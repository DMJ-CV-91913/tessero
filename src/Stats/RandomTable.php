<?php

declare(strict_types=1);

namespace Tessero\Stats;

use Tessero\Native\Library;
use Tessero\Random\Generator;

/**
 * scipy.stats.random_table — random contingency tables with fixed row and column margins
 * (the distribution of tables under the row/column-independence null hypothesis).
 *
 * Pure-PHP facade over Tessero's bit-exact PCG64 {@see Generator}. It has no kernel of its own:
 * it advances the SAME PCG64 stream numpy uses, through the already-exported raw primitives
 * `tsr_pcg64_uint64` (next_uint64) and `tsr_pcg64_random` (next_double), so it is identical on both
 * backends (FFI and the native extension) and bit-exact against
 *   scipy.stats.random_table.rvs(row, col, random_state=np.random.default_rng(seed)).
 *
 * scipy picks the algorithm with _rvs_select: Patefield (rcont2) when ntot > log(ntot+1)*cells,
 * otherwise Boyett (rcont1); the `method` argument forces one. Both are faithful ports of scipy's
 * C `_rcont.c`:
 *
 *   Boyett (rcont1): build a work[] array that repeats each column index c[j] times, Fisher-Yates
 *     shuffle it top-down with random_interval(i), then tally into the table. random_interval is
 *     numpy's masked-rejection bounded integer over next_uint32; since ntot-1 always fits in 32 bits
 *     for a contingency table, it draws uint32 values. PCG64's next_uint32 is BUFFERED (it returns
 *     the low 32 bits of a fresh uint64 and stashes the high 32 for the next call); we reproduce that
 *     buffer in PHP on top of raw next_uint64 draws, so the stream advances exactly as numpy's.
 *
 *   Patefield (rcont2): fill the table cell by cell (last row and column are forced by the margins)
 *     by sampling each entry from its conditional hypergeometric distribution, using only
 *     random_standard_uniform == next_double draws. This is a line-for-line port of scipy's goto-based
 *     rcont2, including the sum-probability renormalisation that redraws the uniform.
 *
 * An int seed passed to scipy would build a legacy MT19937 RandomState and would NOT match — use
 * default_rng(seed). Output is an integer table of shape (len(row), len(col)) (scipy returns float64,
 * but the entries are exact integers; fixtures compare integer-equal).
 */
final class RandomTable
{
    /**
     * Draw one random contingency table with the given row and column sums (size 1).
     *
     * @param list<int>               $row    non-negative row sums
     * @param list<int>               $col    non-negative column sums (must sum to the same total)
     * @param string|null             $method 'boyett', 'patefield', or null to auto-select like scipy
     * @param int|array<int>|Generator|null $rng seed for a PCG64 Generator, or a Generator instance
     *
     * @return array<int, array<int, int>> the len(row) x len(col) table
     */
    public static function rvs(array $row, array $col, ?string $method = null, int|array|Generator|null $rng = null): array
    {
        [$r, $c, $n] = self::processParameters($row, $col);
        $gen = ($rng instanceof Generator) ? $rng : Generator::defaultRng($rng);

        $meth = self::selectMethod($method, $r, $c, $n);

        if ($meth === 'patefield') {
            return self::rcont2($r, $c, $n, $gen);
        }

        return self::rcont1($r, $c, $n, $gen);
    }

    /**
     * Validate margins and return [row, col, total] (scipy _process_parameters).
     *
     * @return array{0: list<int>, 1: list<int>, 2: int}
     */
    private static function processParameters(array $row, array $col): array
    {
        $r = array_values(array_map('intval', $row));
        $c = array_values(array_map('intval', $col));
        foreach ($row as $i => $v) {
            if ((float) $v !== (float) $r[$i]) {
                throw new \InvalidArgumentException('each element of `row` must be an integer');
            }
        }
        foreach ($col as $i => $v) {
            if ((float) $v !== (float) $c[$i]) {
                throw new \InvalidArgumentException('each element of `col` must be an integer');
            }
        }
        foreach ($r as $v) {
            if ($v < 0) {
                throw new \InvalidArgumentException('each element of `row` must be non-negative');
            }
        }
        foreach ($c as $v) {
            if ($v < 0) {
                throw new \InvalidArgumentException('each element of `col` must be non-negative');
            }
        }
        $n = array_sum($r);
        if ($n !== array_sum($c)) {
            throw new \InvalidArgumentException('sums over `row` and `col` must be equal');
        }

        return [$r, $c, $n];
    }

    /**
     * scipy _rvs_select / _process_rvs_method: default picks Patefield when
     * ntot > log(ntot+1) * cells, else Boyett.
     *
     * @param list<int> $r
     * @param list<int> $c
     */
    private static function selectMethod(?string $method, array $r, array $c, int $n): string
    {
        if ($method === 'boyett' || $method === 'patefield') {
            return $method;
        }
        if ($method !== null) {
            throw new \InvalidArgumentException("'{$method}' not recognized, must be one of boyett, patefield");
        }
        $k = count($r) * count($c);
        // n + 1 guards against n == 0; fac = 1.0.
        return ($n > log($n + 1) * $k) ? 'patefield' : 'boyett';
    }

    // ------------------------------------------------------------------ Boyett (rcont1)

    /**
     * Boyett's algorithm: shuffle a column-index work array with random_interval, then tally.
     *
     * @param list<int> $r
     * @param list<int> $c
     *
     * @return array<int, array<int, int>>
     */
    private static function rcont1(array $r, array $c, int $n, Generator $gen): array
    {
        $nr = count($r);
        $nc = count($c);
        $table = [];
        for ($i = 0; $i < $nr; $i++) {
            $table[$i] = array_fill(0, $nc, 0);
        }
        if ($n === 0) {
            return $table;
        }

        // work = [0]*c0 ++ [1]*c1 ++ ...  (rcont1_init)
        $work = [];
        for ($i = 0; $i < $nc; $i++) {
            for ($t = 0; $t < $c[$i]; $t++) {
                $work[] = $i;
            }
        }

        $u32 = new Pcg64Uint32Stream($gen);
        // Fisher-Yates top-down: for i = ntot-1 .. 1: j = random_interval(i); swap.
        for ($i = $n - 1; $i > 0; $i--) {
            $j = $u32->randomInterval($i);
            $tmp = $work[$j];
            $work[$j] = $work[$i];
            $work[$i] = $tmp;
        }

        // Tally: row ir consumes r[ir] work entries in order.
        $idx = 0;
        for ($ir = 0; $ir < $nr; $ir++) {
            for ($t = 0; $t < $r[$ir]; $t++) {
                $table[$ir][$work[$idx]] += 1;
                $idx++;
            }
        }

        return $table;
    }

    // ------------------------------------------------------------------ Patefield (rcont2)

    /**
     * Patefield's algorithm: conditional-hypergeometric cell fill using next_double draws. A faithful
     * port of scipy's goto-based C rcont2.
     *
     * @param list<int> $r
     * @param list<int> $c
     *
     * @return array<int, array<int, int>>
     */
    private static function rcont2(array $r, array $c, int $n, Generator $gen): array
    {
        $nr = count($r);
        $nc = count($c);
        $table = [];
        for ($i = 0; $i < $nr; $i++) {
            $table[$i] = array_fill(0, $nc, 0);
        }
        if ($n === 0) {
            return $table;
        }

        $u = new Pcg64DoubleStream($gen);

        // jwork holds the running column residuals c[0..nc-2]; folded onto the last table row like scipy
        // but kept as its own array here for clarity (last row filled explicitly at the end).
        $jwork = array_fill(0, $nc, 0);
        for ($i = 0; $i < $nc - 1; $i++) {
            $jwork[$i] = $c[$i];
        }

        $jc = $n;
        $ib = 0;
        for ($l = 0; $l < $nr - 1; $l++) {
            $ia = $r[$l]; // first term
            if ($ia === 0) {
                for ($i = 0; $i < $nc; $i++) {
                    $table[$l][$i] = 0;
                }
                continue;
            }
            $ic = $jc; // second term
            $jc -= $r[$l];
            for ($m = 0; $m < $nc - 1; $m++) {
                $id = $jwork[$m]; // third term
                $ie = $ic;        // eighth term
                $ic -= $id;
                $ib = $ie - $ia;  // must be computed before the c[m]==0 skip (used after the row loop)
                if ($c[$m] === 0) {
                    for ($i = 0; $i < $nr; $i++) {
                        $table[$i][$m] = 0;
                    }
                    continue;
                }
                $ii = $ib - $id; // fourth term
                if ($ie === 0) {
                    for ($j = $m; $j < $nc - 1; $j++) {
                        $table[$l][$j] = 0;
                    }
                    $ia = 0;
                    break;
                }
                $z = $u->next();
                $nlm = self::conditional($ia, $ib, $ic, $id, $ie, $ii, $z, $u);
                $table[$l][$m] = $nlm;
                $ia -= $nlm;
                $jwork[$m] -= $nlm;
            }
            // last column of this row is forced.
            $table[$l][$nc - 1] = $ia;
        }

        // Last row is forced: it equals the remaining column residuals (jwork), with the final
        // corner fixed by the running ib.
        for ($i = 0; $i < $nc; $i++) {
            $table[$nr - 1][$i] = $jwork[$i];
        }
        $table[$nr - 1][$nc - 1] = $ib - $table[$nr - 1][$nc - 2];

        return $table;
    }

    /**
     * Sample one cell entry nlm from its conditional hypergeometric distribution given the current
     * margins, by inverting the cumulative probability against the uniform $z. A direct transcription
     * of scipy's labelled goto block (l131..l160), including the sum-probability renormalisation that
     * redraws a uniform via $u->next().
     */
    private static function conditional(int $ia, int $ib, int $ic, int $id, int $ie, int $ii, float $z, Pcg64DoubleStream $u): int
    {
        // l131: mode of the conditional distribution and its probability.
        while (true) {
            $nlm = (int) floor(($ia * $id) / $ie + 0.5);
            $x = exp(
                self::logFactorial($ia) + self::logFactorial($ib) + self::logFactorial($ic) + self::logFactorial($id)
                - self::logFactorial($ie) - self::logFactorial($nlm) - self::logFactorial($id - $nlm)
                - self::logFactorial($ia - $nlm) - self::logFactorial($ii + $nlm)
            );
            if ($x >= $z) {
                return $nlm; // l160
            }

            $sumprb = $x;
            $y = $x;
            $nll = $nlm;
            $lsp = false;
            $lsm = false;

            $state = 'L140';
            while (true) {
                if ($state === 'L140') {
                    $j = ($id - $nlm) * ($ia - $nlm);
                    if ($j === 0) {
                        $state = 'L156';
                        continue;
                    }
                    $nlm += 1;
                    $x *= $j / ($nlm * ($ii + $nlm));
                    $sumprb += $x;
                    if ($sumprb >= $z) {
                        return $nlm; // l160
                    }
                    $state = 'L150';
                    continue;
                }
                if ($state === 'L150') {
                    if ($lsm) {
                        $state = 'L155';
                        continue;
                    }
                    $j = $nll * ($ii + $nll);
                    if ($j === 0) {
                        $state = 'L154';
                        continue;
                    }
                    $nll -= 1;
                    $y *= $j / (($id - $nll) * ($ia - $nll));
                    $sumprb += $y;
                    if ($sumprb >= $z) {
                        return $nll; // l159 -> l160 (nlm = nll)
                    }
                    if (! $lsp) {
                        $state = 'L140';
                        continue;
                    }
                    $state = 'L150';
                    continue;
                }
                if ($state === 'L154') {
                    $lsm = true;
                    $state = 'L155';
                    continue;
                }
                if ($state === 'L155') {
                    if (! $lsp) {
                        $state = 'L140';
                        continue;
                    }
                    // Numerical safety net: renormalise and redraw, then restart from l131.
                    $z = $u->next() * $sumprb;
                    break; // restart outer while (goto l131)
                }
                if ($state === 'L156') {
                    $lsp = true;
                    $state = 'L150';
                    continue;
                }
            }
            // reached only via the L155 redraw -> loop back to l131.
        }
    }

    /**
     * log(k!) — a byte-for-byte port of numpy's logfactorial (the function scipy's _rcont calls): a
     * 126-entry lookup table for k < 126, then the Stirling series (truncated at 1/k^3) beyond. Using
     * numpy's own table (NOT libm lgamma, which differs in the last ULP) is what makes the conditional
     * threshold comparisons — and therefore the sampled table — bit-identical to scipy.
     */
    private static function logFactorial(int $k): float
    {
        if ($k < 126) {
            return self::LOGFACT[$k];
        }
        $halfln2pi = 0.9189385332046728;
        $kf = (float) $k;

        return ($kf + 0.5) * log($kf) - $kf + ($halfln2pi + (1.0 / $kf) * (1.0 / 12.0 - 1.0 / (360.0 * $kf * $kf)));
    }

    /** log(k!) for k = 0..125, copied verbatim from numpy/random/src/logfactorial.c. */
    private const LOGFACT = [
        0,
        0,
        0.69314718055994529,
        1.791759469228055,
        3.1780538303479458,
        4.7874917427820458,
        6.5792512120101012,
        8.5251613610654147,
        10.604602902745251,
        12.801827480081469,
        15.104412573075516,
        17.502307845873887,
        19.987214495661885,
        22.552163853123425,
        25.19122118273868,
        27.89927138384089,
        30.671860106080672,
        33.505073450136891,
        36.395445208033053,
        39.339884187199495,
        42.335616460753485,
        45.380138898476908,
        48.471181351835227,
        51.606675567764377,
        54.784729398112319,
        58.003605222980518,
        61.261701761002001,
        64.557538627006338,
        67.88974313718154,
        71.257038967168015,
        74.658236348830158,
        78.092223553315307,
        81.557959456115043,
        85.054467017581516,
        88.580827542197682,
        92.136175603687093,
        95.719694542143202,
        99.330612454787428,
        102.96819861451381,
        106.63176026064346,
        110.32063971475739,
        114.03421178146171,
        117.77188139974507,
        121.53308151543864,
        125.3172711493569,
        129.12393363912722,
        132.95257503561632,
        136.80272263732635,
        140.67392364823425,
        144.5657439463449,
        148.47776695177302,
        152.40959258449735,
        156.3608363030788,
        160.3311282166309,
        164.32011226319517,
        168.32744544842765,
        172.35279713916279,
        176.39584840699735,
        180.45629141754378,
        184.53382886144948,
        188.6281734236716,
        192.7390472878449,
        196.86618167289001,
        201.00931639928152,
        205.1681994826412,
        209.34258675253685,
        213.53224149456327,
        217.73693411395422,
        221.95644181913033,
        226.1905483237276,
        230.43904356577696,
        234.70172344281826,
        238.97838956183432,
        243.26884900298271,
        247.57291409618688,
        251.89040220972319,
        256.22113555000954,
        260.56494097186322,
        264.92164979855278,
        269.29109765101981,
        273.67312428569369,
        278.06757344036612,
        282.4742926876304,
        286.89313329542699,
        291.32395009427029,
        295.76660135076065,
        300.22094864701415,
        304.68685676566872,
        309.1641935801469,
        313.65282994987905,
        318.1526396202093,
        322.66349912672615,
        327.1852877037752,
        331.71788719692847,
        336.26118197919845,
        340.81505887079902,
        345.37940706226686,
        349.95411804077025,
        354.53908551944079,
        359.1342053695754,
        363.73937555556347,
        368.35449607240474,
        372.97946888568902,
        377.61419787391867,
        382.25858877306001,
        386.91254912321756,
        391.57598821732961,
        396.24881705179155,
        400.93094827891576,
        405.6222961611449,
        410.32277652693733,
        415.03230672824964,
        419.75080559954472,
        424.47819341825709,
        429.21439186665157,
        433.95932399501481,
        438.71291418612117,
        443.47508812091894,
        448.24577274538461,
        453.02489623849613,
        457.81238798127816,
        462.60817852687489,
        467.4121995716082,
        472.22438392698058,
        477.04466549258564,
        481.87297922988796,
    ];
}
