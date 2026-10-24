<?php

declare(strict_types=1);

namespace Tessero\Stats\Qmc;

use Tessero\Exceptions\TesseroException;
use Tessero\NDArray;
use Tessero\Random\SeedSequence;

/**
 * Sobol' low-discrepancy sequence engine (scipy.stats.qmc.Sobol).
 *
 * Generates the Joe & Kuo (2008) base-2 Sobol' sequence, optionally scrambled with the
 * left-matrix-scramble + digital-shift of Matousek (as scipy does). Bit-for-bit identical to
 * scipy.stats.qmc.Sobol for both the deterministic (scramble=False) and scrambled (scramble=True,
 * rng=numpy.random.default_rng(seed)) variants, at bits=30 (default) and bits=64.
 *
 * ## Algorithm
 * The UNSCRAMBLED sequence is fully deterministic: direction numbers v_{j,i} are built from
 * primitive polynomials and initial direction integers (Joe & Kuo "new-joe-kuo-6.21201" table,
 * vendored here as POLY/MINIT up to {@see MAXDIM} dimensions), then points are drawn by the
 * Gray-code recurrence q ^= v_{*,c}, x = q / 2^bits with c the index of the lowest zero bit of the
 * running counter. The first point is the digital shift (zero when unscrambled).
 *
 * The SCRAMBLE draws come from a PCG64 Generator that is the SINGLE spawned child of
 * default_rng(seed) (scipy: `_rng_spawn(rng, 1)[0]`, spawn_key (0,)); Tessero's {@see SeedSequence}
 * reproduces NumPy's spawn tree exactly. The shift integers and the per-dimension lower-triangular
 * LMS matrices (diagonal forced to 1) are drawn with `Generator::integers(0, 2, ...)` in the same
 * order and shape scipy uses, so the scrambled stream matches to the last bit.
 *
 * ## Representation
 * Direction numbers, the running quasi vector and the digital shift are held as MSB-first bit
 * arrays of width `bits` (so there is no 64-bit unsigned overflow in PHP). The float coordinate is
 * `(double)value * 2^-bits`, formed from the value's two 32-bit halves exactly as a C
 * `(double)uint64` conversion would round it.
 *
 * @see S. Joe and F. Y. Kuo, "Constructing Sobol sequences with better two-dimensional
 *      projections", SIAM J. Sci. Comput. 30, 2635-2654 (2008).
 *
 * The direction-number table (POLY/MINIT) is Joe and Kuo's, used under their licence: see
 * licenses/LICENSE.joe-kuo-sobol.txt.
 */
final class Sobol
{
    /** Maximum supported dimensionality (size of the vendored direction-number table). */
    public const MAXDIM = 201;

    private int $d;

    private int $bits;

    private bool $scramble;

    /** @var list<list<array<int,int>>> sv[dim][col] = MSB-first bit array of width `bits` */
    private array $sv;

    /** @var list<array<int,int>> digital shift per dimension (MSB-first bit array) */
    private array $shift;

    /** @var list<array<int,int>> running quasi vector per dimension (MSB-first bit array) */
    private array $quasi;

    private int $numGenerated = 0;

    /**
     * @param int      $d        number of dimensions (1..MAXDIM)
     * @param bool     $scramble apply the LMS + digital-shift scramble (scipy default True)
     * @param int|null $seed     root seed for the scramble rng (numpy.random.default_rng(seed));
     *                           ignored when $scramble is false
     * @param int      $bits     number of bits (precision); 30 by default, up to 64
     */
    public function __construct(int $d, bool $scramble = true, ?int $seed = null, int $bits = 30)
    {
        if ($d < 1 || $d > self::MAXDIM) {
            throw new TesseroException("Sobol: d must be in 1..".self::MAXDIM.".");
        }
        if ($bits < 1 || $bits > 64) {
            throw new TesseroException('Sobol: bits must be in 1..64.');
        }
        $this->d = $d;
        $this->bits = $bits;
        $this->scramble = $scramble;

        $this->sv = self::initSv($d, $bits);

        if (! $scramble) {
            $this->shift = [];
            for ($i = 0; $i < $d; $i++) {
                $this->shift[$i] = array_fill(0, $bits, 0);
            }
        } else {
            $this->shift = $this->applyScramble($seed);
        }
        $this->quasi = $this->shift;
    }

    /**
     * Draw the next `$n` points of the sequence as an (n, d) NDArray of float64 in [0, 1).
     */
    public function random(int $n = 1): NDArray
    {
        if ($n < 0) {
            throw new TesseroException('Sobol: n must be non-negative.');
        }
        $bits = $this->bits;
        $d = $this->d;
        $rows = [];
        for ($k = 0; $k < $n; $k++) {
            $idx = $this->numGenerated + $k;
            if ($idx === 0) {
                // first point is the digital shift (zeros when unscrambled)
                $row = [];
                foreach ($this->quasi as $q) {
                    $row[] = self::bitsToFloat($q, $bits);
                }
                $rows[] = $row;
                continue;
            }
            $c = self::lowestZeroBit($idx - 1);
            $row = [];
            for ($i = 0; $i < $d; $i++) {
                $col = $this->sv[$i][$c];
                $q = $this->quasi[$i];
                for ($t = 0; $t < $bits; $t++) {
                    $q[$t] ^= $col[$t];
                }
                $this->quasi[$i] = $q;
                $row[] = self::bitsToFloat($q, $bits);
            }
            $rows[] = $row;
        }
        $this->numGenerated += $n;

        if ($n === 0) {
            return NDArray::empty([0, $d]);
        }

        return NDArray::array($rows, 'float64');
    }

    /** Reset the engine so the next draw starts the sequence again. */
    public function reset(): void
    {
        $this->quasi = $this->shift;
        $this->numGenerated = 0;
    }

    /**
     * Build the direction-number matrix sv[dim][col], each column an MSB-first bit array of width
     * `bits`. The Joe & Kuo recurrence is run in a 32- or 64-bit "reference frame" (the frame scipy
     * uses internally), then rescaled to the requested bit width.
     *
     * @return list<list<array<int,int>>>
     */
    private static function initSv(int $dim, int $bits): array
    {
        $ref = $bits <= 32 ? 32 : 64;
        $sv = [];
        for ($dd = 0; $dd < $dim; $dd++) {
            // V[i] (i = 1..bits) = direction integer as an MSB-first bit array of width $ref.
            $V = [];
            if ($dd === 0) {
                for ($i = 1; $i <= $bits; $i++) {
                    $b = array_fill(0, $ref, 0);
                    $b[$i - 1] = 1;              // value 1 << (ref - i)
                    $V[$i] = $b;
                }
            } else {
                $p = self::POLY[$dd];
                $s = strlen(decbin($p)) - 1;     // polynomial degree
                $minit = self::MINIT[$dd];
                for ($i = 1; $i <= min($s, $bits); $i++) {
                    $m = $minit[$i - 1];
                    $b = array_fill(0, $ref, 0);  // value m << (ref - i)
                    for ($j = 0; $j < $ref; $j++) {
                        if (($m >> $j) & 1) {
                            $pos = $i - 1 - $j;
                            if ($pos >= 0 && $pos < $ref) {
                                $b[$pos] = 1;
                            }
                        }
                    }
                    $V[$i] = $b;
                }
                for ($i = $s + 1; $i <= $bits; $i++) {
                    $base = $V[$i - $s];
                    $cur = $base;                 // V[i-s]
                    for ($t = 0; $t < $ref; $t++) { // ^= V[i-s] >> s
                        if ($base[$t]) {
                            $np = $t + $s;
                            if ($np < $ref) {
                                $cur[$np] ^= 1;
                            }
                        }
                    }
                    for ($j = 1; $j < $s; $j++) {  // ^= a_j V[i-j], a_j = (poly >> (s-j)) & 1
                        if (($p >> ($s - $j)) & 1) {
                            $vj = $V[$i - $j];
                            for ($t = 0; $t < $ref; $t++) {
                                $cur[$t] ^= $vj[$t];
                            }
                        }
                    }
                    $V[$i] = $cur;
                }
            }
            // rescale to the `bits` frame: the direction value's top i bits (MSB positions 0..i-1)
            // become the top i bits of the output column.
            $col = [];
            for ($i = 1; $i <= $bits; $i++) {
                $out = array_fill(0, $bits, 0);
                for ($q = 0; $q < $i; $q++) {
                    if ($V[$i][$q]) {
                        $out[$q] = 1;
                    }
                }
                $col[$i - 1] = $out;
            }
            $sv[$dd] = $col;
        }

        return $sv;
    }

    /**
     * Draw the scramble from the spawned PCG64 child and apply the LMS to {@see $sv}, returning the
     * digital-shift bit arrays. Mirrors scipy's `Sobol._scramble` exactly.
     *
     * @return list<array<int,int>>
     */
    private function applyScramble(?int $seed): array
    {
        $d = $this->d;
        $bits = $this->bits;
        // scipy: self.rng = _rng_spawn(default_rng(seed), 1)[0]  -> spawn_key (0,)
        $root = new SeedSequence($seed);
        $gen = $root->childWithKey([0])->generator();

        // digital shift: rand_shift = integers(0, 2, (d, bits)); shift[d] = sum_b rand_shift[d,b] << b
        $rs = $gen->integers(0, 2, [$d, $bits])->toArray();
        $shift = [];
        for ($i = 0; $i < $d; $i++) {
            $sb = array_fill(0, $bits, 0);
            for ($b = 0; $b < $bits; $b++) {
                if ($rs[$i][$b]) {
                    $sb[$bits - 1 - $b] = 1;    // value 2^b -> MSB position (bits-1-b)
                }
            }
            $shift[$i] = $sb;
        }

        // LMS: ltm = tril(integers(0, 2, (d, bits, bits))), diagonal forced to 1; column transform
        // out bit i = XOR_{k<=i} ltm[i][k] * in bit k   (MSB indexing).
        $rand = $gen->integers(0, 2, [$d, $bits, $bits])->toArray();
        for ($i = 0; $i < $d; $i++) {
            $M = $rand[$i];
            $newcols = [];
            for ($j = 0; $j < $bits; $j++) {
                $v = $this->sv[$i][$j];
                $nv = array_fill(0, $bits, 0);
                for ($r = 0; $r < $bits; $r++) {
                    $acc = $v[$r];              // diagonal (k == r) is forced to 1
                    for ($k = 0; $k < $r; $k++) {
                        if ($M[$r][$k]) {
                            $acc ^= $v[$k];
                        }
                    }
                    $nv[$r] = $acc;
                }
                $newcols[$j] = $nv;
            }
            $this->sv[$i] = $newcols;
        }

        return $shift;
    }

    /**
     * Convert an MSB-first bit array (value V, width `bits`) to the double `(double)V * 2^-bits`,
     * matching a C `(double)uint` round-to-nearest conversion via the value's 32-bit halves.
     *
     * @param array<int,int> $b
     */
    private static function bitsToFloat(array $b, int $bits): float
    {
        $hi = 0;
        $lo = 0;
        for ($i = 0; $i < $bits; $i++) {
            if ($b[$i]) {
                $w = $bits - 1 - $i;
                if ($w >= 32) {
                    $hi |= (1 << ($w - 32));
                } else {
                    $lo |= (1 << $w);
                }
            }
        }

        return ((float) $hi * 4294967296.0 + (float) $lo) * (2.0 ** -$bits);
    }

    /** Index (0-based) of the lowest zero bit of $x (the Gray-code column selector). */
    private static function lowestZeroBit(int $x): int
    {
        $c = 0;
        while ($x & 1) {
            $x >>= 1;
            $c++;
        }

        return $c;
    }

    // Joe & Kuo (2008) "new-joe-kuo-6.21201" direction-number table, capped at MAXDIM dimensions.
    // POLY[d] is dimension d's primitive polynomial (encoded with implicit top and bottom bits);
    // MINIT[d] lists its initial direction integers m_1..m_s (s = degree). Dimension 0 is the
    // identity (POLY[0] = 1, no MINIT entries).
    private const POLY = [1,3,7,11,13,19,25,37,41,47,55,59,61,67,91,97,103,109,115,131,137,143,145,157,167,171,185,191,193,203,211,213,229,239,241,247,253,285,299,301,333,351,355,357,361,369,391,397,425,451,463,487,501,529,539,545,557,563,601,607,617,623,631,637,647,661,675,677,687,695,701,719,721,731,757,761,787,789,799,803,817,827,847,859,865,875,877,883,895,901,911,949,953,967,971,973,981,985,995,1001,1019,1033,1051,1063,1069,1125,1135,1153,1163,1221,1239,1255,1267,1279,1293,1305,1315,1329,1341,1347,1367,1387,1413,1423,1431,1441,1479,1509,1527,1531,1555,1557,1573,1591,1603,1615,1627,1657,1663,1673,1717,1729,1747,1759,1789,1815,1821,1825,1849,1863,1869,1877,1881,1891,1917,1933,1939,1969,2011,2035,2041,2053,2071,2091,2093,2119,2147,2149,2161,2171,2189,2197,2207,2217,2225,2255,2257,2273,2279,2283,2293,2317,2323,2341,2345,2363,2365,2373,2377,2385,2395,2419,2421,2431,2435,2447,2475,2477,2489,2503,2521];

    private const MINIT = [
        [],
        [1],
        [1,3],
        [1,3,1],
        [1,1,1],
        [1,1,3,3],
        [1,3,5,13],
        [1,1,5,5,17],
        [1,1,5,5,5],
        [1,1,7,11,19],
        [1,1,5,1,1],
        [1,1,1,3,11],
        [1,3,5,5,31],
        [1,3,3,9,7,49],
        [1,1,1,15,21,21],
        [1,3,1,13,27,49],
        [1,1,1,15,7,5],
        [1,3,1,15,13,25],
        [1,1,5,5,19,61],
        [1,3,7,11,23,15,103],
        [1,3,7,13,13,15,69],
        [1,1,3,13,7,35,63],
        [1,3,5,9,1,25,53],
        [1,3,1,13,9,35,107],
        [1,3,1,5,27,61,31],
        [1,1,5,11,19,41,61],
        [1,3,5,3,3,13,69],
        [1,1,7,13,1,19,1],
        [1,3,7,5,13,19,59],
        [1,1,3,9,25,29,41],
        [1,3,5,13,23,1,55],
        [1,3,7,3,13,59,17],
        [1,3,1,3,5,53,69],
        [1,1,5,5,23,33,13],
        [1,1,7,7,1,61,123],
        [1,1,7,9,13,61,49],
        [1,3,3,5,3,55,33],
        [1,3,1,15,31,13,49,245],
        [1,3,5,15,31,59,63,97],
        [1,3,1,11,11,11,77,249],
        [1,3,1,11,27,43,71,9],
        [1,1,7,15,21,11,81,45],
        [1,3,7,3,25,31,65,79],
        [1,3,1,1,19,11,3,205],
        [1,1,5,9,19,21,29,157],
        [1,3,7,11,1,33,89,185],
        [1,3,3,3,15,9,79,71],
        [1,3,7,11,15,39,119,27],
        [1,1,3,1,11,31,97,225],
        [1,1,1,3,23,43,57,177],
        [1,3,7,7,17,17,37,71],
        [1,3,1,5,27,63,123,213],
        [1,1,3,5,11,43,53,133],
        [1,3,5,5,29,17,47,173,479],
        [1,3,3,11,3,1,109,9,69],
        [1,1,1,5,17,39,23,5,343],
        [1,3,1,5,25,15,31,103,499],
        [1,1,1,11,11,17,63,105,183],
        [1,1,5,11,9,29,97,231,363],
        [1,1,5,15,19,45,41,7,383],
        [1,3,7,7,31,19,83,137,221],
        [1,1,1,3,23,15,111,223,83],
        [1,1,5,13,31,15,55,25,161],
        [1,1,3,13,25,47,39,87,257],
        [1,1,1,11,21,53,125,249,293],
        [1,1,7,11,11,7,57,79,323],
        [1,1,5,5,17,13,81,3,131],
        [1,1,7,13,23,7,65,251,475],
        [1,3,5,1,9,43,3,149,11],
        [1,1,3,13,31,13,13,255,487],
        [1,3,3,1,5,63,89,91,127],
        [1,1,3,3,1,19,123,127,237],
        [1,1,5,7,23,31,37,243,289],
        [1,1,5,11,17,53,117,183,491],
        [1,1,1,5,1,13,13,209,345],
        [1,1,3,15,1,57,115,7,33],
        [1,3,1,11,7,43,81,207,175],
        [1,3,1,1,15,27,63,255,49],
        [1,3,5,3,27,61,105,171,305],
        [1,1,5,3,1,3,57,249,149],
        [1,1,3,5,5,57,15,13,159],
        [1,1,1,11,7,11,105,141,225],
        [1,3,3,5,27,59,121,101,271],
        [1,3,5,9,11,49,51,59,115],
        [1,1,7,1,23,45,125,71,419],
        [1,1,3,5,23,5,105,109,75],
        [1,1,7,15,7,11,67,121,453],
        [1,3,7,3,9,13,31,27,449],
        [1,3,1,15,19,39,39,89,15],
        [1,1,1,1,1,33,73,145,379],
        [1,3,1,15,15,43,29,13,483],
        [1,1,7,3,19,27,85,131,431],
        [1,3,3,3,5,35,23,195,349],
        [1,3,3,7,9,27,39,59,297],
        [1,1,3,9,11,17,13,241,157],
        [1,3,7,15,25,57,33,189,213],
        [1,1,7,1,9,55,73,83,217],
        [1,3,3,13,19,27,23,113,249],
        [1,3,5,3,23,43,3,253,479],
        [1,1,5,5,11,5,45,117,217],
        [1,3,3,7,29,37,33,123,147],
        [1,3,1,15,5,5,37,227,223,459],
        [1,1,7,5,5,39,63,255,135,487],
        [1,3,1,7,9,7,87,249,217,599],
        [1,1,3,13,9,47,7,225,363,247],
        [1,3,7,13,19,13,9,67,9,737],
        [1,3,5,5,19,59,7,41,319,677],
        [1,1,5,3,31,63,15,43,207,789],
        [1,1,7,9,13,39,3,47,497,169],
        [1,3,1,7,21,17,97,19,415,905],
        [1,3,7,1,3,31,71,111,165,127],
        [1,1,5,11,1,61,83,119,203,847],
        [1,3,3,13,9,61,19,97,47,35],
        [1,1,7,7,15,29,63,95,417,469],
        [1,3,1,9,25,9,71,57,213,385],
        [1,3,5,13,31,47,101,57,39,341],
        [1,1,3,3,31,57,125,173,365,551],
        [1,3,7,1,13,57,67,157,451,707],
        [1,1,1,7,21,13,105,89,429,965],
        [1,1,5,9,17,51,45,119,157,141],
        [1,3,7,7,13,45,91,9,129,741],
        [1,3,7,1,23,57,67,141,151,571],
        [1,1,3,11,17,47,93,107,375,157],
        [1,3,3,5,11,21,43,51,169,915],
        [1,1,5,3,15,55,101,67,455,625],
        [1,3,5,9,1,23,29,47,345,595],
        [1,3,7,7,5,49,29,155,323,589],
        [1,3,3,7,5,41,127,61,261,717],
        [1,3,7,7,17,23,117,67,129,1009],
        [1,1,3,13,11,39,21,207,123,305],
        [1,1,3,9,29,3,95,47,231,73],
        [1,3,1,9,1,29,117,21,441,259],
        [1,3,1,13,21,39,125,211,439,723],
        [1,1,7,3,17,63,115,89,49,773],
        [1,3,7,13,11,33,101,107,63,73],
        [1,1,5,5,13,57,63,135,437,177],
        [1,1,3,7,27,63,93,47,417,483],
        [1,1,3,1,23,29,1,191,49,23],
        [1,1,3,15,25,55,9,101,219,607],
        [1,3,1,7,7,19,51,251,393,307],
        [1,3,3,3,25,55,17,75,337,3],
        [1,1,1,13,25,17,65,45,479,413],
        [1,1,7,7,27,49,99,161,213,727],
        [1,3,5,1,23,5,43,41,251,857],
        [1,3,3,7,11,61,39,87,383,835],
        [1,1,3,15,13,7,29,7,505,923],
        [1,3,7,1,5,31,47,157,445,501],
        [1,1,3,7,1,43,9,147,115,605],
        [1,3,3,13,5,1,119,211,455,1001],
        [1,1,3,5,13,19,3,243,75,843],
        [1,3,7,7,1,19,91,249,357,589],
        [1,1,1,9,1,25,109,197,279,411],
        [1,3,1,15,23,57,59,135,191,75],
        [1,1,5,15,29,21,39,253,383,349],
        [1,3,3,5,19,45,61,151,199,981],
        [1,3,5,13,9,61,107,141,141,1],
        [1,3,1,11,27,25,85,105,309,979],
        [1,3,3,11,19,7,115,223,349,43],
        [1,1,7,9,21,39,123,21,275,927],
        [1,1,7,13,15,41,47,243,303,437],
        [1,1,1,7,7,3,15,99,409,719],
        [1,3,3,15,27,49,113,123,113,67,469],
        [1,3,7,11,3,23,87,169,119,483,199],
        [1,1,5,15,7,17,109,229,179,213,741],
        [1,1,5,13,11,17,25,135,403,557,1433],
        [1,3,1,1,1,61,67,215,189,945,1243],
        [1,1,7,13,17,33,9,221,429,217,1679],
        [1,1,3,11,27,3,15,93,93,865,1049],
        [1,3,7,7,25,41,121,35,373,379,1547],
        [1,3,3,9,11,35,45,205,241,9,59],
        [1,3,1,7,3,51,7,177,53,975,89],
        [1,1,3,5,27,1,113,231,299,759,861],
        [1,3,3,15,25,29,5,255,139,891,2031],
        [1,3,1,1,13,9,109,193,419,95,17],
        [1,1,7,9,3,7,29,41,135,839,867],
        [1,1,7,9,25,49,123,217,113,909,215],
        [1,1,7,3,23,15,43,133,217,327,901],
        [1,1,3,3,13,53,63,123,477,711,1387],
        [1,1,3,15,7,29,75,119,181,957,247],
        [1,1,1,11,27,25,109,151,267,99,1461],
        [1,3,7,15,5,5,53,145,11,725,1501],
        [1,3,7,1,9,43,71,229,157,607,1835],
        [1,3,3,13,25,1,5,27,471,349,127],
        [1,1,1,1,23,37,9,221,269,897,1685],
        [1,1,3,3,31,29,51,19,311,553,1969],
        [1,3,7,5,5,55,17,39,475,671,1529],
        [1,1,7,1,1,35,47,27,437,395,1635],
        [1,1,7,3,13,23,43,135,327,139,389],
        [1,3,7,3,9,25,91,25,429,219,513],
        [1,1,3,5,13,29,119,201,277,157,2043],
        [1,3,5,3,29,57,13,17,167,739,1031],
        [1,3,3,5,29,21,95,27,255,679,1531],
        [1,3,7,15,9,5,21,71,61,961,1201],
        [1,3,5,13,15,57,33,93,459,867,223],
        [1,1,1,15,17,43,127,191,67,177,1073],
        [1,1,1,15,23,7,21,199,75,293,1611],
        [1,3,7,13,15,39,21,149,65,741,319],
        [1,3,7,11,23,13,101,89,277,519,711]
    ];
}
