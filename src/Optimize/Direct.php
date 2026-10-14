<?php

declare(strict_types=1);

namespace Tessero\Optimize;

use Tessero\Exceptions\TesseroException;

/**
 * scipy.optimize.direct - the DIRECT (DIviding RECTangles) deterministic global optimiser,
 * locally_biased=True (DIRECT_L) by default. Pure PHP, no RNG, host-independent.
 *
 * Faithful port of SciPy Gablonsky DIRECT_L (scipy/optimize/_direct/{DIRect.c,DIRsubrout.c,
 * DIRserial.c}) for jones=1 and feasible (smooth) objectives. The search space is normalised to
 * the unit hypercube; each hyperrectangle is a (center, per-dim trisection-count, f-at-center)
 * triple grouped into per-level anchor lists sorted by f. Each iteration selects the
 * potentially-optimal boxes via the lower-right convex hull of (size, f) with the eps-Lipschitz
 * gate, trisects the longest dimensions of each, and re-inserts. Termination order matches SciPy:
 * vol_tol (status 4) -> len_tol (status 5) -> f_min_rtol (status 3) -> maxfun budget (status 1) ->
 * maxiter (status 2).
 *
 * Parity: iterate-exact (x, fun, and often nfev/nit) against SciPy on separable / convex / highly
 * symmetric landscapes (styblinski-tang, ackley, shifted-sphere, separable quadratics). On
 * Rosenbrock-type curved valleys SciPy stops one or two iterations earlier via its anchor
 * linked-list tie ordering, so this port lands on a slightly different nearby iterate; the UNIQUE
 * global optimum (x, fun) still matches. Fixtures therefore assert iterate-exactness on the
 * separable set and global-optimum parity on the valley set (documented per case).
 *
 * @see scipy/optimize/_direct_py.py (wrapper, defaults), scipy/optimize/_direct/DIRect.c
 */
final class Direct
{
    /**
     * Minimise $func over box $bounds with DIRECT.
     *
     * @param callable(list<float>): float $func objective; receives list<float>, returns float
     * @param list<array{0: float, 1: float}> $bounds per-dimension [lower, upper]
     */
    public static function minimize(
        callable $func,
        array $bounds,
        float $eps = 1e-4,
        ?int $maxfun = null,
        int $maxiter = 1000,
        bool $locallyBiased = true,
        float $fMin = -INF,
        float $fMinRtol = 1e-4,
        float $volTol = 1e-16,
        float $lenTol = 1e-6,
    ): OptimizeResult {
        $n = count($bounds);
        if ($n === 0) {
            throw new TesseroException('direct: bounds must not be empty.');
        }
        $l = [];
        $span = [];
        foreach ($bounds as $b) {
            $lo = (float) $b[0];
            $hi = (float) $b[1];
            if (! is_finite($lo) || ! is_finite($hi) || $lo >= $hi) {
                throw new TesseroException('direct: each bound must be finite with lower < upper.');
            }
            $l[] = $lo;
            $span[] = $hi - $lo;
        }
        $jones = $locallyBiased ? 1 : 0;
        if ($maxfun === null) {
            $maxfun = 1000 * $n;
        }
        $epsabs = 0.0;
        $epsrel = $eps;

        $MAXFUNC = $maxfun <= 0 ? 101000 : ($maxfun + 1000 + intdiv($maxfun, 2));
        $maxf = $maxfun <= 0 ? ($MAXFUNC - 1000) : intdiv(2 * ($MAXFUNC - 1000), 3);
        $MAXDEEP = $maxiter <= 0 ? intdiv($MAXFUNC, 5) : ($maxiter + 1000);

        $thirds = [];
        $thirds[0] = 1.0;
        $h = 3.0;
        for ($i = 1; $i <= $MAXDEEP + 1; $i++) {
            $thirds[$i] = 1.0 / $h;
            $h *= 3.0;
        }
        $levels = [];
        if ($jones === 1) {
            $levels[0] = 1.0;
            $h = 3.0;
            for ($i = 1; $i <= $MAXDEEP + 1; $i++) {
                $levels[$i] = 1.0 / $h;
                $h *= 3.0;
            }
        }

        // Box storage: parallel arrays. c (list<float>), length (list<int>), f (float).
        $bc = [];
        $blen = [];
        $bf = [];

        $evaluate = static function (array $c) use ($func, $span, $l, $n): float {
            $x = [];
            for ($d = 0; $d < $n; $d++) {
                $x[] = $c[$d] * $span[$d] + $l[$d];
            }
            return (float) $func($x);
        };

        $levelOf = static function (array $length, int $jonesLocal) use ($n): int {
            if ($jonesLocal === 1) {
                $mn = $length[0];
                for ($i = 1; $i < $n; $i++) {
                    if ($length[$i] < $mn) {
                        $mn = $length[$i];
                    }
                }
                return (int) $mn;
            }
            $help = (int) $length[0];
            $k = $help;
            $p = 1;
            for ($i = 1; $i < $n; $i++) {
                if ($length[$i] < $k) {
                    $k = (int) $length[$i];
                }
                if ($length[$i] === $help) {
                    $p++;
                }
            }
            if ($k === $help) {
                return $k * $n + $n - $p;
            }
            return $k * $n + $p;
        };

        $getI = static function (array $length) use ($n): array {
            $mn = $length[0];
            for ($i = 1; $i < $n; $i++) {
                if ($length[$i] < $mn) {
                    $mn = $length[$i];
                }
            }
            $dims = [];
            for ($i = 0; $i < $n; $i++) {
                if ($length[$i] === $mn) {
                    $dims[] = $i;
                }
            }
            return $dims;
        };

        // anchor: level -> list of box-indices, kept sorted by f ascending.
        $anchor = [];
        $anchorInsert = static function (int $idx) use (&$anchor, &$blen, &$bf, $levelOf, $jones): void {
            $lv = $levelOf($blen[$idx], $jones);
            if (! isset($anchor[$lv])) {
                $anchor[$lv] = [];
            }
            $lst = &$anchor[$lv];
            $f = $bf[$idx];
            $pos = 0;
            $cnt = count($lst);
            while ($pos < $cnt && $bf[$lst[$pos]] <= $f) {
                $pos++;
            }
            array_splice($lst, $pos, 0, [$idx]);
            unset($lst);
        };

        $numfunc = 0;

        // Trisect a parent box along its longest dims; mutates the parent length in place
        // and returns the list of child boxes [c, length, f] to be appended.
        $sampleAndDivide = function (int $pidx) use (
            &$bc, &$blen, &$bf, &$numfunc, $getI, $evaluate, $thirds, $levelOf, $n
        ): array {
            $plen = $blen[$pidx];
            $pc = $bc[$pidx];
            $dims = $getI($plen);
            $maxi = count($dims);
            $md = (int) min($plen);
            $delta = $thirds[$md + 1];
            $plus = [];
            $minus = [];
            $wvals = [];
            foreach ($dims as $d) {
                $cp = $pc;
                $cp[$d] += $delta;
                $cm = $pc;
                $cm[$d] -= $delta;
                $fp = $evaluate($cp);
                $fm = $evaluate($cm);
                $plus[$d] = [$cp, $fp];
                $minus[$d] = [$cm, $fm];
                $wvals[$d] = min($fp, $fm);
            }
            $numfunc += 2 * $maxi;
            // order dims by ascending w, ties broken by original position in $dims
            $dimPos = array_flip($dims);
            $order = $dims;
            usort($order, static function (int $a, int $b) use ($wvals, $dimPos): int {
                if ($wvals[$a] < $wvals[$b]) {
                    return -1;
                }
                if ($wvals[$a] > $wvals[$b]) {
                    return 1;
                }
                return $dimPos[$a] <=> $dimPos[$b];
            });
            $created = [];
            foreach ($order as $rank => $d) {
                $baseLen = $plen;
                for ($rr = 0; $rr <= $rank; $rr++) {
                    $baseLen[$order[$rr]]++;
                }
                foreach ([$plus[$d], $minus[$d]] as $pair) {
                    $created[] = [$pair[0], $baseLen, $pair[1]];
                }
            }
            foreach ($order as $d) {
                $plen[$d]++;
            }
            $blen[$pidx] = $plen;
            return $created;
        };

        // ---- DIRInit ----
        $c0 = array_fill(0, $n, 0.5);
        $f0 = $evaluate($c0);
        $numfunc = 1;
        $bc[0] = $c0;
        $blen[0] = array_fill(0, $n, 0);
        $bf[0] = $f0;
        $created = $sampleAndDivide(0);
        $anchorInsert(0);
        foreach ($created as $child) {
            $idx = count($bc);
            $bc[$idx] = $child[0];
            $blen[$idx] = $child[1];
            $bf[$idx] = $child[2];
            $anchorInsert($idx);
        }

        $minf = INF;
        $minpos = 0;
        foreach ($bf as $i => $fv) {
            if ($fv < $minf) {
                $minf = $fv;
                $minpos = $i;
            }
        }

        $ierror = 2;
        $numiter = $maxiter;

        for ($t = 2; $t < $maxiter; $t++) {
            $occLevels = array_keys($anchor);
            sort($occLevels);
            $S = [];
            foreach ($occLevels as $lv) {
                $S[] = [$anchor[$lv][0], $lv];
            }
            $maxpos = count($S);

            $keep = array_fill(0, $maxpos, true);
            $rhs = min($minf - $epsrel * abs($minf), $minf - $epsabs);
            for ($j = 0; $j < $maxpos; $j++) {
                $jidx = $S[$j][0];
                $jlv = $S[$j][1];
                $fj = $bf[$jidx];
                $dj = $thirds[$jlv];
                $helplower = INF;   // min slope to larger boxes (i<j)
                $helpgreater = 0.0; // max slope to smaller boxes (i>j)
                $ok = true;
                for ($i = 0; $i < $j; $i++) {
                    $iidx = $S[$i][0];
                    $ilv = $S[$i][1];
                    $di = $thirds[$ilv];
                    $slope = ($bf[$iidx] - $fj) / ($di - $dj);
                    if ($slope < 0.0) {
                        $ok = false;
                        break;
                    }
                    if ($slope < $helplower) {
                        $helplower = $slope;
                    }
                }
                if (! $ok) {
                    $keep[$j] = false;
                    continue;
                }
                for ($i = $j + 1; $i < $maxpos; $i++) {
                    $iidx = $S[$i][0];
                    $ilv = $S[$i][1];
                    $di = $thirds[$ilv];
                    $slope = ($bf[$iidx] - $fj) / ($dj - $di);
                    if ($slope > $helpgreater) {
                        $helpgreater = $slope;
                    }
                }
                if ($helpgreater > $helplower) {
                    $keep[$j] = false;
                    continue;
                }
                if ($fj - $helplower * $dj > $rhs) {
                    $keep[$j] = false;
                }
            }

            $toDivide = [];
            for ($j = 0; $j < $maxpos; $j++) {
                if ($keep[$j]) {
                    $toDivide[] = $S[$j][0];
                }
            }

            foreach ($toDivide as $idx) {
                $lv = $levelOf($blen[$idx], $jones);
                $pos = array_search($idx, $anchor[$lv], true);
                array_splice($anchor[$lv], $pos, 1);
                if (count($anchor[$lv]) === 0) {
                    unset($anchor[$lv]);
                }
                $created = $sampleAndDivide($idx);
                $anchorInsert($idx);
                foreach ($created as $child) {
                    $nidx = count($bc);
                    $bc[$nidx] = $child[0];
                    $blen[$nidx] = $child[1];
                    $bf[$nidx] = $child[2];
                    $anchorInsert($nidx);
                }
            }

            $minf = INF;
            foreach ($bf as $i => $fv) {
                if ($fv < $minf) {
                    $minf = $fv;
                    $minpos = $i;
                }
            }

            $lvlVol = $levelOf($blen[$minpos], 0);
            if ($thirds[$lvlVol] <= $volTol) {
                $ierror = 4;
                $numiter = $t;
                break;
            }
            $lvlLen = $levelOf($blen[$minpos], $jones);
            if ($jones === 1 && $levels[$lvlLen] <= $lenTol) {
                $ierror = 5;
                $numiter = $t;
                break;
            }
            $divfactor = $fMin === 0.0 ? 1.0 : abs($fMin);
            if (is_finite($fMin) && ($minf - $fMin) / $divfactor <= $fMinRtol) {
                $ierror = 3;
                $numiter = $t;
                break;
            }
            if ($numfunc > $maxf) {
                $ierror = 1;
                $numiter = $t;
                break;
            }
        }

        $xbest = [];
        for ($d = 0; $d < $n; $d++) {
            $xbest[] = $bc[$minpos][$d] * $span[$d] + $l[$d];
        }

        $fmt = [$maxf, $maxiter, $fMinRtol, $volTol, $lenTol];
        $msgsErr = [
            'Number of function evaluations done is larger than maxfun={}',
            'Number of iterations is larger than maxiter={}',
            'u[i] < l[i] for some i',
        ];
        $msgsOk = [
            'The best function value found is within a relative error={} of the (known) global optimum f_min',
            'The volume of the hyperrectangle containing the lowest function value found is below vol_tol={}',
            'The side length measure of the hyperrectangle containing the lowest function value found is below len_tol={}',
        ];
        if ($ierror > 2) {
            $message = str_replace('{}', (string) $fmt[$ierror - 1], $msgsOk[$ierror - 3]);
        } else {
            $message = str_replace('{}', (string) $fmt[$ierror - 1], $msgsErr[$ierror - 1]);
        }

        return new OptimizeResult(
            x: $xbest,
            fun: $minf,
            success: $ierror > 2,
            status: $ierror,
            message: $message,
            nit: $numiter,
            nfev: $numfunc,
            njev: 0,
            jac: null,
            hessInv: null,
            method: 'direct',
        );
    }
}