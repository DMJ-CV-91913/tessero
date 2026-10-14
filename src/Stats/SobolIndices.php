<?php

declare(strict_types=1);

namespace Tessero\Stats;

use Tessero\Exceptions\TesseroException;
use Tessero\Stats\Qmc\Sobol;

/**
 * Sobol' global sensitivity indices (scipy.stats.sobol_indices).
 *
 * Computes the first-order and total-order Sobol' sensitivity indices of a model `func` using a
 * Saltelli A/B/AB design drawn from a scrambled 64-bit Sobol' sequence and the `saltelli_2010`
 * estimator (scipy's default `method`). Bit-for-bit identical to scipy.stats.sobol_indices over the
 * verified named test functions, both the first- and total-order indices.
 *
 * ## Method (mirrors scipy's `_sensitivity_analysis`)
 *   1. Draw a scrambled Sobol' sequence of dimension 2*d at bits=64 with the supplied seed
 *      (scipy: `qmc.Sobol(d=2*d, seed=rng, bits=64).random(n)`), transpose to (2*d, n), reshape to
 *      (2, d, n) -> matrices A and B.
 *   2. Map each dimension through its distribution's ppf. Only uniform(loc, scale) distributions are
 *      supported here; `uniform.ppf(q) = loc + scale*q` is applied exactly.
 *   3. Form the AB stack (d pages of A with row i replaced by B's row i).
 *   4. Evaluate `func` on A, B and the flattened AB (same memory layout scipy uses), then subtract
 *      the mean of [f_A, f_B] (the Sobol'-Levitan variance-reduction multiplier) before the estimator.
 *   5. saltelli_2010: s_i = mean(f_B*(f_AB_i - f_A)) / var,  st_i = 0.5*mean((f_A - f_AB_i)^2) / var.
 *
 * `func` must accept a (d, n) matrix (a list of d rows, each a list of n floats) and return a
 * (s, n) matrix (a list of s rows, each a list of n floats); a plain 1-D return (s = 1) is accepted
 * and promoted.
 *
 * @see A. Saltelli et al., "Variance based sensitivity analysis of model output. Design and
 *      estimator for the total sensitivity index", Comput. Phys. Commun. 181 (2010) 259-270.
 */
final class SobolIndices
{
    /**
     * @param callable             $func  model: (d,n) matrix -> (s,n) matrix (or 1-D for s = 1)
     * @param int                  $n     number of A/B samples (must be a power of 2)
     * @param list<array{float,float}> $dists per-dimension [loc, scale] of a uniform distribution
     * @param int|null             $seed  root seed (numpy.random.default_rng(seed)); null = OS entropy
     * @param bool                 $scramble scramble the Sobol' points (scipy always does here)
     *
     * @return array{first_order: list<list<float>>, total_order: list<list<float>>}
     *         each (s, d): first_order[output][input], total_order[output][input]
     */
    public static function sobolIndices(
        callable $func,
        int $n,
        array $dists,
        ?int $seed = null,
        bool $scramble = true
    ): array {
        $d = count($dists);
        if ($d < 1) {
            throw new TesseroException('sobol_indices: dists must be non-empty.');
        }
        if ($n < 1 || ($n & ($n - 1)) !== 0) {
            throw new TesseroException('sobol_indices: n must be a power of 2.');
        }

        // 1. Sobol' points (n, 2d) -> transpose (2d, n) -> reshape (2, d, n) -> A, B.
        $pts = (new Sobol(2 * $d, $scramble, $seed, 64))->random($n)->toArray(); // (n, 2d)
        $A = [];
        $B = [];
        for ($i = 0; $i < $d; $i++) {
            $A[$i] = [];
            $B[$i] = [];
        }
        for ($k = 0; $k < $n; $k++) {
            $row = $pts[$k];
            for ($i = 0; $i < $d; $i++) {
                $A[$i][$k] = $row[$i];
                $B[$i][$k] = $row[$d + $i];
            }
        }

        // 2. ppf per dimension: uniform(loc, scale).ppf(q) = loc + scale*q.
        for ($i = 0; $i < $d; $i++) {
            [$loc, $scale] = $dists[$i];
            for ($k = 0; $k < $n; $k++) {
                $A[$i][$k] = $loc + $scale * $A[$i][$k];
                $B[$i][$k] = $loc + $scale * $B[$i][$k];
            }
        }

        // 3. AB stack: d pages, page i is A with row i replaced by B's row i.
        // 4a. f_A, f_B.
        $fA = self::atLeast2d($func($A), $n);
        $fB = self::atLeast2d($func($B), $n);
        $s = count($fA);

        // 4b. funcAB: scipy flattens AB as moveaxis(AB,0,-1).reshape(d, n*d), evaluates once, then
        // reshape(-1, n, d) + moveaxis(-1,0). We build the (d, n*d) matrix column-block j (0..d-1)
        // holding page j, evaluate func once, and slice back into f_AB[page][output][sample].
        $flat = [];
        for ($i = 0; $i < $d; $i++) {
            $flat[$i] = [];
        }
        // column order: for sample k (0..n-1), for page j (0..d-1) -> one column.
        for ($k = 0; $k < $n; $k++) {
            for ($j = 0; $j < $d; $j++) {
                for ($i = 0; $i < $d; $i++) {
                    // page j = A, but row j replaced by B's row j.
                    $flat[$i][] = ($i === $j) ? $B[$i][$k] : $A[$i][$k];
                }
            }
        }
        $fFlat = self::atLeast2d($func($flat), $n * $d);   // (s, n*d)
        // reshape (s, n*d) -> (s, n, d) -> f_AB[page j][output o][sample k] = fFlat[o][k*d + j].
        $fAB = [];
        for ($j = 0; $j < $d; $j++) {
            $fAB[$j] = [];
            for ($o = 0; $o < $s; $o++) {
                $fAB[$j][$o] = [];
                for ($k = 0; $k < $n; $k++) {
                    $fAB[$j][$o][$k] = $fFlat[$o][$k * $d + $j];
                }
            }
        }

        // 5a. mean over [f_A, f_B] per output; subtract (variance-reduction multiplier).
        $mean = [];
        for ($o = 0; $o < $s; $o++) {
            $sum = 0.0;
            for ($k = 0; $k < $n; $k++) {
                $sum += $fA[$o][$k] + $fB[$o][$k];
            }
            $mean[$o] = $sum / (2 * $n);
        }
        for ($o = 0; $o < $s; $o++) {
            $m = $mean[$o];
            for ($k = 0; $k < $n; $k++) {
                $fA[$o][$k] -= $m;
                $fB[$o][$k] -= $m;
            }
            for ($j = 0; $j < $d; $j++) {
                for ($k = 0; $k < $n; $k++) {
                    $fAB[$j][$o][$k] -= $m;
                }
            }
        }

        // 5b. var = np.var([f_A, f_B], axis=(0,-1)) per output (population variance over 2n values).
        $var = [];
        for ($o = 0; $o < $s; $o++) {
            $mu = 0.0;
            for ($k = 0; $k < $n; $k++) {
                $mu += $fA[$o][$k] + $fB[$o][$k];
            }
            $mu /= (2 * $n);
            $acc = 0.0;
            for ($k = 0; $k < $n; $k++) {
                $acc += ($fA[$o][$k] - $mu) ** 2 + ($fB[$o][$k] - $mu) ** 2;
            }
            $var[$o] = $acc / (2 * $n);
        }

        // 5c. estimator; result shape (s, d): [output][input].
        $first = [];
        $total = [];
        for ($o = 0; $o < $s; $o++) {
            $first[$o] = [];
            $total[$o] = [];
            for ($i = 0; $i < $d; $i++) {
                $sSum = 0.0;
                $tSum = 0.0;
                for ($k = 0; $k < $n; $k++) {
                    $sSum += $fB[$o][$k] * ($fAB[$i][$o][$k] - $fA[$o][$k]);
                    $tSum += ($fA[$o][$k] - $fAB[$i][$o][$k]) ** 2;
                }
                $first[$o][$i] = ($sSum / $n) / $var[$o];
                $total[$o][$i] = (0.5 * ($tSum / $n)) / $var[$o];
            }
        }

        return ['first_order' => $first, 'total_order' => $total];
    }

    /**
     * Promote a 1-D func result (n values) to a (1, n) matrix; pass through a (s, n) matrix.
     *
     * @param mixed $r
     *
     * @return list<list<float>>
     */
    private static function atLeast2d(mixed $r, int $n): array
    {
        $r = is_object($r) && method_exists($r, 'toList') ? $r->toList() : $r;
        if (! is_array($r)) {
            throw new TesseroException('sobol_indices: func must return an array.');
        }
        if ($r !== [] && ! is_array($r[array_key_first($r)])) {
            $r = [$r];   // 1-D -> (1, n)
        }

        return array_map(static fn ($row) => array_map('floatval', array_values($row)), array_values($r));
    }
}
