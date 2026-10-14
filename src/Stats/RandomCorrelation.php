<?php

declare(strict_types=1);

namespace Tessero\Stats;

use Tessero\Random\Generator;

/**
 * scipy.stats.random_correlation — a random correlation matrix with a prescribed set of
 * eigenvalues. Symmetric positive semi-definite with unit diagonal.
 *
 * Pure-PHP facade over Tessero's bit-exact PCG64 {@see Generator} and {@see OrthoGroup}; it has
 * no kernel of its own, so it is identical on both backends (FFI and the native extension).
 *
 * Algorithm (identical to scipy.stats._multivariate.random_correlation_gen.rvs, the numerically
 * stable Davies & Higham construction):
 *   m = ortho_group.rvs(dim)                 # the ONLY rng draw (one dim x dim normal matrix)
 *   m = m @ diag(eigs) @ m.T                 # symmetric psd with the requested spectrum
 *   m = _to_corr(m)                          # Givens rotations to put 1's on the diagonal
 * _to_corr is pure arithmetic (no rng): a sweep of 2x2 Givens rotations, each applied in place as
 * two reference-BLAS `drot` passes (row pair i,j then column pair i,j) so the trace stays equal to
 * the dimension and every diagonal entry becomes 1. The rotation angle for cell (i, j) comes from
 * _givens_to_1(m[i,i], m[j,j], m[i,j]); the partner j is the first later index whose diagonal lies
 * on the opposite side of 1 from m[i,i].
 *
 * Because the single rng draw (ortho_group's normal matrix) and the subsequent deterministic
 * arithmetic both reproduce scipy's, the matrix matches
 *   scipy.stats.random_correlation.rvs(eigs, random_state=np.random.default_rng(seed))
 * to a few ULP (observed max abs diff ~9e-16 across all fixtures — pure BLAS-drot FP reassociation;
 * NOT byte-exact, unlike ortho_group, because _to_corr accumulates rotations). An int seed passed to
 * scipy would build a legacy MT19937 RandomState and would NOT match — use default_rng(seed).
 */
final class RandomCorrelation
{
    /**
     * Draw one random correlation matrix with eigenvalues $eigs (scipy random_correlation.rvs, size 1).
     *
     * @param list<float>              $eigs    eigenvalues; all >= 0 and summing to count($eigs) (= dim)
     * @param int|array<int>|Generator|null $rng seed for a PCG64 Generator, or a Generator instance
     * @param float                    $tol     tolerance for the input checks (scipy default 1e-13)
     * @param float                    $diagTol tolerance for the unit-diagonal check (scipy default 1e-7)
     *
     * @return array<int, array<int, float>> the dim x dim correlation matrix
     */
    public static function rvs(array $eigs, int|array|Generator|null $rng = null, float $tol = 1e-13, float $diagTol = 1e-7): array
    {
        $eigs = array_values(array_map('floatval', $eigs));
        $dim = count($eigs);
        if ($dim <= 1) {
            throw new \InvalidArgumentException("Array 'eigs' must be a vector of length greater than 1.");
        }
        $sum = array_sum($eigs);
        if (abs($sum - $dim) > $tol) {
            throw new \InvalidArgumentException('Sum of eigenvalues must equal dimensionality.');
        }
        foreach ($eigs as $x) {
            if ($x < -$tol) {
                throw new \InvalidArgumentException('All eigenvalues must be non-negative.');
            }
        }

        $gen = ($rng instanceof Generator) ? $rng : Generator::defaultRng($rng);

        // m = ortho_group.rvs(dim) — the sole rng draw.
        $o = OrthoGroup::rvs($dim, $gen);
        $m = method_exists($o, 'toArray') ? $o->toArray() : $o;

        // m = m @ diag(eigs) @ m.T  (scale columns of m by eigs, then multiply by m^T)
        $md = self::matTimesDiagTimesTranspose($m, $eigs, $dim);

        // Carefully rotate to unit diagonal.
        self::toCorr($md, $dim);

        // Check diagonal (scipy raises RuntimeError on failure).
        $maxDiag = 0.0;
        for ($i = 0; $i < $dim; $i++) {
            $maxDiag = max($maxDiag, abs($md[$i][$i] - 1.0));
        }
        if ($maxDiag > $diagTol) {
            throw new \RuntimeException('Failed to generate a valid correlation matrix');
        }

        return $md;
    }

    /**
     * Compute m @ diag(eigs) @ m^T as a dense dim x dim PHP array.
     *
     * @param array<int, array<int, float>> $m
     * @param list<float>                   $eigs
     *
     * @return array<int, array<int, float>>
     */
    private static function matTimesDiagTimesTranspose(array $m, array $eigs, int $dim): array
    {
        // b = m * diag(eigs): column k of m scaled by eigs[k].
        $b = [];
        for ($i = 0; $i < $dim; $i++) {
            $row = [];
            for ($k = 0; $k < $dim; $k++) {
                $row[$k] = (float) $m[$i][$k] * $eigs[$k];
            }
            $b[$i] = $row;
        }
        // out = b @ m^T  ->  out[i][j] = sum_k b[i][k] * m[j][k]
        $out = [];
        for ($i = 0; $i < $dim; $i++) {
            $oi = [];
            for ($j = 0; $j < $dim; $j++) {
                $s = 0.0;
                for ($k = 0; $k < $dim; $k++) {
                    $s += $b[$i][$k] * (float) $m[$j][$k];
                }
                $oi[$j] = $s;
            }
            $out[$i] = $oi;
        }

        return $out;
    }

    /**
     * 2x2 Givens rotation (c, s) that drives the (i,i) entry to 1 for the symmetric block
     * [[aii, aij], [aij, ajj]], exactly as scipy's _givens_to_1.
     *
     * @return array{0: float, 1: float}
     */
    private static function givensTo1(float $aii, float $ajj, float $aij): array
    {
        $aiid = $aii - 1.0;
        $ajjd = $ajj - 1.0;
        if ($ajjd === 0.0) {
            // ajj == 1, so swap to avoid division by zero.
            return [0.0, 1.0];
        }
        $dd = sqrt(max($aij * $aij - $aiid * $ajjd, 0.0));
        // Choose the sign of t to avoid cancellation [Davies & Higham].
        $t = ($aij + self::copysign($dd, $aij)) / $ajjd;
        $c = 1.0 / sqrt(1.0 + $t * $t);
        if ($c === 0.0) {
            $s = 1.0; // underflow
        } else {
            $s = $c * $t;
        }

        return [$c, $s];
    }

    /**
     * Rotate the psd matrix $m in place to put 1's on the diagonal (scipy's _to_corr). Requires the
     * trace to equal the dimension. Mirrors scipy's two in-place BLAS drot calls per Givens step.
     *
     * @param array<int, array<int, float>> $m modified in place
     */
    private static function toCorr(array &$m, int $dim): void
    {
        for ($i = 0; $i < $dim - 1; $i++) {
            $mii = $m[$i][$i];
            if ($mii === 1.0) {
                continue;
            }
            $j = $i + 1;
            if ($mii > 1.0) {
                for ($jj = $i + 1; $jj < $dim; $jj++) {
                    $j = $jj;
                    if ($m[$jj][$jj] < 1.0) {
                        break;
                    }
                }
            } else {
                for ($jj = $i + 1; $jj < $dim; $jj++) {
                    $j = $jj;
                    if ($m[$jj][$jj] > 1.0) {
                        break;
                    }
                }
            }

            [$c, $s] = self::givensTo1($m[$i][$i], $m[$j][$j], $m[$i][$j]);

            // scipy applies g^T m g via two drot passes with the s argument negated:
            //   drot(rows i,j)  then  drot(cols i,j).  Reference BLAS drot on (x, y):
            //     x' = c*x + sr*y ;  y' = c*y - sr*x    (here sr = -s)
            $sr = -$s;

            // Pass 1: rotate row i against row j (over all columns k).
            for ($k = 0; $k < $dim; $k++) {
                $x = $m[$i][$k];
                $y = $m[$j][$k];
                $m[$i][$k] = $c * $x + $sr * $y;
                $m[$j][$k] = $c * $y - $sr * $x;
            }
            // Pass 2: rotate column i against column j (over all rows k), on the updated matrix.
            for ($k = 0; $k < $dim; $k++) {
                $x = $m[$k][$i];
                $y = $m[$k][$j];
                $m[$k][$i] = $c * $x + $sr * $y;
                $m[$k][$j] = $c * $y - $sr * $x;
            }
        }
    }

    private static function copysign(float $mag, float $sgn): float
    {
        $m = abs($mag);

        return ($sgn < 0.0 || ($sgn === 0.0 && (1.0 / $sgn) < 0.0)) ? -$m : $m;
    }
}
