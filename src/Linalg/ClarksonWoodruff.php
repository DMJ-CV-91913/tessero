<?php

declare(strict_types=1);

namespace Tessero\Linalg;

use Tessero\NDArray;
use Tessero\Random\Generator;

/**
 * scipy.linalg.clarkson_woodruff_transform — a CountSketch of a matrix.
 *
 * Given A of shape (n, d), produces A' of shape (sketch_size, d) such that ||Ax|| ≈ ||A'x|| with
 * high probability. The sketch matrix S is (sketch_size, n): each column j has a single nonzero at a
 * random row, set to +1 or -1 with equal probability; A' = S @ A.
 *
 * scipy's cwt_matrix draws, in order from the Generator:
 *   rows  = rng.integers(0, sketch_size, n)      # target row for each column
 *   signs = rng.choice([1, -1], n)               # +1/-1 for each column
 * {@see Generator} reproduces both draws bit-for-bit (integers via Lemire, choice-with-replacement
 * via integers), so the sketch matches scipy exactly (the only residual is ~1e-15 float-summation
 * ordering in S @ A).
 *
 * Reproduces scipy BIT-FOR-BIT when called with the same SEED as numpy.random.default_rng(SEED)
 * (an int seed to scipy would build a legacy MT19937 RandomState and NOT match — fixtures use
 * default_rng(seed)). Pure PHP, host-independent.
 */
final class ClarksonWoodruff
{
    /**
     * @param array<array<float>>|NDArray $inputMatrix A of shape (n, d)
     * @param int                         $sketchSize  number of rows of the sketch
     * @param int|list<int>|Generator|null $rng         seed for a PCG64 Generator, or a Generator
     *
     * @return NDArray the sketch A' of shape (sketch_size, d), float64
     */
    public static function transform(array|NDArray $inputMatrix, int $sketchSize, int|array|Generator|null $rng = null): NDArray
    {
        $A = ($inputMatrix instanceof NDArray) ? $inputMatrix->toArray() : $inputMatrix;
        $A = array_values($A);
        $n = count($A);
        if ($n === 0) {
            throw new \InvalidArgumentException('clarkson_woodruff_transform: empty input matrix.');
        }
        // Each row is the d-vector of a sample; normalise to float.
        $d = is_array($A[0]) ? count($A[0]) : 1;

        $gen = ($rng instanceof Generator) ? $rng : Generator::defaultRng($rng);

        // Draw order matches scipy.linalg._sketches.cwt_matrix exactly.
        $rows = array_map('intval', $gen->integers(0, $sketchSize, [$n])->toArray());
        $signs = array_map('floatval', $gen->choice([1, -1], [$n])->toArray());

        // S' = S @ A: row `rows[j]` of the sketch gets signs[j] * A[j] added.
        $out = [];
        for ($r = 0; $r < $sketchSize; $r++) {
            $out[$r] = array_fill(0, $d, 0.0);
        }
        for ($j = 0; $j < $n; $j++) {
            $target = $rows[$j];
            $s = $signs[$j];
            $arow = is_array($A[$j]) ? $A[$j] : [$A[$j]];
            for ($c = 0; $c < $d; $c++) {
                $out[$target][$c] += $s * (float) $arow[$c];
            }
        }

        return NDArray::array($out, 'float64');
    }
}
