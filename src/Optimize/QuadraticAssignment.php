<?php

declare(strict_types=1);

namespace Tessero\Optimize;

use Tessero\Exceptions\TesseroException;

/**
 * scipy.optimize.quadratic_assignment(A, B, method='faq'): the Fast Approximate QAP (FAQ) algorithm of
 * Vogelstein et al. (2015) for the quadratic assignment / graph matching problem
 *
 *     minimise  trace(A^T P B P^T)   over permutation matrices P   (maximise when $maximize = true).
 *
 * This is a Frank-Wolfe relaxation on the Birkhoff polytope: each iteration linearises the objective, finds a
 * descent direction by a linear assignment on the gradient, and takes a closed-form quadratic line-search step.
 * The default barycenter start (P0 = J/n, all entries 1/n) makes the whole trajectory deterministic with no RNG,
 * so it reproduces scipy's FAQ result exactly. Only the no-seed case (partial_match = None) is implemented, which
 * is scipy's default: then n_seeds = 0, the const / seed blocks vanish, A22 = A and B22 = B.
 *
 * @see Assignment::linearSumAssignment the linear-assignment subproblem solver used for direction + projection
 */
final class QuadraticAssignment
{
    /**
     * Solve the quadratic assignment problem approximately by FAQ (P0 = 'barycenter', no RNG).
     *
     * @param list<list<float>> $A  square cost matrix
     * @param list<list<float>> $B  square cost matrix (same shape as $A)
     * @return array{col_ind: list<int>, fun: float, nit: int}
     *         col_ind = the best permutation found of the nodes of B; fun = objective value; nit = Frank-Wolfe iterations
     */
    public static function faq(array $A, array $B, bool $maximize = false, int $maxiter = 30, float $tol = 0.03): array
    {
        $n = count($A);
        foreach ($A as $row) {
            if (count($row) !== $n) {
                throw new TesseroException('`A` must be square.');
            }
        }
        if (count($B) !== $n) {
            throw new TesseroException('`A` and `B` matrices must be of equal size.');
        }
        foreach ($B as $row) {
            if (count($row) !== $n) {
                throw new TesseroException('`B` must be square.');
            }
        }
        if ($maxiter <= 0) {
            throw new TesseroException("'maxiter' must be a positive integer.");
        }
        if ($tol <= 0) {
            throw new TesseroException("'tol' must be a positive float.");
        }

        $A = self::toFloat($A);
        $B = self::toFloat($B);

        // trivial case: single node (or empty); the only permutation is the identity
        if ($n === 0) {
            return ['col_ind' => [], 'fun' => 0.0, 'nit' => 0];
        }

        $objScalar = $maximize ? -1.0 : 1.0;

        // No seeds (partial_match = None): A22 = A, B22 = B, const_sum = 0, n_unseed = n.
        $A22 = $A;
        $B22 = $B;
        $B22T = self::transpose($B22);   // B22^T
        $A22T = self::transpose($A22);   // A22^T

        // [1] Algorithm 1 Line 1 - barycenter initialisation: P0 = J / n (all entries 1/n)
        $P = [];
        $inv = 1.0 / $n;
        for ($i = 0; $i < $n; $i++) {
            $P[$i] = array_fill(0, $n, $inv);
        }

        $nIter = 0;
        for ($nIter = 1; $nIter <= $maxiter; $nIter++) {
            // [1] Line 3 - gradient of f(P) = -tr(A P B^T P^T): grad = A22 @ P @ B22^T + A22^T @ P @ B22.
            // Match numpy's left-to-right association exactly ((A@P)@B^T), as the chosen assignment direction can
            // hinge on the last-ULP ordering of the gradient when rows are near-degenerate (e.g. symmetric inputs).
            $grad1 = self::matmul(self::matmul($A22, $P), $B22T);
            $grad2 = self::matmul(self::matmul($A22T, $P), $B22);
            $grad = self::add($grad1, $grad2);

            // [1] Line 4 - direction Q from the linear assignment on the gradient
            [, $cols] = Assignment::linearSumAssignment($grad, $maximize);
            // Q = I[cols]: row i of Q is the e_{cols[i]} basis vector (a permutation matrix)

            // [1] Line 5 - closed-form quadratic line-search step size.
            // Objective along P + x(Q - P) is a x^2 + b x + c; with R = P - Q (note the sign),
            // follow scipy's exact algebra so the trajectory matches.
            $R = self::sub($P, $Q = self::permMatrix($cols, $n));

            // b22a = (A22^T @ R  *  (B22^T)[cols]).sum()
            $AR22 = self::matmul($A22T, $R);             // A22.T @ R
            $BR22 = self::matmul($B22, self::transpose($R)); // B22 @ R.T
            $B22T_cols = self::rowGather($B22T, $cols);  // B22.T[cols]
            $b22a = self::elemDotSum($AR22, $B22T_cols);
            $BR22_cols = self::rowGather($BR22, $cols);  // BR22[cols]
            $b22b = self::elemDotSum($A22, $BR22_cols);
            // a = (AR22.T * BR22).sum()
            $a = self::elemDotSum(self::transpose($AR22), $BR22);
            // no seeds => b21 = b12 = 0
            $b = $b22a + $b22b;

            $as = $a * $objScalar;
            $crit = ($a != 0.0) ? (-$b / (2.0 * $a)) : INF;
            if ($as > 0 && $crit >= 0.0 && $crit <= 1.0) {
                $alpha = -$b / (2.0 * $a);
            } else {
                // np.argmin([0, (b + a)*obj_func_scalar]): 0 if first <= second else 1
                $alpha = (0.0 <= ($b + $a) * $objScalar) ? 0.0 : 1.0;
            }

            // [1] Line 6 - update P_{i+1} = alpha*P + (1-alpha)*Q, and the Frank-Wolfe stop test
            $Pnew = [];
            $diffSq = 0.0;
            for ($i = 0; $i < $n; $i++) {
                $Pnew[$i] = [];
                for ($j = 0; $j < $n; $j++) {
                    $v = $alpha * $P[$i][$j] + (1.0 - $alpha) * $Q[$i][$j];
                    $Pnew[$i][$j] = $v;
                    $d = $P[$i][$j] - $v;
                    $diffSq += $d * $d;
                }
            }
            $P = $Pnew;
            if (sqrt($diffSq) / sqrt((float) $n) < $tol) {
                break;
            }
        }
        if ($nIter > $maxiter) {
            $nIter = $maxiter;    // loop ran to completion without an early break
        }

        // [1] Line 8 - project onto permutation matrices: maximise assignment on P
        [, $col] = Assignment::linearSumAssignment($P, true);
        // no seeds => unshuffled_perm = col directly (perm_A = perm_B = range(n))
        $colInd = $col;

        $fun = self::calcScore($A, $B, $colInd);

        return ['col_ind' => array_map('intval', $colInd), 'fun' => $fun, 'nit' => $nIter];
    }

    /**
     * Objective trace(A^T P B P^T), computed as scipy's _calc_score: sum(A * B[perm][:, perm]).
     *
     * @param list<list<float>> $A
     * @param list<list<float>> $B
     * @param list<int> $perm
     */
    private static function calcScore(array $A, array $B, array $perm): float
    {
        $n = count($A);
        $s = 0.0;
        for ($i = 0; $i < $n; $i++) {
            $pi = $perm[$i];
            for ($j = 0; $j < $n; $j++) {
                $s += $A[$i][$j] * $B[$pi][$perm[$j]];
            }
        }

        return $s;
    }

    /** @param list<list<float>> $M @return list<list<float>> */
    private static function toFloat(array $M): array
    {
        return array_map(static fn (array $r): array => array_map('floatval', array_values($r)), array_values($M));
    }

    /**
     * Permutation matrix Q = I[cols]: row i is the basis vector e_{cols[i]}.
     *
     * @param list<int> $cols
     * @return list<list<float>>
     */
    private static function permMatrix(array $cols, int $n): array
    {
        $Q = [];
        for ($i = 0; $i < $n; $i++) {
            $row = array_fill(0, $n, 0.0);
            $row[$cols[$i]] = 1.0;
            $Q[$i] = $row;
        }

        return $Q;
    }

    /** C = A @ B. @param list<list<float>> $Aa @param list<list<float>> $Bb @return list<list<float>> */
    private static function matmul(array $Aa, array $Bb): array
    {
        $n = count($Aa);
        $k = count($Bb);
        $m = $k > 0 ? count($Bb[0]) : 0;
        $C = [];
        for ($i = 0; $i < $n; $i++) {
            $Ai = $Aa[$i];
            $Ci = array_fill(0, $m, 0.0);
            for ($p = 0; $p < $k; $p++) {
                $aip = $Ai[$p];
                if ($aip === 0.0) {
                    continue;
                }
                $Bp = $Bb[$p];
                for ($j = 0; $j < $m; $j++) {
                    $Ci[$j] += $aip * $Bp[$j];
                }
            }
            $C[$i] = $Ci;
        }

        return $C;
    }

    /** @param list<list<float>> $M @return list<list<float>> */
    private static function transpose(array $M): array
    {
        $n = count($M);
        $m = $n > 0 ? count($M[0]) : 0;
        $T = [];
        for ($j = 0; $j < $m; $j++) {
            $T[$j] = [];
            for ($i = 0; $i < $n; $i++) {
                $T[$j][$i] = $M[$i][$j];
            }
        }

        return $T;
    }

    /** @param list<list<float>> $Aa @param list<list<float>> $Bb @return list<list<float>> */
    private static function add(array $Aa, array $Bb): array
    {
        $out = [];
        foreach ($Aa as $i => $row) {
            $out[$i] = [];
            foreach ($row as $j => $v) {
                $out[$i][$j] = $v + $Bb[$i][$j];
            }
        }

        return $out;
    }

    /** @param list<list<float>> $Aa @param list<list<float>> $Bb @return list<list<float>> */
    private static function sub(array $Aa, array $Bb): array
    {
        $out = [];
        foreach ($Aa as $i => $row) {
            $out[$i] = [];
            foreach ($row as $j => $v) {
                $out[$i][$j] = $v - $Bb[$i][$j];
            }
        }

        return $out;
    }

    /**
     * Gather rows of M in the order given by $idx (numpy M[idx]).
     *
     * @param list<list<float>> $M
     * @param list<int> $idx
     * @return list<list<float>>
     */
    private static function rowGather(array $M, array $idx): array
    {
        $out = [];
        foreach ($idx as $k => $i) {
            $out[$k] = $M[$i];
        }

        return $out;
    }

    /** Elementwise product summed over all entries: (A * B).sum(). @param list<list<float>> $Aa @param list<list<float>> $Bb */
    private static function elemDotSum(array $Aa, array $Bb): float
    {
        $s = 0.0;
        foreach ($Aa as $i => $row) {
            $Bi = $Bb[$i];
            foreach ($row as $j => $v) {
                $s += $v * $Bi[$j];
            }
        }

        return $s;
    }
}
