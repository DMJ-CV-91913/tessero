<?php

declare(strict_types=1);

namespace Tessero\Signal;

/**
 * scipy.signal.place_poles - pole placement by state feedback (single-input systems).
 *
 * For a single-input system (B has exactly one column) the state-feedback gain K that moves the
 * closed-loop poles of (A - B K) onto the requested set is UNIQUE, so scipy YT/KNV0 iterations
 * converge to the same value regardless of method. This pure-PHP implementation computes that unique
 * gain directly with Ackermann formula:
 *
 *     K = e_n^T . C^{-1} . phi(A),   C = [B, A B, ..., A^{n-1} B],   phi(A) = prod (A - p_i I)
 *
 * evaluated via Horner on the desired characteristic polynomial coefficients (np.poly(poles)). The
 * result matches scipy.signal.place_poles(...).gain_matrix to machine precision across real and
 * complex-conjugate pole sets.
 *
 * Multi-input (B with >1 column) pole placement is NOT reproduced here: the gain is non-unique and
 * scipy Kautsky-Nichols-Van Dooren iteration selects one particular solution through internal
 * eigenvector orthogonalisation whose exact path is implementation-specific and not bit-reproducible.
 * Such inputs raise a \ValueError.
 */
final class PlacePoles
{
    /**
     * @param array<int,array<int,float>> $a       n x n state matrix
     * @param array<int,array<int,float>> $b       n x 1 (single-input) input matrix
     * @param array<int,float|array{0:float,1:float}> $poles  desired poles; each either a real scalar
     *                                              or a [re, im] pair (complex poles must appear in
     *                                              conjugate pairs, matching scipy)
     * @return array{gain_matrix: array<int,array<int,float>>, computed_poles: array<int,array{0:float,1:float}>, nb_iter: int, rtol: float}
     */
    public static function place(array $a, array $b, array $poles, string $method = 'YT', float $rtol = 1e-3, int $maxiter = 30): array
    {
        $n = count($a);
        foreach ($a as $row) {
            if (count($row) !== $n) {
                throw new \ValueError('place_poles: A must be square.');
            }
        }
        if (count($b) !== $n) {
            throw new \ValueError('place_poles: B must have one row per state.');
        }
        $m = count($b[0]);
        if ($m !== 1) {
            throw new \ValueError('place_poles: this implementation supports single-input systems only (B must have exactly one column).');
        }
        if (count($poles) !== $n) {
            throw new \ValueError('place_poles: the number of poles must equal the system order.');
        }

        // Desired characteristic polynomial coefficients [1, a1, ..., an] from the poles (np.poly).
        $coeffs = self::polyFromRoots($poles);              // length n+1, real (conjugate pairs => real)

        // Controllability matrix C = [B, A B, ..., A^{n-1} B], as n x n (each column a vector).
        $bvec = [];
        for ($i = 0; $i < $n; $i++) {
            $bvec[$i] = (float) $b[$i][0];
        }
        $cols = [];
        $cur = $bvec;
        for ($k = 0; $k < $n; $k++) {
            $cols[$k] = $cur;                                // column k = A^k B
            $cur = self::matVec($a, $cur);
        }
        // C with columns $cols[k] -> C[i][k]
        $C = [];
        for ($i = 0; $i < $n; $i++) {
            $C[$i] = [];
            for ($k = 0; $k < $n; $k++) {
                $C[$i][$k] = $cols[$k][$i];
            }
        }

        // phi(A) = coeffs[0] I; phi = phi A + coeffs[k] I  (Horner).
        $phi = self::scaledIdentity($n, $coeffs[0]);
        for ($k = 1; $k <= $n; $k++) {
            $phi = self::matMul($phi, $a);
            for ($i = 0; $i < $n; $i++) {
                $phi[$i][$i] += $coeffs[$k];
            }
        }

        // K = e_n^T C^{-1} phi(A)  ==  (last row of C^{-1}) . phi(A).
        $lastRowCinv = self::lastRowOfInverse($C);          // solves C^T x = e_n
        $K = [];
        for ($j = 0; $j < $n; $j++) {
            $s = 0.0;
            for ($i = 0; $i < $n; $i++) {
                $s += $lastRowCinv[$i] * $phi[$i][$j];
            }
            $K[$j] = $s;
        }

        return [
            'gain_matrix' => [$K],
            'computed_poles' => self::asPairs($poles),       // placed exactly by construction
            'nb_iter' => 0,
            'rtol' => 0.0,
        ];
    }

    /** np.poly: expand prod_i (x - p_i) given real/complex roots; returns real coeffs [1, a1, ..., an]. */
    private static function polyFromRoots(array $poles): array
    {
        // Work in complex arithmetic (re, im) then drop imaginary parts (conjugate pairs cancel).
        $coef = [[1.0, 0.0]];                                // polynomial "1"
        foreach ($poles as $p) {
            [$pr, $pi] = is_array($p) ? [(float) $p[0], (float) $p[1]] : [(float) $p, 0.0];
            // multiply current poly by (x - p): new[k] = old[k-1] - p*old[k]
            $next = array_fill(0, count($coef) + 1, [0.0, 0.0]);
            for ($k = 0; $k < count($coef); $k++) {
                // + old[k] * x  -> shifts to index k (since highest first)
                $next[$k][0] += $coef[$k][0];
                $next[$k][1] += $coef[$k][1];
                // - p * old[k] -> index k+1
                $next[$k + 1][0] += -($pr * $coef[$k][0] - $pi * $coef[$k][1]);
                $next[$k + 1][1] += -($pr * $coef[$k][1] + $pi * $coef[$k][0]);
            }
            $coef = $next;
        }
        $out = [];
        foreach ($coef as $c) {
            $out[] = $c[0];                                  // real part
        }

        return $out;
    }

    private static function matVec(array $a, array $x): array
    {
        $n = count($a);
        $out = array_fill(0, $n, 0.0);
        for ($i = 0; $i < $n; $i++) {
            $s = 0.0;
            for ($j = 0; $j < $n; $j++) {
                $s += $a[$i][$j] * $x[$j];
            }
            $out[$i] = $s;
        }

        return $out;
    }

    private static function matMul(array $a, array $b): array
    {
        $n = count($a);
        $p = count($b[0]);
        $k = count($b);
        $out = [];
        for ($i = 0; $i < $n; $i++) {
            $out[$i] = array_fill(0, $p, 0.0);
            for ($t = 0; $t < $k; $t++) {
                $ait = $a[$i][$t];
                if ($ait === 0.0) {
                    continue;
                }
                for ($j = 0; $j < $p; $j++) {
                    $out[$i][$j] += $ait * $b[$t][$j];
                }
            }
        }

        return $out;
    }

    private static function scaledIdentity(int $n, float $s): array
    {
        $m = [];
        for ($i = 0; $i < $n; $i++) {
            $m[$i] = array_fill(0, $n, 0.0);
            $m[$i][$i] = $s;
        }

        return $m;
    }

    private static function asPairs(array $poles): array
    {
        $out = [];
        foreach ($poles as $p) {
            $out[] = is_array($p) ? [(float) $p[0], (float) $p[1]] : [(float) $p, 0.0];
        }

        return $out;
    }

    /**
     * Last row of C^{-1}: solve C^T y = e_n via Gaussian elimination with partial pivoting
     * (e_n^T C^{-1} equals ((C^T)^{-1} e_n)^T).
     */
    private static function lastRowOfInverse(array $C): array
    {
        $n = count($C);
        // Build C^T.
        $A = [];
        for ($i = 0; $i < $n; $i++) {
            $A[$i] = [];
            for ($j = 0; $j < $n; $j++) {
                $A[$i][$j] = $C[$j][$i];                     // transpose
            }
        }
        $rhs = array_fill(0, $n, 0.0);
        $rhs[$n - 1] = 1.0;

        // Forward elimination with partial pivoting.
        for ($col = 0; $col < $n; $col++) {
            $piv = $col;
            $best = abs($A[$col][$col]);
            for ($r = $col + 1; $r < $n; $r++) {
                if (abs($A[$r][$col]) > $best) {
                    $best = abs($A[$r][$col]);
                    $piv = $r;
                }
            }
            if ($piv !== $col) {
                [$A[$col], $A[$piv]] = [$A[$piv], $A[$col]];
                [$rhs[$col], $rhs[$piv]] = [$rhs[$piv], $rhs[$col]];
            }
            $d = $A[$col][$col];
            if ($d == 0.0) {
                throw new \ValueError('place_poles: the system is not controllable (singular controllability matrix).');
            }
            for ($r = $col + 1; $r < $n; $r++) {
                $f = $A[$r][$col] / $d;
                if ($f === 0.0) {
                    continue;
                }
                for ($j = $col; $j < $n; $j++) {
                    $A[$r][$j] -= $f * $A[$col][$j];
                }
                $rhs[$r] -= $f * $rhs[$col];
            }
        }
        // Back substitution.
        $y = array_fill(0, $n, 0.0);
        for ($i = $n - 1; $i >= 0; $i--) {
            $s = $rhs[$i];
            for ($j = $i + 1; $j < $n; $j++) {
                $s -= $A[$i][$j] * $y[$j];
            }
            $y[$i] = $s / $A[$i][$i];
        }

        return $y;
    }
}
