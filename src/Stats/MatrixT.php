<?php

declare(strict_types=1);

namespace Tessero\Stats;

use Tessero\Special;

/**
 * scipy.stats.matrix_t(mean, row_spread, col_spread, df) — the matrix-variate Student-t
 * distribution, as a pure-PHP facade. Only the deterministic density surface (pdf / logpdf at a
 * single matrix X) is provided; rvs is RNG-only and out of scope. No kernel of its own, so it is
 * bit-identical on both backends.
 *
 * With m×n mean M, m×m row-spread Σ_r, n×n col-spread Σ_c and df ν, for an m×n point X
 * (writing A = I_n + (X−M)ᵀ Σ_r⁻¹ (X−M) Σ_c⁻¹):
 *
 *   logpdf(X) = −((ν+m+n−1)/2)·log|A|
 *               + multigammaln((ν+m+n−1)/2, n) − multigammaln((ν+n−1)/2, n)
 *               − (m·n/2)·log π − (n/2)·log|Σ_r| − (m/2)·log|Σ_c|
 *
 * (matching scipy.stats._multivariate.matrix_t_gen._logpdf). pdf(X) = exp(logpdf(X)).
 *
 * All linear algebra on the (small) spread matrices is done in plain PHP (LU with partial
 * pivoting) so the result is independent of the active kernel/BLAS.
 */
final class MatrixT
{
    /** @var list<list<float>> m×n mean */
    private array $mean;

    /** @var list<list<float>> m×m inverse of the row spread */
    private array $invRow;

    /** @var list<list<float>> n×n inverse of the col spread */
    private array $invCol;

    private int $m;

    private int $n;

    private float $logdetRow;

    private float $logdetCol;

    private float $df;

    /**
     * @param array|float $mean 2-D m×n mean matrix
     * @param array|float $rowSpread 2-D m×m SPD row-spread matrix
     * @param array|float $colSpread 2-D n×n SPD col-spread matrix
     */
    public function __construct(mixed $mean, mixed $rowSpread, mixed $colSpread, float $df)
    {
        $M = self::to2d($mean);
        $this->m = count($M);
        $this->n = count($M[0]);
        $this->mean = $M;
        $this->df = $df;

        $R = self::to2d($rowSpread);
        $C = self::to2d($colSpread);
        if (count($R) !== $this->m || count($R[0]) !== $this->m) {
            throw new \ValueError('Stats\\MatrixT: row_spread must be m×m');
        }
        if (count($C) !== $this->n || count($C[0]) !== $this->n) {
            throw new \ValueError('Stats\\MatrixT: col_spread must be n×n');
        }
        [$this->invRow, $this->logdetRow] = self::invAndLogdet($R);
        [$this->invCol, $this->logdetCol] = self::invAndLogdet($C);
    }

    public function logpdf(mixed $x): float
    {
        $X = self::to2d($x);
        if (count($X) !== $this->m || count($X[0]) !== $this->n) {
            throw new \ValueError('Stats\\MatrixT: X must be m×n');
        }
        $m = $this->m;
        $n = $this->n;
        // Xc = X - M  (m×n)
        $Xc = [];
        for ($i = 0; $i < $m; $i++) {
            $row = [];
            for ($j = 0; $j < $n; $j++) {
                $row[$j] = $X[$i][$j] - $this->mean[$i][$j];
            }
            $Xc[$i] = $row;
        }
        // A = I_n + Xcᵀ · invRow · Xc · invCol   (n×n)
        $XcT = self::transpose($Xc);                 // n×m
        $T1 = self::matmul($XcT, $this->invRow);     // n×m
        $T2 = self::matmul($T1, $Xc);                // n×n
        $T3 = self::matmul($T2, $this->invCol);      // n×n
        $A = [];
        for ($i = 0; $i < $n; $i++) {
            $row = [];
            for ($j = 0; $j < $n; $j++) {
                $row[$j] = ($i === $j ? 1.0 : 0.0) + $T3[$i][$j];
            }
            $A[$i] = $row;
        }
        [, $logdetA] = self::invAndLogdet($A, false);

        $df = $this->df;
        $logD = -(($df + $m + $n - 1) / 2.0) * $logdetA;
        $logF = (float) Special::multigammaln(($df + $m + $n - 1) / 2.0, $n)
            - (float) Special::multigammaln(($df + $n - 1) / 2.0, $n)
            - ($m * $n / 2.0) * log(M_PI)
            - ($n / 2.0) * $this->logdetRow
            - ($m / 2.0) * $this->logdetCol;

        return $logD + $logF;
    }

    public function pdf(mixed $x): float
    {
        return exp($this->logpdf($x));
    }

    /** Coerce an NDArray / nested array / scalar into a list-of-lists of floats. */
    private static function to2d(mixed $a): array
    {
        if (is_object($a) && method_exists($a, 'toArray')) {
            $a = $a->toArray();
        }
        if (! is_array($a)) {
            return [[(float) $a]];
        }
        $out = [];
        foreach ($a as $row) {
            if (is_object($row) && method_exists($row, 'toArray')) {
                $row = $row->toArray();
            }
            if (! is_array($row)) {
                $row = [$row];
            }
            $out[] = array_map('floatval', array_values($row));
        }
        return array_values($out);
    }

    private static function transpose(array $a): array
    {
        $r = count($a);
        $c = count($a[0]);
        $t = [];
        for ($j = 0; $j < $c; $j++) {
            $col = [];
            for ($i = 0; $i < $r; $i++) {
                $col[$i] = $a[$i][$j];
            }
            $t[$j] = $col;
        }
        return $t;
    }

    private static function matmul(array $a, array $b): array
    {
        $r = count($a);
        $k = count($b);
        $c = count($b[0]);
        $out = [];
        for ($i = 0; $i < $r; $i++) {
            $row = array_fill(0, $c, 0.0);
            for ($t = 0; $t < $k; $t++) {
                $ait = $a[$i][$t];
                $brow = $b[$t];
                for ($j = 0; $j < $c; $j++) {
                    $row[$j] += $ait * $brow[$j];
                }
            }
            $out[$i] = $row;
        }
        return $out;
    }

    /**
     * LU with partial pivoting: returns [inverse, log|det|]. When $needInverse is false the
     * inverse is returned empty (used when only the log-determinant is needed).
     *
     * @return array{0: array, 1: float}
     */
    private static function invAndLogdet(array $a, bool $needInverse = true): array
    {
        $nn = count($a);
        // working copy
        $lu = [];
        for ($i = 0; $i < $nn; $i++) {
            $lu[$i] = array_map('floatval', $a[$i]);
        }
        $perm = range(0, $nn - 1);
        $logdet = 0.0;
        for ($col = 0; $col < $nn; $col++) {
            // pivot
            $piv = $col;
            $best = abs($lu[$col][$col]);
            for ($r = $col + 1; $r < $nn; $r++) {
                $v = abs($lu[$r][$col]);
                if ($v > $best) {
                    $best = $v;
                    $piv = $r;
                }
            }
            if ($best === 0.0) {
                throw new \ValueError('Stats\\MatrixT: singular spread matrix');
            }
            if ($piv !== $col) {
                $tmp = $lu[$piv];
                $lu[$piv] = $lu[$col];
                $lu[$col] = $tmp;
                $pt = $perm[$piv];
                $perm[$piv] = $perm[$col];
                $perm[$col] = $pt;
            }
            $pivot = $lu[$col][$col];
            $logdet += log(abs($pivot));
            for ($r = $col + 1; $r < $nn; $r++) {
                $f = $lu[$r][$col] / $pivot;
                $lu[$r][$col] = $f;
                for ($c = $col + 1; $c < $nn; $c++) {
                    $lu[$r][$c] -= $f * $lu[$col][$c];
                }
            }
        }
        $inv = [];
        if ($needInverse) {
            // solve A·X = I column by column using the stored LU (unit lower L, upper U)
            for ($e = 0; $e < $nn; $e++) {
                // forward solve L y = P e_e
                $y = array_fill(0, $nn, 0.0);
                for ($i = 0; $i < $nn; $i++) {
                    $bi = ($perm[$i] === $e) ? 1.0 : 0.0;
                    $s = $bi;
                    for ($j = 0; $j < $i; $j++) {
                        $s -= $lu[$i][$j] * $y[$j];
                    }
                    $y[$i] = $s;
                }
                // back solve U x = y
                $xv = array_fill(0, $nn, 0.0);
                for ($i = $nn - 1; $i >= 0; $i--) {
                    $s = $y[$i];
                    for ($j = $i + 1; $j < $nn; $j++) {
                        $s -= $lu[$i][$j] * $xv[$j];
                    }
                    $xv[$i] = $s / $lu[$i][$i];
                }
                for ($i = 0; $i < $nn; $i++) {
                    $inv[$i][$e] = $xv[$i];
                }
            }
        }
        return [$inv, $logdet];
    }
}
