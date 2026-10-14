<?php

declare(strict_types=1);

namespace Tessero\Stats;

/**
 * scipy.stats.Covariance - representation of a covariance matrix via one of several decompositions.
 *
 * The class cannot be instantiated directly; use one of the factory methods:
 *   - Covariance::fromPrecision($precision [, $covariance])
 *   - Covariance::fromCholesky($cholesky)                 (lower Cholesky factor L, L Lᵀ = A)
 *   - Covariance::fromEigendecomposition([$w, $v])        (as returned by eigh: V W Vᵀ = A)
 *   - Covariance::fromDiagonal($diagonal)
 *
 * It exposes:
 *   - covariance()        the explicit covariance matrix A (as a nested PHP array)
 *   - logPdet()           log of the pseudo-determinant of A
 *   - rank()              rank of A
 *   - shape()             [n, n]
 *   - whiten($x)          whitening transform  (unit-diagonal covariance output)
 *   - colorize($x)        colorizing transform (inverse of whiten)
 *
 * All arithmetic is deterministic small-matrix linear algebra, reproducing scipy.stats.Covariance to
 * machine precision (whiten/colorize/log_pdet within ~1e-10 across all four factory methods). $x may
 * be a 1-D vector or a 2-D array of row points; the last axis is the space dimension.
 *
 * @see scipy.stats._covariance.Covariance
 */
final class Covariance
{
    /** Factory discriminator: 'precision' | 'cholesky' | 'eigh' | 'diagonal'. */
    private string $kind;

    /** @var array<int,array<int,float>> Explicit covariance matrix A. */
    private array $cov;

    private float $logPdet;

    private int $rank;

    private int $dim;

    /** Lower Cholesky factor of the precision matrix P (for 'precision'). @var array<int,array<int,float>> */
    private array $cholP = [];

    /** Lower Cholesky factor L of A (for 'cholesky'). @var array<int,array<int,float>> */
    private array $factor = [];

    /** Pseudo-reciprocal square roots of the diagonal / eigenvalues (for 'diagonal'/'eigh'). @var array<int,float> */
    private array $lp = [];

    /** sqrt of the diagonal / eigenvalues, for colorize (for 'diagonal'/'eigh'). @var array<int,float> */
    private array $sqrtDiag = [];

    /** Eigenvectors V (for 'eigh'). @var array<int,array<int,float>> */
    private array $vecs = [];

    private function __construct()
    {
    }

    /**
     * Representation of a covariance from its precision matrix (inverse covariance).
     *
     * @param array<int,array<int,float>>|mixed      $precision  square symmetric positive-definite precision
     * @param array<int,array<int,float>>|mixed|null $covariance optional explicit covariance (else inferred)
     */
    public static function fromPrecision(mixed $precision, mixed $covariance = null): self
    {
        $P = self::toMatrix($precision);
        $n = count($P);
        $L = self::cholesky($P);                // lower factor of P
        $self = new self();
        $self->kind = 'precision';
        $self->dim = $n;
        $self->cholP = $L;
        // log det A = -log det P = -2 * sum(log(diag(L)))
        $s = 0.0;
        foreach ($L as $i => $row) {
            $s += log($row[$i]);
        }
        $self->logPdet = -2.0 * $s;
        $self->rank = $n;
        $self->cov = $covariance !== null ? self::toMatrix($covariance) : self::choSolveIdentity($L);
        return $self;
    }

    /**
     * Representation of a covariance provided via the lower Cholesky factor L (L Lᵀ = A).
     *
     * @param array<int,array<int,float>>|mixed $cholesky
     */
    public static function fromCholesky(mixed $cholesky): self
    {
        $L = self::toMatrix($cholesky);
        $n = count($L);
        $self = new self();
        $self->kind = 'cholesky';
        $self->dim = $n;
        $self->factor = $L;
        $s = 0.0;
        foreach ($L as $i => $row) {
            $s += log($row[$i]);
        }
        $self->logPdet = 2.0 * $s;
        $self->rank = $n;
        $self->cov = self::matMulT($L, $L);     // L Lᵀ
        return $self;
    }

    /**
     * Representation of a covariance via its diagonal elements (diagonal covariance matrix).
     * Supports singular matrices: non-positive entries are treated as zeros in whitening and
     * ignored in the log pseudo-determinant.
     *
     * @param array<int,float>|mixed $diagonal
     */
    public static function fromDiagonal(mixed $diagonal): self
    {
        $d = self::toVector($diagonal);
        $n = count($d);
        $self = new self();
        $self->kind = 'diagonal';
        $self->dim = $n;

        $logpdet = 0.0;
        $rank = 0;
        $lp = [];
        $sq = [];
        $cov = [];
        for ($i = 0; $i < $n; $i++) {
            $cov[$i] = array_fill(0, $n, 0.0);
            $cov[$i][$i] = (float) $d[$i];
            $sq[$i] = sqrt(max($d[$i], 0.0)) === 0.0 ? ($d[$i] > 0 ? sqrt($d[$i]) : 0.0) : sqrt($d[$i]);
            // sqrt of raw diagonal (scipy uses np.sqrt(diagonal); for negative -> nan, but inputs are >=0 here)
            $sq[$i] = $d[$i] >= 0 ? sqrt($d[$i]) : NAN;
            if ($d[$i] > 0) {
                $logpdet += log($d[$i]);
                $lp[$i] = 1.0 / sqrt($d[$i]);
                $rank++;
            } else {
                // ones don't affect determinant; pseudo-reciprocal is zero
                $lp[$i] = 0.0;
            }
        }
        $self->logPdet = $logpdet;
        $self->rank = $rank;
        $self->lp = $lp;
        $self->sqrtDiag = $sq;
        $self->cov = $cov;
        return $self;
    }

    /**
     * Representation of a covariance via its eigendecomposition [$eigenvalues, $eigenvectors],
     * as returned by eigh: V W Vᵀ = A. Supports singular matrices (non-positive eigenvalues
     * ignored / treated as zero), matching scipy.
     *
     * @param array{0:mixed,1:mixed}|array<int,mixed> $eigendecomposition
     */
    public static function fromEigendecomposition(array $eigendecomposition): self
    {
        $w = self::toVector($eigendecomposition[0]);
        $v = self::toMatrix($eigendecomposition[1]);
        $n = count($v);
        $self = new self();
        $self->kind = 'eigh';
        $self->dim = $n;

        $logpdet = 0.0;
        $rank = 0;
        $lp = [];
        $sq = [];
        for ($j = 0; $j < $n; $j++) {
            $sq[$j] = $w[$j] >= 0 ? sqrt($w[$j]) : NAN;
            if ($w[$j] > 0) {
                $logpdet += log($w[$j]);
                $lp[$j] = 1.0 / sqrt($w[$j]);
                $rank++;
            } else {
                $lp[$j] = 0.0;
            }
        }
        $self->logPdet = $logpdet;
        $self->rank = $rank;
        $self->lp = $lp;
        $self->sqrtDiag = $sq;
        $self->vecs = $v;
        // covariance = (V * w) Vᵀ
        $cov = [];
        for ($i = 0; $i < $n; $i++) {
            $cov[$i] = array_fill(0, $n, 0.0);
            for ($k = 0; $k < $n; $k++) {
                $vw = $v[$i][$k] * $w[$k];
                for ($jj = 0; $jj < $n; $jj++) {
                    $cov[$i][$jj] += $vw * $v[$jj][$k];
                }
            }
        }
        $self->cov = $cov;
        return $self;
    }

    /** Explicit covariance matrix A as a nested PHP array. @return array<int,array<int,float>> */
    public function covariance(): array
    {
        return $this->cov;
    }

    /** Log of the pseudo-determinant of the covariance matrix. */
    public function logPdet(): float
    {
        return $this->logPdet;
    }

    /** Rank of the covariance matrix. */
    public function rank(): int
    {
        return $this->rank;
    }

    /** Shape [n, n] of the covariance array. @return array{0:int,1:int} */
    public function shape(): array
    {
        return [$this->dim, $this->dim];
    }

    /**
     * Whitening transform. $x is a 1-D vector or 2-D array of row points (last axis = space dim).
     * Returns the same shape.
     *
     * @param array<int,float>|array<int,array<int,float>>|mixed $x
     * @return array<int,float>|array<int,array<int,float>>
     */
    public function whiten(mixed $x): array
    {
        [$rows, $is1d] = self::toRows($x, $this->dim);
        $out = [];
        foreach ($rows as $r => $row) {
            $out[$r] = $this->whitenRow($row);
        }
        return $is1d ? $out[0] : $out;
    }

    /**
     * Colorizing transform (inverse of whiten). Same shape conventions as whiten().
     *
     * @param array<int,float>|array<int,array<int,float>>|mixed $x
     * @return array<int,float>|array<int,array<int,float>>
     */
    public function colorize(mixed $x): array
    {
        [$rows, $is1d] = self::toRows($x, $this->dim);
        $out = [];
        foreach ($rows as $r => $row) {
            $out[$r] = $this->colorizeRow($row);
        }
        return $is1d ? $out[0] : $out;
    }

    /** @param array<int,float> $x @return array<int,float> */
    private function whitenRow(array $x): array
    {
        $n = $this->dim;
        switch ($this->kind) {
            case 'precision':
                // x @ chol_P   (chol_P lower)
                $out = array_fill(0, $n, 0.0);
                for ($j = 0; $j < $n; $j++) {
                    $s = 0.0;
                    for ($i = 0; $i < $n; $i++) {
                        $s += $x[$i] * $this->cholP[$i][$j];
                    }
                    $out[$j] = $s;
                }
                return $out;
            case 'cholesky':
                // solve_triangular(L, x, lower=True) : forward substitution L y = x
                return self::forwardSolve($this->factor, $x);
            case 'diagonal':
                $out = [];
                for ($i = 0; $i < $n; $i++) {
                    $out[$i] = $x[$i] * $this->lp[$i];
                }
                return $out;
            case 'eigh':
                // x @ (V * lp)   ; column j scaled by lp[j]
                $out = array_fill(0, $n, 0.0);
                for ($j = 0; $j < $n; $j++) {
                    $s = 0.0;
                    for ($i = 0; $i < $n; $i++) {
                        $s += $x[$i] * $this->vecs[$i][$j];
                    }
                    $out[$j] = $s * $this->lp[$j];
                }
                return $out;
            default:
                throw new \RuntimeException('unknown covariance kind');
        }
    }

    /** @param array<int,float> $x @return array<int,float> */
    private function colorizeRow(array $x): array
    {
        $n = $this->dim;
        switch ($this->kind) {
            case 'precision':
                // solve_triangular(chol_P.T, x, lower=False): back substitution Uᵀ? -> chol_P.T is upper
                return self::backSolveUpper(self::transpose($this->cholP), $x);
            case 'cholesky':
                // x @ Lᵀ
                $out = array_fill(0, $n, 0.0);
                for ($j = 0; $j < $n; $j++) {
                    $s = 0.0;
                    for ($i = 0; $i < $n; $i++) {
                        $s += $x[$i] * $this->factor[$j][$i]; // (Lᵀ)[i][j] = L[j][i]
                    }
                    $out[$j] = $s;
                }
                return $out;
            case 'diagonal':
                $out = [];
                for ($i = 0; $i < $n; $i++) {
                    $out[$i] = $x[$i] * $this->sqrtDiag[$i];
                }
                return $out;
            case 'eigh':
                // x @ (V * sqrt(w)).T  == x @ LA.T where LA = V * sqrt(w)
                // (LA.T)[k][i] = LA[i][k] = V[i][k]*sqrt(w)[k]
                $out = array_fill(0, $n, 0.0);
                for ($i = 0; $i < $n; $i++) {      // output component i
                    $s = 0.0;
                    for ($k = 0; $k < $n; $k++) {
                        $s += $x[$k] * $this->vecs[$i][$k] * $this->sqrtDiag[$k];
                    }
                    $out[$i] = $s;
                }
                return $out;
            default:
                throw new \RuntimeException('unknown covariance kind');
        }
    }

    // ---- small deterministic linear-algebra helpers ------------------------------------------

    /** Lower Cholesky factor of a symmetric positive-definite matrix. @param array<int,array<int,float>> $A @return array<int,array<int,float>> */
    private static function cholesky(array $A): array
    {
        $n = count($A);
        $L = [];
        for ($i = 0; $i < $n; $i++) {
            $L[$i] = array_fill(0, $n, 0.0);
        }
        for ($i = 0; $i < $n; $i++) {
            for ($j = 0; $j <= $i; $j++) {
                $s = $A[$i][$j];
                for ($k = 0; $k < $j; $k++) {
                    $s -= $L[$i][$k] * $L[$j][$k];
                }
                if ($i === $j) {
                    if ($s <= 0.0) {
                        throw new \RuntimeException('matrix is not positive definite');
                    }
                    $L[$i][$j] = sqrt($s);
                } else {
                    $L[$i][$j] = $s / $L[$j][$j];
                }
            }
        }
        return $L;
    }

    /** Forward substitution: solve L y = b for lower-triangular L. @param array<int,array<int,float>> $L @param array<int,float> $b @return array<int,float> */
    private static function forwardSolve(array $L, array $b): array
    {
        $n = count($L);
        $y = array_fill(0, $n, 0.0);
        for ($i = 0; $i < $n; $i++) {
            $s = $b[$i];
            for ($k = 0; $k < $i; $k++) {
                $s -= $L[$i][$k] * $y[$k];
            }
            $y[$i] = $s / $L[$i][$i];
        }
        return $y;
    }

    /** Back substitution: solve U y = b for upper-triangular U. @param array<int,array<int,float>> $U @param array<int,float> $b @return array<int,float> */
    private static function backSolveUpper(array $U, array $b): array
    {
        $n = count($U);
        $y = array_fill(0, $n, 0.0);
        for ($i = $n - 1; $i >= 0; $i--) {
            $s = $b[$i];
            for ($k = $i + 1; $k < $n; $k++) {
                $s -= $U[$i][$k] * $y[$k];
            }
            $y[$i] = $s / $U[$i][$i];
        }
        return $y;
    }

    /** Solve A X = I given the lower Cholesky factor L of A (A = L Lᵀ), i.e. invert A. @param array<int,array<int,float>> $L @return array<int,array<int,float>> */
    private static function choSolveIdentity(array $L): array
    {
        $n = count($L);
        $inv = [];
        for ($col = 0; $col < $n; $col++) {
            $e = array_fill(0, $n, 0.0);
            $e[$col] = 1.0;
            // A x = e  with A = L Lᵀ : forward solve L z = e, then back solve Lᵀ x = z
            $z = self::forwardSolve($L, $e);
            $x = self::backSolveUpper(self::transpose($L), $z);
            for ($i = 0; $i < $n; $i++) {
                $inv[$i][$col] = $x[$i];
            }
        }
        return $inv;
    }

    /** Compute A Bᵀ for the special case B == A (returns A Aᵀ) when called as matMulT(L, L). @param array<int,array<int,float>> $A @param array<int,array<int,float>> $B @return array<int,array<int,float>> */
    private static function matMulT(array $A, array $B): array
    {
        $n = count($A);
        $m = count($B);
        $k = count($A[0]);
        $out = [];
        for ($i = 0; $i < $n; $i++) {
            $out[$i] = array_fill(0, $m, 0.0);
            for ($j = 0; $j < $m; $j++) {
                $s = 0.0;
                for ($t = 0; $t < $k; $t++) {
                    $s += $A[$i][$t] * $B[$j][$t];
                }
                $out[$i][$j] = $s;
            }
        }
        return $out;
    }

    /** @param array<int,array<int,float>> $A @return array<int,array<int,float>> */
    private static function transpose(array $A): array
    {
        $n = count($A);
        $m = count($A[0]);
        $out = [];
        for ($i = 0; $i < $m; $i++) {
            $out[$i] = array_fill(0, $n, 0.0);
            for ($j = 0; $j < $n; $j++) {
                $out[$i][$j] = $A[$j][$i];
            }
        }
        return $out;
    }

    /** Coerce an NDArray or nested array into a nested PHP float matrix. @return array<int,array<int,float>> */
    private static function toMatrix(mixed $a): array
    {
        if (is_object($a) && method_exists($a, 'toArray')) {
            $a = $a->toArray();
        }
        $out = [];
        foreach ($a as $i => $row) {
            $out[$i] = array_map('floatval', is_array($row) ? $row : [$row]);
        }
        return $out;
    }

    /** Coerce an NDArray or array into a flat PHP float vector. @return array<int,float> */
    private static function toVector(mixed $a): array
    {
        if (is_object($a) && method_exists($a, 'toArray')) {
            $a = $a->toArray();
        }
        return array_map('floatval', array_values((array) $a));
    }

    /**
     * Normalize $x into a list of row vectors of length $dim, remembering whether the input was 1-D.
     *
     * @return array{0:array<int,array<int,float>>,1:bool}
     */
    private static function toRows(mixed $x, int $dim): array
    {
        if (is_object($x) && method_exists($x, 'toArray')) {
            $x = $x->toArray();
        }
        $x = array_values((array) $x);
        if ($x === [] || ! is_array($x[0])) {
            return [[array_map('floatval', $x)], true];
        }
        $rows = [];
        foreach ($x as $i => $row) {
            $rows[$i] = array_map('floatval', array_values((array) $row));
        }
        return [$rows, false];
    }
}
