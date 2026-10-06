<?php

declare(strict_types=1);

namespace Tessero\Linalg;

use FFI\CData;
use Tessero\DType;
use Tessero\Exceptions\ConvergenceError;
use Tessero\Exceptions\LinAlgError;
use Tessero\Exceptions\NotPositiveDefinite;
use Tessero\Exceptions\ShapeError;
use Tessero\Exceptions\SingularMatrix;
use Tessero\NDArray;
use Tessero\Native\Blas;

/**
 * numpy.linalg / scipy.linalg on LAPACK (row-major LAPACKE, float64).
 *
 * Every routine works on a private contiguous float64 copy, so inputs are
 * never modified. Functions marked "stacked" accept (..., M, M) arrays and
 * loop over the leading dimensions like NumPy's gufuncs.
 */
final class Linalg
{
    private const ROW = Blas::ROW_MAJOR;

    /** Solve A x = b (stacked). b may be (M,) or (M, K). */
    public static function solve(mixed $a, mixed $b): NDArray
    {
        $A = self::f64($a);
        $B = self::f64($b);
        self::requireSquare($A, 'solve');
        $n = self::last($A);
        if (count($A->shape()) > 2) {
            return self::stacked($A, fn (NDArray $m, int $i): NDArray => self::solve($m, count($B->shape()) > 2 ? $B->slice($i) : $B), $A->shape());
        }
        $vector = count($B->shape()) === 1;
        if (($B->shape()[0] ?? -1) !== $n) {
            throw new ShapeError("solve: b has {$B->shape()[0]} rows, A is {$n}x{$n}.");
        }
        $nrhs = $vector ? 1 : $B->shape()[1];
        $a = $A->copy();
        $x = $B->copy();
        $ipiv = Blas::ffi()->new('int[' . max(1, $n) . ']');
        $info = Blas::call('LAPACKE_dgesv', self::ROW, $n, $nrhs, self::d($a), max(1, $n), $ipiv, self::d($x), max(1, $nrhs));
        self::info($info, 'solve', singular: true);

        return $x;
    }

    /** Inverse (stacked). */
    public static function inv(mixed $a): NDArray
    {
        $A = self::f64($a);
        self::requireSquare($A, 'inv');
        if (count($A->shape()) > 2) {
            return self::stacked($A, fn (NDArray $m): NDArray => self::inv($m), $A->shape());
        }
        $n = $A->shape()[0];
        $inv = $A->copy();
        if ($n === 0) {
            return $inv;
        }
        $ipiv = Blas::ffi()->new("int[{$n}]");
        self::info(Blas::call('LAPACKE_dgetrf', self::ROW, $n, $n, self::d($inv), $n, $ipiv), 'inv', singular: true);
        self::info(Blas::call('LAPACKE_dgetri', self::ROW, $n, self::d($inv), $n, $ipiv), 'inv', singular: true);

        return $inv;
    }

    /** Determinant (stacked). */
    public static function det(mixed $a): float|NDArray
    {
        [$sign, $logdet] = self::slogdet($a);
        if ($sign instanceof NDArray) {
            return $sign->mul($logdet->exp());
        }

        return $sign * exp($logdet);
    }

    /**
     * Sign and natural log of |det| (stacked), robust against overflow.
     *
     * @return array{0: float|NDArray, 1: float|NDArray}
     */
    public static function slogdet(mixed $a): array
    {
        $A = self::f64($a);
        self::requireSquare($A, 'slogdet');
        if (count($A->shape()) > 2) {
            $lead = array_slice($A->shape(), 0, -2);
            $count = (int) array_product($lead);
            $flat = $A->reshape([$count, ...array_slice($A->shape(), -2)]);
            $signs = [];
            $logs = [];
            for ($i = 0; $i < $count; $i++) {
                // no list destructuring of the call result here: PHP 8.4's tracing JIT
                // mis-compiled it in long runs (docs/project/upstream-bugs.md, issue 5)
                $r = self::slogdet($flat->slice($i));
                $signs[] = $r[0];
                $logs[] = $r[1];
            }

            return [NDArray::fromFlat($signs, $lead), NDArray::fromFlat($logs, $lead)];
        }
        $n = $A->shape()[0];
        if ($n === 0) {
            return [1.0, 0.0];
        }
        $lu = $A->copy();
        $ipiv = Blas::ffi()->new("int[{$n}]");
        $info = Blas::call('LAPACKE_dgetrf', self::ROW, $n, $n, self::d($lu), $n, $ipiv);
        if ($info < 0) {
            throw new LinAlgError("slogdet: illegal argument {$info}.");
        }
        if ($info > 0) {
            return [0.0, -INF];
        }
        $diag = $lu->reshape([$n * $n])->slice('::' . ($n + 1))->toList();
        $piv = self::ints($ipiv, $n);
        $sign = 1.0;
        $log = 0.0;
        for ($i = 0; $i < $n; $i++) {
            if ($piv[$i] !== $i + 1) {
                $sign = -$sign;
            }
            if ($diag[$i] < 0) {
                $sign = -$sign;
            }
            $log += log(abs($diag[$i]));
        }

        return [$sign, $log];
    }

    /** Lower-triangular Cholesky factor L with A = L L^T (stacked). */
    public static function cholesky(mixed $a, bool $upper = false): NDArray
    {
        $A = self::f64($a);
        self::requireSquare($A, 'cholesky');
        if (count($A->shape()) > 2) {
            return self::stacked($A, fn (NDArray $m): NDArray => self::cholesky($m, $upper), $A->shape());
        }
        $n = $A->shape()[0];
        $l = $A->copy();
        $info = Blas::call('LAPACKE_dpotrf', self::ROW, $upper ? 'U' : 'L', $n, self::d($l), max(1, $n));
        if ($info > 0) {
            throw new NotPositiveDefinite("cholesky: leading minor of order {$info} is not positive definite.");
        }
        self::info($info, 'cholesky');

        return $upper ? self::triu($l) : self::tril($l);
    }

    /**
     * QR factorisation. mode 'reduced' (Q: M x K, R: K x N, K = min(M, N)),
     * 'complete' (Q: M x M, R: M x N) or 'r' (R only).
     *
     * @return array{0: NDArray, 1: NDArray}|NDArray
     */
    public static function qr(mixed $a, string $mode = 'reduced'): array|NDArray
    {
        $A = self::f64($a);
        self::require2d($A, 'qr');
        [$m, $n] = $A->shape();
        $k = min($m, $n);
        $work = $A->copy();
        $tau = NDArray::empty([max(1, $k)]);
        if ($k > 0) {
            self::info(Blas::call('LAPACKE_dgeqrf', self::ROW, $m, $n, self::d($work), max(1, $n), self::d($tau)), 'qr');
        }
        $rRows = $mode === 'complete' ? $m : $k;
        $r = self::triu($work->slice('0:' . $rRows, ':')->copy());
        if ($mode === 'complete' && $m > $k) {
            $r = NDArray::concatenate([self::triu($work->slice('0:' . $k, ':')->copy()), NDArray::zeros([$m - $k, $n])], 0);
        }
        if ($mode === 'r') {
            return $r;
        }
        $qCols = $mode === 'complete' ? $m : $k;
        $q = NDArray::zeros([$m, $qCols]);
        $q->slice(':', '0:' . min($n, $qCols))->assign($work->slice(':', '0:' . min($n, $qCols)));
        if ($k > 0) {
            self::info(Blas::call('LAPACKE_dorgqr', self::ROW, $m, $qCols, $k, self::d($q), max(1, $qCols), self::d($tau)), 'qr');
        }

        return [$q, $r];
    }

    /**
     * Eigen-decomposition of a symmetric matrix (lower triangle is read), eigenvalues ascending (stacked).
     *
     * @return array{0: NDArray, 1: NDArray} [w, v] with columns of v the eigenvectors
     */
    public static function eigh(mixed $a, bool $vectors = true): array
    {
        $A = self::f64($a);
        self::requireSquare($A, 'eigh');
        if (count($A->shape()) > 2) {
            $lead = array_slice($A->shape(), 0, -2);
            $count = (int) array_product($lead);
            $n = self::last($A);
            $flat = $A->reshape([$count, $n, $n]);
            $ws = NDArray::empty([$count, $n]);
            $vs = NDArray::empty([$count, $n, $n]);
            for ($i = 0; $i < $count; $i++) {
                [$w, $v] = self::eigh($flat->slice($i), $vectors);
                $ws->slice($i)->assign($w);
                $vs->slice($i)->assign($v);
            }

            return [$ws->reshape([...$lead, $n]), $vs->reshape([...$lead, $n, $n])];
        }
        $n = $A->shape()[0];
        $v = $A->copy();
        $w = NDArray::empty([max(1, $n)]);
        if ($n > 0) {
            $info = Blas::call('LAPACKE_dsyevd', self::ROW, $vectors ? 'V' : 'N', 'L', $n, self::d($v), $n, self::d($w));
            if ($info > 0) {
                throw new ConvergenceError('eigh: the eigenvalue algorithm failed to converge.');
            }
            self::info($info, 'eigh');
        }

        return [$n > 0 ? $w : NDArray::empty([0]), $v];
    }

    public static function eigvalsh(mixed $a): NDArray
    {
        return self::eigh($a, false)[0];
    }

    /**
     * General eigen-decomposition. Eigenvalues/vectors are float64 when all
     * eigenvalues are real, complex128 otherwise (as NumPy does).
     *
     * @return array{0: NDArray, 1: NDArray}
     */
    public static function eig(mixed $a): array
    {
        $A = self::f64($a);
        self::requireSquare($A, 'eig');
        self::require2d($A, 'eig');
        $n = $A->shape()[0];
        $work = $A->copy();
        $wr = NDArray::empty([max(1, $n)]);
        $wi = NDArray::empty([max(1, $n)]);
        $vr = NDArray::empty([max(1, $n), max(1, $n)]);
        $dummy = NDArray::empty([1]);
        $info = Blas::call('LAPACKE_dgeev', self::ROW, 'N', 'V', $n, self::d($work), max(1, $n), self::d($wr), self::d($wi),
            self::d($dummy), 1, self::d($vr), max(1, $n));
        if ($info > 0) {
            throw new ConvergenceError('eig: the QR algorithm failed to compute all eigenvalues.');
        }
        self::info($info, 'eig');
        $im = $wi->slice('0:' . $n)->toList();
        if (max(array_map('abs', $im ?: [0.0])) == 0.0) {
            return [$wr->slice('0:' . $n)->copy(), $vr];
        }
        $w = NDArray::complex($wr->slice('0:' . $n), $wi->slice('0:' . $n));
        $V = NDArray::complex($vr, NDArray::zeros([$n, $n]));
        for ($j = 0; $j < $n; $j++) {
            if ($im[$j] > 0 && $j + 1 < $n) {
                $re = $vr->slice(':', $j);
                $imv = $vr->slice(':', $j + 1);
                $V->slice(':', $j)->assign(NDArray::complex($re, $imv));
                $V->slice(':', $j + 1)->assign(NDArray::complex($re, $imv->neg()));
                $j++;
            }
        }

        return [$w, $V];
    }

    public static function eigvals(mixed $a): NDArray
    {
        return self::eig($a)[0];
    }

    /**
     * Singular value decomposition A = U diag(s) Vt.
     *
     * @return array{0: NDArray, 1: NDArray, 2: NDArray}|NDArray [U, s, Vt], or s alone when $computeUv is false
     */
    public static function svd(mixed $a, bool $fullMatrices = true, bool $computeUv = true): array|NDArray
    {
        $A = self::f64($a);
        self::require2d($A, 'svd');
        [$m, $n] = $A->shape();
        $k = min($m, $n);
        $work = $A->copy();
        $s = NDArray::empty([max(1, $k)]);
        $job = ! $computeUv ? 'N' : ($fullMatrices ? 'A' : 'S');
        $uCols = $fullMatrices ? $m : $k;
        $vtRows = $fullMatrices ? $n : $k;
        $u = NDArray::empty([max(1, $m), max(1, $computeUv ? $uCols : 1)]);
        $vt = NDArray::empty([max(1, $computeUv ? $vtRows : 1), max(1, $n)]);
        if ($k > 0) {
            $info = Blas::call('LAPACKE_dgesdd', self::ROW, $job, $m, $n, self::d($work), max(1, $n), self::d($s),
                self::d($u), max(1, $computeUv ? $uCols : 1), self::d($vt), max(1, $n));
            if ($info > 0) {
                throw new ConvergenceError('svd: the algorithm did not converge.');
            }
            self::info($info, 'svd');
        }
        $s = $k > 0 ? $s : NDArray::empty([0]);
        if (! $computeUv) {
            return $s;
        }

        return [$u->slice('0:' . $m, '0:' . $uCols)->copy(), $s, $vt->slice('0:' . $vtRows, '0:' . $n)->copy()];
    }

    /** Moore-Penrose pseudo-inverse via SVD; singular values below rcond * max(s) are treated as zero. */
    public static function pinv(mixed $a, float $rcond = 1e-15): NDArray
    {
        $A = self::f64($a);
        [$m, $n] = $A->shape();
        if (min($m, $n) === 0) {
            return NDArray::zeros([$n, $m]);
        }
        [$u, $s, $vt] = self::svd($A, false);
        $sv = $s->toList();
        $cut = $rcond * max($sv);
        $inv = NDArray::fromFlat(array_map(static fn (float $x): float => $x > $cut ? 1.0 / $x : 0.0, $sv), [count($sv)]);

        return $vt->t()->mul($inv)->matmul($u->t());
    }

    /**
     * Least squares (numpy.linalg.lstsq via LAPACK gelsd).
     *
     * @return array{0: NDArray, 1: NDArray, 2: int, 3: NDArray} [x, residuals, rank, singular values]
     */
    public static function lstsq(mixed $a, mixed $b, ?float $rcond = null): array
    {
        $A = self::f64($a);
        $B = self::f64($b);
        self::require2d($A, 'lstsq');
        [$m, $n] = $A->shape();
        $vector = count($B->shape()) === 1;
        if ($B->shape()[0] !== $m) {
            throw new ShapeError('lstsq: b must have as many rows as A.');
        }
        $nrhs = $vector ? 1 : $B->shape()[1];
        $rows = max($m, $n, 1);
        $bb = NDArray::zeros([$rows, max(1, $nrhs)]);
        if ($m > 0 && $nrhs > 0) {
            $bb->slice('0:' . $m, '0:' . $nrhs)->assign($vector ? $B->reshape([$m, 1]) : $B);
        }
        $work = $A->copy();
        $k = min($m, $n);
        $s = NDArray::empty([max(1, $k)]);
        $rank = Blas::ffi()->new('int');
        $rcond ??= PHP_FLOAT_EPSILON * max($m, $n);
        if ($k > 0 && $nrhs > 0) {
            $info = Blas::call('LAPACKE_dgelsd', self::ROW, $m, $n, $nrhs, self::d($work), max(1, $n), self::d($bb), max(1, $nrhs),
                self::d($s), $rcond, \FFI::addr($rank));
            if ($info > 0) {
                throw new ConvergenceError('lstsq: SVD did not converge.');
            }
            self::info($info, 'lstsq');
        }
        $x = $bb->slice('0:' . $n, '0:' . $nrhs)->copy();
        $r = (int) $rank->cdata;
        $residuals = NDArray::empty([0]);
        if ($r === $n && $m > $n) {
            $tail = $bb->slice($n . ':' . $m, '0:' . $nrhs);
            $residuals = $tail->square()->sum(0);
        }
        if ($vector) {
            $x = $x->reshape([$n]);
        }

        return [$x, $residuals, $k > 0 ? $r : 0, $k > 0 ? $s : NDArray::empty([0])];
    }

    /** Solve a triangular system (scipy.linalg.solve_triangular). */
    public static function solveTriangular(mixed $a, mixed $b, bool $lower = false, bool $transpose = false, bool $unitDiagonal = false): NDArray
    {
        $A = self::f64($a);
        $B = self::f64($b);
        self::requireSquare($A, 'solveTriangular');
        $n = $A->shape()[0];
        $vector = count($B->shape()) === 1;
        $nrhs = $vector ? 1 : $B->shape()[1];
        $x = $B->copy();
        $work = $A->copy();
        $info = Blas::call('LAPACKE_dtrtrs', self::ROW, $lower ? 'L' : 'U', $transpose ? 'T' : 'N', $unitDiagonal ? 'U' : 'N',
            $n, $nrhs, self::d($work), max(1, $n), self::d($x), max(1, $nrhs));
        self::info($info, 'solveTriangular', singular: true);

        return $x;
    }

    /**
     * LU factorisation with partial pivoting, A = P L U (scipy.linalg.lu).
     *
     * @return array{0: NDArray, 1: NDArray, 2: NDArray}
     */
    public static function lu(mixed $a): array
    {
        $A = self::f64($a);
        self::require2d($A, 'lu');
        [$m, $n] = $A->shape();
        $k = min($m, $n);
        $lu = $A->copy();
        $ipiv = Blas::ffi()->new('int[' . max(1, $k) . ']');
        $info = Blas::call('LAPACKE_dgetrf', self::ROW, $m, $n, self::d($lu), max(1, $n), $ipiv);
        if ($info < 0) {
            throw new LinAlgError("lu: illegal argument {$info}.");
        }
        $perm = range(0, $m - 1);
        $piv = self::ints($ipiv, $k);
        for ($i = 0; $i < $k; $i++) {
            $j = $piv[$i] - 1;
            $t = $perm[$i];
            $perm[$i] = $perm[$j];
            $perm[$j] = $t;
        }
        $p = NDArray::zeros([$m, $m]);
        foreach ($perm as $row => $orig) {
            $p->put([$orig * $m + $row], 1.0);
        }
        $l = self::tril($lu->slice(':', '0:' . $k)->copy(), -1)->add(NDArray::eye($m, $k));
        $u = self::triu($lu->slice('0:' . $k, ':')->copy());

        return [$p, $l, $u];
    }

    /** Vector or matrix norm (numpy.linalg.norm for ord null, 'fro', 'nuc', 1, 2, inf, -inf and vector p-norms). */
    public static function norm(mixed $x, int|float|string|null $ord = null, ?int $axis = null): float|NDArray
    {
        $X = self::f64($x);
        if ($axis !== null) {
            $abs = $X->abs();

            return match (true) {
                $ord === null || $ord === 2 || $ord === 2.0 => $abs->square()->sum($axis)->sqrt(),
                $ord === INF => $abs->max($axis),
                $ord === -INF => $abs->min($axis),
                $ord === 1 || $ord === 1.0 => $abs->sum($axis),
                $ord === 0 || $ord === 0.0 => $X->ne(0)->sum($axis),
                is_numeric($ord) => $abs->pow((float) $ord)->sum($axis)->pow(1.0 / (float) $ord),
                default => throw new ShapeError("norm: unsupported ord '{$ord}' with an axis."),
            };
        }
        $nd = count($X->shape());
        if ($ord === null || ($nd === 2 && $ord === 'fro')) {
            $flat = $X->ravel();
            $scale = $flat->size() > 0 ? $flat->abs()->max() : 0.0;
            if ($scale == 0.0 || ! is_finite($scale)) {
                return $flat->size() > 0 && ! is_finite($scale) ? $scale : 0.0;
            }

            return $scale * sqrt($flat->div($scale)->square()->sum());
        }
        if ($nd === 1) {
            return (float) self::norm($X, $ord, 0);
        }
        if ($nd !== 2) {
            throw new ShapeError('norm: ord needs a 1-D or 2-D input.');
        }

        return match ($ord) {
            'nuc' => (float) self::svd($X, false, false)->sum(),
            2, 2.0 => (float) self::svd($X, false, false)->max(),
            -2, -2.0 => (float) self::svd($X, false, false)->min(),
            1, 1.0 => (float) $X->abs()->sum(0)->max(),
            -1, -1.0 => (float) $X->abs()->sum(0)->min(),
            INF => (float) $X->abs()->sum(1)->max(),
            -INF => (float) $X->abs()->sum(1)->min(),
            default => throw new ShapeError("norm: unsupported matrix ord '{$ord}'."),
        };
    }

    public static function cond(mixed $a): float
    {
        $s = self::svd($a, false, false)->toList();
        $min = min($s);

        return $min == 0.0 ? INF : max($s) / $min;
    }

    public static function matrixRank(mixed $a, ?float $tol = null): int
    {
        $A = self::f64($a);
        $s = self::svd($A, false, false)->toList();
        if ($s === []) {
            return 0;
        }
        $tol ??= max($s) * max($A->shape()) * PHP_FLOAT_EPSILON;

        return count(array_filter($s, static fn (float $v): bool => $v > $tol));
    }

    public static function trace(mixed $a): float
    {
        $A = self::f64($a);
        $n = min($A->shape());

        return (float) $A->contiguous()->reshape([-1])->slice('::' . ($A->shape()[1] + 1))->slice('0:' . $n)->sum();
    }

    /** Lower triangle (numpy.tril). */
    public static function tril(NDArray $a, int $k = 0): NDArray
    {
        [$m, $n] = array_slice($a->shape(), -2);
        $rows = NDArray::arange($m)->reshape([$m, 1]);
        $cols = NDArray::arange($n)->reshape([1, $n]);

        return NDArray::where($cols->sub($rows)->le($k), $a, 0.0);
    }

    /** Upper triangle (numpy.triu). */
    public static function triu(NDArray $a, int $k = 0): NDArray
    {
        [$m, $n] = array_slice($a->shape(), -2);
        $rows = NDArray::arange($m)->reshape([$m, 1]);
        $cols = NDArray::arange($n)->reshape([1, $n]);

        return NDArray::where($cols->sub($rows)->ge($k), $a, 0.0);
    }

    // ------------------------------------------------------------------ internals

    private static function f64(mixed $a): NDArray
    {
        $A = NDArray::asArray($a);
        if ($A->dtype() === DType::Complex128) {
            throw new \Tessero\Exceptions\DTypeError('Complex linear algebra is not supported yet.');
        }

        return $A->astype(DType::Float64)->contiguous();
    }

    /** double* into the BLAS FFI scope for a contiguous float64 array. */
    /**
     * Copy an int[] CData into a PHP list in one call. Hot PHP loops read the
     * copy, never the CData element by element: fewer FFI calls, and no
     * JIT-compiled FFI element access (see docs/project/upstream-bugs.md, issue 5).
     *
     * @return list<int>
     */
    private static function ints(CData $a, int $n): array
    {
        return $n > 0 ? array_values(unpack('l' . $n, \FFI::string($a, 4 * $n))) : [];
    }

    private static function d(NDArray $a): CData
    {
        $a->ptr();                                  // throws if the buffer was already released (never hand LAPACK freed memory)

        return Blas::ptr('double*', $a->buffer()->address + $a->offset());
    }

    private static function last(NDArray $a): int
    {
        $s = $a->shape();

        return $s[count($s) - 1];
    }

    private static function requireSquare(NDArray $a, string $fn): void
    {
        $s = $a->shape();
        if (count($s) < 2 || $s[count($s) - 1] !== $s[count($s) - 2]) {
            throw new ShapeError("{$fn}: expected a square matrix (or a stack of them), got (" . implode(', ', $s) . ').');
        }
    }

    private static function require2d(NDArray $a, string $fn): void
    {
        if (count($a->shape()) !== 2) {
            throw new ShapeError("{$fn}: expected a 2-D array, got " . count($a->shape()) . '-D.');
        }
    }

    /** @param callable(NDArray, int): NDArray $fn */
    private static function stacked(NDArray $a, callable $fn, array $shape): NDArray
    {
        $lead = array_slice($shape, 0, -2);
        $count = (int) array_product($lead);
        $flat = $a->reshape([$count, ...array_slice($shape, -2)]);
        $results = [];
        for ($i = 0; $i < $count; $i++) {
            $results[] = $fn($flat->slice($i), $i);
        }
        $stack = NDArray::stack($results);

        return $stack->reshape([...$lead, ...array_slice($stack->shape(), 1)]);
    }

    private static function info(int $info, string $fn, bool $singular = false): void
    {
        if ($info < 0) {
            throw new LinAlgError("{$fn}: LAPACK reported an illegal value in argument " . (-$info) . '.');
        }
        if ($info > 0) {
            throw $singular ? new SingularMatrix("{$fn}: singular matrix (U[{$info},{$info}] is exactly zero).") : new LinAlgError("{$fn}: LAPACK info {$info}.");
        }
    }
}
