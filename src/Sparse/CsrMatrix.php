<?php

declare(strict_types=1);

namespace Tessero\Sparse;

use FFI\CData;
use Tessero\DType;
use Tessero\Exceptions\ConvergenceError;
use Tessero\Exceptions\ShapeError;
use Tessero\NDArray;
use Tessero\Native\Library;

/**
 * Compressed sparse row matrix of float64 (scipy.sparse.csr_matrix).
 * Column indices within a row are kept sorted and duplicate entries summed.
 */
final class CsrMatrix
{
    private function __construct(
        private readonly NDArray $indptr,
        private readonly NDArray $indices,
        private readonly NDArray $data,
        private readonly int $rows,
        private readonly int $cols,
    ) {
    }

    public static function fromDense(mixed $dense): self
    {
        $a = NDArray::asArray($dense)->astype(DType::Float64)->contiguous();
        if (count($a->shape()) !== 2) {
            throw new ShapeError('fromDense needs a 2-D array.');
        }
        [$m, $n] = $a->shape();
        $ffi = Library::ffi();
        $nnz = $ffi->tsr_dense_nnz($m, $n, self::d($a));
        $indptr = NDArray::empty([$m + 1], DType::Int64);
        $indices = NDArray::empty([$nnz], DType::Int64);
        $data = NDArray::empty([$nnz]);
        Library::check($ffi->tsr_dense_to_csr($m, $n, self::d($a), self::i($indptr), self::i($indices), self::d($data)), 'fromDense');

        return new self($indptr, $indices, $data, $m, $n);
    }

    /**
     * From coordinate triplets (scipy.sparse.coo_matrix(...).tocsr()); duplicates are summed.
     *
     * @param list<int>|NDArray $row
     * @param list<int>|NDArray $col
     * @param list<float>|NDArray $values
     * @param array{int, int} $shape
     */
    public static function fromTriplets(array|NDArray $row, array|NDArray $col, array|NDArray $values, array $shape): self
    {
        [$m, $n] = $shape;
        $r = NDArray::asArray($row, DType::Int64);
        $c = NDArray::asArray($col, DType::Int64);
        $v = NDArray::asArray($values, DType::Float64);
        if ($r->shape() !== $c->shape() || $r->shape() !== $v->shape() || count($r->shape()) !== 1) {
            throw new ShapeError('row, col and values must be 1-D and the same length.');
        }
        if ($r->size() > 0 && ($r->min() < 0 || $r->max() >= $m || $c->min() < 0 || $c->max() >= $n)) {
            throw new ShapeError('Triplet index outside the matrix shape.');
        }
        $order = $r->mul($n)->add($c)->argsort();
        $keys = $r->mul($n)->add($c)->take($order)->toList();
        $vals = $v->take($order)->toList();
        $indptr = array_fill(0, $m + 1, 0);
        $indices = [];
        $data = [];
        $last = -1;
        foreach ($keys as $k => $key) {
            if ($key === $last) {
                $data[count($data) - 1] = $data[count($data) - 1] + $vals[$k];
                continue;
            }
            $last = $key;
            $indices[] = $key % $n;
            $data[] = $vals[$k];
            $rowIdx = intdiv($key, $n);
            $indptr[$rowIdx + 1] = $indptr[$rowIdx + 1] + 1;
        }
        for ($i = 0; $i < $m; $i++) {
            $indptr[$i + 1] = $indptr[$i + 1] + $indptr[$i];
        }

        return new self(
            NDArray::fromFlat($indptr, [$m + 1], DType::Int64),
            NDArray::fromFlat($indices, [count($indices)], DType::Int64),
            NDArray::fromFlat($data, [count($data)]),
            $m,
            $n,
        );
    }

    /** Wrap existing CSR arrays (validated). */
    public static function fromArrays(mixed $indptr, mixed $indices, mixed $data, array $shape): self
    {
        $p = NDArray::asArray($indptr, DType::Int64)->contiguous();
        $ix = NDArray::asArray($indices, DType::Int64)->contiguous();
        $d = NDArray::asArray($data, DType::Float64)->contiguous();
        [$m, $n] = $shape;
        if ($p->size() !== $m + 1 || $ix->size() !== $d->size() || $p->item($m) !== $ix->size()) {
            throw new ShapeError('Inconsistent CSR arrays.');
        }
        if ($ix->size() > 0 && ($ix->min() < 0 || $ix->max() >= $n)) {
            throw new ShapeError('CSR column index outside the matrix.');
        }

        return new self($p, $ix, $d, $m, $n);
    }

    public static function identity(int $n): self
    {
        return new self(NDArray::arange($n + 1), NDArray::arange($n), NDArray::ones([$n]), $n, $n);
    }

    /** @return array{int, int} */
    public function shape(): array
    {
        return [$this->rows, $this->cols];
    }

    public function nnz(): int
    {
        return $this->data->size();
    }

    public function indptr(): NDArray { return $this->indptr; }
    public function indices(): NDArray { return $this->indices; }
    public function data(): NDArray { return $this->data; }

    public function toDense(): NDArray
    {
        $out = NDArray::zeros([$this->rows, $this->cols]);
        if ($this->nnz() > 0) {
            $rowOf = [];
            $ptr = $this->indptr->toList();
            for ($i = 0; $i < $this->rows; $i++) {
                for ($p = $ptr[$i]; $p < $ptr[$i + 1]; $p++) {
                    $rowOf[] = $i * $this->cols;
                }
            }
            $flat = $out->reshape([-1]);
            $flat->put(NDArray::fromFlat($rowOf, [count($rowOf)], DType::Int64)->add($this->indices), $this->data);
        }

        return $out;
    }

    /** A @ x for a dense vector (n,) or matrix (n, k). */
    public function dot(mixed $x): NDArray
    {
        $X = NDArray::asArray($x)->astype(DType::Float64)->contiguous();
        $ffi = Library::ffi();
        if (count($X->shape()) === 1) {
            if ($X->shape()[0] !== $this->cols) {
                throw new ShapeError("Dimension mismatch: ({$this->rows}, {$this->cols}) @ ({$X->shape()[0]},).");
            }
            $y = NDArray::empty([$this->rows]);
            Library::check($ffi->tsr_csr_matvec($this->rows, self::i($this->indptr), self::i($this->indices), self::d($this->data), self::d($X), self::d($y)), 'dot');

            return $y;
        }
        [$n, $k] = $X->shape();
        if ($n !== $this->cols) {
            throw new ShapeError("Dimension mismatch: ({$this->rows}, {$this->cols}) @ ({$n}, {$k}).");
        }
        $y = NDArray::empty([$this->rows, $k]);
        Library::check($ffi->tsr_csr_matmat($this->rows, $k, self::i($this->indptr), self::i($this->indices), self::d($this->data), self::d($X), self::d($y)), 'dot');

        return $y;
    }

    public function transpose(): self
    {
        $tp = NDArray::empty([$this->cols + 1], DType::Int64);
        $ti = NDArray::empty([$this->nnz()], DType::Int64);
        $td = NDArray::empty([$this->nnz()]);
        Library::check(Library::ffi()->tsr_csr_transpose(
            $this->rows, $this->cols, self::i($this->indptr), self::i($this->indices), self::d($this->data),
            self::i($tp), self::i($ti), self::d($td),
        ), 'csr transpose');

        return new self($tp, $ti, $td, $this->cols, $this->rows);
    }

    public function scale(float $alpha): self
    {
        return new self($this->indptr, $this->indices, $this->data->mul($alpha), $this->rows, $this->cols);
    }

    /** Main diagonal as a dense vector. */
    public function diagonal(): NDArray
    {
        $k = min($this->rows, $this->cols);
        $ptr = $this->indptr->toList();
        $idx = $this->indices->toList();
        $val = $this->data->toList();
        $out = array_fill(0, $k, 0.0);
        for ($i = 0; $i < $k; $i++) {
            for ($p = $ptr[$i]; $p < $ptr[$i + 1]; $p++) {
                if ($idx[$p] === $i) {
                    $out[$i] = $out[$i] + $val[$p];
                }
            }
        }

        return NDArray::fromFlat($out, [$k]);
    }

    /**
     * Conjugate gradients (scipy.sparse.linalg.cg) with Jacobi preconditioning; A must be SPD.
     *
     * @return array{0: NDArray, 1: int} [x, info] - info 0 on convergence, else the iteration count reached
     */
    public function cg(mixed $b, mixed $x0 = null, float $rtol = 1e-5, float $atol = 0.0, ?int $maxiter = null, bool $throw = false): array
    {
        return $this->krylov('tsr_csr_cg', $b, $x0, $rtol, $atol, $maxiter ?? 10 * $this->rows, $throw);
    }

    /** BiCGSTAB for general square systems (scipy.sparse.linalg.bicgstab). */
    public function bicgstab(mixed $b, mixed $x0 = null, float $rtol = 1e-5, float $atol = 0.0, ?int $maxiter = null, bool $throw = false): array
    {
        return $this->krylov('tsr_csr_bicgstab', $b, $x0, $rtol, $atol, $maxiter ?? 10 * $this->rows, $throw);
    }

    /** @return array{0: NDArray, 1: int} */
    private function krylov(string $fn, mixed $b, mixed $x0, float $rtol, float $atol, int $maxiter, bool $throw): array
    {
        if ($this->rows !== $this->cols) {
            throw new ShapeError('Iterative solvers need a square matrix.');
        }
        $B = NDArray::asArray($b)->astype(DType::Float64)->contiguous();
        if ($B->shape() !== [$this->rows]) {
            throw new ShapeError('b must be a vector with one entry per row.');
        }
        $x = $x0 === null ? NDArray::zeros([$this->rows]) : NDArray::asArray($x0)->astype(DType::Float64, true)->contiguous();
        $ffi = Library::ffi();
        $res = $ffi->new('double');
        $it = $ffi->{$fn}($this->rows, self::i($this->indptr), self::i($this->indices), self::d($this->data), self::d($B), self::d($x),
            $rtol, $atol, $maxiter, \FFI::addr($res));
        if ($it === -3) {
            Library::check(-3, $fn);
        }
        $info = $it >= 0 ? 0 : -$it;
        if ($info !== 0 && $throw) {
            throw new ConvergenceError(sprintf('%s did not converge in %d iterations (residual %.3e).', $fn, $info, $res->cdata));
        }

        return [$x, $info];
    }

    private static function d(NDArray $a): CData
    {
        return Library::ffi()->cast('double*', $a->ptr());
    }

    private static function i(NDArray $a): CData
    {
        return Library::ffi()->cast('int64_t*', $a->ptr());
    }
}
