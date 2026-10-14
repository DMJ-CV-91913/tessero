<?php

declare(strict_types=1);

namespace Tessero\Sparse\Linalg;

/**
 * scipy.sparse.linalg.LaplacianNd -- the N-D discrete grid Laplacian and its
 * closed-form spectrum (pure-PHP).
 *
 * The Laplacian L is a square, negative-(semi)definite, real symmetric operator
 * with signed integer entries, built as a Kronecker sum of 1-D second-difference
 * matrices on a uniform grid with step h=1. Supports 'dirichlet', 'neumann'
 * (default) and 'periodic' boundary conditions.
 *
 * Exposes toarray(), tosparse() (dense array of the sparse form), eigenvalues(),
 * matvec()/matmat()/dot() (via the dense array) and the adjoint/transpose. The
 * eigenvalues are the analytically known closed form
 *   dirichlet: sum_d -4 sin^2(pi (j_d+1) / (2(n_d+1)))
 *   neumann:   sum_d -4 sin^2(pi  j_d    / (2 n_d))
 *   periodic:  sum_d -4 sin^2(pi floor((j_d+1)/2) / n_d)
 * returned in ascending order -- matching SciPy to machine precision.
 */
class LaplacianNd
{
    /** @var list<int> */
    private array $grid;

    private string $bc;

    private int $size;

    /**
     * @param list<int> $gridShape
     */
    public function __construct(array $gridShape, string $boundaryConditions = 'neumann')
    {
        $grid = [];
        foreach ($gridShape as $g) {
            $grid[] = (int) $g;
        }
        if ($grid === []) {
            throw new \InvalidArgumentException('LaplacianNd: grid_shape must be non-empty');
        }
        if (! in_array($boundaryConditions, ['dirichlet', 'neumann', 'periodic'], true)) {
            throw new \InvalidArgumentException("LaplacianNd: unknown boundary_conditions '{$boundaryConditions}'");
        }
        $this->grid = $grid;
        $this->bc = $boundaryConditions;
        $p = 1;
        foreach ($grid as $g) {
            $p *= $g;
        }
        $this->size = $p;
    }

    /** @return array{0:int,1:int} */
    public function shape(): array
    {
        return [$this->size, $this->size];
    }

    public function ndim(): int
    {
        return 2;
    }

    public function boundaryConditions(): string
    {
        return $this->bc;
    }

    /** @return list<int> */
    public function gridShape(): array
    {
        return $this->grid;
    }

    /**
     * 1-D second-difference matrix of size n with the configured boundary condition.
     *
     * @return list<list<float>>
     */
    private function lap1d(int $n): array
    {
        $L = [];
        for ($i = 0; $i < $n; $i++) {
            $L[] = array_fill(0, $n, 0.0);
        }
        for ($i = 0; $i < $n; $i++) {
            $L[$i][$i] = -2.0;
        }
        for ($i = 0; $i < $n - 1; $i++) {
            $L[$i][$i + 1] = 1.0;
            $L[$i + 1][$i] = 1.0;
        }
        if ($this->bc === 'neumann') {
            $L[0][0] = -1.0;
            $L[$n - 1][$n - 1] = -1.0;
        } elseif ($this->bc === 'periodic') {
            if ($n > 1) {
                $L[0][$n - 1] += 1.0;
                $L[$n - 1][0] += 1.0;
            } else {
                $L[0][0] += 1.0;
            }
        }
        return $L;
    }

    /**
     * Kronecker product of two dense matrices.
     *
     * @param list<list<float>> $A
     * @param list<list<float>> $B
     * @return list<list<float>>
     */
    private static function kron(array $A, array $B): array
    {
        $ra = count($A);
        $ca = $ra > 0 ? count($A[0]) : 0;
        $rb = count($B);
        $cb = $rb > 0 ? count($B[0]) : 0;
        $out = [];
        for ($i = 0; $i < $ra * $rb; $i++) {
            $out[] = array_fill(0, $ca * $cb, 0.0);
        }
        for ($i = 0; $i < $ra; $i++) {
            for ($j = 0; $j < $ca; $j++) {
                $aij = $A[$i][$j];
                if ($aij === 0.0) {
                    continue;
                }
                for ($p = 0; $p < $rb; $p++) {
                    for ($q = 0; $q < $cb; $q++) {
                        $out[$i * $rb + $p][$j * $cb + $q] = $aij * $B[$p][$q];
                    }
                }
            }
        }
        return $out;
    }

    /** @return list<list<float>> identity of size n */
    private static function eye(int $n): array
    {
        $I = [];
        for ($i = 0; $i < $n; $i++) {
            $row = array_fill(0, $n, 0.0);
            $row[$i] = 1.0;
            $I[] = $row;
        }
        return $I;
    }

    /**
     * Dense array of the Laplacian (Kronecker sum of 1-D Laplacians). Values are
     * exact integers represented as floats.
     *
     * @return list<list<float>>
     */
    public function toarray(): array
    {
        $N = count($this->grid);
        $p = $this->size;
        $L = [];
        for ($i = 0; $i < $p; $i++) {
            $L[] = array_fill(0, $p, 0.0);
        }
        for ($i = 0; $i < $N; $i++) {
            $Li = $this->lap1d($this->grid[$i]);
            for ($j = 0; $j < $i; $j++) {
                $Li = self::kron(self::eye($this->grid[$j]), $Li);
            }
            for ($j = $i + 1; $j < $N; $j++) {
                $Li = self::kron($Li, self::eye($this->grid[$j]));
            }
            for ($r = 0; $r < $p; $r++) {
                for ($c = 0; $c < $p; $c++) {
                    $L[$r][$c] += $Li[$r][$c];
                }
            }
        }
        return $L;
    }

    /**
     * Dense array of the sparse representation. The LaplacianNd sparse form is
     * identical to its dense form, so tosparse()->toarray() == toarray().
     *
     * @return list<list<float>>
     */
    public function tosparse(): array
    {
        return $this->toarray();
    }

    /**
     * Closed-form eigenvalues in ascending order. If $m is given, the $m largest
     * (least negative) eigenvalues are returned (scipy returns the m smallest in
     * absolute value, i.e. the largest algebraically, still ascending).
     *
     * @return list<float>
     */
    public function eigenvalues(?int $m = null): array
    {
        $grid = $this->grid;
        if ($m !== null) {
            $grid = [];
            foreach ($this->grid as $g) {
                $grid[] = min($g, $m);
            }
        }
        // enumerate the tensor-product index grid, summing per-dimension terms
        $dims = count($grid);
        $vals = [];
        $total = 1;
        foreach ($grid as $g) {
            $total *= $g;
        }
        for ($lin = 0; $lin < $total; $lin++) {
            // unravel lin into multi-index (C-order)
            $rem = $lin;
            $s = 0.0;
            for ($d = 0; $d < $dims; $d++) {
                $stride = 1;
                for ($e = $d + 1; $e < $dims; $e++) {
                    $stride *= $grid[$e];
                }
                $j = intdiv($rem, $stride);
                $rem -= $j * $stride;
                $n = $this->grid[$d];
                if ($this->bc === 'dirichlet') {
                    $s += -4.0 * sin(M_PI * ($j + 1) / (2 * ($n + 1))) ** 2;
                } elseif ($this->bc === 'neumann') {
                    $s += -4.0 * sin(M_PI * $j / (2 * $n)) ** 2;
                } else { // periodic
                    $s += -4.0 * sin(M_PI * floor(($j + 1) / 2) / $n) ** 2;
                }
            }
            $vals[] = $s;
        }
        sort($vals);
        if ($m !== null) {
            $vals = array_slice($vals, count($vals) - $m);
        }
        return array_map('floatval', $vals);
    }

    /**
     * Matrix-vector product L x via the dense array.
     *
     * @param list<float> $x
     * @return list<float>
     */
    public function matvec(array $x): array
    {
        $L = $this->toarray();
        $p = $this->size;
        if (count($x) !== $p) {
            throw new \InvalidArgumentException('matvec: dimension mismatch');
        }
        $out = [];
        for ($i = 0; $i < $p; $i++) {
            $sum = 0.0;
            $row = $L[$i];
            for ($j = 0; $j < $p; $j++) {
                $sum += $row[$j] * (float) $x[$j];
            }
            $out[] = $sum;
        }
        return $out;
    }

    /**
     * Adjoint == matvec (L is symmetric).
     *
     * @param list<float> $y
     * @return list<float>
     */
    public function rmatvec(array $y): array
    {
        return $this->matvec($y);
    }

    /**
     * Matrix-matrix product L X (X row-major, shape (p, k)).
     *
     * @param list<list<float>> $X
     * @return list<list<float>>
     */
    public function matmat(array $X): array
    {
        return $this->asLinearOperator()->matmat($X);
    }

    /**
     * @param list<float>|list<list<float>> $x
     * @return list<float>|list<list<float>>
     */
    public function dot(array $x): array
    {
        if ($x !== [] && is_array($x[array_key_first($x)])) {
            return $this->matmat($x);
        }
        return $this->matvec($x);
    }

    /** Return this Laplacian as a generic LinearOperator (wrapping its dense array). */
    public function asLinearOperator(): LinearOperator
    {
        return LinearOperator::fromMatrix($this->toarray());
    }
}
