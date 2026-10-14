<?php

declare(strict_types=1);

namespace Tessero\Sparse\Linalg;

/**
 * scipy.sparse.linalg.LinearOperator -- matrix-free linear operator (pure-PHP).
 *
 * A common interface for performing matrix-vector products. Constructed either
 *   - from an explicit dense matrix (list<list<float>>), via fromMatrix(); or
 *   - from matvec / rmatvec / matmat callbacks, via fromCallbacks().
 *
 * Exposes matvec/rmatvec/matmat/dot/__invoke, the adjoint (H), transpose (T),
 * scalar and operator arithmetic (scale, add, compose), shape and dtype -- all
 * deterministic real linear algebra matching SciPy to machine precision.
 *
 * Hand-authored facade shared by both Tessero backends (no kernel marshaling:
 * the products are exact PHP dot-products over stored dense data or callbacks).
 */
class LinearOperator
{
    /** @var array{0:int,1:int} (rows, cols) */
    protected array $shapeTuple;

    protected string $dtypeName;

    /** @var (callable(list<float>):list<float>)|null */
    protected $matvecFn = null;

    /** @var (callable(list<float>):list<float>)|null */
    protected $rmatvecFn = null;

    /** @var (callable(list<list<float>>):list<list<float>>)|null */
    protected $matmatFn = null;

    /**
     * @param array{0:int,1:int} $shape
     */
    protected function __construct(array $shape, string $dtype = 'float64')
    {
        $this->shapeTuple = [(int) $shape[0], (int) $shape[1]];
        $this->dtypeName = $dtype;
    }

    /**
     * Wrap an explicit dense matrix A (list of rows). matvec = A x, rmatvec = A^T x.
     *
     * @param list<list<float>> $matrix
     */
    public static function fromMatrix(array $matrix, string $dtype = 'float64'): self
    {
        $m = count($matrix);
        $n = $m > 0 ? count($matrix[0]) : 0;
        $A = [];
        foreach ($matrix as $row) {
            $A[] = array_map('floatval', array_values($row));
        }
        $op = new self([$m, $n], $dtype);
        $op->matvecFn = static function (array $x) use ($A, $m, $n): array {
            $out = [];
            for ($i = 0; $i < $m; $i++) {
                $s = 0.0;
                $r = $A[$i];
                for ($j = 0; $j < $n; $j++) {
                    $s += $r[$j] * (float) $x[$j];
                }
                $out[] = $s;
            }
            return $out;
        };
        $op->rmatvecFn = static function (array $y) use ($A, $m, $n): array {
            $out = array_fill(0, $n, 0.0);
            for ($i = 0; $i < $m; $i++) {
                $yi = (float) $y[$i];
                $r = $A[$i];
                for ($j = 0; $j < $n; $j++) {
                    $out[$j] += $r[$j] * $yi;
                }
            }
            return $out;
        };
        return $op;
    }

    /**
     * Construct from callbacks (as scipy LinearOperator(shape, matvec, rmatvec, matmat)).
     *
     * @param array{0:int,1:int}                        $shape
     * @param callable(list<float>):list<float>         $matvec
     * @param (callable(list<float>):list<float>)|null  $rmatvec
     * @param (callable(list<list<float>>):list<list<float>>)|null $matmat
     */
    public static function fromCallbacks(array $shape, callable $matvec, ?callable $rmatvec = null, ?callable $matmat = null, string $dtype = 'float64'): self
    {
        $op = new self($shape, $dtype);
        $op->matvecFn = $matvec;
        $op->rmatvecFn = $rmatvec;
        $op->matmatFn = $matmat;
        return $op;
    }

    /** @return array{0:int,1:int} */
    public function shape(): array
    {
        return $this->shapeTuple;
    }

    public function dtype(): string
    {
        return $this->dtypeName;
    }

    public function ndim(): int
    {
        return 2;
    }

    /**
     * Matrix-vector product A x. Accepts a flat list; returns a flat list.
     *
     * @param list<float> $x
     * @return list<float>
     */
    public function matvec(array $x): array
    {
        if (count($x) !== $this->shapeTuple[1]) {
            throw new \InvalidArgumentException('matvec: dimension mismatch');
        }
        if ($this->matvecFn === null) {
            throw new \RuntimeException('matvec not defined for this operator');
        }
        return array_values(array_map('floatval', ($this->matvecFn)(array_map('floatval', array_values($x)))));
    }

    /**
     * Adjoint (here, transpose) matrix-vector product A^H y.
     *
     * @param list<float> $y
     * @return list<float>
     */
    public function rmatvec(array $y): array
    {
        if (count($y) !== $this->shapeTuple[0]) {
            throw new \InvalidArgumentException('rmatvec: dimension mismatch');
        }
        if ($this->rmatvecFn === null) {
            throw new \RuntimeException('rmatvec/adjoint not defined for this operator');
        }
        return array_values(array_map('floatval', ($this->rmatvecFn)(array_map('floatval', array_values($y)))));
    }

    /**
     * Matrix-matrix product A X, X row-major of shape (n, k); returns shape (m, k).
     *
     * @param list<list<float>> $X
     * @return list<list<float>>
     */
    public function matmat(array $X): array
    {
        $n = $this->shapeTuple[1];
        if (count($X) !== $n) {
            throw new \InvalidArgumentException('matmat: dimension mismatch');
        }
        if ($this->matmatFn !== null) {
            return ($this->matmatFn)($X);
        }
        // default: apply matvec to each column
        $k = count($X) > 0 ? count($X[0]) : 0;
        $cols = [];
        for ($c = 0; $c < $k; $c++) {
            $col = [];
            for ($r = 0; $r < $n; $r++) {
                $col[] = (float) $X[$r][$c];
            }
            $cols[] = $this->matvec($col);
        }
        $m = $this->shapeTuple[0];
        $out = [];
        for ($i = 0; $i < $m; $i++) {
            $row = [];
            for ($c = 0; $c < $k; $c++) {
                $row[] = $cols[$c][$i];
            }
            $out[] = $row;
        }
        return $out;
    }

    /**
     * dot(x): matvec if x is 1-D, matmat if x is 2-D (row-major).
     *
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

    /**
     * Adjoint operator A^H (conjugate transpose; for real data this is the transpose).
     */
    public function adjoint(): self
    {
        $op = new self([$this->shapeTuple[1], $this->shapeTuple[0]], $this->dtypeName);
        $op->matvecFn = $this->rmatvecFn;
        $op->rmatvecFn = $this->matvecFn;
        return $op;
    }

    /** Transpose operator A^T (== adjoint for real data). */
    public function transpose(): self
    {
        return $this->adjoint();
    }

    /** scipy property aliases. */
    public function getH(): self
    {
        return $this->adjoint();
    }

    public function getT(): self
    {
        return $this->transpose();
    }

    /**
     * Scalar multiple (alpha * A).
     */
    public function scale(float $alpha): self
    {
        $mv = $this->matvecFn;
        $rmv = $this->rmatvecFn;
        $op = new self($this->shapeTuple, $this->dtypeName);
        $op->matvecFn = $mv === null ? null : static function (array $x) use ($mv, $alpha): array {
            return array_map(static fn ($v) => $alpha * (float) $v, $mv($x));
        };
        $op->rmatvecFn = $rmv === null ? null : static function (array $y) use ($rmv, $alpha): array {
            return array_map(static fn ($v) => $alpha * (float) $v, $rmv($y));
        };
        return $op;
    }

    /**
     * Operator sum (A + B); shapes must match.
     */
    public function add(LinearOperator $other): self
    {
        if ($this->shapeTuple !== $other->shapeTuple) {
            throw new \InvalidArgumentException('add: shape mismatch');
        }
        $a = $this;
        $b = $other;
        $op = new self($this->shapeTuple, $this->dtypeName);
        $op->matvecFn = static function (array $x) use ($a, $b): array {
            $u = $a->matvec($x);
            $v = $b->matvec($x);
            $o = [];
            foreach ($u as $i => $ui) {
                $o[] = $ui + $v[$i];
            }
            return $o;
        };
        $op->rmatvecFn = static function (array $y) use ($a, $b): array {
            $u = $a->rmatvec($y);
            $v = $b->rmatvec($y);
            $o = [];
            foreach ($u as $i => $ui) {
                $o[] = $ui + $v[$i];
            }
            return $o;
        };
        return $op;
    }

    /**
     * Operator composition / product (A @ B): applies B then A.
     */
    public function compose(LinearOperator $other): self
    {
        if ($this->shapeTuple[1] !== $other->shapeTuple[0]) {
            throw new \InvalidArgumentException('compose: shape mismatch');
        }
        $a = $this;
        $b = $other;
        $op = new self([$this->shapeTuple[0], $other->shapeTuple[1]], $this->dtypeName);
        $op->matvecFn = static function (array $x) use ($a, $b): array {
            return $a->matvec($b->matvec($x));
        };
        $op->rmatvecFn = static function (array $y) use ($a, $b): array {
            // (A B)^H y = B^H A^H y
            return $b->rmatvec($a->rmatvec($y));
        };
        return $op;
    }

    /**
     * @param list<float> $x
     * @return list<float>
     */
    public function __invoke(array $x): array
    {
        return $this->matvec($x);
    }

    public function __get(string $name): mixed
    {
        return match ($name) {
            'H' => $this->getH(),
            'T' => $this->getT(),
            'shape' => $this->shapeTuple,
            'dtype' => $this->dtypeName,
            'ndim' => 2,
            default => throw new \RuntimeException("LinearOperator: unknown property '{$name}'"),
        };
    }
}
