<?php

declare(strict_types=1);

namespace Tessero\Optimize;

use FFI\CData;
use Tessero\DType;
use Tessero\Exceptions\ShapeError;
use Tessero\Exceptions\TesseroException;
use Tessero\NDArray;
use Tessero\Native\Library;

/**
 * scipy.optimize.linprog and scipy.optimize.milp on libtessero's native
 * simplex and branch-and-bound.
 *
 *   minimise c.x  subject to  A_ub x <= b_ub,  A_eq x = b_eq,  bounds
 *
 * Bounds follow linprog: null means (0, +inf) for every variable; one
 * [lo, hi] pair applies to all; otherwise one pair per variable. null or
 * +/-INF is an open side. Maximise by negating c (the result's fun is then
 * the negated maximum, as in SciPy).
 */
final class LinearProgramming
{
    public const STATUS = [
        0 => 'Optimization terminated successfully.',
        1 => 'Iteration or node limit reached.',
        2 => 'The problem is infeasible.',
        3 => 'The problem is unbounded.',
        4 => 'Numerical difficulties encountered.',
    ];

    /**
     * @param array<int, array{0: float|int|null, 1: float|int|null}>|array{0: float|int|null, 1: float|int|null}|null $bounds
     */
    public static function linprog(
        mixed $c,
        mixed $A_ub = null,
        mixed $b_ub = null,
        mixed $A_eq = null,
        mixed $b_eq = null,
        ?array $bounds = null,
        int $maxiter = 100_000,
        float $tol = 1e-9,
    ): LinprogResult {
        $p = self::prepare($c, $A_ub, $b_ub, $A_eq, $b_eq, $bounds);
        $n = $p['n'];
        $ffi = Library::ffi();
        $x = NDArray::empty([$n]);
        $yUb = NDArray::zeros([max(1, $p['mUb'])]);
        $yEq = NDArray::zeros([max(1, $p['mEq'])]);
        $red = NDArray::zeros([$n]);
        $fun = $ffi->new('double');
        $nit = $ffi->new('int64_t');
        $status = $ffi->tsr_linprog(
            $n, self::d($p['c']), $p['mUb'], self::dn($p['Aub']), self::dn($p['bub']), $p['mEq'], self::dn($p['Aeq']), self::dn($p['beq']),
            self::d($p['lb']), self::d($p['ub']), $maxiter, $tol,
            self::d($x), \FFI::addr($fun), self::d($yUb), self::d($yEq), self::d($red), \FFI::addr($nit),
        );
        if ($status === -1) {
            throw new \InvalidArgumentException('linprog: c, A_ub, b_ub, A_eq and b_eq must be finite, and bounds must not be NaN.');
        }
        if ($status < 0) {
            Library::check($status, 'linprog');
        }
        $ok = $status === 0;

        return new LinprogResult(
            $ok || $status === 1 ? $x->toList() : null,
            $ok || $status === 1 ? $fun->cdata : null,
            $ok,
            $status,
            self::STATUS[$status] ?? 'Unknown status.',
            $nit->cdata,
            $ok ? array_slice($yUb->toList(), 0, $p['mUb']) : [],
            $ok ? array_slice($yEq->toList(), 0, $p['mEq']) : [],
            $ok ? $red->toList() : [],
        );
    }

    /**
     * Mixed-integer linear program (scipy.optimize.milp). $integrality: one flag per
     * variable (true/1 = integer), or a single bool for all.
     */
    public static function milp(
        mixed $c,
        bool|array $integrality = true,
        mixed $A_ub = null,
        mixed $b_ub = null,
        mixed $A_eq = null,
        mixed $b_eq = null,
        ?array $bounds = null,
        int $nodeLimit = 100_000,
        float $mipRelGap = 0.0,
        float $tol = 1e-9,
    ): MilpResult {
        $p = self::prepare($c, $A_ub, $b_ub, $A_eq, $b_eq, $bounds);
        $n = $p['n'];
        $flags = is_bool($integrality) ? array_fill(0, $n, $integrality ? 1 : 0) : array_map(static fn ($v): int => $v ? 1 : 0, array_values($integrality));
        if (count($flags) !== $n) {
            throw new ShapeError("milp: integrality needs {$n} flags.");
        }
        $ffi = Library::ffi();
        $integ = NDArray::fromFlat($flags, [$n], DType::UInt8);
        $x = NDArray::zeros([$n]);
        $fun = $ffi->new('double');
        $fun->cdata = INF;
        $nodes = $ffi->new('int64_t');
        $bound = $ffi->new('double');
        $status = $ffi->tsr_milp(
            $n, self::d($p['c']), $p['mUb'], self::dn($p['Aub']), self::dn($p['bub']), $p['mEq'], self::dn($p['Aeq']), self::dn($p['beq']),
            self::d($p['lb']), self::d($p['ub']), $ffi->cast('uint8_t*', $integ->ptr()), $nodeLimit, $mipRelGap, $tol,
            self::d($x), \FFI::addr($fun), \FFI::addr($nodes), \FFI::addr($bound),
        );
        if ($status === -1) {
            throw new \InvalidArgumentException('milp: c, A_ub, b_ub, A_eq and b_eq must be finite, and bounds must not be NaN.');
        }
        if ($status < 0) {
            Library::check($status, 'milp');
        }
        // status 1 (node limit) keeps the incumbent when one was found; fun stays INF otherwise
        $hasX = $status === 0 || ($status === 1 && is_finite($fun->cdata));

        return new MilpResult(
            $hasX ? $x->toList() : null,
            $hasX ? $fun->cdata : null,
            $status === 0,
            $status,
            self::STATUS[$status] ?? 'Unknown status.',
            $nodes->cdata,
            $bound->cdata,
        );
    }

    /** @return array<string, mixed> */
    private static function prepare(mixed $c, mixed $A_ub, mixed $b_ub, mixed $A_eq, mixed $b_eq, ?array $bounds): array
    {
        $cv = NDArray::asArray($c)->astype(DType::Float64)->contiguous();
        if (count($cv->shape()) !== 1 || $cv->size() === 0) {
            throw new ShapeError('c must be a non-empty 1-D array.');
        }
        $n = $cv->size();
        [$Aub, $bub, $mUb] = self::block($A_ub, $b_ub, $n, 'A_ub');
        [$Aeq, $beq, $mEq] = self::block($A_eq, $b_eq, $n, 'A_eq');
        $lo = [];
        $hi = [];
        if ($bounds === null) {
            $lo = array_fill(0, $n, 0.0);
            $hi = array_fill(0, $n, INF);
        } else {
            $pairs = (count($bounds) === 2 && ! is_array($bounds[0] ?? null) && ! is_array($bounds[1] ?? null)) ? array_fill(0, $n, $bounds) : array_values($bounds);
            if (count($pairs) !== $n) {
                throw new ShapeError("bounds must be one pair or {$n} pairs.");
            }
            foreach ($pairs as $i => $pair) {
                $pair = array_values((array) $pair);
                $l = $pair[0] ?? null;
                $h = $pair[1] ?? null;
                $lo[] = $l === null ? -INF : (float) $l;
                $hi[] = $h === null ? INF : (float) $h;
                if ($lo[$i] > $hi[$i]) {
                    throw new TesseroException("bounds[{$i}]: lower bound exceeds upper bound.");
                }
            }
        }

        return [
            'n' => $n, 'c' => $cv, 'Aub' => $Aub, 'bub' => $bub, 'mUb' => $mUb, 'Aeq' => $Aeq, 'beq' => $beq, 'mEq' => $mEq,
            'lb' => NDArray::fromFlat($lo, [$n]), 'ub' => NDArray::fromFlat($hi, [$n]),
        ];
    }

    /** @return array{0: ?NDArray, 1: ?NDArray, 2: int} */
    private static function block(mixed $A, mixed $b, int $n, string $name): array
    {
        if ($A === null && $b === null) {
            return [null, null, 0];
        }
        if ($A === null || $b === null) {
            throw new ShapeError("{$name} and its right-hand side must be given together.");
        }
        $Am = NDArray::asArray($A)->astype(DType::Float64)->contiguous();
        $bv = NDArray::asArray($b)->astype(DType::Float64)->contiguous();
        if ($Am->size() === 0) {
            return [null, null, 0];
        }
        if (count($Am->shape()) === 1) {
            $Am = $Am->reshape([1, $Am->size()]);
        }
        [$m, $cols] = $Am->shape();
        if ($cols !== $n || $bv->size() !== $m) {
            throw new ShapeError("{$name} is {$m}x{$cols} but c has {$n} entries / the right-hand side has {$bv->size()}.");
        }

        return [$Am, $bv->ravel(), $m];
    }

    private static function d(NDArray $a): CData
    {
        return Library::ffi()->cast('double*', $a->ptr());
    }

    private static function dn(?NDArray $a): ?CData
    {
        return $a === null ? null : self::d($a);
    }
}
