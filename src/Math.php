<?php

declare(strict_types=1);

namespace Tessero;

use FFI;
use Tessero\Exceptions\DTypeError;
use Tessero\Exceptions\ShapeError;
use Tessero\Exceptions\TesseroException;
use Tessero\Native\Abi;
use Tessero\Native\Library;

/**
 * Universal functions (ufuncs): element-wise functions over arrays of any shape, with NumPy's broadcasting and type rules and an optional output array.
 *
 *     Math::sin($x);                    // new array
 *     Math::hypot($x, $y, out: $buf);   // written into $buf (no allocation)
 *     Math::cbrt($a, $a);               // in place
 *
 * The native extension has the same class as Tessero\Ext\Math, with the same
 * names, signatures and results: both call libtessero's loops.
 *
 * Kernel ufuncs (sin, exp, add, maximum, comparisons, ...) run the NDArray
 * methods, so Math::sin($a) and $a->sin() are identical to the bit. Loop
 * ufuncs (cbrt, erf, gamma, logaddexp, fmax, copysign, fmod, ...) are
 * libtessero's typed inner loops (tsr_ufunc), vectorised and threaded like
 * the other kernels.
 *
 * With `out`, the result is written into that array: directly when the dtype
 * and shape match and it does not partially overlap an input, otherwise
 * through a temporary and a same-kind cast (float64 -> float32 is allowed,
 * float -> int is not). The result broadcasts into `out`.
 */
final class Math
{
    /** name => [NDArray method, inputs, method takes out, summary] */
    private const NATIVE = [
        'negative' => ['neg', 1, true, '-x'],
        'absolute' => ['abs', 1, true, '|x| (modulus for complex)'],
        'abs' => ['abs', 1, true, 'alias of absolute'],
        'square' => ['square', 1, true, 'x * x'],
        'sign' => ['sign', 1, true, '-1, 0 or 1'],
        'sqrt' => ['sqrt', 1, true, 'square root'],
        'exp' => ['exp', 1, true, 'e^x'],
        'expm1' => ['expm1', 1, true, 'e^x - 1, accurate near 0'],
        'log' => ['log', 1, true, 'natural logarithm'],
        'log10' => ['log10', 1, true, 'base-10 logarithm'],
        'log2' => ['log2', 1, true, 'base-2 logarithm'],
        'log1p' => ['log1p', 1, true, 'log(1 + x), accurate near 0'],
        'sin' => ['sin', 1, true, 'sine (radians)'],
        'cos' => ['cos', 1, true, 'cosine (radians)'],
        'tan' => ['tan', 1, true, 'tangent (radians)'],
        'arcsin' => ['arcsin', 1, true, 'inverse sine'],
        'arccos' => ['arccos', 1, true, 'inverse cosine'],
        'arctan' => ['arctan', 1, true, 'inverse tangent'],
        'sinh' => ['sinh', 1, true, 'hyperbolic sine'],
        'cosh' => ['cosh', 1, true, 'hyperbolic cosine'],
        'tanh' => ['tanh', 1, true, 'hyperbolic tangent'],
        'floor' => ['floor', 1, true, 'round down'],
        'ceil' => ['ceil', 1, true, 'round up'],
        'rint' => ['rint', 1, true, 'round to nearest, ties to even'],
        'reciprocal' => ['reciprocal', 1, true, '1 / x'],
        'isnan' => ['isnan', 1, false, 'x is NaN (bool)'],
        'isinf' => ['isinf', 1, false, 'x is +/-INF (bool)'],
        'isfinite' => ['isfinite', 1, false, 'x is finite (bool)'],
        'invert' => ['invert', 1, false, 'bitwise NOT (logical NOT for bool)'],
        'real' => ['real', 1, false, 'real part'],
        'imag' => ['imag', 1, false, 'imaginary part'],
        'conj' => ['conj', 1, false, 'complex conjugate'],
        'angle' => ['angle', 1, false, 'argument of a complex number'],
        'add' => ['add', 2, true, 'x + y'],
        'subtract' => ['sub', 2, true, 'x - y'],
        'multiply' => ['mul', 2, true, 'x * y'],
        'divide' => ['div', 2, true, 'x / y (true division)'],
        'power' => ['pow', 2, true, 'x ** y'],
        'mod' => ['mod', 2, true, 'remainder with the sign of y (Python %)'],
        'remainder' => ['mod', 2, true, 'alias of mod'],
        'floorDivide' => ['floorDiv', 2, true, 'floor(x / y)'],
        'maximum' => ['maximum', 2, true, 'element-wise maximum, NaN propagates'],
        'minimum' => ['minimum', 2, true, 'element-wise minimum, NaN propagates'],
        'arctan2' => ['atan2', 2, true, 'angle of (x=y2, y=x1): atan2(x1, x2)'],
        'hypot' => ['hypot', 2, true, 'sqrt(x^2 + y^2) without overflow'],
        'equal' => ['eq', 2, false, 'x == y (bool)'],
        'notEqual' => ['ne', 2, false, 'x != y (bool)'],
        'less' => ['lt', 2, false, 'x < y (bool)'],
        'lessEqual' => ['le', 2, false, 'x <= y (bool)'],
        'greater' => ['gt', 2, false, 'x > y (bool)'],
        'greaterEqual' => ['ge', 2, false, 'x >= y (bool)'],
        'logicalAnd' => ['logicalAnd', 2, false, 'x AND y (bool)'],
        'logicalOr' => ['logicalOr', 2, false, 'x OR y (bool)'],
        'logicalXor' => ['logicalXor', 2, false, 'x XOR y (bool)'],
    ];

    private const BOOL_RESULT = ['isnan', 'isinf', 'isfinite', 'equal', 'notEqual', 'less', 'lessEqual', 'greater', 'greaterEqual',
        'logicalAnd', 'logicalOr', 'logicalXor'];

    private const FLOAT_RESULT = ['sqrt', 'exp', 'expm1', 'log', 'log10', 'log2', 'log1p', 'sin', 'cos', 'tan', 'arcsin', 'arccos',
        'arctan', 'sinh', 'cosh', 'tanh', 'reciprocal', 'divide', 'arctan2', 'hypot'];

    private function __construct()
    {
    }

    /**
     * Every ufunc: ['sin' => ['nin' => 1, 'engine' => 'kernel'|'loop', 'summary' => ...], ...].
     *
     * @return array<string, array{nin: int, engine: string, summary: string}>
     */
    public static function ufuncs(): array
    {
        $out = [];
        foreach (self::NATIVE as $name => [, $nin, , $summary]) {
            $out[$name] = ['nin' => $nin, 'engine' => 'kernel', 'summary' => $summary];
        }
        $ffi = Library::ffi();
        $n = $ffi->tsr_ufunc_count();
        for ($id = 0; $id < $n; $id++) {
            $out[(string) $ffi->tsr_ufunc_name($id)] = [
                'nin' => $ffi->tsr_ufunc_nin($id), 'engine' => 'loop', 'summary' => (string) $ffi->tsr_ufunc_summary($id),
            ];
        }

        return $out;
    }

    /**
     * Call a ufunc by name: Math::apply('cbrt', $x), Math::apply('fmax', $x, $y, $out).
     * For a one-input ufunc the third argument may be the output array.
     */
    public static function apply(string $name, mixed $x, mixed $y = null, ?NDArray $out = null): NDArray
    {
        if (isset(self::NATIVE[$name])) {
            [$method, $nin, $takesOut] = self::NATIVE[$name];
            if ($nin === 1 && $out === null && $y instanceof NDArray) {
                $out = $y;
            }
            if ($nin === 2 && $y === null) {
                throw new \ArgumentCountError("Math::{$name}() takes two operands");
            }
            $a = self::operand($x, $nin === 2 && $y instanceof NDArray ? $y : null);
            if ($out !== null) {
                self::requireWritable($out);
                // the NDArray methods cast into out freely; ufuncs allow same-kind casts only (as ext-tessero)
                $in = $nin === 2 ? self::resultType($x, $y, $a, self::operand($y, $a)) : $a->dtype();
                $kind = match (true) {
                    in_array($name, self::BOOL_RESULT, true) => 0,
                    in_array($name, self::FLOAT_RESULT, true) => max(2, self::kind($in)),
                    in_array($name, ['absolute', 'abs', 'real', 'imag', 'angle'], true) && $in === DType::Complex128 => 2,
                    default => self::kind($in),
                };
                if ($kind > self::kind($out->dtype())) {
                    throw new DTypeError("Math::{$name}(): cannot cast the result to {$out->dtype()->name()}");
                }
            }
            if ($nin === 1) {
                $r = $takesOut ? $a->{$method}($out) : $a->{$method}();
            } else {
                $r = $takesOut ? $a->{$method}($y, $out) : $a->{$method}($y);
            }

            return $out !== null && $r !== $out ? self::castInto($r, $out, $name) : $r;
        }

        $ffi = Library::ffi();
        $id = $ffi->tsr_ufunc_find($name);
        if ($id < 0) {
            throw new \ValueError("Math::apply(): '{$name}' is not a ufunc (see Math::ufuncs())");
        }
        $nin = $ffi->tsr_ufunc_nin($id);
        if ($nin === 1 && $out === null && $y instanceof NDArray) {
            $out = $y;
        }
        if ($nin === 2 && $y === null) {
            throw new \ArgumentCountError("Math::{$name}() takes two operands");
        }

        return self::loop($ffi, $name, $id, $nin, $x, $nin === 2 ? $y : null, $out);
    }

    /** -x. */
    public static function negative(mixed $x, ?NDArray $out = null): NDArray
    {
        return self::apply('negative', $x, null, $out);
    }

    /** |x| (modulus for complex). */
    public static function absolute(mixed $x, ?NDArray $out = null): NDArray
    {
        return self::apply('absolute', $x, null, $out);
    }

    /** alias of absolute. */
    public static function abs(mixed $x, ?NDArray $out = null): NDArray
    {
        return self::apply('abs', $x, null, $out);
    }

    /** x * x. */
    public static function square(mixed $x, ?NDArray $out = null): NDArray
    {
        return self::apply('square', $x, null, $out);
    }

    /** -1, 0 or 1. */
    public static function sign(mixed $x, ?NDArray $out = null): NDArray
    {
        return self::apply('sign', $x, null, $out);
    }

    /** square root. */
    public static function sqrt(mixed $x, ?NDArray $out = null): NDArray
    {
        return self::apply('sqrt', $x, null, $out);
    }

    /** e^x. */
    public static function exp(mixed $x, ?NDArray $out = null): NDArray
    {
        return self::apply('exp', $x, null, $out);
    }

    /** e^x - 1, accurate near 0. */
    public static function expm1(mixed $x, ?NDArray $out = null): NDArray
    {
        return self::apply('expm1', $x, null, $out);
    }

    /** natural logarithm. */
    public static function log(mixed $x, ?NDArray $out = null): NDArray
    {
        return self::apply('log', $x, null, $out);
    }

    /** base-10 logarithm. */
    public static function log10(mixed $x, ?NDArray $out = null): NDArray
    {
        return self::apply('log10', $x, null, $out);
    }

    /** base-2 logarithm. */
    public static function log2(mixed $x, ?NDArray $out = null): NDArray
    {
        return self::apply('log2', $x, null, $out);
    }

    /** log(1 + x), accurate near 0. */
    public static function log1p(mixed $x, ?NDArray $out = null): NDArray
    {
        return self::apply('log1p', $x, null, $out);
    }

    /** sine (radians). */
    public static function sin(mixed $x, ?NDArray $out = null): NDArray
    {
        return self::apply('sin', $x, null, $out);
    }

    /** cosine (radians). */
    public static function cos(mixed $x, ?NDArray $out = null): NDArray
    {
        return self::apply('cos', $x, null, $out);
    }

    /** tangent (radians). */
    public static function tan(mixed $x, ?NDArray $out = null): NDArray
    {
        return self::apply('tan', $x, null, $out);
    }

    /** inverse sine. */
    public static function arcsin(mixed $x, ?NDArray $out = null): NDArray
    {
        return self::apply('arcsin', $x, null, $out);
    }

    /** inverse cosine. */
    public static function arccos(mixed $x, ?NDArray $out = null): NDArray
    {
        return self::apply('arccos', $x, null, $out);
    }

    /** inverse tangent. */
    public static function arctan(mixed $x, ?NDArray $out = null): NDArray
    {
        return self::apply('arctan', $x, null, $out);
    }

    /** hyperbolic sine. */
    public static function sinh(mixed $x, ?NDArray $out = null): NDArray
    {
        return self::apply('sinh', $x, null, $out);
    }

    /** hyperbolic cosine. */
    public static function cosh(mixed $x, ?NDArray $out = null): NDArray
    {
        return self::apply('cosh', $x, null, $out);
    }

    /** hyperbolic tangent. */
    public static function tanh(mixed $x, ?NDArray $out = null): NDArray
    {
        return self::apply('tanh', $x, null, $out);
    }

    /** round down. */
    public static function floor(mixed $x, ?NDArray $out = null): NDArray
    {
        return self::apply('floor', $x, null, $out);
    }

    /** round up. */
    public static function ceil(mixed $x, ?NDArray $out = null): NDArray
    {
        return self::apply('ceil', $x, null, $out);
    }

    /** round to nearest, ties to even. */
    public static function rint(mixed $x, ?NDArray $out = null): NDArray
    {
        return self::apply('rint', $x, null, $out);
    }

    /** 1 / x. */
    public static function reciprocal(mixed $x, ?NDArray $out = null): NDArray
    {
        return self::apply('reciprocal', $x, null, $out);
    }

    /** x is NaN (bool). */
    public static function isnan(mixed $x, ?NDArray $out = null): NDArray
    {
        return self::apply('isnan', $x, null, $out);
    }

    /** x is +/-INF (bool). */
    public static function isinf(mixed $x, ?NDArray $out = null): NDArray
    {
        return self::apply('isinf', $x, null, $out);
    }

    /** x is finite (bool). */
    public static function isfinite(mixed $x, ?NDArray $out = null): NDArray
    {
        return self::apply('isfinite', $x, null, $out);
    }

    /** bitwise NOT (logical NOT for bool). */
    public static function invert(mixed $x, ?NDArray $out = null): NDArray
    {
        return self::apply('invert', $x, null, $out);
    }

    /** real part. */
    public static function real(mixed $x, ?NDArray $out = null): NDArray
    {
        return self::apply('real', $x, null, $out);
    }

    /** imaginary part. */
    public static function imag(mixed $x, ?NDArray $out = null): NDArray
    {
        return self::apply('imag', $x, null, $out);
    }

    /** complex conjugate. */
    public static function conj(mixed $x, ?NDArray $out = null): NDArray
    {
        return self::apply('conj', $x, null, $out);
    }

    /** argument of a complex number. */
    public static function angle(mixed $x, ?NDArray $out = null): NDArray
    {
        return self::apply('angle', $x, null, $out);
    }

    /** x + y. */
    public static function add(mixed $x, mixed $y, ?NDArray $out = null): NDArray
    {
        return self::apply('add', $x, $y, $out);
    }

    /** x - y. */
    public static function subtract(mixed $x, mixed $y, ?NDArray $out = null): NDArray
    {
        return self::apply('subtract', $x, $y, $out);
    }

    /** x * y. */
    public static function multiply(mixed $x, mixed $y, ?NDArray $out = null): NDArray
    {
        return self::apply('multiply', $x, $y, $out);
    }

    /** x / y (true division). */
    public static function divide(mixed $x, mixed $y, ?NDArray $out = null): NDArray
    {
        return self::apply('divide', $x, $y, $out);
    }

    /** x ** y. */
    public static function power(mixed $x, mixed $y, ?NDArray $out = null): NDArray
    {
        return self::apply('power', $x, $y, $out);
    }

    /** remainder with the sign of y (Python %). */
    public static function mod(mixed $x, mixed $y, ?NDArray $out = null): NDArray
    {
        return self::apply('mod', $x, $y, $out);
    }

    /** alias of mod. */
    public static function remainder(mixed $x, mixed $y, ?NDArray $out = null): NDArray
    {
        return self::apply('remainder', $x, $y, $out);
    }

    /** floor(x / y). */
    public static function floorDivide(mixed $x, mixed $y, ?NDArray $out = null): NDArray
    {
        return self::apply('floorDivide', $x, $y, $out);
    }

    /** element-wise maximum, NaN propagates. */
    public static function maximum(mixed $x, mixed $y, ?NDArray $out = null): NDArray
    {
        return self::apply('maximum', $x, $y, $out);
    }

    /** element-wise minimum, NaN propagates. */
    public static function minimum(mixed $x, mixed $y, ?NDArray $out = null): NDArray
    {
        return self::apply('minimum', $x, $y, $out);
    }

    /** angle of (x=y2, y=x1): atan2(x1, x2). */
    public static function arctan2(mixed $x, mixed $y, ?NDArray $out = null): NDArray
    {
        return self::apply('arctan2', $x, $y, $out);
    }

    /** sqrt(x^2 + y^2) without overflow. */
    public static function hypot(mixed $x, mixed $y, ?NDArray $out = null): NDArray
    {
        return self::apply('hypot', $x, $y, $out);
    }

    /** x == y (bool). */
    public static function equal(mixed $x, mixed $y, ?NDArray $out = null): NDArray
    {
        return self::apply('equal', $x, $y, $out);
    }

    /** x != y (bool). */
    public static function notEqual(mixed $x, mixed $y, ?NDArray $out = null): NDArray
    {
        return self::apply('notEqual', $x, $y, $out);
    }

    /** x < y (bool). */
    public static function less(mixed $x, mixed $y, ?NDArray $out = null): NDArray
    {
        return self::apply('less', $x, $y, $out);
    }

    /** x <= y (bool). */
    public static function lessEqual(mixed $x, mixed $y, ?NDArray $out = null): NDArray
    {
        return self::apply('lessEqual', $x, $y, $out);
    }

    /** x > y (bool). */
    public static function greater(mixed $x, mixed $y, ?NDArray $out = null): NDArray
    {
        return self::apply('greater', $x, $y, $out);
    }

    /** x >= y (bool). */
    public static function greaterEqual(mixed $x, mixed $y, ?NDArray $out = null): NDArray
    {
        return self::apply('greaterEqual', $x, $y, $out);
    }

    /** x AND y (bool). */
    public static function logicalAnd(mixed $x, mixed $y, ?NDArray $out = null): NDArray
    {
        return self::apply('logicalAnd', $x, $y, $out);
    }

    /** x OR y (bool). */
    public static function logicalOr(mixed $x, mixed $y, ?NDArray $out = null): NDArray
    {
        return self::apply('logicalOr', $x, $y, $out);
    }

    /** x XOR y (bool). */
    public static function logicalXor(mixed $x, mixed $y, ?NDArray $out = null): NDArray
    {
        return self::apply('logicalXor', $x, $y, $out);
    }

    /** cube root. */
    public static function cbrt(mixed $x, ?NDArray $out = null): NDArray
    {
        return self::apply('cbrt', $x, null, $out);
    }

    /** 2^x. */
    public static function exp2(mixed $x, ?NDArray $out = null): NDArray
    {
        return self::apply('exp2', $x, null, $out);
    }

    /** round towards zero. */
    public static function trunc(mixed $x, ?NDArray $out = null): NDArray
    {
        return self::apply('trunc', $x, null, $out);
    }

    /** inverse hyperbolic sine. */
    public static function arcsinh(mixed $x, ?NDArray $out = null): NDArray
    {
        return self::apply('arcsinh', $x, null, $out);
    }

    /** inverse hyperbolic cosine. */
    public static function arccosh(mixed $x, ?NDArray $out = null): NDArray
    {
        return self::apply('arccosh', $x, null, $out);
    }

    /** inverse hyperbolic tangent. */
    public static function arctanh(mixed $x, ?NDArray $out = null): NDArray
    {
        return self::apply('arctanh', $x, null, $out);
    }

    /** radians to degrees. */
    public static function degrees(mixed $x, ?NDArray $out = null): NDArray
    {
        return self::apply('degrees', $x, null, $out);
    }

    /** degrees to radians. */
    public static function radians(mixed $x, ?NDArray $out = null): NDArray
    {
        return self::apply('radians', $x, null, $out);
    }

    /** error function (scipy.special.erf). */
    public static function erf(mixed $x, ?NDArray $out = null): NDArray
    {
        return self::apply('erf', $x, null, $out);
    }

    /** complementary error function. */
    public static function erfc(mixed $x, ?NDArray $out = null): NDArray
    {
        return self::apply('erfc', $x, null, $out);
    }

    /** gamma function (scipy.special.gamma). */
    public static function gamma(mixed $x, ?NDArray $out = null): NDArray
    {
        return self::apply('gamma', $x, null, $out);
    }

    /** log|gamma(x)| (scipy.special.gammaln). */
    public static function lgamma(mixed $x, ?NDArray $out = null): NDArray
    {
        return self::apply('lgamma', $x, null, $out);
    }

    /** sign bit is set (bool; true for -0.0). */
    public static function signbit(mixed $x, ?NDArray $out = null): NDArray
    {
        return self::apply('signbit', $x, null, $out);
    }

    /** maximum ignoring NaN. */
    public static function fmax(mixed $x, mixed $y, ?NDArray $out = null): NDArray
    {
        return self::apply('fmax', $x, $y, $out);
    }

    /** minimum ignoring NaN. */
    public static function fmin(mixed $x, mixed $y, ?NDArray $out = null): NDArray
    {
        return self::apply('fmin', $x, $y, $out);
    }

    /** magnitude of x with the sign of y. */
    public static function copysign(mixed $x, mixed $y, ?NDArray $out = null): NDArray
    {
        return self::apply('copysign', $x, $y, $out);
    }

    /** next representable value after x towards y. */
    public static function nextafter(mixed $x, mixed $y, ?NDArray $out = null): NDArray
    {
        return self::apply('nextafter', $x, $y, $out);
    }

    /** 0 where x1 < 0, x2 where x1 == 0, 1 where x1 > 0 (numpy.heaviside). */
    public static function heaviside(mixed $x1, mixed $x2, ?NDArray $out = null): NDArray
    {
        return self::apply('heaviside', $x1, $x2, $out);
    }

    /** Greatest common divisor, element-wise (numpy.gcd; integer inputs). */
    public static function gcd(mixed $x1, mixed $x2, ?NDArray $out = null): NDArray
    {
        return self::apply('gcd', $x1, $x2, $out);
    }

    /** Least common multiple, element-wise (numpy.lcm; integer inputs). */
    public static function lcm(mixed $x1, mixed $x2, ?NDArray $out = null): NDArray
    {
        return self::apply('lcm', $x1, $x2, $out);
    }

    /** x1 ** x2, computed in float64 (numpy.float_power). */
    public static function floatPower(mixed $x1, mixed $x2, ?NDArray $out = null): NDArray
    {
        return self::apply('floatPower', $x1, $x2, $out);
    }

    /** log(e^x + e^y) without overflow. */
    public static function logaddexp(mixed $x, mixed $y, ?NDArray $out = null): NDArray
    {
        return self::apply('logaddexp', $x, $y, $out);
    }

    /** log2(2^x + 2^y) without overflow. */
    public static function logaddexp2(mixed $x, mixed $y, ?NDArray $out = null): NDArray
    {
        return self::apply('logaddexp2', $x, $y, $out);
    }

    /** C remainder with the sign of x (numpy.fmod); integer x % 0 gives 0. */
    public static function fmod(mixed $x, mixed $y, ?NDArray $out = null): NDArray
    {
        return self::apply('fmod', $x, $y, $out);
    }

    // ------------------------------------------------------------------ engine

    private static function loop(FFI $ffi, string $name, int $id, int $nin, mixed $x, mixed $y, ?NDArray $out): NDArray
    {
        if ($out !== null) {
            self::requireWritable($out);
        }
        $b = null;
        if ($nin === 2) {
            $a = self::operand($x, $y instanceof NDArray ? $y : null);
            $b = self::operand($y, $x instanceof NDArray ? $x : null);
            $dt = self::resultType($x, $y, $a, $b);
        } else {
            $a = self::operand($x, null);
            $dt = $a->dtype();
        }
        if ($dt === DType::Complex128) {
            throw new DTypeError("Math::{$name}() does not support complex128");
        }
        $r = $ffi->new('int[3]');
        $how = $ffi->tsr_ufunc_resolve($id, $dt->value, FFI::addr($r[0]), FFI::addr($r[1]), FFI::addr($r[2]));
        if ($how < 0) {
            throw new DTypeError("Math::{$name}() has no loop for {$dt->name()}");
        }
        if ($how === 2) {                                   // fmax on integers is maximum
            return self::apply($r[2] === Abi::MAX ? 'maximum' : 'minimum', $x, $y, $out);
        }
        if ($how === 1) {                                   // trunc on integers is the identity
            $copy = $a->astype($dt, copy: true);

            return $out !== null ? self::castInto($copy, $out, $name) : $copy;
        }
        $loopDt = DType::from($r[0]);
        $outDt = DType::from($r[1]);
        $a = $a->dtype() === $loopDt ? $a : $a->astype($loopDt);
        $shape = $a->shape();
        if ($b !== null) {
            $b = $b->dtype() === $loopDt ? $b : $b->astype($loopDt);
            try {
                $shape = NDArray::broadcastShapes($a->shape(), $b->shape());
            } catch (ShapeError) {
                throw new ShapeError("Math::{$name}(): shapes " . self::shapeStr($a->shape()) . ' and ' . self::shapeStr($b->shape()) . ' cannot be broadcast together');
            }
            $b = $b->shape() === $shape ? $b : $b->broadcastTo($shape);
        }
        $a = $a->shape() === $shape ? $a : $a->broadcastTo($shape);

        $direct = $out !== null && $out->dtype() === $outDt && $out->shape() === $shape
            && ! self::clobbers($out, $a) && ($b === null || ! self::clobbers($out, $b));
        $target = $direct ? $out : NDArray::empty($shape, $outDt);
        if ($target->size() > 0) {
            Library::check($ffi->tsr_ufunc(
                $id, $loopDt->value, count($shape), Library::i64($shape),
                $a->ptr(), Library::i64($a->strides()),
                $b?->ptr(), $b === null ? null : Library::i64($b->strides()),
                $target->ptr(), Library::i64($target->strides()),
            ), "Math::{$name}()");
        }
        if ($out !== null && ! $direct) {
            return self::castInto($target, $out, $name);
        }

        return $target;
    }

    /** A PHP value as an array; a PHP scalar next to an array takes that array's dtype (NumPy's weak scalars). */
    private static function operand(mixed $v, ?NDArray $like): NDArray
    {
        if ($v instanceof NDArray) {
            return $v;
        }
        $scalar = is_int($v) || is_float($v) || is_bool($v);
        if ($scalar && $like !== null) {
            return NDArray::full([], $v, $like->dtype()->withScalar($v));
        }
        if ($scalar || is_array($v)) {
            return NDArray::array($v);
        }

        throw new DTypeError('Math: unsupported operand of type ' . get_debug_type($v));
    }

    private static function resultType(mixed $x, mixed $y, NDArray $a, NDArray $b): DType
    {
        if ((is_int($x) || is_float($x) || is_bool($x)) && $y instanceof NDArray) {
            return $b->dtype()->withScalar($x);
        }
        if ((is_int($y) || is_float($y) || is_bool($y)) && $x instanceof NDArray) {
            return $a->dtype()->withScalar($y);
        }

        return DType::promote($a->dtype(), $b->dtype());
    }

    /** Would writing $out overwrite elements of $in before they are read? (Same buffer, different layout.) */
    private static function clobbers(NDArray $out, NDArray $in): bool
    {
        if ($in->buffer() !== $out->buffer()) {
            return false;
        }
        if ($in->offset() !== $out->offset()) {
            return true;
        }
        $os = $out->strides();
        $shape = $out->shape();
        foreach ($in->strides() as $d => $s) {
            if ($s !== $os[$d] && $shape[$d] > 1) {
                return true;
            }
        }

        return false;
    }

    /** Write $r into $out, broadcasting, with NumPy's same-kind casting rule. */
    private static function castInto(NDArray $r, NDArray $out, string $name): NDArray
    {
        if (self::kind($r->dtype()) > self::kind($out->dtype())) {
            throw new DTypeError("Math::{$name}(): cannot cast the result from {$r->dtype()->name()} to {$out->dtype()->name()}");
        }
        try {
            $b = $r->shape() === $out->shape() ? $r : $r->broadcastTo($out->shape());
        } catch (ShapeError) {
            throw new ShapeError("Math::{$name}(): result of shape " . self::shapeStr($r->shape()) . ' does not fit an output of shape ' . self::shapeStr($out->shape()));
        }

        return $out->assign($b);
    }

    private static function kind(DType $d): int
    {
        return match ($d) {
            DType::Bool => 0,
            DType::UInt8, DType::Int32, DType::Int64 => 1,
            DType::Float32, DType::Float64 => 2,
            DType::Complex128 => 3,
        };
    }

    private static function requireWritable(NDArray $out): void
    {
        if (! $out->isWritable()) {
            throw new TesseroException('out: the array is read-only');
        }
    }

    /** @param list<int> $s */
    private static function shapeStr(array $s): string
    {
        return '(' . implode(', ', $s) . (count($s) === 1 ? ',' : '') . ')';
    }
}
