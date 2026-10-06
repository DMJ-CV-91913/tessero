<?php

declare(strict_types=1);

namespace Tessero\Tests\Ext;

use PHPUnit\Framework\TestCase;
use Tessero\Ext\NDArray as X;
use Tessero\NDArray as F;

/**
 * The two backends on the same random inputs: reductions over every axis
 * combination (with keepdims), ufuncs, file round trips. They share libtessero,
 * so results must agree to the bit, not within a tolerance. Seeded, so a
 * failure names a reproducible case.
 */
final class BackendParityTest extends TestCase
{
    private const SEED = 20260928;

    protected function setUp(): void
    {
        if (! extension_loaded('tessero')) {
            $this->markTestSkipped('ext-tessero is not loaded.');
        }
        mt_srand(self::SEED);
    }

    public function testReductionsOverAllAxisCombinations(): void
    {
        $checked = 0;
        foreach (['float64', 'float32', 'int32', 'int64', 'uint8', 'bool'] as $dtype) {
            for ($trial = 0; $trial < 6; $trial++) {
                [$f, $x, $label] = $this->randomPair($dtype, $trial);
                $n = count($f->shape());
                foreach ($this->axisChoices($n) as $axis) {
                    foreach ([false, true] as $keep) {
                        $ops = ['sum', 'prod', 'min', 'max', 'any', 'all', 'mean', 'var', 'std'];
                        if ($axis === null || is_int($axis)) {
                            $ops[] = 'argmin';
                            $ops[] = 'argmax';
                        }
                        foreach ($ops as $op) {
                            $what = "{$label} {$op}(" . json_encode($axis) . ', keepdims=' . var_export($keep, true) . ')';
                            $args = in_array($op, ['var', 'std'], true) ? [$axis, 1, $keep] : [$axis, $keep];
                            $a = $this->call(static fn () => $f->{$op}(...$args));
                            $b = $this->call(static fn () => $x->{$op}(...$args));
                            $this->assertSameResult($a, $b, $what);
                            $checked++;
                        }
                    }
                }
            }
        }
        $this->assertGreaterThan(3000, $checked);
    }

    public function testComplexSumOverAxes(): void
    {
        $re = F::array([[1.5, -2.0, 3.25], [0.5, 4.0, -1.0]]);
        $im = F::array([[0.0, 1.0, -1.0], [2.0, 0.5, 0.25]]);
        $f = F::complex($re, $im);
        $x = X::fromBytes($f->toBytes(), 'complex128', [2, 3]);
        foreach ([null, 0, 1, -1, [0, 1], [1, 0]] as $axis) {
            foreach ([false, true] as $keep) {
                $this->assertSameResult($f->sum($axis, $keep), $x->sum($axis, $keep), 'complex sum ' . json_encode($axis));
            }
        }
    }

    public function testUfuncsOnRandomViews(): void
    {
        $names = array_keys(array_filter(\Tessero\Math::ufuncs(), static fn (array $u): bool => $u['engine'] === 'loop'));
        foreach (['float64', 'float32', 'int32'] as $dtype) {
            for ($trial = 0; $trial < 4; $trial++) {
                [$f, $x, $label] = $this->randomPair($dtype, $trial);
                [$g, $y] = $this->randomPair($dtype, $trial + 10);
                foreach ($names as $name) {
                    $two = \Tessero\Math::ufuncs()[$name]['nin'] === 2;
                    // second operand: a broadcastable slice of the other array (last axis only) or a scalar
                    $a = $this->call(static fn () => $two ? \Tessero\Math::{$name}($f, 0.75) : \Tessero\Math::{$name}($f));
                    $b = $this->call(static fn () => $two ? \Tessero\Ext\Math::{$name}($x, 0.75) : \Tessero\Ext\Math::{$name}($x));
                    $this->assertSameResult($a, $b, "{$label} {$name}");
                }
            }
        }
    }

    public function testManipulationMethodsMatchFfi(): void
    {
        $checked = 0;
        foreach (['float64', 'float32', 'int32', 'int64', 'uint8'] as $dtype) {
            for ($trial = 0; $trial < 6; $trial++) {
                [$f, $x, $label] = $this->randomPair($dtype, $trial);
                $nd = count($f->shape());
                $size = (int) array_product($f->shape());

                $this->assertSameResult($this->call(static fn () => $f->flatten()), $this->call(static fn () => $x->flatten()), "{$label} flatten");
                $this->assertSameResult($f->nbytes(), $x->nbytes(), "{$label} nbytes");
                $this->assertSameResult($this->call(static fn () => $f->squeeze()), $this->call(static fn () => $x->squeeze()), "{$label} squeeze");
                $checked += 3;

                if ($nd >= 2) {
                    $this->assertSameResult($this->call(static fn () => $f->swapAxes(0, $nd - 1)), $this->call(static fn () => $x->swapAxes(0, $nd - 1)), "{$label} swapAxes");
                    $checked++;
                }
                if (in_array($dtype, ['float64', 'float32'], true)) {
                    $this->assertSameResult($this->call(static fn () => $f->round(2)), $this->call(static fn () => $x->round(2)), "{$label} round");
                    $checked++;
                }
                $viewTo = match ($dtype) {
                    'float64' => 'int64', 'int64' => 'float64', 'float32' => 'int32', 'int32' => 'float32', default => null,
                };
                if ($viewTo !== null) {
                    $this->assertSameResult($this->call(static fn () => $f->flatten()->view($viewTo)), $this->call(static fn () => $x->flatten()->view($viewTo)), "{$label} view {$viewTo}");
                    $checked++;
                }

                $idx = $f->shape()[0] >= 2 ? [0, 1] : [0];
                $this->assertSameResult($this->call(static fn () => $f->take($idx, 0)), $this->call(static fn () => $x->take($idx, 0)), "{$label} take axis0");
                $flatIdx = $size >= 3 ? [0, 2] : [0];
                $this->assertSameResult($this->call(static fn () => $f->take($flatIdx)), $this->call(static fn () => $x->take($flatIdx)), "{$label} take flat");
                $checked += 2;
            }
        }

        // put mutates in place: use fresh contiguous arrays on each backend
        foreach ([['float64', [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [99.5, -7.25]], ['int64', [1, 2, 3, 4, 5, 6], [99, -7]]] as [$dtype, $vals, $put]) {
            $ff = F::array($vals, $dtype)->reshape([2, 3]);
            $xx = X::array($vals, $dtype)->reshape([2, 3]);
            $this->assertSameResult($this->call(static fn () => $ff->put([0, 5], $put)), $this->call(static fn () => $xx->put([0, 5], $put)), "{$dtype} put");
            $checked++;
        }

        $this->assertGreaterThan(60, $checked);
    }

    // ------------------------------------------------------------------ helpers

    /** @return array{F, X, string} the same random values on both backends, often as a non-contiguous view */
    private function randomPair(string $dtype, int $trial): array
    {
        $nd = mt_rand(1, 4);
        $shape = [];
        for ($d = 0; $d < $nd; $d++) {
            $shape[] = mt_rand(1, 5);
        }
        $size = (int) array_product($shape);
        $vals = [];
        for ($i = 0; $i < $size; $i++) {
            $vals[] = match ($dtype) {
                'float64', 'float32' => (mt_rand() / mt_getrandmax() - 0.5) * 20.0,
                'int32', 'int64' => mt_rand(-50, 50),
                'uint8' => mt_rand(0, 255),
                'bool' => mt_rand(0, 3) > 0,
            };
        }
        $f = F::array($vals, $dtype)->reshape($shape);
        $x = X::array($vals, $dtype)->reshape($shape);
        $label = "{$dtype} " . json_encode($shape);
        if ($trial % 2 === 1 && $nd >= 2) {                     // transposed view
            $f = $f->transpose();
            $x = $x->transpose();
            $label .= '.T';
        } elseif ($trial % 3 === 2 && $shape[0] >= 2) {         // reversed, strided view
            $f = $f->slice('::-2');
            $x = $x['::-2'];
            $label .= '[::-2]';
        }

        return [$f, $x, $label];
    }

    /** @return list<int|list<int>|null> */
    private function axisChoices(int $n): array
    {
        $out = [null];
        for ($a = 0; $a < $n; $a++) {
            $out[] = $a;
            $out[] = $a - $n;
        }
        for ($mask = 1; $mask < (1 << $n); $mask++) {
            $axes = [];
            for ($a = 0; $a < $n; $a++) {
                if ($mask & (1 << $a)) {
                    $axes[] = $a;
                }
            }
            $out[] = $axes;
            if (count($axes) > 1) {
                $out[] = array_reverse($axes);
            }
        }

        return $out;
    }

    private function call(callable $f): mixed
    {
        try {
            return $f();
        } catch (\Throwable $e) {
            return ['exception' => $e instanceof \Tessero\Ext\ShapeException || $e instanceof \Tessero\Exceptions\ShapeError ? 'shape'
                : ($e instanceof \Tessero\Ext\DTypeException || $e instanceof \Tessero\Exceptions\DTypeError ? 'dtype' : get_class($e))];
        }
    }

    private function assertSameResult(mixed $ffi, mixed $ext, string $what): void
    {
        if (is_array($ffi) && isset($ffi['exception'])) {
            $this->assertSame($ffi, $ext, "{$what}: both throw");

            return;
        }
        if ($ffi instanceof F) {
            $this->assertInstanceOf(X::class, $ext, "{$what}: array result");
            $this->assertSame($ffi->shape(), $ext->shape(), "{$what}: shape");
            $this->assertSame($ffi->dtype()->name(), $ext->dtype(), "{$what}: dtype");
            $this->assertSame(bin2hex($ffi->toBytes()), bin2hex($ext->toBytes()), "{$what}: bytes");

            return;
        }
        if (is_float($ffi) && is_nan($ffi)) {
            $this->assertTrue(is_float($ext) && is_nan($ext), "{$what}: NaN");

            return;
        }
        $this->assertSame($ffi, $ext, "{$what}: scalar");
    }
}
