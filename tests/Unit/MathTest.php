<?php

declare(strict_types=1);

namespace Tessero\Tests\Unit;

use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\TestCase;
use Tessero\DType;
use Tessero\Exceptions\DTypeError;
use Tessero\Exceptions\ShapeError;
use Tessero\Exceptions\TesseroException;
use Tessero\Math;
use Tessero\NDArray;
use Tessero\Tessero;

/**
 * Tessero\Math (FFI package) against the NumPy / scipy.special fixtures, the
 * NDArray methods, and, when ext-tessero is loaded, Tessero\Ext\Math byte for byte.
 */
final class MathTest extends TestCase
{
    protected function tearDown(): void
    {
        Tessero::setThreads(1);
    }

    public static function ufunc(): iterable
    {
        $d = json_decode((string) file_get_contents(__DIR__ . '/../fixtures/ufunc.json'), true);
        foreach ($d['cases'] as $i => $c) {
            yield sprintf('%02d %s(%s)', $i, $c['f'], implode(', ', array_map(static fn ($a) => $a['dtype'], $c['args']))) => [$c];
        }
    }

    #[DataProvider('ufunc')]
    public function testUfunc(array $c): void
    {
        $args = array_map(static fn (array $a) => self::arr($a), $c['args']);
        $r = Math::{$c['f']}(...$args);
        $this->assertSame($c['expect']['dtype'], $r->dtype()->name(), $c['f'] . ': dtype');
        $this->same($c['expect'], $r, $c['f'], (float) ($c['tol'] ?? 1e-13));
        $out = NDArray::zeros($r->shape(), $r->dtype());
        $this->assertSame($out, Math::{$c['f']}(...[...$args, $out]));
        $this->assertSame($r->toBytes(), $out->toBytes(), $c['f'] . ': out=');

        if (extension_loaded('tessero')) {
            $ext = \Tessero\Ext\Math::{$c['f']}(...array_map(static fn (NDArray $a) => \Tessero\Ext\NDArray::fromBytes($a->toBytes(), $a->dtype()->name(), $a->shape()), $args));
            $this->assertSame($ext->toBytes(), $r->toBytes(), $c['f'] . ': FFI and extension bytes');
        }
    }

    public function testSameUfuncsAsTheExtension(): void
    {
        $u = Math::ufuncs();
        $this->assertCount(78, $u);
        $this->assertSame(['nin' => 1, 'engine' => 'loop', 'summary' => 'cube root'], $u['cbrt']);
        $this->assertSame('kernel', $u['sin']['engine']);
        if (extension_loaded('tessero')) {
            $this->assertSame(\Tessero\Ext\Math::ufuncs(), $u);
        }
        foreach (array_keys($u) as $name) {
            $this->assertTrue(method_exists(Math::class, $name), $name);
        }
    }

    public function testKernelUfuncsAreTheMethods(): void
    {
        $a = NDArray::array([[-2.5, -0.5, 0.0], [0.25, 1.0, 3.75]]);
        $b = NDArray::array([1.5, -2.0, 0.5]);
        foreach (['negative' => 'neg', 'absolute' => 'abs', 'sin' => 'sin', 'floor' => 'floor', 'isnan' => 'isnan', 'tanh' => 'tanh'] as $f => $m) {
            $this->assertSame($a->{$m}()->toBytes(), Math::{$f}($a)->toBytes(), $f);
        }
        foreach (['add' => 'add', 'divide' => 'div', 'power' => 'pow', 'arctan2' => 'atan2', 'less' => 'lt', 'floorDivide' => 'floorDiv'] as $f => $m) {
            $r = Math::{$f}($a, $b);
            $this->assertSame($a->{$m}($b)->toBytes(), $r->toBytes(), $f);
            $this->assertSame($a->{$m}($b)->dtype(), $r->dtype(), $f);
        }
        $this->assertSame([3.0, 4.0], Math::add(1, NDArray::array([2.0, 3.0]))->toList(), 'scalar on the left');
        $this->assertSame(DType::Float32, Math::multiply(2.0, NDArray::array([1.0], DType::Float32))->dtype(), 'weak scalar');
    }

    public function testDtypeRules(): void
    {
        $i = NDArray::array([1, -8, 27], DType::Int32);
        $this->assertSame(DType::Float64, Math::cbrt($i)->dtype());
        // NumPy's value in the reference environment (ADR 0012): the C library's cbrt, 1 ulp above 3 for 27
        $this->assertSame([1.0, -2.0, 3.0000000000000004], Math::cbrt($i)->toList());
        $this->assertSame(DType::Int32, Math::trunc($i)->dtype(), 'identity on integers');
        $this->assertNotSame($i->buffer(), Math::trunc($i)->buffer(), 'identity copies');
        $this->assertSame(DType::Int32, Math::fmax($i, 2)->dtype(), 'fmax on integers is maximum');
        $this->assertSame([2, 2, 27], Math::fmax($i, 2)->toList());
        $this->assertSame([1, -2, 0], Math::fmod(NDArray::array([7, -8, 5]), NDArray::array([3, 3, 0]))->toList(), 'fmod sign of x, x % 0 = 0');
        $this->assertSame(DType::Float32, Math::erf(NDArray::array([0.5], DType::Float32))->dtype());
        $this->assertSame([true, false], Math::signbit(NDArray::array([-0.0, 0.0]))->toList());
        $this->assertSame([2.0], Math::apply('cbrt', [8.0])->toList(), 'PHP array operand');
    }

    public function testErrors(): void
    {
        $this->expectExceptionOnce(DTypeError::class, static fn () => Math::cbrt(NDArray::complex([1.0], [1.0])));
        $this->expectExceptionOnce(ShapeError::class, static fn () => Math::fmax(NDArray::zeros([2]), NDArray::zeros([3])));
        $this->expectExceptionOnce(\ValueError::class, static fn () => Math::apply('nope', [1.0]));
        $this->expectExceptionOnce(\ArgumentCountError::class, static fn () => Math::apply('fmod', [1.0]));
        $this->expectExceptionOnce(DTypeError::class, static fn () => Math::cbrt([8.0], NDArray::zeros([1], DType::Int64)), 'float -> int out');
        $this->expectExceptionOnce(DTypeError::class, static fn () => Math::sqrt([4.0], NDArray::zeros([1], DType::Int64)), 'kernel ufunc out cast');
        $this->expectExceptionOnce(ShapeError::class, static fn () => Math::cbrt(NDArray::zeros([3]), NDArray::zeros([2])));
        $ro = NDArray::zeros([2]);
        $ro->buffer()->freeze();
        $this->assertFalse($ro->isWritable());
        $this->expectExceptionOnce(TesseroException::class, static fn () => Math::cbrt([1.0, 8.0], $ro));
        $this->expectExceptionOnce(TesseroException::class, static fn () => $ro->add(1.0, $ro));
        $this->expectExceptionOnce(TesseroException::class, static fn () => $ro->assign(1.0));
        $this->assertSame([0.0, 0.0], $ro->toList());
    }

    public function testOutputs(): void
    {
        // same-kind downcast and broadcasting into out
        $f32 = NDArray::zeros([2, 3], DType::Float32);
        Math::cbrt(NDArray::array([8.0, 27.0, 64.0]), $f32);
        $this->assertSame([[2.0, 3.0, 4.0], [2.0, 3.0, 4.0]], $f32->toArray());
        // in place
        $a = NDArray::array([1.0, 8.0, 27.0]);
        $this->assertSame($a, Math::cbrt($a, $a));
        $this->assertSame([1.0, 2.0, 3.0000000000000004], $a->toList(), 'numpy.cbrt([1., 8., 27.]) (ADR 0012)');
        // apply('sin', $x, $out)
        $o = NDArray::zeros([1]);
        $this->assertSame($o, Math::apply('exp2', [3.0], $o));
        $this->assertSame([8.0], $o->toList());
    }

    /** A shifted view of the input as out must not read already-overwritten values (both kernels and loops). */
    public function testOverlappingOutputGoesThroughATemporary(): void
    {
        foreach (['add', 'fmax'] as $f) {
            $x = NDArray::arange(6.0);
            $expect = Math::{$f}($x['0:5'], $x['1:6'])->toList();
            Math::{$f}($x['0:5'], $x['1:6'], $x['1:6']);
            $this->assertSame([0.0, ...$expect], $x->toList(), $f);
        }
        $x = NDArray::arange(6.0);
        $x['1:6']->neg($x['1:6']);                                      // identical layout: in place is fine
        $x2 = NDArray::arange(6.0);
        $x2['0:5']->neg($x2['1:6']);
        $this->assertSame([0.0, -0.0, -1.0, -2.0, -3.0, -4.0], $x2->toList(), 'unary shifted');
        $y = NDArray::arange(4.0);
        $y['0:3']->add($y['1:4'], $y['1:4']);
        $this->assertSame([0.0, 1.0, 3.0, 5.0], $y->toList(), 'binary shifted');
    }

    public function testThreadCountDoesNotChangeResults(): void
    {
        $x = NDArray::arange(420000.0)->div(1000.0)->sub(210.0)->reshape([600, 700]);
        $y = NDArray::linspace(-3.0, 3.0, 700);
        foreach ([fn () => Math::cbrt($x), fn () => Math::logaddexp($x, $y), fn () => Math::erf($x['::-1, ::2']), fn () => Math::fmod($x, 0.7)] as $i => $f) {
            Tessero::setThreads(1);
            $one = $f()->toBytes();
            Tessero::setThreads(4);
            $this->assertSame($one, $f()->toBytes(), "case {$i}");
        }
    }

    private function expectExceptionOnce(string $class, callable $f, string $what = ''): void
    {
        try {
            $f();
        } catch (\Throwable $e) {
            $this->assertInstanceOf($class, $e, $what . ': ' . $e->getMessage());

            return;
        }
        $this->fail("expected {$class} {$what}");
    }

    private static function arr(array $enc): NDArray
    {
        $data = array_map(static fn ($v) => match ($v) { 'nan' => NAN, 'inf' => INF, '-inf' => -INF, default => is_string($v) ? (float) $v : $v }, $enc['data']);
        $bytes = match ($enc['dtype']) {
            'float64' => pack('d*', ...$data),
            'float32' => pack('g*', ...$data),
            'int64' => pack('q*', ...$data),
            'int32' => pack('l*', ...$data),
            'bool', 'uint8' => pack('C*', ...array_map('intval', $data)),
        };

        return NDArray::fromBytes($bytes, $enc['dtype'], $enc['shape']);
    }

    private function same(array $enc, NDArray $act, string $what, float $tol): void
    {
        $this->assertSame($enc['shape'], $act->shape(), "{$what}: shape");
        $got = $act->ravel()->toList();
        foreach ($enc['data'] as $i => $v) {
            $e = match ($v) { 'nan' => NAN, 'inf' => INF, '-inf' => -INF, default => is_string($v) ? (float) $v : $v };
            $g = $got[$i];
            if (is_float($e) && is_nan($e)) {
                $this->assertTrue(is_nan($g), "{$what}[{$i}] NaN");
            } elseif (! is_float($e) || is_infinite($e)) {
                $this->assertTrue($e == $g, "{$what}[{$i}] expected " . var_export($e, true) . ' got ' . var_export($g, true));
            } else {
                $this->assertTrue(abs($e - $g) <= $tol * max(1.0, abs($e)), sprintf('%s[%d] expected %.17g got %.17g', $what, $i, $e, $g));
            }
        }
    }
}
