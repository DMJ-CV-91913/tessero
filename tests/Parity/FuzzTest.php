<?php

declare(strict_types=1);

namespace Tessero\Tests\Parity;

use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\TestCase;
use Tessero\DType;
use Tessero\NDArray;

/**
 * 300 random (shape, slice) pairs from NumPy: the view's contents, element-wise
 * arithmetic on the strided view against a broadcast operand, reductions along
 * a random axis and a transposed cast. Exercises negative/zero strides,
 * dimension coalescing and the broadcasting rules end to end.
 */
final class FuzzTest extends TestCase
{
    public static function cases(): iterable
    {
        $cases = json_decode((string) file_get_contents(__DIR__ . '/../fixtures/fuzz.json'), true, 512, JSON_THROW_ON_ERROR);
        foreach ($cases as $i => $c) {
            yield sprintf('%03d (%s)[%s]', $i, implode(',', $c['shape']), $c['spec']) => [$c];
        }
    }

    #[DataProvider('cases')]
    public function testSliceAndStridedKernels(array $c): void
    {
        $base = NDArray::arange((int) array_product($c['shape']))->reshape($c['shape']);
        $view = $base[$c['spec']];
        $view = $view instanceof NDArray ? $view : NDArray::array($view);
        $this->same($c['expect'], $view, 'view');
        // the C slice parser (used by the native extension) must agree with the PHP one
        $cview = $base->sliceNative($c['spec']);
        $this->same($c['expect'], $cview->shape() === [] ? NDArray::array($cview->item()) : $cview, 'C-parsed view');
        $this->assertSame(json_encode($view->toArray(), JSON_PRESERVE_ZERO_FRACTION), $view->toJson(), 'toJson matches json_encode');
        if (! isset($c['binary'])) {
            return;
        }
        $other = NDArray::arange((int) array_product($c['bshape']))->mul(0.5)->sub(3)->reshape($c['bshape']);
        $this->same($c['binary'], $view->mul(1.5)->sub($other), 'view * 1.5 - other');
        $this->same($c['sum_axis'], $view->sum($c['axis']), 'sum');
        $this->same($c['max_axis'], $view->max($c['axis']), 'max');
        $this->same($c['transposed_copy'], $view->t()->astype(DType::Float32), 'transposed float32 copy');
    }

    private function same(array $enc, NDArray $actual, string $what): void
    {
        $this->assertSame($enc['shape'], $actual->shape(), "{$what}: shape");
        $expected = array_map(static fn ($v) => is_string($v) ? (float) $v : $v, $enc['data']);
        $got = $actual->toList();
        foreach ($expected as $i => $v) {
            $this->assertTrue(is_float($v) ? abs($v - $got[$i]) <= 1e-12 * max(1.0, abs($v)) : $v === $got[$i], "{$what}: element {$i} expected " . var_export($v, true) . ' got ' . var_export($got[$i], true));
        }
    }
}
