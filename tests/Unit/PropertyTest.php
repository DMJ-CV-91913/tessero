<?php

declare(strict_types=1);

namespace Tessero\Tests\Unit;

use PHPUnit\Framework\TestCase;
use Tessero\DType;
use Tessero\Fft\Fft;
use Tessero\Math;
use Tessero\NDArray;

/**
 * Property-based tests: algebraic identities and round trips that must hold
 * for every shape, dtype and memory layout, checked on seeded random inputs
 * (random shapes up to 4-D, random dtypes, transposed / reversed / strided
 * views). A failure message names the case; rerun with the same seed.
 *
 * TESSERO_PROPERTY_CASES scales the number of cases (default 40 per property).
 */
final class PropertyTest extends TestCase
{
    private const SEED = 424242;

    private int $cases;

    protected function setUp(): void
    {
        mt_srand(self::SEED);
        $this->cases = max(1, (int) (getenv('TESSERO_PROPERTY_CASES') ?: 40));
    }

    public function testBytesSerializeJsonAndNpyRoundTrips(): void
    {
        $dir = sys_get_temp_dir() . '/tsr_prop_' . getmypid();
        @mkdir($dir);
        for ($i = 0; $i < $this->cases; $i++) {
            [$a, $label] = $this->randomArray();
            $bytes = $a->toBytes();
            $this->assertSame($bytes, NDArray::fromBytes($bytes, $a->dtype(), $a->shape())->toBytes(), "{$label} bytes");
            $this->assertSame($bytes, unserialize(serialize($a))->toBytes(), "{$label} serialize");
            $a->save("{$dir}/p.npy");
            $this->assertSame($bytes, NDArray::load("{$dir}/p.npy")->toBytes(), "{$label} npy");
            if ($a->size() > 0) {
                $this->assertSame($bytes, NDArray::load("{$dir}/p.npy", 'r')->toBytes(), "{$label} npy memmap");
            }
            if ($a->dtype() !== DType::Complex128) {
                // JSON is exact for finite values: 17 significant digits for float64, 9 for float32
                $back = NDArray::array(json_decode($a->toJson(), true), $a->dtype());
                $this->assertSame($bytes, $back->reshape($a->shape())->toBytes(), "{$label} json");
            }
            $this->assertSame($a->toArray(), $a->copy()->toArray(), "{$label} copy");
        }
        foreach (glob("{$dir}/*") ?: [] as $f) {
            unlink($f);
        }
        rmdir($dir);
    }

    public function testLayoutIdentities(): void
    {
        for ($i = 0; $i < $this->cases; $i++) {
            [$a, $label] = $this->randomArray();
            $this->assertSame($a->toBytes(), $a->transpose()->transpose()->toBytes(), "{$label} T.T");
            $this->assertSame($a->toBytes(), $a->reshape([-1])->reshape($a->shape())->toBytes(), "{$label} reshape");
            $this->assertSame($a->toBytes(), $a->contiguous()->toBytes(), "{$label} contiguous");
            if ($a->ndim() > 0 && $a->shape()[0] > 0) {
                $this->assertSame($a->toBytes(), $a->slice('::-1')->slice('::-1')->toBytes(), "{$label} reverse twice");
            }
            $this->assertSame($a->toBytes(), $a->expandDims(0)->squeeze(0)->toBytes(), "{$label} expand/squeeze");
        }
    }

    public function testArithmeticIdentities(): void
    {
        for ($i = 0; $i < $this->cases; $i++) {
            [$a, $label] = $this->randomArray(['float64', 'float32', 'int64', 'int32']);
            [$b] = $this->randomArray([$a->dtype()->name()], $a->shape());
            $int = $a->dtype()->isInteger();
            // exact for every dtype
            $this->assertSame($a->toBytes(), $a->add(0)->toBytes(), "{$label} a+0");
            $this->assertSame($a->toBytes(), $a->mul(1)->toBytes(), "{$label} a*1");
            $this->assertSame($a->add($b)->toBytes(), $b->add($a)->toBytes(), "{$label} a+b = b+a");
            $this->assertSame($a->mul($b)->toBytes(), $b->mul($a)->toBytes(), "{$label} a*b = b*a");
            $this->assertSame($a->maximum($b)->toBytes(), $b->maximum($a)->toBytes(), "{$label} max commutes");
            $this->assertSame($a->neg()->neg()->toBytes(), $a->toBytes(), "{$label} --a");
            if ($int) {
                // two's-complement wrap-around makes these exact for integers
                $this->assertSame($a->toBytes(), $a->add($b)->sub($b)->toBytes(), "{$label} (a+b)-b");
                $this->assertSame($a->sum(), self::scalar($a->sum(array_keys($a->shape()))), "{$label} sum(all axes) = sum()");
                if ($a->size() > 0) {
                    $c = $a->ravel()->cumsum();
                    $this->assertSame($a->sum(), $c->item($c->size() - 1), "{$label} cumsum last = sum");
                }
            } else {
                // one rounding of a+b: the error is bounded by half an ulp of the largest magnitude involved
                $eps = $a->dtype() === DType::Float32 ? 1.2e-7 : 2.3e-16;
                $big = $a->size() ? max((float) $a->abs()->max(), (float) $b->abs()->max()) : 0.0;
                $this->assertTrue(NDArray::allclose($a->add($b)->sub($b), $a, 0.0, 4 * $eps * $big), "{$label} (a+b)-b ~ a");
                $this->assertEqualsWithDelta($a->sum(), self::scalar($a->sum(array_keys($a->shape()))), 1e-3 * max(1.0, abs((float) $a->sum())), "{$label} sum axes");
            }
            $this->assertSame(
                NDArray::where($a->gt($b), $a, $b)->toBytes(),
                $a->maximum($b)->toBytes(),
                "{$label} where(a>b, a, b) = maximum (no NaN here)",
            );
        }
    }

    public function testUfuncIdentities(): void
    {
        for ($i = 0; $i < $this->cases; $i++) {
            [$a, $label] = $this->randomArray(['float64', 'float32']);
            [$b] = $this->randomArray([$a->dtype()->name()], $a->shape());
            $this->assertSame(Math::fmax($a, $b)->toBytes(), Math::fmax($b, $a)->toBytes(), "{$label} fmax commutes");
            $this->assertSame($a->toBytes(), Math::copysign(Math::absolute($a), $a)->toBytes(), "{$label} copysign(|a|, a) = a");
            $this->assertSame(Math::signbit($a)->toBytes(), $a->lt(0)->logicalOr($a->eq(0)->logicalAnd(Math::copysign(1.0, $a)->lt(0)))->toBytes(), "{$label} signbit");
            $this->assertSame(Math::trunc($a)->toBytes(), Math::trunc(Math::trunc($a))->toBytes(), "{$label} trunc idempotent");
            $tol = $a->dtype() === DType::Float32 ? 2e-6 : 1e-14;
            $this->assertTrue(NDArray::allclose(Math::cbrt($a)->pow(3), $a, $tol * 10, $tol), "{$label} cbrt^3");
            $this->assertTrue(NDArray::allclose(Math::logaddexp($a, $a), $a->add(M_LN2), $tol, $tol), "{$label} logaddexp(a, a)");
            $this->assertTrue(NDArray::allclose(Math::erf($a)->add(Math::erfc($a)), NDArray::ones($a->shape()), $tol * 10, $tol * 10), "{$label} erf + erfc = 1");
        }
    }

    public function testSortAndFftProperties(): void
    {
        for ($i = 0; $i < $this->cases; $i++) {
            [$a, $label] = $this->randomArray(['float64', 'int64', 'int32', 'float32'], [mt_rand(0, 300)]);
            $s = $a->sort();
            $this->assertSame($s->toBytes(), $s->sort()->toBytes(), "{$label} sort idempotent");
            $list = $s->toList();
            for ($k = 1; $k < count($list); $k++) {
                $this->assertLessThanOrEqual($list[$k], $list[$k - 1], "{$label} sorted");
            }
            $idx = $a->argsort();
            $this->assertSame($s->toBytes(), $a->take($idx)->toBytes(), "{$label} take(argsort) = sort");
            $ids = $idx->toList();
            sort($ids);
            $this->assertSame(range(0, $a->size() - 1), $a->size() ? $ids : [], "{$label} argsort is a permutation");
            if ($a->size() > 0 && $a->dtype()->isFloat()) {
                $x = $a->astype(DType::Float64);
                $back = Fft::ifft(Fft::fft($x))->real();
                $this->assertTrue(NDArray::allclose($back, $x, 1e-9, 1e-9), "{$label} ifft(fft(x)) = x");
                $r = Fft::irfft(Fft::rfft($x), $x->size());
                $this->assertTrue(NDArray::allclose($r, $x, 1e-9, 1e-9), "{$label} irfft(rfft(x)) = x");
            }
        }
    }

    // ------------------------------------------------------------------ generator

    private static function scalar(mixed $v): mixed
    {
        return $v instanceof NDArray ? $v->item() : $v;
    }

    /**
     * @param list<string>|null $dtypes
     * @param list<int>|null $shape
     * @return array{NDArray, string}
     */
    private function randomArray(?array $dtypes = null, ?array $shape = null): array
    {
        $dtypes ??= ['float64', 'float32', 'int64', 'int32', 'uint8', 'bool', 'complex128'];
        $dtype = $dtypes[mt_rand(0, count($dtypes) - 1)];
        if ($shape === null) {
            $nd = mt_rand(0, 4);
            $shape = [];
            for ($d = 0; $d < $nd; $d++) {
                $shape[] = mt_rand(0, 9) === 0 ? 0 : mt_rand(1, 6);
            }
        }
        $size = (int) array_product($shape);
        $vals = [];
        for ($k = 0; $k < $size; $k++) {
            $vals[] = match ($dtype) {
                'float64' => (mt_rand() / mt_getrandmax() - 0.5) * 10 ** mt_rand(-3, 6),
                'float32' => (float) unpack('g', pack('g', (mt_rand() / mt_getrandmax() - 0.5) * 10 ** mt_rand(-2, 4)))[1],
                'int64' => mt_rand(PHP_INT_MIN >> 2, PHP_INT_MAX >> 2),
                'int32' => mt_rand(-2 ** 30, 2 ** 30),
                'uint8' => mt_rand(0, 255),
                'bool' => mt_rand(0, 1) === 1,
                'complex128' => 0.0,
            };
        }
        if ($dtype === 'complex128') {
            $re = NDArray::array($size ? array_map(static fn () => mt_rand(-1000, 1000) / 7, $vals) : [], 'float64')->reshape($shape);
            $a = NDArray::complex($re, $re->mul(-0.5));
        } else {
            $a = NDArray::array($vals, $dtype)->reshape($shape);
        }
        $label = "{$dtype} " . json_encode($shape);
        $layout = mt_rand(0, 3);
        if ($layout === 1 && count($shape) >= 2) {
            $a = $a->transpose()->copy()->transpose();              // Fortran-ordered data, same values
            $label .= ' F-order';
        } elseif ($layout === 2 && count($shape) >= 1 && $shape[0] > 0) {
            $a = NDArray::stack([$a, $a], 1)->slice(':', 0);         // strided view, same values
            $label .= ' strided';
        } elseif ($layout === 3 && count($shape) >= 1) {
            $a = $a->slice('::-1')->copy()->slice('::-1');          // reversed view, same values
            $label .= ' reversed';
        }

        return [$a, $label];
    }
}
