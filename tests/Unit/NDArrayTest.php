<?php

declare(strict_types=1);

namespace Tessero\Tests\Unit;

use PHPUnit\Framework\TestCase;
use Tessero\DType;
use Tessero\Exceptions\DTypeError;
use Tessero\Exceptions\IndexError;
use Tessero\Exceptions\MemoryError;
use Tessero\Exceptions\ShapeError;
use Tessero\NDArray;
use Tessero\Native\Abi;
use Tessero\Native\Library;
use Tessero\Tessero;

use function Tessero\arange;
use function Tessero\arr;

final class NDArrayTest extends TestCase
{
    protected function tearDown(): void
    {
        Tessero::setMemoryBudget(0);
    }

    public function testCreationAndInference(): void
    {
        $this->assertSame(DType::Int64, arr([1, 2, 3])->dtype());
        $this->assertSame(DType::Float64, arr([1, 2.5])->dtype());
        $this->assertSame(DType::Bool, arr([true, false])->dtype());
        $this->assertSame([2, 3], arr([[1, 2, 3], [4, 5, 6]])->shape());
        $this->assertSame([], arr(7)->shape());
        $this->assertSame(7, arr(7)->toArray());
        $this->assertSame([[0.0, 0.0]], NDArray::zeros([1, 2])->toArray());
        $this->assertSame([3.0, 3.0], NDArray::full(2, 3.0)->toArray());
        $this->assertSame([0], arr([], 'int64')->shape());
        $this->assertSame([[1.0, 0.0], [0.0, 1.0]], NDArray::eye(2)->toArray());
        $this->assertSame([0.0, 0.5, 1.0], NDArray::linspace(0, 1, 3)->toArray());
    }

    public function testRaggedInputIsRejected(): void
    {
        $this->expectException(ShapeError::class);
        arr([[1, 2], [3]]);
    }

    public function testNonNumericInputIsRejected(): void
    {
        $this->expectException(DTypeError::class);
        arr(['a', 'b']);
    }

    public function testViewsShareMemory(): void
    {
        $a = arange(12)->reshape(3, 4);
        $row = $a[1];
        $col = $a[':, 2'];
        $rev = $a['::-1'];
        $a['1, 2'] = 100;
        $this->assertSame([4, 5, 100, 7], $row->toArray());
        $this->assertSame([2, 100, 10], $col->toArray());
        $this->assertSame(100, $rev->item(1, 2));
        $this->assertSame($a->buffer(), $rev->buffer());
        $this->assertFalse($col->isContiguous());
        $this->assertTrue($row->isContiguous());
        $this->assertTrue($a->t()->isFortranContiguous());
    }

    public function testViewKeepsBufferAliveAfterParentIsGone(): void
    {
        $view = (static fn (): NDArray => arange(1000.0)->slice('990:'))();
        gc_collect_cycles();
        $this->assertSame(990.0, $view->item(0));
    }

    public function testNativeMemoryIsReleased(): void
    {
        gc_collect_cycles();
        $before = Tessero::memoryInUse();
        for ($i = 0; $i < 50; $i++) {
            $a = NDArray::ones([1000, 100]);
            $b = $a->mul(2)->add($a->t()->t())->sum(0);
            unset($a, $b);
        }
        gc_collect_cycles();
        $this->assertSame($before, Tessero::memoryInUse());
        $this->assertGreaterThan(800_000, Tessero::peakMemory());
    }

    public function testMemoryBudgetThrowsInsteadOfGrowing(): void
    {
        Tessero::setMemoryBudget(Tessero::memoryInUse() + 1024 * 1024);
        $small = NDArray::zeros(1000);
        $this->assertSame(1000, $small->size());
        $this->expectException(MemoryError::class);
        NDArray::zeros(1_000_000);
    }

    public function testBroadcastingErrorsAreShapeErrors(): void
    {
        $this->expectException(ShapeError::class);
        arange(6)->reshape(2, 3)->add(arange(2));
    }

    public function testPromotionRules(): void
    {
        $i = arr([1, 2], 'int32');
        $this->assertSame(DType::Int32, $i->add(1)->dtype(), 'weak int scalar keeps int32');
        $this->assertSame(DType::Float64, $i->add(1.5)->dtype());
        $this->assertSame(DType::Float64, $i->div(2)->dtype(), 'true division of ints is float64');
        $this->assertSame(DType::Int64, $i->add(arr([1, 2], 'int64'))->dtype());
        $this->assertSame(DType::Float32, arr([1.0], 'float32')->add(2.0)->dtype());
        $this->assertSame(DType::Float64, arr([1.0], 'float32')->add(arr([1], 'int32'))->dtype());
        $this->assertSame(DType::Float32, arr([1.0], 'float32')->add(arr([1], 'uint8'))->dtype());
        $this->assertSame(DType::Bool, $i->gt(1)->dtype());
        $this->assertSame(DType::Int64, arr([true, true])->add(arr([true, false]))->dtype());
        $this->assertSame([8, 27], arr([2, 3])->pow(3)->toArray());
        $this->assertSame(DType::Int64, arr([2, 3])->pow(3)->dtype());
    }

    public function testIndexingForms(): void
    {
        $a = arange(24)->reshape(2, 3, 4);
        $this->assertSame(23, $a['-1, -1, -1']);
        $this->assertSame([2, 1, 3, 4], $a[':, None, ...']->shape());
        $this->assertSame([0, 4, 8], $a['0, :, 0']->toArray());
        $this->assertSame([[3, 2, 1, 0]], $a['0, 0:1, ::-1']->toArray());
        $this->assertSame([12, 0], $a[[1, 0]]->slice(':', 0, 0)->toArray());
        $this->assertSame([21, 22, 23], $a[$a->gt(20)]->toArray());
        $this->assertSame([0, 3], $a->slice(0, 0, '::3')->toArray());
        $this->assertSame([], $a['0, 0, 5:']->toArray());
        $a[$a->lt(2)] = -1;
        $this->assertSame([-1, -1, 2], $a['0, 0, :3']->toArray());
        $a[[1]] = 0;
        $this->assertSame(0, $a->slice(1)->sum());
    }

    public function testIndexOutOfBounds(): void
    {
        $this->expectException(IndexError::class);
        arange(3)[3];
    }

    public function testTakeOutOfBoundsIsCaughtInC(): void
    {
        $this->expectException(IndexError::class);
        arange(3)->take([0, 7]);
    }

    public function testReshapeAndAxes(): void
    {
        $a = arange(24)->reshape(2, -1, 4);
        $this->assertSame([2, 3, 4], $a->shape());
        $this->assertSame([4, 3, 2], $a->t()->shape());
        $this->assertSame([3, 2, 4], $a->swapAxes(0, 1)->shape());
        $this->assertSame([3, 4, 2], $a->moveAxis(0, -1)->shape());
        $this->assertSame([2, 3, 4, 1], $a->expandDims(-1)->shape());
        $this->assertSame([2, 3, 4], $a->expandDims(1)->squeeze(1)->shape());
        $this->assertSame([5, 2, 3, 4], $a->broadcastTo([5, 2, 3, 4])->shape());
        $t = $a->t()->reshape(-1);
        $this->assertSame([0, 12, 4, 16], $t->slice('0:4')->toArray(), 'reshape of a non-contiguous view copies in C order');
    }

    public function testInPlaceOutParameter(): void
    {
        $a = NDArray::ones([3]);
        $b = $a->add(1, out: $a);
        $this->assertSame($a, $b);
        $this->assertSame([2.0, 2.0, 2.0], $a->toArray());
        $a->sqrt(out: $a);
        $this->assertEqualsWithDelta(sqrt(2), $a->item(0), 1e-15);
    }

    public function testOverlappingAssignmentUsesATemporary(): void
    {
        $a = arange(6);
        $a['1:'] = $a[':-1'];
        $this->assertSame([0, 0, 1, 2, 3, 4], $a->toArray());
    }

    public function testComplexArithmetic(): void
    {
        $z = NDArray::complex([1.0, 0.0], [1.0, 2.0]);
        $w = $z->mul($z);
        $this->assertSame([[0.0, 2.0], [-4.0, 0.0]], $w->toArray());
        $this->assertSame([1.0, 0.0], $z->real()->toArray());
        $this->assertSame([[1.0, 3.0]], [$z->sum()]);
        $this->expectException(DTypeError::class);
        $z->max();
    }

    public function testReductionsKeepdimsAndEmpty(): void
    {
        $a = arange(6.0)->reshape(2, 3);
        $this->assertSame([1, 3], $a->sum(0, true)->shape());
        $this->assertSame([[15.0]], $a->sum(null, true)->toArray());
        $this->assertSame(0.0, NDArray::zeros([0])->sum());
        $this->assertTrue(is_nan(NDArray::zeros([0])->mean()));
        $this->assertSame([true, true, true], $a->ge(0)->all(0)->toArray());
        $this->assertSame(1, arr([3, 9, 1])->argmax());
        $this->expectException(ShapeError::class);
        NDArray::zeros([0])->max();
    }

    public function testStringFormatting(): void
    {
        $this->assertSame('array([1., 2.5, 100.], dtype=float64)', (string) arr([1.0, 2.5, 100.0]));
        $this->assertStringContainsString('...', (string) arange(5000));
        $this->assertSame('[[1,2],[3,4]]', json_encode(arr([[1, 2], [3, 4]])));
    }

    public function testIteration(): void
    {
        $rows = [];
        foreach (arange(6)->reshape(3, 2) as $i => $row) {
            $rows[$i] = $row->toArray();
        }
        $this->assertSame([[0, 1], [2, 3], [4, 5]], $rows);
        $this->assertCount(3, arange(6)->reshape(3, 2));
    }

    public function testMatmulPaths(): void
    {
        $a = arange(6.0)->reshape(2, 3);
        $this->assertSame([[5.0, 14.0], [14.0, 50.0]], $a->matmul($a->t())->toArray());
        $this->assertSame(5.0, arr([1.0, 2.0])->dot(arr([1.0, 2.0])));
        $this->assertSame([5, 14], arange(6)->reshape(2, 3)->matmul(arr([0, 1, 2]))->toArray(), 'int64 via the C kernel');
        $this->assertSame([[0.0, 0.0], [0.0, 0.0]], NDArray::zeros([2, 0])->matmul(NDArray::zeros([0, 2]))->toArray());
        $this->expectException(ShapeError::class);
        $a->matmul($a);
    }

    public function testAbiConstantsMatchTheCHeader(): void
    {
        $src = (string) file_get_contents(dirname(__DIR__, 2) . '/csrc/src/ops.h');
        preg_match_all('/\b(OP|U|R)_([A-Z0-9]+)\b(?:\s*=\s*(\d+))?/', $src, $m, PREG_SET_ORDER);
        $counters = ['OP' => 0, 'U' => 0, 'R' => 0];
        foreach ($m as $hit) {
            [$all, $group, $name] = $hit;
            $value = isset($hit[3]) && $hit[3] !== '' ? (int) $hit[3] : $counters[$group];
            $counters[$group] = $value + 1;
            $const = match ($group) {
                'OP' => $name,
                'U' => $name,
                'R' => 'R_' . $name,
            };
            if ($group === 'OP' && in_array($name, ['MAX', 'MIN'], true)) {
                $this->assertSame($value, constant(Abi::class . '::' . $name));
                continue;
            }
            $this->assertSame($value, constant(Abi::class . '::' . $const), "{$group}_{$name}");
        }
    }

    public function testFfiPointerWorkarounds(): void
    {
        // void* -> uintptr_t is 0 on PHP 8.4; Library::address routes through char*
        $ffi = Library::ffi();
        $raw = $ffi->tsr_alloc(64);
        $this->assertNotSame(0, Library::address($raw));
        $ffi->tsr_free($raw, 64);
    }

    /** Fuzzer finding (csrc/fuzz): a blank item made the slice parser read past its comma; blanks are now bounded. */
    public function testSliceSpecBlanks(): void
    {
        $a = NDArray::arange(12.0)->reshape([3, 4]);
        $this->assertSame(6.0, $a->sliceNative('1 , 2')->item());
        $this->assertSame([[4.0, 6.0], [8.0, 10.0]], $a->sliceNative(' 1:3 , ::2 ')->toArray());
        // fuzzer finding: huge steps overflowed the count and stride arithmetic
        $this->assertSame([[0.0, 1.0, 2.0, 3.0]], $a->slice('::4611686018427387904')->toArray());
        $this->assertSame([[8.0]], $a->slice('::-4611686018427387904', '::4611686018427387904')->toArray());
        $this->assertSame([[0.0, 1.0, 2.0, 3.0]], $a->sliceNative('::4611686018427387904')->toArray());
        foreach (['0, ,1', '0,   ', '  '] as $bad) {
            try {
                $a->sliceNative($bad);
                $this->fail("accepted '{$bad}'");
            } catch (\Tessero\Exceptions\TesseroException) {
                $this->assertTrue(true);
            }
        }
    }
}
