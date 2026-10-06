<?php

declare(strict_types=1);

namespace Tessero\Tests\Unit;

use PHPUnit\Framework\TestCase;
use Tessero\Exceptions\MemoryError;
use Tessero\Exceptions\ShapeError;
use Tessero\Exceptions\TesseroException;
use Tessero\NDArray;
use Tessero\Random\Generator;
use Tessero\Tessero;

/**
 * Memory-safety guarantees of the FFI package (docs/architecture/memory-model.md):
 * untrusted shapes can never describe more memory than was allocated, read-only
 * memory is never written from C, released memory is never handed to C, and
 * every allocation is returned. Each test pins a defect found in the audit or a
 * guarantee the operations manual promises.
 */
final class MemorySafetyTest extends TestCase
{
    public function testShapesThatOverflowAreRefused(): void
    {
        $this->throws(ShapeError::class, static fn () => NDArray::fromBytes(str_repeat("\0", 64), 'float64', [-1, -8]));
        $this->throws(MemoryError::class, static fn () => NDArray::fromBytes('', 'float64', [1 << 61, 8]));
        $this->throws(MemoryError::class, static fn () => NDArray::zeros([1 << 40, 1 << 40]));
        $this->throws(MemoryError::class, static fn () => NDArray::empty([1 << 31, 1 << 31, 1 << 31]));
        $this->throws(\Throwable::class, static fn () => NDArray::zeros([0])->reshape(1 << 62, 4));
        $this->throws(\Throwable::class, static fn () => NDArray::zeros([1])->broadcastTo([1 << 40, 1 << 40]));
        $this->throws(ShapeError::class, static fn () => NDArray::zeros(array_fill(0, 33, 1)));
    }

    public function testUnserializeValidatesItsInput(): void
    {
        // serialized arrays come back from caches and queues; a crafted payload must not map memory
        $crafted = static fn (array $shape, string $data, string $dtype = 'float64'): string => 'O:15:"Tessero\NDArray":3:{s:5:"dtype";s:' . strlen($dtype) . ':"' . $dtype
            . '";s:5:"shape";a:' . count($shape) . ':{' . implode('', array_map(static fn ($k, $v) => "i:{$k};i:{$v};", array_keys($shape), $shape)) . '}s:4:"data";s:' . strlen($data) . ':"' . $data . '";}';
        $ok = unserialize($crafted([2], str_repeat("\0", 16)));
        $this->assertSame([0.0, 0.0], $ok->toList(), 'a well-formed payload');
        foreach ([[[-1, -2], 16], [[1 << 61, 8], 0], [[4], 8]] as [$shape, $len]) {
            $this->throws(\Throwable::class, static fn () => unserialize($crafted($shape, str_repeat("\0", $len))), json_encode($shape));
        }
    }

    public function testReadOnlyMemoryIsNeverWrittenFromC(): void
    {
        $f = sys_get_temp_dir() . '/tsr_ro_' . getmypid() . '.f64';
        NDArray::memmap($f, 'w+', [16])->assign(1.0);
        $ro = NDArray::memmap($f, 'r');
        // each of these would write through a PROT_READ mapping, which crashes the process instead of throwing
        $this->throws(TesseroException::class, static fn () => Generator::defaultRng(1)->shuffle($ro));
        $this->throws(TesseroException::class, static fn () => $ro->put([0], [5.0]));
        $this->throws(TesseroException::class, static fn () => $ro->setWhere($ro->gt(0), 0.0));
        $this->throws(TesseroException::class, static fn () => $ro['0:2'] = 3.0);
        $this->throws(TesseroException::class, static fn () => $ro->exp($ro));
        $this->assertSame(16.0, $ro->sum());
        unset($ro);
        unlink($f);
    }

    public function testReleasedBuffersAreNeverUsed(): void
    {
        $f = sys_get_temp_dir() . '/tsr_rel_' . getmypid() . '.f64';
        $m = NDArray::memmap($f, 'w+', [4]);
        $b = $m->buffer();
        unset($m);
        $b->__destruct();                         // what PHP does at shutdown, before other destructors may run
        $b->flush(true);                          // no-op, not a use-after-free of the mapping handle
        $this->throws(TesseroException::class, static fn () => $b->ptr());
        $this->throws(TesseroException::class, static fn () => $b->read(0, 8));
        unset($b);
        unlink($f);
    }

    public function testEveryAllocationIsReturned(): void
    {
        gc_collect_cycles();
        $base = Tessero::memoryInUse();
        $mapped = Tessero::info()['memory_mapped'];
        for ($i = 0; $i < 50; $i++) {
            $a = NDArray::arange(1000.0)->reshape([10, 100]);
            $v = $a->transpose()['::2'];
            $r = \Tessero\Math::logaddexp($v, 1.0)->sum([0, 1]);
            $c = clone $a;
            $s = unserialize(serialize($a));
            try {
                $a->add(NDArray::zeros([3]));
            } catch (ShapeError) {
            }
            unset($a, $v, $r, $c, $s);
        }
        gc_collect_cycles();
        $this->assertSame($base, Tessero::memoryInUse(), 'native bytes back at baseline');
        $this->assertSame($mapped, Tessero::info()['memory_mapped'], 'no mapping left behind');
    }

    public function testBudgetTurnsRunawayAllocationIntoAnException(): void
    {
        $prev = Tessero::memoryBudget();
        Tessero::setMemoryBudget(1 << 20);
        try {
            $this->throws(MemoryError::class, static fn () => NDArray::zeros([1 << 18]));   // 2 MiB > 1 MiB
            // a kernel that runs out of budget mid-way must throw, never return a half-done result
            $big = NDArray::linspace(1.0, 0.0, 100_000);                                 // 800 KB, fits
            $this->throws(MemoryError::class, static fn () => $big->sort());             // keys + scratch do not
            $this->throws(MemoryError::class, static fn () => $big->argsort());
            unset($big);
            $small = NDArray::zeros([1000]);
            $this->assertSame(1000, $small->size(), 'small allocations still succeed');
        } finally {
            Tessero::setMemoryBudget($prev);
        }
    }

    private function throws(string $class, callable $f, string $what = ''): void
    {
        try {
            $f();
        } catch (\Throwable $e) {
            $this->assertInstanceOf($class, $e, $what . ': ' . get_class($e) . ' ' . $e->getMessage());

            return;
        }
        $this->fail("expected {$class} {$what}");
    }
}
