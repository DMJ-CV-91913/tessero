<?php

declare(strict_types=1);

namespace Tessero\Tests\Unit;

use PHPUnit\Framework\TestCase;
use Tessero\NDArray;
use Tessero\Np;
use Tessero\Tessero;

/**
 * Memory ownership (ADR 0015): the layer that allocates memory frees it. Every kernel block a PHP object holds is
 * released by that object, exactly once, whatever order the objects go in, on exceptions too, so a long-running
 * worker (php-fpm, Octane) ends every request where it started.
 */
final class OwnershipTest extends TestCase
{
    public function testViewsKeepTheirBaseAndFreeNothing(): void
    {
        $base = Tessero::memoryInUse();
        $parent = NDArray::array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        $row = $parent['1'];
        $col = $parent[':, 2'];
        $inner = $row['1:'];                                   // a view of a view
        unset($parent);                                        // the parent goes first: the views keep the block
        gc_collect_cycles();
        $this->assertSame([5.0, 6.0], $inner->toList());
        $this->assertSame([3.0, 6.0], $col->toList());
        $this->assertGreaterThan($base, Tessero::memoryInUse(), 'the block is alive while a view is');
        unset($inner, $row);                                   // views in either order
        $this->assertSame([3.0, 6.0], $col->toList());
        unset($col);
        gc_collect_cycles();
        $this->assertSame($base, Tessero::memoryInUse(), 'the last view released the block, once');
    }

    public function testRoutineResultsAreReleasedOnEveryPath(): void
    {
        $base = Tessero::memoryInUse();
        $parts = Np::split(NDArray::array([1.0, 2.0, 3.0, 4.0]), 2);          // a kind 6 sequence of arrays
        $this->assertCount(2, $parts);
        $this->assertSame('float64', (string) Np::resultType(NDArray::array([1.0]), 'float32'));   // a kind 2 string
        unset($parts);
        try {
            Np::split(NDArray::array([1.0, 2.0, 3.0]), 2);                     // not an equal division: throws
            $this->fail('split into unequal parts must throw');
        } catch (\Throwable) {
        }
        gc_collect_cycles();
        $this->assertSame($base, Tessero::memoryInUse());
    }

    /** An FPM-style worker: many requests in one process; the native heap returns to its baseline every time. */
    public function testManyRequestsReturnToBaseline(): void
    {
        $base = Tessero::memoryInUse();
        for ($request = 0; $request < 10000; $request++) {
            $a = NDArray::arange(16)->reshape([4, 4]);
            $v = $a['1:3, ::2'];
            $s = $a->add($v->sum());
            $parts = Np::split($s, 2);
            if ($request % 7 === 0) {
                unset($a);                                     // drop the parent before its view
            }
            try {
                Np::split($v, 3);                              // a failing routine on every request
            } catch (\Throwable) {
            }
            unset($a, $v, $s, $parts);
            if ($request % 1000 === 999) {
                gc_collect_cycles();
                $this->assertSame($base, Tessero::memoryInUse(), "native heap back to baseline after request {$request}");
            }
        }
    }
}
