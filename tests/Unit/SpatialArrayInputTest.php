<?php

declare(strict_types=1);

namespace Tessero\Tests\Unit;

use PHPUnit\Framework\TestCase;
use Tessero\NDArray;
use Tessero\Spatial\ConvexHull;
use Tessero\Spatial\KDTree;

/**
 * KDTree and the 2-D ConvexHull take the points as PHP rows (converted in C through the extension, else cast in one
 * PHP loop) or as an NDArray of either backend, used as it is. Every way in holds the same doubles, so every result
 * has the same bits: compared here as serialized bytes, including the fallback loop (rows of numeric strings).
 */
final class SpatialArrayInputTest extends TestCase
{
    /** @return list<list<float>> */
    private static function points(int $n, int $d, int $seed): array
    {
        $s = $seed;
        $out = [];
        for ($i = 0; $i < $n; $i++) {
            $row = [];
            for ($j = 0; $j < $d; $j++) {
                $s = ($s * 1103515245 + 12345) % 2147483648;
                $row[] = $s / 2147483648.0 - 0.5;
            }
            $out[] = $row;
        }

        return $out;
    }

    /** @return array<string, callable(list<list<float>>): mixed> the ways the same points can be passed */
    private static function inputs(): array
    {
        $ways = [
            'rows' => static fn (array $p): array => $p,
            'rows of strings' => static fn (array $p): array => array_map(static fn (array $r): array => array_map(static fn (float $v): string => sprintf('%.17g', $v), $r), $p),
            'FFI array' => static fn (array $p): object => NDArray::array($p, 'float64'),
        ];
        if (class_exists(\Tessero\Ext\NDArray::class, false)) {
            $ways['extension array'] = static fn (array $p): object => \Tessero\Ext\NDArray::array($p, 'float64');
        }

        return $ways;
    }

    public function testKdTreeGivesTheSameBitsForEveryKindOfInput(): void
    {
        $pts = self::points(2000, 3, 7);
        $q = self::points(50, 3, 11);
        $ref = null;
        foreach (self::inputs() as $name => $way) {
            $tree = new KDTree($way($pts));
            $this->assertSame(2000, $tree->count(), $name);
            $got = serialize([$tree->query($way($q), 4), $tree->query($way($q))]);
            $ref ??= $got;
            $this->assertSame($ref, $got, "{$name}: the same distances and indices, bit for bit");
        }
    }

    public function testConvexHullGivesTheSameBitsForEveryKindOfInput(): void
    {
        $pts = self::points(3000, 2, 5);
        $ref = null;
        foreach (self::inputs() as $name => $way) {
            $h = new ConvexHull($way($pts));
            $got = serialize([$h->vertices, $h->area, $h->volume, $h->ndim]);
            $ref ??= $got;
            $this->assertSame($ref, $got, "{$name}: the same vertices, perimeter and area, bit for bit");
        }
    }

    public function testRaggedRowsAreStillRefused(): void
    {
        $this->expectException(\InvalidArgumentException::class);
        new ConvexHull([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0, 2.0]]);
    }
}
