<?php

declare(strict_types=1);

namespace Tessero\Spatial;

/**
 * An axis-aligned hyperrectangle, mirroring scipy.spatial.Rectangle: the Minkowski-p minimum and maximum
 * distances from the box to a point or to another rectangle, its volume, and splitting along an axis.
 * Pure PHP (no kernel), identical on both backends.
 */
final class Rectangle
{
    /** @var list<float> */
    public array $maxes;

    /** @var list<float> */
    public array $mins;

    public int $m;

    /**
     * @param list<float> $maxes upper corner
     * @param list<float> $mins  lower corner
     */
    public function __construct(array $maxes, array $mins)
    {
        $this->maxes = array_map('floatval', array_values($maxes));
        $this->mins = array_map('floatval', array_values($mins));
        $this->m = count($this->maxes);
    }

    public function volume(): float
    {
        $v = 1.0;
        for ($i = 0; $i < $this->m; $i++) {
            $v *= $this->maxes[$i] - $this->mins[$i];
        }

        return $v;
    }

    /** @param list<float> $x */
    public function minDistancePoint(array $x, float $p = 2.0): float
    {
        $z = [];
        for ($i = 0; $i < $this->m; $i++) {
            $z[$i] = max(0.0, max($this->mins[$i] - $x[$i], $x[$i] - $this->maxes[$i]));
        }

        return self::minkowski($z, $p);
    }

    /** @param list<float> $x */
    public function maxDistancePoint(array $x, float $p = 2.0): float
    {
        $z = [];
        for ($i = 0; $i < $this->m; $i++) {
            $z[$i] = max($this->maxes[$i] - $x[$i], $x[$i] - $this->mins[$i]);
        }

        return self::minkowski($z, $p);
    }

    public function minDistanceRectangle(self $other, float $p = 2.0): float
    {
        $z = [];
        for ($i = 0; $i < $this->m; $i++) {
            $z[$i] = max(0.0, max($this->mins[$i] - $other->maxes[$i], $other->mins[$i] - $this->maxes[$i]));
        }

        return self::minkowski($z, $p);
    }

    public function maxDistanceRectangle(self $other, float $p = 2.0): float
    {
        $z = [];
        for ($i = 0; $i < $this->m; $i++) {
            $z[$i] = max($this->maxes[$i] - $other->mins[$i], $other->maxes[$i] - $this->mins[$i]);
        }

        return self::minkowski($z, $p);
    }

    /**
     * Split along axis $d at coordinate $split, returning [less, greater].
     *
     * @return array{0: self, 1: self}
     */
    public function split(int $d, float $split): array
    {
        $midMax = $this->maxes;
        $midMax[$d] = $split;
        $midMin = $this->mins;
        $midMin[$d] = $split;

        return [new self($midMax, $this->mins), new self($this->maxes, $midMin)];
    }

    /** @param list<float> $z */
    private static function minkowski(array $z, float $p): float
    {
        if (is_infinite($p)) {
            $m = 0.0;
            foreach ($z as $v) {
                $m = max($m, abs($v));
            }

            return $m;
        }
        $s = 0.0;
        foreach ($z as $v) {
            $s += abs($v) ** $p;
        }

        return $s ** (1.0 / $p);
    }
}
