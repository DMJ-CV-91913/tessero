<?php

declare(strict_types=1);

namespace Tessero\Spatial;

/**
 * scipy.spatial.ConvexHull for 2-D point sets, computed with Andrew's monotone chain. The hull vertices are
 * produced in counter-clockwise order (as scipy's ->vertices are for 2-D), the starting index is arbitrary in
 * both this implementation and in Qhull, so callers compare the vertex SET. ->area is the hull perimeter and
 * ->volume is the enclosed polygon area (shoelace) -- matching scipy's convention that in d dimensions
 * ->volume is the d-measure and ->area the (d-1)-measure, so in 2-D area==perimeter and volume==polygon area.
 *
 * 2-D ONLY: a point set whose columns are not exactly 2 is rejected, as is a degenerate (collinear / fewer
 * than three distinct) set, which Qhull reports as a QhullError. Pure PHP (no kernel), identical on both
 * backends.
 */
final class ConvexHull
{
    /**
     * Hull vertex indices into the input points, in counter-clockwise order (start index arbitrary).
     *
     * @var list<int>
     */
    public array $vertices;

    /** Perimeter of the hull (scipy ConvexHull.area in 2-D). */
    public float $area;

    /** Enclosed polygon area of the hull (scipy ConvexHull.volume in 2-D). */
    public float $volume;

    /**
     * @param list<list<float>> $points an (n, 2) array of 2-D points
     */
    public function __construct(array $points)
    {
        $pts = [];
        foreach (array_values($points) as $p) {
            $p = array_values($p);
            if (count($p) !== 2) {
                throw new \InvalidArgumentException('Tessero\\Spatial\\ConvexHull supports 2-D points only');
            }
            $pts[] = [(float) $p[0], (float) $p[1]];
        }
        $n = count($pts);
        if ($n < 3) {
            throw new \InvalidArgumentException('ConvexHull requires at least 3 points');
        }

        // Sort point indices lexicographically by (x, y).
        $idx = range(0, $n - 1);
        usort($idx, static function (int $a, int $b) use ($pts): int {
            if ($pts[$a][0] !== $pts[$b][0]) {
                return $pts[$a][0] <=> $pts[$b][0];
            }

            return $pts[$a][1] <=> $pts[$b][1];
        });

        $cross = static function (int $o, int $a, int $b) use ($pts): float {
            return ($pts[$a][0] - $pts[$o][0]) * ($pts[$b][1] - $pts[$o][1])
                 - ($pts[$a][1] - $pts[$o][1]) * ($pts[$b][0] - $pts[$o][0]);
        };

        // Lower hull.
        $lower = [];
        foreach ($idx as $i) {
            while (count($lower) >= 2 && $cross($lower[count($lower) - 2], $lower[count($lower) - 1], $i) <= 0.0) {
                array_pop($lower);
            }
            $lower[] = $i;
        }

        // Upper hull.
        $upper = [];
        foreach (array_reverse($idx) as $i) {
            while (count($upper) >= 2 && $cross($upper[count($upper) - 2], $upper[count($upper) - 1], $i) <= 0.0) {
                array_pop($upper);
            }
            $upper[] = $i;
        }

        // Concatenate, dropping each list's last point (shared with the other list's start). CCW order.
        array_pop($lower);
        array_pop($upper);
        $hull = array_merge($lower, $upper);

        if (count($hull) < 3) {
            throw new \InvalidArgumentException('input is degenerate (collinear points); QhullError');
        }

        $this->vertices = $hull;

        $k = count($hull);
        $area2 = 0.0;   // twice the signed polygon area
        $perim = 0.0;
        for ($i = 0; $i < $k; $i++) {
            [$x1, $y1] = $pts[$hull[$i]];
            [$x2, $y2] = $pts[$hull[($i + 1) % $k]];
            $area2 += $x1 * $y2 - $x2 * $y1;
            $dx = $x2 - $x1;
            $dy = $y2 - $y1;
            $perim += sqrt($dx * $dx + $dy * $dy);
        }
        $this->volume = abs($area2) / 2.0;
        $this->area = $perim;
    }
}
