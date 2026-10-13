<?php

declare(strict_types=1);

namespace Tessero\Spatial;

/**
 * scipy.spatial.Voronoi (2-D): the Voronoi diagram as the dual of the Delaunay triangulation.
 * Pure PHP (no kernel), identical on both backends.
 *
 * STABLE INVARIANTS reproduced exactly for points in general position (the only case where the
 * diagram is well defined; cocircular/collinear configurations are out of scope, matching the
 * Delaunay de-risk):
 *   ->vertices      the finite Voronoi vertices = circumcenters of the Delaunay triangles. Qhull's
 *                   per-vertex ORDER is internal, so fixtures compare them as a SET (rounded tuples).
 *   ->ridgePoints   every finite OR infinite ridge separates two input sites; the set of unordered
 *                   input-point index pairs equals the set of Delaunay edges. Compared as a SET.
 *   ->pointRegion / ->regions  one region per input point (ordering internal); not asserted beyond
 *                   count == npoints.
 *
 * The per-vertex/-ridge ORDER and the signs of the infinite-ridge directions are Qhull-internal and
 * are NOT asserted; only the vertex SET and the ridge-point SET (both order-free) are invariants.
 */
final class Voronoi
{
    /** @var list<array{0:float,1:float}> input points */
    public array $points;

    /** @var int */
    public int $npoints;

    /** @var int fixed at 2 */
    public int $ndim = 2;

    /** @var list<array{0:float,1:float}> finite Voronoi vertices (Delaunay circumcenters) */
    public array $vertices;

    /** @var list<array{0:int,1:int}> input-point index pairs, one per ridge (unordered) */
    public array $ridgePoints;

    /** @param list<list<float>> $points >= 3 general-position [x, y] points */
    public function __construct(array $points)
    {
        $pts = [];
        foreach (array_values($points) as $p) {
            $p = array_values($p);
            if (count($p) !== 2) {
                throw new \InvalidArgumentException('Voronoi here supports 2-D points only ([x, y])');
            }
            $pts[] = [(float) $p[0], (float) $p[1]];
        }
        if (count($pts) < 3) {
            throw new \InvalidArgumentException('Voronoi requires at least 3 points');
        }
        $this->points = $pts;
        $this->npoints = count($pts);

        $tri = new Delaunay($pts);

        // Voronoi vertices = circumcenters of the Delaunay triangles.
        $this->vertices = [];
        foreach ($tri->simplices as $s) {
            $this->vertices[] = self::circumcenter($pts[$s[0]], $pts[$s[1]], $pts[$s[2]]);
        }

        // ridge_points = Delaunay edges (each triangle edge; shared edges deduped).
        $seen = [];
        $ridges = [];
        foreach ($tri->simplices as $s) {
            foreach ([[$s[0], $s[1]], [$s[1], $s[2]], [$s[2], $s[0]]] as $e) {
                $a = $e[0]; $b = $e[1];
                if ($a > $b) { [$a, $b] = [$b, $a]; }
                $key = $a . ',' . $b;
                if (!isset($seen[$key])) {
                    $seen[$key] = true;
                    $ridges[] = [$a, $b];
                }
            }
        }
        $this->ridgePoints = $ridges;
    }

    /** Circumcenter of triangle (a, b, c). */
    private static function circumcenter(array $a, array $b, array $c): array
    {
        $ax = $a[0]; $ay = $a[1];
        $bx = $b[0]; $by = $b[1];
        $cx = $c[0]; $cy = $c[1];
        $d = 2.0 * ($ax * ($by - $cy) + $bx * ($cy - $ay) + $cx * ($ay - $by));
        $a2 = $ax * $ax + $ay * $ay;
        $b2 = $bx * $bx + $by * $by;
        $c2 = $cx * $cx + $cy * $cy;
        $ux = ($a2 * ($by - $cy) + $b2 * ($cy - $ay) + $c2 * ($ay - $by)) / $d;
        $uy = ($a2 * ($cx - $bx) + $b2 * ($ax - $cx) + $c2 * ($bx - $ax)) / $d;
        return [$ux, $uy];
    }
}
