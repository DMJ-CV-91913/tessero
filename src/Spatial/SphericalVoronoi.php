<?php

declare(strict_types=1);

namespace Tessero\Spatial;

/**
 * scipy.spatial.SphericalVoronoi: the Voronoi diagram of points lying on a d-sphere, computed as the
 * dual of the convex hull of the generators (no Qhull FFI). Pure PHP, identical on both backends.
 *
 * Each facet of the convex hull of the generators corresponds to one Voronoi vertex: the point on the
 * sphere equidistant (geodesically) from the facet's generators, i.e. the sphere/line intersection
 * along the facet's OUTWARD normal. The outward normal is oriented away from the hull centroid (the
 * same orientation Qhull uses for its facet equations), which makes the vertex set an exact geometric
 * invariant of scipy's SphericalVoronoi.vertices.
 *
 * STABLE INVARIANTS reproduced for generators in general position on the sphere (the well-defined
 * case, matching the ConvexHull de-risk -- no degenerate/co-spherical-facet inputs):
 *   ->points   the input generators (validated to lie on the sphere of the given radius/center).
 *   ->vertices the Voronoi vertices = hull-facet circumcenters projected to the sphere. Qhull's
 *              per-vertex ORDER is internal, so callers compare them as an order-free SET (to ~1e-9).
 *
 * The per-region vertex lists (->regions) and the sort_vertices_of_regions ordering are Qhull/traversal
 * internal and are NOT asserted; only the vertex SET is the invariant (same policy as Voronoi/ConvexHull).
 */
final class SphericalVoronoi
{
    /** @var list<list<float>> input generator points (on the sphere) */
    public array $points;

    /** @var int number of generators */
    public int $npoints;

    /** @var int ambient dimension (>= 3 for a 2-sphere; general d supported) */
    public int $ndim;

    /** @var float sphere radius */
    public float $radius;

    /** @var list<float> sphere center */
    public array $center;

    /**
     * Voronoi vertices = convex-hull facet circumcenters projected to the sphere. Order is Qhull-internal
     * (callers compare as a SET).
     *
     * @var list<list<float>>
     */
    public array $vertices;

    /**
     * @param list<list<float>> $points generators lying on the sphere, >= d+1 of them in general position
     * @param float             $radius sphere radius (default 1.0)
     * @param list<float>|null  $center sphere center (default the origin in the input dimension)
     * @param float             $threshold tolerance for the "points lie on the sphere" check
     */
    public function __construct(array $points, float $radius = 1.0, ?array $center = null, float $threshold = 1e-6)
    {
        $pts = [];
        $d = null;
        foreach (array_values($points) as $p) {
            $p = array_values($p);
            if ($d === null) {
                $d = count($p);
            }
            if (count($p) !== $d) {
                throw new \InvalidArgumentException('SphericalVoronoi: all points must have the same number of columns');
            }
            $row = [];
            foreach ($p as $c) {
                $row[] = (float) $c;
            }
            $pts[] = $row;
        }
        if ($d === null || $d < 2) {
            throw new \InvalidArgumentException('SphericalVoronoi requires points with at least 2 columns');
        }
        if ($center === null) {
            $center = array_fill(0, $d, 0.0);
        } else {
            $center = array_values($center);
            if (count($center) !== $d) {
                throw new \InvalidArgumentException('SphericalVoronoi: center dimension must match the points');
            }
            $center = array_map('floatval', $center);
        }
        $n = count($pts);
        if ($n < $d + 1) {
            throw new \InvalidArgumentException('SphericalVoronoi requires at least d+1 generators');
        }

        // validate that every generator lies on the sphere
        foreach ($pts as $p) {
            $r2 = 0.0;
            for ($k = 0; $k < $d; $k++) {
                $dk = $p[$k] - $center[$k];
                $r2 += $dk * $dk;
            }
            if (abs(sqrt($r2) - $radius) > $threshold) {
                throw new \InvalidArgumentException('SphericalVoronoi: generators must lie on the sphere of the given radius/center');
            }
        }

        $this->points = $pts;
        $this->npoints = $n;
        $this->ndim = $d;
        $this->radius = $radius;
        $this->center = $center;

        // Hull of the generators: each facet (simplex of d point indices) yields one Voronoi vertex.
        $hull = new ConvexHull($pts);
        // ConvexHull exposes facet simplices only for d >= 3; in 2-D (circle) the hull is the CCW vertex
        // polygon, whose consecutive vertex pairs are the edge-facets.
        if ($d === 2) {
            $facets = [];
            $poly = $hull->vertices;
            $kk = count($poly);
            for ($e = 0; $e < $kk; $e++) {
                $facets[] = [$poly[$e], $poly[($e + 1) % $kk]];
            }
        } else {
            $facets = $hull->simplices;
        }
        if (count($facets) === 0) {
            throw new \InvalidArgumentException('SphericalVoronoi: hull produced no facets (degenerate input)');
        }

        // Hull centroid (mean of the hull vertices) -- orient each facet normal away from it (outward).
        $hv = $hull->vertices;
        $hc = array_fill(0, $d, 0.0);
        foreach ($hv as $vi) {
            for ($k = 0; $k < $d; $k++) {
                $hc[$k] += $pts[$vi][$k];
            }
        }
        $m = count($hv);
        for ($k = 0; $k < $d; $k++) {
            $hc[$k] /= $m;
        }

        $verts = [];
        foreach ($facets as $simplex) {
            $nrm = $this->facetNormal($pts, $simplex, $d);
            if ($nrm === null) {
                continue; // degenerate facet (should not happen in general position)
            }
            // orient away from the hull centroid
            $base = $pts[$simplex[0]];
            $dot = 0.0;
            for ($k = 0; $k < $d; $k++) {
                $dot += ($base[$k] - $hc[$k]) * $nrm[$k];
            }
            if ($dot < 0.0) {
                for ($k = 0; $k < $d; $k++) {
                    $nrm[$k] = -$nrm[$k];
                }
            }
            // Voronoi vertex = center + radius * outward_unit_normal
            $v = [];
            for ($k = 0; $k < $d; $k++) {
                $v[] = $center[$k] + $radius * $nrm[$k];
            }
            $verts[] = $v;
        }

        $this->vertices = $verts;
    }

    /**
     * Unit normal of the hyperplane through a d-point facet, via the generalized cross product
     * (cofactor expansion of the (d-1) tangent vectors). Returns null if the facet is degenerate.
     *
     * @param list<list<float>> $pts
     * @param list<int>         $combo d point indices
     * @return list<float>|null
     */
    private function facetNormal(array $pts, array $combo, int $d): ?array
    {
        $combo = array_values($combo);
        $base = $pts[$combo[0]];
        $T = [];
        for ($i = 1; $i < $d; $i++) {
            $row = [];
            for ($k = 0; $k < $d; $k++) {
                $row[] = $pts[$combo[$i]][$k] - $base[$k];
            }
            $T[] = $row;
        }
        $nrm = array_fill(0, $d, 0.0);
        for ($k = 0; $k < $d; $k++) {
            $minor = [];
            foreach ($T as $trow) {
                $mrow = [];
                for ($c = 0; $c < $d; $c++) {
                    if ($c === $k) {
                        continue;
                    }
                    $mrow[] = $trow[$c];
                }
                $minor[] = $mrow;
            }
            $sign = ($k % 2 === 0) ? 1.0 : -1.0;
            $nrm[$k] = $sign * $this->det($minor, $d - 1);
        }
        $nn = 0.0;
        for ($k = 0; $k < $d; $k++) {
            $nn += $nrm[$k] * $nrm[$k];
        }
        if ($nn <= 0.0) {
            return null;
        }
        $nn = sqrt($nn);
        for ($k = 0; $k < $d; $k++) {
            $nrm[$k] /= $nn;
        }

        return $nrm;
    }

    /** Determinant of an m x m matrix by Gaussian elimination with partial pivoting. */
    private function det(array $A, int $m): float
    {
        if ($m === 0) {
            return 1.0;
        }
        $det = 1.0;
        for ($col = 0; $col < $m; $col++) {
            $piv = $col;
            $best = abs($A[$col][$col]);
            for ($r = $col + 1; $r < $m; $r++) {
                if (abs($A[$r][$col]) > $best) {
                    $best = abs($A[$r][$col]);
                    $piv = $r;
                }
            }
            if ($best === 0.0) {
                return 0.0;
            }
            if ($piv !== $col) {
                $tmp = $A[$col];
                $A[$col] = $A[$piv];
                $A[$piv] = $tmp;
                $det = -$det;
            }
            $det *= $A[$col][$col];
            $inv = 1.0 / $A[$col][$col];
            for ($r = $col + 1; $r < $m; $r++) {
                $factor = $A[$r][$col] * $inv;
                if ($factor !== 0.0) {
                    for ($c = $col; $c < $m; $c++) {
                        $A[$r][$c] -= $factor * $A[$col][$c];
                    }
                }
            }
        }

        return $det;
    }
}
