<?php

declare(strict_types=1);

namespace Tessero\Spatial;

/**
 * scipy.spatial.HalfspaceIntersection: the vertices of the polytope defined by a set of halfspaces
 * {x : A x + b <= 0}, computed via the convex-hull DUAL (polar duality about a feasible interior
 * point) -- no Qhull FFI. Pure PHP, identical on both backends.
 *
 * Each halfspace a_i . x + b_i <= 0 maps to a dual point  p_i = a_i / (-b_i - a_i . x0)  where x0 is a
 * strictly-interior feasible point. The convex hull of the dual points is the polar dual of the
 * polytope: every FACET of the dual hull corresponds to a VERTEX of the polytope, recovered as
 * x0 - n / c  where (n, c) is the facet's hyperplane  n . y + c = 0  with n oriented outward (away from
 * the dual-hull centroid). The halfspaces that appear as VERTICES of the dual hull are exactly the
 * halfspaces that are facets of the intersection polytope -> dual_vertices.
 *
 * STABLE INVARIANTS reproduced for a bounded polytope in general position (the well-defined case,
 * matching the ConvexHull de-risk):
 *   ->intersections  the polytope vertices (order-free SET, compared to ~1e-9). scipy's per-vertex
 *                     order is Qhull-internal.
 *   ->dual_vertices  the indices of the halfspaces that form facets of the polytope (order-free SET).
 *
 * The per-vertex halfspace incidence lists and the dual-point order are Qhull-internal and are NOT
 * asserted; only the two sets above are the invariants.
 */
final class HalfspaceIntersection
{
    /** @var list<list<float>> input halfspaces, each row [a_1, ..., a_d, b] meaning a.x + b <= 0 */
    public array $halfspaces;

    /** @var list<float> the feasible interior point used for the duality */
    public array $interiorPoint;

    /** @var int ambient dimension d */
    public int $ndim;

    /**
     * Polytope vertices (duals of the dual-hull facets). Order is Qhull-internal; callers compare as a SET.
     *
     * @var list<list<float>>
     */
    public array $intersections;

    /**
     * Sorted indices of the input halfspaces that are facets of the intersection polytope (= vertices of
     * the dual hull).
     *
     * @var list<int>
     */
    public array $dualVertices;

    /**
     * @param list<list<float>> $halfspaces each row [a_1..a_d, b] for a.x + b <= 0; >= d+1 of them, bounded
     * @param list<float>       $interiorPoint a strictly-feasible interior point (a.x0 + b < 0 for all rows)
     */
    public function __construct(array $halfspaces, array $interiorPoint)
    {
        $hs = [];
        $cols = null;
        foreach (array_values($halfspaces) as $row) {
            $row = array_values($row);
            if ($cols === null) {
                $cols = count($row);
            }
            if (count($row) !== $cols) {
                throw new \InvalidArgumentException('HalfspaceIntersection: all halfspace rows must have the same length');
            }
            $hs[] = array_map('floatval', $row);
        }
        if ($cols === null || $cols < 3) {
            throw new \InvalidArgumentException('HalfspaceIntersection requires halfspaces with at least 3 columns ([a_1, a_2, b] for d=2)');
        }
        $d = $cols - 1;
        $x0 = array_values($interiorPoint);
        if (count($x0) !== $d) {
            throw new \InvalidArgumentException('HalfspaceIntersection: interior_point dimension must be d (halfspace columns - 1)');
        }
        $x0 = array_map('floatval', $x0);
        $n = count($hs);
        if ($n < $d + 1) {
            throw new \InvalidArgumentException('HalfspaceIntersection requires at least d+1 halfspaces');
        }

        $this->halfspaces = $hs;
        $this->interiorPoint = $x0;
        $this->ndim = $d;

        // Dual point for each halfspace: p_i = a_i / (-b_i - a_i . x0). The denominator is the (positive)
        // slack of x0 inside halfspace i; a non-positive slack means x0 is not strictly interior.
        $dual = [];
        foreach ($hs as $row) {
            $b = $row[$d];
            $slack = 0.0;
            for ($k = 0; $k < $d; $k++) {
                $slack += $row[$k] * $x0[$k];
            }
            $denom = -($b + $slack);
            if ($denom <= 0.0) {
                throw new \InvalidArgumentException('HalfspaceIntersection: interior_point is not strictly interior to every halfspace');
            }
            $p = [];
            for ($k = 0; $k < $d; $k++) {
                $p[] = $row[$k] / $denom;
            }
            $dual[] = $p;
        }

        $hull = new ConvexHull($dual);
        // ConvexHull exposes facet simplices only for d >= 3; in 2-D the hull is the CCW vertex polygon,
        // whose consecutive vertex pairs are the edge-facets.
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
            throw new \InvalidArgumentException('HalfspaceIntersection: dual hull produced no facets (unbounded or degenerate)');
        }

        // dual-hull centroid (mean of hull vertices) -> orient facet normals outward
        $hv = $hull->vertices;
        $hc = array_fill(0, $d, 0.0);
        foreach ($hv as $vi) {
            for ($k = 0; $k < $d; $k++) {
                $hc[$k] += $dual[$vi][$k];
            }
        }
        $m = count($hv);
        for ($k = 0; $k < $d; $k++) {
            $hc[$k] /= $m;
        }

        $inter = [];
        foreach ($facets as $simplex) {
            $plane = $this->facetPlane($dual, $simplex, $hc, $d);
            if ($plane === null) {
                continue;
            }
            [$nrm, $off] = $plane; // n . y + off = 0, n outward
            if ($off === 0.0) {
                continue; // facet through the duality origin -> vertex at infinity (unbounded)
            }
            // primal vertex = x0 - n / off
            $v = [];
            for ($k = 0; $k < $d; $k++) {
                $v[] = $x0[$k] - $nrm[$k] / $off;
            }
            $inter[] = $v;
        }

        $this->intersections = $inter;
        $dv = $hull->vertices;
        sort($dv);
        $this->dualVertices = $dv;
    }

    /**
     * Facet hyperplane n . y + off = 0 (unit normal) with n oriented OUTWARD -- away from `centroid`,
     * i.e. n . centroid + off < 0. Returns null for a degenerate facet.
     *
     * @param list<list<float>> $pts
     * @param list<int>         $combo d point indices
     * @param list<float>       $centroid dual-hull centroid
     * @return array{0: list<float>, 1: float}|null
     */
    private function facetPlane(array $pts, array $combo, array $centroid, int $d): ?array
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
        $off = 0.0;
        for ($k = 0; $k < $d; $k++) {
            $off -= $nrm[$k] * $base[$k];
        }
        // orient outward: centroid on the negative side
        $sc = $off;
        for ($k = 0; $k < $d; $k++) {
            $sc += $nrm[$k] * $centroid[$k];
        }
        if ($sc > 0.0) {
            for ($k = 0; $k < $d; $k++) {
                $nrm[$k] = -$nrm[$k];
            }
            $off = -$off;
        }

        return [$nrm, $off];
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
