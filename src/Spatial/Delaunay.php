<?php

declare(strict_types=1);

namespace Tessero\Spatial;

/**
 * scipy.spatial.Delaunay: a 2-D Delaunay triangulation of a point set, built with the incremental
 * Bowyer-Watson algorithm. Pure PHP (no kernel), identical on both backends.
 *
 * The TRIANGLE SET reproduces scipy/Qhull exactly for points in general position (no four points
 * cocircular, no three collinear): the Delaunay triangulation is then unique, so the set of
 * triangles -- each stored as a sorted tuple of input-point indices -- is well defined and matches
 * regardless of construction order. The per-simplex vertex ORDER and the order of simplices in the
 * list are NOT asserted against scipy (Qhull's are Qhull-internal); only the set is a stable
 * invariant.
 *
 * ->simplices    the triangulation as a list of [i, j, k] index triples (input-point indices).
 * ->findSimplex  for each query point, the index of the containing simplex (into ->simplices), or
 *                -1 when the point lies outside the convex hull -- a brute-force barycentric sign scan.
 * ->transform    scipy's barycentric transform stack: for simplex s, transform[s][0..ndim-1] is the
 *                inverse of the (vertex - last_vertex) matrix and transform[s][ndim] is the last
 *                vertex r, so barycentric c[:ndim] = Tinv . (x - r), c[ndim] = 1 - sum(c[:ndim]).
 *
 * General-position inputs only (matching the de-risked scope); degenerate/cocircular configurations
 * are out of scope and may disagree with Qhull's symbolic-perturbation tie-breaking.
 */
final class Delaunay
{
    /** @var list<list<float>> the input points (each [x, y]) */
    public array $points;

    /** @var int number of input points */
    public int $npoints;

    /** @var int spatial dimension (fixed at 2) */
    public int $ndim = 2;

    /** @var list<array{0:int,1:int,2:int}> triangles as input-point index triples */
    public array $simplices;

    /** @var list<list<list<float>>> scipy-style barycentric transform, shape (nsimplex, ndim+1, ndim) */
    public array $transform;

    /**
     * @param list<list<float>> $points a list of [x, y] coordinates (>= 3 points, general position)
     */
    public function __construct(array $points)
    {
        $pts = [];
        foreach (array_values($points) as $p) {
            $p = array_values($p);
            if (count($p) !== 2) {
                throw new \InvalidArgumentException('Delaunay here supports 2-D points only ([x, y])');
            }
            $pts[] = [(float) $p[0], (float) $p[1]];
        }
        $n = count($pts);
        if ($n < 3) {
            throw new \InvalidArgumentException('Delaunay requires at least 3 points');
        }
        $this->points = $pts;
        $this->npoints = $n;

        $this->simplices = $this->bowyerWatson($pts);
        $this->transform = $this->buildTransform();
    }

    /**
     * Incremental Bowyer-Watson over a robust super-triangle placed a huge offset (~1e6) about the
     * centroid, so every input point lies strictly inside it. The in-circle test is an
     * orientation-corrected determinant (float64), matching the de-risked prototype.
     *
     * @param  list<list<float>> $pts
     * @return list<array{0:int,1:int,2:int}>
     */
    private function bowyerWatson(array $pts): array
    {
        $n = count($pts);

        // centroid
        $mx = 0.0;
        $my = 0.0;
        foreach ($pts as $p) {
            $mx += $p[0];
            $my += $p[1];
        }
        $mx /= $n;
        $my /= $n;

        $huge = 1.0e6;
        // augmented point array: inputs then the three super-triangle vertices
        $ap = $pts;
        $ap[$n] = [$mx - $huge, $my - $huge];
        $ap[$n + 1] = [$mx + $huge, $my - $huge];
        $ap[$n + 2] = [$mx, $my + $huge];

        // triangles stored as "i,j,k" keys -> [i, j, k]; start with the super-triangle
        $tris = [];
        $tris[$this->triKey($n, $n + 1, $n + 2)] = [$n, $n + 1, $n + 2];

        for ($ip = 0; $ip < $n; $ip++) {
            $px = $pts[$ip][0];
            $py = $pts[$ip][1];

            // triangles whose circumcircle contains the new point are "bad"
            $bad = [];
            foreach ($tris as $k => $t) {
                if ($this->inCircle($ap[$t[0]], $ap[$t[1]], $ap[$t[2]], $px, $py) > 0.0) {
                    $bad[$k] = $t;
                }
            }

            // edges on the boundary of the bad-triangle cavity appear exactly once
            $edgeCount = [];
            $edge = [];
            foreach ($bad as $t) {
                foreach ([[$t[0], $t[1]], [$t[1], $t[2]], [$t[2], $t[0]]] as $e) {
                    $a = $e[0];
                    $b = $e[1];
                    $ek = $a < $b ? ($a . ',' . $b) : ($b . ',' . $a);
                    $edgeCount[$ek] = ($edgeCount[$ek] ?? 0) + 1;
                    $edge[$ek] = [$a, $b];
                }
            }

            // remove bad triangles
            foreach ($bad as $k => $_) {
                unset($tris[$k]);
            }

            // re-triangulate the cavity against the new point
            foreach ($edgeCount as $ek => $c) {
                if ($c === 1) {
                    [$a, $b] = $edge[$ek];
                    $tris[$this->triKey($a, $b, $ip)] = [$a, $b, $ip];
                }
            }
        }

        // drop any triangle still touching the super-triangle vertices
        $result = [];
        foreach ($tris as $t) {
            if ($t[0] < $n && $t[1] < $n && $t[2] < $n) {
                $result[] = [$t[0], $t[1], $t[2]];
            }
        }

        return $result;
    }

    /** Canonical key for a triangle (order-independent) so the same triangle is never stored twice. */
    private function triKey(int $a, int $b, int $c): string
    {
        if ($a > $b) {
            [$a, $b] = [$b, $a];
        }
        if ($b > $c) {
            [$b, $c] = [$c, $b];
        }
        if ($a > $b) {
            [$a, $b] = [$b, $a];
        }

        return $a . ',' . $b . ',' . $c;
    }

    /**
     * Orientation-corrected in-circle determinant: > 0 iff p is strictly inside the circumcircle of
     * triangle (a, b, c), independent of the triangle's winding.
     *
     * @param list<float> $a
     * @param list<float> $b
     * @param list<float> $c
     */
    private function inCircle(array $a, array $b, array $c, float $px, float $py): float
    {
        $area2 = ($b[0] - $a[0]) * ($c[1] - $a[1]) - ($b[1] - $a[1]) * ($c[0] - $a[0]);

        $ax = $a[0] - $px;
        $ay = $a[1] - $py;
        $bx = $b[0] - $px;
        $by = $b[1] - $py;
        $cx = $c[0] - $px;
        $cy = $c[1] - $py;

        $a2 = $ax * $ax + $ay * $ay;
        $b2 = $bx * $bx + $by * $by;
        $c2 = $cx * $cx + $cy * $cy;

        // det of [[ax, ay, a2], [bx, by, b2], [cx, cy, c2]]
        $det = $ax * ($by * $c2 - $b2 * $cy)
            - $ay * ($bx * $c2 - $b2 * $cx)
            + $a2 * ($bx * $cy - $by * $cx);

        return $area2 > 0.0 ? $det : -$det;
    }

    /**
     * scipy-style barycentric transform stack: one (ndim+1, ndim) block per simplex. The first ndim
     * rows are the inverse of the matrix whose columns are (vertex_i - vertex_last); the last row is
     * the last vertex r.
     *
     * @return list<list<list<float>>>
     */
    private function buildTransform(): array
    {
        $out = [];
        foreach ($this->simplices as $t) {
            $v0 = $this->points[$t[0]];
            $v1 = $this->points[$t[1]];
            $v2 = $this->points[$t[2]];

            // T = [[v0x - v2x, v1x - v2x], [v0y - v2y, v1y - v2y]]
            $m00 = $v0[0] - $v2[0];
            $m01 = $v1[0] - $v2[0];
            $m10 = $v0[1] - $v2[1];
            $m11 = $v1[1] - $v2[1];

            $det = $m00 * $m11 - $m01 * $m10;
            if ($det == 0.0) {
                // degenerate (collinear) simplex: emit NaNs like scipy does for flat simplices
                $out[] = [[NAN, NAN], [NAN, NAN], [$v2[0], $v2[1]]];
                continue;
            }
            $inv = 1.0 / $det;

            $out[] = [
                [$m11 * $inv, -$m01 * $inv],
                [-$m10 * $inv, $m00 * $inv],
                [$v2[0], $v2[1]],
            ];
        }

        return $out;
    }

    /**
     * Barycentric coordinates of a point with respect to a given simplex (length ndim+1, summing to 1).
     *
     * @param  list<float> $x [px, py]
     * @return list<float>
     */
    public function barycentric(int $simplex, array $x): array
    {
        $T = $this->transform[$simplex];
        $r = $T[2];
        $dx0 = (float) $x[0] - $r[0];
        $dx1 = (float) $x[1] - $r[1];
        $c0 = $T[0][0] * $dx0 + $T[0][1] * $dx1;
        $c1 = $T[1][0] * $dx0 + $T[1][1] * $dx1;

        return [$c0, $c1, 1.0 - $c0 - $c1];
    }

    /**
     * For each query point, the index of the containing simplex (into ->simplices), or -1 when the
     * point is outside the convex hull. Brute-force barycentric sign scan; a point is inside simplex
     * s iff all barycentric coordinates are >= -tol.
     *
     * @param  list<list<float>> $queryPoints a list of [x, y]
     * @return list<int>
     */
    public function findSimplex(array $queryPoints, float $tol = 1.0e-12): array
    {
        $out = [];
        foreach (array_values($queryPoints) as $q) {
            $q = array_values($q);
            $x = (float) $q[0];
            $y = (float) $q[1];
            $found = -1;
            foreach ($this->transform as $s => $T) {
                $r = $T[2];
                $dx0 = $x - $r[0];
                $dx1 = $y - $r[1];
                $c0 = $T[0][0] * $dx0 + $T[0][1] * $dx1;
                $c1 = $T[1][0] * $dx0 + $T[1][1] * $dx1;
                $c2 = 1.0 - $c0 - $c1;
                if ($c0 >= -$tol && $c1 >= -$tol && $c2 >= -$tol) {
                    $found = $s;
                    break;
                }
            }
            $out[] = $found;
        }

        return $out;
    }
    /**
     * scipy.spatial.tsearch(tri, xi): the simplex containing each query point, or -1 outside
     * the hull. Identical to (new Delaunay($tri->points))->findSimplex($xi); provided as a
     * free-function-style static so callers can mirror scipy's module-level tsearch.
     *
     * @param Delaunay|list<list<float>> $tri a Delaunay triangulation (or its input points)
     * @param list<list<float>> $xi query points
     * @return list<int>
     */
    public static function tsearch(Delaunay|array $tri, array $xi): array
    {
        $d = $tri instanceof Delaunay ? $tri : new Delaunay($tri);
        return $d->findSimplex($xi);
    }
}
