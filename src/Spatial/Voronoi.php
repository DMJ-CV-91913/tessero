<?php

declare(strict_types=1);

namespace Tessero\Spatial;

/**
 * scipy.spatial.Voronoi: the Voronoi diagram as the dual of the Delaunay triangulation. Pure PHP
 * (no kernel), identical on both backends.
 *
 * STABLE INVARIANTS reproduced exactly for points in general position (the only case where the
 * diagram is well defined; cospherical/coplanar configurations are out of scope, matching the
 * Delaunay de-risk):
 *
 *   ->vertices      the finite Voronoi vertices = circumcenters of the Delaunay simplices. Qhull's
 *                   per-vertex ORDER is internal, so fixtures compare them as a SET (rounded tuples).
 *                   This is matched exactly in 2-D AND N-D (de-risked over 120 3-D/4-D sets).
 *   ->ridgePoints   (2-D ONLY) every finite OR infinite ridge separates two input sites; the set of
 *                   unordered input-point index pairs equals the set of Delaunay edges, compared as a
 *                   SET. In d >= 3 the ridge_points set depends on Qhull's full unbounded-Voronoi
 *                   enumeration (upper-hull / point-at-infinity handling) and is NOT reproducible
 *                   from the lower-hull Delaunay alone, so it is left EMPTY for N-D and not asserted.
 *
 * The per-vertex/-ridge ORDER and the signs of the infinite-ridge directions are Qhull-internal and
 * are NOT asserted; only the vertex SET (all dims) and, in 2-D, the ridge-point SET are invariants.
 */
final class Voronoi
{
    /** @var list<list<float>> input points */
    public array $points;

    /** @var int */
    public int $npoints;

    /** @var int spatial dimension */
    public int $ndim = 2;

    /** @var list<list<float>> finite Voronoi vertices (Delaunay circumcenters) */
    public array $vertices;

    /**
     * @var list<array{0:int,1:int}> input-point index pairs, one per ridge (unordered). Populated in
     *                                2-D only; empty in d >= 3 (see class docblock).
     */
    public array $ridgePoints = [];

    /** @param list<list<float>> $points >= ndim+1 general-position points */
    public function __construct(array $points)
    {
        $pts = [];
        $d = null;
        foreach (array_values($points) as $p) {
            $p = array_values($p);
            if ($d === null) {
                $d = count($p);
            }
            if (count($p) !== $d) {
                throw new \InvalidArgumentException('Voronoi: all points must have the same number of columns');
            }
            $row = [];
            foreach ($p as $c) {
                $row[] = (float) $c;
            }
            $pts[] = $row;
        }
        if ($d === null || $d < 2) {
            throw new \InvalidArgumentException('Voronoi requires points with at least 2 columns');
        }
        if (count($pts) < $d + 1) {
            throw new \InvalidArgumentException('Voronoi requires at least ndim+1 points');
        }
        $this->points = $pts;
        $this->npoints = count($pts);
        $this->ndim = $d;

        if ($d === 2) {
            $this->voronoi2d($pts);

            return;
        }

        $this->ndVoronoi($pts, $d);
    }

    /** 2-D Voronoi: circumcenters of Delaunay triangles + ridge_points = Delaunay edges. */
    private function voronoi2d(array $pts): void
    {
        $tri = new Delaunay($pts);

        $this->vertices = [];
        foreach ($tri->simplices as $s) {
            $this->vertices[] = self::circumcenter2d($pts[$s[0]], $pts[$s[1]], $pts[$s[2]]);
        }

        $seen = [];
        $ridges = [];
        foreach ($tri->simplices as $s) {
            foreach ([[$s[0], $s[1]], [$s[1], $s[2]], [$s[2], $s[0]]] as $e) {
                $a = $e[0];
                $b = $e[1];
                if ($a > $b) {
                    [$a, $b] = [$b, $a];
                }
                $key = $a . ',' . $b;
                if (! isset($seen[$key])) {
                    $seen[$key] = true;
                    $ridges[] = [$a, $b];
                }
            }
        }
        $this->ridgePoints = $ridges;
    }

    /**
     * N-D Voronoi (d >= 3): the finite vertices are the circumcenters of the Delaunay simplices
     * (built via the lifting-map lower hull). ridge_points is left empty (see class docblock).
     */
    private function ndVoronoi(array $pts, int $d): void
    {
        $tri = new Delaunay($pts);

        $this->vertices = [];
        foreach ($tri->simplices as $s) {
            $spts = [];
            foreach ($s as $i) {
                $spts[] = $pts[$i];
            }
            $cc = self::circumcenterNd($spts, $d);
            if ($cc !== null) {
                $this->vertices[] = $cc;
            }
        }
        $this->ridgePoints = [];
    }

    /** Circumcenter of triangle (a, b, c) in 2-D. */
    private static function circumcenter2d(array $a, array $b, array $c): array
    {
        $ax = $a[0];
        $ay = $a[1];
        $bx = $b[0];
        $by = $b[1];
        $cx = $c[0];
        $cy = $c[1];
        $dd = 2.0 * ($ax * ($by - $cy) + $bx * ($cy - $ay) + $cx * ($ay - $by));
        $a2 = $ax * $ax + $ay * $ay;
        $b2 = $bx * $bx + $by * $by;
        $c2 = $cx * $cx + $cy * $cy;
        $ux = ($a2 * ($by - $cy) + $b2 * ($cy - $ay) + $c2 * ($ay - $by)) / $dd;
        $uy = ($a2 * ($cx - $bx) + $b2 * ($ax - $cx) + $c2 * ($bx - $ax)) / $dd;

        return [$ux, $uy];
    }

    /**
     * Circumcenter of a d-simplex (d+1 points) in d dimensions: solve the linear system
     * 2 (p_i - p_0) . x = |p_i|^2 - |p_0|^2 for i = 1..d. Returns null if the simplex is degenerate.
     *
     * @param  list<list<float>> $spts d+1 points
     * @return list<float>|null
     */
    private static function circumcenterNd(array $spts, int $d): ?array
    {
        $p0 = $spts[0];
        $norm0 = 0.0;
        for ($k = 0; $k < $d; $k++) {
            $norm0 += $p0[$k] * $p0[$k];
        }
        $A = [];
        $bvec = [];
        for ($i = 1; $i <= $d; $i++) {
            $pi = $spts[$i];
            $row = [];
            $ni = 0.0;
            for ($k = 0; $k < $d; $k++) {
                $row[] = 2.0 * ($pi[$k] - $p0[$k]);
                $ni += $pi[$k] * $pi[$k];
            }
            $A[] = $row;
            $bvec[] = $ni - $norm0;
        }

        return self::solve($A, $bvec, $d);
    }

    /**
     * Solve A x = b for an m x m system via Gaussian elimination with partial pivoting. Returns null
     * if singular.
     *
     * @param  list<list<float>> $A
     * @param  list<float>       $b
     * @return list<float>|null
     */
    private static function solve(array $A, array $b, int $m): ?array
    {
        for ($col = 0; $col < $m; $col++) {
            $piv = $col;
            $best = abs($A[$col][$col]);
            for ($r = $col + 1; $r < $m; $r++) {
                if (abs($A[$r][$col]) > $best) {
                    $best = abs($A[$r][$col]);
                    $piv = $r;
                }
            }
            if ($best < 1e-15) {
                return null;
            }
            if ($piv !== $col) {
                $tmp = $A[$col];
                $A[$col] = $A[$piv];
                $A[$piv] = $tmp;
                $tb = $b[$col];
                $b[$col] = $b[$piv];
                $b[$piv] = $tb;
            }
            $inv = 1.0 / $A[$col][$col];
            for ($r = 0; $r < $m; $r++) {
                if ($r === $col) {
                    continue;
                }
                $factor = $A[$r][$col] * $inv;
                if ($factor !== 0.0) {
                    for ($cc = $col; $cc < $m; $cc++) {
                        $A[$r][$cc] -= $factor * $A[$col][$cc];
                    }
                    $b[$r] -= $factor * $b[$col];
                }
            }
        }
        $x = [];
        for ($i = 0; $i < $m; $i++) {
            $x[$i] = $b[$i] / $A[$i][$i];
        }

        return $x;
    }
}
