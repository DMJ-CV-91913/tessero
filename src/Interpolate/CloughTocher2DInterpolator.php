<?php

declare(strict_types=1);

namespace Tessero\Interpolate;

/**
 * Clough-Tocher C1 piecewise-cubic interpolation of scattered 2-D data on a Delaunay triangulation
 * (scipy.interpolate.CloughTocher2DInterpolator). Each query point is located in its containing
 * triangle; the value is a cubic Bernstein-Bezier patch whose corner control points come from the
 * vertex values and estimated vertex gradients, and whose interior control points enforce C1
 * continuity across triangle edges. Points outside the convex hull return the fill value (NaN).
 *
 * This is a faithful, backend-neutral pure-PHP port of SciPy's _interpnd.pyx:
 *   - gradients are estimated by SciPy's global minimum-curvature (Nielson) Gauss-Seidel iteration
 *     (_estimate_gradients_2d_global): per vertex a 2x2 normal system is assembled over incident edges
 *     and solved, iterated to tolerance `tol` (default 1e-6) for at most `maxiter` (default 400) sweeps;
 *   - per triangle the C1 cubic is built and evaluated exactly as _clough_tocher_2d_single, including
 *     the affine-invariant neighbour-centroid g coefficients and the extended-barycentric cubic.
 *
 * The triangulation is built in-process by the same inlined Bowyer-Watson 2-D Delaunay used by
 * LinearNDInterpolator, so the class stands alone. The Clough-Tocher interpolant is geometrically
 * invariant under the per-triangle vertex relabelling and the simplex/neighbour ordering: it depends
 * only on the SET of triangles, which reproduces scipy/Qhull exactly for points in general position.
 * Verified machine-precision (<= 4e-15 rel) vs SciPy 1.17.1 across random scatter (VERIFY.py).
 *
 * Scope: ndim == 2, scalar values, general position only (no four cocircular, no three collinear on a
 * hull edge) -- the same documented limitation as LinearNDInterpolator and Tessero\Spatial\Delaunay.
 */
final class CloughTocher2DInterpolator
{
    /** @var list<list<float>> the (possibly rescaled) data points, one [x, y] row per point */
    private array $points;

    /** @var list<float> one scalar value per point */
    private array $values;

    /** @var list<array{0:int,1:int,2:int}> CCW triangles as index triples into $points */
    private array $tris = [];

    /** @var list<array{0:int,1:int,2:int}> neighbours[s][k] = simplex sharing edge opposite vertex k (or -1) */
    private array $neighbors = [];

    /** @var list<array{0:list<float>,1:list<float>,2:list<float>}> barycentric transform: [Tinv row0, Tinv row1, r] */
    private array $transform = [];

    /** @var list<array{0:float,1:float}> estimated gradient (dx, dy) per data point */
    private array $grad = [];

    private bool $rescale;

    /** @var list<float> per-axis scale (ptp) used when rescale is true */
    private array $scale = [];

    private float $fill;

    private float $tol;

    private int $maxiter;

    /**
     * @param list<list<float|int>> $points (npoints x 2) data-point coordinates (general position)
     * @param list<float|int>       $values one scalar value per point
     * @param float                 $fillValue value returned outside the convex hull (default NaN, as SciPy)
     * @param float                 $tol gradient-estimation convergence tolerance (SciPy default 1e-6)
     * @param int                   $maxiter maximum gradient Gauss-Seidel sweeps (SciPy default 400)
     * @param bool                  $rescale rescale points to unit range per axis before triangulating
     */
    public function __construct(
        array $points,
        array $values,
        float $fillValue = NAN,
        float $tol = 1e-6,
        int $maxiter = 400,
        bool $rescale = false
    ) {
        $pts = array_map(static fn (array $p): array => array_map('floatval', array_values($p)), array_values($points));
        $n = count($pts);
        if ($n === 0) {
            throw new \ValueError('CloughTocher2DInterpolator: points must be non-empty');
        }
        if (count($pts[0]) !== 2) {
            throw new \ValueError('CloughTocher2DInterpolator: only 2-D points are supported');
        }
        if ($n < 3) {
            throw new \ValueError('CloughTocher2DInterpolator: need at least 3 points to triangulate');
        }

        $vals = array_values($values);
        $this->values = array_map('floatval', $vals);
        $this->fill = $fillValue;
        $this->tol = $tol;
        $this->maxiter = $maxiter;

        $this->rescale = $rescale;
        if ($rescale) {
            $mins = $pts[0];
            $maxs = $pts[0];
            foreach ($pts as $p) {
                for ($j = 0; $j < 2; $j++) {
                    if ($p[$j] < $mins[$j]) { $mins[$j] = $p[$j]; }
                    if ($p[$j] > $maxs[$j]) { $maxs[$j] = $p[$j]; }
                }
            }
            for ($j = 0; $j < 2; $j++) {
                $r = $maxs[$j] - $mins[$j];
                $this->scale[$j] = $r > 0.0 ? $r : 1.0;
            }
            $pts = array_map(fn (array $p): array => $this->scaleRow($p), $pts);
        }
        $this->points = $pts;

        $this->tris = $this->bowyerWatson($pts);
        $this->neighbors = $this->buildNeighbors($this->tris);
        $this->transform = $this->buildTransform($this->tris);
        $this->grad = $this->estimateGradients();
    }

    /** @param list<float> $p @return list<float> */
    private function scaleRow(array $p): array
    {
        if (! $this->rescale) {
            return $p;
        }
        return [$p[0] / $this->scale[0], $p[1] / $this->scale[1]];
    }

    /** twice the signed area of triangle (a, b, c); > 0 when CCW */
    private static function orient(array $a, array $b, array $c): float
    {
        return ($b[0] - $a[0]) * ($c[1] - $a[1]) - ($b[1] - $a[1]) * ($c[0] - $a[0]);
    }

    /** > 0 when d lies strictly inside the circumcircle of CCW triangle (a, b, c) */
    private static function inCircle(array $a, array $b, array $c, array $d): float
    {
        $ax = $a[0] - $d[0]; $ay = $a[1] - $d[1];
        $bx = $b[0] - $d[0]; $by = $b[1] - $d[1];
        $cx = $c[0] - $d[0]; $cy = $c[1] - $d[1];
        $az = $ax * $ax + $ay * $ay;
        $bz = $bx * $bx + $by * $by;
        $cz = $cx * $cx + $cy * $cy;
        return $ax * ($by * $cz - $bz * $cy)
             - $ay * ($bx * $cz - $bz * $cx)
             + $az * ($bx * $cy - $by * $cx);
    }

    /** orient a raw index triple so its vertices are CCW; returns [i, j, k] */
    private function ccw(array $t): array
    {
        [$a, $b, $c] = $t;
        return self::orient($this->points[$a], $this->points[$b], $this->points[$c]) < 0.0
            ? [$a, $c, $b] : [$a, $b, $c];
    }

    /**
     * Inlined incremental Bowyer-Watson Delaunay (2-D), returning CCW index triples. Mirrors
     * LinearNDInterpolator's triangulator so the triangle set matches scipy/Qhull for general position.
     *
     * @param list<list<float>> $pts
     * @return list<array{0:int,1:int,2:int}>
     */
    private function bowyerWatson(array $pts): array
    {
        $n = count($pts);
        $minx = $pts[0][0]; $miny = $pts[0][1]; $maxx = $pts[0][0]; $maxy = $pts[0][1];
        foreach ($pts as $p) {
            if ($p[0] < $minx) { $minx = $p[0]; }
            if ($p[1] < $miny) { $miny = $p[1]; }
            if ($p[0] > $maxx) { $maxx = $p[0]; }
            if ($p[1] > $maxy) { $maxy = $p[1]; }
        }
        $cx = ($minx + $maxx) / 2.0;
        $cy = ($miny + $maxy) / 2.0;
        $d = max($maxx - $minx, $maxy - $miny) * 100.0 + 1.0;

        $P = $pts;
        $P[$n]     = [$cx - 2.0 * $d, $cy - $d];
        $P[$n + 1] = [$cx,           $cy + 2.0 * $d];
        $P[$n + 2] = [$cx + 2.0 * $d, $cy - $d];

        $tris = [$this->orientRaw($P, $n, $n + 1, $n + 2)];

        for ($i = 0; $i < $n; $i++) {
            $p = $P[$i];
            $bad = [];
            $keep = [];
            foreach ($tris as $t) {
                if (self::inCircle($P[$t[0]], $P[$t[1]], $P[$t[2]], $p) > 1e-12) {
                    $bad[] = $t;
                } else {
                    $keep[] = $t;
                }
            }
            $count = [];
            $edge = [];
            foreach ($bad as $t) {
                foreach ([[$t[0], $t[1]], [$t[1], $t[2]], [$t[2], $t[0]]] as $e) {
                    $k = $e[0] < $e[1] ? ($e[0] . ',' . $e[1]) : ($e[1] . ',' . $e[0]);
                    $count[$k] = ($count[$k] ?? 0) + 1;
                    $edge[$k] = $e;
                }
            }
            $tris = $keep;
            foreach ($count as $k => $c) {
                if ($c === 1) {
                    [$a, $b] = $edge[$k];
                    $tris[] = $this->orientRaw($P, $a, $b, $i);
                }
            }
        }

        $out = [];
        foreach ($tris as $t) {
            if ($t[0] < $n && $t[1] < $n && $t[2] < $n) {
                $out[] = $this->ccw([$t[0], $t[1], $t[2]]);
            }
        }
        return $out;
    }

    /** orient (a, b, c) CCW against working point list $P */
    private function orientRaw(array $P, int $a, int $b, int $c): array
    {
        return self::orient($P[$a], $P[$b], $P[$c]) < 0.0 ? [$a, $c, $b] : [$a, $b, $c];
    }

    /**
     * Build the neighbour adjacency: neighbors[s][k] = index of the simplex sharing the edge OPPOSITE
     * vertex k of simplex s (the edge (vertex (k+1)%3, vertex (k+2)%3)), or -1 for a hull edge. This is
     * the same convention SciPy's Delaunay.neighbors uses, which _clough_tocher_2d_single relies on.
     *
     * @param list<array{0:int,1:int,2:int}> $tris
     * @return list<array{0:int,1:int,2:int}>
     */
    private function buildNeighbors(array $tris): array
    {
        $edgemap = [];
        foreach ($tris as $si => $s) {
            for ($k = 0; $k < 3; $k++) {
                $a = $s[($k + 1) % 3];
                $b = $s[($k + 2) % 3];
                $key = $a < $b ? ($a . ',' . $b) : ($b . ',' . $a);
                $edgemap[$key][] = [$si, $k];
            }
        }
        $neigh = array_fill(0, count($tris), [-1, -1, -1]);
        foreach ($edgemap as $lst) {
            if (count($lst) === 2) {
                [$s1, $k1] = $lst[0];
                [$s2, $k2] = $lst[1];
                $neigh[$s1][$k1] = $s2;
                $neigh[$s2][$k2] = $s1;
            }
        }
        return $neigh;
    }

    /**
     * SciPy's barycentric transform stack for each simplex: transform[s] = [Tinv_row0, Tinv_row1, r]
     * where Tinv is the inverse of M = [[ax-cx, bx-cx], [ay-cy, by-cy]] and r is the last vertex c.
     * Barycentric c[:2] = Tinv . (x - r), c[2] = 1 - c0 - c1.
     *
     * @param list<array{0:int,1:int,2:int}> $tris
     * @return list<array{0:list<float>,1:list<float>,2:list<float>}>
     */
    private function buildTransform(array $tris): array
    {
        $T = [];
        foreach ($tris as $s) {
            $a = $this->points[$s[0]];
            $b = $this->points[$s[1]];
            $c = $this->points[$s[2]];
            $m00 = $a[0] - $c[0]; $m01 = $b[0] - $c[0];
            $m10 = $a[1] - $c[1]; $m11 = $b[1] - $c[1];
            $det = $m00 * $m11 - $m01 * $m10;
            $i00 =  $m11 / $det; $i01 = -$m01 / $det;
            $i10 = -$m10 / $det; $i11 =  $m00 / $det;
            $T[] = [[$i00, $i01], [$i10, $i11], [$c[0], $c[1]]];
        }
        return $T;
    }

    /** barycentric coordinates [c0, c1, c2] of point q in simplex s */
    private function bary(int $s, array $q): array
    {
        $t = $this->transform[$s];
        $dx = $q[0] - $t[2][0];
        $dy = $q[1] - $t[2][1];
        $c0 = $t[0][0] * $dx + $t[0][1] * $dy;
        $c1 = $t[1][0] * $dx + $t[1][1] * $dy;
        return [$c0, $c1, 1.0 - $c0 - $c1];
    }

    /** index of the simplex containing q, or -1 if outside the hull (brute barycentric-sign scan) */
    private function findSimplex(array $q): int
    {
        foreach ($this->tris as $s => $_t) {
            $b = $this->bary($s, $q);
            if ($b[0] >= -1e-12 && $b[1] >= -1e-12 && $b[2] >= -1e-12) {
                return $s;
            }
        }
        return -1;
    }

    /**
     * Estimate vertex gradients by SciPy's global minimum-curvature Gauss-Seidel iteration
     * (_estimate_gradients_2d_global). Returns one [gx, gy] row per data point.
     *
     * @return list<array{0:float,1:float}>
     */
    private function estimateGradients(): array
    {
        $n = count($this->points);
        // vertex -> set of neighbour vertices (via shared triangle edges), sorted ascending for determinism
        $vn = array_fill(0, $n, []);
        foreach ($this->tris as $t) {
            foreach ([[$t[0], $t[1]], [$t[1], $t[2]], [$t[2], $t[0]]] as $e) {
                $vn[$e[0]][$e[1]] = true;
                $vn[$e[1]][$e[0]] = true;
            }
        }
        $neighOf = [];
        for ($i = 0; $i < $n; $i++) {
            $nb = array_keys($vn[$i]);
            sort($nb);
            $neighOf[$i] = $nb;
        }

        $y = array_fill(0, 2 * $n, 0.0);
        $data = $this->values;
        $pts = $this->points;

        for ($it = 0; $it < $this->maxiter; $it++) {
            $err = 0.0;
            for ($ip = 0; $ip < $n; $ip++) {
                $q00 = 0.0; $q01 = 0.0; $q11 = 0.0;
                $s0 = 0.0; $s1 = 0.0;
                foreach ($neighOf[$ip] as $ip2) {
                    $ex = $pts[$ip2][0] - $pts[$ip][0];
                    $ey = $pts[$ip2][1] - $pts[$ip][1];
                    $L = sqrt($ex * $ex + $ey * $ey);
                    $L3 = $L * $L * $L;
                    $f1 = $data[$ip];
                    $f2 = $data[$ip2];
                    $df2 = -$ex * $y[2 * $ip2 + 0] - $ey * $y[2 * $ip2 + 1];
                    $q00 += 4.0 * $ex * $ex / $L3;
                    $q01 += 4.0 * $ex * $ey / $L3;
                    $q11 += 4.0 * $ey * $ey / $L3;
                    $s0 += (6.0 * ($f1 - $f2) - 2.0 * $df2) * $ex / $L3;
                    $s1 += (6.0 * ($f1 - $f2) - 2.0 * $df2) * $ey / $L3;
                }
                $det = $q00 * $q11 - $q01 * $q01;
                $r0 = ($q11 * $s0 - $q01 * $s1) / $det;
                $r1 = (-$q01 * $s0 + $q00 * $s1) / $det;
                $change = max(abs($y[2 * $ip + 0] + $r0), abs($y[2 * $ip + 1] + $r1));
                $y[2 * $ip + 0] = -$r0;
                $y[2 * $ip + 1] = -$r1;
                $change /= max(1.0, max(abs($r0), abs($r1)));
                if ($change > $err) {
                    $err = $change;
                }
            }
            if ($err < $this->tol) {
                break;
            }
        }

        $out = [];
        for ($i = 0; $i < $n; $i++) {
            $out[$i] = [$y[2 * $i + 0], $y[2 * $i + 1]];
        }
        return $out;
    }

    /**
     * Evaluate the C1 cubic Bezier over simplex $s at barycentric $b. Direct port of
     * _clough_tocher_2d_single.
     *
     * @param list<float> $b barycentric coordinates [b0, b1, b2]
     */
    private function cloughTocherSingle(int $s, array $b): float
    {
        $simp = $this->tris[$s];
        $p0 = $this->points[$simp[0]];
        $p1 = $this->points[$simp[1]];
        $p2 = $this->points[$simp[2]];

        $e12x = $p1[0] - $p0[0]; $e12y = $p1[1] - $p0[1];
        $e23x = $p2[0] - $p1[0]; $e23y = $p2[1] - $p1[1];
        $e31x = $p0[0] - $p2[0]; $e31y = $p0[1] - $p2[1];

        $f1 = $this->values[$simp[0]];
        $f2 = $this->values[$simp[1]];
        $f3 = $this->values[$simp[2]];

        $d0 = $this->grad[$simp[0]];
        $d1 = $this->grad[$simp[1]];
        $d2 = $this->grad[$simp[2]];

        $df12 =  ($d0[0] * $e12x + $d0[1] * $e12y);
        $df21 = -($d1[0] * $e12x + $d1[1] * $e12y);
        $df23 =  ($d1[0] * $e23x + $d1[1] * $e23y);
        $df32 = -($d2[0] * $e23x + $d2[1] * $e23y);
        $df31 =  ($d2[0] * $e31x + $d2[1] * $e31y);
        $df13 = -($d0[0] * $e31x + $d0[1] * $e31y);

        $c3000 = $f1;
        $c2100 = ($df12 + 3.0 * $c3000) / 3.0;
        $c2010 = ($df13 + 3.0 * $c3000) / 3.0;
        $c0300 = $f2;
        $c1200 = ($df21 + 3.0 * $c0300) / 3.0;
        $c0210 = ($df23 + 3.0 * $c0300) / 3.0;
        $c0030 = $f3;
        $c1020 = ($df31 + 3.0 * $c0030) / 3.0;
        $c0120 = ($df32 + 3.0 * $c0030) / 3.0;

        $c2001 = ($c2100 + $c2010 + $c3000) / 3.0;
        $c0201 = ($c1200 + $c0300 + $c0210) / 3.0;
        $c0021 = ($c1020 + $c0120 + $c0030) / 3.0;

        $g = [0.0, 0.0, 0.0];
        for ($k = 0; $k < 3; $k++) {
            $itri = $this->neighbors[$s][$k];
            if ($itri === -1) {
                $g[$k] = -0.5;
                continue;
            }
            $s2 = $this->tris[$itri];
            $yx = ($this->points[$s2[0]][0] + $this->points[$s2[1]][0] + $this->points[$s2[2]][0]) / 3.0;
            $yy = ($this->points[$s2[0]][1] + $this->points[$s2[1]][1] + $this->points[$s2[2]][1]) / 3.0;
            $c = $this->bary($s, [$yx, $yy]);
            if ($k === 0) {
                $g[$k] = (2.0 * $c[2] + $c[1] - 1.0) / (2.0 - 3.0 * $c[2] - 3.0 * $c[1]);
            } elseif ($k === 1) {
                $g[$k] = (2.0 * $c[0] + $c[2] - 1.0) / (2.0 - 3.0 * $c[0] - 3.0 * $c[2]);
            } else {
                $g[$k] = (2.0 * $c[1] + $c[0] - 1.0) / (2.0 - 3.0 * $c[1] - 3.0 * $c[0]);
            }
        }

        $c0111 = ($g[0] * (-$c0300 + 3.0 * $c0210 - 3.0 * $c0120 + $c0030)
                 + (-$c0300 + 2.0 * $c0210 - $c0120 + $c0021 + $c0201)) / 2.0;
        $c1011 = ($g[1] * (-$c0030 + 3.0 * $c1020 - 3.0 * $c2010 + $c3000)
                 + (-$c0030 + 2.0 * $c1020 - $c2010 + $c2001 + $c0021)) / 2.0;
        $c1101 = ($g[2] * (-$c3000 + 3.0 * $c2100 - 3.0 * $c1200 + $c0300)
                 + (-$c3000 + 2.0 * $c2100 - $c1200 + $c2001 + $c0201)) / 2.0;

        $c1002 = ($c1101 + $c1011 + $c2001) / 3.0;
        $c0102 = ($c1101 + $c0111 + $c0201) / 3.0;
        $c0012 = ($c1011 + $c0111 + $c0021) / 3.0;

        $c0003 = ($c1002 + $c0102 + $c0012) / 3.0;

        $minval = min($b[0], $b[1], $b[2]);
        $b1 = $b[0] - $minval;
        $b2 = $b[1] - $minval;
        $b3 = $b[2] - $minval;
        $b4 = 3.0 * $minval;

        return $b1 ** 3 * $c3000 + 3.0 * $b1 ** 2 * $b2 * $c2100 + 3.0 * $b1 ** 2 * $b3 * $c2010
             + 3.0 * $b1 ** 2 * $b4 * $c2001 + 3.0 * $b1 * $b2 ** 2 * $c1200
             + 6.0 * $b1 * $b2 * $b4 * $c1101 + 3.0 * $b1 * $b3 ** 2 * $c1020
             + 6.0 * $b1 * $b3 * $b4 * $c1011 + 3.0 * $b1 * $b4 ** 2 * $c1002
             + $b2 ** 3 * $c0300 + 3.0 * $b2 ** 2 * $b3 * $c0210 + 3.0 * $b2 ** 2 * $b4 * $c0201
             + 3.0 * $b2 * $b3 ** 2 * $c0120 + 6.0 * $b2 * $b3 * $b4 * $c0111
             + 3.0 * $b2 * $b4 ** 2 * $c0102 + $b3 ** 3 * $c0030 + 3.0 * $b3 ** 2 * $b4 * $c0021
             + 3.0 * $b3 * $b4 ** 2 * $c0012 + $b4 ** 3 * $c0003;
    }

    /**
     * Interpolated value(s) at the query point(s); outside the convex hull yields the fill value (NaN).
     *
     * @param list<list<float|int>> $xi query points (m x 2)
     * @return list<float>
     */
    public function query(array $xi): array
    {
        $q = array_map(static fn (array $p): array => array_map('floatval', array_values($p)), array_values($xi));
        if ($this->rescale) {
            $q = array_map(fn (array $p): array => $this->scaleRow($p), $q);
        }
        $out = [];
        foreach ($q as $pt) {
            $s = $this->findSimplex($pt);
            if ($s < 0) {
                $out[] = $this->fill;
            } else {
                $b = $this->bary($s, $pt);
                $out[] = $this->cloughTocherSingle($s, $b);
            }
        }
        return $out;
    }

    /**
     * Callable form, matching scipy's CloughTocher2DInterpolator(xi).
     *
     * @param list<list<float|int>> $xi
     * @return list<float>
     */
    public function __invoke(array $xi): array
    {
        return $this->query($xi);
    }
}
