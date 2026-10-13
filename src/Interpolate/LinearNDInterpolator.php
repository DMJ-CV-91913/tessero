<?php

declare(strict_types=1);

namespace Tessero\Interpolate;

/**
 * Piecewise-linear interpolation on a 2-D Delaunay triangulation (scipy.interpolate.LinearNDInterpolator):
 * each query point is located in the triangle that contains it and the value is the barycentric-linear blend
 * of that triangle's three vertex values; points outside the convex hull return NaN. Pure PHP and
 * backend-neutral: the triangulation is built in-process by an inlined Bowyer-Watson incremental Delaunay so
 * the class stands alone (it does not depend on Tessero\Spatial\Delaunay, though it computes the same
 * triangulation for general-position inputs). Mirrors SciPy's 2-D, value-only behaviour.
 *
 * Scope: ndim == 2, general position only (no three points collinear on a hull edge, no four cocircular).
 * Under degeneracy the Delaunay triangulation is not unique and scipy (Qhull) may choose a different diagonal;
 * the interpolated value still agrees everywhere the two triangulations cover the same region, but strict
 * parity is only guaranteed for general-position point sets, matching the campaign's documented limitation.
 */
final class LinearNDInterpolator
{
    /** @var list<list<float>> the (possibly rescaled) data points, one [x, y] row per point */
    private array $points;

    /** @var list<list<float>> one row of values per point; scalar data is stored as a width-1 row */
    private array $values;

    /** @var list<array{0:int,1:int,2:int}> CCW triangles as index triples into $points */
    private array $tris = [];

    private bool $scalar;

    private int $width;

    private bool $rescale;

    /** @var list<float> per-axis scale (ptp) used when rescale is true */
    private array $scale = [];

    private float $fill;

    /**
     * @param list<list<float|int>> $points (npoints x 2) data-point coordinates
     * @param list<mixed>           $values one value per point: float, or list<float> (vector-valued)
     * @param float                 $fillValue value returned outside the convex hull (default NaN, as SciPy)
     * @param bool                  $rescale rescale points to unit scale (divide each axis by its range)
     *                                       before triangulating, as SciPy's rescale=True does
     */
    public function __construct(array $points, array $values, float $fillValue = NAN, bool $rescale = false)
    {
        $pts = array_map(static fn (array $p): array => array_map('floatval', array_values($p)), array_values($points));
        $n = count($pts);
        if ($n === 0) {
            throw new \ValueError('LinearNDInterpolator: points must be non-empty');
        }
        $ndim = count($pts[0]);
        if ($ndim !== 2) {
            throw new \ValueError('LinearNDInterpolator: only 2-D points are supported');
        }
        if ($n < 3) {
            throw new \ValueError('LinearNDInterpolator: need at least 3 points to triangulate');
        }

        // normalise values to a list of rows (scalar -> width-1 rows) so scalar and vector share one path
        $vals = array_values($values);
        $this->scalar = ! is_array($vals[0]);
        $rows = [];
        foreach ($vals as $v) {
            $rows[] = $this->scalar ? [(float) $v] : array_map('floatval', array_values($v));
        }
        $this->width = count($rows[0]);
        $this->values = $rows;
        $this->fill = $fillValue;

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
                $this->scale[$j] = $r > 0.0 ? $r : 1.0;   // SciPy: ptp per axis, zero range -> 1
            }
            $pts = array_map(fn (array $p): array => $this->scaleRow($p), $pts);
        }
        $this->points = $pts;
        $this->tris = $this->bowyerWatson($pts);
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

    /**
     * > 0 when d lies strictly inside the circumcircle of the CCW triangle (a, b, c).
     * Determinant of the standard 3x3 in-circle matrix (robust enough for general position).
     */
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
     * Inlined incremental Bowyer-Watson Delaunay (2-D), independent of Qhull. A large super-triangle
     * encloses all points; each point splits the triangles whose circumcircle it violates and re-stitches
     * the cavity boundary. Super-triangle triangles are discarded at the end. Returns CCW index triples.
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

        // super-triangle vertices stored at indices n, n+1, n+2 of a working point list
        $P = $pts;
        $P[$n]     = [$cx - 2.0 * $d, $cy - $d];
        $P[$n + 1] = [$cx,           $cy + 2.0 * $d];
        $P[$n + 2] = [$cx + 2.0 * $d, $cy - $d];

        /** @var list<array{0:int,1:int,2:int}> working triangle set, each kept CCW */
        $tris = [$this->orientRaw($P, $n, $n + 1, $n + 2)];   // super-triangle oriented CCW up front

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
            // collect boundary edges of the cavity (edges shared by exactly one bad triangle)
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

        // drop any triangle touching the super-triangle; re-orient survivors CCW on the real points
        $out = [];
        foreach ($tris as $t) {
            if ($t[0] < $n && $t[1] < $n && $t[2] < $n) {
                $out[] = $this->ccw([$t[0], $t[1], $t[2]]);
            }
        }
        return $out;
    }

    /** orient (a, b, c) CCW against working point list $P (used while the super-triangle is live) */
    private function orientRaw(array $P, int $a, int $b, int $c): array
    {
        return self::orient($P[$a], $P[$b], $P[$c]) < 0.0 ? [$a, $c, $b] : [$a, $b, $c];
    }

    /**
     * Barycentric-linear value row at a single query point, or null if outside the hull.
     *
     * @param list<float> $q
     * @return list<float>|null
     */
    private function evalPoint(array $q): ?array
    {
        foreach ($this->tris as $t) {
            $a = $this->points[$t[0]];
            $b = $this->points[$t[1]];
            $c = $this->points[$t[2]];
            $det = self::orient($a, $b, $c);
            if ($det === 0.0) {
                continue;
            }
            $l0 = self::orient($b, $c, $q) / $det;
            $l1 = self::orient($c, $a, $q) / $det;
            $l2 = self::orient($a, $b, $q) / $det;
            if ($l0 >= -1e-12 && $l1 >= -1e-12 && $l2 >= -1e-12) {
                $va = $this->values[$t[0]];
                $vb = $this->values[$t[1]];
                $vc = $this->values[$t[2]];
                $row = [];
                for ($w = 0; $w < $this->width; $w++) {
                    $row[$w] = $l0 * $va[$w] + $l1 * $vb[$w] + $l2 * $vc[$w];
                }
                return $row;
            }
        }
        return null;
    }

    /**
     * Piecewise-linear value(s) at the query point(s). Points outside the convex hull yield the fill value
     * (NaN by default), scalar or broadcast across all components for vector-valued data.
     *
     * @param list<list<float|int>> $xi query points (m x 2)
     * @return list<mixed> one value per query point (float, or list<float> for vector-valued data)
     */
    public function query(array $xi): array
    {
        $q = array_map(static fn (array $p): array => array_map('floatval', array_values($p)), array_values($xi));
        if ($this->rescale) {
            $q = array_map(fn (array $p): array => $this->scaleRow($p), $q);
        }
        $out = [];
        foreach ($q as $pt) {
            $row = $this->evalPoint($pt);
            if ($row === null) {
                $out[] = $this->scalar ? $this->fill : array_fill(0, $this->width, $this->fill);
            } else {
                $out[] = $this->scalar ? $row[0] : $row;
            }
        }
        return $out;
    }

    /**
     * Callable form, matching scipy's LinearNDInterpolator(xi).
     *
     * @param list<list<float|int>> $xi
     * @return list<mixed>
     */
    public function __invoke(array $xi): array
    {
        return $this->query($xi);
    }
}
