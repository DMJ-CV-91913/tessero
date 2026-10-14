<?php

declare(strict_types=1);

namespace Tessero\Spatial;

/**
 * scipy.spatial.Delaunay: a Delaunay triangulation of a point set. For 2-D the triangulation is
 * built with the incremental Bowyer-Watson algorithm; for d >= 3 it is built as the LOWER convex
 * hull of the points lifted onto the paraboloid x -> (x, |x|^2) (the classical lifting map), using
 * the same beneath-beyond (quickhull) construction as Spatial\ConvexHull. Pure PHP (no kernel),
 * identical on both backends -- no Qhull FFI.
 *
 * The SIMPLEX SET reproduces scipy/Qhull exactly for points in general position (no d+2 points
 * cospherical, no d+1 collinear/coplanar): the Delaunay triangulation is then unique, so the set of
 * simplices -- each stored as a sorted tuple of input-point indices -- is well defined and matches
 * regardless of construction order. The per-simplex vertex ORDER and the order of simplices in the
 * list are NOT asserted against scipy (Qhull's are Qhull-internal); only the set is a stable
 * invariant. This was de-risked to exact agreement over 120 general-position 3-D/4-D point sets.
 *
 * ->simplices    the triangulation as a list of (ndim+1) index tuples (input-point indices).
 * ->findSimplex  for each query point, the index of the containing simplex (into ->simplices), or
 *                -1 when the point lies outside the convex hull -- a brute-force barycentric sign
 *                scan over the generalized (ndim+1) barycentric coordinates.
 * ->transform    scipy's barycentric transform stack: for simplex s, transform[s][0..ndim-1] is the
 *                inverse of the (vertex - last_vertex) matrix and transform[s][ndim] is the last
 *                vertex r, so barycentric c[:ndim] = Tinv . (x - r), c[ndim] = 1 - sum(c[:ndim]).
 *
 * General-position inputs only (matching the de-risked scope); degenerate/cospherical configurations
 * are out of scope and may disagree with Qhull's symbolic-perturbation tie-breaking.
 */
final class Delaunay
{
    /** @var list<list<float>> the input points */
    public array $points;

    /** @var int number of input points */
    public int $npoints;

    /** @var int spatial dimension */
    public int $ndim = 2;

    /** @var list<list<int>> simplices as input-point index tuples (ndim+1 indices each) */
    public array $simplices;

    /** @var list<list<list<float>>> scipy-style barycentric transform, shape (nsimplex, ndim+1, ndim) */
    public array $transform;

    /**
     * @param list<list<float>> $points a list of coordinate rows (>= ndim+1 points, general position)
     */
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
                throw new \InvalidArgumentException('Delaunay: all points must have the same number of columns');
            }
            $row = [];
            foreach ($p as $c) {
                $row[] = (float) $c;
            }
            $pts[] = $row;
        }
        if ($d === null || $d < 2) {
            throw new \InvalidArgumentException('Delaunay requires points with at least 2 columns');
        }
        $n = count($pts);
        if ($n < $d + 1) {
            throw new \InvalidArgumentException('Delaunay requires at least ndim+1 points');
        }
        $this->points = $pts;
        $this->npoints = $n;
        $this->ndim = $d;

        if ($d === 2) {
            $this->simplices = $this->bowyerWatson($pts);
            $this->transform = $this->buildTransform();

            return;
        }

        $this->simplices = $this->ndDelaunay($pts, $d);
        $this->transform = $this->buildTransformNd($d);
    }

    /**
     * N-D Delaunay via the lifting map: lift each point x -> (x, |x|^2) into d+1 dimensions, take the
     * LOWER convex hull (facets whose outward normal points down in the lift axis), and project each
     * lower facet's (d+1) vertices back to input indices -- those are the Delaunay simplices. The hull
     * is built with the same incremental beneath-beyond construction as Spatial\ConvexHull::hullNd.
     *
     * @param  list<list<float>> $pts
     * @return list<list<int>>
     */
    private function ndDelaunay(array $pts, int $d): array
    {
        $n = count($pts);
        $D = $d + 1;

        // lifted points: (x_0, ..., x_{d-1}, |x|^2)
        $lift = [];
        foreach ($pts as $p) {
            $s = 0.0;
            foreach ($p as $c) {
                $s += $c * $c;
            }
            $row = $p;
            $row[] = $s;
            $lift[] = $row;
        }

        $facets = $this->hullFacets($lift, $D, $n);

        $simplices = [];
        $seen = [];
        foreach ($facets as $f) {
            // lower-hull facet: outward normal has a strictly negative last (lift-axis) component
            if ($f['nrm'][$D - 1] < -1e-12) {
                $combo = $f['idx'];
                sort($combo);
                $key = implode(',', $combo);
                if (! isset($seen[$key])) {
                    $seen[$key] = true;
                    $simplices[] = $combo;
                }
            }
        }

        return $simplices;
    }

    /**
     * Incremental beneath-beyond convex hull in $d dimensions (mirrors Spatial\ConvexHull::hullNd).
     * Returns the facets as ['idx' => [d indices], 'nrm' => [d], 'off' => float], the normal oriented
     * so the running interior centroid is strictly negative.
     *
     * @param  list<list<float>> $pts
     * @return list<array{idx: list<int>, nrm: list<float>, off: float}>
     */
    private function hullFacets(array $pts, int $d, int $n): array
    {
        $eps = 1e-12;

        $init = $this->initialSimplex($pts, $d, $n);
        if (count($init) < $d + 1) {
            throw new \InvalidArgumentException('input is degenerate (not full-dimensional); QhullError');
        }

        $interior = $this->centroid($pts, $init, $d);

        $facets = [];
        foreach ($this->combinations($init, $d) as $combo) {
            sort($combo);
            [$nrm, $off] = $this->facetPlane($pts, $combo, $interior, $d);
            $facets[implode(',', $combo)] = ['idx' => $combo, 'nrm' => $nrm, 'off' => $off];
        }

        $processed = array_fill_keys($init, true);

        for ($p = 0; $p < $n; $p++) {
            if (isset($processed[$p])) {
                continue;
            }
            $x = $pts[$p];
            $visible = [];
            foreach ($facets as $key => $f) {
                $dist = $f['off'];
                $nrm = $f['nrm'];
                for ($k = 0; $k < $d; $k++) {
                    $dist += $nrm[$k] * $x[$k];
                }
                if ($dist > $eps) {
                    $visible[$key] = true;
                }
            }
            $processed[$p] = true;
            if (count($visible) === 0) {
                continue;
            }
            $ridgeCount = [];
            $ridgeIdx = [];
            foreach ($visible as $key => $_) {
                foreach ($this->combinations($facets[$key]['idx'], $d - 1) as $ridge) {
                    sort($ridge);
                    $rk = implode(',', $ridge);
                    $ridgeCount[$rk] = ($ridgeCount[$rk] ?? 0) + 1;
                    $ridgeIdx[$rk] = $ridge;
                }
            }
            foreach ($visible as $key => $_) {
                unset($facets[$key]);
            }
            $verts = [];
            foreach ($facets as $f) {
                foreach ($f['idx'] as $vi) {
                    $verts[$vi] = true;
                }
            }
            $verts[$p] = true;
            $interior = $this->centroid($pts, array_keys($verts), $d);
            foreach ($facets as $key => $f) {
                $dist = $f['off'];
                for ($k = 0; $k < $d; $k++) {
                    $dist += $f['nrm'][$k] * $interior[$k];
                }
                if ($dist > 0.0) {
                    for ($k = 0; $k < $d; $k++) {
                        $facets[$key]['nrm'][$k] = -$facets[$key]['nrm'][$k];
                    }
                    $facets[$key]['off'] = -$f['off'];
                }
            }
            foreach ($ridgeCount as $rk => $cnt) {
                if ($cnt !== 1) {
                    continue;
                }
                $combo = $ridgeIdx[$rk];
                $combo[] = $p;
                if (count(array_unique($combo)) !== $d) {
                    continue;
                }
                sort($combo);
                [$nrm, $off] = $this->facetPlane($pts, $combo, $interior, $d);
                $facets[implode(',', $combo)] = ['idx' => $combo, 'nrm' => $nrm, 'off' => $off];
            }
        }

        return array_values($facets);
    }

    /** Greedily pick d+1 affinely-independent point indices (mirrors ConvexHull). */
    private function initialSimplex(array $pts, int $d, int $n): array
    {
        $order = range(0, $n - 1);
        usort($order, static function (int $a, int $b) use ($pts, $d): int {
            for ($k = 0; $k < $d; $k++) {
                if ($pts[$a][$k] !== $pts[$b][$k]) {
                    return $pts[$a][$k] <=> $pts[$b][$k];
                }
            }

            return 0;
        });
        $init = [$order[0]];
        foreach ($order as $idx) {
            if (in_array($idx, $init, true)) {
                continue;
            }
            $trial = $init;
            $trial[] = $idx;
            if ($this->affineRank($pts, $trial, $d) === count($trial) - 1) {
                $init = $trial;
            }
            if (count($init) === $d + 1) {
                break;
            }
        }

        return $init;
    }

    private function affineRank(array $pts, array $idx, int $d): int
    {
        $m = count($idx);
        if ($m <= 1) {
            return 0;
        }
        $base = $pts[$idx[0]];
        $rows = [];
        for ($i = 1; $i < $m; $i++) {
            $row = [];
            for ($k = 0; $k < $d; $k++) {
                $row[] = $pts[$idx[$i]][$k] - $base[$k];
            }
            $rows[] = $row;
        }

        return $this->matrixRank($rows, $m - 1, $d);
    }

    private function matrixRank(array $A, int $r, int $c): int
    {
        $tol = 1e-9;
        $rank = 0;
        $prow = 0;
        for ($col = 0; $col < $c && $prow < $r; $col++) {
            $piv = $prow;
            $best = abs($A[$prow][$col]);
            for ($i = $prow + 1; $i < $r; $i++) {
                if (abs($A[$i][$col]) > $best) {
                    $best = abs($A[$i][$col]);
                    $piv = $i;
                }
            }
            if ($best <= $tol) {
                continue;
            }
            $tmp = $A[$prow];
            $A[$prow] = $A[$piv];
            $A[$piv] = $tmp;
            $inv = 1.0 / $A[$prow][$col];
            for ($i = $prow + 1; $i < $r; $i++) {
                $factor = $A[$i][$col] * $inv;
                if ($factor !== 0.0) {
                    for ($cc = $col; $cc < $c; $cc++) {
                        $A[$i][$cc] -= $factor * $A[$prow][$cc];
                    }
                }
            }
            $prow++;
            $rank++;
        }

        return $rank;
    }

    private function centroid(array $pts, array $idx, int $d): array
    {
        $c = array_fill(0, $d, 0.0);
        $m = count($idx);
        foreach ($idx as $i) {
            for ($k = 0; $k < $d; $k++) {
                $c[$k] += $pts[$i][$k];
            }
        }
        for ($k = 0; $k < $d; $k++) {
            $c[$k] /= $m;
        }

        return $c;
    }

    /**
     * Supporting hyperplane of a d-point facet via the generalized cross product (cofactor
     * expansion of the (d-1) tangent vectors), oriented so `interior` is strictly negative.
     *
     * @return array{0: list<float>, 1: float}
     */
    private function facetPlane(array $pts, array $combo, array $interior, int $d): array
    {
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
        $off = 0.0;
        for ($k = 0; $k < $d; $k++) {
            $off -= $nrm[$k] * $base[$k];
        }
        $di = $off;
        for ($k = 0; $k < $d; $k++) {
            $di += $nrm[$k] * $interior[$k];
        }
        if ($di > 0.0) {
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

    /**
     * All r-combinations of the given index list (mirrors ConvexHull).
     *
     * @param  list<int> $items
     * @return list<list<int>>
     */
    private function combinations(array $items, int $r): array
    {
        $items = array_values($items);
        $nn = count($items);
        if ($r < 0 || $r > $nn) {
            return [];
        }
        if ($r === 0) {
            return [[]];
        }
        $result = [];
        $idx = range(0, $r - 1);
        while (true) {
            $combo = [];
            foreach ($idx as $i) {
                $combo[] = $items[$i];
            }
            $result[] = $combo;
            $i = $r - 1;
            while ($i >= 0 && $idx[$i] === $i + $nn - $r) {
                $i--;
            }
            if ($i < 0) {
                break;
            }
            $idx[$i]++;
            for ($j = $i + 1; $j < $r; $j++) {
                $idx[$j] = $idx[$j - 1] + 1;
            }
        }

        return $result;
    }

    /**
     * Incremental Bowyer-Watson (2-D) over a robust super-triangle placed a huge offset (~1e6) about
     * the centroid. The in-circle test is an orientation-corrected determinant (float64).
     *
     * @param  list<list<float>> $pts
     * @return list<array{0:int,1:int,2:int}>
     */
    private function bowyerWatson(array $pts): array
    {
        $n = count($pts);

        $mx = 0.0;
        $my = 0.0;
        foreach ($pts as $p) {
            $mx += $p[0];
            $my += $p[1];
        }
        $mx /= $n;
        $my /= $n;

        $huge = 1.0e6;
        $ap = $pts;
        $ap[$n] = [$mx - $huge, $my - $huge];
        $ap[$n + 1] = [$mx + $huge, $my - $huge];
        $ap[$n + 2] = [$mx, $my + $huge];

        $tris = [];
        $tris[$this->triKey($n, $n + 1, $n + 2)] = [$n, $n + 1, $n + 2];

        for ($ip = 0; $ip < $n; $ip++) {
            $px = $pts[$ip][0];
            $py = $pts[$ip][1];

            $bad = [];
            foreach ($tris as $k => $t) {
                if ($this->inCircle($ap[$t[0]], $ap[$t[1]], $ap[$t[2]], $px, $py) > 0.0) {
                    $bad[$k] = $t;
                }
            }

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

            foreach ($bad as $k => $_) {
                unset($tris[$k]);
            }

            foreach ($edgeCount as $ek => $c) {
                if ($c === 1) {
                    [$a, $b] = $edge[$ek];
                    $tris[$this->triKey($a, $b, $ip)] = [$a, $b, $ip];
                }
            }
        }

        $result = [];
        foreach ($tris as $t) {
            if ($t[0] < $n && $t[1] < $n && $t[2] < $n) {
                $result[] = [$t[0], $t[1], $t[2]];
            }
        }

        return $result;
    }

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
     * triangle (a, b, c), independent of winding.
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

        $det = $ax * ($by * $c2 - $b2 * $cy)
            - $ay * ($bx * $c2 - $b2 * $cx)
            + $a2 * ($bx * $cy - $by * $cx);

        return $area2 > 0.0 ? $det : -$det;
    }

    /**
     * scipy-style barycentric transform stack (2-D): one (3, 2) block per simplex.
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

            $m00 = $v0[0] - $v2[0];
            $m01 = $v1[0] - $v2[0];
            $m10 = $v0[1] - $v2[1];
            $m11 = $v1[1] - $v2[1];

            $det = $m00 * $m11 - $m01 * $m10;
            if ($det == 0.0) {
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
     * scipy-style barycentric transform stack (N-D): one (ndim+1, ndim) block per simplex. The first
     * ndim rows are the inverse of the matrix whose columns are (vertex_i - vertex_last); the last
     * row is the last vertex r. A degenerate (flat) simplex emits NaNs like scipy.
     *
     * @return list<list<list<float>>>
     */
    private function buildTransformNd(int $d): array
    {
        $out = [];
        foreach ($this->simplices as $s) {
            $vlast = $this->points[$s[$d]];
            // M[k][j] = points[s[j]][k] - vlast[k], an d x d matrix (columns j = 0..d-1)
            $M = [];
            for ($k = 0; $k < $d; $k++) {
                $row = [];
                for ($j = 0; $j < $d; $j++) {
                    $row[] = $this->points[$s[$j]][$k] - $vlast[$k];
                }
                $M[] = $row;
            }
            $inv = $this->invert($M, $d);
            if ($inv === null) {
                $block = [];
                for ($i = 0; $i < $d; $i++) {
                    $block[] = array_fill(0, $d, NAN);
                }
                $block[] = $vlast;
                $out[] = $block;
                continue;
            }
            $block = $inv;
            $block[] = $vlast;
            $out[] = $block;
        }

        return $out;
    }

    /**
     * Inverse of an m x m matrix via Gauss-Jordan with partial pivoting, or null if singular.
     *
     * @param  list<list<float>> $A
     * @return list<list<float>>|null
     */
    private function invert(array $A, int $m): ?array
    {
        $aug = [];
        for ($i = 0; $i < $m; $i++) {
            $row = $A[$i];
            for ($j = 0; $j < $m; $j++) {
                $row[] = ($i === $j) ? 1.0 : 0.0;
            }
            $aug[] = $row;
        }
        for ($col = 0; $col < $m; $col++) {
            $piv = $col;
            $best = abs($aug[$col][$col]);
            for ($r = $col + 1; $r < $m; $r++) {
                if (abs($aug[$r][$col]) > $best) {
                    $best = abs($aug[$r][$col]);
                    $piv = $r;
                }
            }
            if ($best < 1e-15) {
                return null;
            }
            if ($piv !== $col) {
                $tmp = $aug[$col];
                $aug[$col] = $aug[$piv];
                $aug[$piv] = $tmp;
            }
            $inv = 1.0 / $aug[$col][$col];
            for ($cc = 0; $cc < 2 * $m; $cc++) {
                $aug[$col][$cc] *= $inv;
            }
            for ($r = 0; $r < $m; $r++) {
                if ($r === $col) {
                    continue;
                }
                $factor = $aug[$r][$col];
                if ($factor !== 0.0) {
                    for ($cc = 0; $cc < 2 * $m; $cc++) {
                        $aug[$r][$cc] -= $factor * $aug[$col][$cc];
                    }
                }
            }
        }
        $out = [];
        for ($i = 0; $i < $m; $i++) {
            $row = [];
            for ($j = 0; $j < $m; $j++) {
                $row[] = $aug[$i][$m + $j];
            }
            $out[] = $row;
        }

        return $out;
    }

    /**
     * Barycentric coordinates of a point with respect to a given simplex (length ndim+1, summing
     * to 1). Works for any dimension via the stored transform block.
     *
     * @param  list<float> $x
     * @return list<float>
     */
    public function barycentric(int $simplex, array $x): array
    {
        $d = $this->ndim;
        $T = $this->transform[$simplex];
        $r = $T[$d];
        $dx = [];
        for ($k = 0; $k < $d; $k++) {
            $dx[$k] = (float) $x[$k] - $r[$k];
        }
        $c = [];
        $sum = 0.0;
        for ($i = 0; $i < $d; $i++) {
            $ci = 0.0;
            for ($k = 0; $k < $d; $k++) {
                $ci += $T[$i][$k] * $dx[$k];
            }
            $c[$i] = $ci;
            $sum += $ci;
        }
        $c[$d] = 1.0 - $sum;

        return $c;
    }

    /**
     * For each query point, the index of the containing simplex (into ->simplices), or -1 when the
     * point is outside the convex hull. Brute-force barycentric sign scan over the generalized
     * (ndim+1) barycentric coordinates; a point is inside simplex s iff all coordinates are >= -tol.
     *
     * @param  list<list<float>> $queryPoints
     * @return list<int>
     */
    public function findSimplex(array $queryPoints, float $tol = 1.0e-12): array
    {
        $d = $this->ndim;
        $out = [];
        foreach (array_values($queryPoints) as $q) {
            $q = array_values($q);
            $x = [];
            for ($k = 0; $k < $d; $k++) {
                $x[$k] = (float) $q[$k];
            }
            $found = -1;
            foreach ($this->transform as $s => $T) {
                $r = $T[$d];
                $sum = 0.0;
                $inside = true;
                for ($i = 0; $i < $d; $i++) {
                    $ci = 0.0;
                    for ($k = 0; $k < $d; $k++) {
                        $ci += $T[$i][$k] * ($x[$k] - $r[$k]);
                    }
                    if ($ci < -$tol) {
                        $inside = false;
                        break;
                    }
                    $sum += $ci;
                }
                if ($inside && (1.0 - $sum) >= -$tol) {
                    $found = $s;
                    break;
                }
            }
            $out[] = $found;
        }

        return $out;
    }

    /**
     * scipy.spatial.tsearch(tri, xi): the simplex containing each query point, or -1 outside the
     * hull. Identical to (new Delaunay($tri->points))->findSimplex($xi).
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
