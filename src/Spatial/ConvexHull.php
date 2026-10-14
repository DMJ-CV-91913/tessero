<?php

declare(strict_types=1);

namespace Tessero\Spatial;

/**
 * scipy.spatial.ConvexHull. For 2-D point sets the hull is computed with Andrew's monotone chain
 * (CCW vertex order, arbitrary start); for d >= 3 with an incremental beneath-beyond (quickhull)
 * construction over float64 arithmetic -- no Qhull FFI.
 *
 * INVARIANTS matched to scipy/Qhull (the only well-defined outputs): ->vertices as a SET (Qhull's
 * start index and, in 2-D, winding are arbitrary), ->volume (the d-dimensional measure of the solid)
 * and ->area (the (d-1)-dimensional measure of its boundary -- perimeter in 2-D, surface area in 3-D,
 * etc.). The facet/simplex triangulation (->simplices) is NON-UNIQUE: Qhull triangulates a
 * non-simplicial facet (e.g. a square face of a cube) differently from this builder, so simplices are
 * exposed for convenience but are compared only semantically (via the invariants) by the parity harness.
 *
 * Pure PHP (no kernel), identical on both backends. A point set whose columns are < 2 is rejected, as
 * is a degenerate (lower-than-full-dimensional) set, which Qhull reports as a QhullError.
 */
final class ConvexHull
{
    /**
     * Hull vertex indices into the input points. 2-D: CCW order, arbitrary start. d>=3: ascending
     * (the start/order is arbitrary in Qhull too, so callers compare the vertex SET).
     *
     * @var list<int>
     */
    public array $vertices;

    /** Boundary measure: perimeter (2-D), surface area (3-D), ... (scipy ConvexHull.area). */
    public float $area;

    /** Solid measure: polygon area (2-D), volume (3-D), ... (scipy ConvexHull.volume). */
    public float $volume;

    /**
     * Hull facets as index simplices (d indices each for d-D). NON-UNIQUE triangulation; provided
     * for convenience only. 2-D leaves this empty (the hull is the vertex polygon itself).
     *
     * @var list<list<int>>
     */
    public array $simplices = [];

    /** @var int number of input dimensions */
    public int $ndim;

    /**
     * @param list<list<float>> $points an (n, d) array of d-D points (d >= 2)
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
                throw new \InvalidArgumentException('ConvexHull: all points must have the same number of columns');
            }
            $row = [];
            foreach ($p as $c) {
                $row[] = (float) $c;
            }
            $pts[] = $row;
        }
        if ($d === null || $d < 2) {
            throw new \InvalidArgumentException('ConvexHull requires points with at least 2 columns');
        }
        $this->ndim = $d;
        $n = count($pts);
        if ($n < $d + 1) {
            throw new \InvalidArgumentException('ConvexHull requires at least d+1 points');
        }

        if ($d === 2) {
            $this->hull2d($pts);

            return;
        }

        $this->hullNd($pts, $d);
    }

    /** Andrew's monotone chain (2-D). */
    private function hull2d(array $pts): void
    {
        $n = count($pts);
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
        $lower = [];
        foreach ($idx as $i) {
            while (count($lower) >= 2 && $cross($lower[count($lower) - 2], $lower[count($lower) - 1], $i) <= 0.0) {
                array_pop($lower);
            }
            $lower[] = $i;
        }
        $upper = [];
        foreach (array_reverse($idx) as $i) {
            while (count($upper) >= 2 && $cross($upper[count($upper) - 2], $upper[count($upper) - 1], $i) <= 0.0) {
                array_pop($upper);
            }
            $upper[] = $i;
        }
        array_pop($lower);
        array_pop($upper);
        $hull = array_merge($lower, $upper);
        if (count($hull) < 3) {
            throw new \InvalidArgumentException('input is degenerate (collinear points); QhullError');
        }
        $this->vertices = $hull;
        $k = count($hull);
        $area2 = 0.0;
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

    /**
     * Incremental beneath-beyond in d >= 3 dimensions over float64.
     *
     * A facet is an (unordered) set of d point indices spanning a supporting hyperplane; it carries a
     * normal and offset oriented so the current interior point is strictly negative. Points are
     * inserted one at a time: every facet the point can "see" (positive signed distance) is removed,
     * the horizon ridges ((d-1)-subsets on the boundary of the visible region) are found, and a new
     * facet is created from each horizon ridge plus the new point.
     *
     * @param list<list<float>> $pts
     */
    private function hullNd(array $pts, int $d): void
    {
        $n = count($pts);
        $eps = 1e-12;

        $init = $this->initialSimplex($pts, $d, $n);
        if (count($init) < $d + 1) {
            throw new \InvalidArgumentException('input is degenerate (not full-dimensional); QhullError');
        }

        $interior = $this->centroid($pts, $init, $d);

        // facets keyed by sorted-index string -> ['idx' => [..d..], 'nrm' => [..d..], 'off' => float]
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
            // horizon ridges: (d-1)-subsets appearing in exactly one visible facet
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
            // recompute the interior reference from surviving hull vertices plus p
            $verts = [];
            foreach ($facets as $f) {
                foreach ($f['idx'] as $vi) {
                    $verts[$vi] = true;
                }
            }
            $verts[$p] = true;
            $interior = $this->centroid($pts, array_keys($verts), $d);
            // re-orient surviving facets toward the (possibly moved) interior
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
            // build a new facet from each horizon ridge + p
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

        $vset = [];
        $simplices = [];
        foreach ($facets as $f) {
            $simplices[] = $f['idx'];
            foreach ($f['idx'] as $vi) {
                $vset[$vi] = true;
            }
        }
        $vertices = array_keys($vset);
        sort($vertices);

        $apex = $this->centroid($pts, $vertices, $d);
        $vol = 0.0;
        $area = 0.0;
        $factDm1 = $this->factorial($d - 1);
        foreach ($facets as $f) {
            $simplex = $f['idx'];
            $base = $pts[$simplex[0]];
            // tangent matrix M is (d-1) x d
            $M = [];
            for ($i = 1; $i < $d; $i++) {
                $row = [];
                for ($k = 0; $k < $d; $k++) {
                    $row[] = $pts[$simplex[$i]][$k] - $base[$k];
                }
                $M[] = $row;
            }
            // Gram matrix G = M M^T ((d-1)x(d-1)); facet measure = sqrt(det G) / (d-1)!
            $G = [];
            for ($i = 0; $i < $d - 1; $i++) {
                $Grow = [];
                for ($j = 0; $j < $d - 1; $j++) {
                    $s = 0.0;
                    for ($k = 0; $k < $d; $k++) {
                        $s += $M[$i][$k] * $M[$j][$k];
                    }
                    $Grow[] = $s;
                }
                $G[] = $Grow;
            }
            $detG = $this->det($G, $d - 1);
            $facetMeasure = ($detG > 0.0 ? sqrt($detG) : 0.0) / $factDm1;
            $area += $facetMeasure;
            // cone height from apex to the facet hyperplane
            $nrm = $f['nrm'];
            $nn = 0.0;
            $sd = $f['off'];
            for ($k = 0; $k < $d; $k++) {
                $nn += $nrm[$k] * $nrm[$k];
                $sd += $nrm[$k] * $apex[$k];
            }
            $height = $nn > 0.0 ? abs($sd) / sqrt($nn) : 0.0;
            $vol += $facetMeasure * $height / $d;
        }

        $this->vertices = $vertices;
        $this->simplices = $simplices;
        $this->volume = $vol;
        $this->area = $area;
    }

    /** Greedily pick d+1 affinely-independent point indices. */
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

    /** Affine rank of a set of indices: rank of the (p_i - p_0) tangent matrix via elimination. */
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

    /** Numerical rank of an (r x c) matrix via Gaussian elimination with partial pivoting. */
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
     * Supporting hyperplane of a d-point facet: normal spanning the null space of the (d-1) tangent
     * vectors via the generalized cross product (cofactor expansion), oriented so `interior` is
     * strictly negative.
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
        // n_k = (-1)^k * det(T with column k removed)
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

    private function factorial(int $k): float
    {
        $f = 1.0;
        for ($i = 2; $i <= $k; $i++) {
            $f *= $i;
        }

        return $f;
    }

    /**
     * All r-combinations of the given index list.
     *
     * @param list<int> $items
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
}
