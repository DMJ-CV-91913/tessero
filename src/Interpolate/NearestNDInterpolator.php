<?php

declare(strict_types=1);

namespace Tessero\Interpolate;

use Tessero\Spatial\KDTree;

/**
 * Nearest-neighbour interpolation in N > 1 dimensions (scipy.interpolate.NearestNDInterpolator): the value of
 * the nearest data point is returned at each query point. Pure PHP over Tessero\Spatial\KDTree (exact Euclidean
 * nearest neighbour, lowest-index tie-break), so it is backend-neutral and bit-exact (a gather, no arithmetic
 * on the values). Mirrors SciPy: the input points must be at least 2-D ((npoints, ndim), ndim >= 2).
 */
final class NearestNDInterpolator
{
    /** @var list<list<float>> the (possibly rescaled) data points, one row per point */
    private array $points;

    /** @var list<mixed> one value per point: a float, or a list<float> for vector-valued data */
    private array $values;

    private KDTree $tree;

    private int $ndim;

    private bool $rescale;

    /** @var list<float> per-axis scale (ptp) used when rescale is true */
    private array $scale = [];

    /**
     * @param list<list<float|int>> $points (npoints x ndim) data-point coordinates, ndim >= 2
     * @param list<mixed>           $values one value per point: float, or list<float> (vector-valued)
     * @param bool                  $rescale rescale points to unit scale (divide each axis by its range) before
     *                                       building the tree, as SciPy's rescale=True does
     */
    public function __construct(array $points, array $values, bool $rescale = false)
    {
        $pts = array_map(static fn (array $p): array => array_map('floatval', array_values($p)), array_values($points));
        if (count($pts) === 0) {
            throw new \ValueError('NearestNDInterpolator: points must be non-empty');
        }
        $this->ndim = count($pts[0]);
        if ($this->ndim < 2) {
            throw new \ValueError('input data must be at least 2-D');
        }
        $this->rescale = $rescale;
        if ($rescale) {
            $mins = $pts[0];
            $maxs = $pts[0];
            foreach ($pts as $p) {
                for ($j = 0; $j < $this->ndim; $j++) {
                    if ($p[$j] < $mins[$j]) { $mins[$j] = $p[$j]; }
                    if ($p[$j] > $maxs[$j]) { $maxs[$j] = $p[$j]; }
                }
            }
            for ($j = 0; $j < $this->ndim; $j++) {
                $r = $maxs[$j] - $mins[$j];
                $this->scale[$j] = $r > 0.0 ? $r : 1.0;   // SciPy: ptp per axis, zero range -> 1
            }
            $pts = array_map(fn (array $p): array => $this->scaleRow($p), $pts);
        }
        $this->points = $pts;
        $this->values = array_values($values);
        $this->tree = new KDTree($pts);
    }

    /** @param list<float> $p @return list<float> */
    private function scaleRow(array $p): array
    {
        if (! $this->rescale) {
            return $p;
        }
        $o = [];
        for ($j = 0; $j < $this->ndim; $j++) {
            $o[$j] = $p[$j] / $this->scale[$j];
        }
        return $o;
    }

    /**
     * Nearest-neighbour value(s) at the query point(s).
     *
     * @param list<list<float|int>> $xi query points (m x ndim)
     * @return list<mixed> one value per query point (float, or list<float> for vector-valued data)
     */
    public function query(array $xi): array
    {
        $q = array_map(static fn (array $p): array => array_map('floatval', array_values($p)), array_values($xi));
        if ($this->rescale) {
            $q = array_map(fn (array $p): array => $this->scaleRow($p), $q);
        }
        [, $idxs] = $this->tree->query($q, 1);        // k = 1 -> flat list of nearest indices
        $out = [];
        foreach ($idxs as $i) {
            $out[] = $this->values[$i];
        }
        return $out;
    }

    /**
     * Callable form, matching scipy's NearestNDInterpolator(xi).
     *
     * @param list<list<float|int>> $xi
     * @return list<mixed>
     */
    public function __invoke(array $xi): array
    {
        return $this->query($xi);
    }
}
