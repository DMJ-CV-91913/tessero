<?php

declare(strict_types=1);

namespace Tessero\Spatial;

/**
 * A k-d tree for nearest-neighbour queries (scipy.spatial.cKDTree / KDTree). The query is exact Euclidean
 * nearest-neighbour search; results (distances and point indices) match SciPy's. The search runs in the kernel
 * (distance.kdtree_query: a k-d tree with branch-and-bound, nearest first, ties by index) through the native
 * extension when it is loaded, else through FFI, so the class is the same on both backends.
 *
 * @phpstan-type Point list<float>
 */
final class KDTree
{
    /** @var list<list<float>> */
    private array $pts;

    /** the points as a kernel array, made on the first query and kept */
    private ?object $arr = null;

    /** @param list<list<float|int>> $data the points, one row per point */
    public function __construct(array $data)
    {
        $this->pts = array_map(static fn (array $p): array => array_map('floatval', array_values($p)), array_values($data));
    }

    /** The number of points in the tree. */
    public function count(): int
    {
        return count($this->pts);
    }

    /**
     * The k nearest neighbours of each query point: [distances, indices]. For k == 1 each is a flat list (one
     * value per query point); for k > 1 each is a list of min(k, count()) -length rows, as
     * scipy.spatial.cKDTree.query returns.
     *
     * @param list<list<float|int>> $x query points (one row each)
     * @return array{0: mixed, 1: mixed}
     */
    public function query(array $x, int $k = 1): array
    {
        if ($x === []) {
            return [[], []];
        }
        $ext = extension_loaded('tessero') && class_exists(\Tessero\Ext\Distance::class);
        $array = $ext ? \Tessero\Ext\NDArray::class : \Tessero\NDArray::class;
        $this->arr ??= $array::array($this->pts, 'float64');
        $q = $array::array(array_map(static fn (array $p): array => array_map('floatval', array_values($p)), array_values($x)), 'float64');
        $r = $ext ? \Tessero\Ext\Distance::kdtreeQuery($this->arr, $q, $k) : \Tessero\Distance::kdtreeQuery($this->arr, $q, $k);
        $d = $r['d']->toArray();
        $i = $r['i']->toArray();
        if ($k === 1) {
            return [array_column($d, 0), array_column($i, 0)];
        }

        return [$d, $i];
    }
}
