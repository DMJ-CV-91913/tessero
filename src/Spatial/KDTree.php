<?php

declare(strict_types=1);

namespace Tessero\Spatial;

/**
 * A k-d tree for nearest-neighbour queries (scipy.spatial.cKDTree / KDTree). The query is exact Euclidean
 * nearest-neighbour search; results (distances and point indices) match SciPy's. The tree is built once, in the
 * constructor as SciPy builds it, and every query searches it; both run in the kernel (distance.kdtree_query: a k-d
 * tree with branch-and-bound, nearest first, ties by index) through the native extension when it is loaded, else
 * through FFI, so the class is the same on both backends.
 *
 * @phpstan-type Point list<float>
 */
final class KDTree
{
    /** @var list<list<float>> */
    private array $pts;

    /** the points as a kernel array */
    private ?object $arr = null;

    /** the built tree: the idx, nodes and box arrays of distance.kdtree_build (null without points) */
    private ?array $tree = null;

    /** @param list<list<float|int>> $data the points, one row per point */
    public function __construct(array $data)
    {
        $this->pts = array_map(static fn (array $p): array => array_map('floatval', array_values($p)), array_values($data));
        if ($this->pts === []) {
            return;                                    /* a query then reports that the tree has no points */
        }
        [$array, , $build] = self::backend();
        $this->arr = $array::array($this->pts, 'float64');
        $r = $build($this->arr);
        $this->tree = [$r['idx'], $r['nodes'], $r['box']];
    }

    /** @return array{0: class-string, 1: callable, 2: callable} the NDArray class, kdtree_query and kdtree_build of the active backend */
    private static function backend(): array
    {
        if (extension_loaded('tessero') && class_exists(\Tessero\Ext\Distance::class)) {
            return [\Tessero\Ext\NDArray::class, [\Tessero\Ext\Distance::class, 'kdtreeQuery'], [\Tessero\Ext\Distance::class, 'kdtreeBuild']];
        }

        return [\Tessero\NDArray::class, [\Tessero\Distance::class, 'kdtreeQuery'], [\Tessero\Distance::class, 'kdtreeBuild']];
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
        [$array, $query] = self::backend();
        $this->arr ??= $array::array($this->pts, 'float64');
        $q = $array::array(array_map(static fn (array $p): array => array_map('floatval', array_values($p)), array_values($x)), 'float64');
        $r = $this->tree === null ? $query($this->arr, $q, $k) : $query($this->arr, $q, $k, ...$this->tree);
        $d = $r['d']->toArray();
        $i = $r['i']->toArray();
        if ($k === 1) {
            return [array_column($d, 0), array_column($i, 0)];
        }

        return [$d, $i];
    }
}
