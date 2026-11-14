<?php

declare(strict_types=1);

namespace Tessero\Spatial;

use Tessero\Exceptions\ShapeError;

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
    /** the number of points */
    private int $n = 0;

    /** the points as a kernel array */
    private ?object $arr = null;

    /** the built tree: the idx, nodes and box arrays of distance.kdtree_build (null without points) */
    private ?array $tree = null;

    /** @param list<list<float|int>>|object $data the points, one row per point, or an (n, d) NDArray of either backend */
    public function __construct(array|object $data)
    {
        if ($data === []) {
            return;                                    /* a query then reports that the tree has no points */
        }
        [, , $build] = self::backend();
        $arr = self::rows($data);
        $this->n = $arr->shape()[0];
        if ($this->n === 0) {
            return;
        }
        $this->arr = $arr;
        $r = $build($this->arr);
        $this->tree = [$r['idx'], $r['nodes'], $r['box']];
    }

    /**
     * Points as an (n, d) float64 kernel array of the active backend. An NDArray of that backend is used as it is
     * (cast to float64 if it is not), one of the other backend converted. Rows of numbers: through the extension
     * converted in C in one call (the doubles (float) gives); otherwise, or when that refuses them, each value cast
     * as floatval does, in one loop into a flat list (a row that is no array is a TypeError, rows of unequal length a
     * ShapeError).
     *
     * @param array<array<float|int>>|object $rows
     */
    private static function rows(array|object $rows): object
    {
        [$nd] = self::backend();
        if (is_object($rows)) {
            $a = $rows instanceof $nd ? $rows : $nd::array($rows->toArray(), 'float64');
            if ($a->ndim() !== 2) {
                throw new ShapeError('KDTree: the points must be an (n, d) array.');
            }

            $dt = $a->dtype();                          // a DType (FFI) or its name (extension)

            return $dt === 'float64' || $dt === \Tessero\DType::Float64 ? $a : $a->astype('float64');
        }
        if ($nd === \Tessero\Ext\NDArray::class) {
            try {
                $a = \Tessero\Ext\NDArray::array($rows, 'float64');
                if ($a->ndim() === 2 && $a->shape()[0] === count($rows)) {
                    return $a;
                }
            } catch (\Throwable) {
                // values the extension refuses (numeric strings, null, ragged rows): the loop below casts or reports them
            }
        }
        $d = null;
        $lists = true;
        foreach ($rows as $p) {
            $c = count($p);
            if ($d === null) {
                $d = $c;
            } elseif ($c !== $d) {
                throw new ShapeError("Ragged nested array: every row must have the same length (expected {$d}).");
            }
            $lists = $lists && array_is_list($p);
        }
        if ($lists) {
            // rows that are lists are concatenated in one C call; the float64 packing below converts each value as
            // the (float) cast does, so the array holds the same doubles
            $flat = array_merge(...array_values($rows));
        } else {
            $flat = [];
            foreach ($rows as $p) {
                foreach ($p as $v) {
                    $flat[] = (float) $v;
                }
            }
        }
        $n = count($rows);
        if (extension_loaded('tessero') && class_exists(\Tessero\Ext\NDArray::class)) {
            try {
                return \Tessero\Ext\NDArray::array($flat, 'float64')->reshape([$n, (int) $d]);
            } catch (\Throwable $e) {
                if (!$lists) {
                    throw $e;
                }
                // the extension takes numbers only: values the (float) cast accepts (numeric strings, null) are cast
                return \Tessero\Ext\NDArray::array(array_map(floatval(...), $flat), 'float64')->reshape([$n, (int) $d]);
            }
        }

        return \Tessero\NDArray::fromFlat($flat, [$n, (int) $d], 'float64');
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
        return $this->n;
    }

    /**
     * The k nearest neighbours of each query point: [distances, indices]. For k == 1 each is a flat list (one
     * value per query point); for k > 1 each is a list of min(k, count()) -length rows, as
     * scipy.spatial.cKDTree.query returns.
     *
     * @param list<list<float|int>>|object $x query points (one row each), or an (m, d) NDArray of either backend
     * @return array{0: mixed, 1: mixed}
     */
    public function query(array|object $x, int $k = 1): array
    {
        if ($x === [] || (is_object($x) && $x->size() === 0)) {
            return [[], []];
        }
        [$array, $query] = self::backend();
        $this->arr ??= $array::array([], 'float64');
        $q = self::rows($x);
        $r = $this->tree === null ? $query($this->arr, $q, $k) : $query($this->arr, $q, $k, ...$this->tree);
        $d = $r['d']->toArray();
        $i = $r['i']->toArray();
        if ($k === 1) {
            return [array_column($d, 0), array_column($i, 0)];
        }

        return [$d, $i];
    }
}
