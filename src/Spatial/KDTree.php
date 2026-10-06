<?php

declare(strict_types=1);

namespace Tessero\Spatial;

/**
 * A k-d tree for nearest-neighbour queries (scipy.spatial.cKDTree / KDTree). The query is exact Euclidean
 * nearest-neighbour search; results (distances and point indices) match SciPy's. Pure PHP, so it is
 * backend-neutral. The current implementation stores the points and answers queries by exhaustive search,
 * which returns identical neighbours to SciPy's tree for the sizes used in practice here.
 *
 * @phpstan-type Point list<float>
 */
final class KDTree
{
    /** @var list<list<float>> */
    private array $pts;

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
     * value per query point); for k > 1 each is a list of k-length rows, as scipy.spatial.cKDTree.query returns.
     *
     * @param list<list<float|int>> $x query points (one row each)
     * @return array{0: mixed, 1: mixed}
     */
    public function query(array $x, int $k = 1): array
    {
        $dists = [];
        $idxs = [];
        foreach ($x as $q) {
            $q = array_map('floatval', array_values($q));
            $dd = [];
            foreach ($this->pts as $i => $p) {
                $s = 0.0;
                foreach ($p as $j => $pj) {
                    $d = $pj - ($q[$j] ?? 0.0);
                    $s += $d * $d;
                }
                $dd[] = [sqrt($s), $i];
            }
            usort($dd, static fn (array $a, array $b): int => ($a[0] <=> $b[0]) ?: ($a[1] <=> $b[1]));
            if ($k === 1) {
                $dists[] = $dd[0][0];
                $idxs[] = $dd[0][1];
            } else {
                $rd = [];
                $ri = [];
                for ($t = 0; $t < $k && $t < count($dd); $t++) {
                    $rd[] = $dd[$t][0];
                    $ri[] = $dd[$t][1];
                }
                $dists[] = $rd;
                $idxs[] = $ri;
            }
        }
        return [$dists, $idxs];
    }
}
