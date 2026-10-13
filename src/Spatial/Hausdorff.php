<?php

declare(strict_types=1);

namespace Tessero\Spatial;

/**
 * scipy.spatial.distance.directed_hausdorff: the directed Hausdorff distance from point set u to point set v,
 * computed with the Taha-Hanbury early-break scan. The distance is exact and independent of scipy's internal
 * random shuffle (the shuffle only affects which index pair is reported on ties and how early the inner loop
 * breaks), so the index pair is NOT reproduced here. Pure PHP (no kernel), identical on both backends.
 */
final class Hausdorff
{
    /**
     * @param list<list<float>> $u
     * @param list<list<float>> $v
     *
     * @return array{0: float, 1: int, 2: int} [distance, index_u, index_v]
     */
    public static function directed(array $u, array $v): array
    {
        $u = array_values($u);
        $v = array_values($v);
        $nu = count($u);
        $nv = count($v);
        if ($nu === 0 || $nv === 0) {
            throw new \InvalidArgumentException('directed_hausdorff requires non-empty point sets');
        }
        $dim = count($u[0]);
        if ($dim !== count($v[0])) {
            throw new \InvalidArgumentException('u and v must have the same number of columns');
        }

        $cmax = 0.0;
        $index1 = 0;
        $index2 = 0;
        $inf = INF;
        for ($i = 0; $i < $nu; $i++) {
            $ui = $u[$i];
            $cmin = $inf;
            $bestJ = 0;
            for ($j = 0; $j < $nv; $j++) {
                $vj = $v[$j];
                $d = 0.0;
                for ($k = 0; $k < $dim; $k++) {
                    $diff = (float) $ui[$k] - (float) $vj[$k];
                    $d += $diff * $diff;
                }
                if ($d < $cmax) {   // this i can no longer raise the running max -> early break
                    $cmin = 0.0;
                    break;
                }
                if ($d < $cmin) {
                    $cmin = $d;
                    $bestJ = $j;
                }
            }
            if ($cmin !== $inf && $cmin > $cmax && $cmin !== 0.0) {
                $cmax = $cmin;
                $index1 = $i;
                $index2 = $bestJ;
            }
        }

        return [sqrt($cmax), $index1, $index2];
    }
}
