<?php

declare(strict_types=1);

namespace Tessero\Sparse\Linalg;

/**
 * scipy.sparse.linalg.aslinearoperator -- wrap an array / sparse matrix / object
 * with a .dot into a LinearOperator (pure-PHP facade).
 *
 * Accepts:
 *   - a dense 2-D matrix (list<list<float>>),
 *   - a CSR triple ['data'=>..., 'indices'=>..., 'indptr'=>..., 'shape'=>[m,n]],
 *   - an existing LinearOperator (returned as-is),
 *   - a Tessero\Sparse\CsrMatrix-like object exposing toArray()/toDense().
 */
final class Aslinearoperator
{
    private function __construct()
    {
    }

    public static function aslinearoperator(mixed $a): LinearOperator
    {
        if ($a instanceof LinearOperator) {
            return $a;
        }
        if ($a instanceof LaplacianNd) {
            return $a->asLinearOperator();
        }
        if (is_object($a)) {
            if (method_exists($a, 'toArray')) {
                return LinearOperator::fromMatrix(self::toRows($a->toArray()));
            }
            if (method_exists($a, 'toDense')) {
                return LinearOperator::fromMatrix(self::toRows($a->toDense()));
            }
            throw new \InvalidArgumentException('aslinearoperator: unsupported object (no toArray/toDense)');
        }
        if (is_array($a)) {
            // CSR dict form?
            if (isset($a['data'], $a['indices'], $a['indptr'], $a['shape'])) {
                return LinearOperator::fromMatrix(self::csrToDense($a));
            }
            // dense 2-D list
            return LinearOperator::fromMatrix(self::toRows($a));
        }
        throw new \InvalidArgumentException('aslinearoperator: unsupported input type');
    }

    /**
     * @return list<list<float>>
     */
    private static function toRows(mixed $m): array
    {
        $rows = [];
        foreach ((array) $m as $row) {
            $rows[] = array_map('floatval', array_values((array) $row));
        }
        return $rows;
    }

    /**
     * @param array{data:list<float>,indices:list<int>,indptr:list<int>,shape:array{0:int,1:int}} $csr
     * @return list<list<float>>
     */
    private static function csrToDense(array $csr): array
    {
        $m = (int) $csr['shape'][0];
        $n = (int) $csr['shape'][1];
        $data = array_values($csr['data']);
        $indices = array_values($csr['indices']);
        $indptr = array_values($csr['indptr']);
        $dense = [];
        for ($i = 0; $i < $m; $i++) {
            $dense[] = array_fill(0, $n, 0.0);
        }
        for ($i = 0; $i < $m; $i++) {
            $start = (int) $indptr[$i];
            $end = (int) $indptr[$i + 1];
            for ($k = $start; $k < $end; $k++) {
                $dense[$i][(int) $indices[$k]] += (float) $data[$k];
            }
        }
        return $dense;
    }
}
