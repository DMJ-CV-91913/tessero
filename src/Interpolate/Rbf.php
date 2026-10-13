<?php

declare(strict_types=1);

namespace Tessero\Interpolate;

use Tessero\Linalg\Linalg;

/**
 * Radial basis function interpolation of scattered data (scipy.interpolate.Rbf, scalar target values).
 *
 * Faithful reimplementation of SciPy's Rbf (scipy/interpolate/_rbf.py): with n data sites p_i in R^dim and
 * scalar values d_i, the interpolant is  s(x) = sum_j nodes_j * phi(||x - p_j||),  where the node weights solve
 * the dense linear system  (A - smooth*I) nodes = d,  A_ij = phi(||p_i - p_j||). The default shape parameter
 * epsilon is the bounding-hypercube average node spacing  epsilon = (prod(nonzero edge lengths)/n)^(1/#edges),
 * and the default kernel is multiquadric. The distance is Euclidean. This is a deterministic dense solve, so it
 * is a backend-neutral pure-PHP facade over the existing Tessero\Linalg\Linalg::solve kernel (no new kernel
 * code). Verified machine-precision vs SciPy 1.17.1 across all kernels, 1-D/N-D sites, and smooth>=0
 * (VERIFY_Rbf.py).
 *
 * Supported kernels (SciPy names): multiquadric, inverse (= inverse_multiquadric), gaussian, linear, cubic,
 * quintic, thin_plate. User-callable kernels and the vector-valued (N-D) target mode are out of scope; this
 * facade targets scalar values d (SciPy's default scalar mode).
 */
final class Rbf
{
    /** @var list<list<float>> site coordinates, dim-major: coords[k][i] = k-th coordinate of site i */
    private array $coords;

    private int $dim;

    private int $n;

    private float $epsilon;

    private string $function;

    /** @var list<float> solved node weights */
    private array $nodes;

    /**
     * @param list<list<float|int>> $coords dim coordinate arrays (each length n); coords[k][i] is the k-th
     *                                       coordinate of site i. For 1-D data pass a single array: [[x0,x1,...]]
     * @param list<float|int>       $d       scalar value at each site (length n)
     * @param string                $function kernel name
     * @param float|null            $epsilon  shape parameter; null -> SciPy default (bounding-box avg spacing)
     * @param float                 $smooth   smoothing (0 = interpolation)
     */
    public function __construct(
        array $coords,
        array $d,
        string $function = 'multiquadric',
        ?float $epsilon = null,
        float $smooth = 0.0
    ) {
        $cs = [];
        foreach (array_values($coords) as $c) {
            $cs[] = array_map('floatval', array_values($c));
        }
        $dim = count($cs);
        if ($dim < 1) {
            throw new \ValueError('Rbf: need at least one coordinate dimension');
        }
        $n = count($cs[0]);
        foreach ($cs as $c) {
            if (count($c) !== $n) {
                throw new \ValueError('Rbf: all coordinate arrays must be equal length');
            }
        }
        $dv = array_map('floatval', array_values($d));
        if (count($dv) !== $n) {
            throw new \ValueError('Rbf: values d must have the same length as the sites');
        }
        if ($n < 1) {
            throw new \ValueError('Rbf: need at least one data site');
        }

        $function = strtolower($function);
        if ($function === 'inverse_multiquadric') {
            $function = 'inverse';
        }
        if (! in_array($function, ['multiquadric', 'inverse', 'gaussian', 'linear', 'cubic', 'quintic', 'thin_plate'], true)) {
            throw new \ValueError("Rbf: unsupported kernel '{$function}'");
        }

        // Default epsilon: (prod of nonzero bounding-box edge lengths / n) ^ (1 / #nonzero edges).
        if ($epsilon === null) {
            $edges = [];
            foreach ($cs as $c) {
                $e = max($c) - min($c);
                if ($e != 0.0) {
                    $edges[] = $e;
                }
            }
            if (count($edges) === 0) {
                throw new \ValueError('Rbf: all sites coincide; cannot derive a default epsilon');
            }
            $prod = 1.0;
            foreach ($edges as $e) {
                $prod *= $e;
            }
            $epsilon = ($prod / $n) ** (1.0 / count($edges));
        }

        $this->coords = $cs;
        $this->dim = $dim;
        $this->n = $n;
        $this->epsilon = $epsilon;
        $this->function = $function;

        // Build A_ij = phi(||p_i - p_j||) - smooth*I, solve A nodes = d.
        $A = [];
        for ($i = 0; $i < $n; $i++) {
            $row = [];
            for ($j = 0; $j < $n; $j++) {
                $v = $this->phi($this->dist($cs, $i, $cs, $j));
                if ($i === $j) {
                    $v -= $smooth;
                }
                $row[] = $v;
            }
            $A[] = $row;
        }
        $this->nodes = $this->solveDense($A, $dv);
    }

    /** Euclidean distance between site i of set $a and site j of set $b (both dim-major coord lists). */
    private function dist(array $a, int $i, array $b, int $j): float
    {
        $s = 0.0;
        for ($k = 0; $k < $this->dim; $k++) {
            $dk = $a[$k][$i] - $b[$k][$j];
            $s += $dk * $dk;
        }
        return sqrt($s);
    }

    private function phi(float $r): float
    {
        $eps = $this->epsilon;
        return match ($this->function) {
            'multiquadric' => sqrt(($r / $eps) ** 2 + 1.0),
            'inverse' => 1.0 / sqrt(($r / $eps) ** 2 + 1.0),
            'gaussian' => exp(-(($r / $eps) ** 2)),
            'linear' => $r,
            'cubic' => $r ** 3,
            'quintic' => $r ** 5,
            'thin_plate' => $r == 0.0 ? 0.0 : $r * $r * log($r),  // xlogy(r^2, r)
        };
    }

    /**
     * Solve A x = b via the Tessero\Linalg\Linalg::solve kernel; normalise the result to a flat list<float>.
     *
     * @param list<list<float>> $A
     * @param list<float>       $b
     * @return list<float>
     */
    private function solveDense(array $A, array $b): array
    {
        $res = Linalg::solve($A, $b);
        $out = is_object($res) && method_exists($res, 'toArray') ? $res->toArray() : $res;
        $flat = [];
        $arr = is_array($out) ? $out : [$out];
        array_walk_recursive($arr, static function ($v) use (&$flat): void {
            $flat[] = (float) $v;
        });
        return $flat;
    }

    /**
     * Evaluate the interpolant at query sites (dim-major, same convention as the constructor).
     *
     * @param list<list<float|int>> $coords dim query-coordinate arrays (each length nq)
     * @return list<float> interpolated value at each query site
     */
    public function evaluate(array $coords): array
    {
        $qs = [];
        foreach (array_values($coords) as $c) {
            $qs[] = array_map('floatval', array_values($c));
        }
        if (count($qs) !== $this->dim) {
            throw new \ValueError('Rbf: query must have the same number of coordinate dimensions as the sites');
        }
        $nq = count($qs[0]);
        foreach ($qs as $c) {
            if (count($c) !== $nq) {
                throw new \ValueError('Rbf: all query coordinate arrays must be equal length');
            }
        }
        $out = [];
        for ($q = 0; $q < $nq; $q++) {
            $s = 0.0;
            for ($j = 0; $j < $this->n; $j++) {
                $s += $this->nodes[$j] * $this->phi($this->dist($qs, $q, $this->coords, $j));
            }
            $out[] = $s;
        }
        return $out;
    }

    /**
     * @param list<list<float|int>> $coords
     * @return list<float>
     */
    public function __invoke(array $coords): array
    {
        return $this->evaluate($coords);
    }
}
