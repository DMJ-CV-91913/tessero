<?php

declare(strict_types=1);

namespace Tessero\Optimize;

use Tessero\Exceptions\TesseroException;

/**
 * scipy.optimize.shgo with sampling_method='simplicial' (the DEFAULT), iters=1, n default
 * (-> 2**dim + 1 initial vertices). Pure PHP, deterministic.
 *
 * Builds the initial simplicial complex on the box (2**dim corners + 1 centroid) as a faithful
 * port of SciPy Complex.cyclic_product (asymmetric, order-dependent adjacency), finds the
 * minimiser pool (vertices strictly below every graph neighbour), and polishes each with
 * Tessero SLSQP (ftol=1e-12, locally-convex bounds) - SciPy's default local minimiser.
 *
 * @see scipy/optimize/_shgo.py, scipy/optimize/_shgo_lib/_complex.py
 */
final class Shgo
{
    public static function minimize(callable $func, array $bounds, string $samplingMethod = 'simplicial'): OptimizeResult
    {
        if ($samplingMethod !== 'simplicial') {
            throw new TesseroException("shgo: only sampling_method='simplicial' (the default) is supported.");
        }
        if (count($bounds) === 0) {
            throw new TesseroException('shgo: bounds must not be empty.');
        }
        $lb = [];
        $ub = [];
        foreach ($bounds as $b) {
            $lo = (float) $b[0];
            $hi = (float) $b[1];
            if (! is_finite($lo) || ! is_finite($hi) || $lo > $hi) {
                throw new TesseroException('shgo: each bound must be finite with lower <= upper.');
            }
            $lb[] = $lo;
            $ub[] = $hi;
        }
        $full = self::compute($func, $lb, $ub);

        return new OptimizeResult(
            x: $full['xl'][0],
            fun: $full['funl'][0],
            success: true,
            status: 0,
            message: 'Optimization terminated successfully.',
            nit: 1,
            nfev: $full['nfev'],
            njev: 0,
            jac: null,
            hessInv: null,
            method: 'shgo',
        );
    }

    /** @return array{x: list<float>, fun: float, xl: list<list<float>>, funl: list<float>} */
    public static function minimizeFull(callable $func, array $bounds): array
    {
        $lb = [];
        $ub = [];
        foreach ($bounds as $b) {
            $lb[] = (float) $b[0];
            $ub[] = (float) $b[1];
        }
        $full = self::compute($func, $lb, $ub);

        return ['x' => $full['xl'][0], 'fun' => $full['funl'][0], 'xl' => $full['xl'], 'funl' => $full['funl']];
    }

    /** @return array{xl: list<list<float>>, funl: list<float>, nfev: int} */
    private static function compute(callable $func, array $lb, array $ub): array
    {
        [$verts, $adj] = self::buildComplex($lb, $ub);

        $fval = [];
        foreach ($verts as $key => $coords) {
            $fval[$key] = (float) $func($coords);
        }

        $pool = [];
        foreach ($verts as $key => $coords) {
            $isMin = true;
            foreach ($adj[$key] as $nbKey => $_) {
                if (! ($fval[$key] < $fval[$nbKey])) {
                    $isMin = false;
                    break;
                }
            }
            if ($isMin) {
                $pool[] = $key;
            }
        }
        usort($pool, static fn (string $a, string $b): int => $fval[$a] <=> $fval[$b]);

        $xlList = [];
        $funlList = [];
        $seen = [];
        foreach ($pool as $key) {
            if (isset($seen[$key])) {
                continue;
            }
            $seen[$key] = true;
            $start = $verts[$key];
            $lcb = self::constructLcb($start, $adj[$key], $verts, $lb, $ub);
            $res = Minimize::minimize($func, $start, null, 'SLSQP', $lcb, ['ftol' => 1e-12]);
            $xlList[] = $res->x;
            $funlList[] = $res->fun;
        }

        $order = range(0, count($funlList) - 1);
        usort($order, static fn (int $a, int $b): int => $funlList[$a] <=> $funlList[$b]);
        $xl = [];
        $funl = [];
        foreach ($order as $i) {
            $xl[] = $xlList[$i];
            $funl[] = $funlList[$i];
        }

        return ['xl' => $xl, 'funl' => $funl, 'nfev' => count($verts)];
    }

    /** @return array{0: array<string, list<float>>, 1: array<string, array<string, bool>>} */
    private static function buildComplex(array $lb, array $ub): array
    {
        $dim = count($lb);
        $verts = [];
        $adj = [];

        $key = static function (array $coords): string {
            $parts = [];
            foreach ($coords as $c) {
                $parts[] = pack('d', (float) $c);
            }
            return implode('', $parts);
        };
        $V = static function (array $coords) use (&$verts, &$adj, $key): string {
            $coords = array_map('floatval', $coords);
            $k = $key($coords);
            if (! isset($adj[$k])) {
                $adj[$k] = [];
                $verts[$k] = $coords;
            }
            return $k;
        };
        $connect = static function (string $a, string $b) use (&$adj): void {
            if ($a !== $b) {
                $adj[$a][$b] = true;
                $adj[$b][$a] = true;
            }
        };
        $disconnect = static function (string $a, string $b) use (&$adj): void {
            if (isset($adj[$a][$b])) {
                unset($adj[$a][$b], $adj[$b][$a]);
            }
        };

        $origin = $lb;
        $supremum = $ub;
        $vot = $V($origin);
        $vut = $V($supremum);
        $connect($vut, $vot);

        $C0x = [[$vot]];
        $aVo = $origin;
        $aVo[0] = $supremum[0];
        $aVoK = $V($aVo);
        $connect($vot, $aVoK);
        $C1x = [[$aVoK]];
        $abC = [];

        for ($i = 0; $i < $dim - 1; $i++) {
            $C0x[] = [];
            $C1x[] = [];
            $cC0x = [];
            $cC1x = [];
            for ($r = 0; $r <= $i; $r++) {
                $cC0x[] = $C0x[$r];
                $cC1x[] = $C1x[$r];
            }
            foreach ($cC0x as $j => $VL) {
                $VU = $cC1x[$j];
                $cnt = min(count($VL), count($VU));
                for ($k = 0; $k < $cnt; $k++) {
                    $vlK = $VL[$k];
                    $vuK = $VU[$k];
                    $aVl = $verts[$vlK];
                    $aVu = $verts[$vuK];
                    $aVl[$i + 1] = $supremum[$i + 1];
                    $aVu[$i + 1] = $supremum[$i + 1];
                    $aVlK = $V($aVl);
                    $connect($vlK, $aVlK);
                    $aVuK = $V($aVu);
                    $connect($vuK, $aVuK);
                    $connect($aVlK, $aVuK);
                    $connect($vlK, $aVuK);
                    $abC[] = [$vlK, $aVuK];
                    $C0x[$i + 1][] = $vlK;
                    $C0x[$i + 1][] = $vuK;
                    $C1x[$i + 1][] = $aVlK;
                    $C1x[$i + 1][] = $aVuK;
                    $C0x[$j][] = $aVlK;
                    $C1x[$j][] = $aVuK;
                }
            }
            $abCc = $abC;
            foreach ($abCc as $vp) {
                $bV = $verts[$vp[0]];
                $abV = $verts[$vp[1]];
                $bV[$i + 1] = $supremum[$i + 1];
                $abV[$i + 1] = $supremum[$i + 1];
                $bVK = $V($bV);
                $abVK = $V($abV);
                $connect($vp[0], $abVK);
                $connect($bVK, $abVK);
                $abC[] = [$vp[0], $abVK];
                $abC[] = [$bVK, $abVK];
            }
        }

        $disconnect($vot, $vut);
        $vc = [];
        for ($d = 0; $d < $dim; $d++) {
            $vc[] = ($verts[$vot][$d] + $verts[$vut][$d]) / 2.0;
        }
        $vcK = $V($vc);
        $connect($vot, $vcK);
        $connect($vut, $vcK);
        foreach (array_keys($adj[$vot]) as $nbK) {
            $connect($nbK, $vcK);
        }

        return [$verts, $adj];
    }

    /** @return list<array{0: float, 1: float}> */
    private static function constructLcb(array $v, array $nbKeys, array $verts, array $lb, array $ub): array
    {
        $dim = count($v);
        $cb = [];
        for ($i = 0; $i < $dim; $i++) {
            $cb[$i] = [$lb[$i], $ub[$i]];
        }
        foreach (array_keys($nbKeys) as $nbK) {
            $nb = $verts[$nbK];
            for ($i = 0; $i < $dim; $i++) {
                $xi = $nb[$i];
                if ($xi < $v[$i] && $xi > $cb[$i][0]) {
                    $cb[$i][0] = $xi;
                }
                if ($xi > $v[$i] && $xi < $cb[$i][1]) {
                    $cb[$i][1] = $xi;
                }
            }
        }

        return $cb;
    }
}