<?php

declare(strict_types=1);

namespace Tessero\Integrate;

/**
 * Adaptive Gauss-Kronrod quadrature (scipy.integrate.quad). A global adaptive scheme over the 7-15 Gauss-Kronrod
 * rule (QUADPACK's QK15), subdividing the sub-interval of largest error until the estimate meets the tolerance.
 * The returned integral matches scipy's value to the requested tolerance; the error estimate is Tessero's own.
 */
final class Quad
{
    // 7-15 Gauss-Kronrod abscissae (xgk), Kronrod weights (wgk), Gauss weights (wg)
    private const XGK = [0.991455371120813, 0.949107912342759, 0.864864423359769, 0.741531185599394,
        0.586087235467691, 0.405845151377397, 0.207784955007898, 0.000000000000000];
    private const WGK = [0.022935322010529, 0.063092092629979, 0.104790010322250, 0.140653259715525,
        0.169004726639267, 0.190350578064785, 0.204432940075298, 0.209482141084728];
    private const WG = [0.129484966168870, 0.279705391489277, 0.381830050505119, 0.417959183673469];

    /**
     * Integrate $f over [$a, $b].
     *
     * @param callable(float): float $f
     * @return array{0: float, 1: float}  [value, abserr] (as scipy.integrate.quad's first two returns)
     */
    public static function quad(callable $f, float $a, float $b, float $epsabs = 1.49e-8, float $epsrel = 1.49e-8, int $limit = 50): array
    {
        [$total, $toterr] = self::qk15($f, $a, $b);
        $segs = [[$a, $b, $total, $toterr]];
        for ($iter = 0; $iter < $limit; $iter++) {
            if ($toterr <= max($epsabs, $epsrel * abs($total))) {
                break;
            }
            $wi = 0;
            for ($i = 1; $i < count($segs); $i++) {
                if ($segs[$i][3] > $segs[$wi][3]) {
                    $wi = $i;
                }
            }
            [$sa, $sb, $si, $se] = $segs[$wi];
            $mid = 0.5 * ($sa + $sb);
            [$i1, $e1] = self::qk15($f, $sa, $mid);
            [$i2, $e2] = self::qk15($f, $mid, $sb);
            $total += ($i1 + $i2) - $si;
            $toterr += ($e1 + $e2) - $se;
            $segs[$wi] = [$sa, $mid, $i1, $e1];
            $segs[] = [$mid, $sb, $i2, $e2];
        }
        return [$total, $toterr];
    }

    /** One 7-15 Gauss-Kronrod panel on [$a, $b]; returns [integral, abserr]. */
    private static function qk15(callable $f, float $a, float $b): array
    {
        $c = 0.5 * ($a + $b);
        $h = 0.5 * ($b - $a);
        $fc = (float) $f($c);
        $resg = self::WG[3] * $fc;
        $resk = self::WGK[7] * $fc;
        $resabs = abs($resk);
        $fv1 = $fv2 = [];
        for ($j = 0; $j < 3; $j++) {                 // Gauss points (odd Kronrod indices)
            $jtw = 2 * $j + 1;
            $x = $h * self::XGK[$jtw];
            $f1 = (float) $f($c - $x); $f2 = (float) $f($c + $x);
            $fv1[$jtw] = $f1; $fv2[$jtw] = $f2;
            $fsum = $f1 + $f2;
            $resg += self::WG[$j] * $fsum;
            $resk += self::WGK[$jtw] * $fsum;
            $resabs += self::WGK[$jtw] * (abs($f1) + abs($f2));
        }
        for ($j = 0; $j < 4; $j++) {                 // Kronrod-only points (even indices)
            $jtwm1 = 2 * $j;
            $x = $h * self::XGK[$jtwm1];
            $f1 = (float) $f($c - $x); $f2 = (float) $f($c + $x);
            $fv1[$jtwm1] = $f1; $fv2[$jtwm1] = $f2;
            $fsum = $f1 + $f2;
            $resk += self::WGK[$jtwm1] * $fsum;
            $resabs += self::WGK[$jtwm1] * (abs($f1) + abs($f2));
        }
        $reskh = $resk * 0.5;
        $resasc = self::WGK[7] * abs($fc - $reskh);
        for ($j = 0; $j < 7; $j++) {
            $resasc += self::WGK[$j] * (abs($fv1[$j] - $reskh) + abs($fv2[$j] - $reskh));
        }
        $integral = $resk * $h;
        $resasc *= abs($h);
        $abserr = abs(($resk - $resg) * $h);
        if ($resasc != 0.0 && $abserr != 0.0) {
            $abserr = $resasc * min(1.0, (200.0 * $abserr / $resasc) ** 1.5);
        }
        return [$integral, $abserr];
    }
}
