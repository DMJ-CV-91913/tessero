<?php

declare(strict_types=1);

namespace Tessero\Integrate;

/**
 * Adaptive Gauss-Kronrod quadrature (scipy.integrate.quad on a finite interval): QUADPACK's DQAGSE, the routine
 * scipy calls there, ported step for step from scipy's C translation (scipy/integrate/__quadpack.h): the 10-21
 * Gauss-Kronrod rule (DQK21), bisection of the sub-interval of largest error (DQPSRT keeps them ordered) and the
 * epsilon algorithm (DQELG) to extrapolate past endpoint singularities. The same operations on the same numbers, so
 * the value and error estimate are scipy's, from the same number of integrand calls. (Before 13 Nov 2026: a 7-15
 * rule without extrapolation, with its own error estimate.) scipy's warnings for an abnormal end (ier 1 to 5) are not
 * raised; the value and estimate returned are the ones scipy returns with them.
 */
final class Quad
{
    // 10-21 Gauss-Kronrod abscissae (xgk), Kronrod weights (wgk) and Gauss weights (wg), as scipy's dqk21
    private const XGK = [0.995657163025808080735527280689003, 0.973906528517171720077964012084452,
        0.930157491355708226001207180059508, 0.865063366688984510732096688423493,
        0.780817726586416897063717578345042, 0.679409568299024406234327365114874,
        0.562757134668604683339000099272694, 0.433395394129247190799265943165784,
        0.294392862701460198131126603103866, 0.148874338981631210884826001129720, 0.0];
    private const WGK = [0.011694638867371874278064396062192, 0.032558162307964727478818972459390,
        0.054755896574351996031381300244580, 0.075039674810919952767043140916190,
        0.093125454583697605535065465083366, 0.109387158802297641899210590325805,
        0.123491976262065851077958109831074, 0.134709217311473325928054001771707,
        0.142775938577060080797094273138717, 0.147739104901338491374841515972068,
        0.149445554002916905664936468389821];
    private const WG = [0.066671344308688137593568809893332, 0.149451349150580593145776339657697,
        0.219086362515982043995534934228163, 0.269266719309996355091226921569469,
        0.295524224714752870173892994651338];
    private const EPMACH = 2.220446049250313e-16;   // np.finfo(np.float64).eps
    private const UFLOW = 2.2250738585072014e-308;  // np.finfo(np.float64).tiny
    private const OFLOW = 1.7976931348623157e308;   // np.finfo(np.float64).max

    /**
     * Integrate $f over [$a, $b].
     *
     * @param callable(float): float $f
     * @return array{0: float, 1: float}  [value, abserr] (as scipy.integrate.quad's first two returns)
     */
    public static function quad(callable $f, float $a, float $b, float $epsabs = 1.49e-8, float $epsrel = 1.49e-8, int $limit = 50): array
    {
        if ($a == $b) {                              // scipy's shortcut for an empty interval: no calls
            return [0.0, 0.0];
        }
        if ($limit < 1) {
            throw new \ValueError('quad: limit must be at least 1');
        }
        // as scipy: integrate over [min, max] and negate the value when b < a
        $flip = $b < $a;
        [$value, $abserr] = self::qagse($f, min($a, $b), max($a, $b), $epsabs, $epsrel, $limit);
        return [$flip ? -$value : $value, $abserr];
    }

    /**
     * DQAGSE (scipy's C translation, labels kept): returns [result, abserr]. Divisions go through fdiv, which is IEEE
     * division as in C (PHP's / throws on a zero divisor).
     */
    private static function qagse(callable $f, float $a, float $b, float $epsabs, float $epsrel, int $limit): array
    {
        $epmach = self::EPMACH;
        $alist = array_fill(0, $limit, 0.0);
        $blist = array_fill(0, $limit, 0.0);
        $rlist = array_fill(0, $limit, 0.0);
        $elist = array_fill(0, $limit, 0.0);
        $iord = array_fill(0, $limit + 1, 0);
        $alist[0] = $a;
        $blist[0] = $b;
        $ier = 0;
        $ierror = 0;
        $erlarg = 0.0;
        $ertest = 0.0;
        $correc = 0.0;
        $small = 0.0;
        if ($epsabs <= 0.0 && $epsrel < max(50.0 * $epmach, 0.5e-28)) {
            throw new \ValueError('quad: invalid input: epsabs <= 0 and epsrel < max(50 * eps, 5e-29)');
        }

        // first approximation to the integral, and the test on its accuracy
        [$result, $abserr, $defabs, $resabs] = self::qk21($f, $a, $b);
        $dres = abs($result);
        $errbnd = max($epsabs, $epsrel * $dres);
        $last = 1;
        $rlist[0] = $result;
        $elist[0] = $abserr;
        if ($abserr <= 100.0 * $epmach * $defabs && $abserr > $errbnd) {
            $ier = 2;
        }
        if ($limit == 1) {
            $ier = 1;
        }
        if ($ier != 0 || ($abserr <= $errbnd && $abserr != $resabs) || $abserr == 0.0) {
            return [$result, $abserr];
        }

        $rlist2 = array_fill(0, 52, 0.0);
        $res3la = [0.0, 0.0, 0.0];
        $rlist2[0] = $result;
        $errmax = $abserr;
        $maxerr = 0;
        $area = $result;
        $errsum = $abserr;
        $abserr = self::OFLOW;
        $nrmax = 0;
        $nres = 0;
        $numrl2 = 1;
        $ktmin = 0;
        $extrap = false;
        $noext = false;
        $iroff1 = $iroff2 = $iroff3 = 0;

        for ($L = 1; $L < $limit; $L++) {
            $last = $L + 1;
            // bisect the sub-interval with the nrmax-th largest error estimate
            $a1 = $alist[$maxerr];
            $b1 = 0.5 * ($alist[$maxerr] + $blist[$maxerr]);
            $a2 = $b1;
            $b2 = $blist[$maxerr];
            $erlast = $errmax;
            [$area1, $error1, , $defab1] = self::qk21($f, $a1, $b1);
            [$area2, $error2, , $defab2] = self::qk21($f, $a2, $b2);

            // improve the previous approximations to the integral and error, and test for accuracy
            $area12 = $area1 + $area2;
            $error12 = $error1 + $error2;
            $errsum = $errsum + $error12 - $errmax;
            $area = $area + $area12 - $rlist[$maxerr];
            if ($defab1 != $error1 && $defab2 != $error2) {
                if (!(abs($rlist[$maxerr] - $area12) > 1.0e-5 * abs($area12) || $error12 < 0.99 * $errmax)) {
                    if ($extrap) {
                        $iroff2++;
                    } else {
                        $iroff1++;
                    }
                }
                if ($L > 9 && $error12 > $errmax) {
                    $iroff3++;
                }
            }
            $rlist[$maxerr] = $area1;
            $rlist[$L] = $area2;
            $errbnd = max($epsabs, $epsrel * abs($area));

            // roundoff, the subdivision limit, and bad integrand behaviour at some points
            if ($iroff1 + $iroff2 >= 10 || $iroff3 >= 20) {
                $ier = 2;
            }
            if ($iroff2 >= 5) {
                $ierror = 3;
            }
            if ($last == $limit) {
                $ier = 1;
            }
            if (max(abs($a1), abs($b2)) <= (1.0 + 100.0 * $epmach) * (abs($a2) + 1000.0 * self::UFLOW)) {
                $ier = 4;
            }

            // append the newly created intervals to the list
            if (!($error2 > $error1)) {
                $alist[$L] = $a2;
                $blist[$maxerr] = $b1;
                $blist[$L] = $b2;
                $elist[$maxerr] = $error1;
                $elist[$L] = $error2;
            } else {
                $alist[$maxerr] = $a2;
                $alist[$L] = $a1;
                $blist[$L] = $b1;
                $rlist[$maxerr] = $area2;
                $rlist[$L] = $area1;
                $elist[$maxerr] = $error2;
                $elist[$L] = $error1;
            }

            // keep the error estimates in descending order and pick the sub-interval to bisect next
            self::qpsrt($limit, $last, $maxerr, $errmax, $elist, $iord, $nrmax);

            if ($errsum <= $errbnd) {
                goto LINE115;
            }
            if ($ier != 0) {
                break;
            }
            if ($L == 1) {
                goto LINE80;
            }
            if ($noext) {
                continue;
            }
            $erlarg = $erlarg - $erlast;
            if (abs($b1 - $a1) > $small) {
                $erlarg = $erlarg + $error12;
            }
            if (!$extrap) {
                // is the interval to be bisected next the smallest one?
                if (abs($blist[$maxerr] - $alist[$maxerr]) > $small) {
                    continue;
                }
                $extrap = true;
                $nrmax = 1;
            }
            if ($ierror == 3 || $erlarg <= $ertest) {
                goto LINE60;
            }
            // the smallest interval has the largest error: before bisecting, decrease the sum of the errors over
            // the larger intervals (erlarg) and extrapolate
            $jupbnd = $last > 2 + intdiv($limit, 2) ? $limit + 3 - $last : $last;
            for ($k = $nrmax; $k < $jupbnd; $k++) {
                $maxerr = $iord[$nrmax];
                $errmax = $elist[$maxerr];
                if (abs($blist[$maxerr] - $alist[$maxerr]) > $small) {
                    goto LINE90;
                }
                $nrmax++;
            }
            LINE60:
            // extrapolate
            $numrl2++;
            $rlist2[$numrl2] = $area;
            [$reseps, $abseps] = self::qelg($numrl2, $rlist2, $res3la, $nres);
            $ktmin++;
            if ($ktmin > 5 && $abserr < 1.0e-3 * $errsum) {
                $ier = 5;
            }
            if ($abseps >= $abserr) {
                goto LINE70;
            }
            $ktmin = 0;
            $abserr = $abseps;
            $result = $reseps;
            $correc = $erlarg;
            $ertest = max($epsabs, $epsrel * abs($reseps));
            if ($abserr <= $ertest) {
                break;
            }
            LINE70:
            // prepare the bisection of the smallest interval
            if ($numrl2 == 0) {
                $noext = true;
            }
            if ($ier == 5) {
                break;
            }
            $maxerr = $iord[0];
            $errmax = $elist[$maxerr];
            $nrmax = 0;
            $extrap = false;
            $small = $small * 0.5;
            $erlarg = $errsum;
            continue;
            LINE80:
            $small = abs($b - $a) * 0.375;
            $erlarg = $errsum;
            $ertest = $errbnd;
            $rlist2[1] = $area;
            LINE90:
            ;
        }

        // the final result and error estimate
        if ($abserr == self::OFLOW) {
            goto LINE115;
        }
        if ($ier + $ierror == 0) {
            goto LINE110;
        }
        if ($ierror == 3) {
            $abserr = $abserr + $correc;
        }
        if ($ier == 0) {
            $ier = 3;
        }
        if ($result != 0.0 && $area != 0.0) {
            goto LINE105;
        }
        if ($abserr > $errsum) {
            goto LINE115;
        }
        if ($area == 0.0) {
            goto LINE130;
        }
        goto LINE110;
        LINE105:
        if (fdiv($abserr, abs($result)) > fdiv($errsum, abs($area))) {
            goto LINE115;
        }
        LINE110:
        // test on divergence (it only sets ier, which quad does not return)
        goto LINE130;
        LINE115:
        // the global integral sum
        $result = 0.0;
        for ($k = 0; $k <= $L; $k++) {
            $result = $result + $rlist[$k];
        }
        $abserr = $errsum;
        LINE130:
        return [$result, $abserr];
    }

    /** DQPSRT: keeps elist's descending order in iord and selects maxerr, the next sub-interval to bisect. */
    private static function qpsrt(int $limit, int $last, int &$maxerr, float &$ermax, array $elist, array &$iord, int &$nrmax): void
    {
        if ($last <= 2) {
            $iord[0] = 0;
            $iord[1] = 1;
            $maxerr = $iord[$nrmax];
            $ermax = $elist[$maxerr];
            return;
        }
        $errmax = $elist[$maxerr];
        while ($nrmax > 0 && $errmax > $elist[$iord[$nrmax - 1]]) {
            $iord[$nrmax] = $iord[$nrmax - 1];
            $nrmax--;
        }
        $jupbn = $last > intdiv($limit, 2) + 2 ? $limit - $last + 2 : $last - 1;
        $errmin = $elist[$last - 1];
        $jbnd = $jupbn - 1;
        $ibeg = $nrmax + 1;
        $i = $ibeg;
        $found = false;
        if ($ibeg <= $jbnd) {
            for ($i = $ibeg; $i <= $jbnd; $i++) {
                if ($errmax >= $elist[$iord[$i]]) {
                    $found = true;
                    break;
                }
                $iord[$i - 1] = $iord[$i];
            }
        }
        if (!$found) {
            $iord[$jbnd] = $maxerr;
            $iord[$jupbn] = $last - 1;
            $maxerr = $iord[$nrmax];
            $ermax = $elist[$maxerr];
            return;
        }
        $iord[$i - 1] = $maxerr;
        $k = $jbnd;
        for ($j = $i; $j <= $jbnd; $j++) {
            if ($errmin < $elist[$iord[$k]]) {
                break;
            }
            $iord[$k + 1] = $iord[$k];
            $k--;
        }
        if ($errmin < $elist[$iord[$k]]) {
            $iord[$k + 1] = $last - 1;
        } else {
            $iord[$i] = $last - 1;
        }
        $maxerr = $iord[$nrmax];
        $ermax = $elist[$maxerr];
    }

    /**
     * DQELG, the epsilon algorithm: extrapolates the limit of the sequence in epstab[0..n] (updated in place, as are
     * n, res3la and nres); returns [result, abserr].
     */
    private static function qelg(int &$n, array &$epstab, array &$res3la, int &$nres): array
    {
        $epmach = self::EPMACH;
        $startingN = $n;
        $nres++;
        $abserr = self::OFLOW;
        $result = $epstab[$n];
        if ($n < 2) {
            return [$result, max($abserr, 5.0 * $epmach * abs($result))];
        }
        $limexp = 49;
        $epstab[$n + 2] = $epstab[$n];
        $newelm = intdiv($n, 2);
        $epstab[$n] = self::OFLOW;
        $k1 = $n;
        for ($i = 0; $i < $newelm; $i++) {
            $k2 = $k1 - 1;
            $k3 = $k1 - 2;
            $res = $epstab[$k1 + 2];
            $e0 = $epstab[$k3];
            $e1 = $epstab[$k2];
            $e2 = $res;
            $e1abs = abs($e1);
            $delta2 = $e2 - $e1;
            $err2 = abs($delta2);
            $tol2 = max(abs($e2), $e1abs) * $epmach;
            $delta3 = $e1 - $e0;
            $err3 = abs($delta3);
            $tol3 = max($e1abs, abs($e0)) * $epmach;
            if (!($err2 > $tol2 || $err3 > $tol3)) {
                // e0, e1 and e2 are equal to within machine accuracy: convergence
                $result = $res;
                $abserr = $err2 + $err3;
                return [$result, max($abserr, 5.0 * $epmach * abs($result))];
            }
            $e3 = $epstab[$k1];
            $epstab[$k1] = $e1;
            $delta1 = $e1 - $e3;
            $err1 = abs($delta1);
            $tol1 = max($e1abs, abs($e3)) * $epmach;
            if ($err1 <= $tol1 || $err2 <= $tol2 || $err3 <= $tol3) {
                $n = $i + $i;
                break;
            }
            $ss = fdiv(1.0, $delta1) + fdiv(1.0, $delta2) - fdiv(1.0, $delta3);
            $epsinf = abs($ss * $e1);
            if (!($epsinf > 1e-4)) {
                $n = $i + $i;
                break;
            }
            $res = $e1 + fdiv(1.0, $ss);
            $epstab[$k1] = $res;
            $k1 -= 2;
            $error = $err2 + abs($res - $e2) + $err3;
            if (!($error > $abserr)) {
                $abserr = $error;
                $result = $res;
            }
        }
        if ($n == $limexp) {
            $n = 2 * intdiv($limexp, 2);
        }
        $n2rem = $startingN % 2;
        for ($i = 0; $i <= $newelm; $i++) {
            $epstab[2 * $i + $n2rem] = $epstab[2 * $i + 2 + $n2rem];
        }
        if ($n != $startingN) {
            $indx = $startingN - $n;
            for ($i = 0; $i <= $n; $i++) {
                $epstab[$i] = $epstab[$indx];
                $indx++;
            }
        }
        if ($nres < 4) {
            $res3la[$nres - 1] = $result;
            $abserr = self::OFLOW;
        } else {
            $abserr = abs($result - $res3la[2]) + abs($result - $res3la[1]) + abs($result - $res3la[0]);
            $res3la[0] = $res3la[1];
            $res3la[1] = $res3la[2];
            $res3la[2] = $result;
        }
        return [$result, max($abserr, 5.0 * $epmach * abs($result))];
    }

    /**
     * One 10-21 Gauss-Kronrod panel on [$a, $b], QUADPACK's dqk21 step for step: returns [integral, abserr,
     * resabs, resasc] (the integral of |f| and of |f - mean| over the panel).
     */
    private static function qk21(callable $f, float $a, float $b): array
    {
        $xgk = self::XGK;
        $wgk = self::WGK;
        $wg = self::WG;
        $c = 0.5 * ($a + $b);
        $h = 0.5 * ($b - $a);
        $dh = abs($h);
        $resg = 0.0;                                 // the 10-point Gauss rule has no centre node
        $fc = (float) $f($c);
        $resk = $wgk[10] * $fc;
        $resabs = abs($resk);
        $fv1 = $fv2 = [];
        for ($j = 0; $j < 5; $j++) {                 // Gauss points (odd Kronrod indices)
            $jtw = 2 * $j + 1;
            $x = $h * $xgk[$jtw];
            $f1 = (float) $f($c - $x);
            $f2 = (float) $f($c + $x);
            $fv1[$jtw] = $f1;
            $fv2[$jtw] = $f2;
            $fsum = $f1 + $f2;
            $resg += $wg[$j] * $fsum;
            $resk += $wgk[$jtw] * $fsum;
            $resabs += $wgk[$jtw] * (abs($f1) + abs($f2));
        }
        for ($j = 0; $j < 5; $j++) {                 // Kronrod-only points (even indices)
            $jtwm1 = 2 * $j;
            $x = $h * $xgk[$jtwm1];
            $f1 = (float) $f($c - $x);
            $f2 = (float) $f($c + $x);
            $fv1[$jtwm1] = $f1;
            $fv2[$jtwm1] = $f2;
            $fsum = $f1 + $f2;
            $resk += $wgk[$jtwm1] * $fsum;
            $resabs += $wgk[$jtwm1] * (abs($f1) + abs($f2));
        }
        $reskh = $resk * 0.5;
        $resasc = $wgk[10] * abs($fc - $reskh);
        for ($j = 0; $j < 10; $j++) {
            $resasc += $wgk[$j] * (abs($fv1[$j] - $reskh) + abs($fv2[$j] - $reskh));
        }
        $integral = $resk * $h;
        $resabs *= $dh;
        $resasc *= $dh;
        $abserr = abs(($resk - $resg) * $h);
        if ($resasc != 0.0 && $abserr != 0.0) {
            $abserr = $resasc * min(1.0, (200.0 * $abserr / $resasc) ** 1.5);
        }
        if ($resabs > self::UFLOW / (50.0 * self::EPMACH)) {
            $abserr = max(self::EPMACH * 50.0 * $resabs, $abserr);
        }
        return [$integral, $abserr, $resabs, $resasc];
    }

    /**
     * Fixed-order Gauss-Legendre quadrature of $func over [$a, $b] (scipy.integrate.fixed_quad).
     *
     * Uses the native n-point Gauss-Legendre rule (nodes/weights from Tessero\Special::rootsLegendre),
     * affinely mapped from [-1, 1] to [$a, $b]. Returns the fixed-order estimate and null (scipy returns
     * (val, None) -- the order-n value, NOT a converged/adaptive integral).
     *
     * @param callable(float): float $func  integrand evaluated at each mapped node
     * @return array{0: float, 1: null}  [value, null]
     */
    public static function fixedQuad(callable $func, float $a, float $b, int $n = 5): array
    {
        $rl = \Tessero\Special::rootsLegendre($n);           // [x => nodes, w => weights] on [-1, 1]
        $x = $rl["x"];
        $w = $rl["w"];
        $x = \is_object($x) ? $x->toArray() : (array) $x;
        $w = \is_object($w) ? $w->toArray() : (array) $w;

        $half = 0.5 * ($b - $a);
        $mid = 0.5 * ($b + $a);
        $sum = 0.0;
        foreach ($x as $i => $xi) {
            $y = $half * (float) $xi + $mid;                 // map node from [-1,1] to [a,b]
            $sum += (float) $w[$i] * (float) $func($y);
        }

        return [$half * $sum, null];
    }

}
