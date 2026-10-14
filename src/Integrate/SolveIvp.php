<?php

declare(strict_types=1);

namespace Tessero\Integrate;

/**
 * Explicit Runge-Kutta IVP ODE solver (scipy.integrate.solve_ivp).
 *
 * Pure PHP over a user callable $fun(float \, list<float> \): list<float> (dy/dt = f(t, y)).
 * Reproduces scipy adaptive machinery bit-for-bit: _select_initial_step, the RMS error norm with
 * rtol/atol, the I-controller step-size controller (SAFETY/MIN_FACTOR/MAX_FACTOR + error exponent),
 * and the per-method polynomial dense-output interpolant for the requested t_eval points. The control
 * flow is deterministic, so the solution at t_eval matches scipy solve_ivp to machine precision.
 *
 * Methods: RK45 (Dormand-Prince 5(4), default) and RK23 (Bogacki-Shampine 3(2)). Dense-output / t_eval
 * path and finite t_span only (events, args, implicit stiff solvers are out of scope).
 */
final class SolveIvp
{
    private const SAFETY = 0.9;
    private const MIN_FACTOR = 0.2;
    private const MAX_FACTOR = 10.0;

    // ---- scipy.integrate DOP853 (Dormand-Prince 8(5,3)) coefficients ----
    private const DOP_C = [0, 0.05260015195876773, 0.078900227938151601, 0.1183503419072274, 0.28164965809277259, 0.33333333333333331, 0.25, 0.30769230769230771, 0.6512820512820513, 0.59999999999999998, 0.8571428571428571, 1, 1, 0.10000000000000001, 0.20000000000000001, 0.77777777777777779];
    private const DOP_A = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0.05260015195876773, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0.0197250569845379, 0.059175170953613701, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0.029587585476806851, 0, 0.088762756430420545, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0.24136513415926669, 0, -0.88454947932828609, 0.92483400326179199, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0.037037037037037035, 0, 0, 0.17082860872947386, 0.12546768756682242, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0.037109375, 0, 0, 0.17025221101954405, 0.060216538980455959, -0.017578125, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0.037092000118504789, 0, 0, 0.17038392571223998, 0.10726203044637328, -0.015319437748624402, 0.0082737891638140233, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0.62411095871607569, 0, 0, -3.3608926294469414, -0.86821934684172597, 27.59209969944671, 20.154067550477894, -43.489884181069961, 0, 0, 0, 0, 0, 0, 0, 0], [0.47766253643826434, 0, 0, -2.4881146199716677, -0.59029082683684297, 21.230051448181193, 15.279233632882423, -33.288210968984863, -0.020331201708508627, 0, 0, 0, 0, 0, 0, 0], [-0.9371424300859873, 0, 0, 5.1863724288440638, 1.0914373489967295, -8.1497870107469268, -18.520065659996959, 22.739487099350505, 2.4936055526796523, -3.0467644718982196, 0, 0, 0, 0, 0, 0], [2.273310147516538, 0, 0, -10.534495466737249, -2.0008720582248625, -17.958931863118799, 27.94888452941996, -2.8589982771350235, -8.8728569335306293, 12.360567175794303, 0.64339274601576357, 0, 0, 0, 0, 0], [0.054293734116568765, 0, 0, 0, 0, 4.4503128927524092, 1.8915178993145003, -5.8012039600105849, 0.3111643669578199, -0.15216094966251609, 0.20136540080403034, 0.044710615727772587, 0, 0, 0, 0], [0.056167502283047954, 0, 0, 0, 0, 0, 0.25350021021662483, -0.2462390374708025, -0.12419142326381637, 0.15329179827876568, 0.0082010522956346907, 0.0075678976605456994, -0.0082979999999999998, 0, 0, 0], [0.031834648163502142, 0, 0, 0, 0, 0.028300909672366776, 0.053541988307438566, -0.054923748571390991, 0, 0, -0.00010834732869724932, 0.00038257109083565839, -0.00034046500868740456, 0.1413124436746325, 0, 0], [-0.42889630158379194, 0, 0, 0, 0, -4.697621415361164, 7.6834211960625991, 4.0689898183971103, 0.35672718745528109, 0, 0, 0, -0.0013990241651590145, 2.9475147891527724, -9.1509584721798696, 0]];
    private const DOP_B = [0.054293734116568765, 0, 0, 0, 0, 4.4503128927524092, 1.8915178993145003, -5.8012039600105849, 0.3111643669578199, -0.15216094966251609, 0.20136540080403034, 0.044710615727772587];
    private const DOP_E3 = [-0.18980075407240762, 0, 0, 0, 0, 4.4503128927524092, 1.8915178993145003, -5.8012039600105849, -0.42268232132379191, -0.15216094966251609, 0.20136540080403034, 0.022651792198360821, 0];
    private const DOP_E5 = [0.01312004499419488, 0, 0, 0, 0, -1.2251564463762044, -0.4957589496572502, 1.6643771824549864, -0.35032884874997366, 0.33417911871301748, 0.08192320648511571, -0.022355307863886294, 0];
    private const DOP_D = [[-8.4289382761090135, 0, 0, 0, 0, 0.56671495351937773, -3.0689499459498917, 2.3846676565120699, 2.1170345824450281, -0.87139158377797299, 2.2404374302607883, 0.63157877876946877, -0.088990336451333307, 18.148505520854727, -9.194632392478356, -4.4360363875948936], [10.427508642579134, 0, 0, 0, 0, 242.28349177525817, 165.20045171727028, -374.5467547226902, -22.113666853125306, 7.7334326684722638, -30.674084731089398, -9.3321305264302286, 15.697238121770845, -31.139403219565178, -9.3529243588444793, 35.816841486394082], [19.985053242002433, 0, 0, 0, 0, -387.03730874935178, -189.17813819516758, 527.80815920542364, -11.573902539959629, 6.8812326946963003, -1.0006050966910838, 0.77771377980534429, -2.7782057523535082, -60.196695231264123, 84.320405506677162, 11.992291136182789], [-25.69393346270375, 0, 0, 0, 0, -154.18974869023643, -231.5293791760455, 357.63911791061412, 93.405324183624316, -37.458323136451632, 104.0996495089623, 29.840293426660502, -43.533456590011141, 96.324553959188279, -39.177261675615441, -149.72683625798564]];

    /**
     * @param callable(float, list<float>): (list<float>|float) $fun
     * @param array{0: float, 1: float} $tSpan
     * @param list<float>|float $y0
     * @param list<float> $tEval
     * @return array{t: list<float>, y: list<list<float>>}  y shaped (n_states x n_tEval), like scipy sol.y
     */
    public static function solve(callable $fun, array $tSpan, array|float $y0, array $tEval, string $method = 'RK45', float $rtol = 1e-3, float $atol = 1e-6): array
    {
        $y0v = is_array($y0) ? array_values(array_map('floatval', $y0)) : [(float) $y0];
        if (strtoupper($method) === 'DOP853') {
            $n0 = count($y0v);
            $callD = static function (float $t, array $yy) use ($fun, $n0): array {
                $r = $fun($t, $yy); $r = is_array($r) ? array_values($r) : [$r]; $o = [];
                foreach ($r as $v) { $o[] = (float) $v; }
                if (count($o) !== $n0) { throw new \RuntimeException('solve_ivp: RHS returned ' . count($o) . ' components, expected ' . $n0); }
                return $o;
            };
            $colsD = self::solveDop853($callD, (float) $tSpan[0], (float) $tSpan[1], $y0v, array_map('floatval', array_values($tEval)), $rtol, $atol);
            $ysolD = [];
            for ($i = 0; $i < $n0; $i++) { $row = []; foreach ($colsD as $cc) { $row[] = $cc[$i]; } $ysolD[] = $row; }
            return ['t' => array_map('floatval', array_values($tEval)), 'y' => $ysolD];
        }
        [$C, $A, $B, $E, $P, $nStages, $errOrder] = self::tableau($method);
        $y0 = is_array($y0) ? array_values(array_map('floatval', $y0)) : [(float) $y0];
        $n = count($y0);
        $t0 = (float) $tSpan[0];
        $tf = (float) $tSpan[1];
        $direction = $tf !== $t0 ? ($tf > $t0 ? 1.0 : -1.0) : 1.0;
        $call = static function (float $t, array $y) use ($fun, $n): array {
            $r = $fun($t, $y);
            $r = is_array($r) ? array_values($r) : [$r];
            $out = [];
            foreach ($r as $v) { $out[] = (float) $v; }
            if (count($out) !== $n) { throw new \RuntimeException('solve_ivp: RHS returned ' . count($out) . ' components, expected ' . $n); }
            return $out;
        };
        $f0 = $call($t0, $y0);
        $hAbs = self::selectInitialStep($call, $t0, $y0, $f0, $direction, $errOrder, $rtol, $atol);
        $errExp = -1.0 / ($errOrder + 1);
        $t = $t0; $y = $y0; $f = $f0; $seg = [];
        $minStep = 10.0 * (abs($t) * PHP_FLOAT_EPSILON + PHP_FLOAT_MIN);
        while ($direction * ($t - $tf) < 0.0) {
            if ($hAbs < $minStep) { $hAbs = $minStep; }
            $accepted = false; $rejected = false; $yNew = $y; $fNew = $f; $K = []; $hUsed = 0.0;
            while (! $accepted) {
                if ($hAbs < $minStep) { throw new \RuntimeException('solve_ivp: step size below minimum; integration failed'); }
                $h = $hAbs * $direction; $tNew = $t + $h;
                if ($direction * ($tNew - $tf) > 0.0) { $tNew = $tf; }
                $h = $tNew - $t; $hAbs = abs($h);
                [$yNew, $fNew, $K] = self::rkStep($call, $t, $y, $f, $h, $C, $A, $B, $nStages);
                $hUsed = $h;
                $errorNorm = 0.0;
                for ($i = 0; $i < $n; $i++) {
                    $e = 0.0;
                    for ($s = 0; $s <= $nStages; $s++) { $e += $K[$s][$i] * $E[$s]; }
                    $e *= $h;
                    $scale = $atol + max(abs($y[$i]), abs($yNew[$i])) * $rtol;
                    $rr = $e / $scale; $errorNorm += $rr * $rr;
                }
                $errorNorm = sqrt($errorNorm / $n);
                if ($errorNorm < 1.0) {
                    $factor = $errorNorm === 0.0 ? self::MAX_FACTOR : min(self::MAX_FACTOR, self::SAFETY * $errorNorm ** $errExp);
                    if ($rejected) { $factor = min(1.0, $factor); }
                    $hAbs *= $factor; $accepted = true;
                } else {
                    $hAbs *= max(self::MIN_FACTOR, self::SAFETY * $errorNorm ** $errExp); $rejected = true;
                }
            }
            $deg = count($P[0]); $Q = [];
            for ($i = 0; $i < $n; $i++) {
                $rowq = [];
                for ($j = 0; $j < $deg; $j++) {
                    $acc = 0.0;
                    for ($s = 0; $s <= $nStages; $s++) { $acc += $K[$s][$i] * $P[$s][$j]; }
                    $rowq[] = $acc;
                }
                $Q[] = $rowq;
            }
            $seg[] = [$t, $hUsed, $Q, $y];
            $t = $t + $hUsed; $y = $yNew; $f = $fNew;
        }
        $cols = [];
        foreach ($tEval as $teRaw) {
            $te = (float) $teRaw; $col = null;
            foreach ($seg as [$ta, $hseg, $Q, $ya]) {
                $tb = $ta + $hseg;
                $lo = $direction > 0 ? $ta : $tb; $hi = $direction > 0 ? $tb : $ta;
                if ($te >= $lo - 1e-15 && $te <= $hi + 1e-15) {
                    $x = ($te - $ta) / $hseg; $deg = count($Q[0]); $col = [];
                    for ($i = 0, $ni = count($ya); $i < $ni; $i++) {
                        $acc = 0.0; $xp = $x;
                        for ($j = 0; $j < $deg; $j++) { $acc += $Q[$i][$j] * $xp; $xp *= $x; }
                        $col[] = $ya[$i] + $hseg * $acc;
                    }
                    break;
                }
            }
            if ($col === null) { $col = (abs($te - $t0) < 1e-12) ? $y0 : $y; }
            $cols[] = $col;
        }
        $ysol = [];
        for ($i = 0; $i < $n; $i++) {
            $rowOut = [];
            foreach ($cols as $c) { $rowOut[] = $c[$i]; }
            $ysol[] = $rowOut;
        }
        return ['t' => array_map('floatval', array_values($tEval)), 'y' => $ysol];
    }

    /**
     * One RK step; returns [y_new, f_new, K] where K has nStages+1 rows.
     *
     * @param callable(float, list<float>): list<float> $call
     * @param list<float> $y
     * @param list<float> $f
     * @return array{0: list<float>, 1: list<float>, 2: list<list<float>>}
     */
    private static function rkStep(callable $call, float $t, array $y, array $f, float $h, array $C, array $A, array $B, int $nStages): array
    {
        $n = count($y); $K = []; $K[0] = $f;
        for ($s = 1; $s < $nStages; $s++) {
            $ys = $y;
            for ($i = 0; $i < $n; $i++) {
                $dy = 0.0;
                for ($j = 0; $j < $s; $j++) { $dy += $K[$j][$i] * $A[$s][$j]; }
                $ys[$i] = $y[$i] + $dy * $h;
            }
            $K[$s] = $call($t + $C[$s] * $h, $ys);
        }
        $yNew = $y;
        for ($i = 0; $i < $n; $i++) {
            $acc = 0.0;
            for ($s = 0; $s < $nStages; $s++) { $acc += $K[$s][$i] * $B[$s]; }
            $yNew[$i] = $y[$i] + $h * $acc;
        }
        $fNew = $call($t + $h, $yNew); $K[$nStages] = $fNew;
        return [$yNew, $fNew, $K];
    }

    /**
     * scipy _select_initial_step.
     *
     * @param callable(float, list<float>): list<float> $call
     * @param list<float> $y0
     * @param list<float> $f0
     */
    private static function selectInitialStep(callable $call, float $t0, array $y0, array $f0, float $direction, int $order, float $rtol, float $atol): float
    {
        $n = count($y0);
        if ($n === 0) { return INF; }
        $d0 = 0.0; $d1 = 0.0; $scale = [];
        for ($i = 0; $i < $n; $i++) {
            $sc = $atol + abs($y0[$i]) * $rtol; $scale[$i] = $sc;
            $a = $y0[$i] / $sc; $b = $f0[$i] / $sc;
            $d0 += $a * $a; $d1 += $b * $b;
        }
        $d0 = sqrt($d0 / $n); $d1 = sqrt($d1 / $n);
        $h0 = ($d0 < 1e-5 || $d1 < 1e-5) ? 1e-6 : 0.01 * $d0 / $d1;
        $y1 = [];
        for ($i = 0; $i < $n; $i++) { $y1[$i] = $y0[$i] + $h0 * $direction * $f0[$i]; }
        $f1 = $call($t0 + $h0 * $direction, $y1);
        $d2 = 0.0;
        for ($i = 0; $i < $n; $i++) { $rr = ($f1[$i] - $f0[$i]) / $scale[$i]; $d2 += $rr * $rr; }
        $d2 = sqrt($d2 / $n) / $h0;
        if ($d1 <= 1e-15 && $d2 <= 1e-15) { $h1 = max(1e-6, $h0 * 1e-3); }
        else { $h1 = (0.01 / max($d1, $d2)) ** (1.0 / ($order + 1)); }
        return min(100.0 * $h0, $h1);
    }

    /**
     * Butcher tableau + dense-output matrix P for a method.
     *
     * @return array{0: list<float>, 1: list<list<float>>, 2: list<float>, 3: list<float>, 4: list<list<float>>, 5: int, 6: int}
     */
    private static function tableau(string $method): array
    {
        return match (strtoupper($method)) {
            'RK45' => [
                [0.0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1.0],
                [[0, 0, 0, 0, 0], [1 / 5, 0, 0, 0, 0], [3 / 40, 9 / 40, 0, 0, 0], [44 / 45, -56 / 15, 32 / 9, 0, 0],
                 [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729, 0], [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656]],
                [35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84],
                [-71 / 57600, 0.0, 71 / 16695, -71 / 1920, 17253 / 339200, -22 / 525, 1 / 40],
                [[1, -8048581381 / 2820520608, 8663915743 / 2820520608, -12715105075 / 11282082432], [0, 0, 0, 0],
                 [0, 131558114200 / 32700410799, -68118460800 / 10900136933, 87487479700 / 32700410799],
                 [0, -1754552775 / 470086768, 14199869525 / 1410260304, -10690763975 / 1880347072],
                 [0, 127303824393 / 49829197408, -318862633887 / 49829197408, 701980252875 / 199316789632],
                 [0, -282668133 / 205662961, 2019193451 / 616988883, -1453857185 / 822651844],
                 [0, 40617522 / 29380423, -110615467 / 29380423, 69997945 / 29380423]],
                6, 4,
            ],
            'RK23' => [
                [0.0, 1 / 2, 3 / 4],
                [[0, 0, 0], [1 / 2, 0, 0], [0, 3 / 4, 0]],
                [2 / 9, 1 / 3, 4 / 9],
                [5 / 72, -1 / 12, -1 / 9, 1 / 8],
                [[1, -4 / 3, 5 / 9], [0, 1, -2 / 3], [0, 4 / 3, -8 / 9], [0, -1, 1]],
                3, 2,
            ],
            default => throw new \RuntimeException('solve_ivp: unsupported method (RK45, RK23): ' . $method),
        };
    }

    /**
     * scipy.integrate.solve_ivp method='DOP853' (Dormand-Prince 8(5,3) explicit Runge-Kutta with 7th-order
     * dense output). Same adaptive machinery as RK45/RK23 -- _select_initial_step, the I-controller step
     * selection, finite t_span and dense-output t_eval -- but with 12 stages, the combined err5/err3 error
     * norm, 3 extra stages for the interpolant, and the alternating x/(1-x) Horner dense-output evaluation.
     * Matches scipy's sol.y to machine precision.
     *
     * @param callable(float, list<float>): list<float> $call  validated RHS f(t, y) -> list<float>
     * @param list<float> $y0
     * @param list<float> $tEval
     * @return list<list<float>>  cols: one state-vector per t_eval point
     */
    private static function solveDop853(callable $call, float $t0, float $tf, array $y0, array $tEval, float $rtol, float $atol): array
    {
        $C = self::DOP_C; $A = self::DOP_A; $B = self::DOP_B; $E3 = self::DOP_E3; $E5 = self::DOP_E5; $D = self::DOP_D;
        $nStages = 12; $nExt = 16; $errOrder = 7;
        $n = count($y0);
        $direction = $tf !== $t0 ? ($tf > $t0 ? 1.0 : -1.0) : 1.0;
        $f0 = $call($t0, $y0);
        $hAbs = self::selectInitialStep($call, $t0, $y0, $f0, $direction, $errOrder, $rtol, $atol);
        $errExp = -1.0 / ($errOrder + 1);
        $t = $t0; $y = $y0; $f = $f0; $seg = [];
        $minStep = 10.0 * (abs($t) * PHP_FLOAT_EPSILON + PHP_FLOAT_MIN);
        while ($direction * ($t - $tf) < 0.0) {
            if ($hAbs < $minStep) { $hAbs = $minStep; }
            $accepted = false; $rejected = false; $yNew = $y; $fNew = $f; $K = []; $hUsed = 0.0;
            while (! $accepted) {
                if ($hAbs < $minStep) { throw new \RuntimeException('solve_ivp: step size below minimum; integration failed'); }
                $h = $hAbs * $direction; $tNew = $t + $h;
                if ($direction * ($tNew - $tf) > 0.0) { $tNew = $tf; }
                $h = $tNew - $t; $hAbs = abs($h);
                // explicit RK step: K[0]=f, stages 1..nStages-1, then y_new via B, then K[nStages]=f(t+h,y_new)
                $K = []; $K[0] = $f;
                for ($s = 1; $s < $nStages; $s++) {
                    $ys = $y;
                    for ($i = 0; $i < $n; $i++) {
                        $dy = 0.0;
                        for ($j = 0; $j < $s; $j++) { $dy += $K[$j][$i] * $A[$s][$j]; }
                        $ys[$i] = $y[$i] + $dy * $h;
                    }
                    $K[$s] = $call($t + $C[$s] * $h, $ys);
                }
                $yNew = $y;
                for ($i = 0; $i < $n; $i++) {
                    $acc = 0.0;
                    for ($s = 0; $s < $nStages; $s++) { $acc += $K[$s][$i] * $B[$s]; }
                    $yNew[$i] = $y[$i] + $h * $acc;
                }
                $fNew = $call($t + $h, $yNew); $K[$nStages] = $fNew;
                // combined DOP853 error norm (err5 primary, err3 correction)
                $err5n2 = 0.0; $err3n2 = 0.0;
                for ($i = 0; $i < $n; $i++) {
                    $e5 = 0.0; $e3 = 0.0;
                    for ($s = 0; $s <= $nStages; $s++) { $e5 += $K[$s][$i] * $E5[$s]; $e3 += $K[$s][$i] * $E3[$s]; }
                    $scale = $atol + max(abs($y[$i]), abs($yNew[$i])) * $rtol;
                    $e5 /= $scale; $e3 /= $scale;
                    $err5n2 += $e5 * $e5; $err3n2 += $e3 * $e3;
                }
                if ($err5n2 === 0.0 && $err3n2 === 0.0) {
                    $errorNorm = 0.0;
                } else {
                    $denom = $err5n2 + 0.01 * $err3n2;
                    $errorNorm = abs($h) * $err5n2 / sqrt($denom * $n);
                }
                if ($errorNorm < 1.0) {
                    $factor = $errorNorm === 0.0 ? self::MAX_FACTOR : min(self::MAX_FACTOR, self::SAFETY * $errorNorm ** $errExp);
                    if ($rejected) { $factor = min(1.0, $factor); }
                    $hAbs *= $factor; $accepted = true; $hUsed = $h;
                } else {
                    $hAbs *= max(self::MIN_FACTOR, self::SAFETY * $errorNorm ** $errExp); $rejected = true;
                }
            }
            // dense output: 3 extra stages (indices nStages+1 .. nExt-1), then F (7 rows)
            for ($s = $nStages + 1; $s < $nExt; $s++) {
                $ys = $y;
                for ($i = 0; $i < $n; $i++) {
                    $dy = 0.0;
                    for ($j = 0; $j < $s; $j++) { $dy += $K[$j][$i] * $A[$s][$j]; }
                    $ys[$i] = $y[$i] + $dy * $hUsed;
                }
                $K[$s] = $call($t + $C[$s] * $hUsed, $ys);
            }
            $fOld = $K[0]; $deltaY = [];
            for ($i = 0; $i < $n; $i++) { $deltaY[$i] = $yNew[$i] - $y[$i]; }
            $F = [];
            $F[0] = $deltaY;
            $F[1] = []; for ($i = 0; $i < $n; $i++) { $F[1][$i] = $hUsed * $fOld[$i] - $deltaY[$i]; }
            $F[2] = []; for ($i = 0; $i < $n; $i++) { $F[2][$i] = 2.0 * $deltaY[$i] - $hUsed * ($fNew[$i] + $fOld[$i]); }
            for ($r = 0; $r < 4; $r++) {
                $F[3 + $r] = [];
                for ($i = 0; $i < $n; $i++) {
                    $acc = 0.0;
                    for ($s = 0; $s < $nExt; $s++) { $acc += $D[$r][$s] * $K[$s][$i]; }
                    $F[3 + $r][$i] = $hUsed * $acc;
                }
            }
            $seg[] = [$t, $hUsed, $F, $y];
            $t += $hUsed; $y = $yNew; $f = $fNew;
        }
        $cols = [];
        foreach ($tEval as $teRaw) {
            $te = (float) $teRaw; $col = null;
            foreach ($seg as [$ta, $hseg, $F, $ya]) {
                $tb = $ta + $hseg;
                $lo = $direction > 0 ? $ta : $tb; $hi = $direction > 0 ? $tb : $ta;
                if ($te >= $lo - 1e-15 && $te <= $hi + 1e-15) {
                    $x = ($te - $ta) / $hseg; $col = array_fill(0, $n, 0.0);
                    // scipy evaluates sum over reversed(F) with alternating *x (even index in the reversed list) / *(1-x)
                    for ($k = 6; $k >= 0; $k--) {
                        for ($i = 0; $i < $n; $i++) { $col[$i] += $F[$k][$i]; }
                        $ii = 6 - $k;
                        if ($ii % 2 === 0) { for ($i = 0; $i < $n; $i++) { $col[$i] *= $x; } }
                        else { for ($i = 0; $i < $n; $i++) { $col[$i] *= (1.0 - $x); } }
                    }
                    for ($i = 0; $i < $n; $i++) { $col[$i] += $ya[$i]; }
                    break;
                }
            }
            if ($col === null) { $col = (abs($te - $t0) < 1e-12) ? $y0 : $y; }
            $cols[] = $col;
        }
        return $cols;
    }
}
