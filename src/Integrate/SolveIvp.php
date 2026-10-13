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

    /**
     * @param callable(float, list<float>): (list<float>|float) $fun
     * @param array{0: float, 1: float} $tSpan
     * @param list<float>|float $y0
     * @param list<float> $tEval
     * @return array{t: list<float>, y: list<list<float>>}  y shaped (n_states x n_tEval), like scipy sol.y
     */
    public static function solve(callable $fun, array $tSpan, array|float $y0, array $tEval, string $method = 'RK45', float $rtol = 1e-3, float $atol = 1e-6): array
    {
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
}
