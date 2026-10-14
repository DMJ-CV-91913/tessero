<?php

declare(strict_types=1);

namespace Tessero\Integrate;

/**
 * Stateful ODE stepper (scipy.integrate.ode and scipy.integrate.complex_ode).
 *
 * scipy's legacy `ode` object wraps a Fortran integrator (dopri5 = Hairer DOPRI5, dop853 = Hairer DOP853,
 * vode/lsoda = ODEPACK) behind a stateful stepping API:
 *
 *     r = ode(f).set_integrator('dopri5').set_initial_value(y0, t0)
 *     y1 = r.integrate(t1)   # advance from the current (t, y) to t1, return y(t1)
 *     y2 = r.integrate(t2)   # advance again from (t1, y1) to t2
 *
 * This is a thin stateful facade over {@see SolveIvp}: each integrate(t) call runs the underlying adaptive
 * engine over the segment [t_current, t_target] with t_eval = [t_target], then stores (t_target, y) as the
 * new state. dopri5 -> RK45 (Dormand-Prince 5(4)), dop853 -> DOP853 (Dormand-Prince 8(5,3)); vode/lsoda map
 * to BDF for the stiff path (vode's default is Adams for non-stiff, bdf for stiff via method='bdf'; here the
 * stiff BDF engine is used). The explicit non-stiff integrators (dopri5/dop853) converge to the same ODE
 * solution as scipy's Fortran codes at tight tolerances (agreement ~1e-8 at rtol=1e-9/atol=1e-12); the
 * stiff Fortran codes (vode/lsoda) use genuinely different step/order control, so numerical parity is only
 * claimed for the explicit engines.
 *
 * complex_ode: scipy.integrate.complex_ode(f) integrates a complex-valued system by splitting y into real
 * and imaginary parts, forming the 2n-dimensional real system [Re f, Im f], stepping it, and recombining.
 * Construct this facade with $complex = true (or via {@see complexOde}); y0 may then be given as
 * [[re, im], ...] pairs and integrate(t) returns [[re, im], ...].
 *
 * Pure PHP, host-independent.
 */
final class Ode
{
    /** @var callable(float, list<float>): (list<float>|float) */
    private $fun;
    private bool $complex;
    private string $method = 'RK45';
    private float $rtol = 1e-6;
    private float $atol = 1e-12;
    /** @var list<float> real state vector (length n for real; length 2n interleaved re,im for complex) */
    private array $y = [];
    private float $t = 0.0;
    private bool $success = true;
    /** @var list<mixed> extra params appended to f (scipy set_f_params) */
    private array $fparams = [];

    /**
     * @param callable(float, list<float>): (list<float>|float) $fun  RHS f(t, y[, *params]); for complex mode
     *        f receives/returns complex components encoded as [re, im] pairs (see integrate()).
     */
    public function __construct(callable $fun, bool $complex = false)
    {
        $this->fun = $fun;
        $this->complex = $complex;
    }

    /** scipy.integrate.complex_ode(f): construct a complex-valued stepper. */
    public static function complexOde(callable $fun): self
    {
        return new self($fun, true);
    }

    /**
     * scipy ode.set_integrator(name, **opts). Recognised names: 'dopri5' (-> RK45), 'dop853' (-> DOP853),
     * 'vode'/'lsoda' (-> BDF). Options 'rtol' and 'atol' are honoured; others are accepted and ignored.
     */
    public function setIntegrator(string $name, float $rtol = 1e-6, float $atol = 1e-12): self
    {
        $this->method = match (strtolower($name)) {
            'dopri5'       => 'RK45',
            'dop853'       => 'DOP853',
            'vode', 'lsoda', 'bdf' => 'BDF',
            'rk45'         => 'RK45',
            'rk23'         => 'RK23',
            'radau'        => 'Radau',
            default        => throw new \RuntimeException("Tessero\\Integrate\\Ode: unknown integrator '{$name}'"),
        };
        $this->rtol = $rtol;
        $this->atol = $atol;
        return $this;
    }

    /**
     * scipy ode.set_initial_value(y, t). For real mode $y is a list<float> (or scalar). For complex mode $y
     * is a list of [re, im] pairs (or a single [re, im]); it is flattened to the internal real state.
     *
     * @param list<float>|list<array{0: float, 1: float}>|float $y
     */
    public function setInitialValue(array|float $y, float $t = 0.0): self
    {
        $this->t = $t;
        $this->success = true;
        if ($this->complex) {
            $pairs = is_array($y) && isset($y[0]) && is_array($y[0]) ? $y : [$y];
            $flat = [];
            foreach ($pairs as $c) {
                $flat[] = (float) $c[0];
                $flat[] = (float) ($c[1] ?? 0.0);
            }
            $this->y = $flat;
        } else {
            $this->y = is_array($y) ? array_values(array_map('floatval', $y)) : [(float) $y];
        }
        return $this;
    }

    /** scipy ode.set_f_params(*args): extra positional args passed to f after (t, y). */
    public function setFParams(mixed ...$args): self
    {
        $this->fparams = array_values($args);
        return $this;
    }

    /**
     * scipy ode.integrate(t): advance the solution from the current time to $t and return y(t).
     *
     * Real mode returns list<float>. Complex mode returns list<array{0: float, 1: float}> (re/im pairs).
     *
     * @return list<float>|list<array{0: float, 1: float}>
     */
    public function integrate(float $t): array
    {
        if ($t === $this->t) {
            return $this->currentY();
        }
        $rhs = $this->buildRhs();
        try {
            $res = SolveIvp::solve($rhs, [$this->t, $t], $this->y, [$t], $this->method, $this->rtol, $this->atol);
            // SolveIvp returns y shaped (n_states x n_tEval); take the single t_eval column.
            $col = [];
            foreach ($res['y'] as $row) {
                $col[] = (float) $row[count($row) - 1];
            }
            $this->y = $col;
            $this->t = $t;
            $this->success = true;
        } catch (\Throwable $e) {
            $this->success = false;
        }
        return $this->currentY();
    }

    /** scipy ode.successful(): whether the last integrate() step completed without error. */
    public function successful(): bool
    {
        return $this->success;
    }

    /** Current integration time. */
    public function t(): float
    {
        return $this->t;
    }

    /**
     * Current solution value (same encoding integrate() returns).
     *
     * @return list<float>|list<array{0: float, 1: float}>
     */
    public function y(): array
    {
        return $this->currentY();
    }

    /** @return list<float>|list<array{0: float, 1: float}> */
    private function currentY(): array
    {
        if (! $this->complex) {
            return $this->y;
        }
        $out = [];
        for ($i = 0, $n = count($this->y); $i < $n; $i += 2) {
            $out[] = [$this->y[$i], $this->y[$i + 1]];
        }
        return $out;
    }

    /** Build the real-valued RHS callable fed to SolveIvp (handles complex splitting + f-params). */
    private function buildRhs(): callable
    {
        $fun = $this->fun;
        $params = $this->fparams;
        if (! $this->complex) {
            return static function (float $t, array $y) use ($fun, $params): array {
                $r = $fun($t, $y, ...$params);
                $r = is_array($r) ? array_values($r) : [$r];
                $o = [];
                foreach ($r as $v) { $o[] = (float) $v; }
                return $o;
            };
        }
        // Complex mode: internal state is [re0, im0, re1, im1, ...]. Present y to the user's f as
        // [[re,im],...] pairs; f returns [[re,im],...]; re-flatten the derivative.
        return static function (float $t, array $y) use ($fun, $params): array {
            $pairs = [];
            for ($i = 0, $n = count($y); $i < $n; $i += 2) {
                $pairs[] = [$y[$i], $y[$i + 1]];
            }
            $r = $fun($t, $pairs, ...$params);
            $r = is_array($r) ? array_values($r) : [$r];
            $o = [];
            foreach ($r as $c) {
                if (is_array($c)) {
                    $o[] = (float) $c[0];
                    $o[] = (float) ($c[1] ?? 0.0);
                } else {
                    // real scalar derivative for a complex component -> imaginary part 0
                    $o[] = (float) $c;
                    $o[] = 0.0;
                }
            }
            return $o;
        };
    }
}
