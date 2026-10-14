<?php

declare(strict_types=1);

namespace Tessero\Optimize;

/**
 * Barrier subproblem for scipy.optimize trust-constr interior-point method
 * (port of scipy's BarrierSubproblem in _trustregion_constr/tr_interior_point.py):
 *
 *   minimize  fun(x) - barrier_parameter * sum(log s)
 *   s.t.      constr_eq(x)      = 0
 *             constr_ineq(x) + s = 0
 *
 * Works on the augmented variable z = [x, s]. Methods are public so they can be
 * passed to the equality-constrained SQP as [$sub, 'method'] callables.
 */
final class BarrierSubproblemState
{
    public bool $terminate = false;

    /** @var list<float> */
    private array $x0;
    /** @var list<float> */
    private array $s0;
    /** @var callable(list<float>): float */
    private $fun;
    /** @var callable(list<float>): list<float> */
    private $grad;
    /** @var callable(list<float>, list<float>, list<float>): callable */
    private $lagrHessX;
    private int $nVars;
    private int $nIneq;
    private int $nEq;
    /** @var callable(list<float>): array{0: list<float>, 1: list<float>} */
    private $constr;
    /** @var callable(list<float>): array{0: list<list<float>>, 1: list<list<float>>} */
    private $jac;
    private float $barrierParameter;
    private float $tolerance;
    /** @var list<bool> */
    private array $enforce;
    /** @var callable */
    private $globalStop;
    private float $xtol;
    /** @var list<float> */
    private array $lb;
    /** @var list<float> */
    private array $ub;
    private TrustConstr $owner;

    /**
     * @param list<float> $x0
     * @param list<float> $s0
     * @param callable(list<float>): float $fun
     * @param callable(list<float>): list<float> $grad
     * @param callable(list<float>, list<float>, list<float>): callable $lagrHessX
     * @param callable(list<float>): array{0: list<float>, 1: list<float>} $constr
     * @param callable(list<float>): array{0: list<list<float>>, 1: list<list<float>>} $jac
     * @param list<bool> $enforce
     * @param list<float> $lb
     * @param list<float> $ub
     */
    public function __construct(array $x0, array $s0, callable $fun, callable $grad, callable $lagrHessX, int $nVars, int $nIneq, int $nEq, callable $constr, callable $jac, float $barrierParameter, float $tolerance, array $enforce, callable $globalStop, float $xtol, array $lb, array $ub, TrustConstr $owner)
    {
        $this->x0 = $x0;
        $this->s0 = $s0;
        $this->fun = $fun;
        $this->grad = $grad;
        $this->lagrHessX = $lagrHessX;
        $this->nVars = $nVars;
        $this->nIneq = $nIneq;
        $this->nEq = $nEq;
        $this->constr = $constr;
        $this->jac = $jac;
        $this->barrierParameter = $barrierParameter;
        $this->tolerance = $tolerance;
        $this->enforce = $enforce;
        $this->globalStop = $globalStop;
        $this->xtol = $xtol;
        $this->lb = $lb;
        $this->ub = $ub;
        $this->owner = $owner;
    }

    public function update(float $bp, float $tol): void
    {
        $this->barrierParameter = $bp;
        $this->tolerance = $tol;
    }

    /** @param list<float> $z @return list<float> */
    private function getSlack(array $z): array
    {
        return array_slice($z, $this->nVars, $this->nIneq);
    }

    /** @param list<float> $z @return list<float> */
    private function getVars(array $z): array
    {
        return array_slice($z, 0, $this->nVars);
    }

    /**
     * @param list<float> $z
     * @return array{0: float, 1: list<float>}
     */
    public function functionAndConstraints(array $z): array
    {
        $x = $this->getVars($z);
        $s = $this->getSlack($z);
        $outOfBounds = false;
        for ($i = 0; $i < $this->nVars; $i++) {
            if ($x[$i] < $this->lb[$i] || $x[$i] > $this->ub[$i]) {
                $outOfBounds = true;
                break;
            }
        }
        if ($outOfBounds) {
            $f = INF;
            $cEq = array_fill(0, $this->nEq, 0.0);
            $cIneq = array_fill(0, $this->nIneq, 0.0);
        } else {
            $f = ($this->fun)($x);
            [$cEq, $cIneq] = ($this->constr)($x);
        }

        return [$this->computeFunction($f, $cIneq, $s), $this->computeConstr($cIneq, $cEq, $s)];
    }

    /**
     * @param list<float> $cIneq
     * @param list<float> $s
     */
    private function computeFunction(float $f, array $cIneq, array $s): float
    {
        // enforce_feasibility: lock s_i = -c_ineq_i (Nocedal & Wright p.576)
        foreach ($this->enforce as $i => $on) {
            if ($on) {
                $s[$i] = -$cIneq[$i];
            }
        }
        $sumLog = 0.0;
        foreach ($s as $si) {
            if ($si > 0) {
                $sumLog += log($si);
            } else {
                $sumLog = -INF;
                break;
            }
        }

        return $f - $this->barrierParameter * $sumLog;
    }

    /**
     * @param list<float> $cIneq
     * @param list<float> $cEq
     * @param list<float> $s
     * @return list<float>
     */
    private function computeConstr(array $cIneq, array $cEq, array $s): array
    {
        $out = $cEq;
        foreach ($cIneq as $i => $ci) {
            $out[] = $ci + $s[$i];
        }

        return array_values($out);
    }

    /**
     * Scaling: diag([ones(n_vars), s]). Returns [diag, matvec-closure].
     *
     * @param list<float> $z
     * @return array{0: list<float>, 1: callable}
     */
    public function scaling(array $z): array
    {
        $s = $this->getSlack($z);
        $diag = array_merge(array_fill(0, $this->nVars, 1.0), $s);
        $dot = static function (array $v) use ($diag): array {
            $out = [];
            foreach ($v as $i => $val) {
                $out[] = $diag[$i] * $val;
            }

            return $out;
        };

        return [$diag, $dot];
    }

    /**
     * Scaled gradient and Jacobian of the barrier problem.
     *
     * @param list<float> $z
     * @return array{0: list<float>, 1: list<list<float>>}
     */
    public function gradientAndJacobian(array $z): array
    {
        $x = $this->getVars($z);
        $s = $this->getSlack($z);
        $g = ($this->grad)($x);
        [$jEq, $jIneq] = ($this->jac)($x);

        $grad = $g;
        for ($i = 0; $i < $this->nIneq; $i++) {
            $grad[] = -$this->barrierParameter;
        }

        if ($this->nIneq === 0) {
            return [$grad, $jEq];
        }

        // jacobian = [ J_eq      0 ]
        //            [ J_ineq  diag(s) ]
        $nCols = $this->nVars + $this->nIneq;
        $J = [];
        foreach ($jEq as $row) {
            $r = $row;
            for ($c = 0; $c < $this->nIneq; $c++) {
                $r[] = 0.0;
            }
            $J[] = $r;
        }
        foreach ($jIneq as $i => $row) {
            $r = $row;
            for ($c = 0; $c < $this->nIneq; $c++) {
                $r[] = ($c === $i) ? $s[$i] : 0.0;
            }
            $J[] = $r;
        }

        return [$grad, $J];
    }

    /**
     * Scaled Lagrangian Hessian of the barrier problem as a matvec closure:
     *   [ Hx      0   ]
     *   [ 0    S Hs S ]
     *
     * @param list<float> $z
     * @param list<float> $v
     */
    public function lagrangianHessian(array $z, array $v): callable
    {
        $x = $this->getVars($z);
        $s = $this->getSlack($z);
        $vEq = array_slice($v, 0, $this->nEq);
        $vIneq = array_slice($v, $this->nEq, $this->nIneq);
        $Hx = ($this->lagrHessX)($x, $vEq, $vIneq); // matvec closure over x-space

        $sHsS = [];
        if ($this->nIneq > 0) {
            $vTail = array_slice($v, count($v) - $this->nIneq, $this->nIneq);
            for ($i = 0; $i < $this->nIneq; $i++) {
                // primal-dual for v>0, primal otherwise
                $sHsS[] = ($vTail[$i] > 0) ? $vTail[$i] * $s[$i] : $this->barrierParameter;
            }
        }

        $nVars = $this->nVars;
        $nIneq = $this->nIneq;

        return static function (array $vec) use ($Hx, $sHsS, $nVars, $nIneq): array {
            $vx = array_slice($vec, 0, $nVars);
            $out = $Hx($vx);
            if ($nIneq > 0) {
                $vs = array_slice($vec, $nVars, $nIneq);
                for ($i = 0; $i < $nIneq; $i++) {
                    $out[] = $sHsS[$i] * $vs[$i];
                }
            }

            return $out;
        };
    }

    /**
     * Stop criteria for the barrier subproblem SQP (local), consulting the
     * global criteria which may set $this->terminate.
     *
     * @param array<string, mixed> $state
     * @param list<float> $z
     * @param array{niter: int, stop_cond: int, hits_boundary: bool} $cgInfo
     */
    public function stopCriteria(array &$state, array $z, bool $lastFailed, float $optimality, float $constrViolation, float $trustRadius, float $penalty, array $cgInfo): bool
    {
        $x = $this->getVars($z);
        if (($this->globalStop)($state, $x, $lastFailed, $trustRadius, $penalty, $cgInfo, $this->barrierParameter, $this->tolerance)) {
            $this->terminate = true;

            return true;
        }
        $gCond = $optimality < $this->tolerance && $constrViolation < $this->tolerance;
        $xCond = $trustRadius < $this->xtol;

        return $gCond || $xCond;
    }
}
