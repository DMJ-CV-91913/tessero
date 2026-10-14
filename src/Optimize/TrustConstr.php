<?php

declare(strict_types=1);

namespace Tessero\Optimize;

use Tessero\Exceptions\TesseroException;

/**
 * scipy.optimize.minimize(method='trust-constr') -- the trust-region constrained
 * optimizer. A faithful pure-PHP port of scipy's _trustregion_constr package:
 *
 *   - equality_constrained_sqp  (Byrd-Omojokun trust-region SQP)
 *   - tr_interior_point         (barrier interior-point for inequalities)
 *   - projected_cg / modified_dogleg  (the QP subproblems)
 *   - dense QR-with-column-pivoting projections (SVD fallback on rank deficiency)
 *   - BFGS Hessian-update strategy (default when no hess supplied)
 *   - 2-point finite-difference gradient / Jacobian
 *
 * Supports every constraint class scipy's trust-constr accepts: box Bounds,
 * LinearConstraint (lb <= A x <= ub), and NonlinearConstraint (lb <= f(x) <= ub),
 * with equality, one-sided and two-sided interval bounds, mixed together.
 *
 * All arithmetic is plain float arrays -- no FFI/kernel dependency, matching the
 * rest of Tessero\Optimize.
 */
final class TrustConstr
{
    /** @var callable(list<float>): float */
    private $fun;
    /** @var (callable(list<float>): list<float>)|null */
    private $grad;
    public int $nfev = 0;
    public int $njev = 0;

    // ---- BFGS state (used when objective hess is not supplied) -------------
    /** @var list<list<float>>|null */
    private ?array $bfgsB = null;
    private bool $bfgsFirst = true;
    private int $bfgsN = 0;

    /**
     * @param callable(list<float>): float $fun
     * @param (callable(list<float>): list<float>)|null $grad
     */
    private function __construct(callable $fun, ?callable $grad)
    {
        $this->fun = $fun;
        $this->grad = $grad;
    }

    /**
     * Entry point from Minimize::minimize. $bounds is a list of [lo,ub] pairs or a Bounds
     * object; $options may carry 'constraints' (Bounds|LinearConstraint|NonlinearConstraint
     * or a list thereof), 'hess', 'gtol', 'xtol', 'barrier_tol', 'maxiter',
     * 'initial_tr_radius', 'initial_constr_penalty', 'initial_barrier_parameter',
     * 'initial_barrier_tolerance'.
     *
     * @param callable(list<float>): float $fun
     * @param list<float> $x0
     * @param (callable(list<float>): list<float>)|null $jac
     * @param list<array{0: float|null, 1: float|null}>|Bounds|null $bounds
     * @param array<string, mixed> $options
     */
    public static function run(callable $fun, array $x0, ?callable $jac, array|Bounds|null $bounds, array $options): OptimizeResult
    {
        $self = new self($fun, $jac);

        return $self->solve($x0, $bounds, $options);
    }

    /**
     * @param list<float> $x0
     * @param list<array{0: float|null, 1: float|null}>|Bounds|null $bounds
     * @param array<string, mixed> $options
     */
    private function solve(array $x0, array|Bounds|null $bounds, array $options): OptimizeResult
    {
        $n = count($x0);

        $gtol = (float) ($options['gtol'] ?? 1e-8);
        $xtol = (float) ($options['xtol'] ?? 1e-8);
        $barrierTol = (float) ($options['barrier_tol'] ?? 1e-8);
        $maxiter = (int) ($options['maxiter'] ?? 1000);
        $initTr = (float) ($options['initial_tr_radius'] ?? 1.0);
        $initPenalty = (float) ($options['initial_constr_penalty'] ?? 1.0);
        $initBp = (float) ($options['initial_barrier_parameter'] ?? 0.1);
        $initBtol = (float) ($options['initial_barrier_tolerance'] ?? 0.1);

        /** @var callable(list<float>): list<list<float>>|null $objHess */
        $objHess = (isset($options['hess']) && is_callable($options['hess'])) ? $options['hess'] : null;
        if ($objHess === null) {
            $this->bfgsInit($n);
        }

        // --- Assemble the list of "prepared" constraints -------------------
        // Each prepared constraint: ['fun'=>cb, 'jac'=>cb, 'hess'=>cb|null, 'lb'=>vec, 'ub'=>vec]
        $prepared = [];
        $constraints = $options['constraints'] ?? [];
        if ($constraints instanceof LinearConstraint || $constraints instanceof NonlinearConstraint) {
            $constraints = [$constraints];
        }
        foreach ($constraints as $c) {
            $prepared[] = $this->prepareConstraint($c, $n);
        }

        // --- Box bounds -> one prepared (identity) constraint --------------
        $lbFd = array_fill(0, $n, -INF);
        $ubFd = array_fill(0, $n, INF);
        $boundsLb = null;
        $boundsUb = null;
        if ($bounds instanceof Bounds) {
            [$boundsLb, $boundsUb] = $bounds->resolved($n);
        } elseif (is_array($bounds)) {
            $boundsLb = [];
            $boundsUb = [];
            foreach ($bounds as $pair) {
                $boundsLb[] = ($pair[0] === null) ? -INF : (float) $pair[0];
                $boundsUb[] = ($pair[1] === null) ? INF : (float) $pair[1];
            }
        }
        if ($boundsLb !== null) {
            $lbFd = $boundsLb;
            $ubFd = $boundsUb;
            // scipy nudges the trust-constr bounds outward by nextafter so the
            // strict-feasibility check never rejects a point exactly on the bound.
            $lbM = [];
            $ubM = [];
            for ($i = 0; $i < $n; $i++) {
                $lbM[] = is_finite($boundsLb[$i]) ? self::nextafter($boundsLb[$i], -INF) : $boundsLb[$i];
                $ubM[] = is_finite($boundsUb[$i]) ? self::nextafter($boundsUb[$i], INF) : $boundsUb[$i];
            }
            $I = self::eye($n);
            $prepared[] = [
                'fun' => static fn (array $x): array => $x,
                'jac' => static fn (array $x): array => $I,
                'hess' => null,
                'lb' => $lbM,
                'ub' => $ubM,
            ];
        }

        // --- Build the canonical constraint (eq + ineq) --------------------
        $canon = $this->buildCanonical($prepared, $n);
        $nEq = $canon['n_eq'];
        $nIneq = $canon['n_ineq'];
        $cfun = $canon['fun'];   // x -> [c_eq, c_ineq]
        $cjac = $canon['jac'];   // x -> [J_eq, J_ineq]
        $chess = $canon['hess']; // (x, v_eq, v_ineq) -> dense n x n matrix

        // Hessian of the Lagrangian as a matvec closure.
        $lagrHessX = function (array $x, array $vEq, array $vIneq) use ($objHess, $chess, $n): callable {
            $Hcon = $chess($x, $vEq, $vIneq); // dense n x n
            if ($objHess !== null) {
                $Hobj = $objHess($x); // dense n x n
                $H = self::matAdd($Hobj, $Hcon);
            } else {
                // BFGS: combine as matvec without forming a dense sum unnecessarily
                $B = $this->bfgsB;

                return function (array $p) use ($B, $Hcon): array {
                    return self::vecAdd(self::matVec($B, $p), self::matVec($Hcon, $p));
                };
            }

            return static fn (array $p): array => self::matVec($H, $p);
        };

        // Gradient wrapper that drives the BFGS update from consecutive points.
        $lastX = null;
        $lastG = null;
        $gradTracked = function (array $x) use (&$lastX, &$lastG, $objHess): array {
            $g = $this->gradAt($x);
            if ($objHess === null && $lastX !== null) {
                $this->bfgsUpdate(self::vecSub($x, $lastX), self::vecSub($g, $lastG));
            }
            $lastX = $x;
            $lastG = $g;

            return $g;
        };

        $f0 = $this->funAt($x0);
        $g0 = $gradTracked($x0);

        $state = [
            'nit' => 0,
            'tr_radius' => $initTr,
            'cg_niter' => 0,
            'status' => null,
            'x' => $x0,
            'optimality' => INF,
            'constr_violation' => INF,
            'v' => [],          // barrier eq-system multipliers
        ];

        if ($nIneq === 0) {
            // ---- equality_constrained_sqp -----------------------------------
            $stopCriteria = function (array &$state, array $x, bool $lastFailed, float $optimality, float $constrViolation, float $trRadius, float $penalty, array $cgInfo) use ($gtol, $xtol, $maxiter): bool {
                if (! $lastFailed) {
                    $state['x'] = $x;
                    $state['optimality'] = $optimality;
                    $state['constr_violation'] = $constrViolation;
                }
                $state['nit']++;
                $state['tr_radius'] = $trRadius;
                $state['cg_niter'] += $cgInfo['niter'];
                if ($optimality < $gtol && $constrViolation < $gtol) {
                    $state['status'] = 1;
                } elseif ($trRadius < $xtol) {
                    $state['status'] = 2;
                } elseif ($state['nit'] >= $maxiter) {
                    $state['status'] = 0;
                } else {
                    $state['status'] = null;
                }

                return in_array($state['status'], [0, 1, 2, 3], true);
            };

            $funAndConstr = function (array $x) use ($cfun): array {
                $f = $this->funAt($x);
                [$cEq, ] = $cfun($x);

                return [$f, $cEq];
            };
            $gradAndJac = function (array $x) use ($gradTracked, $cjac): array {
                $g = $gradTracked($x);
                [$jEq, ] = $cjac($x);

                return [$g, $jEq];
            };

            [$cEq0, ] = $cfun($x0);
            [$jEq0, ] = $cjac($x0);
            $lagrHess2 = static fn (array $x, array $v): callable => $lagrHessX($x, $v, []);

            $x = $this->equalityConstrainedSqp($funAndConstr, $gradAndJac, $lagrHess2, $x0, $f0, $g0, $cEq0, $jEq0, $stopCriteria, $state, $initPenalty, $initTr, null, null, null);
            $method = 'equality_constrained_sqp';
        } else {
            // ---- tr_interior_point ------------------------------------------
            [, $cIneq0] = $cfun($x0);
            $ipStop = function (array &$state, array $x, bool $lastFailed, float $trRadius, float $penalty, array $cgInfo, float $bp, float $btol) use ($gradTracked, $cfun, $cjac, $gtol, $xtol, $barrierTol, $maxiter, $nEq, $nIneq, $g0): bool {
                // True optimality / constraint violation of the ORIGINAL problem
                // using the barrier SQP multipliers stashed in $state['v'].
                $g = $this->gradAt($x);
                [$jEq, $jIneq] = $cjac($x);
                [$cEq, $cIneq] = $cfun($x);
                $cv = 0.0;
                foreach ($cEq as $v) {
                    $cv = max($cv, abs($v));
                }
                foreach ($cIneq as $v) {
                    $cv = max($cv, $v); // feasible iff c_ineq <= 0
                }
                $cv = max($cv, 0.0);
                $lag = $g;
                $vfull = $state['v'];
                if (count($vfull) >= $nEq + $nIneq) {
                    $vEq = array_slice($vfull, 0, $nEq);
                    $vIneq = array_slice($vfull, $nEq, $nIneq);
                    if ($nEq > 0) {
                        $lag = self::vecAdd($lag, self::matTVec($jEq, $vEq));
                    }
                    if ($nIneq > 0) {
                        $lag = self::vecAdd($lag, self::matTVec($jIneq, $vIneq));
                    }
                }
                $opt = self::infNorm($lag);

                $state['_opt'] = $opt;
                $state['_cv'] = $cv;
                $state['nit']++;
                $state['tr_radius'] = $trRadius;
                $state['cg_niter'] += $cgInfo['niter'];
                if ($opt < $gtol && $cv < $gtol) {
                    $state['status'] = 1;
                } elseif ($trRadius < $xtol && $bp < $barrierTol) {
                    $state['status'] = 2;
                } elseif ($state['nit'] >= $maxiter) {
                    $state['status'] = 0;
                } else {
                    $state['status'] = null;
                }
                $state['x'] = $x;
                $state['optimality'] = $opt;
                $state['constr_violation'] = $cv;

                return in_array($state['status'], [0, 1, 2, 3], true);
            };

            $x = $this->trInteriorPoint($this->fun, $gradTracked, $lagrHessX, $n, $nIneq, $nEq, $cfun, $cjac, $x0, $cIneq0, $ipStop, $canon['keep_feasible'], $xtol, $state, $initBp, $initBtol, $initPenalty, $initTr, $lbFd, $ubFd);
            $method = 'tr_interior_point';
        }

        $status = $state['status'] ?? 0;
        if (in_array($status, [1, 2], true) && $state['constr_violation'] > $gtol) {
            $status = 4;
        }
        $success = in_array($status, [1, 2], true);
        $messages = [
            0 => 'The maximum number of function evaluations is exceeded.',
            1 => '`gtol` termination condition is satisfied.',
            2 => '`xtol` termination condition is satisfied.',
            3 => '`callback` raised `StopIteration`.',
            4 => "Constraint violation exceeds 'gtol'",
        ];

        return new OptimizeResult(
            $x,
            $this->funAt($x),
            $success,
            $status,
            $messages[$status] ?? '',
            $state['nit'],
            $this->nfev,
            $this->njev,
            $this->gradAt($x),
            null,
            $method,
        );
    }

    // ======================================================================
    // equality_constrained_sqp  (Byrd-Omojokun trust-region SQP)
    // ======================================================================
    /**
     * @param callable(list<float>): array{0: float, 1: list<float>} $funAndConstr
     * @param callable(list<float>): array{0: list<float>, 1: list<list<float>>} $gradAndJac
     * @param callable(list<float>, list<float>): callable $lagrHess
     * @param list<float> $x0
     * @param list<float> $grad0
     * @param list<float> $constr0
     * @param list<list<float>> $jac0
     * @param callable $stopCriteria
     * @param array<string, mixed> $state
     * @param list<float>|null $trustLb
     * @param list<float>|null $trustUb
     * @param (callable(list<float>): array{0: list<float>, 1: callable})|null $scaling
     * @return list<float>
     */
    private function equalityConstrainedSqp(callable $funAndConstr, callable $gradAndJac, callable $lagrHess, array $x0, float $fun0, array $grad0, array $constr0, array $jac0, callable $stopCriteria, array &$state, float $initialPenalty, float $initialTrustRadius, ?array $trustLb, ?array $trustUb, ?callable $scaling): array
    {
        $PENALTY_FACTOR = 0.3;
        $LARGE = 0.9;
        $INTER = 0.3;
        $SUFF = 1e-8;
        $ENL_L = 7.0;
        $ENL_S = 2.0;
        $MAXR = 0.5;
        $MINR = 0.1;
        $SOC = 0.1;
        $TR = 0.8;
        $BOX = 0.5;

        $n = count($x0);
        if ($trustLb === null) {
            $trustLb = array_fill(0, $n, -INF);
        }
        if ($trustUb === null) {
            $trustUb = array_fill(0, $n, INF);
        }
        if ($scaling === null) {
            $scaling = static fn (array $z): array => [array_fill(0, count($z), 1.0), static fn (array $v): array => $v];
        }

        $x = $x0;
        $trustRadius = $initialTrustRadius;
        $penalty = $initialPenalty;
        $f = $fun0;
        $c = $grad0;
        $b = $constr0;
        $A = $jac0;
        [, $Sdot] = $scaling($x);
        $proj = $this->projections($A, $n);
        $v = self::neg2($proj['LS']($c));
        $H = $lagrHess($x, $v);
        $optimality = self::infNorm(count($A) > 0 ? self::vecAdd($c, self::matTVec($A, $v)) : $c);
        $constrViolation = count($b) > 0 ? self::infNorm($b) : 0.0;
        $cgInfo = ['niter' => 0, 'stop_cond' => 0, 'hits_boundary' => false];
        $state['v'] = $v;
        $lastFailed = false;

        while (! $stopCriteria($state, $x, $lastFailed, $optimality, $constrViolation, $trustRadius, $penalty, $cgInfo)) {
            $dn = $this->modifiedDogleg($A, $proj['Y'], $b, $TR * $trustRadius, self::scaleVec($BOX, $trustLb), self::scaleVec($BOX, $trustUb));
            $cT = self::vecAdd($H($dn), $c);
            $bT = array_fill(0, count($b), 0.0);
            $trustRadiusT = sqrt(max(0.0, $trustRadius ** 2 - self::norm22($dn)));
            $lbT = self::vecSub($trustLb, $dn);
            $ubT = self::vecSub($trustUb, $dn);
            [$dt, $cgInfo] = $this->projectedCg($H, $cT, $proj['Z'], $proj['Y'], $bT, $trustRadiusT, $lbT, $ubT);
            $d = self::vecAdd($dn, $dt);

            $quadraticModel = 0.5 * self::dot($H($d), $d) + self::dot($c, $d);
            $linearizedConstr = self::vecAdd(self::matVec($A, $d), $b);
            $vpred = max(1e-16, self::norm2vec($b) - self::norm2vec($linearizedConstr));
            $previousPenalty = $penalty;
            if ($quadraticModel > 0) {
                $newPenalty = $quadraticModel / ((1 - $PENALTY_FACTOR) * $vpred);
                $penalty = max($penalty, $newPenalty);
            }
            $predictedReduction = -$quadraticModel + $penalty * $vpred;

            $merit = $f + $penalty * self::norm2vec($b);
            $xNext = self::vecAdd($x, $Sdot($d));
            [$fNext, $bNext] = $funAndConstr($xNext);
            $meritNext = $fNext + $penalty * self::norm2vec($bNext);
            $actualReduction = $merit - $meritNext;
            $reductionRatio = $actualReduction / $predictedReduction;

            if ($reductionRatio < $SUFF && self::norm2vec($dn) <= $SOC * self::norm2vec($dt)) {
                $y = self::neg2($proj['Y']($bNext));
                [, $t, $intersect] = $this->boxIntersections($d, $y, $trustLb, $trustUb, false);
                $xSoc = self::vecAdd($x, $Sdot(self::vecAdd($d, self::scaleVec($t, $y))));
                [$fSoc, $bSoc] = $funAndConstr($xSoc);
                $meritSoc = $fSoc + $penalty * self::norm2vec($bSoc);
                $redSoc = ($merit - $meritSoc) / $predictedReduction;
                if ($intersect && $redSoc >= $SUFF) {
                    $xNext = $xSoc;
                    $fNext = $fSoc;
                    $bNext = $bSoc;
                    $reductionRatio = $redSoc;
                }
            }

            $normD = self::norm2vec($d);
            if ($reductionRatio >= $LARGE) {
                $trustRadius = max($ENL_L * $normD, $trustRadius);
            } elseif ($reductionRatio >= $INTER) {
                $trustRadius = max($ENL_S * $normD, $trustRadius);
            } elseif ($reductionRatio < $SUFF) {
                $trustReduction = (1 - $SUFF) / (1 - $reductionRatio);
                $newTrustRadius = $trustReduction * $normD;
                if ($newTrustRadius >= $MAXR * $trustRadius) {
                    $trustRadius *= $MAXR;
                } elseif ($newTrustRadius >= $MINR * $trustRadius) {
                    $trustRadius = $newTrustRadius;
                } else {
                    $trustRadius *= $MINR;
                }
            }

            if ($reductionRatio >= $SUFF) {
                $x = $xNext;
                $f = $fNext;
                $b = $bNext;
                [$c, $A] = $gradAndJac($x);
                [, $Sdot] = $scaling($x);
                $proj = $this->projections($A, $n);
                $v = self::neg2($proj['LS']($c));
                $H = $lagrHess($x, $v);
                $lastFailed = false;
                $optimality = self::infNorm(count($A) > 0 ? self::vecAdd($c, self::matTVec($A, $v)) : $c);
                $constrViolation = count($b) > 0 ? self::infNorm($b) : 0.0;
                $state['v'] = $v;
            } else {
                $penalty = $previousPenalty;
                $lastFailed = true;
            }
        }

        return $x;
    }

    // ======================================================================
    // tr_interior_point  (barrier interior-point)
    // ======================================================================
    /**
     * @param callable(list<float>): float $fun
     * @param callable(list<float>): list<float> $grad
     * @param callable(list<float>, list<float>, list<float>): callable $lagrHessX
     * @param callable(list<float>): array{0: list<float>, 1: list<float>} $constr
     * @param callable(list<float>): array{0: list<list<float>>, 1: list<list<float>>} $jac
     * @param list<float> $x0
     * @param list<float> $constrIneq0
     * @param list<bool> $enforceFeas
     * @param array<string, mixed> $state
     * @param list<float> $lbFd
     * @param list<float> $ubFd
     * @return list<float>
     */
    private function trInteriorPoint(callable $fun, callable $grad, callable $lagrHessX, int $nVars, int $nIneq, int $nEq, callable $constr, callable $jac, array $x0, array $constrIneq0, callable $stopCriteria, array $enforceFeas, float $xtol, array &$state, float $initBp, float $initTol, float $initPenalty, float $initTr, array $lbFd, array $ubFd): array
    {
        $BOUNDARY = 0.995;
        $DECAY = 0.2;
        $ENL = 5.0;

        if ($enforceFeas === []) {
            $enforceFeas = array_fill(0, $nIneq, false);
        }
        $bp = $initBp;
        $tol = $initTol;
        $tr = $initTr;
        $s0 = [];
        foreach ($constrIneq0 as $ci) {
            $s0[] = max(-1.5 * $ci, 1.0);
        }

        $sub = new BarrierSubproblemState($x0, $s0, $fun, $grad, $lagrHessX, $nVars, $nIneq, $nEq, $constr, $jac, $bp, $tol, $enforceFeas, $stopCriteria, $xtol, $lbFd, $ubFd, $this);

        $z = array_merge($x0, $s0);
        [$f0, $c0] = $sub->functionAndConstraints($z);
        [$g0, $j0] = $sub->gradientAndJacobian($z);

        $trustLb = array_merge(array_fill(0, $nVars, -INF), array_fill(0, $nIneq, -$BOUNDARY));
        $trustUb = array_fill(0, $nVars + $nIneq, INF);

        while (true) {
            $z = $this->equalityConstrainedSqp(
                [$sub, 'functionAndConstraints'],
                [$sub, 'gradientAndJacobian'],
                [$sub, 'lagrangianHessian'],
                $z, $f0, $g0, $c0, $j0,
                [$sub, 'stopCriteria'],
                $state, $initPenalty, $tr, $trustLb, $trustUb,
                [$sub, 'scaling'],
            );
            if ($sub->terminate) {
                break;
            }
            $tr = max($initTr, $ENL * $state['tr_radius']);
            $bp *= $DECAY;
            $tol *= $DECAY;
            $sub->update($bp, $tol);
            [$f0, $c0] = $sub->functionAndConstraints($z);
            [$g0, $j0] = $sub->gradientAndJacobian($z);
        }

        return array_slice($z, 0, $nVars);
    }

    // ======================================================================
    // projected_cg  (Algorithm 6.2, Gould/Hribar/Nocedal 2001)
    // ======================================================================
    /**
     * @param callable(list<float>): list<float> $Hdot
     * @param list<float> $c
     * @param callable(list<float>): list<float> $Zf
     * @param callable(list<float>): list<float> $Yf
     * @param list<float> $b
     * @param list<float>|null $lb
     * @param list<float>|null $ub
     * @return array{0: list<float>, 1: array{niter: int, stop_cond: int, hits_boundary: bool}}
     */
    private function projectedCg(callable $Hdot, array $c, callable $Zf, callable $Yf, array $b, float $trustRadius = INF, ?array $lb = null, ?array $ub = null, ?float $tol = null, ?int $maxIter = null, ?int $maxInfeasibleIter = null): array
    {
        $CLOSE = 1e-25;
        $n = count($c);
        $m = count($b);

        $x = $Yf(self::neg2($b));
        $r = $Zf(self::vecAdd($Hdot($x), $c));
        $g = $Zf($r);
        $p = self::neg2($g);
        $Hp = $Hdot($p);
        $rtG = self::norm22($g);

        $trDistance = $trustRadius - self::norm2vec($x);
        if ($trDistance < 0) {
            throw new TesseroException('trust-constr: projected CG subproblem has no solution.');
        }
        if ($trDistance < $CLOSE) {
            return [$x, ['niter' => 0, 'stop_cond' => 2, 'hits_boundary' => true]];
        }
        if ($tol === null) {
            $tol = max(min(0.01 * sqrt($rtG), 0.1 * $rtG), $CLOSE);
        }
        if ($lb === null) {
            $lb = array_fill(0, $n, -INF);
        }
        if ($ub === null) {
            $ub = array_fill(0, $n, INF);
        }
        if ($maxIter === null) {
            $maxIter = $n - $m;
        }
        $maxIter = min($maxIter, $n - $m);
        if ($maxInfeasibleIter === null) {
            $maxInfeasibleIter = $n - $m;
        }

        $hitsBoundary = false;
        $stopCond = 1;
        $counter = 0;
        $lastFeasibleX = array_fill(0, $n, 0.0);
        $k = 0;
        for ($i = 0; $i < $maxIter; $i++) {
            if ($rtG < $tol) {
                $stopCond = 4;
                break;
            }
            $k++;
            $ptHp = self::dot($Hp, $p);
            if ($ptHp <= 0) {
                if (is_infinite($trustRadius)) {
                    throw new TesseroException('trust-constr: negative curvature not allowed for unrestricted problems.');
                }
                [, $alpha, $intersect] = $this->boxSphereIntersections($x, $p, $lb, $ub, $trustRadius, true);
                if ($intersect) {
                    $x = self::vecAdd($x, self::scaleVec($alpha, $p));
                }
                $x = $this->reinforceBox($x, $lb, $ub);
                $stopCond = 3;
                $hitsBoundary = true;
                break;
            }
            $alpha = $rtG / $ptHp;
            $xNext = self::vecAdd($x, self::scaleVec($alpha, $p));
            if (self::norm2vec($xNext) >= $trustRadius) {
                [, $theta, $intersect] = $this->boxSphereIntersections($x, self::scaleVec($alpha, $p), $lb, $ub, $trustRadius, false);
                if ($intersect) {
                    $x = self::vecAdd($x, self::scaleVec($theta * $alpha, $p));
                }
                $x = $this->reinforceBox($x, $lb, $ub);
                $stopCond = 2;
                $hitsBoundary = true;
                break;
            }
            if ($this->insideBox($xNext, $lb, $ub)) {
                $counter = 0;
            } else {
                $counter++;
            }
            if ($counter > 0) {
                [, $theta, $intersect] = $this->boxSphereIntersections($x, self::scaleVec($alpha, $p), $lb, $ub, $trustRadius, false);
                if ($intersect) {
                    $lastFeasibleX = $this->reinforceBox(self::vecAdd($x, self::scaleVec($theta * $alpha, $p)), $lb, $ub);
                    $counter = 0;
                }
            }
            if ($counter > $maxInfeasibleIter) {
                break;
            }
            $rNext = self::vecAdd($r, self::scaleVec($alpha, $Hp));
            $gNext = $Zf($rNext);
            $rtGNext = self::norm22($gNext);
            $beta = $rtGNext / $rtG;
            $p = self::vecAdd(self::neg2($gNext), self::scaleVec($beta, $p));
            $x = $xNext;
            $g = $gNext;
            $r = $gNext;
            $rtG = self::norm22($g);
            $Hp = $Hdot($p);
        }

        if (! $this->insideBox($x, $lb, $ub)) {
            $x = $lastFeasibleX;
            $hitsBoundary = true;
        }

        return [$x, ['niter' => $k, 'stop_cond' => $stopCond, 'hits_boundary' => $hitsBoundary]];
    }

    // ======================================================================
    // modified_dogleg
    // ======================================================================
    /**
     * @param list<list<float>> $A
     * @param callable(list<float>): list<float> $Yf
     * @param list<float> $b
     * @param list<float> $lb
     * @param list<float> $ub
     * @return list<float>
     */
    private function modifiedDogleg(array $A, callable $Yf, array $b, float $trustRadius, array $lb, array $ub): array
    {
        $newtonPoint = self::neg2($Yf($b));
        if ($this->insideBox($newtonPoint, $lb, $ub) && self::norm2vec($newtonPoint) <= $trustRadius) {
            return $newtonPoint;
        }
        $g = self::matTVec($A, $b);
        $Ag = self::matVec($A, $g);
        $cauchyPoint = self::scaleVec(-self::dot($g, $g) / self::dot($Ag, $Ag), $g);
        $origin = array_fill(0, count($cauchyPoint), 0.0);

        $z = $cauchyPoint;
        $p = self::vecSub($newtonPoint, $cauchyPoint);
        [, $alpha, $intersect] = $this->boxSphereIntersections($z, $p, $lb, $ub, $trustRadius, false);
        if ($intersect) {
            $x1 = self::vecAdd($z, self::scaleVec($alpha, $p));
        } else {
            $z = $origin;
            $p = $cauchyPoint;
            [, $alpha, ] = $this->boxSphereIntersections($z, $p, $lb, $ub, $trustRadius, false);
            $x1 = self::vecAdd($z, self::scaleVec($alpha, $p));
        }
        $z = $origin;
        $p = $newtonPoint;
        [, $alpha, ] = $this->boxSphereIntersections($z, $p, $lb, $ub, $trustRadius, false);
        $x2 = self::vecAdd($z, self::scaleVec($alpha, $p));

        $r1 = self::norm2vec(self::vecAdd(self::matVec($A, $x1), $b));
        $r2 = self::norm2vec(self::vecAdd(self::matVec($A, $x2), $b));

        return $r1 < $r2 ? $x1 : $x2;
    }

    // ======================================================================
    // Intersections
    // ======================================================================
    /**
     * @param list<float> $z
     * @param list<float> $d
     * @return array{0: float, 1: float, 2: bool}
     */
    private function sphereIntersections(array $z, array $d, float $trustRadius, bool $entireLine): array
    {
        if (self::norm2vec($d) == 0.0) {
            return [0.0, 0.0, false];
        }
        if (is_infinite($trustRadius)) {
            return $entireLine ? [-INF, INF, true] : [0.0, 1.0, true];
        }
        $a = self::dot($d, $d);
        $b = 2 * self::dot($z, $d);
        $c = self::dot($z, $z) - $trustRadius ** 2;
        $disc = $b * $b - 4 * $a * $c;
        if ($disc < 0) {
            return [0.0, 0.0, false];
        }
        $sq = sqrt($disc);
        $aux = $b + ($b >= 0 ? $sq : -$sq);
        $ta = -$aux / (2 * $a);
        $tb = -2 * $c / $aux;
        if ($ta > $tb) {
            [$ta, $tb] = [$tb, $ta];
        }
        if ($entireLine) {
            return [$ta, $tb, true];
        }
        if ($tb < 0 || $ta > 1) {
            return [0.0, 0.0, false];
        }

        return [max(0.0, $ta), min(1.0, $tb), true];
    }

    /**
     * @param list<float> $z
     * @param list<float> $d
     * @param list<float> $lb
     * @param list<float> $ub
     * @return array{0: float, 1: float, 2: bool}
     */
    private function boxIntersections(array $z, array $d, array $lb, array $ub, bool $entireLine): array
    {
        if (self::norm2vec($d) == 0.0) {
            return [0.0, 0.0, false];
        }
        $n = count($d);
        // Check coordinates with d==0: if z violates bounds there, no intersection.
        for ($i = 0; $i < $n; $i++) {
            if ($d[$i] == 0.0) {
                if ($z[$i] < $lb[$i] || $z[$i] > $ub[$i]) {
                    return [0.0, 0.0, false];
                }
            }
        }
        $taCands = [];
        $tbCands = [];
        for ($i = 0; $i < $n; $i++) {
            if ($d[$i] == 0.0) {
                continue;
            }
            $tLb = ($lb[$i] - $z[$i]) / $d[$i];
            $tUb = ($ub[$i] - $z[$i]) / $d[$i];
            $taCands[] = min($tLb, $tUb);
            $tbCands[] = max($tLb, $tUb);
        }
        $ta = max($taCands);
        $tb = min($tbCands);
        $intersect = $ta <= $tb;
        if (! $entireLine) {
            if ($tb < 0 || $ta > 1) {
                return [0.0, 0.0, false];
            }
            $ta = max(0.0, $ta);
            $tb = min(1.0, $tb);
        }

        return [$ta, $tb, $intersect];
    }

    /**
     * @param list<float> $z
     * @param list<float> $d
     * @param list<float> $lb
     * @param list<float> $ub
     * @return array{0: float, 1: float, 2: bool}
     */
    private function boxSphereIntersections(array $z, array $d, array $lb, array $ub, float $trustRadius, bool $entireLine): array
    {
        [$taB, $tbB, $ib] = $this->boxIntersections($z, $d, $lb, $ub, $entireLine);
        [$taS, $tbS, $is] = $this->sphereIntersections($z, $d, $trustRadius, $entireLine);
        $ta = max($taB, $taS);
        $tb = min($tbB, $tbS);
        $intersect = $ib && $is && $ta <= $tb;

        return [$ta, $tb, $intersect];
    }

    /**
     * @param list<float> $x
     * @param list<float> $lb
     * @param list<float> $ub
     */
    private function insideBox(array $x, array $lb, array $ub): bool
    {
        foreach ($x as $i => $v) {
            if ($v < $lb[$i] || $v > $ub[$i]) {
                return false;
            }
        }

        return true;
    }

    /**
     * @param list<float> $x
     * @param list<float> $lb
     * @param list<float> $ub
     * @return list<float>
     */
    private function reinforceBox(array $x, array $lb, array $ub): array
    {
        $out = [];
        foreach ($x as $i => $v) {
            $out[] = min(max($v, $lb[$i]), $ub[$i]);
        }

        return $out;
    }

    // ======================================================================
    // projections: dense QR with column pivoting, SVD fallback.
    // Returns ['Z'=>cb, 'LS'=>cb, 'Y'=>cb] linear-operator closures.
    //   Z x  = x - A^T (A A^T)^-1 A x   (null-space projection)
    //   LS x = (A A^T)^-1 A x           (least-squares / pinv(A^T))
    //   Y x  = A^T (A A^T)^-1 x         (row-space / min-norm solve of A y = x)
    // ======================================================================
    /**
     * @param list<list<float>> $A  shape (m, n)
     * @return array{Z: callable, LS: callable, Y: callable, m: int, n: int}
     */
    private function projections(array $A, int $nHint = 0): array
    {
        $m = count($A);
        $n = $m > 0 ? count($A[0]) : $nHint;
        if ($m === 0) {
            return [
                'Z' => static fn (array $x): array => $x,
                'LS' => static fn (array $x): array => [],
                'Y' => static fn (array $x): array => array_fill(0, $n, 0.0),
                'm' => 0,
                'n' => $n,
            ];
        }

        $orthTol = 1e-12;
        $maxRefin = 3;
        $tol = 1e-15;

        // QR with column pivoting of A^T  (shape n x m).
        $At = self::transpose($A);
        [$Q, $R, $P] = self::pivotedQr($At); // Q: n x k, R: k x m, P: list length m
        $k = count($R);
        $lastRow = $R[$k - 1] ?? [];
        if ($k < $m || self::infNorm($lastRow) < $tol) {
            return $this->svdProjections($A, $m, $n, $orthTol, $maxRefin, $tol);
        }

        // solveR: R y = rhs  (R is k x m upper-triangular, k==m here).
        $solveR = static function (array $rhs) use ($R, $m): array {
            $y = array_fill(0, $m, 0.0);
            for ($i = $m - 1; $i >= 0; $i--) {
                $s = $rhs[$i];
                for ($j = $i + 1; $j < $m; $j++) {
                    $s -= $R[$i][$j] * $y[$j];
                }
                $y[$i] = $s / $R[$i][$i];
            }

            return $y;
        };
        $solveRT = static function (array $rhs) use ($R, $m): array {
            $w = array_fill(0, $m, 0.0);
            for ($i = 0; $i < $m; $i++) {
                $s = $rhs[$i];
                for ($j = 0; $j < $i; $j++) {
                    $s -= $R[$j][$i] * $w[$j];
                }
                $w[$i] = $s / $R[$i][$i];
            }

            return $w;
        };

        $Zf = function (array $x) use ($A, $Q, $P, $solveR, $m, $orthTol, $maxRefin): array {
            $aux = $solveR(self::matTVec($Q, $x)); // Q^T x  (Q is n x m -> matTVec gives m)
            $vv = array_fill(0, $m, 0.0);
            foreach ($P as $idx => $pj) {
                $vv[$pj] = $aux[$idx];
            }
            $z = self::vecSub($x, self::matTVec($A, $vv));
            $r = 0;
            while (self::orthogonality($A, $z) > $orthTol && $r < $maxRefin) {
                $aux = $solveR(self::matTVec($Q, $z));
                foreach ($P as $idx => $pj) {
                    $vv[$pj] = $aux[$idx];
                }
                $z = self::vecSub($z, self::matTVec($A, $vv));
                $r++;
            }

            return $z;
        };
        $LSf = static function (array $x) use ($Q, $P, $solveR, $m): array {
            $aux = $solveR(self::matTVec($Q, $x));
            $z = array_fill(0, $m, 0.0);
            foreach ($P as $idx => $pj) {
                $z[$pj] = $aux[$idx];
            }

            return $z;
        };
        $Yf = static function (array $x) use ($Q, $P, $solveRT): array {
            $permuted = [];
            foreach ($P as $pj) {
                $permuted[] = $x[$pj];
            }
            $aux = $solveRT($permuted);

            return self::matVec($Q, $aux); // Q aux (Q is n x m)
        };

        return ['Z' => $Zf, 'LS' => $LSf, 'Y' => $Yf, 'm' => $m, 'n' => $n];
    }

    /**
     * SVD-based projections (rank-deficient fallback).
     *
     * @param list<list<float>> $A
     * @return array{Z: callable, LS: callable, Y: callable, m: int, n: int}
     */
    private function svdProjections(array $A, int $m, int $n, float $orthTol, int $maxRefin, float $tol): array
    {
        [$U, $s, $Vt] = self::svdThin($A); // U: m x r, s: r, Vt: r x n
        // keep s > tol
        $keep = [];
        foreach ($s as $i => $sv) {
            if ($sv > $tol) {
                $keep[] = $i;
            }
        }
        $Uk = [];
        foreach ($U as $row) {
            $nr = [];
            foreach ($keep as $i) {
                $nr[] = $row[$i];
            }
            $Uk[] = $nr;
        }
        $sk = [];
        $Vtk = [];
        foreach ($keep as $i) {
            $sk[] = $s[$i];
            $Vtk[] = $Vt[$i];
        }
        $r = count($sk);

        // helper: inv(A A^T) applied via SVD building blocks
        // LS x = U (1/s) V^T x  ;  Z x = x - A^T LS x ; Y x = V (1/s) U^T x
        $VtDot = static function (array $x) use ($Vtk, $r): array {
            $out = [];
            for ($i = 0; $i < $r; $i++) {
                $out[] = self::dot($Vtk[$i], $x);
            }

            return $out;
        };
        $UDot = static function (array $a) use ($Uk, $m, $r): array {
            $out = array_fill(0, $m, 0.0);
            for ($row = 0; $row < $m; $row++) {
                $sVal = 0.0;
                for ($i = 0; $i < $r; $i++) {
                    $sVal += $Uk[$row][$i] * $a[$i];
                }
                $out[$row] = $sVal;
            }

            return $out;
        };
        $UtDot = static function (array $x) use ($Uk, $m, $r): array {
            $out = array_fill(0, $r, 0.0);
            for ($i = 0; $i < $r; $i++) {
                $sVal = 0.0;
                for ($row = 0; $row < $m; $row++) {
                    $sVal += $Uk[$row][$i] * $x[$row];
                }
                $out[$i] = $sVal;
            }

            return $out;
        };
        $VtTDot = static function (array $a) use ($Vtk, $n, $r): array {
            $out = array_fill(0, $n, 0.0);
            for ($i = 0; $i < $r; $i++) {
                $coeff = $a[$i];
                $row = $Vtk[$i];
                for ($j = 0; $j < $n; $j++) {
                    $out[$j] += $row[$j] * $coeff;
                }
            }

            return $out;
        };

        $LSf = static function (array $x) use ($VtDot, $UDot, $sk, $r): array {
            $a = $VtDot($x);
            for ($i = 0; $i < $r; $i++) {
                $a[$i] /= $sk[$i];
            }

            return $UDot($a);
        };
        $Zf = function (array $x) use ($A, $VtDot, $UDot, $sk, $r, $orthTol, $maxRefin): array {
            $apply = function (array $xx) use ($VtDot, $UDot, $sk, $r): array {
                $a = $VtDot($xx);
                for ($i = 0; $i < $r; $i++) {
                    $a[$i] /= $sk[$i];
                }

                return $UDot($a);
            };
            $z = self::vecSub($x, self::matTVec($A, $apply($x)));
            $k = 0;
            while (self::orthogonality($A, $z) > $orthTol && $k < $maxRefin) {
                $z = self::vecSub($z, self::matTVec($A, $apply($z)));
                $k++;
            }

            return $z;
        };
        $Yf = static function (array $x) use ($UtDot, $VtTDot, $sk, $r): array {
            $a = $UtDot($x);
            for ($i = 0; $i < $r; $i++) {
                $a[$i] /= $sk[$i];
            }

            return $VtTDot($a);
        };

        return ['Z' => $Zf, 'LS' => $LSf, 'Y' => $Yf, 'm' => $m, 'n' => $n];
    }

    /**
     * orthogonality(A, g) = ||A g|| / (||A||_F ||g||)   (projections refinement test).
     *
     * @param list<list<float>> $A
     * @param list<float> $g
     */
    private static function orthogonality(array $A, array $g): float
    {
        $normG = self::norm2vec($g);
        $normAf = 0.0;
        foreach ($A as $row) {
            foreach ($row as $v) {
                $normAf += $v * $v;
            }
        }
        $normAf = sqrt($normAf);
        if ($normG == 0.0 || $normAf == 0.0) {
            return 0.0;
        }

        return self::norm2vec(self::matVec($A, $g)) / ($normAf * $normG);
    }

    // ======================================================================
    // Canonical constraint builder
    // ======================================================================
    /**
     * @param list<array{fun: callable, jac: callable, hess: callable|null, lb: list<float>, ub: list<float>}> $prepared
     * @return array{n_eq: int, n_ineq: int, fun: callable, jac: callable, hess: callable, keep_feasible: list<bool>}
     */
    private function buildCanonical(array $prepared, int $n): array
    {
        $specs = [];
        $totalEq = 0;
        $totalIneq = 0;
        foreach ($prepared as $con) {
            $lb = $con['lb'];
            $ub = $con['ub'];
            $m = count($lb);
            $equal = [];
            $less = [];
            $greater = [];
            $interval = [];
            for ($i = 0; $i < $m; $i++) {
                $loInf = is_infinite($lb[$i]) && $lb[$i] < 0;
                $upInf = is_infinite($ub[$i]) && $ub[$i] > 0;
                if ($lb[$i] === $ub[$i]) {
                    $equal[] = $i;
                } elseif ($loInf && ! $upInf) {
                    $less[] = $i;
                } elseif ($upInf && ! $loInf) {
                    $greater[] = $i;
                } elseif (! $loInf && ! $upInf) {
                    $interval[] = $i;
                }
                // both-inf rows contribute nothing
            }
            $nEq = count($equal);
            $nIneq = count($less) + count($greater) + 2 * count($interval);
            $specs[] = [
                'fun' => $con['fun'],
                'jac' => $con['jac'],
                'hess' => $con['hess'],
                'm' => $m,
                'lb' => $lb,
                'ub' => $ub,
                'equal' => $equal,
                'less' => $less,
                'greater' => $greater,
                'interval' => $interval,
                'n_eq' => $nEq,
                'n_ineq' => $nIneq,
            ];
            $totalEq += $nEq;
            $totalIneq += $nIneq;
        }

        $fun = function (array $x) use ($specs): array {
            $eq = [];
            $ineq = [];
            foreach ($specs as $s) {
                $f = ($s['fun'])($x);
                if (! is_array($f)) {
                    $f = [$f];
                }
                $f = array_values($f);
                foreach ($s['equal'] as $i) {
                    $eq[] = $f[$i] - $s['lb'][$i];
                }
                foreach ($s['less'] as $i) {
                    $ineq[] = $f[$i] - $s['ub'][$i];
                }
                foreach ($s['greater'] as $i) {
                    $ineq[] = $s['lb'][$i] - $f[$i];
                }
                foreach ($s['interval'] as $i) {
                    $ineq[] = $f[$i] - $s['ub'][$i];
                }
                foreach ($s['interval'] as $i) {
                    $ineq[] = $s['lb'][$i] - $f[$i];
                }
            }

            return [$eq, $ineq];
        };

        $jac = function (array $x) use ($specs, $n): array {
            $jEq = [];
            $jIneq = [];
            foreach ($specs as $s) {
                $J = ($s['jac'])($x); // m x n
                $J = array_map('array_values', $J);
                foreach ($s['equal'] as $i) {
                    $jEq[] = $J[$i];
                }
                foreach ($s['less'] as $i) {
                    $jIneq[] = $J[$i];
                }
                foreach ($s['greater'] as $i) {
                    $jIneq[] = self::negRow($J[$i]);
                }
                foreach ($s['interval'] as $i) {
                    $jIneq[] = $J[$i];
                }
                foreach ($s['interval'] as $i) {
                    $jIneq[] = self::negRow($J[$i]);
                }
            }

            return [$jEq, $jIneq];
        };

        $hess = function (array $x, array $vEq, array $vIneq) use ($specs, $n): array {
            $total = self::zeros($n, $n);
            $ie = 0;
            $ii = 0;
            foreach ($specs as $s) {
                $vcEq = array_slice($vEq, $ie, $s['n_eq']);
                $ie += $s['n_eq'];
                $nle = count($s['less']);
                $nge = count($s['greater']);
                $niv = count($s['interval']);
                $vL = array_slice($vIneq, $ii, $nle);
                $ii += $nle;
                $vG = array_slice($vIneq, $ii, $nge);
                $ii += $nge;
                $vIl = array_slice($vIneq, $ii, $niv);
                $ii += $niv;
                $vIg = array_slice($vIneq, $ii, $niv);
                $ii += $niv;

                if ($s['hess'] === null) {
                    continue; // linear constraint -> zero Hessian
                }
                $v = array_fill(0, $s['m'], 0.0);
                foreach ($s['equal'] as $k => $i) {
                    $v[$i] = $vcEq[$k];
                }
                foreach ($s['less'] as $k => $i) {
                    $v[$i] = $vL[$k];
                }
                foreach ($s['greater'] as $k => $i) {
                    $v[$i] = -$vG[$k];
                }
                foreach ($s['interval'] as $k => $i) {
                    $v[$i] = $vIl[$k] - $vIg[$k];
                }
                $Hc = ($s['hess'])($x, $v); // dense n x n
                $total = self::matAdd($total, $Hc);
            }

            return $total;
        };

        $keepFeasible = array_fill(0, $totalIneq, false);

        return [
            'n_eq' => $totalEq,
            'n_ineq' => $totalIneq,
            'fun' => $fun,
            'jac' => $jac,
            'hess' => $hess,
            'keep_feasible' => $keepFeasible,
        ];
    }

    /**
     * Convert a scipy-style constraint object into prepared form
     * ['fun','jac','hess','lb'(vec),'ub'(vec)].
     *
     * @return array{fun: callable, jac: callable, hess: callable|null, lb: list<float>, ub: list<float>}
     */
    private function prepareConstraint(LinearConstraint|NonlinearConstraint $c, int $n): array
    {
        if ($c instanceof LinearConstraint) {
            $A = $c->A;
            $m = count($A);
            $fun = static fn (array $x): array => self::matVec($A, $x);
            $jac = static fn (array $x): array => $A;
            $hess = null; // linear -> zero
            [$lb, $ub] = $this->resolveBounds($c->lb, $c->ub, $m);

            return ['fun' => $fun, 'jac' => $jac, 'hess' => $hess, 'lb' => $lb, 'ub' => $ub];
        }

        // NonlinearConstraint
        $cfun = $c->fun;
        $wrappedFun = static function (array $x) use ($cfun): array {
            $r = $cfun($x);

            return is_array($r) ? array_values(array_map('floatval', $r)) : [(float) $r];
        };
        // determine m from a probe is deferred; we resolve bounds lazily with
        // the length of the constraint value at x (handled in buildCanonical via
        // lb/ub vectors) -> so we must know m now. Probe with a dummy to size it.
        // We size lb/ub using the declared bounds if arrays, else broadcast after probe.
        $lbRaw = $c->lb;
        $ubRaw = $c->ub;
        // Jacobian: user-supplied callable or 2-point finite difference.
        if (is_callable($c->jac)) {
            $cjac = $c->jac;
            $jac = static function (array $x) use ($cjac): array {
                $J = $cjac($x);

                return array_map(static fn ($row): array => array_values(array_map('floatval', is_array($row) ? $row : [$row])), $J);
            };
        } else {
            $jac = fn (array $x): array => $this->fdJac($wrappedFun, $x);
        }
        $hess = is_callable($c->hess) ? $c->hess : null;

        // resolve bound vectors; need m. Probe is unavoidable if scalar bounds.
        $mProbe = null;
        if (is_array($lbRaw)) {
            $mProbe = count($lbRaw);
        } elseif (is_array($ubRaw)) {
            $mProbe = count($ubRaw);
        }
        // store deferred-length info by wrapping lb/ub resolution now if known,
        // otherwise we do a single evaluation at a placeholder -- but we don't
        // have x0 here. Caller passes constraints before x0 probe; so require
        // vector bounds OR we resolve at first canonical call. Simplest: resolve
        // using a lazy constraint whose lb/ub get fixed on first fun() call.
        if ($mProbe === null) {
            // Must know constraint dimension. Defer: wrap so buildCanonical sees
            // vectors after a one-time probe stored in a closure cache.
            $cache = ['lb' => null, 'ub' => null, 'm' => null];
            $resolve = function (array $x) use (&$cache, $wrappedFun, $lbRaw, $ubRaw): void {
                if ($cache['m'] === null) {
                    $m = count($wrappedFun($x));
                    $cache['m'] = $m;
                    [$cache['lb'], $cache['ub']] = $this->resolveBounds($lbRaw, $ubRaw, $m);
                }
            };
            // We cannot return scalar lb/ub into buildCanonical (needs vectors).
            // Resolve immediately requires an x; throw a clear error instead so
            // callers pass vector bounds or we probe with zeros. Probe with zeros:
            $xProbe = array_fill(0, $n, 0.0);
            $m = count($wrappedFun($xProbe));
            [$lb, $ub] = $this->resolveBounds($lbRaw, $ubRaw, $m);
        } else {
            [$lb, $ub] = $this->resolveBounds($lbRaw, $ubRaw, $mProbe);
        }

        return ['fun' => $wrappedFun, 'jac' => $jac, 'hess' => $hess, 'lb' => $lb, 'ub' => $ub];
    }

    /**
     * @param list<float>|float $lb
     * @param list<float>|float $ub
     * @return array{0: list<float>, 1: list<float>}
     */
    private function resolveBounds(array|float|int $lb, array|float|int $ub, int $m): array
    {
        $lbv = is_array($lb) ? array_map('floatval', array_values($lb)) : array_fill(0, $m, (float) $lb);
        $ubv = is_array($ub) ? array_map('floatval', array_values($ub)) : array_fill(0, $m, (float) $ub);

        return [$lbv, $ubv];
    }

    // ======================================================================
    // Objective fun / grad / FD helpers
    // ======================================================================
    /** @param list<float> $x */
    private function funAt(array $x): float
    {
        $this->nfev++;

        return (float) ($this->fun)($x);
    }

    /**
     * @param list<float> $x
     * @return list<float>
     */
    private function gradAt(array $x): array
    {
        if ($this->grad !== null) {
            $this->njev++;

            return array_values(array_map('floatval', ($this->grad)($x)));
        }

        return $this->fdGrad($this->fun, $x);
    }

    /**
     * 2-point finite-difference gradient (scipy's default for trust-constr).
     *
     * @param callable(list<float>): float $f
     * @param list<float> $x
     * @return list<float>
     */
    private function fdGrad(callable $f, array $x): array
    {
        $n = count($x);
        $f0 = $this->funAt($x);
        $g = [];
        $eps = sqrt(2.220446049250313e-16);
        for ($i = 0; $i < $n; $i++) {
            $h = $eps * max(1.0, abs($x[$i]));
            $xp = $x;
            $xp[$i] += $h;
            $g[] = ($this->funAt($xp) - $f0) / $h;
        }

        return $g;
    }

    /**
     * 2-point finite-difference Jacobian.
     *
     * @param callable(list<float>): list<float> $fun
     * @param list<float> $x
     * @return list<list<float>>
     */
    private function fdJac(callable $fun, array $x): array
    {
        $n = count($x);
        $f0 = array_values(array_map('floatval', $fun($x)));
        $m = count($f0);
        $eps = sqrt(2.220446049250313e-16);
        $cols = [];
        for ($i = 0; $i < $n; $i++) {
            $h = $eps * max(1.0, abs($x[$i]));
            $xp = $x;
            $xp[$i] += $h;
            $fp = array_values(array_map('floatval', $fun($xp)));
            $col = [];
            for ($r = 0; $r < $m; $r++) {
                $col[] = ($fp[$r] - $f0[$r]) / $h;
            }
            $cols[] = $col;
        }
        // transpose cols (n x m) -> J (m x n)
        $J = self::zeros($m, $n);
        for ($r = 0; $r < $m; $r++) {
            for ($i = 0; $i < $n; $i++) {
                $J[$r][$i] = $cols[$i][$r];
            }
        }

        return $J;
    }

    // ======================================================================
    // BFGS Hessian update strategy (dense 'hess' form, skip_update exception)
    // ======================================================================
    private function bfgsInit(int $n): void
    {
        $this->bfgsN = $n;
        $this->bfgsFirst = true;
        $this->bfgsB = self::eye($n);
    }

    /**
     * @param list<float> $dx
     * @param list<float> $dg
     */
    private function bfgsUpdate(array $dx, array $dg): void
    {
        $minCurv = 1e-8;
        if (self::allZero($dx) || self::allZero($dg)) {
            return;
        }
        if ($this->bfgsFirst) {
            $this->bfgsB = self::matScale($this->bfgsAutoScale($dx, $dg), $this->bfgsB);
            $this->bfgsFirst = false;
        }
        $w = $dx;
        $z = $dg;
        $wz = self::dot($w, $z);
        $Bw = self::matVec($this->bfgsB, $w);
        $wBw = self::dot($Bw, $w);
        if ($wBw <= 0.0) {
            $scale = $this->bfgsAutoScale($dx, $dg);
            $this->bfgsB = self::matScale($scale, self::eye($this->bfgsN));
            $Bw = self::matVec($this->bfgsB, $w);
            $wBw = self::dot($Bw, $w);
        }
        if ($wz <= $minCurv * $wBw) {
            return; // skip_update
        }
        // B <- B + y y^T / wz - (B s)(B s)^T / sBs
        $n = $this->bfgsN;
        $B = $this->bfgsB;
        for ($i = 0; $i < $n; $i++) {
            $yi = $z[$i];
            $bi = $Bw[$i];
            for ($j = 0; $j < $n; $j++) {
                $B[$i][$j] += $yi * $z[$j] / $wz - $bi * $Bw[$j] / $wBw;
            }
        }
        $this->bfgsB = $B;
    }

    /**
     * @param list<float> $dx
     * @param list<float> $dg
     */
    private function bfgsAutoScale(array $dx, array $dg): float
    {
        $s2 = self::dot($dx, $dx);
        $y2 = self::dot($dg, $dg);
        $ys = abs(self::dot($dg, $dx));
        if ($ys == 0.0 || $y2 == 0.0 || $s2 == 0.0) {
            return 1.0;
        }

        return $y2 / $ys;
    }

    // ======================================================================
    // Dense linear algebra helpers (self-contained, pure PHP)
    // ======================================================================
    /**
     * Householder QR with column pivoting of B (p x q). Returns [Q(p x k), R(k x q), piv],
     * economic k = min(p, q), such that B[:, piv] == Q R.
     *
     * @param list<list<float>> $B
     * @return array{0: list<list<float>>, 1: list<list<float>>, 2: list<int>}
     */
    private static function pivotedQr(array $B): array
    {
        $p = count($B);
        $q = $p > 0 ? count($B[0]) : 0;
        $k = min($p, $q);
        // work on a copy R (p x q); Q starts as I_p
        $R = array_map(static fn ($row): array => array_map('floatval', $row), $B);
        $Q = self::eye($p);
        $piv = range(0, $q - 1);

        for ($i = 0; $i < $k; $i++) {
            // choose pivot column with max norm over rows i..p-1
            $best = $i;
            $bestN = -1.0;
            for ($j = $i; $j < $q; $j++) {
                $nrm = 0.0;
                for ($r = $i; $r < $p; $r++) {
                    $nrm += $R[$r][$j] * $R[$r][$j];
                }
                if ($nrm > $bestN) {
                    $bestN = $nrm;
                    $best = $j;
                }
            }
            if ($best !== $i) {
                for ($r = 0; $r < $p; $r++) {
                    [$R[$r][$i], $R[$r][$best]] = [$R[$r][$best], $R[$r][$i]];
                }
                [$piv[$i], $piv[$best]] = [$piv[$best], $piv[$i]];
            }
            // Householder on column i, rows i..p-1
            $xnorm = 0.0;
            for ($r = $i; $r < $p; $r++) {
                $xnorm += $R[$r][$i] * $R[$r][$i];
            }
            $xnorm = sqrt($xnorm);
            if ($xnorm == 0.0) {
                continue;
            }
            $x0 = $R[$i][$i];
            $alpha = ($x0 >= 0 ? -1.0 : 1.0) * $xnorm;
            $v = array_fill(0, $p, 0.0);
            for ($r = $i; $r < $p; $r++) {
                $v[$r] = $R[$r][$i];
            }
            $v[$i] -= $alpha;
            $vnorm = 0.0;
            for ($r = $i; $r < $p; $r++) {
                $vnorm += $v[$r] * $v[$r];
            }
            $vnorm = sqrt($vnorm);
            if ($vnorm < 1e-300) {
                continue;
            }
            for ($r = $i; $r < $p; $r++) {
                $v[$r] /= $vnorm;
            }
            // R[i:,:] -= 2 v (v^T R[i:,:])
            for ($col = 0; $col < $q; $col++) {
                $dotv = 0.0;
                for ($r = $i; $r < $p; $r++) {
                    $dotv += $v[$r] * $R[$r][$col];
                }
                $dotv *= 2.0;
                for ($r = $i; $r < $p; $r++) {
                    $R[$r][$col] -= $dotv * $v[$r];
                }
            }
            // Q[:,i:] -= 2 (Q[:,i:] v) v^T
            for ($row = 0; $row < $p; $row++) {
                $dotv = 0.0;
                for ($r = $i; $r < $p; $r++) {
                    $dotv += $Q[$row][$r] * $v[$r];
                }
                $dotv *= 2.0;
                for ($r = $i; $r < $p; $r++) {
                    $Q[$row][$r] -= $dotv * $v[$r];
                }
            }
        }

        // economic: Q -> p x k, R -> k x q
        $Qe = [];
        for ($row = 0; $row < $p; $row++) {
            $nr = [];
            for ($c = 0; $c < $k; $c++) {
                $nr[] = $Q[$row][$c];
            }
            $Qe[] = $nr;
        }
        $Re = [];
        for ($r = 0; $r < $k; $r++) {
            $Re[] = $R[$r];
        }

        return [$Qe, $Re, $piv];
    }

    /**
     * Thin SVD of A (m x n): returns [U(m x r), s(r), Vt(r x n)] via eigen-decomp
     * of the smaller Gram matrix. r = min(m, n). Adequate for the small dense
     * Jacobians arising here.
     *
     * @param list<list<float>> $A
     * @return array{0: list<list<float>>, 1: list<float>, 2: list<list<float>>}
     */
    private static function svdThin(array $A): array
    {
        $m = count($A);
        $n = $m > 0 ? count($A[0]) : 0;
        // Work with the smaller Gram matrix.
        if ($m <= $n) {
            // G = A A^T (m x m); eigenpairs -> U, s; V = A^T U / s
            $G = self::zeros($m, $m);
            for ($i = 0; $i < $m; $i++) {
                for ($j = $i; $j < $m; $j++) {
                    $s = self::dot($A[$i], $A[$j]);
                    $G[$i][$j] = $s;
                    $G[$j][$i] = $s;
                }
            }
            [$evals, $evecs] = self::jacobiEig($G); // evals desc, evecs columns
            $r = $m;
            $sVals = [];
            $U = self::zeros($m, $r);
            for ($c = 0; $c < $r; $c++) {
                $lam = max(0.0, $evals[$c]);
                $sVals[] = sqrt($lam);
                for ($i = 0; $i < $m; $i++) {
                    $U[$i][$c] = $evecs[$i][$c];
                }
            }
            $Vt = self::zeros($r, $n);
            for ($c = 0; $c < $r; $c++) {
                if ($sVals[$c] > 1e-300) {
                    // v_c = A^T u_c / s_c
                    $uc = [];
                    for ($i = 0; $i < $m; $i++) {
                        $uc[] = $U[$i][$c];
                    }
                    $vc = self::matTVec($A, $uc); // n-vector
                    $inv = 1.0 / $sVals[$c];
                    for ($j = 0; $j < $n; $j++) {
                        $Vt[$c][$j] = $vc[$j] * $inv;
                    }
                }
            }

            return [$U, $sVals, $Vt];
        }
        // m > n: G = A^T A (n x n); eigenpairs -> V, s; U = A V / s
        $G = self::zeros($n, $n);
        $At = self::transpose($A);
        for ($i = 0; $i < $n; $i++) {
            for ($j = $i; $j < $n; $j++) {
                $s = self::dot($At[$i], $At[$j]);
                $G[$i][$j] = $s;
                $G[$j][$i] = $s;
            }
        }
        [$evals, $evecs] = self::jacobiEig($G);
        $r = $n;
        $sVals = [];
        $Vt = self::zeros($r, $n);
        for ($c = 0; $c < $r; $c++) {
            $lam = max(0.0, $evals[$c]);
            $sVals[] = sqrt($lam);
            for ($j = 0; $j < $n; $j++) {
                $Vt[$c][$j] = $evecs[$j][$c];
            }
        }
        $U = self::zeros($m, $r);
        for ($c = 0; $c < $r; $c++) {
            if ($sVals[$c] > 1e-300) {
                $vc = $Vt[$c];
                $uc = self::matVec($A, $vc); // m-vector
                $inv = 1.0 / $sVals[$c];
                for ($i = 0; $i < $m; $i++) {
                    $U[$i][$c] = $uc[$i] * $inv;
                }
            }
        }

        return [$U, $sVals, $Vt];
    }

    /**
     * Jacobi eigenvalue algorithm for a symmetric matrix. Returns [evals(desc), evecs]
     * with eigenvectors as columns, sorted by descending eigenvalue.
     *
     * @param list<list<float>> $A
     * @return array{0: list<float>, 1: list<list<float>>}
     */
    private static function jacobiEig(array $A): array
    {
        $n = count($A);
        $a = array_map(static fn ($row): array => array_map('floatval', $row), $A);
        $V = self::eye($n);
        for ($sweep = 0; $sweep < 100; $sweep++) {
            $off = 0.0;
            for ($p = 0; $p < $n; $p++) {
                for ($q = $p + 1; $q < $n; $q++) {
                    $off += $a[$p][$q] * $a[$p][$q];
                }
            }
            if ($off < 1e-30) {
                break;
            }
            for ($p = 0; $p < $n; $p++) {
                for ($q = $p + 1; $q < $n; $q++) {
                    if (abs($a[$p][$q]) < 1e-300) {
                        continue;
                    }
                    $app = $a[$p][$p];
                    $aqq = $a[$q][$q];
                    $apq = $a[$p][$q];
                    $phi = 0.5 * atan2(2 * $apq, $aqq - $app);
                    $cs = cos($phi);
                    $sn = sin($phi);
                    for ($i = 0; $i < $n; $i++) {
                        $aip = $a[$i][$p];
                        $aiq = $a[$i][$q];
                        $a[$i][$p] = $cs * $aip - $sn * $aiq;
                        $a[$i][$q] = $sn * $aip + $cs * $aiq;
                    }
                    for ($i = 0; $i < $n; $i++) {
                        $api = $a[$p][$i];
                        $aqi = $a[$q][$i];
                        $a[$p][$i] = $cs * $api - $sn * $aqi;
                        $a[$q][$i] = $sn * $api + $cs * $aqi;
                    }
                    for ($i = 0; $i < $n; $i++) {
                        $vip = $V[$i][$p];
                        $viq = $V[$i][$q];
                        $V[$i][$p] = $cs * $vip - $sn * $viq;
                        $V[$i][$q] = $sn * $vip + $cs * $viq;
                    }
                }
            }
        }
        $evals = [];
        for ($i = 0; $i < $n; $i++) {
            $evals[] = $a[$i][$i];
        }
        // sort descending
        $idx = range(0, $n - 1);
        usort($idx, static fn ($i, $j): int => $evals[$j] <=> $evals[$i]);
        $sortedE = [];
        $sortedV = self::zeros($n, $n);
        foreach ($idx as $c => $orig) {
            $sortedE[] = $evals[$orig];
            for ($i = 0; $i < $n; $i++) {
                $sortedV[$i][$c] = $V[$i][$orig];
            }
        }

        return [$sortedE, $sortedV];
    }

    // --- small vector/matrix primitives -----------------------------------
    /**
     * @param list<float> $a
     * @param list<float> $b
     */
    private static function dot(array $a, array $b): float
    {
        $s = 0.0;
        foreach ($a as $i => $v) {
            $s += $v * $b[$i];
        }

        return $s;
    }

    /** @param list<float> $a @param list<float> $b @return list<float> */
    private static function vecAdd(array $a, array $b): array
    {
        $out = [];
        foreach ($a as $i => $v) {
            $out[] = $v + $b[$i];
        }

        return $out;
    }

    /** @param list<float> $a @param list<float> $b @return list<float> */
    private static function vecSub(array $a, array $b): array
    {
        $out = [];
        foreach ($a as $i => $v) {
            $out[] = $v - $b[$i];
        }

        return $out;
    }

    /** @param list<float> $a @return list<float> */
    private static function neg2(array $a): array
    {
        return array_map(static fn (float $v): float => -$v, $a);
    }

    /** @param list<float> $a @return list<float> */
    private static function scaleVec(float $s, array $a): array
    {
        return array_map(static fn (float $v): float => $s * $v, $a);
    }

    /** @param list<float> $a */
    private static function norm2vec(array $a): float
    {
        return sqrt(self::norm22($a));
    }

    /** @param list<float> $a */
    private static function norm22(array $a): float
    {
        $s = 0.0;
        foreach ($a as $v) {
            $s += $v * $v;
        }

        return $s;
    }

    /** @param list<float> $a */
    private static function infNorm(array $a): float
    {
        $m = 0.0;
        foreach ($a as $v) {
            $av = abs($v);
            if ($av > $m) {
                $m = $av;
            }
        }

        return $m;
    }

    /** @param list<float> $a */
    private static function allZero(array $a): bool
    {
        foreach ($a as $v) {
            if ($v !== 0.0) {
                return false;
            }
        }

        return true;
    }

    /** @param list<float> $row @return list<float> */
    private static function negRow(array $row): array
    {
        return array_map(static fn (float $v): float => -$v, $row);
    }

    /**
     * @param list<list<float>> $A  (m x n)
     * @param list<float> $x  (n)
     * @return list<float>  (m)
     */
    private static function matVec(array $A, array $x): array
    {
        $out = [];
        foreach ($A as $row) {
            $s = 0.0;
            foreach ($row as $j => $v) {
                $s += $v * $x[$j];
            }
            $out[] = $s;
        }

        return $out;
    }

    /**
     * A^T x  for A (m x n), x (m)  -> (n).
     *
     * @param list<list<float>> $A
     * @param list<float> $x
     * @return list<float>
     */
    private static function matTVec(array $A, array $x): array
    {
        $m = count($A);
        if ($m === 0) {
            return [];
        }
        $n = count($A[0]);
        $out = array_fill(0, $n, 0.0);
        for ($i = 0; $i < $m; $i++) {
            $xi = $x[$i];
            $row = $A[$i];
            for ($j = 0; $j < $n; $j++) {
                $out[$j] += $row[$j] * $xi;
            }
        }

        return $out;
    }

    /**
     * @param list<list<float>> $A
     * @param list<list<float>> $B
     * @return list<list<float>>
     */
    private static function matAdd(array $A, array $B): array
    {
        $out = [];
        foreach ($A as $i => $row) {
            $nr = [];
            foreach ($row as $j => $v) {
                $nr[] = $v + $B[$i][$j];
            }
            $out[] = $nr;
        }

        return $out;
    }

    /**
     * @param list<list<float>> $A
     * @return list<list<float>>
     */
    private static function matScale(float $s, array $A): array
    {
        return array_map(static fn ($row): array => array_map(static fn (float $v): float => $s * $v, $row), $A);
    }

    /**
     * @param list<list<float>> $A
     * @return list<list<float>>
     */
    private static function transpose(array $A): array
    {
        $m = count($A);
        if ($m === 0) {
            return [];
        }
        $n = count($A[0]);
        $T = self::zeros($n, $m);
        for ($i = 0; $i < $m; $i++) {
            for ($j = 0; $j < $n; $j++) {
                $T[$j][$i] = $A[$i][$j];
            }
        }

        return $T;
    }

    /** @return list<list<float>> */
    private static function zeros(int $rows, int $cols): array
    {
        $out = [];
        for ($i = 0; $i < $rows; $i++) {
            $out[] = array_fill(0, $cols, 0.0);
        }

        return $out;
    }

    /** @return list<list<float>> */
    private static function eye(int $n): array
    {
        $out = self::zeros($n, $n);
        for ($i = 0; $i < $n; $i++) {
            $out[$i][$i] = 1.0;
        }

        return $out;
    }

    /** IEEE nextafter(x, toward). */
    private static function nextafter(float $x, float $toward): float
    {
        if ($x == $toward) {
            return $x;
        }
        if (is_nan($x) || is_nan($toward)) {
            return NAN;
        }
        if ($x == 0.0) {
            return $toward > 0 ? 4.9e-324 : -4.9e-324;
        }
        $bits = unpack('q', pack('d', $x))[1];
        if (($x < $toward) === ($x > 0)) {
            $bits++;
        } else {
            $bits--;
        }

        return unpack('d', pack('q', $bits))[1];
    }
}
