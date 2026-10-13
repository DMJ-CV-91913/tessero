<?php

declare(strict_types=1);

namespace Tessero\Optimize;

use Tessero\Random\Generator;
use Tessero\Special;

/**
 * Pure-PHP port of scipy.optimize.dual_annealing with no_local_search=True
 * (traditional Generalized Simulated Annealing, no L-BFGS-B refinement).
 *
 * Bit-exact against scipy when the same PCG64 Generator seed is used, because:
 *   - RNG draws are issued in the identical order/shape as scipy, and
 *   - Tessero\Random\Generator reproduces NumPy PCG64 draws exactly, and
 *   - Tessero\Special::gammaln reproduces scipy.special.gammaln.
 *
 * Scope: no_local_search=True ONLY. The default local-search path (L-BFGS-B)
 * is intentionally NOT implemented here: Tessero lbfgsb currently diverges
 * from scipy, so a bit-exact dual_annealing WITH local search is not yet
 * achievable. See VERIFY.txt / wiring.txt notes.
 *
 * Signature mirrors scipy.optimize.dual_annealing(func, bounds, ...).
 *
 * @param callable $func objective, receives float[] x, returns float
 */
final class DualAnnealing
{
    private const TAIL_LIMIT = 1.0e8;
    private const MIN_VISIT_BOUND = 1.0e-10;

    /**
     * @param array{0:float,1:float}[] $bounds
     * @return array{x: float[], fun: float, nit: int, nfev: int, success: bool, message: string[]}
     */
    public static function minimize(
        callable $func,
        array $bounds,
        int $maxiter = 1000,
        float $initialTemp = 5230.0,
        float $restartTempRatio = 2.0e-5,
        float $visit = 2.62,
        float $accept = -5.0,
        float $maxfun = 1.0e7,
        int|array|null $rng = null,
        bool $noLocalSearch = true
    ): array {
        if (!$noLocalSearch) {
            throw new \RuntimeException(
                "DualAnnealing: only no_local_search=true is supported "
                . "(local-search path blocked on lbfgsb divergence)."
            );
        }

        $lower = [];
        $upper = [];
        foreach ($bounds as $b) {
            $lower[] = (float) $b[0];
            $upper[] = (float) $b[1];
        }
        $dim = count($lower);

        if ($restartTempRatio <= 0.0 || $restartTempRatio >= 1.0) {
            throw new \InvalidArgumentException("Restart temperature ratio has to be in range (0, 1)");
        }
        for ($i = 0; $i < $dim; $i++) {
            if (!is_finite($lower[$i]) || !is_finite($upper[$i])) {
                throw new \InvalidArgumentException("Some bounds values are inf values or nan values");
            }
            if (!($lower[$i] < $upper[$i])) {
                throw new \InvalidArgumentException("Bounds are not consistent min < max");
            }
        }

        $gen = ($rng instanceof Generator) ? $rng : Generator::defaultRng($rng);

        $boundRange = [];
        for ($i = 0; $i < $dim; $i++) {
            $boundRange[$i] = $upper[$i] - $lower[$i];
        }

        // --- VisitingDistribution invariant factors (visit_param fixed) ---
        $vp = $visit;
        $factor2 = exp((4.0 - $vp) * log($vp - 1.0));
        $factor3 = exp((2.0 - $vp) * log(2.0) / ($vp - 1.0));
        $factor4p = sqrt(M_PI) * $factor2 / ($factor3 * (3.0 - $vp));
        $factor5 = 1.0 / ($vp - 1.0) - 0.5;
        $d1 = 2.0 - $factor5;
        $factor6 = M_PI * (1.0 - $factor5) / sin(M_PI * (1.0 - $factor5))
            / exp((float) Special::gammaln($d1));

        // --- objective wrapper (nfev counter) ---
        $nfev = 0;
        $fun = static function (array $x) use ($func, &$nfev): float {
            $nfev++;
            return (float) $func($x);
        };

        // ---- visit_fn: returns float[] of length $m ----
        $visitFn = static function (float $temperature, int $m) use (
            $gen, $vp, $factor4p, $factor6
        ): array {
            // x,y = rng.normal(size=(m,2)).T  => x = col0, y = col1
            $draw = $gen->normal(0.0, 1.0, [$m, 2]);
            $rows = $draw->toArray();
            $xs = [];
            $ys = [];
            foreach ($rows as $r) {
                $xs[] = $r[0];
                $ys[] = $r[1];
            }
            $factor1 = exp(log($temperature) / ($vp - 1.0));
            $factor4 = $factor4p * $factor1;
            $sigmax = exp(-($vp - 1.0) * log($factor6 / $factor4) / (3.0 - $vp));
            $out = [];
            for ($i = 0; $i < $m; $i++) {
                $xi = $xs[$i] * $sigmax;
                $den = exp(($vp - 1.0) * log(abs($ys[$i])) / (3.0 - $vp));
                $out[$i] = $xi / $den;
            }
            return $out;
        };

        // ---- visiting: produce x_visit (float[]) ----
        $visiting = static function (array $x, int $step, float $temperature) use (
            $gen, $visitFn, $dim, $lower, $boundRange
        ): array {
            if ($step < $dim) {
                $visits = $visitFn($temperature, $dim);
                // uniform(size=2) -> [upper_sample, lower_sample]
                $u = $gen->uniform(0.0, 1.0, 2)->toArray();
                $upperSample = $u[0];
                $lowerSample = $u[1];
                $xVisit = [];
                for ($i = 0; $i < $dim; $i++) {
                    $v = $visits[$i];
                    if ($v > DualAnnealing::TAIL_LIMIT) {
                        $v = DualAnnealing::TAIL_LIMIT * $upperSample;
                    } elseif ($v < -DualAnnealing::TAIL_LIMIT) {
                        $v = -DualAnnealing::TAIL_LIMIT * $lowerSample;
                    }
                    $xv = $v + $x[$i];
                    $a = $xv - $lower[$i];
                    $bb = fmod($a, $boundRange[$i]) + $boundRange[$i];
                    $xv = fmod($bb, $boundRange[$i]) + $lower[$i];
                    if (abs($xv - $lower[$i]) < DualAnnealing::MIN_VISIT_BOUND) {
                        $xv += 1.0e-10;
                    }
                    $xVisit[$i] = $xv;
                }
                return $xVisit;
            }
            // single-coordinate change
            $xVisit = $x; // copy
            $visit = $visitFn($temperature, 1)[0];
            if ($visit > DualAnnealing::TAIL_LIMIT) {
                $visit = DualAnnealing::TAIL_LIMIT * $gen->uniform();
            } elseif ($visit < -DualAnnealing::TAIL_LIMIT) {
                $visit = -DualAnnealing::TAIL_LIMIT * $gen->uniform();
            }
            $index = $step - $dim;
            $xVisit[$index] = $visit + $x[$index];
            $a = $xVisit[$index] - $lower[$index];
            $bb = fmod($a, $boundRange[$index]) + $boundRange[$index];
            $xVisit[$index] = fmod($bb, $boundRange[$index]) + $lower[$index];
            if (abs($xVisit[$index] - $lower[$index]) < DualAnnealing::MIN_VISIT_BOUND) {
                $xVisit[$index] += DualAnnealing::MIN_VISIT_BOUND;
            }
            return $xVisit;
        };

        // ---- EnergyState reset ----
        $ebest = null;
        $xbest = null;
        $currentEnergy = 0.0;
        $currentLocation = [];

        $reset = static function (?array $x0) use (
            $gen, $fun, $lower, $upper, $dim,
            &$currentEnergy, &$currentLocation, &$ebest, &$xbest
        ): void {
            if ($x0 === null) {
                $currentLocation = $gen->uniform($lower, $upper, $dim)->toArray();
            } else {
                $currentLocation = $x0;
            }
            $initError = true;
            $reinit = 0;
            while ($initError) {
                $currentEnergy = $fun($currentLocation);
                if (!is_finite($currentEnergy)) {
                    if ($reinit >= 1000) {
                        throw new \RuntimeException(
                            "Stopping algorithm because function create NaN or "
                            . "(+/-) infinity values even with trying new random parameters"
                        );
                    }
                    $currentLocation = $gen->uniform($lower, $upper, $dim)->toArray();
                    $reinit++;
                } else {
                    $initError = false;
                }
                if ($ebest === null && $xbest === null) {
                    $ebest = $currentEnergy;
                    $xbest = $currentLocation;
                }
            }
        };

        $reset(null);

        $temperatureRestart = $initialTemp * $restartTempRatio;

        // ---- StrategyChain state ----
        $emin = $currentEnergy;
        $xmin = $currentLocation;
        $notImprovedIdx = 0;
        $notImprovedMaxIdx = 1000;
        $temperatureStep = 0.0;
        $energyStateImproved = false;

        $message = [];
        $success = true;
        $iteration = 0;

        $t1 = exp(($visit - 1.0) * log(2.0)) - 1.0;

        $needToStop = false;
        while (!$needToStop) {
            for ($i = 0; $i < $maxiter; $i++) {
                $s = (float) $i + 2.0;
                $t2 = exp(($visit - 1.0) * log($s)) - 1.0;
                $temperature = $initialTemp * $t1 / $t2;
                if ($iteration >= $maxiter) {
                    $message[] = "Maximum number of iteration reached";
                    $needToStop = true;
                    break;
                }
                if ($temperature < $temperatureRestart) {
                    $reset(null);
                    break;
                }

                // ---- strategy_chain.run(i, temperature) ----
                $temperatureStep = $temperature / (float) ($i + 1);
                $notImprovedIdx++;
                $stopMsg = null;
                $n2 = count($currentLocation) * 2;
                for ($j = 0; $j < $n2; $j++) {
                    if ($j === 0) {
                        $energyStateImproved = ($i === 0);
                    }
                    $xVisit = $visiting($currentLocation, $j, $temperature);
                    $e = $fun($xVisit);
                    if ($e < $currentEnergy) {
                        $currentEnergy = $e;
                        $currentLocation = $xVisit;
                        if ($e < $ebest) {
                            $ebest = $e;
                            $xbest = $xVisit;
                            $energyStateImproved = true;
                            $notImprovedIdx = 0;
                        }
                    } else {
                        // accept_reject(j, e, x_visit)
                        $r = $gen->uniform();
                        $pqvTemp = 1.0 - ((1.0 - $accept)
                            * ($e - $currentEnergy) / $temperatureStep);
                        if ($pqvTemp <= 0.0) {
                            $pqv = 0.0;
                        } else {
                            $pqv = exp(log($pqvTemp) / (1.0 - $accept));
                        }
                        if ($r <= $pqv) {
                            $currentEnergy = $e;
                            $currentLocation = $xVisit;
                            $xmin = $currentLocation;
                        }
                        if ($notImprovedIdx >= $notImprovedMaxIdx) {
                            if ($j === 0 || $currentEnergy < $emin) {
                                $emin = $currentEnergy;
                                $xmin = $currentLocation;
                            }
                        }
                    }
                    if ($nfev >= $maxfun) {
                        $stopMsg = "Maximum number of function call reached during annealing";
                        break;
                    }
                }
                if ($stopMsg !== null) {
                    $message[] = $stopMsg;
                    $needToStop = true;
                    $success = false;
                    break;
                }
                // no_local_search=true: skip local_search()
                $iteration++;
            }
        }

        return [
            "x" => $xbest,
            "fun" => $ebest,
            "nit" => $iteration,
            "nfev" => $nfev,
            "success" => $success,
            "message" => $message,
        ];
    }
}
