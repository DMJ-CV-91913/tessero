<?php

declare(strict_types=1);

namespace Tessero\Optimize;

use Tessero\Exceptions\TesseroException;
use Tessero\Random\Generator;

/**
 * scipy.optimize.basinhopping, written in pure PHP over Tessero bit-exact PCG64 Generator
 * (numpy.random.Generator). Basin-hopping is a two-phase global optimiser: a random Monte-Carlo
 * displacement of the coordinates followed by a local minimisation, with each step accepted or
 * rejected by a Metropolis criterion.
 *
 * Every random draw is taken from the SAME PCG64 stream in the SAME order as SciPy reference
 * implementation: for each of the niter cycles the default RandomDisplacement take_step draws
 * rng.uniform(-stepsize, stepsize, shape(x)) (n draws) and the Metropolis test then draws a single
 * rng.uniform(). Both the displacement and the acceptance test share ONE Generator, exactly as SciPy
 * constructs them from the one rng. Seeded with a numpy Generator (rng=np.random.default_rng(seed)),
 * the port reproduces SciPy exact accept/reject sequence and final minimiser.
 *
 * Scope (defaults matching this pure-PHP port):
 *   - default take_step = RandomDisplacement wrapped by AdaptiveStepsize. Because niter is kept below
 *     interval (50), AdaptiveStepsize never rescales stepsize, so step magnitude is constant and no
 *     extra draws occur.
 *   - default accept_test = Metropolis(T). No user accept_test, no callback.
 *   - local minimiser = scipy.optimize.minimize via Tessero\Optimize\Minimize::minimize with the given
 *     method (BFGS or L-BFGS-B), itself a step-for-step port of SciPy algorithm.
 *
 * The objective is a plain PHP callable taking a list<float> and returning float, mirroring func(x).
 *
 * @see scipy/optimize/_basinhopping.py (BasinHoppingRunner, RandomDisplacement, Metropolis)
 */
final class BasinHopping
{
    /**
     * Minimise $func by basin-hopping.
     *
     * @param callable(list<float>): float $func objective; receives a list<float>, returns float
     * @param list<float> $x0  initial guess
     * @param int|array<int>|null $seed  seed for the PCG64 Generator (numpy default_rng semantics)
     * @param string $method  local minimiser passed to Minimize::minimize (BFGS or L-BFGS-B)
     */
    public static function solve(
        callable $func,
        array $x0,
        int|array|null $seed = null,
        int $niter = 100,
        float $T = 1.0,
        float $stepsize = 0.5,
        string $method = "L-BFGS-B",
    ): OptimizeResult {
        $n = count($x0);
        if ($n === 0) {
            throw new TesseroException("basinhopping: x0 must not be empty.");
        }
        if ($niter < 0) {
            throw new TesseroException("basinhopping: niter must be non-negative.");
        }
        if ($niter >= 50) {
            // AdaptiveStepsize rescales stepsize every interval=50 steps; this constant-stepsize port
            // is only exact below that threshold. Keep fixtures small (see fixtures.py).
            throw new TesseroException("basinhopping: this pure-PHP port requires niter < 50 (constant stepsize).");
        }

        $x0 = array_map("floatval", array_values($x0));
        $beta = $T != 0.0 ? 1.0 / $T : INF;
        $rng = Generator::defaultRng($seed);

        $minimizer = static fn (array $xx): OptimizeResult => Minimize::minimize($func, $xx, null, $method);

        // ---- initial local minimisation (no RNG draw) ----
        $minres = $minimizer($x0);
        $x = $minres->x;                                     // current accepted coordinates
        $incumbentFun = $minres->fun;                        // incumbent_minres.fun
        $incumbentSuccess = $minres->success;
        $nfev = $minres->nfev;
        $njev = $minres->njev;
        $minFailures = $minres->success ? 0 : 1;

        // ---- lowest-energy structure (Storage) ----
        $bestX = $minres->x;
        $bestFun = $minres->fun;
        $bestSuccess = $minres->success;
        $i = 0;

        // ---- main Monte-Carlo loop ----
        for ($i = 0; $i < $niter; $i++) {
            // take_step: RandomDisplacement -> x += uniform(-stepsize, stepsize, shape(x))
            $disp = $rng->uniform(-$stepsize, $stepsize, [$n]);
            $dv = $disp->toList();
            $xAfter = [];
            for ($j = 0; $j < $n; $j++) {
                $xAfter[$j] = $x[$j] + $dv[$j];
            }

            // local minimisation of the displaced point
            $minres = $minimizer($xAfter);
            $nfev += $minres->nfev;
            $njev += $minres->njev;
            if (! $minres->success) {
                $minFailures++;
            }
            $fNew = $minres->fun;

            // Metropolis accept_reject: w = exp(min(0, -(f_new - f_old) * beta)); accept if w >= uniform()
            $prod = -($fNew - $incumbentFun) * $beta;
            $w = exp(min(0.0, $prod));
            $rand = (float) $rng->uniform();                 // scalar draw, same stream
            $accept = ($w >= $rand) && ($minres->success || ! $incumbentSuccess);

            if ($accept) {
                $x = $minres->x;
                $incumbentFun = $minres->fun;
                $incumbentSuccess = $minres->success;
                // Storage.update: keep the lowest successful minimum
                if ($minres->success && ($minres->fun < $bestFun || ! $bestSuccess)) {
                    $bestX = $minres->x;
                    $bestFun = $minres->fun;
                    $bestSuccess = $minres->success;
                }
            }
        }

        return new OptimizeResult(
            x: array_values($bestX),
            fun: $bestFun,
            success: $bestSuccess,
            status: $bestSuccess ? 0 : 1,
            message: "requested number of basinhopping iterations completed successfully",
            nit: $i,
            nfev: $nfev,
            njev: $njev,
            method: "basinhopping",
        );
    }
}
