<?php

declare(strict_types=1);

namespace Tessero\Optimize;

use Tessero\Exceptions\TesseroException;
use Tessero\NDArray;
use Tessero\Random\Generator;

/**
 * scipy.optimize.differential_evolution for the 'best1bin' strategy, written in pure PHP over Tessero's
 * bit-exact PCG64 Generator (numpy.random.Generator). Because every random draw - the Latin-hypercube
 * initialisation, the per-generation mutation-constant dither, the trial-vector index shuffle, the binomial
 * crossover mask and the out-of-bounds re-sampling - is taken from the SAME PCG64 stream in the SAME order as
 * SciPy's reference implementation, the solver reproduces SciPy's exact iterate sequence (x, nfev and nit) when
 * seeded with a numpy Generator, not merely a converged optimum.
 *
 * Scope: strategy='best1bin', updating='immediate', polish=False, init='latinhypercube', no constraints,
 * no integrality, scalar (non-vectorised) objective over list<float>. The objective is a plain PHP callable
 * taking a list<float> and returning float, mirroring SciPy's func(parameters).
 *
 * @see scipy/optimize/_differentialevolution.py (DifferentialEvolutionSolver)
 */
final class DifferentialEvolution
{
    private const MACHEPS = 2.220446049250313e-16;

    /**
     * Minimise $func over box $bounds with differential evolution (best1bin).
     *
     * @param callable(list<float>): float $func objective; receives a list<float>, returns float
     * @param list<array{0: float, 1: float}> $bounds  per-dimension [lower, upper]
     * @param int|array<int>|null $seed  seed for the PCG64 Generator (numpy default_rng семантика)
     * @param float|array{0: float, 1: float} $mutation  F; a 2-tuple enables per-generation dither (scipy default (0.5, 1))
     */
    public static function solve(
        callable $func,
        array $bounds,
        int|array|null $seed = null,
        string $strategy = 'best1bin',
        int $maxiter = 1000,
        int $popsize = 15,
        float $tol = 0.01,
        float|array $mutation = [0.5, 1.0],
        float $recombination = 0.7,
        float $atol = 0.0,
        bool $polish = false,
    ): OptimizeResult {
        if ($strategy !== 'best1bin') {
            throw new TesseroException("differential_evolution: only strategy 'best1bin' is supported (got '{$strategy}').");
        }
        if ($polish) {
            throw new TesseroException('differential_evolution: polish=true is out of scope for this pure-PHP port.');
        }

        // ---- bounds / limits (scipy stores limits = [lower; upper]) ----
        $n = count($bounds);
        if ($n === 0) {
            throw new TesseroException('differential_evolution: bounds must not be empty.');
        }
        $lb = [];
        $ub = [];
        foreach ($bounds as $b) {
            $lo = (float) $b[0];
            $hi = (float) $b[1];
            $lb[] = $lo;
            $ub[] = $hi;
            if (! is_finite($lo) || ! is_finite($hi) || $lo > $hi) {
                throw new TesseroException('differential_evolution: each bound must be finite with lower <= upper.');
            }
        }
        // scipy: __scale_arg1 = 0.5*(lb+ub), __scale_arg2 = |lb-ub|
        $scaleMid = [];
        $scaleSpan = [];
        for ($j = 0; $j < $n; $j++) {
            $scaleMid[$j] = 0.5 * ($lb[$j] + $ub[$j]);
            $scaleSpan[$j] = abs($lb[$j] - $ub[$j]);
        }

        // ---- mutation constant / dither (scipy: self.scale, self.dither) ----
        $dither = null;
        $scale = 0.0;
        if (is_array($mutation)) {
            if (count($mutation) < 2) {
                throw new TesseroException('differential_evolution: mutation tuple needs two values.');
            }
            $d0 = (float) $mutation[0];
            $d1 = (float) $mutation[1];
            $dither = $d0 <= $d1 ? [$d0, $d1] : [$d1, $d0];   // scipy sorts the dither bounds
        } else {
            $scale = (float) $mutation;
        }
        $cr = $recombination;

        $np = $popsize * $n;                                  // num_population_members
        if ($np < 5) {
            // scipy clamps popsize so best1bin always has >= 5 selectable members
            $np = 5;
        }

        $maxfun = PHP_INT_MAX;                                // scipy default maxfun = (maxiter+1)*NP, effectively unbounded here

        $rng = Generator::defaultRng($seed);

        // ---- init_population_lhs ----
        // samples = segsize*U(NP,N) + linspace(0,1,NP,endpoint=False)[:,None]
        $segsize = 1.0 / $np;
        $uFlat = $rng->uniform(0.0, 1.0, [$np, $n])->toList();   // row-major, length NP*N
        $samples = [];
        for ($i = 0; $i < $np; $i++) {
            $off = $i / $np;                                     // linspace(0,1,NP,endpoint=False)[i] == i/NP
            $row = [];
            for ($j = 0; $j < $n; $j++) {
                $row[$j] = $segsize * $uFlat[$i * $n + $j] + $off;
            }
            $samples[$i] = $row;
        }
        // population[:, j] = samples[order, j] with order = permutation(NP) per column
        $pop = [];
        for ($i = 0; $i < $np; $i++) {
            $pop[$i] = array_fill(0, $n, 0.0);
        }
        for ($j = 0; $j < $n; $j++) {
            $order = $rng->permutation($np)->toList();
            for ($i = 0; $i < $np; $i++) {
                $pop[$i][$j] = $samples[$order[$i]][$j];
            }
        }

        // ---- initial energies + promote lowest to index 0 ----
        $energies = [];
        $nfev = 0;
        for ($i = 0; $i < $np; $i++) {
            $energies[$i] = (float) $func(self::scaleParameters($pop[$i], $scaleMid, $scaleSpan));
            $nfev++;
        }
        self::promoteLowest($pop, $energies, $np);

        // persistent index array shuffled in place by _select_samples (scipy: self._random_population_index)
        $ridx = NDArray::arange($np);

        $nit = 0;
        $warning = false;
        for ($nit = 1; $nit <= $maxiter; $nit++) {
            // dither: scale re-drawn once per generation, BEFORE the candidate loop
            if ($dither !== null) {
                $scale = (float) $rng->uniform($dither[0], $dither[1]);
            }

            // ---- one generation, updating='immediate' ----
            for ($cand = 0; $cand < $np; $cand++) {
                if ($nfev > $maxfun) {
                    $warning = true;
                    break 2;
                }
                // _mutate: fill_point first, THEN _select_samples, THEN crossover draw (scipy draw order)
                $fillPoint = (int) $rng->integers($n);
                [$r0, $r1] = self::selectSamples($rng, $ridx, $cand, $np);

                // best1: pop[0] + scale*(pop[r0]-pop[r1])
                $best = $pop[0];
                $pr0 = $pop[$r0];
                $pr1 = $pop[$r1];
                $bprime = [];
                for ($j = 0; $j < $n; $j++) {
                    $bprime[$j] = $best[$j] + $scale * ($pr0[$j] - $pr1[$j]);
                }

                // binomial crossover
                $crossU = $rng->uniform(0.0, 1.0, [$n])->toList();
                $trial = $pop[$cand];
                for ($j = 0; $j < $n; $j++) {
                    if ($crossU[$j] < $cr) {
                        $trial[$j] = $bprime[$j];
                    }
                }
                $trial[$fillPoint] = $bprime[$fillPoint];        // crossovers[fill_point] = True

                // _ensure_constraint: resample out-of-[0,1] entries, in index order
                $oob = [];
                for ($j = 0; $j < $n; $j++) {
                    if ($trial[$j] > 1.0 || $trial[$j] < 0.0) {
                        $oob[] = $j;
                    }
                }
                if ($oob !== []) {
                    $repl = $rng->uniform(0.0, 1.0, [count($oob)])->toList();
                    foreach ($oob as $k => $j) {
                        $trial[$j] = $repl[$k];
                    }
                }

                // energy of the trial
                $energy = (float) $func(self::scaleParameters($trial, $scaleMid, $scaleSpan));
                $nfev++;

                // _accept_trial (feasible/feasible): energy_trial <= energy_orig
                if ($energy <= $energies[$cand]) {
                    $pop[$cand] = $trial;
                    $energies[$cand] = $energy;
                    if ($energy <= $energies[0]) {
                        self::promoteLowest($pop, $energies, $np);
                    }
                }
            }

            // converged(): std(energies) <= atol + tol*|mean(energies)|
            if (self::converged($energies, $np, $atol, $tol)) {
                break;
            }
        }
        if ($nit > $maxiter) {
            $nit = $maxiter;
            $warning = true;
        }

        // best solution = argmin energies (scipy keeps it at index 0 via promotion)
        $bi = 0;
        for ($i = 1; $i < $np; $i++) {
            if ($energies[$i] < $energies[$bi]) {
                $bi = $i;
            }
        }
        $x = self::scaleParameters($pop[$bi], $scaleMid, $scaleSpan);

        return new OptimizeResult(
            x: array_values($x),
            fun: $energies[$bi],
            success: ! $warning,
            status: $warning ? 1 : 0,
            message: $warning ? 'Maximum number of iterations has been exceeded.' : 'Optimization terminated successfully.',
            nit: $nit,
            nfev: $nfev,
            method: 'differential_evolution',
        );
    }

    /**
     * _scale_parameters: scale a point in [0,1] to the parameter box using scipy's midpoint form,
     * scaled = mid + (trial - 0.5) * span.
     *
     * @param list<float> $t
     * @param list<float> $mid
     * @param list<float> $span
     * @return list<float>
     */
    private static function scaleParameters(array $t, array $mid, array $span): array
    {
        $out = [];
        foreach ($t as $j => $tj) {
            $out[$j] = $mid[$j] + ($tj - 0.5) * $span[$j];
        }

        return $out;
    }

    /**
     * _select_samples for best1 (number_samples=5): shuffle the persistent index array, take the first 6,
     * drop the candidate, keep the first 5; best1 uses the first two (r0, r1).
     *
     * @return array{0: int, 1: int}
     */
    private static function selectSamples(Generator $rng, NDArray $ridx, int $candidate, int $np): array
    {
        $rng->shuffle($ridx);                                    // in place, same stream/order as numpy
        $idx = $ridx->toList();
        $picked = [];
        for ($k = 0; $k < 6 && $k < $np; $k++) {                 // first number_samples+1 = 6 entries
            if ($idx[$k] !== $candidate) {
                $picked[] = (int) $idx[$k];
                if (count($picked) === 5) {
                    break;
                }
            }
        }

        return [$picked[0], $picked[1]];
    }

    /**
     * _promote_lowest_energy: swap the lowest-energy member into index 0 (all members feasible here).
     *
     * @param list<list<float>> $pop
     * @param list<float> $energies
     */
    private static function promoteLowest(array &$pop, array &$energies, int $np): void
    {
        $l = 0;
        for ($i = 1; $i < $np; $i++) {
            if ($energies[$i] < $energies[$l]) {
                $l = $i;
            }
        }
        if ($l !== 0) {
            [$pop[0], $pop[$l]] = [$pop[$l], $pop[0]];
            [$energies[0], $energies[$l]] = [$energies[$l], $energies[0]];
        }
    }

    /**
     * converged(): population-energy standard deviation (population std, ddof=0) within atol + tol*|mean|.
     *
     * @param list<float> $energies
     */
    private static function converged(array $energies, int $np, float $atol, float $tol): bool
    {
        $mean = 0.0;
        for ($i = 0; $i < $np; $i++) {
            if (! is_finite($energies[$i])) {
                return false;
            }
            $mean += $energies[$i];
        }
        $mean /= $np;
        $var = 0.0;
        for ($i = 0; $i < $np; $i++) {
            $d = $energies[$i] - $mean;
            $var += $d * $d;
        }
        $std = sqrt($var / $np);                                 // numpy std, ddof=0

        return $std <= $atol + $tol * abs($mean);
    }
}
