<?php

declare(strict_types=1);

namespace Tessero\Integrate;

use Tessero\Random\SeedSequence;

/**
 * scipy.integrate.qmc_quad — Quasi-Monte Carlo quadrature in N dimensions.
 *
 * The integral over the box [a_i, b_i] is estimated `n_estimates` times, each with a fresh,
 * independently-scrambled {@see Halton} engine of `n_points` points; the reported integral is the
 * mean of the estimates. Each estimate maps the engine's points into the box
 * (`x = sample*(b-a)+a`) and averages `func(x) * prod(b-a)` over the points.
 *
 * Reproduces scipy's RNG draw order BIT-FOR-BIT when called with `qrng=qmc.Halton(d, seed=rng)` for
 * `rng = numpy.random.default_rng(SEED)`. The spawn tree (via {@see SeedSequence}) is:
 *
 *   default_rng(SEED)           -> root SeedSequence(SEED),   spawn_key ()
 *   Halton(seed=rng)            -> _initialize spawns once:   self.rng spawn_key (0,)
 *   _rng_spawn(qrng.rng, N)     -> N children:                spawn_key (0,0),(0,1),...,(0,N-1)
 *   Halton(seed=rngs[i])        -> spawns once more:          self.rng spawn_key (0,i,0)
 *
 * Estimate 0 uses the base Halton (self.rng spawn_key (0,)); estimate i (>0) uses a Halton whose
 * self.rng has spawn_key (0,i,0). The scramble draws inside each Halton match scipy exactly, so the
 * sample points — and therefore the integral — match to full working precision (observed rel 0.0).
 *
 * Only the real, non-log branch with lower < upper limits is implemented (the common case the
 * fixtures exercise). Pure PHP, host-independent.
 */
final class QmcQuad
{
    /**
     * @param callable          $func        integrand; receives an n_points×d array of points (list of
     *                                        rows) and returns a list of n_points scalar values
     * @param list<float>       $a           lower limits (length d)
     * @param list<float>       $b           upper limits (length d)
     * @param int|list<int>     $seed        the SEED used for numpy.random.default_rng(seed)
     * @param int               $nEstimates  number of independent QMC estimates (scipy default 8)
     * @param int               $nPoints     points per estimate (scipy default 1024)
     *
     * @return array{integral: float, standard_error: float}
     */
    public static function qmcQuad(
        callable $func,
        array $a,
        array $b,
        int|array $seed,
        int $nEstimates = 8,
        int $nPoints = 1024
    ): array {
        $dim = count($a);
        if (count($b) !== $dim) {
            throw new \InvalidArgumentException('qmc_quad: a and b must have the same length.');
        }

        // The sign of the integral flips once per limit with b < a; scipy swaps those limits and
        // multiplies the result by (-1)^(#swaps).
        $sign = 1.0;
        for ($k = 0; $k < $dim; $k++) {
            if ($b[$k] < $a[$k]) {
                [$a[$k], $b[$k]] = [$b[$k], $a[$k]];
                $sign = -$sign;
            } elseif ($a[$k] === $b[$k]) {
                return ['integral' => 0.0, 'standard_error' => 0.0];
            }
        }

        $A = 1.0;
        for ($k = 0; $k < $dim; $k++) {
            $A *= ($b[$k] - $a[$k]);
        }
        $dA = $A / $nPoints;

        $root = new SeedSequence($seed);           // default_rng(seed): spawn_key ()

        $estimates = [];
        // Estimate 0: base Halton, self.rng = root.spawn(1)[0] with spawn_key (0,).
        $qrng = new Halton($dim, self::engineSeq($root, [0]));
        for ($i = 0; $i < $nEstimates; $i++) {
            $sample = $qrng->random($nPoints);     // n_points × d in [0,1)
            $x = [];
            foreach ($sample as $row) {
                $pt = [];
                for ($k = 0; $k < $dim; $k++) {
                    $pt[] = $row[$k] * ($b[$k] - $a[$k]) + $a[$k];
                }
                $x[] = $pt;
            }
            $vals = $func($x);                     // list of n_points scalars
            $s = 0.0;
            foreach ($vals as $v) {
                $s += (float) $v * $dA;
            }
            $estimates[] = $s;

            // Next engine: Halton(seed=rngs[i]); rngs[i] has spawn_key (0,i); the new Halton spawns
            // it once more -> self.rng spawn_key (0,i,0).
            $qrng = new Halton($dim, self::engineSeq($root, [0, $i, 0]));
        }

        $mean = array_sum($estimates) / $nEstimates;
        $integral = $mean * $sign;

        // sample std with ddof=1, then sem = std / sqrt(n_estimates).
        $se = 0.0;
        if ($nEstimates > 1) {
            $var = 0.0;
            foreach ($estimates as $e) {
                $var += ($e - $mean) ** 2;
            }
            $var /= ($nEstimates - 1);
            $se = sqrt($var) / sqrt($nEstimates);
        }

        return ['integral' => $integral, 'standard_error' => $se];
    }

    /**
     * Build the SeedSequence a Halton engine uses as self.rng, given the ROOT sequence and the full
     * spawn key. We construct it directly (rather than threading child indices through live spawn
     * state) because the key is deterministic for each estimate: (0,) for estimate 0 and (0,i,0) for
     * estimate i. The assembled entropy is identical either way.
     *
     * @param list<int> $spawnKey
     */
    private static function engineSeq(SeedSequence $root, array $spawnKey): SeedSequence
    {
        return $root->childWithKey($spawnKey);
    }
}
