<?php

declare(strict_types=1);

namespace Tessero\Stats;

use Tessero\Special;

/**
 * scipy.stats.normal_inverse_gamma(mu, lmbda, a, b) — the normal-inverse-gamma distribution, as a
 * pure-PHP facade. Only the deterministic surface (pdf / logpdf at (x, s2) and mean / var) is
 * provided; rvs is RNG-only and out of scope. No kernel of its own, so it is bit-identical on both
 * backends.
 *
 * With location μ, precision scaling λ>0, shape a>0 and scale b>0, for a point (x, s2>0):
 *
 *   logpdf(x, s2) = ½·(log λ − log(2π·s2))
 *                   + a·log b − gammaln(a)
 *                   − (a+1)·log s2
 *                   − (2b + λ·(x−μ)²) / (2·s2)
 *
 * (matching scipy.stats._multivariate.normal_inverse_gamma_gen._logpdf). pdf = exp(logpdf).
 * mean()  = (μ, b/(a−1))             defined for a>1 (else NaN component).
 * var()   = (b/(λ(a−1)), b²/((a−1)²(a−2)))  for a>2 (else NaN component).
 */
final class NormalInverseGamma
{
    public function __construct(
        private readonly float $mu = 0.0,
        private readonly float $lmbda = 1.0,
        private readonly float $a = 1.0,
        private readonly float $b = 1.0,
    ) {
        if (! ($this->lmbda > 0.0 && $this->a > 0.0 && $this->b > 0.0)) {
            throw new \ValueError('Stats\\NormalInverseGamma: require lmbda > 0, a > 0, b > 0');
        }
    }

    public function logpdf(float $x, float $s2): float
    {
        if (! ($s2 > 0.0)) {
            return -INF;
        }
        $t1 = 0.5 * (log($this->lmbda) - log(2.0 * M_PI * $s2));
        $t2 = $this->a * log($this->b) - (float) Special::gammaln($this->a);
        $t3 = -($this->a + 1.0) * log($s2);
        $t4 = -(2.0 * $this->b + $this->lmbda * ($x - $this->mu) ** 2) / (2.0 * $s2);

        return $t1 + $t2 + $t3 + $t4;
    }

    public function pdf(float $x, float $s2): float
    {
        return exp($this->logpdf($x, $s2));
    }

    /** @return array{0: float, 1: float} (mean of x, mean of s2); s2-mean is NaN when a ≤ 1. */
    public function mean(): array
    {
        $ms2 = ($this->a > 1.0) ? $this->b / ($this->a - 1.0) : NAN;

        return [$this->mu, $ms2];
    }

    /** @return array{0: float, 1: float} (var of x, var of s2); NaN components when a ≤ 1 / a ≤ 2. */
    public function var(): array
    {
        $vx = ($this->a > 1.0) ? $this->b / ($this->lmbda * ($this->a - 1.0)) : NAN;
        $vs2 = ($this->a > 2.0)
            ? ($this->b ** 2) / (($this->a - 1.0) ** 2 * ($this->a - 2.0))
            : NAN;

        return [$vx, $vs2];
    }
}
