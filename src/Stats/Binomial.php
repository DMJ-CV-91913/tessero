<?php

declare(strict_types=1);

namespace Tessero\Stats;

use Tessero\NDArray;
use Tessero\Stats;

/**
 * scipy.stats.Binomial(n, p) — the new (SciPy 1.15+) random-variable class API for the binomial
 * distribution, as a pure-PHP facade over Tessero's existing kernel-backed binomial distribution
 * ({@see Stats::binom()} → {@see Distribution}). No kernel of its own, so it is identical on both
 * backends (the frozen distribution underneath is the one verified by the stats fixtures). The
 * cumulative mass function is the kernel's regularised incomplete beta (betainc) and the quantile is
 * its inverse (betaincinv), exactly as scipy's binom is defined — this facade just forwards to them.
 *
 * The 1.15 method names map onto the legacy frozen-distribution names:
 *   pmf/logpmf → same (the RV class also exposes pdf/logpdf as aliases of pmf/logpmf for a discrete
 *   distribution — forwarded here too); cdf/logcdf → same; ccdf → sf; logccdf → logsf; icdf → ppf;
 *   iccdf → isf; ilogcdf(y) → ppf(exp(y)); ilogccdf(y) → isf(exp(y)); mean/median/variance →
 *   mean/median/var; standard_deviation → std; skewness → stats('s'); kurtosis → stats('k') + 3
 *   (NON-excess / Pearson, as the RV API returns it); entropy/support → same; mode = floor((n+1)p).
 *
 * moment(order, kind): raw = Distribution::moment (the legacy non-central moment, closed form).
 * central / standardized are computed by a numerically stable DIRECT SUMMATION over the support
 *   central_r = Σ_{k=0}^{n} pmf(k) · (k - mean)^r ,  standardized_r = central_r / std^r
 * — this avoids the catastrophic cancellation of the binomial-expansion-of-raw-moments route (which
 * drifts to ~5e-12 for large n) and reproduces scipy's modern class to ~5e-14.
 *
 * Parameters n and p are scalars (the discrete binomial pmf is not broadcast over parameter arrays in
 * this facade); the method argument may be a number, a list, or an {@see NDArray} and is mapped over
 * exactly as in SciPy. Scalar arguments return floats.
 */
final class Binomial
{
    private readonly Distribution $dist;

    /**
     * @param mixed $n the number of trials; a non-negative integer (scalar)
     * @param mixed $p the success probability in [0, 1] (scalar)
     */
    public function __construct(private readonly mixed $n = 1, private readonly mixed $p = 0.5)
    {
        $this->dist = Stats::binom($n, $p);
    }

    // ---- densities (mass) -------------------------------------------------------------------

    public function pmf(mixed $k): NDArray|float
    {
        return $this->dist->pmf($k);
    }

    public function logpmf(mixed $k): NDArray|float
    {
        return $this->dist->logpmf($k);
    }

    /** The RV class aliases pdf → pmf for a discrete distribution. */
    public function pdf(mixed $k): NDArray|float
    {
        return $this->dist->pmf($k);
    }

    /** The RV class aliases logpdf → logpmf for a discrete distribution. */
    public function logpdf(mixed $k): NDArray|float
    {
        return $this->dist->logpmf($k);
    }

    // ---- cumulative / survival --------------------------------------------------------------

    public function cdf(mixed $x): NDArray|float
    {
        return $this->dist->cdf($x);
    }

    public function logcdf(mixed $x): NDArray|float
    {
        return $this->dist->logcdf($x);
    }

    /** Complementary CDF — the survival function (scipy ccdf == legacy sf). */
    public function ccdf(mixed $x): NDArray|float
    {
        return $this->dist->sf($x);
    }

    /** log of the complementary CDF (scipy logccdf == legacy logsf). */
    public function logccdf(mixed $x): NDArray|float
    {
        return $this->dist->logsf($x);
    }

    // ---- inverses ---------------------------------------------------------------------------

    /** Inverse CDF — the quantile / percent-point function (scipy icdf == legacy ppf). */
    public function icdf(mixed $p): NDArray|float
    {
        return $this->dist->ppf($p);
    }

    /** Inverse complementary CDF (scipy iccdf == legacy isf). */
    public function iccdf(mixed $p): NDArray|float
    {
        return $this->dist->isf($p);
    }

    /** Inverse of logcdf: icdf(exp(y)). */
    public function ilogcdf(mixed $y): NDArray|float
    {
        return $this->dist->ppf($this->expOf($y));
    }

    /** Inverse of logccdf: iccdf(exp(y)). */
    public function ilogccdf(mixed $y): NDArray|float
    {
        return $this->dist->isf($this->expOf($y));
    }

    // ---- summary statistics -----------------------------------------------------------------

    public function mean(): NDArray|float
    {
        return $this->dist->mean();
    }

    public function median(): NDArray|float
    {
        return $this->dist->median();
    }

    public function variance(): NDArray|float
    {
        return $this->dist->var();
    }

    public function standard_deviation(): NDArray|float
    {
        return $this->dist->std();
    }

    public function skewness(): NDArray|float
    {
        return $this->dist->stats('s');
    }

    /**
     * The Pearson (NON-excess) kurtosis: the RV API returns the full fourth standardized moment,
     * which is the legacy excess kurtosis (stats('k')) plus 3.
     */
    public function kurtosis(): NDArray|float
    {
        $k = $this->dist->stats('k');

        return $k instanceof NDArray ? $k->add(3.0) : $k + 3.0;
    }

    public function entropy(): NDArray|float
    {
        return $this->dist->entropy();
    }

    /** The mode — floor((n + 1) p) for the binomial. */
    public function mode(): NDArray|float
    {
        return (float) floor(((float) $this->n + 1.0) * (float) $this->p);
    }

    /** @return array{0: NDArray|float, 1: NDArray|float} the support's end points (0, n) */
    public function support(): array
    {
        return $this->dist->support();
    }

    /**
     * The raw ('raw' — non-central), 'central' or 'standardized' moment of the given order.
     *
     * Raw = Distribution::moment (the legacy non-central moment about 0). Central / standardized are
     * computed by stable direct summation over the support (see the class doc).
     *
     * @param string $kind one of 'raw', 'central', 'standardized'
     */
    public function moment(int $order, string $kind = 'raw'): NDArray|float
    {
        return match ($kind) {
            'raw' => $this->dist->moment($order),
            'central' => $this->centralMoment($order),
            'standardized' => $this->standardizedMoment($order),
            default => throw new \ValueError("Binomial::moment(): kind must be 'raw', 'central' or 'standardized'"),
        };
    }

    /** Σ_k pmf(k) (k - mean)^order over the full support {0, …, n} — stable, no cancellation. */
    private function centralMoment(int $order): float
    {
        $n = (int) $this->n;
        $mean = (float) $this->meanScalar();
        $sum = 0.0;
        for ($k = 0; $k <= $n; $k++) {
            $pk = (float) $this->dist->pmf($k);
            $sum += $pk * (($k - $mean) ** $order);
        }

        return $sum;
    }

    /** central_order / std^order; 0 when std is 0 (the degenerate p ∈ {0, 1}). */
    private function standardizedMoment(int $order): float
    {
        $std = $this->stdScalar();
        if ($std == 0.0) {
            return 0.0;
        }

        return $this->centralMoment($order) / ($std ** $order);
    }

    private function meanScalar(): float
    {
        $m = $this->dist->mean();

        return $m instanceof NDArray ? (float) $m->item() : (float) $m;
    }

    private function stdScalar(): float
    {
        $s = $this->dist->std();

        return $s instanceof NDArray ? (float) $s->item() : (float) $s;
    }

    /** exp(y) for a number, an NDArray, or a (nested) PHP array — mirroring how SciPy maps over the input. */
    private function expOf(mixed $y): mixed
    {
        if ($y instanceof NDArray) {
            return $y->exp();
        }
        if (is_array($y)) {
            return NDArray::array($y)->exp();
        }

        return exp((float) $y);
    }
}
