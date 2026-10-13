<?php

declare(strict_types=1);

namespace Tessero\Stats;

use Tessero\NDArray;
use Tessero\Stats;

/**
 * scipy.stats.Normal(mu, sigma) — the new (SciPy 1.15+) random-variable class API for the normal
 * distribution, as a pure-PHP facade over Tessero's existing kernel-backed normal distribution
 * ({@see Stats::norm()} → {@see Distribution}). No kernel of its own, so it is identical on both
 * backends (the frozen distribution underneath is the one verified by the stats fixtures).
 *
 * The 1.15 method names map onto the legacy frozen-distribution names:
 *   pdf/logpdf/cdf/logcdf  → same; ccdf → sf; logccdf → logsf; icdf → ppf; iccdf → isf;
 *   ilogcdf(y) → ppf(exp(y)); ilogccdf(y) → isf(exp(y)); mean/median/variance → mean/median/var;
 *   standard_deviation → std; skewness → stats('s'); kurtosis → stats('k') + 3 (NON-excess / Pearson,
 *   as the RV API returns it); moment(order, kind) → raw = Distribution::moment (non-central),
 *   central/standardized derived from the zero-mean / standard normal; entropy/support → same;
 *   mode = mu.
 *
 * Parameters may be numbers or {@see NDArray}s and broadcast against the method argument exactly as in
 * SciPy; scalar parameters with scalar arguments return floats.
 */
final class Normal
{
    private readonly Distribution $dist;

    /** The standard normal N(0, 1), reused for standardized moments. */
    private ?Distribution $standard = null;

    /** The zero-mean normal N(0, sigma), reused for central moments. */
    private ?Distribution $centered = null;

    /**
     * @param mixed $mu    the mean (loc); a number or NDArray
     * @param mixed $sigma the standard deviation (scale); a number or NDArray
     */
    public function __construct(private readonly mixed $mu = 0.0, private readonly mixed $sigma = 1.0)
    {
        $this->dist = Stats::norm($mu, $sigma);
    }

    // ---- densities --------------------------------------------------------------------------

    public function pdf(mixed $x): NDArray|float
    {
        return $this->dist->pdf($x);
    }

    public function logpdf(mixed $x): NDArray|float
    {
        return $this->dist->logpdf($x);
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
     * which is the legacy excess kurtosis (stats('k')) plus 3 (= 3.0 for every normal).
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

    /** The mode — for the normal it is the mean mu. */
    public function mode(): NDArray|float
    {
        return $this->mean();
    }

    /** @return array{0: NDArray|float, 1: NDArray|float} the support's end points (-inf, inf) */
    public function support(): array
    {
        return $this->dist->support();
    }

    /**
     * The raw ('raw' — non-central), 'central' or 'standardized' moment of the given order.
     *
     * Raw = Distribution::moment (the legacy non-central moment about 0). Central = the moment of the
     * zero-mean normal N(0, sigma). Standardized = the moment of the standard normal N(0, 1).
     *
     * @param string $kind one of 'raw', 'central', 'standardized'
     */
    public function moment(int $order, string $kind = 'raw'): NDArray|float
    {
        return match ($kind) {
            'raw' => $this->dist->moment($order),
            'central' => ($this->centered ??= Stats::norm(0.0, $this->sigma))->moment($order),
            'standardized' => ($this->standard ??= Stats::norm(0.0, 1.0))->moment($order),
            default => throw new \ValueError("Normal::moment(): kind must be 'raw', 'central' or 'standardized'"),
        };
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
