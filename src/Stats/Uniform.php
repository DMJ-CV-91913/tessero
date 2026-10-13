<?php

declare(strict_types=1);

namespace Tessero\Stats;

use Tessero\NDArray;
use Tessero\Stats;

/**
 * scipy.stats.Uniform(a, b) — the new (SciPy 1.15+) random-variable class API for the continuous
 * uniform distribution on the interval [a, b], as a pure-PHP facade over Tessero's existing
 * kernel-backed uniform distribution ({@see Stats::uniform()} → {@see Distribution}). No kernel of
 * its own, so it is identical on both backends (the frozen distribution underneath is the one
 * verified by the stats fixtures).
 *
 * Parameterisation: scipy's modern class takes the end points (a, b) = (lower, upper). Tessero's
 * {@see Stats::uniform()} is the legacy (loc, scale) frozen distribution with support
 * [loc, loc + scale]; this facade therefore wraps Stats::uniform($a, $b - $a) — a = loc, width
 * b - a = scale.
 *
 * The 1.15 method names map onto the legacy frozen-distribution names:
 *   pdf/logpdf/cdf/logcdf  → same; ccdf → sf; logccdf → logsf; icdf → ppf; iccdf → isf;
 *   ilogcdf(y) → ppf(exp(y)); ilogccdf(y) → isf(exp(y)); mean/median/variance → mean/median/var;
 *   standard_deviation → std; skewness → stats('s') (= 0); kurtosis → stats('k') + 3 (NON-excess /
 *   Pearson, as the RV API returns it: -1.2 + 3 = 1.8); moment(order, kind) → raw =
 *   Distribution::moment (non-central), central = moment of the zero-mean uniform of the same
 *   width, standardized = moment of the unit-variance uniform [-√3, √3]; entropy/support → same;
 *   mode = the interval midpoint (a + b) / 2.
 *
 * NOTE on precision: scipy's MODERN Uniform computes kurtosis / the standardized 4th moment by
 * numerical quadrature, so those drift by ~1e-12 across parameters. This facade (like the stats
 * fixtures) uses the analytic frozen-distribution values instead, which are exact and stable.
 *
 * Parameters may be numbers or {@see NDArray}s and broadcast against the method argument exactly as
 * in SciPy; scalar parameters with scalar arguments return floats.
 */
final class Uniform
{
    private readonly Distribution $dist;

    /** The unit-variance uniform on [-√3, √3], reused for standardized moments. */
    private ?Distribution $standard = null;

    /** The zero-mean uniform of the same width (loc = -width/2), reused for central moments. */
    private ?Distribution $centered = null;

    /** The lower end point a (= loc). */
    private readonly mixed $a;

    /** The upper end point b (= loc + scale). */
    private readonly mixed $b;

    /** The width b - a (= scale). */
    private readonly mixed $width;

    /**
     * @param mixed $a the lower end point (loc); a number or NDArray
     * @param mixed $b the upper end point (loc + scale); a number or NDArray
     */
    public function __construct(mixed $a = 0.0, mixed $b = 1.0)
    {
        $this->a = $a;
        $this->b = $b;
        $this->width = $this->sub($b, $a);
        $this->dist = Stats::uniform($a, $this->width);
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

    /** The skewness — 0 for every uniform. */
    public function skewness(): NDArray|float
    {
        return $this->dist->stats('s');
    }

    /**
     * The Pearson (NON-excess) kurtosis: the RV API returns the full fourth standardized moment,
     * which is the legacy excess kurtosis (stats('k') = -1.2) plus 3 (= 1.8 for every uniform).
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

    /** The mode — any point in [a, b]; scipy returns the interval midpoint (a + b) / 2. */
    public function mode(): NDArray|float
    {
        $mid = $this->add($this->a, $this->b);

        return $mid instanceof NDArray ? $mid->mul(0.5) : (float) $mid * 0.5;
    }

    /** @return array{0: NDArray|float, 1: NDArray|float} the support's end points (a, b) */
    public function support(): array
    {
        return $this->dist->support();
    }

    /**
     * The raw ('raw' — non-central), 'central' or 'standardized' moment of the given order.
     *
     * Raw = Distribution::moment (the legacy non-central moment about 0). Central = the moment of
     * the zero-mean uniform of the same width (loc = -width/2). Standardized = the moment of the
     * unit-variance uniform on [-√3, √3].
     *
     * @param string $kind one of 'raw', 'central', 'standardized'
     */
    public function moment(int $order, string $kind = 'raw'): NDArray|float
    {
        return match ($kind) {
            'raw' => $this->dist->moment($order),
            'central' => ($this->centered ??= Stats::uniform($this->negHalf($this->width), $this->width))->moment($order),
            'standardized' => ($this->standard ??= Stats::uniform(-M_SQRT3, 2.0 * M_SQRT3))->moment($order),
            default => throw new \ValueError("Uniform::moment(): kind must be 'raw', 'central' or 'standardized'"),
        };
    }

    // ---- small arithmetic helpers (number | NDArray) ----------------------------------------

    /** b - a, for numbers or NDArrays. */
    private function sub(mixed $x, mixed $y): mixed
    {
        if ($x instanceof NDArray || $y instanceof NDArray) {
            $xn = $x instanceof NDArray ? $x : NDArray::array($x);

            return $xn->sub($y);
        }

        return (float) $x - (float) $y;
    }

    /** a + b, for numbers or NDArrays. */
    private function add(mixed $x, mixed $y): mixed
    {
        if ($x instanceof NDArray || $y instanceof NDArray) {
            $xn = $x instanceof NDArray ? $x : NDArray::array($x);

            return $xn->add($y);
        }

        return (float) $x + (float) $y;
    }

    /** -width / 2, for a number or NDArray. */
    private function negHalf(mixed $w): mixed
    {
        return $w instanceof NDArray ? $w->mul(-0.5) : -0.5 * (float) $w;
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
