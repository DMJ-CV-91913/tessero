<?php

declare(strict_types=1);

namespace Tessero\Stats;

use Tessero\NDArray;
use Tessero\Stats;

/**
 * scipy.stats.make_distribution(dist) — the meta-wrapper that turns a legacy rv_continuous (e.g.
 * scipy.stats.norm) into a new (SciPy 1.15+) ContinuousDistribution-style random-variable CLASS.
 * Calling the returned class with the base distribution's shape parameters yields a frozen new-API
 * distribution object whose pdf/cdf/logpdf/... equal the wrapped base distribution.
 *
 * In SciPy this is purely a bridge: make_distribution(stats.X)(**shapes) evaluates exactly like the
 * legacy frozen stats.X(*shapes) at loc 0, scale 1 — the new infrastructure re-exposes the same
 * mathematics under the 1.15 method names. Tessero already has that mathematics in its kernel-backed
 * frozen {@see Distribution} (Stats::X(...)), so this facade is a thin re-dispatch over it: no kernel
 * of its own, hence identical on both backends (the frozen distribution underneath is the one already
 * verified by the stats fixtures).
 *
 * Two construction forms mirror SciPy's:
 *   $Norm = MakeDistribution::of('norm');   // the family class — make_distribution(stats.norm)
 *   $X    = $Norm();                          // freeze: the standard normal, new-API object
 *   $Y    = MakeDistribution::of('gamma')(['a' => 2.5]);     // standard gamma(a=2.5)
 * or, in one step:
 *   $X = MakeDistribution::make('gamma', ['a' => 2.5]);
 *
 * The 1.15 method names map onto the legacy frozen-distribution names exactly as the other new-API
 * facades do (see {@see Normal}): pdf/logpdf/cdf/logcdf → same; ccdf → sf; logccdf → logsf;
 * icdf → ppf; iccdf → isf; variance → var; mean/median/support/entropy/moment → same.
 *
 * make_distribution standardizes the base (strips loc/scale); loc/scale are reintroduced in SciPy via
 * arithmetic (scale * X + loc), which equals the legacy X(loc, scale). This facade therefore builds
 * the base at loc 0, scale 1 and accepts only the distribution's shape parameters, matching
 * make_distribution(stats.X)(**shapes).
 */
final class MakeDistribution
{
    /**
     * The ordered shape-parameter names for every base distribution make_distribution accepts here.
     * Empty for shapeless families (norm, expon, uniform, cauchy, laplace). The order is scipy's
     * positional shape order, so named kwargs marshal to Stats::<name>(...) correctly.
     *
     * @var array<string, list<string>>
     */
    private const SHAPES = [
        'norm'    => [],
        'expon'   => [],
        'uniform' => [],
        'cauchy'  => [],
        'laplace' => [],
        'gamma'   => ['a'],
        'chi2'    => ['df'],
        't'       => ['df'],
        'lognorm' => ['s'],
        'beta'    => ['a', 'b'],
    ];

    private readonly Distribution $dist;

    /**
     * @param string               $name  the scipy distribution name ('norm', 'gamma', ...)
     * @param array<string, mixed>  $shape the shape parameters by name (empty for shapeless families)
     */
    public function __construct(string $name, array $shape = [])
    {
        if (! isset(self::SHAPES[$name])) {
            throw new \ValueError("make_distribution: unsupported base distribution '{$name}'");
        }
        $order = self::SHAPES[$name];
        $args = [];
        foreach ($order as $p) {
            if (! array_key_exists($p, $shape)) {
                throw new \ValueError("make_distribution('{$name}'): missing shape parameter '{$p}'");
            }
            $args[] = $shape[$p];
        }
        foreach (array_keys($shape) as $given) {
            if (! in_array($given, $order, true)) {
                throw new \ValueError("make_distribution('{$name}'): unexpected parameter '{$given}'");
            }
        }
        // loc 0, scale 1 (standardized base) — exactly what make_distribution(stats.X)() produces.
        $this->dist = match ($name) {
            'norm'    => Stats::norm(0.0, 1.0),
            'expon'   => Stats::expon(0.0, 1.0),
            'uniform' => Stats::uniform(0.0, 1.0),
            'cauchy'  => Stats::cauchy(0.0, 1.0),
            'laplace' => Stats::laplace(0.0, 1.0),
            'gamma'   => Stats::gamma($args[0], 0.0, 1.0),
            'chi2'    => Stats::chi2($args[0], 0.0, 1.0),
            't'       => Stats::t($args[0], 0.0, 1.0),
            'lognorm' => Stats::lognorm($args[0], 0.0, 1.0),
            'beta'    => Stats::beta($args[0], $args[1], 0.0, 1.0),
            default   => throw new \ValueError("make_distribution: unsupported base distribution '{$name}'"),
        };
    }

    /**
     * The family class — SciPy's make_distribution(stats.<name>). Invoke the returned factory with the
     * shape parameters to freeze a distribution object.
     */
    public static function of(string $name): MakeDistributionFactory
    {
        if (! isset(self::SHAPES[$name])) {
            throw new \ValueError("make_distribution: unsupported base distribution '{$name}'");
        }

        return new MakeDistributionFactory($name);
    }

    /**
     * One-step construction: make_distribution(stats.X)(**shapes).
     *
     * @param array<string, mixed> $shape
     */
    public static function make(string $name, array $shape = []): self
    {
        return new self($name, $shape);
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

    public function entropy(): NDArray|float
    {
        return $this->dist->entropy();
    }

    /** The raw (non-central) moment of the given order — the new-API default moment kind. */
    public function moment(int $order): NDArray|float
    {
        return $this->dist->moment($order);
    }

    /** @return array{0: NDArray|float, 1: NDArray|float} the support's end points */
    public function support(): array
    {
        return $this->dist->support();
    }
}
