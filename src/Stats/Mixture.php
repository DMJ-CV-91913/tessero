<?php

declare(strict_types=1);

namespace Tessero\Stats;

use Tessero\NDArray;

/**
 * scipy.stats.Mixture(components, weights) — the new (SciPy 1.15+) finite mixture of distributions,
 * as a pure-PHP facade over a list of base {@see Distribution}s. No kernel of its own, so it is
 * identical on both backends.
 *
 * With components fᵢ/Fᵢ and weights wᵢ (wᵢ ≥ 0, Σwᵢ = 1):
 *   pdf(x)  = Σ wᵢ fᵢ(x)
 *   cdf(x)  = Σ wᵢ Fᵢ(x)        ;  ccdf = 1 − cdf
 *   mean    = Σ wᵢ meanᵢ
 *   variance= Σ wᵢ (varᵢ + meanᵢ²) − mean²     (law of total variance)
 *   icdf(q) = the unique root of cdf(x) = q, found by a bracketed bisection/Brent search
 *
 * Scalar arguments return floats; arrays broadcast element-wise.
 */
final class Mixture
{
    /** @var list<Distribution> */
    private readonly array $components;

    /** @var list<float> */
    private readonly array $weights;

    /**
     * @param list<Distribution> $components the mixture's component distributions
     * @param list<float>|null   $weights    the mixing weights (default: uniform); normalised to sum 1
     */
    public function __construct(array $components, ?array $weights = null)
    {
        $n = count($components);
        if ($n === 0) {
            throw new \ValueError('Stats\\Mixture: at least one component is required');
        }
        $this->components = array_values($components);
        $w = $weights ?? array_fill(0, $n, 1.0 / $n);
        if (count($w) !== $n) {
            throw new \ValueError('Stats\\Mixture: weights must match the number of components');
        }
        $sum = 0.0;
        foreach ($w as $wi) {
            if ($wi < 0.0) {
                throw new \ValueError('Stats\\Mixture: weights must be non-negative');
            }
            $sum += (float) $wi;
        }
        $this->weights = array_map(static fn ($wi): float => (float) $wi / $sum, array_values($w));
    }

    public function pdf(mixed $x): NDArray|float
    {
        return $this->map($x, fn (float $v): float => $this->combine(fn (Distribution $c) => (float) $c->pdf($v)));
    }

    public function logpdf(mixed $x): NDArray|float
    {
        return $this->map($x, fn (float $v): float => log($this->combine(fn (Distribution $c) => (float) $c->pdf($v))));
    }

    public function cdf(mixed $x): NDArray|float
    {
        return $this->map($x, fn (float $v): float => $this->cdfScalar($v));
    }

    public function logcdf(mixed $x): NDArray|float
    {
        return $this->map($x, fn (float $v): float => log($this->cdfScalar($v)));
    }

    public function ccdf(mixed $x): NDArray|float
    {
        return $this->map($x, fn (float $v): float => $this->ccdfScalar($v));
    }

    public function logccdf(mixed $x): NDArray|float
    {
        // ccdf = Σ wᵢ·sfᵢ(x); taking the log of that directly is stable in the far tail (avoids 1 − cdf).
        return $this->map($x, fn (float $v): float => log($this->ccdfScalar($v)));
    }

    public function mean(): float
    {
        return $this->combine(fn (Distribution $c) => (float) $c->mean());
    }

    public function variance(): float
    {
        $mean = $this->mean();
        $ex2 = $this->combine(static function (Distribution $c): float {
            $m = (float) $c->mean();

            return (float) $c->var() + $m * $m;
        });

        return $ex2 - $mean * $mean;
    }

    public function standard_deviation(): float
    {
        return sqrt($this->variance());
    }

    public function median(): float
    {
        return $this->icdfScalar(0.5);
    }

    /** Inverse CDF (quantile) by bracketed root-finding on the strictly-increasing mixture CDF. */
    public function icdf(mixed $q): NDArray|float
    {
        return $this->map($q, fn (float $p): float => $this->icdfScalar($p));
    }

    public function iccdf(mixed $q): NDArray|float
    {
        return $this->map($q, fn (float $p): float => $this->icdfScalar(1.0 - $p));
    }

    /** @return array{0: float, 1: float} the union of the component supports. */
    public function support(): array
    {
        $lo = INF;
        $hi = -INF;
        foreach ($this->components as $c) {
            [$a, $b] = $c->support();
            $lo = min($lo, (float) $a);
            $hi = max($hi, (float) $b);
        }

        return [$lo, $hi];
    }

    private function cdfScalar(float $v): float
    {
        return $this->combine(fn (Distribution $c) => (float) $c->cdf($v));
    }

    /** Complementary CDF Σ wᵢ·sfᵢ(x) — kept in survival space so the far tail does not cancel to 0. */
    private function ccdfScalar(float $v): float
    {
        return $this->combine(fn (Distribution $c) => (float) $c->sf($v));
    }

    /** @param callable(Distribution): float $f */
    private function combine(callable $f): float
    {
        $s = 0.0;
        foreach ($this->components as $i => $c) {
            $s += $this->weights[$i] * $f($c);
        }

        return $s;
    }

    /** Solve cdf(x) = p by expanding a bracket then bisecting to full double precision. */
    private function icdfScalar(float $p): float
    {
        if ($p <= 0.0) {
            return $this->support()[0];
        }
        if ($p >= 1.0) {
            return $this->support()[1];
        }
        // Seed the bracket from the weighted spread of the component quantiles.
        $lo = INF;
        $hi = -INF;
        foreach ($this->components as $c) {
            $lo = min($lo, (float) $c->ppf($p));
            $hi = max($hi, (float) $c->ppf($p));
        }
        if (! is_finite($lo)) {
            $lo = -1.0;
        }
        if (! is_finite($hi)) {
            $hi = 1.0;
        }
        if ($lo === $hi) {
            $lo -= 1.0;
            $hi += 1.0;
        }
        // Bracket so that cdf(lo) <= p < cdf(hi); icdf is sup{x : cdf(x) <= p} (SciPy's convention —
        // the RIGHT edge of any flat region whose level equals p). Keeping the lo invariant as the
        // non-strict side (cdf <= p) makes the bisection land on that right edge.
        $span = max($hi - $lo, 1e-6);
        $guard = 0;
        while ($this->cdfScalar($lo) > $p && $guard++ < 200) {
            $lo -= $span;
            $span *= 2.0;
        }
        $span = max($hi - $lo, 1e-6);
        $guard = 0;
        while ($this->cdfScalar($hi) <= $p && $guard++ < 200) {
            $hi += $span;
            $span *= 2.0;
        }
        // Bisection to full precision: mid joins lo (cdf <= p) or hi (cdf > p).
        for ($i = 0; $i < 200; $i++) {
            $mid = 0.5 * ($lo + $hi);
            if ($mid === $lo || $mid === $hi) {
                break;
            }
            if ($this->cdfScalar($mid) <= $p) {
                $lo = $mid;
            } else {
                $hi = $mid;
            }
            if ($hi - $lo <= 1e-15 * (1.0 + abs($mid))) {
                break;
            }
        }

        return 0.5 * ($lo + $hi);
    }

    private function map(mixed $x, callable $f): NDArray|float
    {
        if ($x instanceof NDArray) {
            return NDArray::array($this->mapArray($x->toArray(), $f));
        }
        if (is_array($x)) {
            return NDArray::array($this->mapArray($x, $f));
        }

        return $f((float) $x);
    }

    private function mapArray(array $a, callable $f): array
    {
        $out = [];
        foreach ($a as $k => $v) {
            $out[$k] = is_array($v) ? $this->mapArray($v, $f) : $f((float) $v);
        }

        return $out;
    }
}
