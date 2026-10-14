<?php

declare(strict_types=1);

namespace Tessero\Stats;

use Tessero\NDArray;

/**
 * scipy.stats.truncate(X, lb, ub) — the new (SciPy 1.15+) distribution transform that restricts a
 * base random variable X to the interval [lb, ub] and renormalises, as a pure-PHP facade over a base
 * {@see Distribution} (Tessero's kernel-backed frozen distribution). No kernel of its own, so it is
 * identical on both backends.
 *
 * With base CDF F, base PDF f and normalising mass Z = F(ub) − F(lb):
 *   pdf(x)  = f(x) / Z                       for lb ≤ x ≤ ub (0 elsewhere)
 *   cdf(x)  = (F(x) − F(lb)) / Z             clamped to [0, 1]
 *   ccdf(x) = 1 − cdf(x)
 *   icdf(q) = F⁻¹(F(lb) + q·Z)               (the base ppf of the renormalised quantile)
 *   logpdf  = log pdf ; logcdf = log cdf ; logccdf = log ccdf
 *
 * The support is exactly [lb, ub]. Scalar arguments return floats; arrays broadcast element-wise.
 */
final class Truncated
{
    private readonly float $lb;

    private readonly float $ub;

    /** Normalising mass Z = F(ub) − F(lb). */
    private readonly float $z;

    /** Base CDF at the lower bound, F(lb). */
    private readonly float $flb;

    public function __construct(private readonly Distribution $base, float $lb, float $ub)
    {
        if (! ($lb < $ub)) {
            throw new \ValueError('Stats\\Truncated: require lb < ub');
        }
        $this->lb = $lb;
        $this->ub = $ub;
        $this->flb = (float) $base->cdf($lb);
        $fub = (float) $base->cdf($ub);
        $this->z = $fub - $this->flb;
        if (! ($this->z > 0.0)) {
            throw new \ValueError('Stats\\Truncated: base assigns no mass to [lb, ub]');
        }
    }

    public function pdf(mixed $x): NDArray|float
    {
        return $this->map($x, fn (float $v): float => ($v >= $this->lb && $v <= $this->ub)
            ? (float) $this->base->pdf($v) / $this->z
            : 0.0);
    }

    public function logpdf(mixed $x): NDArray|float
    {
        return $this->map($x, fn (float $v): float => ($v >= $this->lb && $v <= $this->ub)
            ? (float) $this->base->logpdf($v) - log($this->z)
            : -INF);
    }

    public function cdf(mixed $x): NDArray|float
    {
        return $this->map($x, fn (float $v): float => $this->cdfScalar($v));
    }

    public function logcdf(mixed $x): NDArray|float
    {
        return $this->map($x, fn (float $v): float => log($this->cdfScalar($v)));
    }

    /** Complementary CDF (survival). */
    public function ccdf(mixed $x): NDArray|float
    {
        return $this->map($x, fn (float $v): float => 1.0 - $this->cdfScalar($v));
    }

    public function logccdf(mixed $x): NDArray|float
    {
        return $this->map($x, fn (float $v): float => log(1.0 - $this->cdfScalar($v)));
    }

    /** Inverse CDF (quantile): the base ppf of the renormalised quantile. */
    public function icdf(mixed $q): NDArray|float
    {
        return $this->map($q, fn (float $p): float => (float) $this->base->ppf($this->flb + $p * $this->z));
    }

    /** Inverse complementary CDF. */
    public function iccdf(mixed $q): NDArray|float
    {
        return $this->map($q, fn (float $p): float => (float) $this->base->ppf($this->flb + (1.0 - $p) * $this->z));
    }

    public function median(): float
    {
        return (float) $this->base->ppf($this->flb + 0.5 * $this->z);
    }

    /** @return array{0: float, 1: float} the support's end points [lb, ub]. */
    public function support(): array
    {
        return [$this->lb, $this->ub];
    }

    private function cdfScalar(float $v): float
    {
        if ($v <= $this->lb) {
            return 0.0;
        }
        if ($v >= $this->ub) {
            return 1.0;
        }

        return ((float) $this->base->cdf($v) - $this->flb) / $this->z;
    }

    /** Apply a scalar transform to a number, (nested) array or NDArray, mirroring SciPy's broadcasting. */
    private function map(mixed $x, callable $f): NDArray|float
    {
        if ($x instanceof NDArray) {
            $x = $x->toArray();
            $out = $this->mapArray($x, $f);

            return NDArray::array($out);
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
