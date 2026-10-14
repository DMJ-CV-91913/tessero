<?php

declare(strict_types=1);

namespace Tessero\Stats;

use Tessero\NDArray;
use Tessero\Special;

/**
 * scipy.stats.order_statistic(X, n, r) — the distribution of the r-th order statistic (the r-th
 * smallest of n i.i.d. draws) of a continuous base random variable X, as a pure-PHP facade over a
 * base {@see Distribution}. The transform is closed-form in the base CDF/PDF and the regularised
 * incomplete beta function, so it is identical on both backends.
 *
 * With base CDF F, base PDF f and u = F(x):
 *   cdf(x)  = I_u(r, n − r + 1)                         (regularised incomplete beta)
 *   pdf(x)  = n·C(n−1, r−1)·u^(r−1)·(1−u)^(n−r)·f(x)
 *   icdf(q) = F⁻¹( I⁻¹_q(r, n − r + 1) )                (base ppf of the inverse incomplete beta)
 *   ccdf    = 1 − cdf
 *
 * n and r are integers with 1 ≤ r ≤ n. Scalar arguments return floats; arrays broadcast element-wise.
 */
final class OrderStatistic
{
    private readonly int $n;

    private readonly int $r;

    /** The log of the normalising combinatorial factor n·C(n−1, r−1) = n! / ((r−1)!·(n−r)!). */
    private readonly float $logNorm;

    public function __construct(private readonly Distribution $base, int $n, int $r)
    {
        if ($n < 1 || $r < 1 || $r > $n) {
            throw new \ValueError('Stats\\OrderStatistic: require 1 <= r <= n');
        }
        $this->n = $n;
        $this->r = $r;
        // log(n! / ((r-1)! (n-r)!)) via lgamma, stable for large n.
        $this->logNorm = Special::gammaln($n + 1) - Special::gammaln($r) - Special::gammaln($n - $r + 1);
    }

    public function pdf(mixed $x): NDArray|float
    {
        return $this->map($x, fn (float $v): float => $this->pdfScalar($v));
    }

    public function logpdf(mixed $x): NDArray|float
    {
        return $this->map($x, fn (float $v): float => log($this->pdfScalar($v)));
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
        return $this->map($x, fn (float $v): float => log($this->ccdfScalar($v)));
    }

    /** Inverse CDF: base ppf of the inverse regularised incomplete beta. */
    public function icdf(mixed $q): NDArray|float
    {
        return $this->map($q, function (float $p): float {
            $u = (float) Special::betaincinv($this->r, $this->n - $this->r + 1, $p);

            return (float) $this->base->ppf($u);
        });
    }

    public function iccdf(mixed $q): NDArray|float
    {
        // betainccinv(a, b, p) gives u with I_u(a, b) = 1 − p, i.e. the (1 − p) quantile, stable for small p.
        return $this->map($q, function (float $p): float {
            $u = (float) Special::betainccinv($this->r, $this->n - $this->r + 1, $p);

            return (float) $this->base->ppf($u);
        });
    }

    public function median(): float
    {
        $u = (float) Special::betaincinv($this->r, $this->n - $this->r + 1, 0.5);

        return (float) $this->base->ppf($u);
    }

    /** @return array{0: float, 1: float} the base support (the order statistic lives on the same support). */
    public function support(): array
    {
        [$a, $b] = $this->base->support();

        return [(float) $a, (float) $b];
    }

    private function cdfScalar(float $v): float
    {
        $u = (float) $this->base->cdf($v);
        if ($u <= 0.0) {
            return 0.0;
        }
        if ($u >= 1.0) {
            return 1.0;
        }

        return (float) Special::betainc($this->r, $this->n - $this->r + 1, $u);
    }

    /** ccdf = 1 − I_u(r, n−r+1) = I_{1−u}(n−r+1, r) = betaincc(r, n−r+1, u), stable near 1. */
    private function ccdfScalar(float $v): float
    {
        $u = (float) $this->base->cdf($v);
        if ($u <= 0.0) {
            return 1.0;
        }
        if ($u >= 1.0) {
            return 0.0;
        }

        return (float) Special::betaincc($this->r, $this->n - $this->r + 1, $u);
    }

    private function pdfScalar(float $v): float
    {
        $u = (float) $this->base->cdf($v);
        $f = (float) $this->base->pdf($v);
        if ($f <= 0.0 || $u <= 0.0 || $u >= 1.0) {
            // Edge cases: u^(r-1)(1-u)^(n-r) may still be finite at the boundary, but f·… -> 0.
            if ($f <= 0.0) {
                return 0.0;
            }
        }
        // log-space for numerical stability: logNorm + (r-1)log u + (n-r)log(1-u) + log f.
        $logp = $this->logNorm;
        if ($this->r - 1 > 0) {
            $logp += ($this->r - 1) * log($u);
        }
        if ($this->n - $this->r > 0) {
            $logp += ($this->n - $this->r) * log(1.0 - $u);
        }
        $logp += log($f);

        return exp($logp);
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
