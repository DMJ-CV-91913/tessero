<?php

declare(strict_types=1);

namespace Tessero\Stats;

use Tessero\NDArray;

/**
 * The monotone / symmetric change-of-variable distribution transforms from the new (SciPy 1.15+)
 * random-variable API:
 *   scipy.stats.exp(X) — Y = exp(X), supported on (0, ∞)
 *   scipy.stats.log(X) — Y = log(X), for a positive base X, supported on (−∞, ∞) (image of the base)
 *   scipy.stats.abs(X) — Y = |X|, folded at 0, supported on [0, ∞)
 *
 * Each is a pure-PHP facade over a base {@see Distribution} (Tessero's kernel-backed frozen
 * distribution) with closed-form CDF/PDF, so it is identical on both backends.
 *
 * With base CDF F, base PDF f:
 *   EXP:  cdf(y) = F(ln y),            pdf(y) = f(ln y) / y        (y > 0),  icdf(q) = exp(F⁻¹(q))
 *   LOG:  cdf(y) = F(eʸ),             pdf(y) = f(eʸ)·eʸ,           icdf(q) = ln(F⁻¹(q))
 *   ABS:  cdf(y) = F(y) − F(−y),      pdf(y) = f(y) + f(−y)        (y ≥ 0),  icdf by bracketed bisection
 *
 * Construct via {@see self::exp()}, {@see self::log()} or {@see self::abs()}. Scalar arguments return
 * floats; arrays broadcast element-wise.
 */
final class Transformed
{
    private const EXP = 'exp', LOG = 'log', ABS = 'abs';

    private function __construct(private readonly Distribution $base, private readonly string $kind)
    {
    }

    /** Y = exp(X). */
    public static function exp(Distribution $base): self
    {
        return new self($base, self::EXP);
    }

    /** Y = log(X), for a positive base X. */
    public static function log(Distribution $base): self
    {
        return new self($base, self::LOG);
    }

    /** Y = |X|. */
    public static function abs(Distribution $base): self
    {
        return new self($base, self::ABS);
    }

    public function pdf(mixed $x): NDArray|float
    {
        return $this->map($x, fn (float $y): float => $this->pdfScalar($y));
    }

    public function logpdf(mixed $x): NDArray|float
    {
        return $this->map($x, fn (float $y): float => log($this->pdfScalar($y)));
    }

    public function cdf(mixed $x): NDArray|float
    {
        return $this->map($x, fn (float $y): float => $this->cdfScalar($y));
    }

    public function logcdf(mixed $x): NDArray|float
    {
        return $this->map($x, fn (float $y): float => log($this->cdfScalar($y)));
    }

    public function ccdf(mixed $x): NDArray|float
    {
        return $this->map($x, fn (float $y): float => 1.0 - $this->cdfScalar($y));
    }

    public function logccdf(mixed $x): NDArray|float
    {
        return $this->map($x, fn (float $y): float => log(1.0 - $this->cdfScalar($y)));
    }

    public function icdf(mixed $q): NDArray|float
    {
        return $this->map($q, fn (float $p): float => $this->icdfScalar($p));
    }

    public function iccdf(mixed $q): NDArray|float
    {
        return $this->map($q, fn (float $p): float => $this->icdfScalar(1.0 - $p));
    }

    public function median(): float
    {
        return $this->icdfScalar(0.5);
    }

    /** @return array{0: float, 1: float} the transformed support. */
    public function support(): array
    {
        [$a, $b] = $this->base->support();
        $a = (float) $a;
        $b = (float) $b;

        if ($this->kind === self::EXP) {
            return [exp($a), exp($b)];
        }
        if ($this->kind === self::LOG) {
            return [$a > 0 ? log($a) : -INF, log($b)];
        }
        // ABS: lower end is 0 if the base straddles 0, else the nearer end point to 0.
        $lo = ($a <= 0.0 && $b >= 0.0) ? 0.0 : min(abs($a), abs($b));

        return [$lo, max(abs($a), abs($b))];
    }

    private function cdfScalar(float $y): float
    {
        return match ($this->kind) {
            self::EXP => $y <= 0.0 ? 0.0 : (float) $this->base->cdf(log($y)),
            self::LOG => (float) $this->base->cdf(exp($y)),
            self::ABS => $y <= 0.0 ? 0.0 : (float) $this->base->cdf($y) - (float) $this->base->cdf(-$y),
        };
    }

    private function pdfScalar(float $y): float
    {
        return match ($this->kind) {
            self::EXP => $y <= 0.0 ? 0.0 : (float) $this->base->pdf(log($y)) / $y,
            self::LOG => (float) $this->base->pdf(exp($y)) * exp($y),
            self::ABS => $y < 0.0 ? 0.0 : (float) $this->base->pdf($y) + (float) $this->base->pdf(-$y),
        };
    }

    private function icdfScalar(float $p): float
    {
        if ($this->kind === self::EXP) {
            return exp((float) $this->base->ppf($p));
        }
        if ($this->kind === self::LOG) {
            return log((float) $this->base->ppf($p));
        }
        // ABS: solve F(y) - F(-y) = p on [0, ∞) by bracketed bisection.
        if ($p <= 0.0) {
            return 0.0;
        }
        if ($p >= 1.0) {
            return INF;
        }
        $lo = 0.0;
        $hi = 1.0;
        $guard = 0;
        while ($this->cdfScalar($hi) < $p && $guard++ < 200) {
            $hi *= 2.0;
        }
        for ($i = 0; $i < 200; $i++) {
            $mid = 0.5 * ($lo + $hi);
            if ($mid === $lo || $mid === $hi) {
                break;
            }
            $fm = $this->cdfScalar($mid) - $p;
            if ($fm === 0.0) {
                return $mid;
            }
            if ($fm < 0.0) {
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
