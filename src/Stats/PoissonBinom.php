<?php

declare(strict_types=1);

namespace Tessero\Stats;

use Tessero\DType;
use Tessero\NDArray;
use Tessero\Native\Library;
use Tessero\Random\Generator;

/**
 * A frozen scipy.stats.poisson_binom distribution: the number of successes in n independent Bernoulli trials
 * whose success probabilities are the vector p.
 *
 * Unlike the other scipy.stats distributions (Tessero\Stats\Distribution), poisson_binom's shape parameter is a
 * whole vector, which the broadcast distribution framework cannot express. The numerics live in the kernel
 * (tsr_poisson_binom / tsr_poisson_binom_rvs): the pmf is a discrete convolution and every other method follows
 * from it through the same rv_discrete machinery the registered distributions use. The native extension's
 * Tessero\Ext\Stats::poissonBinom has the same methods and results.
 */
final class PoissonBinom
{
    /* method codes mirror the kernel's DM_* enum (csrc/src/fn.h) */
    private const PDF = 0, LOGPDF = 1, CDF = 2, LOGCDF = 3, SF = 4, LOGSF = 5, PPF = 6, ISF = 7;
    private const STATS = 8, ENTROPY = 9, SUPPORT = 10, MOMENT = 11;

    private static ?Generator $defaultRng = null;

    /** @var list<float> the probability vector */
    private readonly array $prob;

    public function __construct(mixed $p, private readonly float $loc = 0.0)
    {
        $arr = $p instanceof NDArray ? $p->astype(DType::Float64) : NDArray::array($p, DType::Float64);
        $this->prob = array_map('floatval', $arr->flatten()->toList());
    }

    public function pmf(mixed $k): NDArray|float
    {
        return $this->elem(self::PDF, $k);
    }

    public function logpmf(mixed $k): NDArray|float
    {
        return $this->elem(self::LOGPDF, $k);
    }

    public function cdf(mixed $k): NDArray|float
    {
        return $this->elem(self::CDF, $k);
    }

    public function logcdf(mixed $k): NDArray|float
    {
        return $this->elem(self::LOGCDF, $k);
    }

    public function sf(mixed $k): NDArray|float
    {
        return $this->elem(self::SF, $k);
    }

    public function logsf(mixed $k): NDArray|float
    {
        return $this->elem(self::LOGSF, $k);
    }

    public function ppf(mixed $q): NDArray|float
    {
        return $this->elem(self::PPF, $q);
    }

    public function isf(mixed $q): NDArray|float
    {
        return $this->elem(self::ISF, $q);
    }

    /**
     * Mean ('m'), variance ('v'), skew ('s') and/or excess kurtosis ('k').
     *
     * @return float|array<string, float> one value, or the requested ones keyed by mean, var, skew, kurtosis
     */
    public function stats(string $moments = 'mv'): float|array
    {
        $mask = 0;
        foreach (str_split($moments) as $c) {
            $mask |= match ($c) {
                'm' => 1, 'v' => 2, 's' => 4, 'k' => 8,
                default => throw new \ValueError("PoissonBinom::stats(): moments must contain only 'm', 'v', 's', 'k'"),
            };
        }
        $out = $this->kernel(self::STATS, [(float) $mask], 4);
        $res = [];
        foreach (['mean' => [1, 0], 'var' => [2, 1], 'skew' => [4, 2], 'kurtosis' => [8, 3]] as $name => [$bit, $idx]) {
            if ($mask & $bit) {
                $res[$name] = $out[$idx];
            }
        }

        return count($res) === 1 ? reset($res) : $res;
    }

    public function mean(): float
    {
        return $this->stats('m');
    }

    public function var(): float
    {
        return $this->stats('v');
    }

    public function std(): float
    {
        return sqrt($this->stats('v'));
    }

    public function median(): float
    {
        return $this->elem(self::PPF, 0.5);
    }

    public function entropy(): float
    {
        return $this->kernel(self::ENTROPY, [], 1)[0];
    }

    /** Non-central moment of the given order. */
    public function moment(int $order): float
    {
        return $this->kernel(self::MOMENT, [(float) $order], 1)[0];
    }

    /** @return array{0: float, 1: float} the support's end points */
    public function support(): array
    {
        $out = $this->kernel(self::SUPPORT, [], 2);

        return [$out[0], $out[1]];
    }

    /** @return array{0: float, 1: float} the central interval holding `confidence` of the mass */
    public function interval(float $confidence): array
    {
        if (! ($confidence >= 0 && $confidence <= 1)) {
            throw new \ValueError('PoissonBinom::interval(): confidence must be between 0 and 1 inclusive');
        }

        return [$this->elem(self::PPF, (1.0 - $confidence) / 2), $this->elem(self::PPF, (1.0 + $confidence) / 2)];
    }

    /**
     * Random variates drawn from $randomState's stream, exactly as scipy.stats draws them from a
     * numpy.random.Generator.
     *
     * @param int|list<int>|null $size output shape; null draws a single variate
     */
    public function rvs(int|array|null $size = null, ?Generator $randomState = null): NDArray|int
    {
        $gen = $randomState ?? (self::$defaultRng ??= Generator::defaultRng());
        $ffi = Library::ffi();
        $shape = $size === null ? [] : (is_int($size) ? [$size] : array_values($size));
        $nsamp = 1;
        foreach ($shape as $d) {
            $nsamp *= $d;
        }
        $n = count($this->prob);
        $pbuf = $ffi->new('double[' . max($n, 1) . ']');
        for ($i = 0; $i < $n; $i++) {
            $pbuf[$i] = $this->prob[$i];
        }
        $obuf = $ffi->new('double[' . max($nsamp, 1) . ']');
        $rc = $ffi->tsr_poisson_binom_rvs($gen->stateHandle(), $ffi->cast('double*', $pbuf), $n, $this->loc, $nsamp,
            $ffi->cast('double*', $obuf));
        if ($rc < 0) {
            throw new \ValueError('Stats::poissonBinom()->rvs(): ' . (string) $ffi->tsr_fn_error());
        }
        $vals = [];
        for ($i = 0; $i < $nsamp; $i++) {
            $vals[$i] = (int) $obuf[$i];
        }

        return $shape === [] ? $vals[0] : NDArray::array($vals, DType::Int64)->reshape($shape);
    }

    private function elem(int $method, mixed $x): NDArray|float
    {
        if (is_int($x) || is_float($x)) {
            return $this->kernel($method, [(float) $x], 1)[0];
        }
        $xa = $x instanceof NDArray ? $x->astype(DType::Float64) : NDArray::array($x, DType::Float64);
        $shape = $xa->shape();
        $xs = array_map('floatval', $xa->flatten()->toList());
        $out = $this->kernel($method, $xs, count($xs));

        return NDArray::array($out, DType::Float64)->reshape($shape);
    }

    /**
     * @param list<float> $xs
     * @return list<float>
     */
    private function kernel(int $method, array $xs, int $outN): array
    {
        $ffi = Library::ffi();
        $n = count($this->prob);
        $pbuf = $ffi->new('double[' . max($n, 1) . ']');
        for ($i = 0; $i < $n; $i++) {
            $pbuf[$i] = $this->prob[$i];
        }
        $nx = count($xs);
        $xbuf = $ffi->new('double[' . max($nx, 1) . ']');
        for ($i = 0; $i < $nx; $i++) {
            $xbuf[$i] = $xs[$i];
        }
        $obuf = $ffi->new('double[' . max($outN, 1) . ']');
        $rc = $ffi->tsr_poisson_binom($method, $ffi->cast('double*', $xbuf), $nx, $ffi->cast('double*', $pbuf), $n,
            $this->loc, $ffi->cast('double*', $obuf));
        if ($rc < 0) {
            throw new \ValueError('Stats::poissonBinom(): ' . (string) $ffi->tsr_fn_error());
        }
        $out = [];
        for ($i = 0; $i < $outN; $i++) {
            $out[$i] = $obuf[$i];
        }

        return $out;
    }
}
