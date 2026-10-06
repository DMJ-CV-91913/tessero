<?php

declare(strict_types=1);

namespace Tessero\Stats;

use FFI;
use Tessero\DType;
use Tessero\Exceptions\ShapeError;
use Tessero\NDArray;
use Tessero\Native\Library;
use Tessero\Native\Registry;
use Tessero\Random\Generator;

/**
 * A frozen scipy.stats distribution: Stats::gamma(2.5, loc: 1.0, scale: 3.0).
 *
 * Every method is one element-wise call into the kernel's distribution machinery (ADR 0011): SciPy's
 * rv_continuous / rv_discrete ported with the same support handling and generic fallbacks, so the results
 * match scipy.stats. Parameters may be numbers or arrays; they broadcast against x as in SciPy. The native
 * extension's Tessero\Ext\Distribution has the same methods and results.
 */
final class Distribution
{
    private const PDF = 0, LOGPDF = 1, CDF = 2, LOGCDF = 3, SF = 4, LOGSF = 5, PPF = 6, ISF = 7;
    private const STATS = 8, ENTROPY = 9, SUPPORT = 10, MOMENT = 11;

    private static ?Generator $defaultRng = null;

    private readonly bool $discrete;

    /** @param list<mixed> $params shape parameters, loc and (continuous) scale */
    public function __construct(private readonly string $name, private readonly array $params)
    {
        $info = Registry::info($name);
        if ($info['kind'] !== 'dist') {
            throw new \ValueError("{$name} is not a distribution");
        }
        $this->discrete = (bool) $info['discrete'];
    }

    /** The SciPy name ("gamma"). */
    public function name(): string
    {
        return substr($this->name, 6);
    }

    /** @return list<mixed> the shape parameters, loc and scale */
    public function args(): array
    {
        return $this->params;
    }

    public function pdf(mixed $x): NDArray|float
    {
        return $this->call(self::PDF, $x);
    }

    public function logpdf(mixed $x): NDArray|float
    {
        return $this->call(self::LOGPDF, $x);
    }

    /** Probability mass function (discrete distributions). */
    public function pmf(mixed $k): NDArray|float
    {
        return $this->call(self::PDF, $k);
    }

    public function logpmf(mixed $k): NDArray|float
    {
        return $this->call(self::LOGPDF, $k);
    }

    public function cdf(mixed $x): NDArray|float
    {
        return $this->call(self::CDF, $x);
    }

    public function logcdf(mixed $x): NDArray|float
    {
        return $this->call(self::LOGCDF, $x);
    }

    public function sf(mixed $x): NDArray|float
    {
        return $this->call(self::SF, $x);
    }

    public function logsf(mixed $x): NDArray|float
    {
        return $this->call(self::LOGSF, $x);
    }

    public function ppf(mixed $q): NDArray|float
    {
        return $this->call(self::PPF, $q);
    }

    public function isf(mixed $q): NDArray|float
    {
        return $this->call(self::ISF, $q);
    }

    /**
     * Mean ('m'), variance ('v'), skew ('s') and/or excess kurtosis ('k').
     *
     * @return NDArray|float|array<string, NDArray|float> one value, or the requested ones keyed by
     *                                                    mean, var, skew, kurtosis (in that order)
     */
    public function stats(string $moments = 'mv'): NDArray|float|array
    {
        $mask = 0;
        foreach (str_split($moments) as $c) {
            $mask |= match ($c) {
                'm' => 1, 'v' => 2, 's' => 4, 'k' => 8,
                default => throw new \ValueError("Distribution::stats(): moments must contain only 'm', 'v', 's', 'k'"),
            };
        }
        $all = Registry::ufunc($this->name, [(float) $mask, ...$this->params], null, self::STATS);
        $out = [];
        foreach (['mean' => 1, 'var' => 2, 'skew' => 4, 'kurtosis' => 8] as $k => $bit) {
            if ($mask & $bit) {
                $out[$k] = $all[$k];
            }
        }

        return count($out) === 1 ? reset($out) : $out;
    }

    public function mean(): NDArray|float
    {
        return $this->stats('m');
    }

    public function var(): NDArray|float
    {
        return $this->stats('v');
    }

    public function std(): NDArray|float
    {
        $v = $this->stats('v');

        return $v instanceof NDArray ? $v->sqrt() : sqrt($v);
    }

    public function median(): NDArray|float
    {
        return $this->call(self::PPF, 0.5);
    }

    public function entropy(): NDArray|float
    {
        return Registry::ufunc($this->name, $this->params, null, self::ENTROPY);
    }

    /** Non-central moment of the given order. */
    public function moment(int $order): NDArray|float
    {
        return $this->call(self::MOMENT, (float) $order);
    }

    /** @return array{0: NDArray|float, 1: NDArray|float} the support's end points */
    public function support(): array
    {
        $r = Registry::ufunc($this->name, $this->params, null, self::SUPPORT);

        return [$r['a'], $r['b']];
    }

    /** @return array{0: NDArray|float, 1: NDArray|float} the central interval holding `confidence` of the mass */
    public function interval(float $confidence): array
    {
        if (! ($confidence >= 0 && $confidence <= 1)) {
            throw new \ValueError('Distribution::interval(): confidence must be between 0 and 1 inclusive');
        }

        return [$this->call(self::PPF, (1.0 - $confidence) / 2), $this->call(self::PPF, (1.0 + $confidence) / 2)];
    }

    /**
     * Random variates, drawn from $randomState's stream exactly as scipy.stats draws them from a
     * numpy.random.Generator (rvs(size, random_state=numpy.random.default_rng(seed))).
     *
     * @param int|list<int>|null $size output shape; null: the parameters' broadcast shape
     */
    public function rvs(int|array|null $size = null, ?Generator $randomState = null): NDArray|float|int
    {
        $gen = $randomState ?? (self::$defaultRng ??= Generator::defaultRng());
        $info = Registry::info($this->name);
        $ffi = Library::ffi();
        $ops = [];
        $shapes = [];
        foreach ($this->params as $p) {
            $a = $p instanceof NDArray ? $p->astype(DType::Float64) : NDArray::array($p, DType::Float64);
            // SciPy ignores leading length-1 dimensions of the parameters (_argcheck_rvs)
            $sh = $a->shape();
            while ($sh !== [] && $sh[0] === 1) {
                array_shift($sh);
            }
            $a = $a->reshape($sh);
            $ops[] = $a;
            $shapes[] = $sh;
        }
        $bshape = NDArray::broadcastShapes(...$shapes);
        $shape = $size === null ? $bshape : (is_int($size) ? [$size] : array_values($size));
        $nb = count($bshape);
        $ns = count($shape);
        $bcheck = $nb < $ns ? array_merge(array_fill(0, $ns - $nb, 1), $bshape) : $bshape;
        $scheck = $nb > $ns ? array_merge(array_fill(0, $nb - $ns, 1), $shape) : $shape;
        foreach ($bcheck as $k => $d) {
            if ($d !== 1 && $d !== $scheck[$k]) {
                throw new \ValueError('Distribution::rvs(): size does not match the broadcast shape of the parameters');
            }
        }
        $shape = $scheck;
        $out = NDArray::empty($shape, DType::Float64);
        $nd = count($shape);
        $nop = count($ops) + 1;
        $ptrs = $ffi->new("void*[{$nop}]");
        $strides = [];
        $keep = [];
        foreach ($ops as $k => $a) {
            $b = $a->broadcastTo($shape);
            $keep[] = $b;
            $ptrs[$k] = $ffi->cast('void*', $b->ptr());
            array_push($strides, ...$b->strides());
        }
        $ptrs[$nop - 1] = $ffi->cast('void*', $out->ptr());
        array_push($strides, ...$out->strides());
        $rc = $ffi->tsr_fn_rvs($info['id'], $gen->stateHandle(), $nd, $nd > 0 ? Library::i64($shape) : null, $nop, $ptrs,
            $nd > 0 ? Library::i64($strides) : null);
        unset($keep);
        if ($rc < 0) {
            throw new \ValueError($this->label('rvs') . ': ' . (string) $ffi->tsr_fn_error());
        }
        if ($this->discrete) {
            $out = $out->astype(DType::Int64);
        }

        return $nd === 0 ? $out->item() : $out;
    }

    private function call(int $method, mixed $x): NDArray|float
    {
        return Registry::ufunc($this->name, [$x, ...$this->params], null, $method);
    }

    private function label(string $method): string
    {
        return 'Stats::' . $this->name() . "()->{$method}()";
    }
}
