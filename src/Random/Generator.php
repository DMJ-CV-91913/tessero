<?php

declare(strict_types=1);

namespace Tessero\Random;

use FFI\CData;
use Tessero\DType;
use Tessero\Exceptions\ShapeError;
use Tessero\Exceptions\TesseroException;
use Tessero\NDArray;
use Tessero\Native\Library;

/**
 * PCG64 generator seeded through NumPy's SeedSequence.
 *
 *   Generator::defaultRng(42)->random(5)   produces exactly   numpy.random.default_rng(42).random(5)
 *
 * Stream-identical to NumPy: every distribution method (GeneratorMethods, generated from the kernel's
 * function registry and drawn with NumPy's own distribution code), integers, choice (with replacement,
 * with or without p), permutation and shuffle.
 */
final class Generator
{
    use GeneratorMethods;

    private CData $state;

    private function __construct()
    {
        $this->state = Library::ffi()->new('uint64_t[6]');
    }

    /** @param int|list<int>|null $seed non-negative; null draws 128 bits from the OS */
    public static function defaultRng(int|array|null $seed = null): self
    {
        $words = [];
        if ($seed === null) {
            $words = array_values(unpack('V4', random_bytes(16)));
        } else {
            foreach (is_int($seed) ? [$seed] : $seed as $s) {
                if (! is_int($s) || $s < 0) {
                    throw new TesseroException('Seeds must be non-negative integers.');
                }
                if (is_array($seed) && $s > 0xFFFFFFFF) {
                    throw new TesseroException('Seed array elements must fit in 32 bits.');
                }
                // an int seed becomes little-endian 32-bit words, as numpy's SeedSequence does
                $words[] = $s & 0xFFFFFFFF;
                if ($s > 0xFFFFFFFF) {
                    $words[] = ($s >> 32) & 0xFFFFFFFF;
                }
            }
        }
        $ffi = Library::ffi();
        $n = count($words);
        $entropy = $ffi->new('uint32_t[' . max(1, $n) . ']');
        foreach ($words as $i => $w) {
            $entropy[$i] = $w;
        }
        $pool = $ffi->new('uint64_t[4]');
        $ffi->tsr_seed_sequence($entropy, $n, $pool, 4);
        $g = new self();
        $ffi->tsr_pcg64_seed($g->state, $pool[0], $pool[1], $pool[2], $pool[3]);

        return $g;
    }

    /** Integers in [low, high); with one bound, [0, low). */
    public function integers(int $low, ?int $high = null, int|array|null $size = null): NDArray|int
    {
        if ($high === null) {
            [$low, $high] = [0, $low];
        }
        if ($high <= $low) {
            throw new TesseroException('integers: high must be greater than low.');
        }
        [$shape, $n] = self::size($size);
        $out = NDArray::empty($shape, DType::Int64);
        if ($n > 0) {
            $ffi = Library::ffi();
            $ffi->tsr_pcg64_integers($this->state, $n, $low, $high, $ffi->cast('int64_t*', $out->ptr()));
        }

        return $size === null ? $out->item() : $out;
    }

    /**
     * `$length` random bytes as a binary string (numpy.random.Generator.bytes). Even $length 0 advances
     * the stream by one 32-bit draw, exactly as NumPy does.
     */
    public function bytes(int $length): string
    {
        if ($length < 0) {
            throw new TesseroException('bytes: length must be non-negative.');
        }
        $ffi = Library::ffi();
        $buf = $ffi->new('uint8_t[' . max(1, $length) . ']');
        $ffi->tsr_pcg64_bytes($this->state, $length, $buf);

        return $length === 0 ? '' : \FFI::string($buf, $length);
    }

    /**
     * Sample from a 1-D array (or arange(a) for an int). Probabilities $p use
     * NumPy's CDF-inversion, so the draws match numpy's choice(..., p=p).
     */
    public function choice(int|array|NDArray $a, int|array|null $size = null, bool $replace = true, ?array $p = null): mixed
    {
        $pop = is_int($a) ? NDArray::arange($a) : NDArray::asArray($a);
        if (count($pop->shape()) !== 1) {
            throw new ShapeError('choice: population must be 1-D.');
        }
        $n = $pop->shape()[0];
        if ($n === 0) {
            throw new TesseroException('choice: population is empty.');
        }
        [$shape, $count] = self::size($size);
        if ($p !== null) {
            if (count($p) !== $n) {
                throw new ShapeError('choice: p must have one probability per element.');
            }
            if (! $replace) {
                throw new TesseroException('choice without replacement and with p is not supported.');
            }
            $cdf = NDArray::array($p, DType::Float64)->cumsum();
            $cdf = $cdf->div($cdf->item($n - 1));
            $u = $this->random($shape === [] ? [1] : $shape)->ravel()->toList();
            $cdfList = $cdf->toList();
            $idx = array_map(static function (float $x) use ($cdfList): int {
                $lo = 0;
                $hi = count($cdfList);
                while ($lo < $hi) {                      // searchsorted(side='right')
                    $mid = ($lo + $hi) >> 1;
                    if ($cdfList[$mid] <= $x) {
                        $lo = $mid + 1;
                    } else {
                        $hi = $mid;
                    }
                }

                return $lo;
            }, $u);
            $picked = $pop->take($idx);
        } elseif ($replace) {
            $picked = $pop->take($this->integers(0, $n, [max(1, $count)]));
        } else {
            if ($count > $n) {
                throw new TesseroException('choice: cannot take a larger sample than the population without replacement.');
            }
            $picked = $pop->take($this->permutation($n)->slice('0:' . $count));
        }
        if ($size === null) {
            return $picked->item(0);
        }

        return $picked->reshape($shape);
    }

    /** Random permutation of arange(n), or a shuffled copy of an array along axis 0. */
    public function permutation(int|NDArray|array $x): NDArray
    {
        $arr = is_int($x) ? NDArray::arange($x) : NDArray::asArray($x)->copy();
        $this->shuffle($arr);

        return $arr;
    }

    /** Shuffle along axis 0 in place. */
    public function shuffle(NDArray $x): void
    {
        if ($x->shape() === []) {
            throw new ShapeError('shuffle needs at least one dimension.');
        }
        if (! $x->isWritable()) {
            // a read-only memmap is mapped PROT_READ: writing it from C would crash the process
            throw new \Tessero\Exceptions\TesseroException('shuffle: the array is read-only.');
        }
        $work = $x->isContiguous() ? $x : $x->copy();
        $n = $x->shape()[0];
        $item = $n > 0 ? intdiv($work->nbytes(), $n) : 1;
        Library::check(Library::ffi()->tsr_pcg64_shuffle($this->state, $n, max(1, $item), $work->ptr()), 'shuffle');
        if ($work !== $x) {
            $x->assign($work);
        }
    }

    /**
     * Draw from a Dirichlet distribution (numpy.random.Generator.dirichlet). Each sample is a vector of
     * length count($alpha) that sums to 1, built from independent standard-gamma draws then normalised.
     *
     * @param list<int|float> $alpha
     */
    public function dirichlet(array $alpha, int|array|null $size = null): NDArray
    {
        $k = count($alpha);
        foreach ($alpha as $a) {
            if ($a < 0) {
                throw new TesseroException('dirichlet: alpha must be non-negative.');
            }
        }
        [$shape, $rows] = self::size($size);
        $rows = max(1, $rows);
        $flat = [];
        for ($i = 0; $i < $rows; $i++) {
            foreach ($alpha as $a) {
                $flat[] = (float) $a;
            }
        }
        $out = [];
        if ($k > 0) {
            $g = $this->standardGamma(NDArray::array($flat, DType::Float64))->toList();
            for ($i = 0; $i < $rows; $i++) {
                $s = 0.0;
                for ($j = 0; $j < $k; $j++) {
                    $s += $g[$i * $k + $j];
                }
                $inv = 1.0 / $s;
                for ($j = 0; $j < $k; $j++) {
                    $out[] = $g[$i * $k + $j] * $inv;
                }
            }
        }

        return NDArray::array($out, DType::Float64)->reshape([...$shape, $k]);
    }

    /**
     * Draw samples from a multinomial distribution (numpy.random.Generator.multinomial): for each of the
     * len($pvals) categories, a count, built from sequential conditional binomial draws.
     *
     * @param list<int|float> $pvals
     */
    public function multinomial(int $n, array $pvals, int|array|null $size = null): NDArray
    {
        $k = count($pvals);
        if ($k < 1) {
            throw new TesseroException('multinomial: pvals must have at least one element.');
        }
        $head = 0.0;
        for ($j = 0; $j < $k - 1; $j++) {
            $head += $pvals[$j];
        }
        if ($head > 1.0 + 1e-12) {
            throw new TesseroException('multinomial: sum(pvals[:-1]) > 1.0');
        }
        [$shape, $rows] = self::size($size);
        $rows = max(1, $rows);
        $out = [];
        for ($i = 0; $i < $rows; $i++) {
            $remaining = $n;
            $psum = 1.0;
            for ($j = 0; $j < $k - 1; $j++) {
                $pj = $psum > 0.0 ? $pvals[$j] / $psum : 0.0;
                if ($pj > 1.0) {
                    $pj = 1.0;
                }
                $bj = $remaining > 0 ? $this->binomial($remaining, $pj) : 0;
                $out[] = $bj;
                $remaining -= $bj;
                $psum -= $pvals[$j];
            }
            $out[] = $remaining;
        }

        return NDArray::array($out, DType::Int64)->reshape([...$shape, $k]);
    }

    /**
     * Randomly permute values along an axis, each slice independently (numpy.random.Generator.permuted).
     * With $axis null the flattened array is permuted. Returns a new array (or writes into $out).
     */
    public function permuted(NDArray $x, ?int $axis = null, ?NDArray $out = null): NDArray
    {
        $ffi = Library::ffi();
        if ($axis === null) {
            $work = $x->flatten();
            $n = $work->size();
            if ($n > 1) {
                $item = intdiv($work->nbytes(), $n);
                Library::check($ffi->tsr_pcg64_shuffle($this->state, $n, max(1, $item), $work->ptr()), 'permuted');
            }
            $res = $work->reshape($x->shape());
        } else {
            $moved = $x->moveAxis($axis, -1)->copy();
            $shape = $moved->shape();
            $l = $shape[count($shape) - 1];
            $n = $moved->size();
            if ($l > 1) {
                $item = intdiv($moved->nbytes(), $n);
                $buf = $moved->buffer();
                $rows = intdiv($n, $l);
                for ($r = 0; $r < $rows; $r++) {
                    Library::check($ffi->tsr_pcg64_shuffle($this->state, $l, $item, $buf->ptr($r * $l * $item)), 'permuted');
                }
            }
            $res = $moved->moveAxis(-1, $axis)->copy();
        }
        if ($out !== null) {
            return $out->assign($res);
        }

        return $res;
    }

    /**
     * Draw from a multivariate hypergeometric distribution (numpy.random.Generator.multivariate_hypergeometric,
     * method='marginals'): sequential univariate hypergeometric draws over the colours.
     *
     * @param list<int> $colors
     */
    public function multivariateHypergeometric(array $colors, int $nsample, int|array|null $size = null, string $method = 'marginals'): NDArray
    {
        if ($method !== 'marginals') {
            throw new TesseroException("multivariate_hypergeometric: only method='marginals' is supported.");
        }
        $k = count($colors);
        $total = 0;
        foreach ($colors as $c) {
            if ($c < 0) {
                throw new TesseroException('multivariate_hypergeometric: colors must be nonnegative.');
            }
            $total += $c;
        }
        if ($nsample < 0) {
            throw new TesseroException('multivariate_hypergeometric: nsample must be nonnegative.');
        }
        if ($nsample > $total) {
            throw new TesseroException('multivariate_hypergeometric: nsample > sum(colors)');
        }
        [$shape, $rows] = self::size($size);
        $rows = max(1, $rows);
        $out = [];
        for ($i = 0; $i < $rows; $i++) {
            $remaining = $total;
            $ns = $nsample;
            for ($j = 0; $j < $k - 1; $j++) {
                $ng = $colors[$j];
                $nb = $remaining - $colors[$j];
                $r = ($ns > 0 && $remaining > 0) ? $this->hypergeometric($ng, $nb, $ns) : 0;
                $out[] = $r;
                $ns -= $r;
                $remaining -= $colors[$j];
            }
            $out[] = $ns;
        }

        return NDArray::array($out, DType::Int64)->reshape([...$shape, $k]);
    }

    /** @return list<string> the 6 state words as decimal strings (for persistence) */
    /**
     * The PCG64 state vector, for the kernel calls that draw from this generator (scipy.stats rvs).
     *
     * @internal
     */
    public function stateHandle(): CData
    {
        return $this->state;
    }

    public function getState(): array
    {
        $out = [];
        for ($i = 0; $i < 6; $i++) {
            $out[] = sprintf('%u', $this->state[$i]);
        }

        return $out;
    }

    /** @param list<string|int> $words */
    public function setState(array $words): void
    {
        if (count($words) !== 6) {
            throw new TesseroException('PCG64 state has 6 words.');
        }
        foreach (array_values($words) as $i => $w) {
            $this->state[$i] = is_int($w) ? $w : self::u64((string) $w);
        }
    }

    private static function u64(string $decimal): int
    {
        // parse an unsigned 64-bit decimal into PHP's signed int with the same bits
        $hi = 0;
        $lo = 0;
        foreach (str_split($decimal) as $ch) {
            $d = ord($ch) - 48;
            // (hi, lo) = (hi, lo) * 10 + d  with 32-bit limbs
            $lo = $lo * 10 + $d;
            $hi = $hi * 10 + ($lo >> 32);
            $lo &= 0xFFFFFFFF;
            $hi &= 0xFFFFFFFF;
        }

        return ($hi << 32) | $lo;
    }

    private function d(NDArray $a): CData
    {
        return Library::ffi()->cast('double*', $a->ptr());
    }

    /** @return array{0: list<int>, 1: int} */
    private static function size(int|array|null $size): array
    {
        $shape = $size === null ? [] : (is_int($size) ? [$size] : array_values($size));

        return [$shape, (int) array_product($shape)];
    }
}

