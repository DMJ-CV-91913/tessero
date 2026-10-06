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

