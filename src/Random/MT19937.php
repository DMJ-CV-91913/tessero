<?php

declare(strict_types=1);

namespace Tessero\Random;

/**
 * Bit-exact MT19937 matching NumPy's legacy {@see \numpy.random.RandomState} bit generator.
 *
 * NumPy's modern RNG (PCG64 + SeedSequence) is already reproduced by Tessero's kernel
 * ({@see Generator} / {@see SeedSequence}). This class covers the *legacy* generator that
 * `numpy.random.RandomState` is built on -- the 624-word Mersenne Twister with Matsumoto-
 * Nishimura tempering and NumPy's two seeding paths:
 *
 *   - a scalar int seed `0 <= s < 2**32` -> `mt19937_seed` (`init_genrand`),
 *   - an int array seed -> `mt19937_init_by_array`.
 *
 * (NumPy rejects scalar int seeds outside `[0, 2**32)` with a ValueError; only arrays carry
 * wider entropy, exactly as reproduced here.)
 *
 * The draw methods reproduce legacy `mtrand` semantics, which differ from the Generator ones:
 *   - `randomSample()` = `(a>>5)*2**26 + (b>>6)) / 2**53` from two 32-bit words,
 *   - `standardNormal()` = the legacy Box-Muller *polar* gauss (NOT the ziggurat),
 *   - `randint(low, high)` / `randomInterval(max)` = masked-rejection bounded integers
 *     (32-bit draw when the range fits in 32 bits, else a 64-bit draw),
 *   - `permutation(n)` / `shuffle()` = legacy Fisher-Yates using `randomInterval(i)`.
 *
 * Verified bit-for-bit against `numpy.random.RandomState` for int and array seeds across
 * randomSample / standardNormal / randint / permutation. This unlocks
 * `scipy.stats.multiscale_graphcorr`, whose permutation p-value seeds one MT19937
 * RandomState per replication.
 *
 * Pure-PHP (no kernel routine): MGC uses the stream at modest volume and the same code path
 * serves both backends, mirroring {@see SeedSequence} and {@see \Tessero\Stats\Pcg64Uint32Stream}.
 *
 * The algorithm follows the MT19937 reference code of Matsumoto and Nishimura (BSD-3-Clause) as distributed
 * in NumPy (randomkit, MIT); the copyright notices and licence terms are in licenses/LICENSE.mt19937.md.
 */
final class MT19937
{
    private const N = 624;
    private const M = 397;
    private const MATRIX_A = 0x9908b0df;
    private const UPPER_MASK = 0x80000000;
    private const LOWER_MASK = 0x7fffffff;
    private const UINT32 = 0xffffffff;

    /** @var array<int,int> 624 state words (each a uint32 held in a PHP int) */
    private array $mt;

    private int $pos = self::N;

    private bool $hasGauss = false;

    private float $gauss = 0.0;

    /**
     * @param int|list<int> $seed scalar int in [0, 2**32) (-> init_genrand) or an int array
     *                            (-> init_by_array), exactly as numpy.random.RandomState accepts.
     */
    public function __construct(int|array $seed)
    {
        $this->mt = array_fill(0, self::N, 0);
        if (is_int($seed)) {
            if ($seed < 0 || $seed > self::UINT32) {
                throw new \InvalidArgumentException('MT19937 scalar seed must be in [0, 2**32).');
            }
            $this->initGenrand($seed);
        } else {
            $this->initByArray(array_values($seed));
        }
    }

    /** numpy mt19937_seed / init_genrand. */
    private function initGenrand(int $s): void
    {
        $mt = &$this->mt;
        $mt[0] = $s & self::UINT32;
        for ($i = 1; $i < self::N; $i++) {
            $prev = $mt[$i - 1] ^ ($mt[$i - 1] >> 30);
            // 1812433253 * prev + i, kept to 32 bits. Use the low/high split to stay exact in PHP's
            // 64-bit int (1812433253 * 0xffffffff overflows 53-bit float mantissa but not int64).
            $mt[$i] = (1812433253 * $prev + $i) & self::UINT32;
        }
        $this->pos = self::N;
    }

    /** numpy mt19937_init_by_array. */
    private function initByArray(array $initKey): void
    {
        $initKey = array_map(static fn ($v) => ((int) $v) & self::UINT32, $initKey);
        $keyLen = count($initKey);
        $this->initGenrand(19650218);
        $mt = &$this->mt;

        $i = 1;
        $j = 0;
        $k = max(self::N, $keyLen);
        for (; $k > 0; $k--) {
            $prev = $mt[$i - 1] ^ ($mt[$i - 1] >> 30);
            $mt[$i] = (($mt[$i] ^ ($prev * 1664525)) + $initKey[$j] + $j) & self::UINT32;
            $i++;
            $j++;
            if ($i >= self::N) {
                $mt[0] = $mt[self::N - 1];
                $i = 1;
            }
            if ($j >= $keyLen) {
                $j = 0;
            }
        }
        for ($k = self::N - 1; $k > 0; $k--) {
            $prev = $mt[$i - 1] ^ ($mt[$i - 1] >> 30);
            $mt[$i] = (($mt[$i] ^ ($prev * 1566083941)) - $i) & self::UINT32;
            $i++;
            if ($i >= self::N) {
                $mt[0] = $mt[self::N - 1];
                $i = 1;
            }
        }
        $mt[0] = self::UPPER_MASK; // 0x80000000, MSB = 1 ensures non-zero initial array
    }

    /** The twist: regenerate all 624 words. */
    private function generate(): void
    {
        $mt = &$this->mt;
        $mag01 = [0, self::MATRIX_A];
        for ($kk = 0; $kk < self::N - self::M; $kk++) {
            $y = ($mt[$kk] & self::UPPER_MASK) | ($mt[$kk + 1] & self::LOWER_MASK);
            $mt[$kk] = $mt[$kk + self::M] ^ ($y >> 1) ^ $mag01[$y & 1];
        }
        for (; $kk < self::N - 1; $kk++) {
            $y = ($mt[$kk] & self::UPPER_MASK) | ($mt[$kk + 1] & self::LOWER_MASK);
            $mt[$kk] = $mt[$kk + (self::M - self::N)] ^ ($y >> 1) ^ $mag01[$y & 1];
        }
        $y = ($mt[self::N - 1] & self::UPPER_MASK) | ($mt[0] & self::LOWER_MASK);
        $mt[self::N - 1] = $mt[self::M - 1] ^ ($y >> 1) ^ $mag01[$y & 1];
        $this->pos = 0;
    }

    /** The next tempered 32-bit word (numpy next_uint32 / genrand_uint32). */
    public function nextUint32(): int
    {
        if ($this->pos >= self::N) {
            $this->generate();
        }
        $y = $this->mt[$this->pos];
        $this->pos++;
        $y ^= ($y >> 11);
        $y ^= ($y << 7) & 0x9d2c5680;
        $y ^= ($y << 15) & 0xefc60000;
        $y ^= ($y >> 18);

        return $y & self::UINT32;
    }

    /** A single double in [0, 1) (numpy legacy next_double / random_sample). */
    public function nextDouble(): float
    {
        $a = $this->nextUint32() >> 5; // 27 bits
        $b = $this->nextUint32() >> 6; // 26 bits

        return ($a * 67108864.0 + $b) / 9007199254740992.0;
    }

    /**
     * @return list<float> `n` doubles in [0, 1) (numpy RandomState.random_sample(n)).
     */
    public function randomSample(int $n): array
    {
        $out = [];
        for ($i = 0; $i < $n; $i++) {
            $out[] = $this->nextDouble();
        }

        return $out;
    }

    /** A single standard normal (numpy legacy gauss: Box-Muller polar). */
    public function nextGauss(): float
    {
        if ($this->hasGauss) {
            $this->hasGauss = false;

            return $this->gauss;
        }
        do {
            $x1 = 2.0 * $this->nextDouble() - 1.0;
            $x2 = 2.0 * $this->nextDouble() - 1.0;
            $r2 = $x1 * $x1 + $x2 * $x2;
        } while ($r2 >= 1.0 || $r2 === 0.0);
        $f = sqrt(-2.0 * log($r2) / $r2);
        $this->gauss = $f * $x1;
        $this->hasGauss = true;

        return $f * $x2;
    }

    /**
     * @return list<float> `n` standard normals (numpy RandomState.standard_normal(n)).
     */
    public function standardNormal(int $n): array
    {
        $out = [];
        for ($i = 0; $i < $n; $i++) {
            $out[] = $this->nextGauss();
        }

        return $out;
    }

    /**
     * A uniform integer in [0, $max] by numpy's smallest-mask rejection (rk_interval). Uses a 32-bit
     * draw when the range fits in 32 bits, else a 64-bit draw -- matching numpy's legacy buffered path.
     */
    public function randomInterval(int $max): int
    {
        if ($max === 0) {
            return 0;
        }
        $mask = $max;
        $mask |= $mask >> 1;
        $mask |= $mask >> 2;
        $mask |= $mask >> 4;
        $mask |= $mask >> 8;
        $mask |= $mask >> 16;
        $mask |= $mask >> 32;

        if ($max <= self::UINT32) {
            do {
                $v = $this->nextUint32() & $mask;
            } while ($v > $max);

            return $v;
        }
        do {
            $hi = $this->nextUint32();
            $lo = $this->nextUint32();
            $v = ((($hi << 32) | $lo)) & $mask;
        } while ($v > $max);

        return $v;
    }

    /**
     * @return list<int> `size` integers in [low, high) (numpy RandomState.randint, default int path).
     */
    public function randint(int $low, int $high, int $size): array
    {
        $rng = $high - $low - 1; // inclusive max offset
        $out = [];
        for ($i = 0; $i < $size; $i++) {
            $out[] = $low + $this->randomInterval($rng);
        }

        return $out;
    }

    /**
     * @return list<int> a permutation of [0, n) (numpy RandomState.permutation(n)).
     */
    public function permutation(int $n): array
    {
        $arr = range(0, $n - 1);
        for ($i = $n - 1; $i > 0; $i--) {
            $j = $this->randomInterval($i);
            $tmp = $arr[$i];
            $arr[$i] = $arr[$j];
            $arr[$j] = $tmp;
        }

        return $arr;
    }

    /**
     * In-place legacy Fisher-Yates shuffle (numpy RandomState.shuffle).
     *
     * @param array<int,mixed> $arr
     */
    public function shuffle(array &$arr): void
    {
        $n = count($arr);
        for ($i = $n - 1; $i > 0; $i--) {
            $j = $this->randomInterval($i);
            $tmp = $arr[$i];
            $arr[$i] = $arr[$j];
            $arr[$j] = $tmp;
        }
    }
}
