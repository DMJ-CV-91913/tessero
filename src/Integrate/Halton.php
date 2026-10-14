<?php

declare(strict_types=1);

namespace Tessero\Integrate;

use Tessero\NDArray;
use Tessero\Random\Generator;
use Tessero\Random\SeedSequence;

/**
 * Scrambled Halton low-discrepancy sequence (scipy.stats.qmc.Halton with scramble=True).
 *
 * One scrambled van der Corput sequence per dimension, using the first `d` primes as bases. The
 * scramble is Owen's "randomized Halton" (Owen 2017): for each base, a stack of
 * `ceil(54/log2(base)) - 1` random permutations of `[0, base)` is drawn by shuffling with the
 * engine's own PCG64 Generator. A point's digits are each remapped through the matching permutation
 * layer.
 *
 * The engine's Generator is a SPAWNED child of the seed passed to `qmc.Halton(seed=rng)` — scipy's
 * QMCEngine._initialize does `self.rng = _rng_spawn(rng, 1)[0]`. {@see SeedSequence} reproduces that
 * spawn bit-for-bit, so the permutation draws (and hence every sample) match scipy exactly.
 *
 * Pure PHP, host-independent (no kernel maths beyond the PCG64 stream, no LAPACK).
 */
final class Halton
{
    /** First 100 primes — enough for any realistic qmc_quad dimensionality. */
    private const PRIMES = [
        2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
        73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151,
        157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233,
        239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317,
        331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419,
        421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503,
        509, 521, 523, 541,
    ];

    private int $d;

    /** @var list<int> base (prime) per dimension */
    private array $base;

    /** @var list<list<list<int>>> permutations[dim][layer] = permutation of [0, base) */
    private array $permutations;

    private int $numGenerated = 0;

    /**
     * @param int          $d   dimensionality (number of variables)
     * @param SeedSequence $seq the engine's SeedSequence (ALREADY the spawned child, matching
     *                          scipy's self.rng) — its generator() drives the scramble draws
     */
    public function __construct(int $d, SeedSequence $seq)
    {
        if ($d < 1) {
            throw new \InvalidArgumentException('Halton: d must be a positive integer.');
        }
        if ($d > count(self::PRIMES)) {
            throw new \InvalidArgumentException('Halton: d exceeds the available prime table.');
        }
        $this->d = $d;
        $this->base = array_slice(self::PRIMES, 0, $d);

        $gen = $seq->generator();
        $this->permutations = [];
        foreach ($this->base as $b) {
            $this->permutations[] = self::vdcPermutations($b, $gen);
        }
    }

    /**
     * Draw `$n` points in [0, 1)^d, continuing the sequence from the current index.
     *
     * @return list<list<float>> an n×d array
     */
    public function random(int $n): array
    {
        $cols = [];
        foreach ($this->base as $i => $b) {
            $cols[$i] = self::vdcScrambled($n, $b, $this->numGenerated, $this->permutations[$i]);
        }
        $out = [];
        for ($r = 0; $r < $n; $r++) {
            $row = [];
            for ($c = 0; $c < $this->d; $c++) {
                $row[] = $cols[$c][$r];
            }
            $out[] = $row;
        }
        $this->numGenerated += $n;

        return $out;
    }

    public function d(): int
    {
        return $this->d;
    }

    /**
     * Owen permutations for one base: ceil(54/log2(base)) - 1 shuffled copies of [0, base), drawn in
     * order from $gen (scipy._qmc._van_der_corput_permutations).
     *
     * @return list<list<int>>
     */
    private static function vdcPermutations(int $base, Generator $gen): array
    {
        $count = (int) ceil(54.0 / log($base, 2)) - 1;
        $perms = [];
        for ($k = 0; $k < $count; $k++) {
            $perm = NDArray::arange($base);
            $gen->shuffle($perm);
            $perms[] = array_map('intval', $perm->toArray());
        }

        return $perms;
    }

    /**
     * Owen-scrambled van der Corput digits (scipy _qmc_cy._cy_van_der_corput_scrambled_loop).
     *
     * @param list<list<int>> $permutations
     *
     * @return list<float>
     */
    private static function vdcScrambled(int $n, int $base, int $startIndex, array $permutations): array
    {
        $nperm = count($permutations);
        $seq = [];
        for ($i = 0; $i < $n; $i++) {
            $quotient = $startIndex + $i;
            $b2r = 1.0 / $base;
            $acc = 0.0;
            for ($j = 0; $j < $nperm; $j++) {
                $remainder = $quotient % $base;
                $remainder = $permutations[$j][$remainder];
                $acc += $remainder * $b2r;
                $b2r /= $base;
                $quotient = intdiv($quotient, $base);
            }
            $seq[] = $acc;
        }

        return $seq;
    }
}
