<?php

declare(strict_types=1);

namespace Tessero\Sparse;

use Tessero\Exceptions\TesseroException;
use Tessero\NDArray;
use Tessero\Random\Generator;

/**
 * scipy.sparse.random_array / random / rand — sparse arrays/matrices of uniformly random
 * values in [0, 1) at a requested density, with the sparsity structure drawn from a PCG64
 * Generator.
 *
 * Pure-PHP facade over Tessero's bit-exact PCG64 {@see Generator}; identical on both backends
 * (FFI and the native extension) because the only entropy source is the Generator. The result
 * is bit-for-bit identical to scipy when the SAME seed is used with rng=np.random.default_rng(seed):
 * an int seed passed to scipy builds a legacy MT19937 RandomState whose stream differs from PCG64
 * and would be unmatchable, so parity fixtures MUST use default_rng(seed).
 *
 * Algorithm (identical to scipy.sparse._construct._random):
 *   - size = round(density * prod(shape)) structurally-nonzero entries
 *   - raveled_ind = rng.choice(prod(shape), size=size, replace=False)      # see {@see choiceNoReplace}
 *   - (row, col) = np.unravel_index(raveled_ind, shape, order='F')         # column-major
 *   - data = rng.uniform(size=size)                                        # drawn AFTER the indices
 * The coo components are then handed to coo_array((data, (row, col)), shape).
 *
 * The reproduction of numpy.random.Generator.choice(pop, size, replace=False, p=None) is the crux.
 * numpy uses one of two algorithms (see numpy/random/_generator.pyx), both driven by the Lemire
 * bounded-integer draw random_bounded_uint64(bitgen, 0, k, 0, 0) — i.e. Generator::integers(0, k+1),
 * NOT the masked-rejection random_interval used by shuffle()/permutation():
 *   - Floyd's algorithm (default, when pop <= 10000 or size <= pop/50):
 *       for j in [pop-size, pop):
 *           val = integers(0, j+1)
 *           if val unseen: use val; else use j            # both marked seen
 *       then _shuffle_int(size, 1): Fisher-Yates over the size results, i=size-1..1, j=integers(0,i+1)
 *   - tail-shuffle (when pop > 10000 and size > pop/50):
 *       idx = range(pop); _shuffle_int(pop, max(pop-size,1)): i=pop-1..pop-size, j=integers(0,i+1)
 *       return idx[pop-size:]
 */
final class SparseRandom
{
    /**
     * scipy.sparse.random_array(shape, density, rng=...). Returns the COO triplets as a map
     * {'row': int[], 'col': int[], 'data': float[], 'shape': [m, n]}.
     *
     * @param array{int,int}            $shape   (rows, cols)
     * @param float                     $density in [0, 1]
     * @param int|array<int>|Generator  $rng     seed for a PCG64 Generator, or a Generator instance
     *
     * @return array{row: list<int>, col: list<int>, data: list<float>, shape: array{int,int}}
     */
    public static function randomArray(array $shape, float $density = 0.01, int|array|Generator $rng = 0): array
    {
        if ($density < 0.0 || $density > 1.0) {
            throw new TesseroException('random_array: density expected to be 0 <= density <= 1.');
        }
        $shape = array_values($shape);
        $m = (int) $shape[0];
        $n = (int) $shape[1];
        $tot = $m * $n;
        $size = (int) round($density * $tot);

        $g = $rng instanceof Generator ? $rng : Generator::defaultRng($rng);

        $raveled = self::choiceNoReplace($g, $tot, $size);

        // np.unravel_index(raveled, (m, n), order='F') -> column-major: row = k % m, col = k // m
        $row = [];
        $col = [];
        foreach ($raveled as $k) {
            $row[] = $k % $m;
            $col[] = intdiv($k, $m);
        }

        // data = rng.uniform(size=size), drawn AFTER the index choice (stream order matters)
        $data = $size > 0 ? $g->uniform(0.0, 1.0, [$size])->toList() : [];

        return ['row' => $row, 'col' => $col, 'data' => $data, 'shape' => [$m, $n]];
    }

    /**
     * scipy.sparse.random(m, n, density, rng=...) — identical math to {@see randomArray}; scipy only
     * differs in returning a coo_matrix rather than a coo_array.
     *
     * @return array{row: list<int>, col: list<int>, data: list<float>, shape: array{int,int}}
     */
    public static function random(int $m, ?int $n = null, float $density = 0.01, int|array|Generator $rng = 0): array
    {
        return self::randomArray([$m, $n ?? $m], $density, $rng);
    }

    /**
     * scipy.sparse.rand(m, n, density, rng=...) — alias of {@see random} (uniform [0,1) data only).
     *
     * @return array{row: list<int>, col: list<int>, data: list<float>, shape: array{int,int}}
     */
    public static function rand(int $m, ?int $n = null, float $density = 0.01, int|array|Generator $rng = 0): array
    {
        return self::randomArray([$m, $n ?? $m], $density, $rng);
    }

    /**
     * numpy.random.Generator.choice(pop, size, replace=False, p=None), shuffle=True — returns the
     * chosen raveled indices in numpy's exact order. All index draws use the Lemire bounded path
     * (Generator::integers), matching numpy's random_bounded_uint64(..., use_masked=0).
     *
     * @return list<int>
     */
    public static function choiceNoReplace(Generator $g, int $pop, int $size): array
    {
        if ($size < 0 || $size > $pop) {
            throw new TesseroException('choice: cannot take a larger sample than the population without replacement.');
        }
        if ($size === 0) {
            return [];
        }

        $cutoff = 50; // numpy uses 50 when shuffle=True (scipy's default), 20 when shuffle=False
        if ($pop > 10000 && $size > intdiv($pop, $cutoff)) {
            // tail-shuffle branch: shuffle the last `size` positions of arange(pop), take the tail
            $idx = range(0, $pop - 1);
            $first = max($pop - $size, 1);
            for ($i = $pop - 1; $i >= $first; $i--) {
                $j = $g->integers(0, $i + 1);        // Lemire closed [0, i]
                $t = $idx[$j];
                $idx[$j] = $idx[$i];
                $idx[$i] = $t;
            }

            return array_slice($idx, $pop - $size);
        }

        // Floyd's algorithm
        $seen = [];
        $idx = array_fill(0, $size, 0);
        for ($j = $pop - $size; $j < $pop; $j++) {
            $val = $g->integers(0, $j + 1);          // Lemire closed [0, j]
            if (! isset($seen[$val])) {
                $seen[$val] = true;
                $idx[$j - ($pop - $size)] = $val;
            } else {
                $seen[$j] = true;
                $idx[$j - ($pop - $size)] = $j;
            }
        }
        // _shuffle_int(size, 1): Fisher-Yates over the size results
        for ($i = $size - 1; $i >= 1; $i--) {
            $jj = $g->integers(0, $i + 1);           // Lemire closed [0, i]
            $t = $idx[$jj];
            $idx[$jj] = $idx[$i];
            $idx[$i] = $t;
        }

        return $idx;
    }
}
