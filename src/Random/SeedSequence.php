<?php

declare(strict_types=1);

namespace Tessero\Random;

/**
 * NumPy SeedSequence spawning, reproduced exactly on top of Tessero's {@see Generator}.
 *
 * NumPy's `default_rng(seed)` builds a `SeedSequence` with `entropy = seed`, an empty `spawn_key`,
 * and `pool_size = 4`. `SeedSequence.spawn(n)` returns children whose `spawn_key` is the parent's
 * spawn_key with an extra index appended: `parent_spawn_key + (k,)` for k in the next n indices.
 *
 * A child's 256-bit pool (and therefore its PCG64 stream) is fully determined by its "assembled
 * entropy", which NumPy forms as:
 *
 *     assembled = pad(coerce(entropy), pool_size=4) + list(spawn_key)
 *
 * where coerce() turns an int into little-endian 32-bit words. Feeding that assembled word array to
 * `tsr_seed_sequence` (via {@see Generator::defaultRng} with an int-array seed) yields a Generator
 * whose stream is byte-identical to `numpy.random.Generator(PCG64(child_seed_sequence))`. This has
 * been verified bit-for-bit against NumPy for flat and nested spawn keys and for >32-bit seeds.
 *
 * This is the crux that unlocks scipy.integrate.qmc_quad (which spawns child QMCEngines via
 * scipy._lib._util._rng_spawn → SeedSequence.spawn).
 */
final class SeedSequence
{
    private const POOL_SIZE = 4;

    /** @var list<int> little-endian 32-bit entropy words of the root seed */
    private array $entropyWords;

    /** @var list<int> this node's spawn key (tuple of child indices) */
    private array $spawnKey;

    private int $childrenSpawned = 0;

    /**
     * @param int|list<int>|null $seed root seed (as passed to numpy.random.default_rng); null draws
     *                                 128 bits from the OS, matching Generator::defaultRng(null).
     * @param list<int>          $spawnKey internal — the spawn key of this node (empty for a root)
     */
    public function __construct(int|array|null $seed = null, array $spawnKey = [])
    {
        if ($seed === null) {
            $this->entropyWords = array_values(unpack('V4', random_bytes(16)));
        } elseif (is_int($seed)) {
            $this->entropyWords = self::coerceInt($seed);
        } else {
            $this->entropyWords = array_values($seed);
        }
        $this->spawnKey = array_values($spawnKey);
    }

    /**
     * Spawn `$n` independent child SeedSequences, exactly like NumPy's SeedSequence.spawn.
     *
     * @return list<self>
     */
    public function spawn(int $n): array
    {
        $children = [];
        for ($i = 0; $i < $n; $i++) {
            $key = $this->spawnKey;
            $key[] = $this->childrenSpawned + $i;
            $children[] = new self($this->entropyWords, $key);
        }
        $this->childrenSpawned += $n;

        return $children;
    }

    /**
     * Build a PCG64 {@see Generator} seeded from this SeedSequence's assembled entropy — the stream
     * is identical to numpy.random.Generator(PCG64(this_seed_sequence)).
     */
    public function generator(): Generator
    {
        return Generator::defaultRng($this->assembledEntropy());
    }

    /**
     * NumPy's assembled entropy: the root words padded to pool_size, with the spawn key appended.
     *
     * @return list<int>
     */
    public function assembledEntropy(): array
    {
        $run = $this->entropyWords;
        while (count($run) < self::POOL_SIZE) {
            $run[] = 0;
        }

        return array_merge($run, $this->spawnKey);
    }

    /**
     * A descendant SeedSequence of the SAME root entropy with an explicit spawn key. Lets callers
     * address a specific node in the spawn tree directly (e.g. qmc_quad's deterministic (0,i,0)
     * engine keys) without threading live spawn state.
     *
     * @param list<int> $spawnKey
     */
    public function childWithKey(array $spawnKey): self
    {
        return new self($this->entropyWords, $spawnKey);
    }

    /** Coerce a non-negative int seed to little-endian 32-bit words (numpy's _coerce_to_uint32_array). */
    private static function coerceInt(int $seed): array
    {
        if ($seed < 0) {
            throw new \InvalidArgumentException('Seeds must be non-negative integers.');
        }
        if ($seed === 0) {
            return [0];
        }
        $words = [];
        while ($seed > 0) {
            $words[] = $seed & 0xFFFFFFFF;
            $seed >>= 32;
        }

        return $words;
    }
}
