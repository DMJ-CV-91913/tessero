<?php

declare(strict_types=1);

namespace Tessero\Cluster\Hierarchy;

/**
 * scipy.cluster.hierarchy.DisjointSet: union-find (merge-find) with path halving in find (__getitem__) and
 * merge-by-size in merge(). Deterministic: on equal-size merges the element inserted first becomes the root,
 * and iteration / subsets() follow insertion order. Pure PHP (no kernel); identical on both backends.
 *
 * Elements are hashable scalars (int|string|float|bool). Insertion order is tracked for the deterministic
 * tie-break and for __iter__/subsets ordering, matching scipy exactly.
 */
final class DisjointSet implements \Countable, \IteratorAggregate, \ArrayAccess
{
    public int $nSubsets = 0;

    /** @var array<int|string, int> element => subtree size (valid only at roots) */
    private array $sizes = [];

    /** @var array<int|string, int|string> element => parent element */
    private array $parents = [];

    /** @var array<int|string, int|string> circular linked list over each subset */
    private array $nbrs = [];

    /** @var array<int|string, int> element => insertion index */
    private array $indices = [];

    /** @var array<int|string, int|string|float|bool> normalized-key => original element value */
    private array $values = [];

    /** @param iterable<int|string|float|bool>|null $elements */
    public function __construct(?iterable $elements = null)
    {
        if ($elements !== null) {
            foreach ($elements as $x) {
                $this->add($x);
            }
        }
    }

    /** Normalize an element to an array key preserving type distinctions scipy's dict would keep. */
    private static function key(int|string|float|bool $x): int|string
    {
        if (is_int($x)) {
            return 'i:' . $x;
        }
        if (is_bool($x)) {
            return $x ? 'i:1' : 'i:0';   // Python: True == 1, False == 0 as dict keys
        }
        if (is_float($x)) {
            return (float) $x === floor($x) && is_finite($x) ? 'i:' . (int) $x : 'f:' . $x;
        }

        return 's:' . $x;
    }

    /** Add an element to the disjoint set (no-op if present). scipy DisjointSet.add. */
    public function add(int|string|float|bool $x): void
    {
        $k = self::key($x);
        if (isset($this->indices[$k])) {
            return;
        }
        $this->sizes[$k] = 1;
        $this->parents[$k] = $k;
        $this->nbrs[$k] = $k;
        $this->indices[$k] = count($this->indices);
        $this->values[$k] = $x;
        $this->nSubsets++;
    }

    /** Find the root element of $x using path halving. scipy __getitem__. */
    private function findKey(int|string|float|bool $x): int|string
    {
        $k = self::key($x);
        if (! isset($this->indices[$k])) {
            throw new \OutOfBoundsException('DisjointSet: element not found');
        }
        while ($this->indices[$k] !== $this->indices[$this->parents[$k]]) {
            $this->parents[$k] = $this->parents[$this->parents[$k]];
            $k = $this->parents[$k];
        }

        return $k;
    }

    /** The root element value of $x (scipy __getitem__ / find). */
    public function find(int|string|float|bool $x): int|string|float|bool
    {
        return $this->values[$this->findKey($x)];
    }

    /**
     * Merge the subsets containing $x and $y. The smaller subset is attached to the larger; on a tie the
     * earlier-inserted root wins. Returns true iff they were in different subsets. scipy merge.
     */
    public function merge(int|string|float|bool $x, int|string|float|bool $y): bool
    {
        $xr = $this->findKey($x);
        $yr = $this->findKey($y);
        if ($this->indices[$xr] === $this->indices[$yr]) {
            return false;
        }
        // (sizes[xr], indices[yr]) < (sizes[yr], indices[xr]) -> swap so xr is the (larger / earlier) parent
        $lt = $this->sizes[$xr] < $this->sizes[$yr]
            || ($this->sizes[$xr] === $this->sizes[$yr] && $this->indices[$yr] < $this->indices[$xr]);
        if ($lt) {
            [$xr, $yr] = [$yr, $xr];
        }
        $this->parents[$yr] = $xr;
        $this->sizes[$xr] += $this->sizes[$yr];
        [$this->nbrs[$xr], $this->nbrs[$yr]] = [$this->nbrs[$yr], $this->nbrs[$xr]];
        $this->nSubsets--;

        return true;
    }

    /** Whether $x and $y are in the same subset. scipy connected. */
    public function connected(int|string|float|bool $x, int|string|float|bool $y): bool
    {
        return $this->indices[$this->findKey($x)] === $this->indices[$this->findKey($y)];
    }

    /**
     * The subset containing $x, as a list of element values in the subset's internal circular order
     * (scipy returns a Python set; order is not significant). scipy subset.
     *
     * @return list<int|string|float|bool>
     */
    public function subset(int|string|float|bool $x): array
    {
        $k = self::key($x);
        if (! isset($this->indices[$k])) {
            throw new \OutOfBoundsException('DisjointSet: element not found');
        }
        $result = [$this->values[$k]];
        $nxt = $this->nbrs[$k];
        while ($this->indices[$nxt] !== $this->indices[$k]) {
            $result[] = $this->values[$nxt];
            $nxt = $this->nbrs[$nxt];
        }

        return $result;
    }

    /** The size of the subset containing $x, without materializing it. scipy subset_size. */
    public function subsetSize(int|string|float|bool $x): int
    {
        $k = self::key($x);
        if (! isset($this->indices[$k])) {
            throw new \OutOfBoundsException('DisjointSet: element not found');
        }

        return $this->sizes[$this->findKey($x)];
    }

    /**
     * All subsets, each a list of element values, in the order subsets are first encountered scanning
     * elements by insertion order. scipy subsets.
     *
     * @return list<list<int|string|float|bool>>
     */
    public function subsets(): array
    {
        $result = [];
        $seen = [];
        foreach ($this->indices as $k => $_) {
            if (! isset($seen[$k])) {
                $xset = $this->subset($this->values[$k]);
                foreach ($xset as $v) {
                    $seen[self::key($v)] = true;
                }
                $result[] = $xset;
            }
        }

        return $result;
    }

    public function count(): int
    {
        return count($this->indices);
    }

    /** @return \Traversable<int, int|string|float|bool> elements in insertion order */
    public function getIterator(): \Traversable
    {
        foreach (array_keys($this->indices) as $k) {
            yield $this->values[$k];
        }
    }

    public function offsetExists(mixed $offset): bool
    {
        return isset($this->indices[self::key($offset)]);
    }

    /** DisjointSet[$x] -> root element (scipy __getitem__). */
    public function offsetGet(mixed $offset): mixed
    {
        return $this->find($offset);
    }

    public function offsetSet(mixed $offset, mixed $value): void
    {
        throw new \LogicException('DisjointSet does not support item assignment');
    }

    public function offsetUnset(mixed $offset): void
    {
        throw new \LogicException('DisjointSet does not support item deletion');
    }
}
