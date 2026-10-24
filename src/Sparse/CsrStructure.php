<?php

declare(strict_types=1);

namespace Tessero\Sparse;

use Tessero\Exceptions\ShapeError;

/**
 * Structural checks for compressed sparse (CSR / CSC) arrays, in pure PHP so the backend-agnostic readers
 * (Io\HarwellBoeing, Io\SparseNpz) and CsrMatrix::fromArrays apply the same rules. The sparse kernels index
 * data/indices through indptr, so arrays from a file must pass these before they reach native code.
 *
 * @internal
 */
final class CsrStructure
{
    private function __construct()
    {
    }

    /**
     * indptr has $major + 1 entries, starts at 0, never decreases and ends at $nnz; every index lies in
     * [0, $minor). CSR: major = rows, minor = cols. CSC: major = cols, minor = rows.
     *
     * @param list<mixed> $indptr
     * @param list<mixed>|null $indices null when the caller has already range-checked them
     */
    public static function validate(array $indptr, ?array $indices, int $nnz, int $major, int $minor, string $what): void
    {
        if ($major < 0 || $minor < 0) {
            throw new ShapeError("{$what}: negative matrix dimension.");
        }
        if (count($indptr) !== $major + 1 || ($indices !== null && count($indices) !== $nnz)) {
            throw new ShapeError("{$what}: inconsistent sparse arrays (indptr must have " . ($major + 1) . ' entries and indices one per stored value).');
        }
        if ($indptr[0] !== 0) {
            throw new ShapeError("{$what}: indptr must start at 0.");
        }
        $prev = 0;
        for ($i = 1; $i <= $major; $i++) {
            $p = $indptr[$i];
            if (! is_int($p) || $p < $prev) {
                throw new ShapeError("{$what}: indptr must be non-decreasing integers.");
            }
            $prev = $p;
        }
        if ($prev !== $nnz) {
            throw new ShapeError("{$what}: indptr must end at the number of stored values ({$nnz}).");
        }
        if ($indices !== null) {
            foreach ($indices as $ix) {
                if (! is_int($ix) || $ix < 0 || $ix >= $minor) {
                    throw new ShapeError("{$what}: sparse index outside the matrix.");
                }
            }
        }
    }
}
