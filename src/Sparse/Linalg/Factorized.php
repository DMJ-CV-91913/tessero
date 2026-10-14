<?php

declare(strict_types=1);

namespace Tessero\Sparse\Linalg;

/**
 * Pre-factorised solve callable (scipy.sparse.linalg.factorized).
 *
 * scipy's factorized(A) returns a function solve(b) -> x that reuses a single LU factorisation of A for
 * repeated right-hand sides. This mirrors that: an invokable object wrapping a {@see SuperLU}, so both
 *
 *     $solve = Factorized::of($A, $splinalgClass);
 *     $x = $solve($b);        // __invoke, like scipy's returned callable
 *     $x = $solve->solve($b); // explicit method, identical result
 *
 * give the unique A^{-1} b to machine precision (see {@see SuperLU} for the reproducibility rationale).
 */
final class Factorized
{
    private function __construct(private readonly SuperLU $lu)
    {
    }

    /**
     * @param list<list<float>> $A        dense square matrix
     * @param class-string      $splinalg backend SparseLinalg class (FFI or Ext)
     */
    public static function of(array $A, string $splinalg): self
    {
        return new self(new SuperLU($A, $splinalg));
    }

    /**
     * Solve A x = b for the pre-factorised A.
     *
     * @param list<float>|list<list<float>> $b
     * @return list<float>|list<list<float>>
     */
    public function __invoke(array $b): array
    {
        return $this->lu->solve($b);
    }

    /**
     * Explicit solve (same as invoking the object).
     *
     * @param list<float>|list<list<float>> $b
     * @return list<float>|list<list<float>>
     */
    public function solve(array $b): array
    {
        return $this->lu->solve($b);
    }
}
