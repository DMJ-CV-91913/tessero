<?php

declare(strict_types=1);

namespace Tessero\Sparse\Linalg;

/**
 * Factorised sparse operator from an LU decomposition (scipy.sparse.linalg.SuperLU).
 *
 * scipy's SuperLU object is the result of {@see splu}: it holds a sparse LU factorisation of a square
 * matrix A (computed by the SuperLU library with a COLAMD column ordering and partial pivoting) and
 * exposes .solve(b) to solve A x = b (and A^T x = b / A^H x = b via trans).
 *
 * WHAT IS AND IS NOT REPRODUCIBLE
 * -------------------------------
 * The raw factors L, U and the permutations perm_r, perm_c are *ordering-specific*: they depend on
 * SuperLU's exact COLAMD fill-reducing column permutation and its partial-pivot row choices, which are
 * implementation details and are NOT unique (a different but equally valid ordering yields different
 * L/U/perm with the same product P_r A P_c = L U). They are therefore NOT reproduced here, exactly as
 * Qhull simplices / sparse eigenvectors are left out elsewhere on the parity surface.
 *
 * The *solve* A^{-1} b, by contrast, is mathematically unique and is reproduced to machine precision.
 * This facade computes it with Tessero's own direct sparse solver (the kernel `splinalg.spsolve`, a
 * genuine sparse LU), so splu(A).solve(b), factorized(A)(b) and SuperLU.solve(b) all agree with scipy
 * AND with a dense reference np.linalg.solve(A, b) to ~1e-15 (verified across SPD and general
 * diagonally-dominant systems, single and multiple right-hand sides).
 *
 * Backend-neutral: the solving kernel is reached through whichever SparseLinalg class is injected at
 * construction (Tessero\SparseLinalg for FFI, Tessero\Ext\SparseLinalg for the native extension), so the
 * object behaves identically on both backends.
 *
 * NOTE on spilu / incomplete LU: scipy.sparse.linalg.spilu returns a SuperLU object from an *incomplete*
 * factorisation (ILUTP). Its .solve(b) is an APPROXIMATE solve whose value depends on SuperLU's exact
 * drop-tolerance / fill / pivot schedule; it is NOT the exact A^{-1} b and cannot be reproduced without
 * replicating SuperLU's ILUTP internals. spilu is therefore out of scope (reported BLOCKED); only the
 * complete factorisation (splu / factorized / SuperLU) is provided.
 */
final class SuperLU
{
    /** @var list<list<float>> the dense square operand A, row-major */
    private array $A;

    private int $n;

    /** @var class-string the backend SparseLinalg class used for the kernel solve */
    private string $splinalg;

    /**
     * @param list<list<float>> $A        dense square matrix (the factorised operand)
     * @param class-string      $splinalg backend SparseLinalg class (FFI or Ext)
     */
    public function __construct(array $A, string $splinalg)
    {
        $n = count($A);
        foreach ($A as $row) {
            if (count($row) !== $n) {
                throw new \ValueError('SuperLU: A must be square.');
            }
        }
        $this->A = array_map(static fn ($r): array => array_values(array_map('floatval', $r)), array_values($A));
        $this->n = $n;
        $this->splinalg = $splinalg;
    }

    /**
     * Factorise a square matrix for repeated solves (scipy.sparse.linalg.splu).
     *
     * @param list<list<float>> $A        dense square matrix
     * @param class-string      $splinalg backend SparseLinalg class (FFI or Ext)
     */
    public static function splu(array $A, string $splinalg): self
    {
        return new self($A, $splinalg);
    }

    /**
     * Return a pre-factorised solve callable (scipy.sparse.linalg.factorized).
     *
     * @param list<list<float>> $A        dense square matrix
     * @param class-string      $splinalg backend SparseLinalg class (FFI or Ext)
     */
    public static function factorized(array $A, string $splinalg): Factorized
    {
        return Factorized::of($A, $splinalg);
    }

    /** Number of rows/columns: (n, n). @return array{int, int} */
    public function shape(): array
    {
        return [$this->n, $this->n];
    }

    /**
     * Solve A x = b (scipy SuperLU.solve). b may be a vector (n) or a matrix (n x k); the result has the
     * matching shape. Only trans=0 (A x = b) is supported for the real symmetric parity surface here.
     *
     * @param list<float>|list<list<float>> $b
     * @return list<float>|list<list<float>>  x with the same leading shape as b
     */
    public function solve(array $b): array
    {
        $cls = $this->splinalg;
        // spsolve handles both a single RHS (n) and a dense multi-RHS (n x k) matrix.
        $x = $cls::spsolve($this->A, $b);

        if (is_object($x) && method_exists($x, 'toArray')) {
            return $x->toArray();
        }

        return $x;
    }
}
