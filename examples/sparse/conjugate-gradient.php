<?php

/**
 * Solve a sparse symmetric positive-definite system with conjugate gradient.
 * SciPy: scipy.sparse.linalg.cg(A, b)  ->  (x, info)   (info == 0 means converged)
 *
 * FFI-only today (the extension has no sparse matrices yet), so this skips under ext.
 */

declare(strict_types=1);

use Tessero\Sparse\CsrMatrix;

ffi_only('scipy.sparse');

// A is SPD; store it as CSR. Solve A x = b.
$A = CsrMatrix::fromDense([[4.0, 1.0], [1.0, 3.0]]);
[$x, $info] = $A->cg([1.0, 2.0]);

check($info === 0, 'CG reported convergence (info == 0)');
check_close($x, [0.09090909090909094, 0.6363636363636364], 'CG solution', 1e-6, 1e-9);

// Residual check: A x should recover b.
check_close($A->dot(ND::array($x))->toList(), [1.0, 2.0], 'A x == b', 1e-6, 1e-9);

say('sparse CG ok on ' . backend());
