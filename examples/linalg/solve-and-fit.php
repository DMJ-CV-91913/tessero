<?php

/**
 * Dense linear algebra: solve a square system, and least-squares fit an overdetermined one.
 * NumPy: np.linalg.solve(A, b); np.linalg.lstsq(A, y, rcond=None)
 *
 * FFI-only today (the extension has no Linalg yet; see TSR-103), so this skips under ext.
 */

declare(strict_types=1);

use Tessero\Linalg\Linalg;

ffi_only('numpy.linalg');

// Exact 2x2 solve: 3x + 2y = 12, x + 2y = 8  ->  (x, y) = (2, 3).
$x = Linalg::solve([[3.0, 2.0], [1.0, 2.0]], [12.0, 8.0]);
check_close($x->toList(), [2.0, 3.0], 'solve 2x2', 1e-9, 1e-9);

// Least-squares line fit y = a + b*x to points exactly on y = 1 + 2x.
$design = [[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]];   // columns: intercept, x
$y = [1.0, 3.0, 5.0, 7.0];
[$coef, $residuals, $rank, $sv] = Linalg::lstsq($design, $y);
check_close($coef, [1.0, 2.0], 'lstsq recovers intercept 1, slope 2', 1e-9, 1e-9);
check($rank === 2, 'full column rank');

say('solve + lstsq ok on ' . backend());
