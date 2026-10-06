<?php

/**
 * Element-wise math (ufuncs), scalar and array broadcasting. Runs on both backends.
 * NumPy: a + 1; np.sqrt(x); a.prod(); np.add(mat, col) broadcasting (3,) over (2, 3)
 *
 * Scenario: turn a series of monthly simple returns into growth factors and compound them,
 * then a little broadcasting. ND/M are the FFI or extension classes depending on the backend.
 */

declare(strict_types=1);

$returns = ND::array([0.01, -0.02, 0.03, 0.00, 0.015]);   // monthly simple returns
$growth = M::add($returns, 1.0);                          // 1 + r, scalar broadcast
check_close($growth->toList(), [1.01, 0.98, 1.03, 1.00, 1.015], '1 + r', 1e-12, 1e-12);

$compounded = $growth->prod();                            // product of all growth factors
check_close($compounded, 1.01 * 0.98 * 1.03 * 1.00 * 1.015, 'compounded growth', 1e-12, 1e-12);
say('compounded growth = ' . round((float) $compounded, 6) . '  on ' . backend());

$x = ND::array([1.0, 4.0, 9.0, 16.0]);
check_close(M::sqrt($x)->toList(), [1.0, 2.0, 3.0, 4.0], 'sqrt', 1e-12, 1e-12);

// Broadcasting: a (3,) vector adds across each row of a (2, 3) matrix.
$mat = ND::array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
$col = ND::array([10.0, 20.0, 30.0]);
check_close(M::add($mat, $col)->toList(), [[11.0, 22.0, 33.0], [14.0, 25.0, 36.0]], 'broadcast add', 1e-12, 1e-12);
