<?php

/**
 * Validate an incoming numeric array by shape and finiteness with the Laravel rule.
 * In a Laravel app:
 *   $request->validate(['matrix' => new Tessero\Laravel\Rules\NumericArray(ndim: 2, shape: [2, 3])]);
 *
 * This exercises the rule's static checker directly (pure PHP, both backends). The cast
 * (AsNDArray model casting), queued-job and Octane-safe-settings patterns need a running
 * Laravel app and are shown in the Laravel guide (docs/guide/laravel.md).
 */

declare(strict_types=1);

require_once __DIR__ . '/../../laravel/tests/stubs.php';   // minimal Illuminate interface stubs

use Tessero\Laravel\Rules\NumericArray;

// A valid 2x3 numeric matrix passes: check() returns null (no error).
check(NumericArray::check([[1, 2, 3], [4, 5, 6]], ndim: 2, shape: [2, 3]) === null, 'valid 2x3 passes');

// Wrong column count is rejected with a helpful message.
$err = NumericArray::check([[1, 2], [3, 4]], ndim: 2, shape: [2, 3]);
check(is_string($err) && str_contains($err, 'length 3'), 'wrong column count rejected');

// Non-numeric, ragged, and non-finite inputs are rejected.
check(NumericArray::check([[1, 'x', 3]], ndim: 2) !== null, 'non-numeric rejected');
check(NumericArray::check([[1, 2, 3], [4, 5]]) !== null, 'ragged rows rejected');
check(NumericArray::check([[1.0, INF, 3.0]], ndim: 2) !== null, 'non-finite rejected');

say('NumericArray validation ok on ' . backend());
