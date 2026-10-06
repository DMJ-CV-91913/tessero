<?php

// Router for `php -S` used by tests/e2e/preload.sh: proves the library works in a
// non-CLI SAPI where ffi.enable=preload forbids FFI::cdef at request time.
declare(strict_types=1);

require __DIR__ . '/../autoload.php';

use Tessero\Linalg\Linalg;
use Tessero\NDArray;
use Tessero\Random\Generator;
use Tessero\Tessero;

header('Content-Type: application/json');
$cdefAllowed = true;
try {
    FFI::cdef('int abs(int);');
} catch (\Throwable) {
    $cdefAllowed = false;
}
$a = Generator::defaultRng(1)->normal(size: [200, 200]);
$x = Linalg::solve($a, NDArray::ones([200]));
echo json_encode([
    'sapi' => PHP_SAPI,
    'ffi_enable' => ini_get('ffi.enable'),
    'cdef_allowed_at_request_time' => $cdefAllowed,
    'info' => Tessero::info(),
    'residual' => Linalg::norm($a->matmul($x)->sub(1.0)),
    'sum' => NDArray::arange(1_000_000.0)->sum(),
]);
