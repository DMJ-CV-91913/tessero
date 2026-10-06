<?php
// PHP FFI: passing a pointer-arithmetic temporary (`$p + 8`) to a function breaks later
// arithmetic on the source pointer with "Unsupported operand types: FFI\CData + int".
// Run: php -d ffi.enable=1 ffi-pointer-temporary-repro.php
$ffi = FFI::cdef('void *malloc(size_t n); void free(void *p);', PHP_OS_FAMILY === 'Darwin' ? 'libSystem.B.dylib' : 'libc.so.6');
$raw = $ffi->malloc(64);
$p = $ffi->cast('char *', $raw);

FFI::memcpy($p + 8, 'abcdefgh', 8);       // temporary passed straight to a function
echo "1. FFI::memcpy(\$p + 8, ...): ok\n";

try {
    $r = $p + 16;                          // expected: a char* 16 bytes in
    echo "2. \$p + 16 afterwards: ok\n";
} catch (TypeError $e) {
    echo "2. \$p + 16 afterwards: FAILED - ", $e->getMessage(), "\n";
}
$ffi->free($raw);
// Note: if `$x = $p + 8;` (result kept in a variable) runs once before step 1, the failure does not occur.
