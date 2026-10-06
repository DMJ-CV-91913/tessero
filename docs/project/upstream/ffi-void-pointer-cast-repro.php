<?php
// PHP FFI: FFI::cast('uintptr_t', <void* CData>) yields 0; the same pointer cast to char* first gives the address.
// Run: php -d ffi.enable=1 ffi-void-pointer-cast-repro.php
$ffi = FFI::cdef('void *malloc(size_t n); void free(void *p);', PHP_OS_FAMILY === 'Darwin' ? 'libSystem.B.dylib' : 'libc.so.6');
$raw = $ffi->malloc(64);                                  // CData of type void*
$direct = $ffi->cast('uintptr_t', $raw)->cdata;
$viaChar = $ffi->cast('uintptr_t', $ffi->cast('char *', $raw))->cdata;
printf("void* -> uintptr_t:          %d\n", $direct);
printf("void* -> char* -> uintptr_t: %d\n", $viaChar);
echo $direct === $viaChar ? "consistent\n" : "INCONSISTENT (expected equal, non-zero addresses)\n";
$ffi->free($raw);
