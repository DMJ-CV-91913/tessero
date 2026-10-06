<?php

/**
 * opcache.preload script for PHP-FPM / Apache / FrankenPHP workers.
 *
 *   ; php.ini (production)
 *   ffi.enable = preload
 *   opcache.enable = 1
 *   opcache.preload = /var/www/app/vendor/tessero/tessero/resources/preload.php
 *   opcache.preload_user = www-data
 *
 * With ffi.enable=preload, FFI::cdef() is refused at request time; only
 * libraries bound here (FFI::load with an FFI_SCOPE) are reachable, through
 * FFI::scope(). This binds libtessero as scope "tessero" and, if one is found,
 * an LP64 OpenBLAS/LAPACKE as scope "tessero_blas", then preloads every
 * Tessero class. That last step is required, not an optimisation: in this
 * mode PHP only lets *preloaded* functions call FFI APIs such as new/cast,
 * so the Tessero classes must themselves be preloaded.
 *
 * Environment overrides: TESSERO_LIB, TESSERO_BLAS (absolute paths).
 */

declare(strict_types=1);

(static function (): void {
    $root = dirname(__DIR__);
    require_once $root . '/src/Native/Library.php';
    $tmp = sys_get_temp_dir();

    // --- libtessero -------------------------------------------------------
    $lib = \Tessero\Native\Library::locate();
    if ($lib === null) {
        error_log('[tessero] preload: no libtessero for ' . \Tessero\Native\Library::platform() . '; build it with `make -C csrc`.');
    } else {
        $header = "#define FFI_SCOPE \"tessero\"\n#define FFI_LIB \"" . addslashes(realpath($lib) ?: $lib) . "\"\n"
            . \Tessero\Native\Library::header();
        // The header names the library FFI will dlopen, so it must not be a file another user could
        // have planted in the shared temp directory: create it exclusively (tempnam: O_EXCL, mode 0600),
        // load it, and delete it. The parsed scope lives in the preloaded process; the file is not reused.
        $file = tempnam($tmp, 'tessero-preload-');
        if ($file === false || file_put_contents($file, $header) !== strlen($header)) {
            error_log('[tessero] preload: cannot write a temporary header in ' . $tmp);
        } else {
            try {
                FFI::load($file);
            } finally {
                @unlink($file);
            }
        }
    }

    // --- optional BLAS/LAPACKE -------------------------------------------
    require_once $root . '/src/Native/Blas.php';
    foreach (\Tessero\Native\Blas::preloadCandidates() as [$path, $prefix, $cdef]) {
        $header = "#define FFI_SCOPE \"tessero_blas\"\n#define FFI_LIB \"" . addslashes($path) . "\"\n" . $cdef;
        $file = tempnam($tmp, 'tessero-blas-');                     // exclusive, 0600; see above
        if ($file === false || file_put_contents($file, $header) !== strlen($header)) {
            continue;
        }
        try {
            $ffi = @FFI::load($file);
            if ($ffi !== null) {
                $ffi->{$prefix . 'LAPACKE_dgesv'};
                break;
            }
        } catch (\Throwable) {
            // try the next candidate
        } finally {
            @unlink($file);
        }
    }

    // --- precompile the PHP sources ----------------------------------------
    if (function_exists('opcache_compile_file')) {
        // scandir rather than RecursiveDirectoryIterator: the latter sees nothing on some FUSE/overlay mounts
        $walk = static function (string $dir) use (&$walk): array {
            $files = [];
            foreach (scandir($dir) ?: [] as $name) {
                if ($name === '.' || $name === '..') {
                    continue;
                }
                $path = $dir . '/' . $name;
                if (is_dir($path)) {
                    array_push($files, ...$walk($path));
                } elseif (str_ends_with($name, '.php')) {
                    $files[] = $path;
                }
            }

            return $files;
        };
        // dependencies first so every class can be linked: exceptions, enums, natives, base class, then the rest
        $files = $walk($root . '/src');
        usort($files, static function (string $a, string $b): int {
            $rank = static fn (string $f): int => match (true) {
                str_contains($f, '/Exceptions/TesseroException.php') => 0,
                str_contains($f, '/Exceptions/LinAlgError.php') => 1,
                str_contains($f, '/Exceptions/') => 2,
                str_contains($f, '/Native/') || str_ends_with($f, '/DType.php') => 3,
                str_contains($f, '/Internal/') => 4,
                default => 5,
            };

            return [$rank($a), $a] <=> [$rank($b), $b];
        });
        $already = array_flip(array_map(static fn (string $f): string => realpath($f) ?: $f, get_included_files()));
        foreach ($files as $file) {
            if (isset($already[realpath($file) ?: $file])) {
                continue;
            }
            if (str_ends_with($file, '/functions.php')) {
                require_once $file;
                continue;
            }
            // OperandBase declares its class conditionally (ext-tessero loaded or not): include it for real
            if (str_ends_with($file, '/Internal/OperandBase.php')) {
                require_once $file;
                continue;
            }
            @opcache_compile_file($file);
        }
    }
})();
