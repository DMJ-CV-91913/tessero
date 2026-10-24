<?php

declare(strict_types=1);

namespace Tessero\Native;

use FFI;
use FFI\CData;
use Tessero\Exceptions\DTypeError;
use Tessero\Exceptions\IndexError;
use Tessero\Exceptions\LibraryUnavailable;
use Tessero\Exceptions\MemoryError;
use Tessero\Exceptions\ShapeError;
use Tessero\Exceptions\TesseroException;

/**
 * Loads libtessero once per process.
 *
 * Resolution order:
 *   1. FFI::scope('tessero')      - the library was bound by opcache.preload
 *                                   (required under PHP-FPM/Apache with ffi.enable=preload);
 *   2. FFI::cdef(header, path)    - CLI, or any SAPI with ffi.enable=1.
 * The shared object is looked up in $TESSERO_LIB, then the prebuilt binary
 * shipped for this platform in lib/<os>-<arch>/.
 *
 * Pointers into native memory are created by Buffer::ptr(). Pointer
 * arithmetic on CData (`$p + $n`) is deliberately never used: on PHP 8.4
 * passing such a temporary to a function silently breaks later arithmetic
 * on the source pointer (see docs/project/upstream-bugs.md, issue 2).
 */
final class Library
{
    public const SCOPE = 'tessero';

    private static ?FFI $ffi = null;

    private static ?string $path = null;

    private static string $mode = 'unloaded';

    /**
     * The bound library. Internal: it reaches raw native memory, so application code must not use it.
     * Under ffi.enable=preload it is a preloaded method, so any request code that calls it gets full FFI
     * access; PHP cannot restrict that (Reflection reaches private members too), see docs/operations/security.md.
     *
     * @internal
     */
    public static function ffi(): FFI
    {
        return self::$ffi ?? self::load();
    }

    public static function available(): bool
    {
        try {
            self::ffi();

            return true;
        } catch (LibraryUnavailable) {
            return false;
        }
    }

    /** 'scope' (preloaded), 'cdef' or 'unloaded'. */
    public static function mode(): string
    {
        return self::$mode;
    }

    public static function path(): ?string
    {
        return self::$path ?? self::locate();
    }

    public static function header(): string
    {
        $raw = file_get_contents(self::headerPath());
        if ($raw === false) {
            throw new LibraryUnavailable('tessero.h is missing from the package.');
        }

        return (string) preg_replace('#/\*.*?\*/#s', '', $raw);
    }

    public static function headerPath(): string
    {
        return dirname(__DIR__, 2) . '/csrc/include/tessero.h';
    }

    public static function platform(): string
    {
        $os = match (PHP_OS_FAMILY) {
            'Darwin' => 'darwin',
            'Windows' => 'windows',
            default => 'linux',
        };
        $arch = strtolower(php_uname('m'));
        $arch = match ($arch) {
            'amd64', 'x64' => 'x86_64',
            'arm64' => PHP_OS_FAMILY === 'Darwin' ? 'arm64' : 'aarch64',
            default => $arch,
        };
        if ($os === 'linux' && self::isMusl()) {
            $os = 'linux-musl';
        }

        return "{$os}-{$arch}";
    }

    /**
     * The library to load: $TESSERO_LIB when set (read from the process environment only, never from
     * SAPI/request variables; it must be an absolute path to a file that neither it nor its directory is
     * group- or world-writable), else the bundled binary.
     *
     * @throws LibraryUnavailable when TESSERO_LIB is set but not acceptable
     */
    public static function locate(): ?string
    {
        $env = getenv('TESSERO_LIB', true);
        if (is_string($env) && $env !== '') {
            return self::trustedFile($env, 'TESSERO_LIB');
        }
        $file = match (PHP_OS_FAMILY) {
            'Darwin' => 'libtessero.dylib',
            'Windows' => 'tessero.dll',
            default => 'libtessero.so',
        };
        $candidate = dirname(__DIR__, 2) . '/lib/' . self::platform() . '/' . $file;

        return is_file($candidate) ? $candidate : null;
    }

    /**
     * A shared library named by an environment variable, checked before anything dlopen()s it: an absolute
     * path (a relative one resolves against the working directory, the web root under php-cgi), an existing
     * file, and on POSIX neither the file nor its directory writable by group or others (fileperms & 0022),
     * so no other account can swap the code PHP is about to run. Returns the resolved path.
     *
     * @internal
     *
     * @throws LibraryUnavailable
     */
    public static function trustedFile(string $path, string $var): string
    {
        $absolute = PHP_OS_FAMILY === 'Windows'
            ? preg_match('#^([A-Za-z]:[\\\\/]|\\\\\\\\)#', $path) === 1
            : str_starts_with($path, '/');
        if (! $absolute || str_contains($path, "\0") || str_contains($path, '://')) {
            throw new LibraryUnavailable("{$var} must be an absolute path to a shared library, got '{$path}'.");
        }
        // a permission check must not use a cached stat (before PHP 8.4, chmod() leaves it stale)
        clearstatcache();
        $real = @realpath($path);
        if ($real === false || ! is_file($real)) {
            throw new LibraryUnavailable("{$var} points to '{$path}', which is not a readable file.");
        }
        if (PHP_OS_FAMILY !== 'Windows') {
            foreach ([$real, dirname($real)] as $p) {
                $perms = @fileperms($p);
                if ($perms === false || ($perms & 0o022) !== 0) {
                    throw new LibraryUnavailable(sprintf(
                        "%s: refusing to load '%s': %s is group- or world-writable (mode %o). Run chmod go-w on it.",
                        $var, $real, $p === $real ? 'the file' : "its directory '{$p}'", $perms === false ? 0 : $perms & 0o7777,
                    ));
                }
            }
        }

        return $real;
    }

    /**
     * The native memory budget a non-CLI process starts with, or null to leave the kernel's setting alone.
     * Native buffers do not count toward memory_limit, so outside the CLI the FFI package caps them by
     * default at max(256 MiB, memory_limit), as the extension does. TESSERO_MEMORY_BUDGET (bytes, process
     * environment) overrides it; 0 or a negative value opts out (unlimited). memory_limit = -1 also means
     * unlimited. Tessero::setMemoryBudget() changes the budget at any time; under FPM the default is
     * re-applied when a request loads the library while the budget is 0.
     *
     * @internal exposed for tests
     */
    public static function defaultBudget(string $sapi, string|false $env, string $memoryLimit, int $current): ?int
    {
        if ($sapi === 'cli' || $sapi === 'phpdbg' || $sapi === 'embed') {
            return null;
        }
        if (is_string($env) && is_numeric(trim($env))) {
            $v = (int) trim($env);

            return $v > 0 ? $v : 0;
        }
        if ($current !== 0) {
            return null;
        }
        $limit = function_exists('ini_parse_quantity') ? @ini_parse_quantity($memoryLimit) : (int) $memoryLimit;
        if ($limit <= 0) {
            return null;
        }

        return max(256 << 20, $limit);
    }

    private static function applyDefaultBudget(FFI $ffi): void
    {
        $budget = self::defaultBudget(PHP_SAPI, getenv('TESSERO_MEMORY_BUDGET', true), (string) ini_get('memory_limit'), $ffi->tsr_budget());
        if ($budget !== null) {
            $ffi->tsr_set_budget($budget);
        }
        // A non-canonical FP environment (flush-to-zero, directed rounding, unmasked traps: e.g. set by another
        // library linked with -ffast-math) silently changes results: warn once, never fail the load. Calling thread
        // only: an OpenMP region here, before php-fpm forks its workers, would break libgomp in the children.
        $buf = $ffi->new('char[256]');
        if ($ffi->tsr_fpenv_check_thread($buf, 256) !== 0) {
            trigger_error('Tessero: non-canonical floating-point environment: ' . \FFI::string($buf), E_USER_WARNING);
        }
        // The BLAS libtessero links (used by matmul) gets the same web-SAPI default as the extension's
        // tessero.blas_threads=auto: one thread per worker, since php-fpm already runs a worker per core.
        $threads = self::defaultBlasThreads(PHP_SAPI, getenv('TESSERO_BLAS_THREADS', true));
        if ($threads !== null) {
            $ffi->tsr_set_blas_threads($threads);
        }
    }

    /**
     * The BLAS thread cap to apply at load, or null to leave the BLAS alone: TESSERO_BLAS_THREADS=n (n >= 1)
     * wins, 0 means leave alone, and otherwise web SAPIs get 1 and the CLI keeps the BLAS default.
     */
    public static function defaultBlasThreads(string $sapi, string|false $env): ?int
    {
        if ($env !== false && $env !== '' && preg_match('/^\d+$/', $env) === 1) {
            $n = (int) $env;

            return $n >= 1 ? $n : null;
        }

        return in_array($sapi, ['cli', 'phpdbg', 'embed'], true) ? null : 1;
    }

    private static function load(): FFI
    {
        if (! extension_loaded('ffi')) {
            throw new LibraryUnavailable('ext-ffi is not loaded.');
        }
        try {
            $scoped = FFI::scope(self::SCOPE);
        } catch (\Throwable) {
            $scoped = null;                                    // not preloaded; fall through to cdef
        }
        if ($scoped !== null) {
            self::checkVersion($scoped, 'the preloaded library');
            self::applyDefaultBudget($scoped);
            self::$ffi = $scoped;
            self::$mode = 'scope';
            self::$path = 'preloaded';

            return self::$ffi;
        }
        $enable = strtolower((string) ini_get('ffi.enable'));
        if (! in_array($enable, ['1', 'on', 'true', 'yes'], true) && ! ($enable === 'preload' && PHP_SAPI === 'cli')) {
            throw new LibraryUnavailable(
                'ffi.enable=' . $enable . ' in the ' . PHP_SAPI . ' SAPI and libtessero was not preloaded. '
                . 'Add vendor/tessero/tessero/resources/preload.php to opcache.preload (see docs/deployment.md).'
            );
        }
        $path = self::locate();
        if ($path === null) {
            throw new LibraryUnavailable(
                'No prebuilt libtessero for ' . self::platform() . '. Build it with `make -C vendor/tessero/tessero/csrc` or set TESSERO_LIB.'
            );
        }
        try {
            $ffi = FFI::cdef(self::header(), $path);
        } catch (\FFI\Exception $e) {
            // typically a symbol the header declares but the binary lacks: an older or foreign build
            throw new LibraryUnavailable("Could not load {$path}: " . $e->getMessage()
                . ' (is it a libtessero build of this version? Rebuild with `make -C csrc` or fix TESSERO_LIB.)', 0, $e);
        }
        self::checkVersion($ffi, $path);
        self::applyDefaultBudget($ffi);
        self::$ffi = $ffi;
        self::$mode = 'cdef';
        self::$path = $path;

        return self::$ffi;
    }

    /**
     * The binary must be the version the PHP code was written for: struct layouts and constants are
     * shared through tessero.h, and a mismatch would corrupt memory rather than fail.
     */
    private static function checkVersion(FFI $ffi, string $what): void
    {
        $abi = $ffi->tsr_abi_version();
        if ($abi !== \Tessero\Tessero::ABI_VERSION) {
            throw new LibraryUnavailable("libtessero ({$what}) implements C ABI {$abi}, tessero/tessero needs ABI " . \Tessero\Tessero::ABI_VERSION
                . '. Use the binary shipped with the package, or rebuild it with `make -C csrc`.');
        }
        $v = (string) $ffi->tsr_version();
        if ($v !== \Tessero\Tessero::VERSION) {
            throw new LibraryUnavailable("libtessero {$v} ({$what}) does not match tessero/tessero " . \Tessero\Tessero::VERSION
                . '. Use the binary shipped with the package, or rebuild it with `make -C csrc`.');
        }
    }

    /** Integer address of any pointer CData (void* is routed through char*: a direct void*->uintptr_t cast yields 0 on PHP 8.4). */
    public static function address(CData $pointer): int
    {
        $ffi = self::ffi();

        return $ffi->cast('uintptr_t', $ffi->cast('char*', $pointer))->cdata;
    }

    /** @param list<int> $values */
    public static function i64(array $values): CData
    {
        $n = max(1, count($values));
        $arr = self::ffi()->new("int64_t[{$n}]");
        foreach ($values as $i => $v) {
            $arr[$i] = $v;
        }

        return $arr;
    }

    /** Map a negative libtessero return code to an exception. */
    public static function check(int $rc, string $what): int
    {
        if ($rc >= 0) {
            return $rc;
        }

        throw match ($rc) {
            -1 => new TesseroException("{$what}: invalid argument."),
            -2 => new IndexError("{$what}: index out of bounds."),
            -3 => new MemoryError("{$what}: out of memory or over the Tessero memory budget."),
            -4 => new DTypeError("{$what}: not supported for this dtype."),
            -5 => new ShapeError("{$what}: more than 32 dimensions."),
            -6 => new ShapeError("{$what}: shapes are not compatible."),
            -7 => new TesseroException("{$what}: the array is not contiguous; copy it first."),
            -8 => new \Tessero\Exceptions\ConvergenceError("{$what}: the iterative method did not converge (singular system)."),
            -9 => new TesseroException("{$what}: I/O error."),
            default => new TesseroException("{$what}: error {$rc}."),
        };
    }

    private static function isMusl(): bool
    {
        static $musl = null;
        if ($musl === null) {
            $musl = @is_file('/etc/alpine-release') || (@glob('/lib/ld-musl-*') ?: []) !== [];
        }

        return $musl;
    }
}
