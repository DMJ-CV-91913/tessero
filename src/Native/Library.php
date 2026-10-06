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
 * Pointers into native memory are created with ptr($address). Pointer
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

    public static function locate(): ?string
    {
        $env = getenv('TESSERO_LIB');
        if (is_string($env) && $env !== '' && is_file($env)) {
            return $env;
        }
        $file = match (PHP_OS_FAMILY) {
            'Darwin' => 'libtessero.dylib',
            'Windows' => 'tessero.dll',
            default => 'libtessero.so',
        };
        $candidate = dirname(__DIR__, 2) . '/lib/' . self::platform() . '/' . $file;

        return is_file($candidate) ? $candidate : null;
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
        $v = (string) $ffi->tsr_version();
        if ($v !== \Tessero\Tessero::VERSION) {
            throw new LibraryUnavailable("libtessero {$v} ({$what}) does not match tessero/tessero " . \Tessero\Tessero::VERSION
                . '. Use the binary shipped with the package, or rebuild it with `make -C csrc`.');
        }
    }

    /** char* at an absolute address. The caller keeps the owning Buffer alive. */
    public static function ptr(int $address): CData
    {
        return self::ffi()->cast('char*', $address);
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
