<?php

declare(strict_types=1);

namespace Tessero;

use Tessero\Native\Blas;
use Tessero\Native\Library;
use Tessero\Random\Generator;

/**
 * Process-wide settings and diagnostics.
 */
final class Tessero
{
    public const VERSION = '0.2.0';

    /** The libtessero C ABI generation this package is written against (csrc/include/tessero_version.h, ADR 0014). */
    public const ABI_VERSION = 2;

    /**
     * Cap native memory for this process (0 = unlimited). Allocations over the
     * cap throw MemoryError instead of letting a web worker grow without bound;
     * native buffers do not count toward memory_limit.
     */
    public static function setMemoryBudget(int $bytes): void
    {
        Library::ffi()->tsr_set_budget(max(0, $bytes));
    }

    public static function memoryBudget(): int
    {
        return Library::ffi()->tsr_budget();
    }

    public static function memoryInUse(): int
    {
        return Library::ffi()->tsr_allocated();
    }

    public static function peakMemory(): int
    {
        return Library::ffi()->tsr_peak();
    }

    /**
     * Cap the BLAS thread pools: the BLAS this package loads over FFI (Linalg) and the one libtessero links
     * (matmul). Web SAPIs default to 1 (TESSERO_BLAS_THREADS overrides), since php-fpm runs a worker per core.
     */
    public static function setBlasThreads(int $threads): void
    {
        Blas::setThreads($threads);
        Library::ffi()->tsr_set_blas_threads(max(1, $threads));
    }

    /** The C ABI generation of the loaded libtessero. */
    public static function abiVersion(): int
    {
        return Library::ffi()->tsr_abi_version();
    }

    /**
     * Whether the loaded build has a capability: "repro_f32", "gemm_canonical", "lapack", "openblas_threads",
     * "openmp" or "fpenv_check". Unknown names give false.
     */
    public static function feature(string $name): bool
    {
        return Library::ffi()->tsr_feature($name) === 1;
    }

    /**
     * Threads for large element-wise kernels and MDP sweeps (OpenMP builds).
     * Default 1: PHP-FPM already runs a worker per core. Results are identical
     * for any thread count.
     */
    public static function setThreads(int $threads): void
    {
        Library::ffi()->tsr_set_threads(max(1, $threads));
    }

    public static function threads(): int
    {
        return Library::ffi()->tsr_get_threads();
    }

    public static function rng(int|array|null $seed = null): Generator
    {
        return Generator::defaultRng($seed);
    }

    /** @return array<string, mixed> */
    public static function info(): array
    {
        $ffi = Library::ffi();
        $simd = ['baseline (SSE2)', 'AVX2 (x86-64-v3)', 'AVX-512 (x86-64-v4)', 'NEON'][$ffi->tsr_simd_level()] ?? 'unknown';

        return [
            'tessero' => self::VERSION,
            'libtessero' => (string) $ffi->tsr_version(),
            'abi' => $ffi->tsr_abi_version(),
            'features' => array_values(array_filter(
                ['repro_f32', 'gemm_canonical', 'lapack', 'openblas_threads', 'openmp', 'fpenv_check'],
                static fn (string $f): bool => $ffi->tsr_feature($f) === 1,
            )),
            'blas_threads' => $ffi->tsr_get_blas_threads(),
            'library' => Library::path(),
            'load_mode' => Library::mode(),
            'platform' => Library::platform(),
            'simd' => $simd,
            'blas' => Blas::library(),
            'openmp' => (bool) $ffi->tsr_openmp(),
            'threads' => $ffi->tsr_get_threads(),
            'php' => PHP_VERSION,
            'sapi' => PHP_SAPI,
            'jit' => function_exists('opcache_get_status') ? (opcache_get_status(false)['jit']['on'] ?? false) : false,
            'memory_in_use' => self::memoryInUse(),
            'memory_budget' => self::memoryBudget(),
            'memory_mapped' => $ffi->tsr_mmap_bytes(),     // file-backed, outside the budget
        ];
    }
}
