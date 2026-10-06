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

    public static function setBlasThreads(int $threads): void
    {
        Blas::setThreads($threads);
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
