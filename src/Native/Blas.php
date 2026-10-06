<?php

declare(strict_types=1);

namespace Tessero\Native;

use FFI;
use FFI\CData;
use Tessero\DType;
use Tessero\Exceptions\LibraryUnavailable;

/**
 * Optional BLAS + LAPACKE binding (OpenBLAS, LP64 / 32-bit integers).
 *
 * Tried in order: $TESSERO_BLAS, the system OpenBLAS, and the OpenBLAS that
 * SciPy's wheels bundle (symbols prefixed "scipy_"; `pip install scipy` is
 * the no-root way to get a tuned BLAS). ILP64 builds (libopenblas64_,
 * numpy's bundled copy) are refused: their integers are 64-bit and passing
 * 32-bit ints would corrupt memory.
 *
 * Without a BLAS, matmul falls back to libtessero's blocked kernel and
 * Linalg throws LibraryUnavailable.
 */
final class Blas
{
    public const ROW_MAJOR = 101;
    public const NO_TRANS = 111;
    public const TRANS = 112;

    private const CDEF = <<<'C'
        void {P}cblas_dgemm(int order, int ta, int tb, int m, int n, int k, double alpha, const double *a, int lda,
                            const double *b, int ldb, double beta, double *c, int ldc);
        void {P}cblas_sgemm(int order, int ta, int tb, int m, int n, int k, float alpha, const float *a, int lda,
                            const float *b, int ldb, float beta, float *c, int ldc);
        void {P}openblas_set_num_threads(int n);
        char *{P}openblas_get_config(void);
        int {P}LAPACKE_dgesv(int layout, int n, int nrhs, double *a, int lda, int *ipiv, double *b, int ldb);
        int {P}LAPACKE_dgetrf(int layout, int m, int n, double *a, int lda, int *ipiv);
        int {P}LAPACKE_dgetri(int layout, int n, double *a, int lda, const int *ipiv);
        int {P}LAPACKE_dpotrf(int layout, char uplo, int n, double *a, int lda);
        int {P}LAPACKE_dgeqrf(int layout, int m, int n, double *a, int lda, double *tau);
        int {P}LAPACKE_dorgqr(int layout, int m, int n, int k, double *a, int lda, const double *tau);
        int {P}LAPACKE_dsyevd(int layout, char jobz, char uplo, int n, double *a, int lda, double *w);
        int {P}LAPACKE_dgeev(int layout, char jobvl, char jobvr, int n, double *a, int lda, double *wr, double *wi,
                             double *vl, int ldvl, double *vr, int ldvr);
        int {P}LAPACKE_dgesdd(int layout, char jobz, int m, int n, double *a, int lda, double *s, double *u, int ldu,
                              double *vt, int ldvt);
        int {P}LAPACKE_dgelsd(int layout, int m, int n, int nrhs, double *a, int lda, double *b, int ldb, double *s,
                              double rcond, int *rank);
        int {P}LAPACKE_dtrtrs(int layout, char uplo, char trans, char diag, int n, int nrhs, const double *a, int lda,
                              double *b, int ldb);
        C;

    private const PREFIXES = ['', 'scipy_'];

    private static ?FFI $ffi = null;

    private static string $prefix = '';

    private static ?string $library = null;

    private static bool $tried = false;

    public static function available(): bool
    {
        return self::load() !== null;
    }

    public static function library(): ?string
    {
        self::load();

        return self::$library;
    }

    public static function ffi(): FFI
    {
        return self::load() ?? throw new LibraryUnavailable(
            'No LP64 OpenBLAS with LAPACKE was found. Install libopenblas (apt install libopenblas0 liblapacke) '
            . 'or `pip install scipy`, or set TESSERO_BLAS to the shared library.'
        );
    }

    /** Call a BLAS/LAPACKE symbol by its unprefixed name. */
    public static function call(string $fn, mixed ...$args): mixed
    {
        return self::ffi()->{self::$prefix . $fn}(...$args);
    }

    public static function setThreads(int $n): void
    {
        if (self::available()) {
            self::call('openblas_set_num_threads', max(1, $n));
        }
    }

    /** C = A (m x k) @ B (k x n), all row-major and contiguous. */
    public static function gemm(DType $dt, int $m, int $n, int $k, CData $a, CData $b, CData $c): void
    {
        $ffi = self::ffi();
        $base = Library::ffi();
        if ($dt === DType::Float64) {
            $ffi->{self::$prefix . 'cblas_dgemm'}(self::ROW_MAJOR, self::NO_TRANS, self::NO_TRANS, $m, $n, $k, 1.0,
                $ffi->cast('double*', $base->cast('uintptr_t', $a)->cdata), $k,
                $ffi->cast('double*', $base->cast('uintptr_t', $b)->cdata), $n, 0.0,
                $ffi->cast('double*', $base->cast('uintptr_t', $c)->cdata), $n);
        } else {
            $ffi->{self::$prefix . 'cblas_sgemm'}(self::ROW_MAJOR, self::NO_TRANS, self::NO_TRANS, $m, $n, $k, 1.0,
                $ffi->cast('float*', $base->cast('uintptr_t', $a)->cdata), $k,
                $ffi->cast('float*', $base->cast('uintptr_t', $b)->cdata), $n, 0.0,
                $ffi->cast('float*', $base->cast('uintptr_t', $c)->cdata), $n);
        }
    }

    /** Pointer of the given C type in the BLAS FFI scope at an absolute address. */
    public static function ptr(string $type, int $address): CData
    {
        return self::ffi()->cast($type, $address);
    }

    private static function load(): ?FFI
    {
        if (self::$tried) {
            return self::$ffi;
        }
        self::$tried = true;
        if (! extension_loaded('ffi')) {
            return null;
        }
        // bound by resources/preload.php (the only way under ffi.enable=preload in FPM)
        try {
            $scoped = FFI::scope('tessero_blas');
            foreach (self::PREFIXES as $prefix) {
                try {
                    $scoped->{$prefix . 'LAPACKE_dgesv'};
                } catch (\Throwable) {
                    continue;
                }
                self::$ffi = $scoped;
                self::$prefix = $prefix;
                self::$library = 'preloaded';

                return $scoped;
            }
        } catch (\Throwable) {
            // not preloaded
        }
        $enable = strtolower((string) ini_get('ffi.enable'));
        if (! in_array($enable, ['1', 'on', 'true', 'yes'], true) && ! ($enable === 'preload' && PHP_SAPI === 'cli')) {
            return null;
        }
        foreach (self::candidates() as $path) {
            foreach (self::PREFIXES as $prefix) {
                try {
                    $ffi = FFI::cdef(str_replace('{P}', $prefix, self::CDEF), $path);
                    // resolve a LAPACKE symbol eagerly: a BLAS-only library must not be picked
                    $ffi->{$prefix . 'LAPACKE_dgesv'};
                } catch (\Throwable) {
                    continue;
                }
                self::$ffi = $ffi;
                self::$prefix = $prefix;
                self::$library = $path;
                $threads = getenv('TESSERO_BLAS_THREADS');
                if ($threads !== false && $threads !== '') {
                    $ffi->{$prefix . 'openblas_set_num_threads'}((int) $threads);
                } elseif (PHP_SAPI !== 'cli') {
                    // FPM/Apache already run one PHP process per core
                    $ffi->{$prefix . 'openblas_set_num_threads'}(1);
                }

                return $ffi;
            }
        }

        return null;
    }

    /**
     * [path, prefix, header] triples for resources/preload.php.
     *
     * @return list<array{0: string, 1: string, 2: string}>
     */
    public static function preloadCandidates(): array
    {
        $out = [];
        foreach (self::candidates() as $path) {
            $resolved = str_contains($path, '/') ? $path : self::resolveSoname($path);
            if ($resolved === null) {
                continue;
            }
            foreach (self::PREFIXES as $prefix) {
                $out[] = [$resolved, $prefix, str_replace('{P}', $prefix, self::CDEF)];
            }
        }

        return $out;
    }

    private static function resolveSoname(string $name): ?string
    {
        foreach (['/usr/lib/x86_64-linux-gnu', '/usr/lib/aarch64-linux-gnu', '/usr/lib64', '/usr/lib', '/usr/local/lib'] as $dir) {
            if (is_file("{$dir}/{$name}")) {
                return "{$dir}/{$name}";
            }
        }

        return null;
    }

    /** @return list<string> */
    private static function candidates(): array
    {
        $out = [];
        $env = getenv('TESSERO_BLAS');
        if (is_string($env) && $env !== '') {
            $out[] = $env;
        }
        array_push($out, 'libopenblas.so.0', 'libopenblas.so', 'liblapacke.so.3', '/opt/homebrew/opt/openblas/lib/libopenblas.dylib',
            '/usr/local/opt/openblas/lib/libopenblas.dylib', 'libopenblas.dll');
        $home = getenv('HOME') ?: '/root';
        foreach ([
            '/usr/local/lib/python3*/dist-packages/scipy.libs/libscipy_openblas-*.so',
            '/usr/lib/python3*/site-packages/scipy.libs/libscipy_openblas-*.so',
            '/usr/local/lib/python3*/site-packages/scipy.libs/libscipy_openblas-*.so',
            $home . '/.local/lib/python3*/site-packages/scipy.libs/libscipy_openblas-*.so',
        ] as $pattern) {
            foreach (glob($pattern) ?: [] as $match) {
                if (! str_contains(basename($match), 'openblas64')) {
                    $out[] = $match;
                }
            }
        }

        return array_values(array_unique($out));
    }
}
