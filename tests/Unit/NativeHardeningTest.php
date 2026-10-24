<?php

declare(strict_types=1);

namespace Tessero\Tests\Unit;

use PHPUnit\Framework\TestCase;
use Tessero\DType;
use Tessero\Exceptions\IndexError;
use Tessero\Exceptions\LibraryUnavailable;
use Tessero\Exceptions\ShapeError;
use Tessero\Exceptions\TesseroException;
use Tessero\NDArray;
use Tessero\Native\Blas;
use Tessero\Native\Buffer;
use Tessero\Native\Library;
use Tessero\Native\Registry;
use Tessero\Sparse\CsrMatrix;
use Tessero\Tessero;

/**
 * Regression tests for the FFI binding hardening of the 2026-10 review: library path checks, the default
 * native memory budget, routine-result ownership in Registry, the 32-bit BLAS integer guard, Krylov return
 * codes and the removed raw-memory factories. Deterministic: fixed inputs, no timing.
 */
final class NativeHardeningTest extends TestCase
{
    private string $dir;

    protected function setUp(): void
    {
        if (! Library::available()) {
            $this->markTestSkipped('libtessero is not available');
        }
        $this->dir = sys_get_temp_dir() . '/tsr_native_' . getmypid();
        @mkdir($this->dir, 0o755);
        @chmod($this->dir, 0o755);
    }

    protected function tearDown(): void
    {
        gc_collect_cycles();
        if (isset($this->dir)) {
            @chmod($this->dir, 0o755);
            foreach (glob($this->dir . '/*') ?: [] as $f) {
                @chmod($f, 0o755);
                is_dir($f) ? @rmdir($f) : @unlink($f);
            }
            @rmdir($this->dir);
        }
    }

    // ------------------------------------------------------------------ 6. library hijacking

    public function testLibraryPathsFromTheEnvironmentAreChecked(): void
    {
        $this->throwsLike(LibraryUnavailable::class, 'absolute', static fn () => Library::trustedFile('libtessero.so', 'TESSERO_LIB'));
        $this->throwsLike(LibraryUnavailable::class, 'absolute', static fn () => Library::trustedFile('./lib/x.so', 'TESSERO_LIB'));
        $this->throwsLike(LibraryUnavailable::class, 'absolute', static fn () => Library::trustedFile('phar:///x.so', 'TESSERO_BLAS'));
        $missing = $this->dir . '/missing.so';
        $this->throwsLike(LibraryUnavailable::class, 'not a readable file', static fn () => Library::trustedFile($missing, 'TESSERO_LIB'));
        if (PHP_OS_FAMILY === 'Windows') {
            return;
        }
        $lib = $this->dir . '/lib.so';
        file_put_contents($lib, "\x7fELF");
        chmod($lib, 0o644);
        $this->assertSame(realpath($lib), Library::trustedFile($lib, 'TESSERO_LIB'));
        chmod($lib, 0o666);
        $this->throwsLike(LibraryUnavailable::class, 'writable', static fn () => Library::trustedFile($lib, 'TESSERO_LIB'));
        chmod($lib, 0o664);
        $this->throwsLike(LibraryUnavailable::class, 'writable', static fn () => Library::trustedFile($lib, 'TESSERO_LIB'));
        chmod($lib, 0o644);
        chmod($this->dir, 0o777);
        $this->throwsLike(LibraryUnavailable::class, 'directory', static fn () => Library::trustedFile($lib, 'TESSERO_BLAS'));
        chmod($this->dir, 0o755);
    }

    public function testEnvironmentOverridesUseTheCheck(): void
    {
        $saved = getenv('TESSERO_LIB', true);
        putenv('TESSERO_LIB=relative/libtessero.so');
        try {
            $this->throwsLike(LibraryUnavailable::class, 'TESSERO_LIB', static fn () => Library::locate());
        } finally {
            putenv($saved === false ? 'TESSERO_LIB' : "TESSERO_LIB={$saved}");
        }

        $savedBlas = getenv('TESSERO_BLAS', true);
        putenv('TESSERO_BLAS=libopenblas.so');
        try {
            $candidates = new \ReflectionMethod(Blas::class, 'candidates');
            $this->throwsLike(LibraryUnavailable::class, 'TESSERO_BLAS', static fn () => $candidates->invoke(null));
        } finally {
            putenv($savedBlas === false ? 'TESSERO_BLAS' : "TESSERO_BLAS={$savedBlas}");
        }
    }

    // ------------------------------------------------------------------ 11. default memory budget

    public function testDefaultBudgetForWebSapis(): void
    {
        $mib = 1 << 20;
        $this->assertNull(Library::defaultBudget('cli', false, '128M', 0), 'the CLI keeps the kernel default');
        $this->assertSame(256 * $mib, Library::defaultBudget('fpm-fcgi', false, '128M', 0), 'at least 256 MiB');
        $this->assertSame(1024 * $mib, Library::defaultBudget('apache2handler', false, '1G', 0), 'memory_limit when larger');
        $this->assertNull(Library::defaultBudget('fpm-fcgi', false, '-1', 0), 'memory_limit = -1 stays unlimited');
        $this->assertNull(Library::defaultBudget('fpm-fcgi', false, '128M', 5 * $mib), 'an explicit budget is kept');
        $this->assertSame(5000, Library::defaultBudget('fpm-fcgi', '5000', '128M', 0), 'TESSERO_MEMORY_BUDGET wins');
        $this->assertSame(0, Library::defaultBudget('fpm-fcgi', '0', '128M', 0), 'TESSERO_MEMORY_BUDGET=0 opts out');
        $this->assertSame(0, Library::defaultBudget('litespeed', '-1', '128M', 0), 'TESSERO_MEMORY_BUDGET=-1 opts out');
        $this->assertSame(256 * $mib, Library::defaultBudget('fpm-fcgi', 'garbage', '64M', 0), 'a non-numeric value is ignored');
    }

    // ------------------------------------------------------------------ 7. Registry result ownership

    public function testMalformedArrayResultsAreFreedExactlyOnce(): void
    {
        $ffi = Library::ffi();
        $resultValue = new \ReflectionMethod(Registry::class, 'resultValue');
        gc_collect_cycles();
        $base = Tessero::memoryInUse();
        foreach ([
            'unknown dtype' => [99, 1, [8], 64],
            'shape past the block' => [DType::Float64->value, 1, [1000], 64],
            'ndim over 32' => [DType::Float64->value, 40, [1], 64],
            'negative dimension' => [DType::Float64->value, 1, [-1], 64],
        ] as $what => [$dtype, $ndim, $shape, $bytes]) {
            $r = self::arrayResult($ffi, $dtype, $ndim, $shape, $bytes);
            $this->throwsLike(TesseroException::class, 'malformed', static fn () => $resultValue->invoke(null, $ffi, $r), $what);
            $this->assertSame($base, Tessero::memoryInUse(), "{$what}: block freed once, not leaked");
        }
        // a valid result is adopted: its Buffer frees it
        $ok = $resultValue->invoke(null, $ffi, self::arrayResult($ffi, DType::Float64->value, 1, [8], 64));
        $this->assertSame([8], $ok->shape());
        unset($ok);
        gc_collect_cycles();
        $this->assertSame($base, Tessero::memoryInUse());
    }

    public function testASequenceWithABadItemFreesEverything(): void
    {
        $ffi = Library::ffi();
        $resultValue = new \ReflectionMethod(Registry::class, 'resultValue');
        gc_collect_cycles();
        $base = Tessero::memoryInUse();
        $size = \FFI::sizeof($ffi->type('tsr_result'));
        $block = $ffi->tsr_alloc(2 * $size);
        $items = $ffi->cast('tsr_result*', $block);
        \FFI::memcpy(\FFI::addr($items[0]), \FFI::addr(self::arrayResult($ffi, DType::Float64->value, 1, [8], 64)), $size);
        \FFI::memcpy(\FFI::addr($items[1]), \FFI::addr(self::arrayResult($ffi, 99, 1, [8], 64)), $size);
        $seq = $ffi->new('tsr_result');
        $seq->kind = 6;
        $seq->arr->shape[0] = 2;
        $seq->arr->data = $block;
        $seq->bytes = 2 * $size;
        // before the fix the bad item's block was freed by its Buffer and again by the catch block
        $this->throwsLike(TesseroException::class, 'malformed', static fn () => $resultValue->invoke(null, $ffi, $seq));
        gc_collect_cycles();
        $this->assertSame($base, Tessero::memoryInUse(), 'item blocks and the sequence block all freed once');

        // release() frees a whole unconsumed sequence (results after a failing one in routine())
        $block = $ffi->tsr_alloc(2 * $size);
        $items = $ffi->cast('tsr_result*', $block);
        \FFI::memcpy(\FFI::addr($items[0]), \FFI::addr(self::arrayResult($ffi, DType::Float64->value, 1, [8], 64)), $size);
        \FFI::memcpy(\FFI::addr($items[1]), \FFI::addr(self::arrayResult($ffi, DType::Int64->value, 1, [2], 64)), $size);
        $seq = $ffi->new('tsr_result');
        $seq->kind = 6;
        $seq->arr->shape[0] = 2;
        $seq->arr->data = $block;
        $seq->bytes = 2 * $size;
        (new \ReflectionMethod(Registry::class, 'release'))->invoke(null, $ffi, $seq);
        $this->assertSame($base, Tessero::memoryInUse());
        $this->assertSame(0, $seq->kind, 'marked released');
    }

    public function testRoutineArgumentErrorsAreClean(): void
    {
        // a later argument that cannot become an array throws after a string argument was marshalled;
        // the string buffer is freed in a finally block (not observable from PHP: it is not tsr_alloc memory)
        $this->throwsLike(\Throwable::class, '', static fn () => Registry::routine('np.histogram', ['auto', [[1.0], [1.0, 2.0]], null, false, null]));
        $this->assertSame(2, count(Registry::routine('np.histogram', [[1.0, 2.0, 3.0], 2, null, false, null])), 'the next call works');
    }

    // ------------------------------------------------------------------ 8. BLAS int32 narrowing

    public function testBlasDimensionsMustFitAnInt(): void
    {
        $p = Library::ffi()->new('char[64]');
        $this->throwsLike(ShapeError::class, '32-bit', static fn () => Blas::gemm(DType::Float64, (1 << 32) + 2, 2, 2, $p, $p, $p));
        $this->throwsLike(ShapeError::class, '32-bit', static fn () => Blas::gemm(DType::Float32, 2, 2, 0x80000000, $p, $p, $p));
        $this->throwsLike(ShapeError::class, '32-bit', static fn () => Blas::call('LAPACKE_dgetrf', Blas::ROW_MAJOR, 0x80000000, 2, $p, 2, $p));
        if (Blas::available()) {
            $a = NDArray::array([[1.0, 2.0], [3.0, 4.0]]);
            $this->assertSame([[7.0, 10.0], [15.0, 22.0]], $a->matmul($a)->toArray(), 'normal sizes still multiply');
        }
    }

    // ------------------------------------------------------------------ 9. kernel return codes

    public function testKrylovErrorCodesAreNotIterationCounts(): void
    {
        // 1-D Laplacian, n = 40: neither solver converges in 3 iterations
        $n = 40;
        $rows = $cols = $vals = [];
        for ($i = 0; $i < $n; $i++) {
            foreach ([$i - 1 => -1.0, $i => 2.0, $i + 1 => -1.0] as $j => $v) {
                if ($j >= 0 && $j < $n) {
                    $rows[] = $i;
                    $cols[] = $j;
                    $vals[] = $v;
                }
            }
        }
        $A = CsrMatrix::fromTriplets($rows, $cols, $vals, [$n, $n]);
        $b = array_fill(0, $n, 1.0);
        // -2 / -3 are also TSR_EINDEX / TSR_ENOMEM: two or three iterations without convergence used to raise
        foreach ([2, 3] as $maxiter) {
            [, $info] = $A->cg($b, null, 1e-12, 0.0, $maxiter);
            $this->assertSame($maxiter, $info, "cg maxiter={$maxiter} reports non-convergence");
            [, $info] = $A->bicgstab($b, null, 1e-14, 0.0, $maxiter);
            $this->assertGreaterThan(0, $info, "bicgstab maxiter={$maxiter} reports non-convergence");
        }
        [$x, $info] = $A->cg($b, null, 1e-10);
        $this->assertSame(0, $info);

        // a structure the kernel rejects is an error, never a convergence count
        $good = CsrMatrix::fromTriplets($rows, $cols, $vals, [$n, $n]);
        $bad = \Closure::bind(static fn (): CsrMatrix => new CsrMatrix($good->indptr(), NDArray::full([count($vals)], 99, DType::Int64), $good->data(), $n, $n), null, CsrMatrix::class)();
        $this->throwsLike(IndexError::class, 'tsr_csr_cg', static fn () => $bad->cg($b));
        $this->throwsLike(IndexError::class, 'tsr_csr_bicgstab', static fn () => $bad->bicgstab($b));

        // x0 is read and written n times by the kernel
        $this->throwsLike(ShapeError::class, 'x0', static fn () => $A->cg($b, [0.0, 0.0]));
    }

    public function testCheckedKernelCallsKeepTheirResults(): void
    {
        $this->assertSame([0.0, 0.25, 0.5, 0.75, 1.0], NDArray::linspace(0.0, 1.0, 5)->toList());
        $a = NDArray::arange(6);
        $this->assertSame([1, 3, 5], $a->filter(NDArray::array([false, true, false, true, false, true]))->toList());
        $this->assertSame([[0.0, 0.0], [0.0, 0.0]], NDArray::zeros([2, 0])->matmul(NDArray::zeros([0, 2]))->toArray(), 'k = 0 fills zeros');
        $this->assertSame('[1,2,3]', NDArray::array([1, 2, 3])->toJson());
    }

    // ------------------------------------------------------------------ 10. raw-memory primitives

    public function testRawMemoryFactoriesAreGone(): void
    {
        $this->assertFalse(method_exists(Buffer::class, 'wrap'), 'Buffer::wrap() wrapped any address');
        $this->assertFalse(method_exists(Buffer::class, 'adopt'), 'Buffer::adopt() took ownership of any address');
        $this->assertFalse(method_exists(Library::class, 'ptr'), 'Library::ptr() made a pointer to any address');
        $this->throwsLike(\LogicException::class, 'internal', static fn () => Buffer::adopter());
    }

    // ------------------------------------------------------------------ helpers

    /** @param list<int> $shape */
    private static function arrayResult(\FFI $ffi, int $dtype, int $ndim, array $shape, int $bytes): \FFI\CData
    {
        $r = $ffi->new('tsr_result');
        $r->kind = 3;
        $r->arr->dtype = $dtype;
        $r->arr->ndim = $ndim;
        foreach ($shape as $d => $len) {
            $r->arr->shape[$d] = $len;
        }
        $r->arr->data = $ffi->tsr_alloc($bytes);
        $r->bytes = $bytes;

        return $r;
    }

    private function throwsLike(string $class, string $needle, callable $f, string $what = ''): void
    {
        try {
            $f();
        } catch (\Throwable $e) {
            $this->assertInstanceOf($class, $e, $what . ': ' . get_class($e) . ' ' . $e->getMessage());
            $this->assertStringContainsStringIgnoringCase($needle, $e->getMessage(), $what);

            return;
        }
        $this->fail("expected {$class} ({$needle}) {$what}");
    }
}
