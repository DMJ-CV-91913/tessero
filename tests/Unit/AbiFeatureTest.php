<?php

declare(strict_types=1);

namespace Tessero\Tests\Unit;

use PHPUnit\Framework\TestCase;
use Tessero\Native\Library;
use Tessero\Tessero;

/** The C ABI handshake, the capability query and the BLAS thread defaults of the FFI package (ADR 0014). */
final class AbiFeatureTest extends TestCase
{
    public function testTheLoadedLibraryImplementsThePackagesAbi(): void
    {
        $this->assertSame(Tessero::ABI_VERSION, Tessero::abiVersion());
        $this->assertSame(Tessero::ABI_VERSION, Tessero::info()['abi']);
    }

    public function testFeatureQueries(): void
    {
        $this->assertTrue(Tessero::feature('repro_f32'));
        $this->assertTrue(Tessero::feature('gemm_canonical'));
        $this->assertFalse(Tessero::feature('no-such-capability'));
        $this->assertFalse(Tessero::feature(''));
        $info = Tessero::info();
        $this->assertContains('repro_f32', $info['features']);
        if (PHP_OS_FAMILY === 'Linux') {
            $this->assertTrue(Tessero::feature('lapack'));
        }
    }

    public function testSelfcheckOnACanonicalHost(): void
    {
        $r = Tessero::selfcheck();
        $this->assertSame(0, $r['fpenv'], $r['fpenv_reason']);
        $this->assertSame(0, $r['selfcheck']);
        $this->assertTrue($r['canonical']);
        $this->assertTrue(Tessero::feature('fpenv_check'));
    }

    public function testLinkedWithNodeleteOnLinux(): void
    {
        if (PHP_OS_FAMILY !== 'Linux') {
            $this->assertFalse(Tessero::feature('nodelete'));     // ELF-only (ADR 0014)

            return;
        }
        $this->assertTrue(Tessero::feature('nodelete'), 'libtessero must carry DF_1_NODELETE (ADR 0014)');
    }

    public function testBlasThreadDefaults(): void
    {
        $this->assertNull(Library::defaultBlasThreads('cli', false));
        $this->assertNull(Library::defaultBlasThreads('phpdbg', false));
        $this->assertSame(1, Library::defaultBlasThreads('fpm-fcgi', false));
        $this->assertSame(1, Library::defaultBlasThreads('apache2handler', ''));
        $this->assertSame(4, Library::defaultBlasThreads('fpm-fcgi', '4'));
        $this->assertNull(Library::defaultBlasThreads('fpm-fcgi', '0'));
        $this->assertSame(1, Library::defaultBlasThreads('fpm-fcgi', 'many'));   // malformed: keep the safe default
    }

    public function testFftCacheDefaults(): void
    {
        $this->assertNull(Library::defaultFftCache('cli', false));                 // the library default, 256 MiB
        $this->assertNull(Library::defaultFftCache('phpdbg', ''));
        $this->assertSame(32 << 20, Library::defaultFftCache('fpm-fcgi', false));
        $this->assertSame(32 << 20, Library::defaultFftCache('apache2handler', 'lots'));   // malformed: safe default
        $this->assertSame(0, Library::defaultFftCache('fpm-fcgi', '0'));
        $this->assertSame(268435456, Library::defaultFftCache('cli', '268435456'));
    }

    public function testAllocCacheDefaults(): void
    {
        $this->assertNull(Library::defaultAllocCache('cli', false));                 // the library default, 64 MiB
        $this->assertNull(Library::defaultAllocCache('embed', ''));
        $this->assertSame(16 << 20, Library::defaultAllocCache('fpm-fcgi', false));
        $this->assertSame(16 << 20, Library::defaultAllocCache('apache2handler', 'lots'));   // malformed: safe default
        $this->assertSame(0, Library::defaultAllocCache('fpm-fcgi', '0'));
        $this->assertSame(134217728, Library::defaultAllocCache('cli', '134217728'));
    }

    public function testAllocCacheLimitRoundTrip(): void
    {
        $saved = Tessero::allocCacheLimit();
        Tessero::setAllocCacheLimit(0);
        $this->assertSame(0, Tessero::allocCacheLimit());
        $this->assertSame(0, Tessero::allocCacheBytes());
        Tessero::setAllocCacheLimit(-1);
        $this->assertSame(0, Tessero::allocCacheLimit());
        Tessero::setAllocCacheLimit($saved);
        $this->assertSame($saved, Tessero::allocCacheLimit());
    }

    public function testFftCacheLimitRoundTrip(): void
    {
        $saved = Tessero::fftCacheLimit();
        Tessero::setFftCacheLimit(0);
        $this->assertSame(0, Tessero::fftCacheLimit());
        $this->assertSame(0, Tessero::fftCacheBytes());
        Tessero::setFftCacheLimit(-1);
        $this->assertSame(0, Tessero::fftCacheLimit());
        Tessero::setFftCacheLimit($saved);
        $this->assertSame($saved, Tessero::fftCacheLimit());
    }

    public function testSetBlasThreadsReachesTheLinkedBlas(): void
    {
        if (! Tessero::feature('openblas_threads')) {
            $this->markTestSkipped('no controllable BLAS in this process');
        }
        $before = Library::ffi()->tsr_get_blas_threads();
        try {
            Tessero::setBlasThreads(2);
            $this->assertSame(2, Library::ffi()->tsr_get_blas_threads());
        } finally {
            Tessero::setBlasThreads(max(1, $before));
        }
    }
}
