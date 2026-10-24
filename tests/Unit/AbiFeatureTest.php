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
