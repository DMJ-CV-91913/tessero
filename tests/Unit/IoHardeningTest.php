<?php

declare(strict_types=1);

namespace Tessero\Tests\Unit;

use PHPUnit\Framework\TestCase;
use Tessero\Exceptions\ShapeError;
use Tessero\Exceptions\TesseroException;
use Tessero\Io\HarwellBoeing;
use Tessero\Io\IdlSave;
use Tessero\Io\IoGuard;
use Tessero\Io\MatIO;
use Tessero\Io\MatrixMarket;
use Tessero\Io\NetCDF;
use Tessero\Io\Npy;
use Tessero\Io\SparseNpz;
use Tessero\NDArray;

/**
 * Regression tests for the PHP file readers treating their input as hostile (2026-10 FFI-binding review):
 * decompression caps, record chains that must move forward, header counts checked against the bytes
 * present, CSR structure validated on load, and stream wrappers refused. Every input is built here from
 * fixed bytes, so the tests are deterministic.
 */
final class IoHardeningTest extends TestCase
{
    private string $dir;

    protected function setUp(): void
    {
        $this->dir = sys_get_temp_dir() . '/tsr_iohard_' . getmypid();
        @mkdir($this->dir);
    }

    protected function tearDown(): void
    {
        gc_collect_cycles();
        foreach (glob($this->dir . '/*') ?: [] as $f) {
            @unlink($f);
        }
        @rmdir($this->dir);
    }

    // ------------------------------------------------------------------ 1. decompression caps

    public function testNpzMembersAreCappedBeforeInflating(): void
    {
        $this->requireZip();
        $path = $this->dir . '/bomb.npz';
        Npy::saveZ($path, ['z' => NDArray::zeros([1 << 18])]);           // 2 MiB of zeros, a few KiB on disk
        $this->assertLessThan(64 << 10, filesize($path));
        $this->throwsLike(TesseroException::class, 'maxBytes', static fn () => Npy::loadZ($path, 1 << 20));
        $this->assertSame([1 << 18], Npy::loadZ($path, 4 << 20)['z']->shape(), 'a sufficient limit still loads');
        $this->throwsLike(\ValueError::class, 'maxBytes', static fn () => Npy::loadZ($path, 0));
    }

    public function testNpzMemberCountIsCapped(): void
    {
        $this->requireZip();
        $path = $this->dir . '/many.npz';
        $zip = new \ZipArchive();
        $zip->open($path, \ZipArchive::CREATE | \ZipArchive::OVERWRITE);
        for ($i = 0; $i <= IoGuard::MAX_ZIP_ENTRIES; $i++) {
            $zip->addFromString("m{$i}.npy", '');
        }
        $zip->close();
        $this->throwsLike(TesseroException::class, 'members', static fn () => Npy::loadZ($path));
        $this->throwsLike(TesseroException::class, 'members', static fn () => SparseNpz::load($path));
    }

    public function testSparseNpzIsCappedBeforeInflating(): void
    {
        $this->requireZip();
        $path = $this->dir . '/sparse.npz';
        $n = 1 << 18;
        SparseNpz::save($path, ['data' => array_fill(0, $n, 0.0), 'indices' => array_fill(0, $n, 0),
            'indptr' => array_merge([0], array_fill(0, 1, $n)), 'shape' => [1, 1]]);
        $this->throwsLike(TesseroException::class, 'maxBytes', static fn () => SparseNpz::load($path, 1 << 20));
        $this->assertSame([1, 1], SparseNpz::load($path)['shape']);
    }

    public function testMatFileInflationIsCapped(): void
    {
        $inner = MatIO::write(['x' => ['shape' => [1, 1], 'class' => 'double', 'data' => [1.0]]]);
        $body = substr($inner, 128) . str_repeat("\0", 4 << 20);         // a real variable, then 4 MiB of padding
        $z = gzcompress($body);
        $mat = substr($inner, 0, 128) . pack('VV', 15, strlen($z)) . $z;  // miCOMPRESSED
        $this->throwsLike(TesseroException::class, 'maxBytes', static fn () => MatIO::read($mat, 1 << 20));
        $this->throwsLike(TesseroException::class, 'maxBytes', static fn () => MatIO::info($mat, 1 << 20));
        $this->assertSame([1.0], MatIO::read($mat)['x']['data'], 'within the default limit it reads');
    }

    public function testIdlInflationIsCapped(): void
    {
        $z = gzcompress(str_repeat("\0", 4 << 20));
        $sav = "SR\x00\x06" . self::idlRecord(10, 4, $z) . pack('NNNN', 6, 0, 0, 0);
        $this->throwsLike(TesseroException::class, 'maxBytes', static fn () => IdlSave::read($sav, 1 << 20));
    }

    // ------------------------------------------------------------------ 2. readsav record chain

    public function testIdlRecordChainMustMoveForward(): void
    {
        // a record whose next-record offset points at itself used to loop forever
        $self = "SR\x00\x04" . pack('NNNN', 10, 4, 0, 0);
        $this->throwsLike(TesseroException::class, 'next record', static fn () => IdlSave::read($self));
        // ... or backwards, into the signature
        $back = "SR\x00\x04" . pack('NNNN', 10, 0, 0, 0) . str_repeat("\0", 16);
        $this->throwsLike(TesseroException::class, 'next record', static fn () => IdlSave::read($back));
        // ... or past the end of the file
        $past = "SR\x00\x04" . pack('NNNN', 10, 1 << 30, 0, 0);
        $this->throwsLike(TesseroException::class, 'next record', static fn () => IdlSave::read($past));
        // compressed: a next-record offset inside the record header gave readBytes() a negative length
        $neg = "SR\x00\x06" . pack('NNNN', 10, 8, 0, 0) . str_repeat("\0", 16);
        $this->throwsLike(TesseroException::class, 'readsav', static fn () => IdlSave::read($neg));
        // a well-formed chain still reads to the end marker
        $ok = "SR\x00\x04" . pack('NNNN', 10, 20, 0, 0) . pack('NNNN', 6, 0, 0, 0);
        $this->assertSame([], IdlSave::read($ok));
    }

    // ------------------------------------------------------------------ 3. declared counts

    public function testMatDimensionsMustMatchTheData(): void
    {
        $mat = MatIO::write(['x' => ['shape' => [1, 1], 'class' => 'double', 'data' => [2.5]]]);
        $this->assertSame([2.5], MatIO::read($mat)['x']['data']);
        // dims sub-element data sits at bytes 160..167: declare 100000 x 100000 over one stored value
        $big = substr_replace($mat, pack('VV', 100000, 100000), 160, 8);
        $this->throwsLike(TesseroException::class, 'declares', static fn () => MatIO::read($big));
        // a zero dimension with a huge other one must not loop over the huge one
        $empty = substr_replace($mat, pack('VV', 0, 0xFFFFFFFF), 160, 8);
        $this->assertSame([], MatIO::read($empty)['x']['data']);
        // an element whose byte count runs past the end of the file
        $past = substr_replace($mat, pack('V', 0x7FFFFFF0), 132, 4);
        $this->throwsLike(TesseroException::class, 'declares', static fn () => MatIO::read($past));
        // the per-file value cap
        $this->throwsLike(TesseroException::class, 'maxElements', static fn () => MatIO::read(
            MatIO::write(['y' => ['shape' => [1, 4], 'class' => 'double', 'data' => [1.0, 2.0, 3.0, 4.0]]]), IoGuard::DEFAULT_MAX_BYTES, 3));
    }

    public function testNetcdfHeaderCountsAreChecked(): void
    {
        $this->assertSame([1, 2, 3, 4], NetCDF::read(self::cdf())['variables']['v']['data'], 'the well-formed file reads');
        $this->throwsLike(TesseroException::class, 'past the end', static fn () => NetCDF::read(self::cdf(dimLen: 1_000_000)));
        $this->throwsLike(TesseroException::class, 'undefined dimension', static fn () => NetCDF::read(self::cdf(dimId: 5)));
        $this->throwsLike(TesseroException::class, 'nc_type', static fn () => NetCDF::read(self::cdf(ncType: 9)));
        $this->throwsLike(TesseroException::class, 'count', static fn () => NetCDF::read(self::cdf(varCount: 0x7FFFFFFF)));
        $this->throwsLike(TesseroException::class, 'maxElements', static fn () => NetCDF::read(self::cdf(), 3));
        // a record variable: numrecs drives the loop, so it is checked against the bytes present
        $this->assertSame([1, 2, 3, 4], NetCDF::read(self::cdf(dimLen: 0, numrecs: 4, vsize: 1))['variables']['v']['data']);
        $this->throwsLike(TesseroException::class, 'past the end', static fn () => NetCDF::read(self::cdf(dimLen: 0, numrecs: 1000, vsize: 1)));
        $this->throwsLike(TesseroException::class, 'maxElements', static fn () => NetCDF::read(self::cdf(dimLen: 0, numrecs: 0x7FFFFFFF, vsize: 1)));
        // a truncated header is an exception, not a PHP warning
        $this->throwsLike(TesseroException::class, 'netcdf', static fn () => NetCDF::read(substr(self::cdf(), 0, 30)));
    }

    public function testMatrixMarketSizeLineIsCheckedBeforeAllocating(): void
    {
        $ok = "%%MatrixMarket matrix coordinate real general\n2 2 1\n2 1 5\n";
        $this->assertSame([[0.0, 0.0], [5.0, 0.0]], MatrixMarket::read($ok)['data']);
        $huge = "%%MatrixMarket matrix coordinate real general\n100000000 100000000 0\n";
        $this->throwsLike(TesseroException::class, 'maxElements', static fn () => MatrixMarket::read($huge));
        $short = "%%MatrixMarket matrix array real general\n1000 1000\n1\n";
        $this->throwsLike(TesseroException::class, 'truncated', static fn () => MatrixMarket::read($short));
        $nnz = "%%MatrixMarket matrix coordinate real general\n2 2 5000000\n1 1 1\n";
        $this->throwsLike(TesseroException::class, 'truncated', static fn () => MatrixMarket::read($nnz));
        $outside = "%%MatrixMarket matrix coordinate real general\n2 2 1\n3 1 5\n";
        $this->throwsLike(TesseroException::class, 'outside', static fn () => MatrixMarket::read($outside));
    }

    // ------------------------------------------------------------------ 4. CSR/CSC structure on load

    public function testHarwellBoeingStructureIsValidated(): void
    {
        $good = HarwellBoeing::write(2, 2, [0, 1, 2], [0, 1], [1.5, 2.5]);
        $this->assertSame([0, 1, 2], HarwellBoeing::read($good)['indptr']);
        $badIndex = HarwellBoeing::write(2, 2, [0, 1, 2], [0, 7], [1.5, 2.5]);
        $this->throwsLike(ShapeError::class, 'outside', static fn () => HarwellBoeing::read($badIndex));
        $decreasing = HarwellBoeing::write(2, 2, [0, 2, 1], [0, 1], [1.5, 2.5]);
        $this->throwsLike(ShapeError::class, 'indptr', static fn () => HarwellBoeing::read($decreasing));
        $notZero = HarwellBoeing::write(2, 2, [1, 1, 2], [0, 1], [1.5, 2.5]);
        $this->throwsLike(ShapeError::class, 'start at 0', static fn () => HarwellBoeing::read($notZero));
    }

    public function testSparseNpzStructureIsValidated(): void
    {
        $this->requireZip();
        $path = $this->dir . '/csr.npz';
        SparseNpz::save($path, ['data' => [1.0, 2.0], 'indices' => [0, 1], 'indptr' => [0, 1, 2], 'shape' => [2, 2]]);
        $this->assertSame([0, 1, 2], SparseNpz::load($path)['indptr']->toList());
        SparseNpz::save($path, ['data' => [1.0, 2.0], 'indices' => [0, 9], 'indptr' => [0, 1, 2], 'shape' => [2, 2]]);
        $this->throwsLike(ShapeError::class, 'outside', static fn () => SparseNpz::load($path));
        SparseNpz::save($path, ['data' => [1.0, 2.0], 'indices' => [0, 1], 'indptr' => [0, 5, 2], 'shape' => [2, 2]]);
        $this->throwsLike(ShapeError::class, 'indptr', static fn () => SparseNpz::load($path));
        SparseNpz::save($path, ['data' => [1.0, 2.0], 'indices' => [0, 1], 'indptr' => [0, 1], 'shape' => [2, 2]]);
        $this->throwsLike(ShapeError::class, 'inconsistent', static fn () => SparseNpz::load($path));
    }

    // ------------------------------------------------------------------ 5. stream wrappers

    public function testFileLoadersRefuseStreamWrappers(): void
    {
        $paths = ['php://filter/resource=/etc/hostname', 'phar:///tmp/x.phar/a.npy', 'http://127.0.0.1/a.npy',
            'data:text/plain;base64,AAAA', 'compress.zlib:///tmp/x'];
        foreach ($paths as $p) {
            $this->throwsLike(\ValueError::class, 'local file', static fn () => Npy::load($p), $p);
            $this->throwsLike(\ValueError::class, 'local file', static fn () => NDArray::load($p), $p);
            $this->throwsLike(\ValueError::class, 'local file', static fn () => Npy::save($p, NDArray::zeros([1])), $p);
            $this->throwsLike(\ValueError::class, 'local file', static fn () => Npy::loadZ($p), $p);
            $this->throwsLike(\ValueError::class, 'local file', static fn () => Npy::saveZ($p, []), $p);
            $this->throwsLike(\ValueError::class, 'local file', static fn () => SparseNpz::load($p), $p);
            $this->throwsLike(\ValueError::class, 'local file', static fn () => NDArray::memmap($p, 'r', [1]), $p);
        }
        $this->assertFalse(IoGuard::isStreamWrapper('C:\\data\\a.npy'), 'a Windows drive letter is a local path');
        $this->assertFalse(IoGuard::isStreamWrapper('/srv/data:2026/a.npy'));
    }

    // ------------------------------------------------------------------ 13. memmap w+ through a symlink

    public function testMemmapCreateRefusesASymlink(): void
    {
        if (PHP_OS_FAMILY === 'Windows') {
            $this->markTestSkipped('symlinks need privileges on Windows');
        }
        $victim = $this->dir . '/victim.bin';
        file_put_contents($victim, 'keep me');
        $link = $this->dir . '/link.bin';
        symlink($victim, $link);
        $this->throwsLike(TesseroException::class, 'symbolic link', static fn () => NDArray::memmap($link, 'w+', [4]));
        $this->throwsLike(TesseroException::class, 'symbolic link', static fn () => NDArray::openMemmap($link, 'w+', [4]));
        $this->assertSame('keep me', file_get_contents($victim), 'the link target was not truncated');
        $m = NDArray::memmap($link, 'r', [7], 'uint8');                   // reading through a link is fine
        $this->assertSame(7, $m->size());
        unset($m);
    }

    // ------------------------------------------------------------------ helpers

    /** An IDL record: type, next-record offset (absolute), the 4 unknown bytes, then $body. */
    private static function idlRecord(int $type, int $start, string $body): string
    {
        return pack('NNNN', $type, $start + 16 + strlen($body), 0, 0) . $body;
    }

    /** A netCDF-3 classic file with one int8 variable 'v' over dimension 'x' holding [1, 2, 3, 4]. */
    private static function cdf(int $dimLen = 4, int $dimId = 0, int $ncType = 1, int $varCount = 1, int $numrecs = 0, int $vsize = 4): string
    {
        $str = static fn (string $s): string => pack('N', strlen($s)) . $s . str_repeat("\0", (4 - strlen($s) % 4) % 4);
        $head = 'CDF' . chr(1) . pack('N', $numrecs)
            . pack('NN', 0x0A, 1) . $str('x') . pack('N', $dimLen)
            . pack('NN', 0, 0)
            . pack('NN', 0x0B, $varCount) . $str('v') . pack('NN', 1, $dimId) . pack('NN', 0, 0)
            . pack('NN', $ncType, $vsize);
        $begin = strlen($head) + 4;

        return $head . pack('N', $begin) . "\x01\x02\x03\x04";
    }

    private function requireZip(): void
    {
        if (! class_exists(\ZipArchive::class)) {
            $this->markTestSkipped('ext-zip is not loaded');
        }
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
