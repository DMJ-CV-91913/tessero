<?php

declare(strict_types=1);

namespace Tessero\Tests\Unit;

use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\TestCase;
use Tessero\DType;
use Tessero\Exceptions\DTypeError;
use Tessero\Exceptions\ShapeError;
use Tessero\Exceptions\TesseroException;
use Tessero\Math;
use Tessero\NDArray;
use Tessero\Tessero;

/**
 * .npy load/save and memory maps in the FFI package, against files written by
 * NumPy (tests/fixtures/npy), and, when ext-tessero is loaded, the same files
 * and calls through Tessero\Ext\NDArray: same bytes, same errors.
 */
final class FilesTest extends TestCase
{
    private string $dir;

    protected function setUp(): void
    {
        $this->dir = sys_get_temp_dir() . '/tsr_files_' . getmypid() . '_' . bin2hex(random_bytes(3));
        mkdir($this->dir);
    }

    protected function tearDown(): void
    {
        gc_collect_cycles();
        foreach (glob($this->dir . '/*') ?: [] as $f) {
            @unlink($f);
        }
        @rmdir($this->dir);
    }

    public static function npyFiles(): iterable
    {
        $m = json_decode((string) file_get_contents(__DIR__ . '/../fixtures/npy.json'), true);
        foreach ($m['files'] as $name => $meta) {
            yield $name => [$name, $meta];
        }
    }

    #[DataProvider('npyFiles')]
    public function testLoadNumpyFiles(string $name, array $meta): void
    {
        $path = __DIR__ . "/../fixtures/npy/{$name}.npy";
        $expect = base64_decode($meta['c_bytes']);
        $a = NDArray::load($path);
        $this->assertSame($meta['shape'], $a->shape());
        $this->assertSame($meta['dtype'], $a->dtype()->name());
        $this->assertSame($expect, $a->toBytes(), 'load');
        if (! in_array(0, $meta['shape'], true)) {
            foreach (['r', 'c', 'r+'] as $mode) {
                $copy = $this->dir . "/{$name}.npy";
                copy($path, $copy);
                $m = NDArray::load($copy, $mode);
                $this->assertTrue($m->isMemmap(), $mode);
                $this->assertSame($meta['shape'], $m->shape());
                $this->assertSame($expect, $m->toBytes(), "memmap {$mode}");
                unset($m);
            }
        }
        // save() writes the file NumPy writes (C-order arrays; NumPy keeps Fortran order, Tessero writes C)
        if (! $meta['fortran'] && $name !== 'f8_v2') {
            $a->save($this->dir . '/out.npy');
            $this->assertSame(file_get_contents($path), file_get_contents($this->dir . '/out.npy'), 'save is byte-identical to numpy.save');
        }
        if (extension_loaded('tessero')) {
            $x = \Tessero\Ext\NDArray::load($path);
            $this->assertSame($expect, $x->toBytes(), 'extension load');
            $this->assertSame($a->shape(), $x->shape());
            if (! $meta['fortran'] && $name !== 'f8_v2') {
                $x->save($this->dir . '/ext.npy');
                $this->assertSame(file_get_contents($path), file_get_contents($this->dir . '/ext.npy'), 'extension save');
            }
            if (! in_array(0, $meta['shape'], true)) {
                $this->assertSame($expect, \Tessero\Ext\NDArray::load($path, 'r')->toBytes(), 'extension memmap');
            }
        }
    }

    public function testInvalidFilesFailTheSameWayOnBothBackends(): void
    {
        $cases = ['bad_bigendian' => DTypeError::class, 'bad_structured' => DTypeError::class,
            'bad_truncated' => TesseroException::class, 'bad_magic' => TesseroException::class];
        foreach ($cases as $name => $class) {
            $path = __DIR__ . "/../fixtures/npy/{$name}.npy";
            foreach ([null, 'r'] as $mode) {
                if ($name === 'bad_truncated' && $mode === 'r') {
                    continue;                                   // a mapping of a short file is refused by size instead
                }
                $this->assertThrows($class, static fn () => NDArray::load($path, $mode), "{$name} {$mode}");
                if (extension_loaded('tessero')) {
                    $ext = $class === DTypeError::class ? \Tessero\Ext\DTypeException::class : \Tessero\Ext\Exception::class;
                    $this->assertThrows($ext, static fn () => \Tessero\Ext\NDArray::load($path, $mode), "ext {$name} {$mode}");
                }
            }
        }
        $this->assertThrows(ShapeError::class, static fn () => NDArray::load(__DIR__ . '/../fixtures/npy/bad_truncated.npy', 'r'));
    }

    public function testMemmapLifecycle(): void
    {
        $f = $this->dir . '/a.f64';
        $a = NDArray::memmap($f, 'w+', [3, 2]);
        $this->assertTrue($a->isMemmap());
        $this->assertFalse($a->isReadonly());
        $this->assertSame(48, filesize($f));
        $this->assertSame(realpath($f), $a->filename());
        $a['1:'] = [[1, 2], [3, 4]];
        $a->flush();
        $a->flush(true);
        $v = $a[':, 1'];
        $info = Tessero::info();
        $this->assertGreaterThanOrEqual(48, $info['memory_mapped']);
        unset($a);
        $this->assertSame([0.0, 2.0, 4.0], $v->toList(), 'a view keeps the mapping alive');
        unset($v);
        gc_collect_cycles();
        $this->assertSame(0, Tessero::info()['memory_mapped'], 'released with the last view');

        $r = NDArray::memmap($f, 'r');
        $this->assertSame([6], $r->shape());
        $this->assertEquals(10.0, $r->sum());
        $this->assertTrue($r->isReadonly());
        $this->assertThrows(TesseroException::class, static fn () => $r[0] = 1);
        $this->assertThrows(TesseroException::class, static fn () => Math::negative($r, $r));
        $this->assertThrows(TesseroException::class, static fn () => $r->assign(0));
        $h = clone $r;
        $h[0] = 5;
        $this->assertFalse($h->isMemmap(), 'a clone is a heap copy');
        $this->assertEquals(5.0, $h->item(0));
        $this->assertEquals(0.0, $r->item(0));

        $c = NDArray::memmap($f, 'c', [3, 2]);
        $c[2] = [-1, -1];
        unset($c);
        $this->assertSame([3.0, 4.0], NDArray::memmap($f, 'r', [3, 2])[2]->toList(), 'copy-on-write leaves the file');

        $p = NDArray::memmap($f, 'r+', [2], 'float64', 48);        // extends the file
        $p->assign([7, 8]);
        $p->flush(true);
        clearstatcache();
        $this->assertSame(64, filesize($f));
        $this->assertSame([7.0, 8.0], NDArray::memmap($f, 'r', null, 'float64', 48)->toList());

        $i = NDArray::memmap($this->dir . '/i.i32', 'w+', [4], 'int32');
        $i->assign([1, -2, 3, -4]);
        $this->assertSame([1, 2, 3, 4], Math::absolute($i)->toList());
        $this->assertSame(-2, NDArray::memmap($this->dir . '/i.i32', 'r', null, 'int32')->sum());
        if (extension_loaded('tessero')) {
            $this->assertSame($i->toBytes(), \Tessero\Ext\NDArray::memmap($this->dir . '/i.i32', 'r', null, 'int32')->toBytes(), 'extension reads the same file');
        }
    }

    public function testMemmapErrorsMatchTheExtension(): void
    {
        $f = $this->dir . '/e.f64';
        NDArray::memmap($f, 'w+', [6])->flush();
        $cases = [
            [fn ($X) => $X::memmap($this->dir . '/missing', 'r'), TesseroException::class, 'Tessero\Ext\Exception'],
            [fn ($X) => $X::memmap('php://memory', 'r+', [1]), \ValueError::class, \ValueError::class],
            [fn ($X) => $X::memmap($f, 'x'), \ValueError::class, \ValueError::class],
            [fn ($X) => $X::memmap($f, 'r', [100]), ShapeError::class, 'Tessero\Ext\ShapeException'],
            [fn ($X) => $X::memmap($f, 'r', null, 'float64', 4), \ValueError::class, \ValueError::class],
            [fn ($X) => $X::memmap($this->dir . '/n', 'w+'), \ValueError::class, \ValueError::class],
            [fn ($X) => $X::memmap($f, 'r', [0]), ShapeError::class, 'Tessero\Ext\ShapeException'],
            [fn ($X) => $X::load($f, 'w+'), \ValueError::class, \ValueError::class],
        ];
        foreach ($cases as $k => [$call, $ffi, $ext]) {
            $this->assertThrows($ffi, static fn () => $call(NDArray::class), "case {$k}");
            if (extension_loaded('tessero')) {
                $this->assertThrows($ext, static fn () => $call(\Tessero\Ext\NDArray::class), "ext case {$k}");
            }
        }
    }

    public function testOpenMemmapCreatesNumpyReadableFiles(): void
    {
        $f = $this->dir . '/m.npy';
        $a = NDArray::openMemmap($f, 'w+', [4, 3], 'float32');
        $a->assign(NDArray::arange(12.0)->reshape([4, 3]));
        $a->flush(true);
        unset($a);
        $b = NDArray::load($f);
        $this->assertSame(DType::Float32, $b->dtype());
        $this->assertSame([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0], [9.0, 10.0, 11.0]], $b->toArray());
        $r = NDArray::openMemmap($f, 'r');
        $this->assertSame([4, 3], $r->shape());
        $this->assertTrue($r->isReadonly());
        // Fortran order: the file holds the transpose; the mapping has F strides
        $g = $this->dir . '/f.npy';
        $F = NDArray::openMemmap($g, 'w+', [2, 3], 'int64', fortranOrder: true);
        $F->assign([[1, 2, 3], [4, 5, 6]]);
        unset($F);
        $this->assertSame([[1, 2, 3], [4, 5, 6]], NDArray::load($g)->toArray());
        $this->assertSame([[1, 2, 3], [4, 5, 6]], NDArray::load($g, 'r')->toArray());
        if (extension_loaded('tessero')) {
            $this->assertSame([[1, 2, 3], [4, 5, 6]], \Tessero\Ext\NDArray::load($g)->toArray());
            $this->assertSame(file_get_contents($f), (function () use ($f) {
                $e = \Tessero\Ext\NDArray::openMemmap($this->dir . '/e.npy', 'w+', [4, 3], 'float32');
                $e->assign(\Tessero\Ext\NDArray::arange(12.0)->reshape([4, 3]));
                unset($e);

                return file_get_contents($this->dir . '/e.npy');
            })(), 'both backends write the same .npy file');
        }
    }

    /** open_basedir is process-wide and can only be tightened, so this runs in a child process. */
    public function testOpenBasedirIsEnforced(): void
    {
        $f = $this->dir . '/ob.f64';
        NDArray::memmap($f, 'w+', [2])->flush();
        $root = dirname(__DIR__, 2);
        $script = $this->dir . '/ob.php';
        file_put_contents($script, '<?php require ' . var_export($root . '/tests/autoload.php', true) . ';
            echo json_encode(Tessero\NDArray::memmap(' . var_export($f, true) . ', "r")->shape()), "\n";
            try { Tessero\NDArray::memmap("/etc/hostname", "r", null, "uint8"); echo "ALLOWED\n"; }
            catch (Tessero\Exceptions\TesseroException $e) { echo "refused\n"; }');
        $cmd = escapeshellarg(PHP_BINARY) . ' -d ffi.enable=1 -d open_basedir=' . escapeshellarg($this->dir . PATH_SEPARATOR . $root)
            . ' ' . escapeshellarg($script) . ' 2>&1';
        $out = (string) shell_exec($cmd);
        $this->assertSame("[2]\nrefused\n", $out, $out);
    }

    private function assertThrows(string $class, callable $f, string $what = ''): void
    {
        try {
            $f();
        } catch (\Throwable $e) {
            $this->assertInstanceOf($class, $e, $what . ': ' . get_class($e) . ' ' . $e->getMessage());

            return;
        }
        $this->fail("expected {$class}: {$what}");
    }
}
