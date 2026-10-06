<?php

declare(strict_types=1);

namespace Tessero\Laravel\Tests;

use PHPUnit\Framework\TestCase;
use Tessero\Laravel\Casts\AsNDArray;
use Tessero\Laravel\Casts\TensorCast;
use Tessero\Laravel\Rules\NumericArray;
use Tessero\Laravel\TesseroManager;

require_once __DIR__ . '/stubs.php';

final class BridgeTest extends TestCase
{
    /** @return list<string> */
    private static function backends(): array
    {
        $b = [];
        if (extension_loaded('tessero')) {
            $b[] = 'ext';
        }
        if (extension_loaded('ffi') && \Tessero\Native\Library::available()) {
            $b[] = 'ffi';
        }

        return $b;
    }

    public function testCastRoundTripsJsonAndBinary(): void
    {
        foreach (['json', 'binary'] as $format) {
            $caster = AsNDArray::castUsing([$format]);
            $stored = $caster->set(null, 'w', [[0.1, 2.0], [3.5, -1.0]], []);
            $this->assertIsString($stored);
            $back = $caster->get(null, 'w', $stored, []);
            $this->assertSame([[0.1, 2.0], [3.5, -1.0]], $back->toArray(), $format);
            $this->assertSame([[0.1, 2.0], [3.5, -1.0]], $caster->serialize(null, 'w', $back, []));
        }
        $this->assertSame('[[0.1,2.0],[3.5,-1.0]]', AsNDArray::castUsing([])->set(null, 'w', [[0.1, 2.0], [3.5, -1.0]], []));
        $this->assertNull(AsNDArray::castUsing([])->get(null, 'w', null, []));
    }

    public function testBinaryIsExactAndDtypeArgumentApplies(): void
    {
        $caster = TensorCast::castUsing(['binary', 'float32']);
        $stored = $caster->set(null, 'f', [1 / 3, 2.5], []);
        $payload = json_decode($stored, true);
        $this->assertSame('float32', $payload['dtype']);
        $this->assertSame([2], $payload['shape']);
        $back = $caster->get(null, 'f', $stored, []);
        $dt = $back->dtype();
        $this->assertSame('float32', is_string($dt) ? $dt : $dt->name());
        $this->assertSame((float) unpack('g', pack('g', 1 / 3))[1], $back->toArray()[0]);
    }

    public function testNumericArrayRule(): void
    {
        $this->assertNull(NumericArray::check([[1, 2.5], [3, 4]], ndim: 2));
        $this->assertNull(NumericArray::check([[1, 2, 3]], shape: [null, 3]));
        $this->assertStringContainsString('rectangular', NumericArray::check([[1, 2], [3]]));
        $this->assertStringContainsString('only numbers', NumericArray::check([1, '2']));
        $this->assertStringContainsString('finite', NumericArray::check([1.0, NAN]));
        $this->assertStringContainsString('limit', NumericArray::check(range(1, 11), maxElements: 10));
        $this->assertStringContainsString('dimension', NumericArray::check([1, 2], ndim: 2));
        $this->assertStringContainsString('list arrays', NumericArray::check(['a' => 1]));
        $messages = [];
        (new NumericArray(ndim: 1))->validate('prices', [[1]], function (string $m) use (&$messages): void { $messages[] = $m; });
        $this->assertSame(['The :attribute must have 1 dimension(s), got 2.'], $messages);
    }

    public function testManagerGivesTheSameAnswersOnEveryBackend(): void
    {
        $backends = self::backends();
        if ($backends === []) {
            $this->markTestSkipped('no backend');
        }
        $results = [];
        foreach ($backends as $b) {
            $m = new TesseroManager(['backend' => $b, 'threads' => 1, 'memory_budget' => 0, 'epsilon' => 1e-9]);
            $m->applyRuntimeSettings();
            $this->assertSame($b, $m->backend());
            $lp = $m->linprog([-40, -30], [[2, 1], [1, 1]], [100, 80], null, null, [[0, 40], [0, null]]);
            $milp = $m->milp([10, 7, 12], true, [[-6, -5, -8]], [-10], null, null, [0, 1]);
            $mdp = $m->solveMdp([[[1, 0], [0, 1]], [[0, 1], [1, 0]]], [[0, 0], [1, 0]], 0.9);
            $results[$b] = [
                'lp' => [$lp['x'], $lp['fun'], $lp['ineqlin']],
                'milp' => [$milp['x'], $milp['fun']],
                'mdp' => $mdp['policy'],
                'rng' => $m->normal(12345, 2, 1.5, 2.0)->toArray(),
                'arr' => $m->array([[1, 2], [3, 4]])->sum(),
            ];
        }
        $first = reset($results);
        $this->assertEqualsWithDelta([20.0, 60.0], $first['lp'][0], 1e-9);
        $this->assertSame([-1.3476500729092624, 4.02745691625822], $first['rng']);
        $this->assertSame([1, 0], $first['mdp']);
        foreach ($results as $b => $r) {
            $this->assertEquals($first, $r, "backend {$b} agrees");
        }
    }

    public function testMemmapThroughTheManager(): void
    {
        $manager = new TesseroManager(['backend' => 'auto']);
        $file = sys_get_temp_dir() . '/tessero-bridge-' . getmypid() . '.bin';
        $a = $manager->memmap($file, 'w+', [2, 2]);
        $a->assign([[1.0, 2.0], [3.0, 4.0]]);
        $a->flush(true);
        unset($a);
        $this->assertSame(10.0, $manager->memmap($file, 'r', [2, 2])->sum());
        unlink($file);
        $npy = $file . '.npy';
        $m = $manager->openMemmap($npy, 'w+', [3], 'int32');
        $m->assign([1, 2, 3]);
        unset($m);
        $this->assertSame([1, 2, 3], $manager->load($npy)->toList());
        $this->assertTrue($manager->load($npy, 'r')->isMemmap());
        unlink($npy);
    }
}
