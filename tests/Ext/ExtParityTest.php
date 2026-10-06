<?php

declare(strict_types=1);

namespace Tessero\Tests\Ext;

use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\TestCase;
use Tessero\Ext\Engine;
use Tessero\Ext\Math;
use Tessero\Ext\NDArray as X;

/**
 * Runs the NumPy-generated fixtures through the native extension (skipped
 * when ext-tessero is not loaded), so both backends are held to the same
 * reference: slicing/stride fuzz, element-wise, reductions, LP/MILP, MDP, RNG.
 */
final class ExtParityTest extends TestCase
{
    protected function setUp(): void
    {
        if (! extension_loaded('tessero')) {
            $this->markTestSkipped('ext-tessero is not loaded.');
        }
    }

    public static function fuzz(): iterable
    {
        foreach (json_decode((string) file_get_contents(__DIR__ . '/../fixtures/fuzz.json'), true) as $i => $c) {
            yield sprintf('%03d [%s]', $i, $c['spec']) => [$c];
        }
    }

    public static function parity(): iterable
    {
        $f = json_decode((string) file_get_contents(__DIR__ . '/../fixtures/parity.json'), true);
        foreach ($f['cases'] as $i => $c) {
            if (in_array($c['family'], ['binary', 'binary_scalar', 'unary', 'reduce', 'slice', 'lp', 'mdp', 'random', 'where'], true)) {
                yield sprintf('%03d %s %s', $i, $c['family'], is_string($c['op'] ?? null) ? $c['op'] : '') => [$c];
            }
        }
    }

    #[DataProvider('fuzz')]
    public function testFuzz(array $c): void
    {
        $base = X::arange((int) array_product($c['shape']))->reshape($c['shape']);
        $v = $base[$c['spec']];
        $v = $v instanceof X ? $v : X::array($v);
        $this->same($c['expect'], $v, 'view');
        if (! isset($c['binary'])) {
            return;
        }
        $other = X::arange((int) array_product($c['bshape']))->mul(0.5)->sub(3)->reshape($c['bshape']);
        $this->same($c['binary'], $v * 1.5 - $other, 'operators on a strided view');
        $this->same($c['sum_axis'], $v->sum($c['axis']), 'sum');
        $this->same($c['max_axis'], $v->max($c['axis']), 'max');
        $this->same($c['transposed_copy'], $v->t()->astype('float32'), 'transposed cast');
        $this->assertSame(json_encode($v->toArray(), JSON_PRESERVE_ZERO_FRACTION), $v->toJson());
    }

    #[DataProvider('parity')]
    public function testParity(array $c): void
    {
        switch ($c['family']) {
            case 'binary':
                $this->same($c['expect'], self::arr($c['a'])->{$c['op']}(self::arr($c['b'])), $c['op'], 1e-12);
                break;
            case 'binary_scalar':
                $b = is_string($c['b']) ? (float) $c['b'] : $c['b'];
                $this->same($c['expect'], self::arr($c['a'])->{$c['op']}($b), $c['op'], 1e-12);
                break;
            case 'unary':
                $this->same($c['expect'], self::arr($c['a'])->{$c['op']}(), $c['op'], 1e-13);
                break;
            case 'reduce':
                $a = self::arr($c['a']);
                if (is_array($c['axis']) || ! in_array($c['op'], ['sum', 'prod', 'min', 'max', 'mean', 'argmin', 'argmax', 'var', 'std', 'cumsum', 'cumprod'], true)) {
                    $this->markTestSkipped('multi-axis reductions are FFI-only for now');
                }
                $r = match ($c['op']) {
                    'var', 'std' => $a->{$c['op']}($c['axis'], $c['ddof']),
                    default => $a->{$c['op']}($c['axis']),
                };
                $this->same($c['expect'], $r, $c['op'], 1e-12);
                break;
            case 'slice':
                $v = self::arr($c['a'])[$c['spec']];
                $this->same($c['expect'], $v instanceof X ? $v : X::array($v), 'slice');
                break;
            case 'where':
                $this->same($c['expect'], X::where(self::arr($c['c']), self::arr($c['a']), self::arr($c['b'])), 'where');
                break;
            case 'random':
                $this->random($c);
                break;
            case 'lp':
                $this->lp($c);
                break;
            case 'mdp':
                $pi = Engine::mdpPolicyIteration(self::arr($c['P']), self::arr($c['R']), (float) $c['gamma']);
                $this->same($c['expect_policy'], $pi['policy'], 'policy');
                $this->same($c['expect_V'], $pi['values'], 'values', 1e-10);
                $vi = Engine::mdpValueIteration(self::arr($c['P']), self::arr($c['R']), (float) $c['gamma'], 1e-9, 100000);
                $this->same($c['expect_policy'], $vi['policy'], 'VI policy');
                $fh = Engine::mdpFiniteHorizon(self::arr($c['P']), self::arr($c['R']), 5);
                $this->same($c['expect_fh_V'], $fh['values'], 'finite horizon', 1e-12);
                break;
        }
    }

    private function random(array $c): void
    {
        if (is_array($c['seed'])) {
            $this->markTestSkipped('array seeds are FFI-only');
        }
        $s = $c['seed'];
        $e = $c['expect'];
        match ($c['op']) {
            'random' => $this->same($e, Engine::random($s, $c['size']), 'random'),
            'normal' => $this->same($e, Engine::normal($s, $c['size'], (float) $c['loc'], (float) $c['scale']), 'normal'),
            'integers', 'integers_big' => $this->same($e, Engine::integers($s, $c['size'], $c['low'], $c['high']), 'integers'),
            'uniform' => $this->same($e, Engine::random($s, $c['size'], (float) $c['low'], (float) $c['high']), 'uniform'),
            default => $this->markTestSkipped('op not bound in the extension'),
        };
    }

    private function lp(array $c): void
    {
        $e = $c['expect'];
        if ($c['op'] === 'milp') {
            $r = Engine::milp(self::arr($c['c']), array_map('boolval', $c['integrality']['data']), self::arr($c['A_ub']), self::arr($c['b_ub']),
                null, null, array_map(static fn ($u): array => [0.0, (float) $u], $c['ub']['data']));
            $this->assertSame($e['status'], $r['status']);
            $this->assertEqualsWithDelta((float) $e['fun'], $r['fun'], 1e-9 * max(1.0, abs((float) $e['fun'])));

            return;
        }
        $hasEq = $c['A_eq']['shape'][0] > 0;
        $bounds = array_map(static fn (array $b): array => array_map(static fn ($v) => $v === null ? null : (float) $v, $b), $c['bounds']);
        $r = Engine::linprog(self::arr($c['c']), self::arr($c['A_ub']), self::arr($c['b_ub']),
            $hasEq ? self::arr($c['A_eq']) : null, $hasEq ? self::arr($c['b_eq']) : null, $bounds);
        $this->assertSame($e['status'], $r['status'], $r['message']);
        if ($e['status'] === 0) {
            $this->assertEqualsWithDelta((float) $e['fun'], $r['fun'], 1e-9 * max(1.0, abs((float) $e['fun'])));
            foreach ($e['ineq']['data'] as $i => $y) {
                $this->assertEqualsWithDelta((float) $y, $r['ineqlin'][$i], 1e-7);
            }
        }
    }

    public static function ufunc(): iterable
    {
        $d = json_decode((string) file_get_contents(__DIR__ . '/../fixtures/ufunc.json'), true);
        foreach ($d['cases'] as $i => $c) {
            yield sprintf('%02d %s(%s)', $i, $c['f'], implode(', ', array_map(static fn ($a) => $a['dtype'], $c['args']))) => [$c];
        }
    }

    /** Loop ufuncs against NumPy / scipy.special: values and result dtype. */
    #[DataProvider('ufunc')]
    public function testUfunc(array $c): void
    {
        $args = array_map(static fn (array $a) => self::arr($a), $c['args']);
        $r = Math::{$c['f']}(...$args);
        $this->assertSame($c['expect']['dtype'], $r->dtype(), $c['f'] . ': dtype');
        $this->same($c['expect'], $r, $c['f'], (float) ($c['tol'] ?? 1e-13));
        // the same call writing into a preallocated output gives the same bytes
        $out = X::zeros($r->shape(), $r->dtype());
        $this->assertSame($out, Math::{$c['f']}(...[...$args, $out]));
        $this->assertSame($r->toBytes(), $out->toBytes(), $c['f'] . ': out=');
    }

    /** Kernel ufuncs are the NDArray methods: identical bytes. */
    public function testKernelUfuncsMatchMethods(): void
    {
        $a = X::array([[-2.5, -0.5, 0.0], [0.25, 1.0, 3.75]]);
        $b = X::array([1.5, -2.0, 0.5]);
        $unary = ['negative' => 'neg', 'absolute' => 'abs', 'square' => 'square', 'sign' => 'sign', 'exp' => 'exp', 'expm1' => 'expm1',
            'sin' => 'sin', 'cos' => 'cos', 'tan' => 'tan', 'arctan' => 'arctan', 'sinh' => 'sinh', 'cosh' => 'cosh', 'tanh' => 'tanh',
            'floor' => 'floor', 'ceil' => 'ceil', 'rint' => 'rint', 'isnan' => 'isnan', 'isfinite' => 'isfinite', 'isinf' => 'isinf'];
        foreach ($unary as $f => $m) {
            $this->assertSame($a->{$m}()->toBytes(), Math::{$f}($a)->toBytes(), $f);
        }
        $binary = ['add' => 'add', 'subtract' => 'sub', 'multiply' => 'mul', 'divide' => 'div', 'power' => 'pow', 'mod' => 'mod',
            'floorDivide' => 'floorDiv', 'maximum' => 'maximum', 'minimum' => 'minimum', 'arctan2' => 'atan2', 'hypot' => 'hypot',
            'equal' => 'eq', 'less' => 'lt', 'greaterEqual' => 'ge'];
        foreach ($binary as $f => $m) {
            $this->assertSame($a->{$m}($b)->toBytes(), Math::{$f}($a, $b)->toBytes(), $f);
            $this->assertSame($a->{$m}($b)->dtype(), Math::{$f}($a, $b)->dtype(), $f . ' dtype');
        }
    }

    /** Loop ufuncs split large arrays across threads; the bytes must not change. */
    public function testUfuncThreadsIdentical(): void
    {
        $x = \Tessero\Ext\Engine::normal(7, [600, 700]);            // 420k elements, above the parallel threshold
        $y = \Tessero\Ext\Engine::normal(8, [700]);
        $cases = [
            fn () => Math::cbrt($x),                          // one coalesced run -> chunked
            fn () => Math::logaddexp($x, $y),                 // broadcast -> parallel over rows
            fn () => Math::erf($x['::-1, ::2']),              // strided view
            fn () => Math::fmod($x, 0.7),
        ];
        $prev = \Tessero\Ext\Engine::getMaxThreads();
        foreach ($cases as $i => $f) {
            \Tessero\Ext\Engine::setMaxThreads(1);
            $one = $f()->toBytes();
            \Tessero\Ext\Engine::setMaxThreads(4);
            $this->assertSame($one, $f()->toBytes(), "case {$i}");
        }
        \Tessero\Ext\Engine::setMaxThreads($prev);
    }

    private static function arr(array $enc): X
    {
        $data = array_map(static fn ($v) => match ($v) { 'nan' => NAN, 'inf' => INF, '-inf' => -INF, default => is_string($v) ? (float) $v : $v }, $enc['data']);
        if ($enc['dtype'] === 'complex128') {
            throw new \RuntimeException('complex fixtures are not used here');
        }

        return X::fromBytes(self::pack($data, $enc['dtype']), $enc['dtype'], $enc['shape']);
    }

    private static function pack(array $data, string $dtype): string
    {
        return match ($dtype) {
            'float64' => pack('d*', ...$data),
            'float32' => pack('g*', ...$data),
            'int64' => pack('q*', ...$data),
            'int32' => pack('l*', ...$data),
            'bool', 'uint8' => pack('C*', ...array_map('intval', $data)),
        };
    }

    private function same(mixed $enc, X|int|float $actual, string $what, float $tol = 0.0): void
    {
        if (! is_array($enc)) {
            $enc = ['shape' => [], 'data' => [$enc]];
        }
        $act = $actual instanceof X ? $actual : X::array($actual);
        $this->assertSame($enc['shape'], $act->shape(), "{$what}: shape");
        $got = $act->toList();
        foreach ($enc['data'] as $i => $v) {
            $e = match ($v) { 'nan' => NAN, 'inf' => INF, '-inf' => -INF, default => is_string($v) ? (float) $v : $v };
            $g = $got[$i];
            if (is_float($e) && is_nan($e)) {
                $this->assertTrue(is_nan($g), "{$what}[{$i}] NaN");
                continue;
            }
            if ($tol === 0.0 || ! is_float($e) || is_infinite($e)) {
                $this->assertTrue($e == $g && (is_bool($e) === is_bool($g)), "{$what}[{$i}] expected " . var_export($e, true) . ' got ' . var_export($g, true));
            } else {
                $this->assertTrue(abs($e - $g) <= $tol * max(1.0, abs($e)), sprintf('%s[%d] expected %.17g got %.17g', $what, $i, $e, $g));
            }
        }
    }
}
