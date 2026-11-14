<?php

declare(strict_types=1);

namespace Tessero\Tests\Unit;

use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\TestCase;
use Tessero\Exceptions\TesseroException;
use Tessero\Linalg;
use Tessero\NDArray;
use Tessero\Optimize\LeastSquares;

/**
 * curveFit runs MINPACK's lmdif as scipy.optimize.curve_fit does (linalg.lmdif). Every data set and model here is
 * built from + - * / alone (each rounded once, the same in PHP and NumPy), so both hold the same doubles, and the
 * fits are compared with SciPy's bit for bit: the parameters, the residuals, the number of evaluations and the
 * status, for a per-point model and for whole-array models on each backend, each fit run twice.
 */
final class CurveFitMinpackTest extends TestCase
{
    /**
     * scipy.optimize.curve_fit (SciPy 1.17.1, NumPy 2.4.4, full_output=True) on the problems below: popt, pcov
     * (row-major), nfev, ier and the SHA-1 of fvec as little-endian doubles; each double as big-endian hex.
     */
    private const SCIPY = [
        'rational_12' => [['40080aa9df520c42', '3fe5a7348d46a84d', '3fc6987b3d1f0be1'],
            ['3f3fe84dbb05855b', 'bf2c7bcccf189213', 'bf3a12f55a507f7f', 'bf2c7bcccf189213', '3f34efc4e11ac6f4', '3f381f0cdcde2f64', 'bf3a12f55a507f7f', '3f381f0cdcde2f64', '3f40961b41986e66'],
            30, 1, '16bc7c9a3cc2b3023e26d0e62976431c10f4a1c4'],
        'rational_12_sigma_abs' => [['40080ea98d11ffd8', '3fe5992c572fea15', '3fc648f5bc723acb'],
            ['3fa2a53f83c07345', 'bf9253ea810b060b', 'bfa0e980bb5603bf', 'bf9253ea810b060b', '3f9467e0c97e94b4', '3f9a1f853954d018', 'bfa0e980bb5603bf', '3f9a1f853954d018', '3fa3f425f6fa51b7'],
            30, 1, 'f71561174ae6ff1ecd9cd5823e767b5c3bbbd1b1'],
        'mm_20_scalar_sigma' => [['400fff869dff8354', '4003fd3e9e9c29d5'],
            ['3f16ee1f98e02611', '3f2be5a9b7df5690', '3f2be5a9b7df5690', '3f45700b74f09a56'],
            19, 1, 'b8f1547e15755c70a19147728a05d683a1aeb3b1'],
        'poly5_40_sigma' => [['3fefe9e533861fa8', 'bfdf438217ad836d', '3fce5b8972dffa2c', 'bf8e59e0df35590d', '3f34af971016ec07'],
            ['3ef75c6a2ae35e5e', 'bf132e4a547b58a3', '3f11f871e9855e37', 'bef94b3c21abbe7c', '3ec7ce2785cf006c', 'bf132e4a547b58a3', '3f37e9da887813c2', 'bf3a30761bae6a59', '3f2402cf8a44ca7f', 'bef3d07ad7892fc1', '3f11f871e9855e37', 'bf3a30761bae6a59', '3f3f114965daa05d', 'bf28ebb3473b4252', '3ef97eaf6b9864fc', 'bef94b3c21abbe7c', '3f2402cf8a44ca7f', 'bf28ebb3473b4252', '3f14a786b6ae3396', 'bee5a1d4f40e2db3', '3ec7ce2785cf006c', 'bef3d07ad7892fc1', '3ef97eaf6b9864fc', 'bee5a1d4f40e2db3', '3eb70f864167a151'],
            13, 1, '9178bf99d6c9ef70f84caf64ad71039b2d0b6421'],
        'dead_param_10' => [['400008b04086f0cd', '3feff05c58a26dce', '0000000000000000'],
            ['7ff0000000000000', '7ff0000000000000', '7ff0000000000000', '7ff0000000000000', '7ff0000000000000', '7ff0000000000000', '7ff0000000000000', '7ff0000000000000', '7ff0000000000000'],
            9, 3, '4cc88e67e6592db4198294f7cff46364df9aa2d0'],
        'line_6' => [['3fff7f44c118de5c', '3ff20ea0ea0ea0e4'],
            ['3f41a793af044103', 'bf5611789310a3df', 'bf5611789310a3df', '3f743aae74b143bc'],
            7, 3, 'f1ea54d40dad3083d624ae7e016241db9a293efd'],
    ];

    /**
     * name => [per-point model, whole-array model, x, y, p0, sigma, absolute_sigma]. The whole-array models take
     * floats or, for a Jacobian's evaluations in one call, (1, n) arrays, and give every element NumPy's
     * operations in NumPy's order.
     *
     * @return array<string, array{0: callable, 1: callable, 2: list<float>, 3: list<float>, 4: list<float>, 5: list<float>|null, 6: bool}>
     */
    private static function problems(): array
    {
        $one = static fn (string $nd, mixed $v): object => is_object($v) ? $v : $nd::array([$v], 'float64');
        $p = [];
        $x = array_map(static fn (int $i): float => $i * 0.5, range(0, 11));
        $y = [];
        foreach ($x as $i => $v) {
            $y[] = 3.0 / (1 + 0.7 * $v) + 0.2 + 0.01 * (($i * 7) % 5 - 2);
        }
        $rat = static fn (float $x, float $a, float $b, float $c): float => $a / (1 + $b * $x) + $c;
        $ratA = static fn (string $nd): \Closure => static fn ($x, $a, $b, $c) => $one($nd, $a)->div($x->mul($b)->add(1.0))->add($c);
        $p['rational_12'] = [$rat, $ratA, $x, $y, [1.0, 1.0, 0.0], null, false];
        $p['rational_12_sigma_abs'] = [$rat, $ratA, $x, $y, [1.0, 1.0, 0.0], array_map(static fn (float $v): float => 0.1 + 0.01 * $v, $x), true];
        $x = array_map(static fn (int $i): float => 0.5 + $i, range(0, 19));
        $y = [];
        foreach ($x as $i => $v) {
            $y[] = 4.0 * $v / (2.5 + $v) + 0.02 * (($i * 3) % 7 - 3) / 3.0;
        }
        $p['mm_20_scalar_sigma'] = [static fn (float $x, float $a, float $b): float => $a * $x / ($b + $x),
            static fn (string $nd): \Closure => static fn ($x, $a, $b) => $x->mul($a)->div($x->add($b)), $x, $y, [1.0, 1.0], [0.05], false];
        $x = array_map(static fn (int $i): float => $i * 0.1, range(0, 39));
        $y = [];
        foreach ($x as $i => $v) {
            $y[] = 1.0 - 0.5 * $v + 0.25 * $v * $v - 0.02 * $v * $v * $v + 0.001 * $v * $v * $v * $v + 0.01 * (($i * 5) % 3 - 1);
        }
        $p['poly5_40_sigma'] = [static fn (float $x, float $a, float $b, float $c, float $d, float $e): float => $a + $x * ($b + $x * ($c + $x * ($d + $x * $e))),
            static fn (string $nd): \Closure => static fn ($x, $a, $b, $c, $d, $e) => $x->mul($e)->add($d)->mul($x)->add($c)->mul($x)->add($b)->mul($x)->add($a),
            $x, $y, [1.0, 0.0, 0.0, 0.0, 0.0], array_map(static fn (float $v): float => 0.5 + 0.1 * $v, $x), false];
        $x = array_map(static fn (int $i): float => $i * 0.1, range(0, 9));
        $y = [];
        foreach ($x as $i => $v) {
            $y[] = 2.0 * $v + 1.0 + 0.01 * (($i * 3) % 4 - 1.5);
        }
        $p['dead_param_10'] = [static fn (float $x, float $a, float $b, float $c): float => $a * $x + $b + 0.0 * $c,
            static fn (string $nd): \Closure => static fn ($x, $a, $b, $c) => $x->mul($a)->add($b)->add(is_object($c) ? $c->mul(0.0) : 0.0 * $c),
            $x, $y, [1.0, 1.0, 0.0], null, false];
        $p['line_6'] = [static fn (float $x, float $a, float $b): float => $a * $x + $b,
            static fn (string $nd): \Closure => static fn ($x, $a, $b) => $x->mul($a)->add($b),
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0], [1.2, 3.1, 5.0, 6.9, 9.1, 11.0], [0.0, 0.0], null, false];

        return $p;
    }

    /** @return iterable<string, array{string, string}> */
    public static function cases(): iterable
    {
        $modes = ['per point' => 'point', 'FFI arrays' => NDArray::class];
        if (class_exists(\Tessero\Ext\NDArray::class, false)) {
            $modes['extension arrays'] = \Tessero\Ext\NDArray::class;
        }
        foreach (array_keys(self::SCIPY) as $name) {
            foreach ($modes as $label => $mode) {
                yield "{$name}, {$label}" => [$name, $mode];
            }
        }
    }

    private static function hex(float $v): string
    {
        return bin2hex(pack('E', $v));
    }

    #[DataProvider('cases')]
    public function testTheFitIsScipysBitForBit(string $name, string $mode): void
    {
        [$point, $array, $x, $y, $p0, $sigma, $abs] = self::problems()[$name];
        [$popt, $pcov, $nfev, $ier, $fvecSha1] = self::SCIPY[$name];
        $f = $mode === 'point' ? $point : $array($mode);
        $xd = $mode === 'point' ? $x : $mode::array($x, 'float64');
        $fit = LeastSquares::curveFit($f, $xd, $y, $p0, $sigma, $abs, true);
        $this->assertSame($popt, array_map(self::hex(...), $fit[0]), 'popt');
        $this->assertSame($nfev, $fit[2]['nfev'], 'nfev');
        $this->assertSame($ier, $fit[4], 'ier');
        $this->assertSame($fvecSha1, sha1(pack('e*', ...$fit[2]['fvec'])), 'fvec');
        // pcov: leastsq's inverse of R^T R. SciPy forms it in its BLAS/LAPACK (dtrtri, then invR @ invR.T), whose
        // summation order depends on the build; the kernel uses one fixed order. Within 1 ulp of SciPy's.
        $bits = static fn (float $v): int => unpack('q', pack('d', $v))[1];
        foreach (array_merge(...$fit[1]) as $k => $v) {
            $ref = unpack('E', hex2bin($pcov[$k]))[1];
            $this->assertTrue(self::hex($v) === $pcov[$k] || abs($bits($v) - $bits($ref)) <= 1, "pcov[{$k}]: " . self::hex($v) . " against SciPy's {$pcov[$k]}");
        }
        // the same fit again: the same bits
        $this->assertSame(serialize($fit), serialize(LeastSquares::curveFit($f, $xd, $y, $p0, $sigma, $abs, true)), 'a second run');
    }

    public function testTheJacobianInOneCallIsKeptOnlyWhenItHasTheBitsOfTheSeparateCalls(): void
    {
        [, $array, $x, $y, $p0] = self::problems()['rational_12'];
        $model = $array(NDArray::class);
        $calls = ['floats' => 0, 'arrays' => 0];
        $counted = static function ($x, ...$p) use ($model, &$calls) {
            $calls[is_object($p[0]) ? 'arrays' : 'floats']++;

            return $model($x, ...$p);
        };
        $xd = NDArray::array($x, 'float64');
        [$ref] = LeastSquares::curveFit($counted, $xd, $y, $p0);
        $this->assertGreaterThan(1, $calls['arrays'], 'a model that broadcasts gets the Jacobian in one call');
        // a model that refuses arrays as parameters (PHP's unary minus on an object) is called per evaluation
        $floatsOnly = static fn ($x, float $a, float $b, float $c) => $model($x, $a, $b, $c);
        $this->assertSame($ref, LeastSquares::curveFit($floatsOnly, $xd, $y, $p0)[0]);
        // a model whose one call gives other values is caught by the comparison and called per evaluation
        $odd = static function ($x, ...$p) use ($model) {
            $v = $model($x, ...$p);

            return is_object($p[0]) ? $v->add(1e-9) : $v;
        };
        $this->assertSame($ref, LeastSquares::curveFit($odd, $xd, $y, $p0)[0]);
    }

    public function testATamperedStateIsRefusedNotRead(): void
    {
        $out = array_values(Linalg::lmdif(null, null, false, [1.0, 1.0], [1.0, 2.0, 3.0]));
        $this->assertSame(1, $out[1]);
        $state = $out[0]->toList();
        $state[2] = 1e9;                                       // n
        $this->expectException(\ValueError::class);
        Linalg::lmdif($state, [1.0, 2.0, 3.0]);
    }

    public function testMoreParametersThanPointsIsRefused(): void
    {
        $this->expectException(TesseroException::class);
        LeastSquares::curveFit(static fn (float $x, float $a, float $b, float $c): float => $a + $b * $x + $c, [0.0, 1.0], [1.0, 2.0], [1.0, 1.0, 1.0]);
    }
}
