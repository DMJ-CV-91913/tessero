<?php

declare(strict_types=1);

namespace Tessero\Tests\Unit;

use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\TestCase;
use Tessero\Exceptions\TesseroException;
use Tessero\Linalg;
use Tessero\Optimize\LeastSquares;

/**
 * solve runs MINPACK's lmdif (lmder with a Jacobian) as scipy.optimize.leastsq does, leastSquares MINPACK's lmder as
 * least_squares with method 'lm' does (linalg.lmdif). The residuals are built from + - * / alone, so their doubles
 * are the same in PHP and in NumPy on every platform, and the results are compared with SciPy's bit for bit.
 */
final class LeastSquaresMinpackTest extends TestCase
{
    /**
     * SciPy 1.17.1 (NumPy 2.4.4): leastsq(full_output=True), with and without Dfun, and least_squares(method='lm'),
     * with and without jac, on the problems below. x and cov_x (row-major) as big-endian hex, the residuals and
     * least_squares' jac as the SHA-1 of their little-endian doubles (C order).
     */
    private const LEASTSQ = [
        'rational_12' => [['40080aa9df520c42', '3fe5a7348d46a84d', '3fc6987b3d1f0be1'], 30, 1, '16bc7c9a3cc2b3023e26d0e62976431c10f4a1c4',
            ['400210cdd5a4ae94', 'bff020898cc5bbc2', 'bffd8693a05ab998', 'bff020898cc5bbc2', '3ff7b548f9ad596f', '3ffb507dda53152c', 'bffd8693a05ab998', '3ffb507dda53152c', '4002c832ac456f5d']],
        'rosenbrock' => [['3ff0000000000000', '3ff0000000000000'], 54, 2, 'e129f27c5103bc5cc44bcdf0a15e160d445066ff',
            ['3feffffffffffffc', '4000000002013c7a', '4000000002013c7a', '40100a3d74a65197']],
        'poly5_40' => [['3fefea30e1e6b72f', 'bfdf4737746bbc43', '3fce64c1070d4c76', 'bf8e96fd12ee5d09', '3f35afc4d068da1e'], 13, 1, '17dd3a1d98996ce84ea06f9582fb42009b6cd9b8',
            ['3fde11dd0258d281', 'bff6be91c53c3d8f', '3ff41f8892097eb6', 'bfdb284ea6b173ab', '3fa8c045ad154a64', 'bff6be91c53c3d8f', '4019051a25d9e116', 'c0199ef0d353fd4c', '4002b461e3a1fabd', 'bfd1e8d659165451', '3ff41f8892097eb6', 'c0199ef0d353fd4c', '401c39d71b512f36', 'c0058dd8d74c90f7', '3fd5477a0dd51754', 'bfdb284ea6b173ab', '4002b461e3a1fabd', 'c0058dd8d74c90f7', '3ff0f67eaacad361', 'bfc11d69139d4f6e', '3fa8c045ad154a64', 'bfd1e8d659165451', '3fd5477a0dd51754', 'bfc11d69139d4f6e', '3f918dc047a2bf56']],
    ];

    /** leastsq with Dfun: x, nfev, njev, ier, residuals, cov_x */
    private const LEASTSQ_DFUN = [
        'rational_12' => [['40080aa9df9a3b2a', '3fe5a7348c326678', '3fc6987b3744db39'], 9, 7, 1, 'eff2c36d1f4f0ef84708d27e4f0936cefa007425',
            ['400210cda8496207', 'bff02089a1fc65f5', 'bffd869396800108', 'bff02089a1fc65f5', '3ff7b5495d69fcb6', '3ffb507e49d8771f', 'bffd869396800108', '3ffb507e49d8771f', '4002c832e3ecc0ae']],
        'rosenbrock' => [['3ff0000000000000', '3ff0000000000000'], 21, 16, 4, 'e129f27c5103bc5cc44bcdf0a15e160d445066ff',
            ['3feffffffffffffe', '3fffffffffffffff', '3fffffffffffffff', '40100a3d70a3d70b']],
    ];

    /** least_squares(method='lm'): x, nfev, status, jac, cost, message */
    private const LM = [
        'rational_12' => [['40080aa9df96db46', '3fe5a7348c20247b', '3fc6987b371dd5a5'], 9, 2, '05a66d2f1233886a7ad93f836d5604546e2d4c53', '3f4fca9fda97dbc7', '`ftol` termination condition is satisfied.'],
        'rosenbrock' => [['3ff0000000000000', '3ff0000000000000'], 21, 1, '5df80e82270258a595d34578e45960260381ccdf', '0000000000000000', '`gtol` termination condition is satisfied.'],
        'poly5_40' => [['3fefea30e2348df3', 'bfdf4737738a122d', '3fce64c1007344c8', 'bf8e96fcd4df28c1', '3f35afc3920b00e7'], 3, 2, 'c9e7007d1749443ebd0307f5ccb2d57378ff9dba', '3f55c2247311f5de', '`ftol` termination condition is satisfied.'],
    ];

    /** least_squares(method='lm', jac=...): x, nfev, njev, status, jac, cost */
    private const LM_JAC = [
        'rational_12' => [['40080aa9df9a3b2a', '3fe5a7348c326678', '3fc6987b3744db39'], 9, 7, 2, '4c41e4bc2ffff776c4d6fe7f9ec5fa61a9d3bc33', '3f4fca9fda97db7f'],
        'rosenbrock' => [['3ff0000000000000', '3ff0000000000000'], 21, 16, 1, '93f7b7a4cffab72726f316ae988d74fcb9c79d29', '0000000000000000'],
    ];

    /**
     * name => [residuals, Jacobian or null, x0], the same operations in the same order as the SciPy script's.
     *
     * @return array<string, array{0: callable, 1: callable|null, 2: list<float>}>
     */
    private static function problems(): array
    {
        $x12 = array_map(static fn (int $i): float => $i * 0.5, range(0, 11));
        $y12 = [];
        foreach ($x12 as $i => $v) {
            $y12[] = 3.0 / (1 + 0.7 * $v) + 0.2 + 0.01 * (($i * 7) % 5 - 2);
        }
        $rat = static function (array $p) use ($x12, $y12): array {
            $out = [];
            foreach ($x12 as $i => $x) {
                $out[] = $p[0] / (1 + $p[1] * $x) + $p[2] - $y12[$i];
            }

            return $out;
        };
        $ratJac = static function (array $p) use ($x12): array {
            $out = [];
            foreach ($x12 as $x) {
                $d = 1 + $p[1] * $x;
                $out[] = [1.0 / $d, -($p[0] * $x) / ($d * $d), 1.0];
            }

            return $out;
        };
        $x40 = array_map(static fn (int $i): float => $i * 0.1, range(0, 39));
        $y40 = [];
        foreach ($x40 as $i => $v) {
            $y40[] = 1.0 - 0.5 * $v + 0.25 * $v * $v - 0.02 * $v * $v * $v + 0.001 * $v * $v * $v * $v + 0.01 * (($i * 5) % 3 - 1);
        }
        $poly = static function (array $p) use ($x40, $y40): array {
            $out = [];
            foreach ($x40 as $i => $x) {
                $out[] = $p[0] + $x * ($p[1] + $x * ($p[2] + $x * ($p[3] + $x * $p[4]))) - $y40[$i];
            }

            return $out;
        };

        return [
            'rational_12' => [$rat, $ratJac, [1.0, 1.0, 0.0]],
            'rosenbrock' => [static fn (array $p): array => [10.0 * ($p[1] - $p[0] * $p[0]), 1.0 - $p[0]],
                static fn (array $p): array => [[-20.0 * $p[0], 10.0], [-1.0, 0.0]], [-1.2, 1.0]],
            'poly5_40' => [$poly, null, [1.0, 0.0, 0.0, 0.0, 0.0]],
        ];
    }

    private static function hex(float $v): string
    {
        return bin2hex(pack('E', $v));
    }

    /** @param list<float> $v */
    private static function sha(array $v): string
    {
        return sha1(pack('e*', ...$v));
    }

    /**
     * cov_x is the inverse of R^T R. SciPy forms it in its BLAS/LAPACK (dtrtri, then invR @ invR.T), whose summation
     * order depends on the build; the kernel uses one fixed order. Within 1 ulp of SciPy's.
     *
     * @param list<list<float>> $cov
     * @param list<string> $ref
     */
    private function assertCovWithinAnUlp(array $cov, array $ref): void
    {
        $bits = static fn (float $v): int => unpack('q', pack('d', $v))[1];
        foreach (array_merge(...$cov) as $k => $v) {
            $r = unpack('E', hex2bin($ref[$k]))[1];
            $this->assertTrue(self::hex($v) === $ref[$k] || abs($bits($v) - $bits($r)) <= 1, "cov_x[{$k}]: " . self::hex($v) . " against SciPy's {$ref[$k]}");
        }
    }

    /** @return iterable<string, array{string}> */
    public static function leastsqCases(): iterable
    {
        foreach (array_keys(self::LEASTSQ) as $name) {
            yield $name => [$name];
        }
    }

    /** @return iterable<string, array{string}> */
    public static function jacobianCases(): iterable
    {
        foreach (array_keys(self::LEASTSQ_DFUN) as $name) {
            yield $name => [$name];
        }
    }

    #[DataProvider('leastsqCases')]
    public function testSolveIsLeastsqBitForBit(string $name): void
    {
        [$f, , $x0] = self::problems()[$name];
        [$x, $nfev, $ier, $fvec, $cov] = self::LEASTSQ[$name];
        $r = LeastSquares::solve($f, $x0);
        $this->assertSame($x, array_map(self::hex(...), $r['x']), 'x');
        $this->assertSame($nfev, $r['nfev'], 'nfev');
        $this->assertSame($ier, $r['status'], 'ier');
        $this->assertTrue($r['success']);
        $this->assertSame($fvec, self::sha($r['fun']), 'residuals');
        $this->assertCovWithinAnUlp($r['cov_x'], $cov);
        $this->assertSame(serialize($r), serialize(LeastSquares::solve($f, $x0)), 'a second run');
    }

    #[DataProvider('jacobianCases')]
    public function testSolveWithAJacobianIsLeastsqWithDfunBitForBit(string $name): void
    {
        [$f, $jac, $x0] = self::problems()[$name];
        [$x, $nfev, $njev, $ier, $fvec, $cov] = self::LEASTSQ_DFUN[$name];
        $r = LeastSquares::solve($f, $x0, $jac);
        $this->assertSame($x, array_map(self::hex(...), $r['x']), 'x');
        $this->assertSame([$nfev, $njev, $ier], [$r['nfev'], $r['njev'], $r['status']], 'nfev, njev, ier');
        $this->assertSame($fvec, self::sha($r['fun']), 'residuals');
        $this->assertCovWithinAnUlp($r['cov_x'], $cov);
    }

    #[DataProvider('leastsqCases')]
    public function testLeastSquaresIsMethodLmBitForBit(string $name): void
    {
        [$f, , $x0] = self::problems()[$name];
        [$x, $nfev, $status, $jac, $cost, $message] = self::LM[$name];
        $r = LeastSquares::leastSquares($f, $x0);
        $this->assertSame($x, array_map(self::hex(...), $r->x), 'x');
        $this->assertSame([$nfev, 0, $status, $message], [$r->nfev, $r->njev, $r->status, $r->message]);
        $this->assertTrue($r->success);
        // jac: approx_derivative's 2-point differences at x, the same bits
        $this->assertSame($jac, self::sha(array_merge(...$r->jac)), 'jac');
        // cost: SciPy's 0.5 * np.dot(f, f) sums in its BLAS, the kernel pairwise; within 1 ulp
        $ref = unpack('E', hex2bin($cost))[1];
        $this->assertEqualsWithDelta($ref, $r->fun, abs($ref) * PHP_FLOAT_EPSILON, 'cost');
    }

    #[DataProvider('jacobianCases')]
    public function testLeastSquaresWithAJacobianIsMethodLmBitForBit(string $name): void
    {
        [$f, $jacf, $x0] = self::problems()[$name];
        [$x, $nfev, $njev, $status, $jac, $cost] = self::LM_JAC[$name];
        $r = LeastSquares::leastSquares($f, $x0, $jacf);
        $this->assertSame($x, array_map(self::hex(...), $r->x), 'x');
        $this->assertSame([$nfev, $njev, $status], [$r->nfev, $r->njev, $r->status]);
        $this->assertSame($jac, self::sha(array_merge(...$r->jac)), 'jac');
        $ref = unpack('E', hex2bin($cost))[1];
        $this->assertEqualsWithDelta($ref, $r->fun, abs($ref) * PHP_FLOAT_EPSILON, 'cost');
    }

    public function testTheKernelsTwoPointDifferencesAreThoseOfThePhpLoop(): void
    {
        // leastSquares without jac differences in the kernel (linalg.lmdif jac 2); the same differences written in
        // PHP and passed as jac give the same iterations, bit for bit
        [$f, , $x0] = self::problems()['poly5_40'];
        $php = static function (array $x) use ($f): array {
            $f0 = $f($x);
            $J = array_fill(0, count($f0), []);
            foreach ($x as $j => $xj) {
                $x1 = $x;
                $x1[$j] = $xj + (sqrt(PHP_FLOAT_EPSILON) * ($xj >= 0.0 ? 1.0 : -1.0)) * max(1.0, abs($xj));
                $dx = $x1[$j] - $xj;
                foreach ($f($x1) as $i => $v) {
                    $J[$i][] = ($v - $f0[$i]) / $dx;
                }
            }

            return $J;
        };
        $kernel = LeastSquares::leastSquares($f, $x0);
        $loop = LeastSquares::leastSquares($f, $x0, $php);
        $this->assertSame([$kernel->x, $kernel->fun, $kernel->nfev, $kernel->status, $kernel->jac], [$loop->x, $loop->fun, $loop->nfev, $loop->status, $loop->jac]);
    }

    public function testAJacobianStateWaitsForTheJacobianAtOnePoint(): void
    {
        $out = array_values(Linalg::lmdif(null, null, false, [1.0, 1.0], null, null, 1.49012e-8, 1.49012e-8, 0.0, 0, 0.0, 100.0, 1));
        $this->assertSame(1, $out[1]);
        $out = array_values(Linalg::lmdif($out[0], [1.0, 2.0, 3.0]));
        $this->assertSame(-1, $out[1], 'task -1: the Jacobian');
        $this->assertSame([[1.0, 1.0]], $out[2]->toArray());
        $this->expectException(\ValueError::class);
        Linalg::lmdif($out[0], [1.0, 2.0, 3.0]);                  // three values where the 3 x 2 Jacobian is due
    }

    public function testFewerResidualsThanParametersAndTinyTolerancesAreRefused(): void
    {
        try {
            LeastSquares::solve(static fn (array $p): array => [$p[0] + $p[1]], [1.0, 1.0]);
            $this->fail('one residual for two parameters');
        } catch (TesseroException) {
            $this->addToAssertionCount(1);
        }
        $this->expectException(TesseroException::class);
        LeastSquares::leastSquares(static fn (array $p): array => [$p[0], $p[0] - 1.0], [0.0], ftol: 1e-20);
    }
}
