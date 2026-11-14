<?php

declare(strict_types=1);

namespace Tessero\Tests\Unit;

use PHPUnit\Framework\TestCase;

/**
 * eigsh on CSR (thick-restart Lanczos with a search of the complement for missed repeats) against eigenvalues known in
 * closed form: the 2-D Poisson matrix, and two identical blocks, whose doubled eigenvalues a Krylov space from one
 * start vector cannot hold twice (the complement search must find each second copy). The same call gives the same
 * bytes every time, and on both backends.
 */
final class EigshLanczosTest extends TestCase
{
    /** @return list<array{class-string, class-string}> [Sparse, SparseLinalg] of each backend */
    private static function backends(): array
    {
        $b = [[\Tessero\Sparse::class, \Tessero\SparseLinalg::class]];
        if (extension_loaded('tessero') && class_exists(\Tessero\Ext\SparseLinalg::class)) {
            $b[] = [\Tessero\Ext\Sparse::class, \Tessero\Ext\SparseLinalg::class];
        }

        return $b;
    }

    /** the 2-D Poisson matrix on a g x g grid (Dirichlet): 4 on the diagonal, -1 to the four neighbours */
    private static function poisson(string $sparse, int $g): mixed
    {
        $n = $g * $g;
        $off1 = array_map(static fn (int $i): float => ($i + 1) % $g === 0 ? 0.0 : -1.0, range(0, $n - 2));

        return $sparse::diags([array_fill(0, $n - $g, -1.0), $off1, array_fill(0, $n, 4.0), $off1, array_fill(0, $n - $g, -1.0)], [-$g, -1, 0, 1, $g], [$n, $n]);
    }

    /** two copies of tridiag(-1, 2, -1) of size b on the diagonal: every eigenvalue 2 - 2 cos(i pi / (b + 1)) twice */
    private static function twoBlocks(string $sparse, int $b): mixed
    {
        $off = array_map(static fn (int $i): float => $i === $b - 1 ? 0.0 : -1.0, range(0, 2 * $b - 2));

        return $sparse::diags([$off, array_fill(0, 2 * $b, 2.0), $off], [-1, 0, 1], [2 * $b, 2 * $b]);
    }

    /** @return list<float> */
    private static function values(mixed $w): array
    {
        return (is_array($w) ? reset($w) : $w)->toList();
    }

    /** @param list<float> $want */
    private function assertEigen(array $want, array $got, string $what): void
    {
        sort($want);
        $this->assertCount(count($want), $got, $what);
        foreach ($want as $i => $v) {
            $this->assertEqualsWithDelta($v, $got[$i], 1e-12 * (1.0 + abs($v)), "{$what}: eigenvalue {$i}");
        }
    }

    public function testPoissonEigenvaluesMatchTheClosedForm(): void
    {
        $g = 30;
        $all = [];
        for ($i = 1; $i <= $g; $i++) {
            for ($j = 1; $j <= $g; $j++) {
                $all[] = 4.0 - 2.0 * cos($i * M_PI / ($g + 1)) - 2.0 * cos($j * M_PI / ($g + 1));
            }
        }
        sort($all);
        foreach (self::backends() as [$sparse, $solvers]) {
            $A = self::poisson($sparse, $g);
            // the top of this spectrum holds pairs (i, j) and (j, i): repeated eigenvalues
            $this->assertEigen(array_slice($all, -6), self::values($solvers::eigsh($A, 6, 'LA')), "$solvers LA");
            $this->assertEigen(array_slice($all, -6), self::values($solvers::eigsh($A, 6, 'LM')), "$solvers LM");
            $this->assertEigen(array_slice($all, 0, 6), self::values($solvers::eigsh($A, 6, 'SA')), "$solvers SA");
        }
    }

    public function testEachSecondCopyOfARepeatedEigenvalueIsFound(): void
    {
        $b = 300;
        $lam = static fn (int $i): float => 2.0 - 2.0 * cos($i * M_PI / ($b + 1));
        foreach (self::backends() as [$sparse, $solvers]) {
            $A = self::twoBlocks($sparse, $b);
            $this->assertEigen([$lam($b), $lam($b), $lam($b - 1), $lam($b - 1)], self::values($solvers::eigsh($A, 4, 'LA')), "$solvers LA, doubled");
            $this->assertEigen([$lam(1), $lam(1), $lam(2), $lam(2), $lam(3)], self::values($solvers::eigsh($A, 5, 'SA')), "$solvers SA, doubled");
        }
    }

    public function testAStartVectorBlindToABlockStillGivesEverySecondCopy(): void
    {
        // v0 zero on the second block: the blocks are uncoupled, so no rounding brings that block into the Krylov space,
        // and the main run sees each eigenvalue once. Every second copy must come from the complement search.
        $b = 200;
        $lam = static fn (int $i): float => 2.0 - 2.0 * cos($i * M_PI / ($b + 1));
        $v0 = array_merge(array_map(static fn (int $i): float => 1.0 + 0.25 * sin(0.7 * $i), range(0, $b - 1)), array_fill(0, $b, 0.0));
        foreach (self::backends() as [$sparse, $solvers]) {
            $array = str_replace('Sparse', 'NDArray', $sparse);
            $A = self::twoBlocks($sparse, $b);
            $w = self::values($solvers::eigsh($A, 4, 'LA', null, $array::array($v0, 'float64')));
            $this->assertEigen([$lam($b), $lam($b), $lam($b - 1), $lam($b - 1)], $w, "$solvers LA, v0 blind to block 2");
        }
    }

    public function testTheSameCallGivesTheSameBytesEveryTimeAndOnBothBackends(): void
    {
        $ref = [];
        foreach (self::backends() as $k => [$sparse, $solvers]) {
            foreach (['poisson' => self::poisson($sparse, 30), 'blocks' => self::twoBlocks($sparse, 300)] as $name => $A) {
                $first = pack('e*', ...self::values($solvers::eigsh($A, 6, 'LA')));
                $this->assertSame($first, pack('e*', ...self::values($solvers::eigsh($A, 6, 'LA'))), "$solvers $name: a second run");
                $ref[$name] ??= $first;
                $this->assertSame($ref[$name], $first, "$solvers $name: the same bytes as the FFI backend");
            }
        }
    }
}
