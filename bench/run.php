<?php

/**
 * Tessero micro-benchmarks with regression gates.
 *
 *   php -d ffi.enable=1 bench/run.php [--json] [--no-gates]
 *
 * Each case reports the median of N runs. Gates are deliberately loose
 * (CI machines are noisy): they catch order-of-magnitude regressions such as
 * a kernel falling back to a scalar loop, not 10% drift. bench/numpy_bench.py
 * prints the same cases for NumPy on the same machine.
 */

declare(strict_types=1);

require __DIR__ . '/../tests/autoload.php';

use Tessero\Fft\Fft;
use Tessero\Linalg\Linalg;
use Tessero\NDArray;
use Tessero\Mdp\MarkovDecisionProcess;
use Tessero\Native\Blas;
use Tessero\Optimize\LinearProgramming;
use Tessero\Random\Generator;
use Tessero\Tessero;

$json = in_array('--json', $argv, true);
$gates = ! in_array('--no-gates', $argv, true);

function median_ms(callable $fn, int $runs = 15): float
{
    $fn(); // warm-up
    $t = [];
    for ($i = 0; $i < $runs; $i++) {
        $s = hrtime(true);
        $fn();
        $t[] = (hrtime(true) - $s) / 1e6;
    }
    sort($t);

    return $t[intdiv(count($t), 2)];
}

$rng = Generator::defaultRng(0);
$a = $rng->random(1_000_000);
$b = $rng->random(1_000_000);
$out = NDArray::empty([1_000_000]);
$m600 = $rng->random([600, 600]);
$m2 = $rng->random([600, 600]);
$big2d = $rng->random([1000, 1000]);
$spd = $m600->matmul($m600->t())->add(NDArray::eye(600)->mul(600));
$sig = $rng->random(1 << 20);
$phpA = $a->toList();

// LP: 200 x 200 dense, feasible and bounded (random positive A, b)
$lpA = $rng->uniform(0.1, 1.0, [200, 200])->toArray();
$lpB = $rng->uniform(50, 100, 200)->toList();
$lpC = $rng->uniform(-1, 0, 200)->toList();
// MDP: 20 000 states x 4 actions, 5 successors per (s, a)
$mdpS = 20_000;
$mdpA = 4;
$next = $rng->integers(0, $mdpS, [$mdpA * $mdpS, 5])->toArray();
$rew = $rng->random([$mdpS, $mdpA])->toArray();
$tr = [];
foreach ($next as $row => $succ) {
    foreach ($succ as $s2) {
        $tr[] = [$row % $mdpS, intdiv($row, $mdpS), $s2, 0.2];
    }
}
$rw = [];
foreach ($rew as $s => $acts) {
    foreach ($acts as $act => $r) {
        $rw[] = [$s, $act, $r];
    }
}
$mdp = MarkovDecisionProcess::fromTransitions($mdpS, $mdpA, $tr, $rw);
unset($tr, $rw, $next, $rew);

$cases = [
    // name => [callable, gate_ms]
    'add 1M f64 (new array)' => [fn () => $a->add($b), 6.0],
    'add 1M f64 (out=)' => [fn () => $a->add($b, out: $out), 4.0],
    'mul scalar 1M' => [fn () => $a->mul(2.5), 6.0],
    'sqrt 1M' => [fn () => $a->sqrt(), 8.0],
    'exp 1M' => [fn () => $a->exp(), 20.0],
    'log 1M' => [fn () => $a->log(), 20.0],
    'sum 1M (pairwise)' => [fn () => $a->sum(), 3.0],
    'sum axis 0 1000x1000' => [fn () => $big2d->sum(0), 15.0],
    'transpose copy 1000x1000' => [fn () => $big2d->t()->copy(), 20.0],
    'boolean mask 1M' => [fn () => $a->filter($a->gt(0.5)), 15.0],
    'sort 1M' => [fn () => $a->sort(), 120.0],
    'matmul 600x600' => [fn () => $m600->matmul($m2), Blas::available() ? 40.0 : 400.0],
    'solve 600x600' => [fn () => Linalg::solve($m600, $b->slice('0:600')), 60.0],
    'cholesky 600x600' => [fn () => Linalg::cholesky($spd), 40.0],
    'svd 300x300' => [fn () => Linalg::svd($m600->slice('0:300', '0:300')), 400.0],
    'fft 2^20 complex' => [fn () => Fft::fft($sig), 200.0],
    'rfft 2^20 real' => [fn () => Fft::rfft($sig), 150.0],
    'fft 1_000_003 (prime, Bluestein)' => [fn () => Fft::fft($sig->slice('0:1000003')), 1500.0],
    'rng normal 1M' => [fn () => $rng->normal(size: 1_000_000), 20.0],
    'PHP list -> NDArray 1M' => [fn () => NDArray::fromFlat($phpA, [1_000_000]), 120.0],
    'NDArray -> PHP list 1M' => [fn () => $a->toList(), 150.0],
    'linprog 200x200 dense' => [fn () => LinearProgramming::linprog($lpC, $lpA, $lpB), 150.0],
    'MDP value iteration 20k x 4' => [fn () => $mdp->valueIteration(0.95, 1e-6), 2500.0],
    'MDP modified PI 20k x 4' => [fn () => $mdp->policyIteration(0.95, evalSweeps: 20), 1500.0],
];

$results = [];
$failed = [];
foreach ($cases as $name => [$fn, $gate]) {
    $ms = median_ms($fn, str_contains($name, 'prime') || str_contains($name, 'sort') || str_contains($name, 'MDP') ? 5 : 15);
    $results[$name] = round($ms, 3);
    if ($gates && $ms > $gate) {
        $failed[] = sprintf('%s: %.2f ms > gate %.1f ms', $name, $ms, $gate);
    }
}

if ($json) {
    echo json_encode(['info' => Tessero::info(), 'ms' => $results], JSON_PRETTY_PRINT), "\n";
} else {
    $info = Tessero::info();
    printf("Tessero %s | %s | SIMD %s | BLAS %s | PHP %s%s\n\n", $info['tessero'], $info['platform'], $info['simd'], $info['blas'] ? basename($info['blas']) : 'none', PHP_VERSION, $info['jit'] ? ' (JIT)' : '');
    foreach ($results as $name => $ms) {
        printf("  %-36s %10.3f ms\n", $name, $ms);
    }
}
if ($failed !== []) {
    fwrite(STDERR, "\nBenchmark gates failed:\n  " . implode("\n  ", $failed) . "\n");
    exit(1);
}
