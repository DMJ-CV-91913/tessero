<?php

/**
 * Executes the examples printed in docs/ and checks the results the docs
 * claim, so the manuals cannot drift from the code. CI runs it in the docs job:
 *
 *   php -d ffi.enable=1 [-d extension=tessero] tests/docs/examples.php
 */

declare(strict_types=1);

require __DIR__ . '/../autoload.php';

use Tessero\Fft\Fft;
use Tessero\Linalg\Linalg;
use Tessero\Mdp\MarkovDecisionProcess;
use Tessero\Optimize\LinearProgramming as LP;
use Tessero\Random\Generator;

use function Tessero\{arange, arr, linspace};

$fail = 0;
$n = 0;
function check(string $what, bool $ok): void
{
    global $fail, $n;
    $n++;
    if (! $ok) {
        $fail++;
        fwrite(STDERR, "FAIL: {$what}\n");
    }
}
function close(array|float $a, array|float $b, float $tol = 1e-9): bool
{
    if (is_float($a) || is_int($a)) {
        return abs($a - $b) <= $tol * max(1.0, abs($b));
    }
    foreach ($a as $i => $v) {
        if (! close(is_array($v) ? $v : (float) $v, is_array($b[$i]) ? $b[$i] : (float) $b[$i], $tol)) {
            return false;
        }
    }

    return count($a) === count($b);
}

// ---- quickstart: arrays
$a = arr([[1, 2, 3], [4, 5, 6]]);
$b = arr([0.5, 1.5, 2.5]);
check('shape', $a->shape() === [2, 3]);
check('dtype', $a->dtype()->name() === 'int64');
check('broadcast mul', $a->mul($b)->toArray() === [[0.5, 3.0, 7.5], [2.0, 7.5, 15.0]]);
check('sum', $a->sum() === 21);
check('sum axis', $a->sum(axis: 0)->toList() === [5, 7, 9]);
check('keepdims', $a->mean(axis: 1, keepdims: true)->shape() === [2, 1]);

// ---- quickstart: slicing
$m = arange(12)->reshape(3, 4);
check('slice', $m['1:, ::2']->toArray() === [[4, 6], [8, 10]]);
check('ellipsis', $m['..., -1']->toList() === [3, 7, 11]);
check('reverse', $m['::-1']->toArray()[0] === [8, 9, 10, 11]);
$m[$m->gt(8)] = 0;
check('mask assign', $m->toArray()[2] === [8, 0, 0, 0]);

// ---- quickstart: lstsq, fft
$rng = Generator::defaultRng(42);
$X = $rng->normal(size: [200, 3]);
$y = $X->matmul(arr([1.5, -2.0, 0.5]))->add($rng->normal(0, 0.1, 200));
[$beta] = Linalg::lstsq($X, $y);
check('lstsq', close($beta->toList(), [1.5, -2.0, 0.5], 0.05));
$spec = Fft::rfft(linspace(0, 1, 1024)->mul(2 * M_PI * 50)->sin())->abs();
check('rfft peak at 50', $spec->argmax() === 50);

// ---- quickstart + guide: LP
$r = LP::linprog([-40, -30], [[2, 1], [1, 1]], [100, 80]);
check('lp x', close($r->x, [20.0, 60.0]));
check('lp fun', close($r->fun, -2600.0));
check('lp duals', close($r->ineqlinMarginals, [-10.0, -20.0]));

// ---- quickstart: MDP
$P = [[[0.9, 0.1], [0.0, 1.0]], [[1.0, 0.0], [1.0, 0.0]]];
$R = [[10, -5], [0, -5]];
$sol = MarkovDecisionProcess::fromDense($P, $R)->policyIteration(gamma: 0.95);
check('mdp policy', $sol->policy->toList() === [0, 1]);

// ---- guide/linear-programming: facility location MILP
$fixed = [12, 10, 14];
$serve = [[4, 6, 9], [5, 4, 7], [6, 3, 4]];   // serve[i][j]: cost of serving customer j from site i
// variables: open_0..2, assign_ij (9)  -> 12 variables
$c = array_merge($fixed, array_merge(...$serve));
$Aeq = [];
$beq = [];
for ($j = 0; $j < 3; $j++) {                   // each customer served exactly once
    $row = array_fill(0, 12, 0);
    for ($i = 0; $i < 3; $i++) {
        $row[3 + 3 * $i + $j] = 1;
    }
    $Aeq[] = $row;
    $beq[] = 1;
}
$Aub = [];
$bub = [];
for ($i = 0; $i < 3; $i++) {                   // assign_ij <= open_i
    for ($j = 0; $j < 3; $j++) {
        $row = array_fill(0, 12, 0);
        $row[3 + 3 * $i + $j] = 1;
        $row[$i] = -1;
        $Aub[] = $row;
        $bub[] = 0;
    }
}
$mr = LP::milp($c, true, $Aub, $bub, $Aeq, $beq, [0, 1]);
check('milp success', $mr->success);
check('milp open', array_map('intval', array_map('round', array_slice($mr->x, 0, 3))) === [0, 1, 0]);
check('milp cost', close($mr->fun, 26.0));

// ---- guide/mdp: inventory model via fromTransitions
$cap = 3;
$S = $cap + 1;
$A = $cap + 1;              // order 0..cap units
$demand = [0 => 0.3, 1 => 0.4, 2 => 0.3];
$T = [];
$Rw = [];
for ($s = 0; $s < $S; $s++) {
    for ($a = 0; $a < $A; $a++) {
        $stock = min($cap, $s + $a);
        $exp = -2.0 * $a - 1.0 * $stock;       // order cost 2/unit, holding 1/unit
        foreach ($demand as $d => $p) {
            $sold = min($stock, $d);
            $exp += $p * (8.0 * $sold);        // price 8
            $T[] = [$s, $a, $stock - $sold, $p];
        }
        $Rw[] = [$s, $a, $exp];
    }
}
$inv = MarkovDecisionProcess::fromTransitions($S, $A, $T, $Rw);
$vi = $inv->valueIteration(0.9, 1e-10);
$pi = $inv->policyIteration(0.9);
check('inventory vi==pi', $vi->policy->toList() === $pi->policy->toList());
check('inventory converged', $vi->converged && $pi->converged);
check('inventory values', close($vi->values->toList(), $pi->values->toList(), 1e-6));

// ---- extension examples
if (extension_loaded('tessero')) {
    $X = Tessero\Ext\NDArray::class;
    $p = $X::array([42.1, 39.8, 55.0, 61.2]);
    $q = $X::array([10, 12, 8, 9]);
    check('ext revenue', close(($p * $q)->sum(), 42.1 * 10 + 39.8 * 12 + 55.0 * 8 + 61.2 * 9));
    $norm = ($p - $p->mean()) / $p->std();
    check('ext json', json_encode($norm) === json_encode($norm->toList()));
    $plan = Tessero\Ext\Engine::linprog([-40, -30], [[2, 1], [1, 1]], [100, 80]);
    check('ext lp', close($plan['x'], [20.0, 60.0]) && close($plan['ineqlin'], [-10.0, -20.0]));
    $load = $X::array(range(1, 48));
    check('ext hour profile', $load->reshape(-1, 24)->mean(0)->shape() === [24]);
    $ffi = arr([1.0, 2.0]);
    $ext = Tessero\Ext\NDArray::fromBytes($ffi->toBytes(), $ffi->dtype()->name(), $ffi->shape());
    $back = Tessero\NDArray::fromBytes($ext->toBytes(), $ext->dtype(), $ext->shape());
    check('bytes round trip', $back->toList() === [1.0, 2.0] && $ext->toList() === [1.0, 2.0]);
    // ---- guide/ufuncs
    $M = Tessero\Ext\Math::class;
    $x = $X::array([[0.25, 1.0, 4.0], [9.0, 16.0, 25.0]]);
    check('ufunc sqrt', $M::sqrt($x)->toArray() === [[0.5, 1.0, 2.0], [3.0, 4.0, 5.0]]);
    check('ufunc broadcast', $M::hypot($x, $X::array([3.0, 4.0, 0.0]))->shape() === [2, 3]);
    check('ufunc cbrt', $M::cbrt($X::array([8, -27]))->toList() === [2.0, -3.0000000000000004]);   // numpy.cbrt (ADR 0012)
    $buf = $X::zeros([2, 3]);
    check('ufunc out', $M::exp($x, $buf) === $buf && $buf->toBytes() === $x->exp()->toBytes());
    check('ufunc apply', $M::apply('cbrt', $X::array([27.0]))->toList() === [3.0000000000000004]);
    check('ufunc catalogue', isset($M::ufuncs()['sin']['engine']));
    // ---- guide/memmap
    $mf = sys_get_temp_dir() . '/tessero-docs-' . getmypid() . '.f64';
    $cube = $X::memmap($mf, 'w+', [48, 3], 'float64');
    $cube['0:24'] = 1.0;
    $cube->flush();
    $cube->flush(sync: true);
    unset($cube);
    $lmp = $X::memmap($mf, 'r', [48, 3]);
    check('memmap view', $lmp[':, 1']->mean() === 0.5 && $lmp->isReadonly());
    check('memmap tail max', $lmp['-24:']->max(0)->toList() === [0.0, 0.0, 0.0]);
    $raw = $X::memmap($mf, 'r', null, 'float64', offset: 8 * 3 * 24);
    check('memmap inferred', $raw->shape() === [72] && $raw->sum() === 0.0);
    unset($lmp, $raw);
    unlink($mf);
    check('operand on ffi', ($ffi + 1)->toList() === [2.0, 3.0]);
}

// ---- guide/ufuncs, guide/memmap (.npy), guide/math: the same calls on each backend
function backend_examples(string $X, string $M, string $tag): void
{
    $x = $X::array([[0.25, 1.0, 4.0], [9.0, 16.0, 25.0]]);
    check("{$tag} ufunc sqrt", $M::sqrt($x)->toArray() === [[0.5, 1.0, 2.0], [3.0, 4.0, 5.0]]);
    check("{$tag} ufunc cbrt", $M::cbrt($X::array([8, -27]))->toList() === [2.0, -3.0000000000000004]);   // numpy.cbrt (ADR 0012)
    check("{$tag} ufunc fmax", $M::fmax($X::array([1.0, NAN]), $X::array([NAN, 2.0]))->toList() === [1.0, 2.0]);
    $buf = $X::zeros([2, 3]);
    check("{$tag} ufunc out", $M::exp($x, $buf) === $buf);
    check("{$tag} ufunc count", count($M::ufuncs()) === 78);
    // multi-axis reductions with keepdims
    $c = $X::arange(24.0)->reshape([2, 3, 4]);
    check("{$tag} sum axes", $c->sum([0, 2])->toList() === [60.0, 92.0, 124.0]);
    check("{$tag} keepdims", $c->sum([0, 2], true)->shape() === [1, 3, 1] && $c->mean(null, true)->shape() === [1, 1, 1]);
    // .npy
    $dir = sys_get_temp_dir();
    $f = "{$dir}/tessero-docs-{$tag}-" . getmypid() . '.npy';
    $X::arange(6.0)->reshape([2, 3])->save($f);
    check("{$tag} npy load", $X::load($f)->toArray() === [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
    $m = $X::load($f, 'r');
    check("{$tag} npy mmap", $m->isMemmap() && $m->isReadonly() && $m->shape() === [2, 3]);
    unset($m);
    $o = $X::openMemmap($f, 'w+', [2, 2], 'float32');
    $M::exp($X::zeros([2, 2]), $o);
    unset($o);
    check("{$tag} openMemmap", $X::load($f)->toList() === [1.0, 1.0, 1.0, 1.0] && $X::load($f)->dtype() == ($tag === 'ffi' ? Tessero\DType::Float32 : 'float32'));
    unlink($f);
}
backend_examples(Tessero\NDArray::class, Tessero\Math::class, 'ffi');
if (extension_loaded('tessero')) {
    backend_examples(Tessero\Ext\NDArray::class, Tessero\Ext\Math::class, 'ext');
}
// guide/statistics: the same calls on each backend, with SciPy's/NumPy's values (pinned reference, ADR 0012)
function statistics_examples(string $St, string $Np, string $Sp, string $G, string $tag): void
{
    $z = $St::norm();
    check("{$tag} norm cdf/ppf", $z->cdf(1.96) === 0.9750021048517795 && $z->ppf(0.975) === 1.959963984540054);
    $g = $St::gamma(2.0, scale: 3.0);
    check("{$tag} gamma moments", $g->mean() === 6.0 && $g->var() === 18.0 && $g->median() === 5.035040970049984);
    check("{$tag} gamma interval", $g->interval(0.9) === [1.066084532095986, 14.231593555171731]);
    check("{$tag} gamma stats", $g->stats(moments: 'mv') === ['mean' => 6.0, 'var' => 18.0]);
    check("{$tag} gamma rvs", $g->rvs(size: 3, randomState: $G::defaultRng(7))->toList() === [5.004765876904312, 4.011647087112922, 3.437703595760081]);
    check("{$tag} poisson/binom", $St::poisson(4.5)->pmf(3) === 0.168717884924555 && $St::binom(10, 0.3)->sf(4) === 0.15026833259999992);
    $x = [2.1, 3.4, 1.9, 5.6, 4.4, 3.3, 2.8];
    $y = [3.9, 4.1, 5.2, 6.0, 4.8, 5.5, 6.3];
    check("{$tag} ttestInd", $St::ttestInd($x, $y) === ['statistic' => -2.9318974619301956, 'pvalue' => 0.012559627601277113, 'df' => 12.0]);
    check("{$tag} pearsonr", $St::pearsonr($x, $y) === ['statistic' => 0.3295765529354113, 'pvalue' => 0.47037875019519426]);
    check("{$tag} describe", array_keys($St::describe($x)) === ['nobs', 'minmax', 'mean', 'variance', 'skewness', 'kurtosis']);
    check("{$tag} np quantiles", $Np::median($x) === 3.3 && $Np::percentile($x, 90) === 4.880000000000001 && $Np::quantile($x, 0.25, method: 'nearest') === 2.8);
    check("{$tag} special", $Sp::gammainc(2.0, 1.5) === 0.4421745996289252 && $Sp::erfinv(0.5) === 0.4769362762044699 && $Sp::jv(1.0, 2.5) === 0.4970941024642741);
    $rng = $G::defaultRng(42);
    check("{$tag} generator gamma/poisson", $rng->gamma(2.0, size: 3)->toList() === [2.0918172704999494, 2.8353455897858653, 1.8372155803071377]
        && $rng->poisson(3.0, size: 4)->toList() === [2, 4, 1, 7]);
}
// guide/arrays: NumPy's array functions (Np)
function array_function_examples(string $Np, string $tag): void
{
    $a = [[1, 2, 3], [4, 5, 6]];
    check("{$tag} np flip", $Np::flip($a, axis: 1)->toArray() === [[3, 2, 1], [6, 5, 4]]);
    check("{$tag} np concatenate", $Np::concatenate([$a, [[7, 8, 9]]])->shape() === [3, 3]);
    $parts = $Np::split(range(0, 8), 3);
    check("{$tag} np split", count($parts) === 3 && $parts[2]->toList() === [6, 7, 8]);
    [$xx, $yy] = $Np::meshgrid([1, 2, 3], [4, 5], indexing: 'ij');
    check("{$tag} np meshgrid", $xx->shape() === [3, 2] && $yy->toArray() === [[4, 5], [4, 5], [4, 5]]);
    check("{$tag} np pad", $Np::pad([1, 2, 3], [2, 1], mode: 'reflect')->toList() === [3, 2, 1, 2, 3, 2]);
    check("{$tag} np interp", $Np::interp([0.5, 2.5], [0, 1, 2], [0, 10, 20])->toList() === [5.0, 20.0]);
}
array_function_examples(Tessero\Np::class, 'ffi');
if (extension_loaded('tessero')) {
    array_function_examples('Tessero\\Ext\\Np', 'ext');
}
statistics_examples(Tessero\Stats::class, Tessero\Np::class, Tessero\Special::class, Generator::class, 'ffi');
if (extension_loaded('tessero')) {
    statistics_examples('Tessero\\Ext\\Stats', 'Tessero\\Ext\\Np', 'Tessero\\Ext\\Special', 'Tessero\\Ext\\Random\\Generator', 'ext');
}
// guide/fft: real-input transform
$r = Fft::rfft(arr([1.0, 2.0, 3.0, 4.0]));
check('rfft', $r->real()->toList() === [10.0, -2.0, -2.0] && close($r->imag()->toList(), [0.0, 2.0, 0.0], 1e-15));
check('irfft', close(Fft::irfft($r)->toList(), [1.0, 2.0, 3.0, 4.0], 1e-15));

fwrite(STDERR, sprintf("docs examples: %d checks, %d failed\n", $n, $fail));
exit($fail === 0 ? 0 : 1);
