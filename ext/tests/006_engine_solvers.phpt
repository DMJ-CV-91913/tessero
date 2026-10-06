--TEST--
Engine: linprog with shadow prices, milp, MDP solvers, NumPy-identical RNG, FFT, settings
--EXTENSIONS--
tessero
--FILE--
<?php
use Tessero\Ext\Engine;
use Tessero\Ext\NDArray;
$r = Engine::linprog([-40, -30], [[2, 1], [1, 1]], [100, 80], null, null, [[0, 40], [0, null]]);
printf("%s x=[%g, %g] fun=%g duals=[%g, %g]\n", $r['message'], $r['x'][0], $r['x'][1], $r['fun'], $r['ineqlin'][0], $r['ineqlin'][1]);
$u = Engine::linprog([-1, 0], [[0, 1]], [1]);
echo $u['status'], " ", $u['message'], "\n";
$m = Engine::milp([-10, -13, -7, -8], true, [[5, 6, 4, 3]], [10], null, null, [0, 1]);
echo json_encode($m['x']), " ", $m['fun'], "\n";
$P = NDArray::array([[[1, 0], [0, 1]], [[0, 1], [1, 0]]], 'float64');
$R = [[0, 0], [1, 0]];
$pi = Engine::mdpPolicyIteration($P, $R, 0.9);
echo $pi['policy']->toJson(), " ", round($pi['values']->item(0), 10), " ", round($pi['values']->item(1), 10), "\n";
$fh = Engine::mdpFiniteHorizon($P, $R, 2);
echo $fh['values']->toJson(), " ", $fh['policy']->toJson(), "\n";
$csr = ['indptr' => [0, 1, 2, 3, 4], 'indices' => [0, 1, 1, 0], 'data' => [1.0, 1.0, 1.0, 1.0]];
echo Engine::mdpValueIteration($csr, $R, 0.9, 1e-12)['policy']->toJson(), "\n";
try { Engine::mdpValueIteration([[[0.5, 0.4], [0, 1]]], [[0], [1]], 0.9); } catch (Tessero\Ext\Exception $e) { echo $e->getMessage(), "\n"; }
echo Engine::random(12345, 3)->toJson(), "\n";
echo Engine::normal(12345, 2, 1.5, 2.0)->toJson(), " ", Engine::integers(12345, 5, 0, 10)->toJson(), "\n";
echo Engine::fft([1.0, 2.0, 3.0, 4.0])->toJson(), "\n";
Engine::setMaxThreads(2);
echo Engine::getMaxThreads() >= 1 ? "threads ok" : "bad", " ", ini_get('tessero.epsilon'), "\n";
?>
--EXPECT--
Optimization terminated successfully. x=[20, 60] fun=-2600 duals=[-10, -20]
3 The problem is unbounded.
[0,1,0,1] -21
[1,0] 9 10
[[1.0,2.0],[0.0,1.0],[0.0,0.0]] [[1,0],[0,0]]
[1,0]
Transition probabilities of state 0 under action 0 sum to 0.9, not 1
[0.22733602246716966,0.31675833970975287,0.7973654573327341]
[-1.3476500729092624,4.02745691625822] [6,2,7,3,2]
[[10.0,0.0],[-2.0,2.0],[-2.0,0.0],[-2.0,-2.0]]
threads ok 1e-6
