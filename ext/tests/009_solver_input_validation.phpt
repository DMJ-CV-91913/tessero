--TEST--
Engine: linprog/milp reject NaN and INF in the problem data (as SciPy does)
--EXTENSIONS--
tessero
--FILE--
<?php
use Tessero\Ext\Engine;
foreach ([
    fn () => Engine::linprog([1.0, NAN]),
    fn () => Engine::linprog([1.0, 1.0], [[1.0, INF]], [1.0]),
    fn () => Engine::milp([1.0, NAN], true),
] as $call) {
    try { $call(); echo "accepted\n"; } catch (ValueError $e) { echo get_class($e), ": ", str_contains($e->getMessage(), 'finite') ? 'finite' : $e->getMessage(), "\n"; }
}
// infinite bounds stay legal
$r = Engine::linprog([1.0], null, null, null, null, [[-INF, INF]]);
echo $r['status'], "\n";
?>
--EXPECT--
ValueError: finite
ValueError: finite
ValueError: finite
3
