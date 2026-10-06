--TEST--
Engine: MDP solvers validate dense and CSR models (NaN, negative, sums, malformed CSR, rewards)
--EXTENSIONS--
tessero
--FILE--
<?php
use Tessero\Ext\Engine;
$R = [[10, -5], [0, -5]];
$cases = [
    'nan P'       => [[[[NAN, 1.0], [0.0, 1.0]], [[1.0, 0.0], [1.0, 0.0]]], $R],
    'negative P'  => [[[[1.5, -0.5], [0.0, 1.0]], [[1.0, 0.0], [1.0, 0.0]]], $R],
    'nan R'       => [[[[0.9, 0.1], [0.0, 1.0]], [[1.0, 0.0], [1.0, 0.0]]], [[NAN, 0], [0, 0]]],
    'csr short'   => [['indptr' => [0, 2, 3, 4, 9], 'indices' => [0, 1, 1, 0, 0], 'data' => [0.9, 0.1, 1.0, 1.0, 1.0]], $R],
    'csr nan'     => [['indptr' => [0, 2, 3, 4, 5], 'indices' => [0, 1, 1, 0, 0], 'data' => [NAN, 0.1, 1.0, 1.0, 1.0]], $R],
    'csr index'   => [['indptr' => [0, 2, 3, 4, 5], 'indices' => [0, 7, 1, 0, 0], 'data' => [0.9, 0.1, 1.0, 1.0, 1.0]], $R],
];
foreach ($cases as $name => [$P, $r]) {
    try { Engine::mdpValueIteration($P, $r, 0.9); echo "$name: accepted\n"; }
    catch (Throwable $e) { echo "$name: ", get_class($e), "\n"; }
}
$ok = Engine::mdpValueIteration(['indptr' => [0, 2, 3, 4, 5], 'indices' => [0, 1, 1, 0, 0], 'data' => [0.9, 0.1, 1.0, 1.0, 1.0]], $R, 0.95);
echo json_encode($ok['policy']->toList()), "\n";
?>
--EXPECT--
nan P: Tessero\Ext\Exception
negative P: Tessero\Ext\Exception
nan R: Tessero\Ext\Exception
csr short: Tessero\Ext\ShapeException
csr nan: Tessero\Ext\Exception
csr index: Tessero\Ext\IndexException
[0,1]
