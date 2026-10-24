--TEST--
Memory ownership (ADR 0015): views keep their base, routine results are released on every path, a request loop returns to baseline
--EXTENSIONS--
tessero
--FILE--
<?php
use Tessero\Ext\Engine;
use Tessero\Ext\NDArray;
use Tessero\Ext\Np;

$base = Engine::memoryInUse();
$parent = NDArray::array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
$row = $parent["1"];
$col = $parent[":, 2"];
$inner = $row["1:"];
unset($parent);                                   // the parent goes first: the views keep the block
gc_collect_cycles();
var_dump($inner->toList(), $col->toList(), Engine::memoryInUse() > $base);
unset($inner, $row);
var_dump($col->toList());
unset($col);
gc_collect_cycles();
var_dump(Engine::memoryInUse() === $base);

$parts = Np::split(NDArray::array([1.0, 2.0, 3.0, 4.0]), 2);          // a kind 6 sequence
var_dump(count($parts), (string) Np::resultType(NDArray::array([1.0]), "float32"));   // a kind 2 string
unset($parts);
try { Np::split(NDArray::array([1.0, 2.0, 3.0]), 2); } catch (Throwable) { echo "split throws\n"; }
gc_collect_cycles();
var_dump(Engine::memoryInUse() === $base);

for ($request = 0; $request < 10000; $request++) {
    $a = NDArray::arange(16)->reshape([4, 4]);
    $v = $a["1:3, ::2"];
    $s = $a->add($v->sum());
    $parts = Np::split($s, 2);
    if ($request % 7 === 0) unset($a);
    try { Np::split($v, 3); } catch (Throwable) {}   // no $e: an exception keeps its call arguments alive
    unset($a, $v, $s, $parts);
}
gc_collect_cycles();
var_dump(Engine::memoryInUse() === $base);
?>
--EXPECT--
array(2) {
  [0]=>
  float(5)
  [1]=>
  float(6)
}
array(2) {
  [0]=>
  float(3)
  [1]=>
  float(6)
}
bool(true)
array(2) {
  [0]=>
  float(3)
  [1]=>
  float(6)
}
bool(true)
int(2)
string(7) "float64"
split throws
bool(true)
bool(true)
