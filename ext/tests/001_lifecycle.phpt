--TEST--
NDArray: views keep their root alive, memory returns to zero, clone is deep
--EXTENSIONS--
tessero
--FILE--
<?php
use Tessero\Ext\NDArray;
use Tessero\Ext\Engine;
$base = Engine::memoryInUse();
function tail() { $big = NDArray::arange(1000.0); return $big['990:']; }
$v = tail();
gc_collect_cycles();
var_dump($v->isView(), $v->item(0), Engine::memoryInUse() > $base);
$w = $v['::2'];
unset($v);
var_dump($w->toList());
unset($w);
gc_collect_cycles();
var_dump(Engine::memoryInUse() === $base);
$a = NDArray::arange(6)->reshape(2, 3);
$t = $a->t();
$c = clone $t;
$a['0, 0'] = 100;
var_dump($t->item(0, 0), $c->item(0, 0), $c->isView(), $c->isContiguous());
for ($i = 0; $i < 2000; $i++) { $x = NDArray::ones([100, 100]); $y = ($x * 2 + 1)->t()['::2']; }
unset($x, $y, $a, $t, $c);
gc_collect_cycles();
var_dump(Engine::memoryInUse() === $base);
?>
--EXPECT--
bool(true)
float(990)
bool(true)
array(5) {
  [0]=>
  float(990)
  [1]=>
  float(992)
  [2]=>
  float(994)
  [3]=>
  float(996)
  [4]=>
  float(998)
}
bool(true)
int(100)
int(0)
bool(false)
bool(true)
bool(true)
