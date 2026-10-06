--TEST--
NDArray: foreach, count, JSON (json_encode and toJson), serialize round trip
--EXTENSIONS--
tessero
--FILE--
<?php
use Tessero\Ext\NDArray;
$a = NDArray::array([[1.5, 2.0], [NAN, -0.0]]);
foreach ($a as $i => $row) { echo $i, "=", $row->toJson(), " "; }
foreach (NDArray::array([3, 4]) as $v) { echo $v, " "; }
echo count($a), "\n";
echo $a->toJson(), "\n";
echo json_encode(['m' => NDArray::array([[1, 2]])]), "\n";
$s = serialize(NDArray::arange(6.0)->reshape(2, 3)->t());
$b = unserialize($s);
echo json_encode($b->shape()), " ", $b->toJson(), " ", var_export($b->isContiguous(), true), "\n";
var_dump($b == NDArray::arange(6.0)->reshape(2, 3)->t()->copy(), $b == NDArray::zeros([3, 2]));
echo (string) NDArray::array([1, 2]), "\n";
?>
--EXPECT--
0=[1.5,2.0] 1=[null,-0.0] 3 4 2
[[1.5,2.0],[null,-0.0]]
{"m":[[1,2]]}
[3,2] [[0.0,3.0],[1.0,4.0],[2.0,5.0]] true
bool(true)
bool(false)
array([1,2], shape=(2,), dtype=int64)
