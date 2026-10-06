--TEST--
NDArray: arithmetic operators, broadcasting, reflected scalars, compound assignment
--EXTENSIONS--
tessero
--FILE--
<?php
use Tessero\Ext\NDArray;
$a = NDArray::array([[1.0, 2.0], [3.0, 4.0]]);
$row = NDArray::array([10, 20]);
echo ($a + $row)->toJson(), "\n";
echo ($a * 2)->toJson(), " ", (2 * $a)->toJson(), "\n";
echo (1 - $a)->toJson(), " ", (12 / $a)->toJson(), "\n";
echo ($a ** 2)->toJson(), " ", (2 ** $a)->toJson(), "\n";
echo (-$a)->toJson(), " ", (NDArray::array([7, -7]) % 3)->toJson(), "\n";
$b = $a;
$b += 1;
echo $b->toJson(), " ", $a->toJson(), "\n";
echo (NDArray::array([1, 2]) / 2)->dtype(), " ", (NDArray::array([1, 2], 'int32') + 1)->dtype(), " ", (NDArray::array([true]) + NDArray::array([true]))->toJson(), "\n";
try { $a + NDArray::zeros(3); } catch (Tessero\Ext\ShapeException $e) { echo $e->getMessage(), "\n"; }
?>
--EXPECT--
[[11.0,22.0],[13.0,24.0]]
[[2.0,4.0],[6.0,8.0]] [[2.0,4.0],[6.0,8.0]]
[[0.0,-1.0],[-2.0,-3.0]] [[12.0,6.0],[4.0,3.0]]
[[1.0,4.0],[9.0,16.0]] [[2.0,4.0],[8.0,16.0]]
[[-1.0,-2.0],[-3.0,-4.0]] [1,2]
[[2.0,3.0],[4.0,5.0]] [[1.0,2.0],[3.0,4.0]]
float64 int32 [2]
Shapes (2, 2) and (3,) cannot be broadcast together
