--TEST--
NDArray: reductions with axes, statistics, sort, matmul, where, dtype casts
--EXTENSIONS--
tessero
--FILE--
<?php
use Tessero\Ext\NDArray;
$a = NDArray::array([[3, 1, 2], [9, 7, 8]]);
echo $a->sum(), " ", $a->sum(0)->toJson(), " ", $a->max(1)->toJson(), " ", $a->argmin(1)->toJson(), "\n";
echo $a->mean(), " ", $a->var(null, 1), " ", round($a->std(1)->item(0), 12), "\n";
echo $a->cumsum(1)->toJson(), " ", $a->sort()->toJson(), " ", $a->argsort()->toJson(), "\n";
echo $a->matmul(NDArray::array([1, 0, 1]))->toJson(), " ", NDArray::eye(2)->matmul([[1.0, 2.0], [3.0, 4.0]])->toJson(), "\n";
echo NDArray::where($a->gt(5), $a, 0)->toJson(), "\n";
echo $a->astype('float32')->dtype(), " ", $a->div(2)->astype('int32')->toJson(), " ", NDArray::array([0.5, 1.5, 2.5])->rint()->toJson(), "\n";
echo NDArray::linspace(0, 1, 5)->toJson(), " ", NDArray::full([2], 7)->toJson(), " ", NDArray::array([4.0, 9.0])->sqrt()->toJson(), "\n";
var_dump(NDArray::array([true, false])->any(), NDArray::array([true, false])->all(), NDArray::zeros(0)->sum());
try { NDArray::zeros(0)->max(); } catch (Tessero\Ext\ShapeException $e) { echo $e->getMessage(), "\n"; }
?>
--EXPECT--
30 [12,8,10] [3,9] [1,1]
5 11.6 0.816496580928
[[3,4,6],[9,16,24]] [[1,2,3],[7,8,9]] [[1,2,0],[1,2,0]]
[5,17] [[1.0,2.0],[3.0,4.0]]
[[0,0,0],[9,7,8]]
float32 [[1,0,1],[4,3,4]] [0.0,2.0,2.0]
[0.0,0.25,0.5,0.75,1.0] [7,7] [2.0,3.0]
bool(true)
bool(false)
float(0)
zero-size array to a reduction operation that has no identity
