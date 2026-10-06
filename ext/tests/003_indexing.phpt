--TEST--
NDArray: string slices, negative strides, masks, fancy indexing, assignment
--EXTENSIONS--
tessero
--FILE--
<?php
use Tessero\Ext\NDArray;
$a = NDArray::arange(24)->reshape(2, 3, 4);
echo $a['1, ::-1, ::2']->toJson(), "\n";
echo $a['..., -1']->toJson(), "\n";
var_dump($a['1, 2, 3'], isset($a[1]), isset($a[5]), isset($a['0, 1:2']), isset($a['0,,']));
echo $a[':, None, 0, 0']->toJson(), " ", json_encode($a[':, None, 0, 0']->shape()), "\n";
$m = NDArray::arange(10);
echo $m[$m->gt(6)]->toJson(), " ", $m[[9, 0, -1]]->toJson(), "\n";
$m[$m->lt(3)] = -1;
$m['5:'] = NDArray::array([50, 60, 70, 80, 90]);
$m[4] = 40;
echo $m->toJson(), "\n";
$m['1:'] = $m[':-1'];
echo $m->toJson(), "\n";
foreach (['10', '0:x', '::0'] as $bad) {
    try { $m[$bad]; } catch (Tessero\Ext\IndexException $e) { echo get_class($e), ": ", $e->getMessage(), "\n"; }
}
try { unset($m[0]); } catch (Tessero\Ext\IndexException $e) { echo $e->getMessage(), "\n"; }
?>
--EXPECT--
[[20,22],[16,18],[12,14]]
[[3,7,11],[15,19,23]]
int(23)
bool(true)
bool(false)
bool(true)
bool(false)
[[0],[12]] [2,1]
[7,8,9] [9,0,9]
[-1,-1,-1,3,40,50,60,70,80,90]
[-1,-1,-1,-1,3,40,50,60,70,80]
Tessero\Ext\IndexException: Index '10' is out of bounds for an array of shape (10,)
Tessero\Ext\IndexException: Invalid index '0:x'
Tessero\Ext\IndexException: Invalid index '::0'
Array elements cannot be unset
