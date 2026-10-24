--TEST--
Engine::selfcheck(): FP-environment gate + canonical KAT; NumPy float semantics fixed by the numerics review
--EXTENSIONS--
tessero
--FILE--
<?php
use Tessero\Ext\Engine;
use Tessero\Ext\NDArray;
$s = Engine::selfcheck();
echo implode(',', array_keys($s)), "\n";
var_dump(is_int($s['fpenv']), is_string($s['fpenv_reason']), is_int($s['selfcheck']));
echo $s['fpenv'], " ", $s['fpenv_reason'], " ", $s['selfcheck'], " ", var_export($s['canonical'], true), "\n";
// floor_divide is npy_divmod: 1.0 // 0.1 == 9 (floor(1.0 / 0.1) would be 10)
echo NDArray::array([1.0, -1.0], 'float64')->floorDiv(NDArray::array([0.1, 0.1], 'float64'))->toJson(), "\n";
?>
--EXPECT--
fpenv,fpenv_reason,selfcheck,canonical
bool(true)
bool(true)
bool(true)
0 canonical 0 true
[9.0,-10.0]
