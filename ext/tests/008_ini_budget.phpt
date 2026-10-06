--TEST--
INI: tessero.memory_budget turns runaway allocations into MemoryException
--EXTENSIONS--
tessero
--INI--
tessero.memory_budget=1M
--FILE--
<?php
use Tessero\Ext\NDArray;
$small = NDArray::zeros(1000);
try { NDArray::zeros(1000000); } catch (Tessero\Ext\MemoryException $e) { echo get_class($e), "\n"; }
echo Tessero\Ext\Engine::memoryBudget(), "\n";
?>
--EXPECT--
Tessero\Ext\MemoryException
1048576
