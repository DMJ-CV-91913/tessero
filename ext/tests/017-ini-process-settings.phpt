--TEST--
INI: tessero.memory_budget=-1 is an explicit unlimited, tessero.blas_threads applies in the CLI when set explicitly
--EXTENSIONS--
tessero
--SKIPIF--
<?php if (PHP_ZTS) die('skip NTS only: under ZTS these settings are PHP_INI_SYSTEM'); ?>
--INI--
tessero.memory_budget=-1
tessero.blas_threads=1
--FILE--
<?php
use Tessero\Ext\Engine;
echo 'budget ', Engine::memoryBudget(), "\n";
echo 'blas ', Engine::feature('openblas_threads') ? Engine::getBlasThreads() : 1, "\n";
// an explicit budget still wins over the automatic default, and a refused allocation stays a MemoryException
ini_set('tessero.memory_budget', '2M');
echo 'budget ', Engine::memoryBudget(), "\n";
try { Tessero\Ext\NDArray::zeros(1000000); } catch (Tessero\Ext\MemoryException $e) { echo get_class($e), "\n"; }
ini_set('tessero.memory_budget', '-1');
echo 'budget ', Engine::memoryBudget(), "\n";
?>
--EXPECT--
budget 0
blas 1
budget 2097152
Tessero\Ext\MemoryException
budget 0
