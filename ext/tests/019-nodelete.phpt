--TEST--
The extension is linked with -z nodelete on Linux (never unmapped at shutdown while OpenMP workers may run; ADR 0014)
--EXTENSIONS--
tessero
--SKIPIF--
<?php if (PHP_OS_FAMILY !== 'Linux') die('skip ELF only'); ?>
--FILE--
<?php
var_dump(\Tessero\Ext\Engine::feature('nodelete'));
?>
--EXPECT--
bool(true)
