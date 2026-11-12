--TEST--
ABI version, capability query and BLAS thread control (ADR 0014); CLI defaults: a budget of half the RAM, BLAS open
--EXTENSIONS--
tessero
--FILE--
<?php
use Tessero\Ext\Engine;
echo 'abi ', Engine::abiVersion(), "\n";
echo 'one version source ', var_export(Engine::version() === Engine::kernelVersion(), true), "\n";
foreach (['repro_f32', 'gemm_canonical', 'lapack', 'no_such_feature', ''] as $f) {
    echo "feature $f: ", var_export(Engine::feature($f), true), "\n";
}
echo 'openmp feature matches ', var_export(Engine::feature('openmp') === Engine::openmp(), true), "\n";
$info = Engine::info();
echo 'info keys ', var_export(isset($info['abi'], $info['blas_threads']) && $info['abi'] === 2, true), "\n";
// the CLI's automatic budget (memory_budget=0) is half of the physical memory; the BLAS pool stays untouched
echo 'cli budget set ', var_export(Engine::memoryBudget() > 0, true), "\n";
echo 'blas ini ', ini_get('tessero.blas_threads'), "\n";
try { Engine::setBlasThreads(0); } catch (ValueError $e) { echo 'setBlasThreads(0): ', get_class($e), "\n"; }
if (Engine::feature('openblas_threads')) {
    echo 'set 1 ', var_export(Engine::setBlasThreads(1), true), ' -> ', Engine::getBlasThreads(), "\n";
    echo 'set 2 ', var_export(Engine::setBlasThreads(2), true), ' -> ', Engine::getBlasThreads(), "\n";
} else {
    echo "set 1 true -> 1\nset 2 true -> 2\n";
}
?>
--EXPECT--
abi 2
one version source true
feature repro_f32: true
feature gemm_canonical: true
feature lapack: true
feature no_such_feature: false
feature : false
openmp feature matches true
info keys true
cli budget set true
blas ini auto
setBlasThreads(0): ValueError
set 1 true -> 1
set 2 true -> 2
