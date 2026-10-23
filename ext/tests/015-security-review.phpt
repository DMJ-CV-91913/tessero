--TEST--
Security review 2026-10: inputs that wrote out of bounds, read static data with a bad dtype, recursed without bound or hit UB
--EXTENSIONS--
tessero
--FILE--
<?php
use Tessero\Ext\NDArray as X;
function t(string $what, callable $f): void {
    try { $r = $f(); echo "$what: ", $r === null ? 'ok' : json_encode($r), "\n"; } catch (Throwable $e) { echo "$what: ", get_class($e), "\n"; }
}
// eye(): an extreme diagonal offset overflowed the index math and wrote past the n*m block
t('eye k min', fn () => X::eye(2, 2, PHP_INT_MIN)->flatten()->toList());
t('eye k max', fn () => X::eye(2, 2, PHP_INT_MAX)->flatten()->toList());
t('eye k 1', fn () => X::eye(2, 3, 1)->flatten()->toList());
// where(): a non-numeric operand produced dtype -1 and indexed the item-size table with it
t('where null', fn () => X::where([true, false], null, 1.0));
t('where string', fn () => X::where([true, false], 'x', 1.0));
t('where ok', fn () => X::where([true, false], 1.0, 2.0)->toList());
// nested-array scan: a self-referencing array recursed until the C stack overflowed
$a = [1.0];
$a[] = &$a;
t('self reference', fn () => X::array([X::zeros([1]), $a]));
$deep = X::zeros([1]);
for ($i = 0; $i < 40; $i++) { $deep = [$deep]; }
t('nested 40 deep', fn () => X::array($deep));
// arange(): a non-finite length was cast to int64 (undefined behaviour)
t('arange inf', fn () => X::arange(0, INF));
t('arange nan step', fn () => X::arange(0, 1, NAN));
t('arange ok', fn () => X::arange(0, 3)->toList());
?>
--EXPECT--
eye k min: [0,0,0,0]
eye k max: [0,0,0,0]
eye k 1: [0,1,0,0,0,1]
where null: Tessero\Ext\DTypeException
where string: Tessero\Ext\DTypeException
where ok: [1,2]
self reference: Tessero\Ext\DTypeException
nested 40 deep: Tessero\Ext\ShapeException
arange inf: Tessero\Ext\MemoryException
arange nan step: Tessero\Ext\MemoryException
arange ok: [0,1,2]
