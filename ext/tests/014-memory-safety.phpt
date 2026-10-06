--TEST--
Memory safety: untrusted shapes (fromBytes, unserialize), overflowing sizes, read-only writes, no leaks
--EXTENSIONS--
tessero
--FILE--
<?php
use Tessero\Ext\{NDArray as X, Math, Engine};
function t(string $what, callable $f): void {
    try { $f(); echo "$what: accepted\n"; } catch (Throwable $e) { echo "$what: ", get_class($e), "\n"; }
}
t('negative dims', fn () => X::fromBytes(str_repeat("\0", 64), 'float64', [-1, -8]));
t('wrapping dims', fn () => X::fromBytes('', 'float64', [1 << 61, 8]));
t('huge zeros', fn () => X::zeros([1 << 40, 1 << 40]));
t('reshape wrap', fn () => X::zeros([0])->reshape(1 << 62, 4));

$payload = fn (array $shape, int $len) => 'O:19:"Tessero\Ext\NDArray":3:{s:5:"dtype";s:7:"float64";s:5:"shape";a:' . count($shape) . ':{'
    . implode('', array_map(fn ($k, $v) => "i:$k;i:$v;", array_keys($shape), $shape)) . '}s:4:"data";s:' . $len . ':"' . str_repeat("\0", $len) . '";}';
echo json_encode(unserialize($payload([2], 16))->toList()), "\n";
t('unserialize negative', fn () => unserialize($payload([-1, -2], 16)));
t('unserialize wrap', fn () => unserialize($payload([1 << 61, 8], 0)));
t('unserialize short', fn () => unserialize($payload([4], 8)));

$f = sys_get_temp_dir() . '/tsr_ms_' . getmypid() . '.f64';
X::memmap($f, 'w+', [8])->assign(1.0);
$ro = X::memmap($f, 'r');
t('write ro', function () use ($ro) { $ro[0] = 2.0; });
t('out ro', fn () => Math::exp($ro, $ro));
echo $ro->sum(), "\n";
unset($ro);
unlink($f);

$prev = Engine::memoryBudget();
$big = X::linspace(1.0, 0.0, 100000);
Engine::setMemoryBudget(Engine::memoryInUse() + 1200000);   // room for the copy, not for the sort's keys and scratch
t('sort over budget', fn () => $big->sort());
t('argsort over budget', fn () => $big->argsort());
Engine::setMemoryBudget($prev);
unset($big);

$base = Engine::info()['memory_in_use'];
for ($i = 0; $i < 100; $i++) {
    $a = X::arange(1000.0)->reshape([10, 100]);
    $v = $a->transpose()['::2'];
    $s = Math::logaddexp($v, 1.0)->sum([0, 1]);
    $c = clone $a;
    $u = unserialize(serialize($a));
    try { $a->add(X::zeros([3])); } catch (Throwable) {}
    unset($a, $v, $s, $c, $u);
}
echo Engine::info()['memory_in_use'] === $base ? "no leak\n" : "LEAK\n";
?>
--EXPECT--
negative dims: Tessero\Ext\ShapeException
wrapping dims: Tessero\Ext\ShapeException
huge zeros: Tessero\Ext\MemoryException
reshape wrap: Tessero\Ext\ShapeException
[0,0]
unserialize negative: Tessero\Ext\Exception
unserialize wrap: Tessero\Ext\Exception
unserialize short: Tessero\Ext\Exception
write ro: Tessero\Ext\Exception
out ro: Tessero\Ext\Exception
8
sort over budget: Tessero\Ext\MemoryException
argsort over budget: Tessero\Ext\MemoryException
no leak
