--TEST--
NDArray::memmap: modes, flush, views outliving the root, read-only enforcement, shape inference, offsets, errors
--EXTENSIONS--
tessero
--FILE--
<?php
use Tessero\Ext\{NDArray as X, Math, Engine};
$dir = sys_get_temp_dir() . '/tsr_mm_' . getmypid();
@mkdir($dir);
$f = "$dir/a.f64";

$a = X::memmap($f, 'w+', [3, 2]);
var_dump($a->isMemmap(), $a->isReadonly(), filesize($f));
$a['1:'] = [[1, 2], [3, 4]];
$a->flush();
$a->flush(true);
$v = $a[':, 1'];
$before = Engine::info()['memory_mapped'];
unset($a);
echo "view keeps the mapping: ", $v->toJson(), " ", Engine::info()['memory_mapped'] === $before ? "still mapped" : "UNMAPPED", "\n";
unset($v);
echo "released: ", Engine::info()['memory_mapped'], "\n";

$r = X::memmap($f, 'r');
echo json_encode($r->shape()), " ", $r->sum(), " ", $r->dtype(), "\n";
try { $r[0] = 1; } catch (Tessero\Ext\Exception $e) { echo "read-only\n"; }
try { Math::negative($r, $r); } catch (Tessero\Ext\Exception $e) { echo "read-only out\n"; }
$h = clone $r;
$h[0] = 5;
echo "clone is a heap copy: ", var_export($h->isMemmap(), true), " ", $h->item(0), "\n";

$c = X::memmap($f, 'c', [3, 2]);
$c[2] = [-1, -1];
unset($c);
echo "copy-on-write leaves the file: ", X::memmap($f, 'r', [3, 2])[2]->toJson(), "\n";

$p = X::memmap($f, 'r+', [2], 'float64', 48);      // extends the file to 64 bytes
$p->assign([7, 8]);
$p->flush(true);
clearstatcache();
echo filesize($f), " ", X::memmap($f, 'r', null, 'float64', 48)->toJson(), "\n";

$i = X::memmap("$dir/i.i32", 'w+', [4], 'int32');
$i->assign([1, -2, 3, -4]);
echo Math::absolute($i)->toJson(), " ", X::memmap("$dir/i.i32", 'r', null, 'int32')->sum(), "\n";

foreach ([
    fn () => X::memmap("$dir/missing", 'r'),
    fn () => X::memmap('php://memory', 'r+', [1]),
    fn () => X::memmap($f, 'x'),
    fn () => X::memmap($f, 'r', [100]),
    fn () => X::memmap($f, 'r', null, 'float64', 4),
    fn () => X::memmap("$dir/n", 'w+'),
    fn () => X::memmap($f, 'r', [0]),
] as $k => $bad) {
    try { $bad(); echo "$k accepted\n"; } catch (Throwable $e) { echo "$k ", get_class($e), "\n"; }
}
unset($r, $h, $p, $i);
foreach (glob("$dir/*") as $g) unlink($g);
rmdir($dir);
echo "done\n";
?>
--EXPECT--
bool(true)
bool(false)
int(48)
view keeps the mapping: [0.0,2.0,4.0] still mapped
released: 0
[6] 10 float64
read-only
read-only out
clone is a heap copy: false 5
copy-on-write leaves the file: [3.0,4.0]
64 [7.0,8.0]
[1,2,3,4] -2
0 Tessero\Ext\Exception
1 ValueError
2 ValueError
3 Tessero\Ext\ShapeException
4 ValueError
5 ValueError
6 Tessero\Ext\ShapeException
done
