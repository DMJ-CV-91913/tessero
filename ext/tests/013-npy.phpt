--TEST--
NDArray::load / save / openMemmap: .npy round trips, Fortran order, memory-mapped .npy, invalid headers
--EXTENSIONS--
tessero
--FILE--
<?php
use Tessero\Ext\NDArray as X;
$dir = sys_get_temp_dir() . '/tsr_npy_' . getmypid();
@mkdir($dir);

$a = X::arange(12.0)->reshape([3, 4]);
$a['::2, 1:3']->save("$dir/v.npy");                          // a non-contiguous view is written in C order
echo X::load("$dir/v.npy")->toJson(), "\n";
X::array([true, false, true])->save("$dir/b.npy");
echo X::load("$dir/b.npy")->toJson(), " ", X::load("$dir/b.npy")->dtype(), "\n";
X::array(2.5)->save("$dir/s.npy");
echo json_encode(X::load("$dir/s.npy")->shape()), " ", X::load("$dir/s.npy", 'r')->item(), "\n";

$m = X::openMemmap("$dir/m.npy", 'w+', [2, 3], 'int32');
$m->assign([[1, 2, 3], [4, 5, 6]]);
unset($m);
echo X::load("$dir/m.npy")->toJson(), " ", filesize("$dir/m.npy"), "\n";
$r = X::load("$dir/m.npy", 'r');
var_dump($r->isMemmap(), $r->isReadonly());
$w = X::openMemmap("$dir/m.npy", 'r+');
$w[0] = [7, 7, 7];
unset($w, $r);
echo X::load("$dir/m.npy")->toJson(), "\n";

$f = X::openMemmap("$dir/f.npy", 'w+', [2, 3], 'float64', true);
$f->assign([[1, 2, 3], [4, 5, 6]]);
echo json_encode($f->strides()), " ";
unset($f);
echo X::load("$dir/f.npy")->toJson(), " ", X::load("$dir/f.npy", 'c')->toJson(), "\n";

file_put_contents("$dir/bad.npy", "\x93NUMPY\x01\x00\x10\x00{'descr': '>f8'}");
file_put_contents("$dir/junk.npy", "hello");
foreach ([
    fn () => X::load("$dir/bad.npy"),
    fn () => X::load("$dir/junk.npy"),
    fn () => X::load("$dir/nope.npy"),
    fn () => X::load("$dir/m.npy", 'w+'),
    fn () => X::openMemmap("$dir/x.npy", 'w+'),
] as $k => $bad) {
    try { $bad(); echo "$k accepted\n"; } catch (Throwable $e) { echo "$k ", get_class($e), "\n"; }
}
foreach (glob("$dir/*") as $g) unlink($g);
rmdir($dir);
echo "done\n";
?>
--EXPECT--
[[1.0,2.0],[9.0,10.0]]
[true,false,true] bool
[] 2.5
[[1,2,3],[4,5,6]] 152
bool(true)
bool(true)
[[7,7,7],[4,5,6]]
[8,16] [[1.0,2.0,3.0],[4.0,5.0,6.0]] [[1.0,2.0,3.0],[4.0,5.0,6.0]]
0 Tessero\Ext\DTypeException
1 Tessero\Ext\Exception
2 Tessero\Ext\Exception
3 ValueError
4 ValueError
done
