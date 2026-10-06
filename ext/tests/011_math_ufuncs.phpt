--TEST--
Math ufuncs: kernel and loop families, broadcasting, dtype rules, out= (direct, cast, in place, overlap), errors
--EXTENSIONS--
tessero
--FILE--
<?php
use Tessero\Ext\{NDArray as X, Math};
$a = X::array([[0.25, 1.0, 4.0], [9.0, 16.0, 25.0]]);
echo Math::sqrt($a)->toJson(), " ", Math::sqrt($a)->toBytes() === $a->sqrt()->toBytes() ? "same as method" : "DIFFERENT", "\n";
echo Math::cbrt(X::array([8, -27, 64]))->toJson(), " ", Math::cbrt(X::array([8]))->dtype(), "\n";
echo Math::hypot($a, X::array([3.0, 4.0, 0.0]))->toJson(), "\n";          // broadcast [2,3] with [3]
echo Math::fmax(X::array([NAN, 1.0]), [2.0, NAN])->toJson(), " ", Math::maximum(X::array([NAN, 1.0]), [2.0, NAN])->toJson(), "\n";
echo Math::fmax(X::array([1, 5]), X::array([3, 2]))->dtype(), " ", Math::fmod(X::array([-7, 7, 5]), X::array([2, -2, 0]))->toJson(), "\n";
echo Math::copysign(X::array([1.0, 2.0], 'float32'), -1.0)->dtype(), " ", Math::trunc(X::array([3, -4]))->dtype(), " ", Math::signbit(X::array([-0.0, 0.0]))->toJson(), "\n";
printf("%.12f %.12f %.1f\n", Math::logaddexp(0.0, 0.0)->item(), Math::gamma(X::array([0.5]))->item(0) ** 2, Math::degrees(X::array([M_PI]))->item(0));

// out= : direct write, returns the same object
$o = X::zeros([2, 3]);
var_dump(Math::exp($a, $o) === $o);
echo $o->toBytes() === $a->exp()->toBytes() ? "direct ok\n" : "direct WRONG\n";
// out= with a cast (float64 result into float32 output)
$o32 = X::zeros([2, 3], 'float32');
Math::sqrt($a, $o32);
echo $o32->toJson(), "\n";
// in place, and a strided output view
$b = X::array([1.0, 4.0, 9.0, 16.0]);
Math::sqrt($b, $b);
echo $b->toJson(), "\n";
$z = X::zeros(6);
Math::cbrt(X::array([1.0, 8.0, 27.0]), $z['::2']);
echo $z->toJson(), "\n";
// overlapping output (shifted view of the input) goes through a temporary
$s = X::array([1.0, 2.0, 3.0, 4.0]);
Math::add($s['0:3'], 10, $s['1:4']);
echo $s->toJson(), "\n";
$t = X::array([1.0, 2.0, 3.0, 4.0]);
Math::copysign($t['0:3'], -1.0, $t['1:4']);
echo $t->toJson(), "\n";

echo count(Math::ufuncs()) > 60 ? "catalogue ok\n" : "catalogue short\n";
echo Math::apply('erf', X::array([0.0]))->toJson(), " ", Math::apply('nextafter', 1.0, 2.0)->item() > 1.0 ? "apply ok" : "apply WRONG", "\n";

foreach ([
    fn () => Math::sqrt($a, X::zeros([2, 3], 'int64')),          // float -> int is not same-kind
    fn () => Math::sqrt($a, X::zeros([3, 2])),                   // wrong shape
    fn () => Math::hypot($a, X::array([1.0, 2.0])),               // not broadcastable
    fn () => Math::cbrt(Tessero\Ext\Engine::fft([1.0, 2.0])),     // complex
    fn () => Math::sqrt($a, [1, 2]),                              // out must be an NDArray
    fn () => Math::apply('nope', 1.0),
] as $i => $bad) {
    try { $bad(); echo "$i accepted\n"; } catch (Throwable $e) { echo "$i ", get_class($e), "\n"; }
}
?>
--EXPECT--
[[0.5,1.0,2.0],[3.0,4.0,5.0]] same as method
[2.0,-3.0000000000000004,4.0] float64
[[3.010398644698074,4.123105625617661,4.0],[9.486832980505138,16.492422502470642,25.0]]
[2.0,1.0] [null,null]
int64 [-1,1,0]
float32 int64 [true,false]
0.693147180560 3.141592653590 180.0
bool(true)
direct ok
[[0.5,1.0,2.0],[3.0,4.0,5.0]]
[1.0,2.0,3.0,4.0]
[1.0,0.0,2.0,0.0,3.0000000000000004,0.0]
[1.0,11.0,12.0,13.0]
[1.0,-1.0,-2.0,-3.0]
catalogue ok
[0.0] apply ok
0 Tessero\Ext\DTypeException
1 Tessero\Ext\ShapeException
2 Tessero\Ext\ShapeException
3 Tessero\Ext\DTypeException
4 TypeError
5 ValueError
