<?php

/**
 * Dump every public method of Tessero's user-facing classes as JSON (used by
 * tools/gap_inventory.py to verify the NumPy/SciPy gap analysis against the
 * code, and by the developer-guide hub page).
 *
 *   php -d ffi.enable=1 [-d extension=tessero] tools/api-dump.php > api.json
 */

declare(strict_types=1);

$root = dirname(__DIR__);
require $root . '/tests/autoload.php';
spl_autoload_register(static function (string $c) use ($root): void {
    if (str_starts_with($c, 'Tessero\\Laravel\\')) {
        $f = $root . '/laravel/src/' . str_replace('\\', '/', substr($c, 16)) . '.php';
        if (is_file($f)) {
            require $f;
        }
    }
});
if (is_file($root . '/laravel/tests/stubs.php')) {
    require_once $root . '/laravel/tests/stubs.php';
}
if (! class_exists('Illuminate\\Support\\ServiceProvider')) {
    eval('namespace Illuminate\\Support; abstract class ServiceProvider { public function __construct(public $app = null) {} }');
}
if (! class_exists('Illuminate\\Support\\Facades\\Facade')) {
    eval('namespace Illuminate\\Support\\Facades; abstract class Facade {}');
}

$classes = ['Tessero\\NDArray', 'Tessero\\Math', 'Tessero\\Tessero', 'Tessero\\Linalg\\Linalg', 'Tessero\\Fft\\Fft',
    'Tessero\\ScipyLinalg', 'Tessero\\Constants', 'Tessero\\Distance', 'Tessero\\Signal', 'Tessero\\SignalWindows', 'Tessero\\Ndimage', 'Tessero\\Csgraph',
    'Tessero\\Random\\Generator', 'Tessero\\Sparse\\CsrMatrix', 'Tessero\\Optimize\\Minimize', 'Tessero\\Optimize\\Root',
    'Tessero\\Optimize\\LeastSquares', 'Tessero\\Optimize\\LinearProgramming', 'Tessero\\Optimize\\Assignment', 'Tessero\\Mdp\\MarkovDecisionProcess',
    'Tessero\\Io\\Npy', 'Tessero\\Integrate\\Quad', 'Tessero\\Spatial\\KDTree', 'Tessero\\Datetime\\Datetime', 'Tessero\\Ext\\NDArray', 'Tessero\\Ext\\Math', 'Tessero\\Ext\\Engine', 'Tessero\\Ext\\Random\\Generator',
    'Tessero\\Np', 'Tessero\\Special', 'Tessero\\Stats', 'Tessero\\Stats\\Distribution', 'Tessero\\Ext\\Np', 'Tessero\\Ext\\Special',
    'Tessero\\Ext\\Stats', 'Tessero\\Ext\\Distribution', 'Tessero\\Ext\\Fft\\Fft',
    'Tessero\\Ext\\Linalg', 'Tessero\\Ext\\ScipyLinalg', 'Tessero\\Ext\\Signal', 'Tessero\\Ext\\SignalWindows',
    'Tessero\\Ext\\Distance', 'Tessero\\Ext\\Ndimage', 'Tessero\\Ext\\Csgraph',
    'Tessero\\Laravel\\TesseroManager', 'Tessero\\Laravel\\Rules\\NumericArray'];

function sig(ReflectionMethod $m): string
{
    $p = [];
    foreach ($m->getParameters() as $x) {
        $t = $x->getType();
        $s = ($t ? (string) $t . ' ' : '') . ($x->isVariadic() ? '...' : '') . '$' . $x->getName();
        if ($x->isOptional() && ! $x->isVariadic()) {
            try {
                $v = $x->getDefaultValue();
                $s .= ' = ' . (is_array($v) ? '[]' : ($v === null ? 'null' : (is_object($v) ? '…' : var_export($v, true))));
            } catch (Throwable) {
                $s .= ' = …';
            }
        }
        $p[] = $s;
    }
    $r = $m->getReturnType();

    return $m->getName() . '(' . implode(', ', $p) . ')' . ($r ? ': ' . $r : '');
}

$out = [];
foreach ($classes as $c) {
    if (! class_exists($c)) {
        continue;                                   // e.g. Tessero\Ext\* without the extension
    }
    $r = new ReflectionClass($c);
    $ms = [];
    foreach ($r->getMethods(ReflectionMethod::IS_PUBLIC) as $m) {
        if ($m->getDeclaringClass()->getName() !== $c) {
            continue;
        }
        $n = $m->getName();
        if ((str_starts_with($n, '__') && $n !== '__construct') || str_starts_with($n, 'offset')) {
            continue;
        }
        $d = $m->getDocComment();
        $doc = '';
        if ($d) {
            foreach (preg_split('/\R/', $d) as $l) {
                $l = trim((string) preg_replace('#^\s*/\*\*\s?|\s*\*/\s*$|^\s*\*\s?#', '', $l));
                if ($l === '' || $l[0] === '@') {
                    if ($doc !== '') {
                        break;
                    }
                    continue;
                }
                $doc .= ($doc ? ' ' : '') . $l;
            }
        }
        if ($doc === '' && $c === 'Tessero\\Ext\\Math') {
            $doc = Tessero\Ext\Math::ufuncs()[$n]['summary'] ?? '';
        }
        $ms[] = ['name' => $n, 'static' => $m->isStatic(), 'sig' => sig($m), 'doc' => $doc];
    }
    $entry = ['class' => $c, 'methods' => $ms];
    $consts = array_keys($r->getConstants());
    if ($consts !== []) {
        $entry['constants'] = $consts;
    }
    $out[] = $entry;
}
echo json_encode($out, JSON_PRETTY_PRINT | JSON_UNESCAPED_SLASHES), "\n";
