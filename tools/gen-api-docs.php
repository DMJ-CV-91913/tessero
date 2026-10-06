<?php

/**
 * Generate the API reference (docs/reference/api/*.md) from the code itself:
 * PHP classes by Reflection (signatures + docblocks), ext-tessero classes when
 * the extension is loaded, and the C ABI from csrc/include/tessero.h.
 *
 *   php -d ffi.enable=1 -d extension=tessero tools/gen-api-docs.php
 *
 * CI regenerates and fails if the committed pages are stale.
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
// Reflection needs the Laravel base classes to exist; outside an application
// declare empty doc-only stand-ins (never loaded at runtime).
if (! class_exists('Illuminate\\Support\\ServiceProvider')) {
    eval('namespace Illuminate\\Support; abstract class ServiceProvider { public function __construct(public $app = null) {} }');
}
if (! class_exists('Illuminate\\Console\\Command')) {
    eval('namespace Illuminate\\Console; abstract class Command {}');
}
if (! class_exists('Illuminate\\Support\\Facades\\Facade')) {
    eval('namespace Illuminate\\Support\\Facades; abstract class Facade {}');
}

$out = $root . '/docs/reference/api';
@mkdir($out, 0777, true);

$groups = [
    'Core arrays' => ['Tessero\\NDArray', 'Tessero\\Math', 'Tessero\\DType', 'Tessero\\Tessero'],
    'Linear algebra, FFT, random' => ['Tessero\\Linalg\\Linalg', 'Tessero\\Fft\\Fft', 'Tessero\\Random\\Generator'],
    'Statistics and special functions' => ['Tessero\\Special', 'Tessero\\Stats', 'Tessero\\Stats\\Distribution', 'Tessero\\Np',
        'Tessero\\Native\\Registry'],
    'Sparse' => ['Tessero\\Sparse\\CsrMatrix'],
    'Optimisation' => ['Tessero\\Optimize\\Minimize', 'Tessero\\Optimize\\Root', 'Tessero\\Optimize\\LeastSquares', 'Tessero\\Optimize\\LinearProgramming',
        'Tessero\\Optimize\\OptimizeResult', 'Tessero\\Optimize\\LinprogResult', 'Tessero\\Optimize\\MilpResult'],
    'Markov decision processes' => ['Tessero\\Mdp\\MarkovDecisionProcess', 'Tessero\\Mdp\\MdpResult'],
    'I/O' => ['Tessero\\Io\\Npy'],
    'Native layer (FFI)' => ['Tessero\\Native\\Library', 'Tessero\\Native\\Buffer', 'Tessero\\Native\\Blas', 'Tessero\\Native\\Abi'],
    'Exceptions' => ['Tessero\\Exceptions\\TesseroException', 'Tessero\\Exceptions\\ShapeError', 'Tessero\\Exceptions\\IndexError',
        'Tessero\\Exceptions\\DTypeError', 'Tessero\\Exceptions\\MemoryError', 'Tessero\\Exceptions\\LibraryUnavailable',
        'Tessero\\Exceptions\\LinAlgError', 'Tessero\\Exceptions\\SingularMatrix', 'Tessero\\Exceptions\\NotPositiveDefinite',
        'Tessero\\Exceptions\\ConvergenceError'],
    'Laravel bridge' => ['Tessero\\Laravel\\TesseroManager', 'Tessero\\Laravel\\TesseroServiceProvider', 'Tessero\\Laravel\\Casts\\AsNDArray',
        'Tessero\\Laravel\\Casts\\TensorCast', 'Tessero\\Laravel\\Rules\\NumericArray'],
];
if (extension_loaded('tessero')) {
    $groups['Native extension (ext-tessero)'] = ['Tessero\\Ext\\NDArray', 'Tessero\\Ext\\Math', 'Tessero\\Ext\\Engine', 'Tessero\\Ext\\Operand',
        'Tessero\\Ext\\Special', 'Tessero\\Ext\\Stats', 'Tessero\\Ext\\Distribution', 'Tessero\\Ext\\Np', 'Tessero\\Ext\\Random\\Generator', 'Tessero\\Ext\\Exception',
        'Tessero\\Ext\\ShapeException', 'Tessero\\Ext\\IndexException', 'Tessero\\Ext\\DTypeException', 'Tessero\\Ext\\MemoryException'];
}

function doc_text(string|false $doc): string
{
    if ($doc === false) {
        return '';
    }
    $lines = preg_split('/\R/', $doc);
    $out = [];
    foreach ($lines as $l) {
        $l = preg_replace('#^\s*/\*\*\s?|\s*\*/\s*$|^\s*\*\s?#', '', $l);
        if (str_starts_with(trim($l), '@')) {
            continue;
        }
        $out[] = rtrim($l);
    }

    return trim(implode("\n", $out));
}

function type_str(?ReflectionType $t): string
{
    return $t === null ? '' : (string) $t;
}

function signature(ReflectionMethod $m): string
{
    $params = [];
    foreach ($m->getParameters() as $p) {
        $s = type_str($p->getType());
        $s .= ($s !== '' ? ' ' : '') . ($p->isVariadic() ? '...' : '') . '$' . $p->getName();
        if ($p->isDefaultValueAvailable()) {
            try {
                $v = $p->getDefaultValue();
                $s .= ' = ' . (is_array($v) ? '[]' : ($v === null ? 'null' : (is_object($v) ? get_class($v) . '::' . ($v->name ?? '') : var_export($v, true))));
            } catch (Throwable) {
            }
        } elseif ($p->isOptional() && ! $p->isVariadic()) {
            $s .= ' = …';
        }
        $params[] = $s;
    }
    $ret = type_str($m->getReturnType());

    return ($m->isStatic() ? 'static ' : '') . $m->getName() . '(' . implode(', ', $params) . ')' . ($ret !== '' ? ': ' . $ret : '');
}

function slug(string $class): string
{
    return strtolower(str_replace('\\', '-', $class));
}

$ufuncs = class_exists('Tessero\\Ext\\Math') ? Tessero\Ext\Math::ufuncs() : [];
$index = "# API reference\n\nGenerated from the code by `tools/gen-api-docs.php` (signatures and docblocks). "
    . "The C ABI is in [C ABI](c-abi.md).\n\n";
$count = 0;
foreach ($groups as $group => $classes) {
    $index .= "## {$group}\n\n| Class | Summary |\n|---|---|\n";
    foreach ($classes as $class) {
        if (! class_exists($class) && ! interface_exists($class) && ! enum_exists($class)) {
            continue;
        }
        $r = new ReflectionClass($class);
        $summary = strtok(doc_text($r->getDocComment()) ?: '', "\n") ?: '';
        $index .= '| [`' . $class . '`](' . slug($class) . '.md) | ' . str_replace('|', '\\|', $summary) . " |\n";
        $md = '# ' . $r->getShortName() . "\n\n`" . $class . '`';
        if ($r->getParentClass()) {
            $md .= ' extends `' . $r->getParentClass()->getName() . '`';
        }
        $ifaces = array_values(array_filter($r->getInterfaceNames(), static fn (string $i): bool => ! in_array($i, ['UnitEnum', 'BackedEnum', 'Stringable'], true)));
        if ($ifaces !== []) {
            $md .= ' implements `' . implode('`, `', $ifaces) . '`';
        }
        $md .= ($r->isInternal() ? ' *(native extension)*' : '') . "\n\n";
        $d = doc_text($r->getDocComment());
        if ($d !== '') {
            $md .= $d . "\n\n";
        }
        if ($r->isEnum()) {
            $md .= "## Cases\n\n";
            foreach ((new ReflectionEnum($class))->getCases() as $case) {
                $md .= '- `' . $case->getName() . '`' . ($case instanceof ReflectionEnumBackedCase ? ' = `' . var_export($case->getBackingValue(), true) . '`' : '') . "\n";
            }
            $md .= "\n";
        }
        $consts = array_filter($r->getReflectionConstants(), static fn ($c) => $c->isPublic() && ! ($r->isEnum() && $c->isEnumCase()));
        if ($consts !== []) {
            $md .= "## Constants\n\n| Name | Value |\n|---|---|\n";
            foreach ($consts as $c) {
                $v = $c->getValue();
                $md .= '| `' . $c->getName() . '` | `' . (is_array($v) ? json_encode($v) : var_export($v, true)) . "` |\n";
            }
            $md .= "\n";
        }
        $methods = array_filter($r->getMethods(ReflectionMethod::IS_PUBLIC), static fn ($m) => $m->getDeclaringClass()->getName() === $class && ! str_starts_with($m->getName(), '__') || in_array($m->getName(), ['__construct', '__serialize', '__unserialize', '__toString'], true) && $m->getDeclaringClass()->getName() === $class);
        usort($methods, static fn ($a, $b) => [$a->isStatic() ? 0 : 1, $a->getName()] <=> [$b->isStatic() ? 0 : 1, $b->getName()]);
        if ($methods !== []) {
            $md .= "## Methods\n\n";
            foreach ($methods as $m) {
                if ($m->isPrivate() || ($m->getName() === '__construct' && ! $m->isPublic())) {
                    continue;
                }
                $md .= '### ' . $m->getName() . "\n\n```php\n" . signature($m) . "\n```\n\n";
                $doc = doc_text($m->getDocComment());
                if ($doc === '' && $class === 'Tessero\\Ext\\Math' && isset($ufuncs[$m->getName()])) {
                    $u = $ufuncs[$m->getName()];
                    $doc = $u['summary'] . '. ' . ($u['engine'] === 'kernel' ? 'Kernel ufunc (same code as the NDArray method).' : 'Loop ufunc.');
                }
                $md .= $doc . "\n\n";
            }
        }
        file_put_contents($out . '/' . slug($class) . '.md', rtrim(preg_replace("/\n{3,}/", "\n\n", $md)) . "\n");
        $count++;
    }
    $index .= "\n";
}
file_put_contents($out . '/index.md', $index);

// ---- C ABI
$h = (string) file_get_contents($root . '/csrc/include/tessero.h');
$c = "# C ABI (libtessero)\n\n`csrc/include/tessero.h` is the complete ABI. It is parsed verbatim by "
    . "`FFI::cdef()`/`FFI::load()` and compiled into ext-tessero, so it holds declarations only. Strides are in bytes; "
    . "every kernel returns 0 or a negative error code (see [error codes](../error-codes.md)).\n\n";
$sections = preg_split('#\n/\* ---- #', $h);
foreach ($sections as $i => $sec) {
    if ($i === 0) {
        // header comment (conventions) + the version/SIMD probes before the first section
        $c .= "## Conventions and version\n\n```c\n" . trim($sec) . "\n```\n\n";
        continue;
    }
    [$title, $body] = explode("\n", $sec, 2) + [1 => ''];
    $title = trim(preg_replace('# ----.*#', '', $title));
    $c .= '## ' . ucfirst($title) . "\n\n```c\n" . trim($body) . "\n```\n\n";
}
file_put_contents($out . '/c-abi.md', $c);
fwrite(STDERR, "API reference: {$count} classes + C ABI -> docs/reference/api/\n");
