<?php

declare(strict_types=1);

namespace Tessero\Native;

use FFI;
use FFI\CData;
use Tessero\DType;
use Tessero\Exceptions\DTypeError;
use Tessero\Exceptions\ShapeError;
use Tessero\Exceptions\TesseroException;
use Tessero\NDArray;

/**
 * Calls into libtessero's function registry (ADR 0011): scipy.special functions, statistics and the other
 * SciPy-style routines. The kernel describes every function (arguments, outputs, parameters) and this class
 * interprets that description, so each registry entry is callable without a hand-written binding; the
 * native extension does the same in C (ext/tessero_fn.c), which keeps the two backends identical.
 *
 * Façade classes (Tessero\Special, Tessero\Stats, ...) are generated from the registry by
 * tools/parity/gen-facades.php and forward here.
 *
 * @internal
 */
final class Registry
{
    /** @var array<string, array<string, mixed>>|null name => info (with 'id') */
    private static ?array $registry = null;

    private function __construct()
    {
    }

    /** @return array<string, array<string, mixed>> every registry entry by name */
    public static function all(): array
    {
        if (self::$registry === null) {
            $ffi = Library::ffi();
            $n = $ffi->tsr_fn_count();
            $buf = $ffi->new('char[16384]');
            $reg = [];
            for ($id = 0; $id < $n; $id++) {
                $len = $ffi->tsr_fn_info($id, $buf, 16384);
                if ($len >= 16384) {
                    $big = $ffi->new('char[' . ($len + 1) . ']');
                    $ffi->tsr_fn_info($id, $big, $len + 1);
                    $json = FFI::string($big, $len);
                } else {
                    $json = FFI::string($buf, $len);
                }
                $info = json_decode($json, true, 16, JSON_THROW_ON_ERROR);
                $info['id'] = $id;
                $reg[$info['name']] = $info;
            }
            self::$registry = $reg;
        }

        return self::$registry;
    }

    /** @return array<string, mixed> */
    public static function info(string $name): array
    {
        $all = self::all();
        if (! isset($all[$name])) {
            throw new \ValueError("'{$name}' is not a Tessero function");
        }

        return $all[$name];
    }

    // ------------------------------------------------------------------ element-wise functions

    /**
     * Evaluate an element-wise registry function (a ufunc, or a distribution method when $method >= 0 on a
     * "dist" entry). All-scalar inputs give floats; otherwise arrays broadcast as in NumPy. Several outputs
     * come back as an array keyed by output name.
     *
     * @param list<mixed> $args
     * @param NDArray|list<NDArray>|null $out
     */
    public static function ufunc(string $name, array $args, NDArray|array|null $out = null, int $method = -1): NDArray|float|array
    {
        $info = self::info($name);
        $ffi = Library::ffi();
        if ($info['kind'] === 'dist') {
            $nin = $ffi->new('int');
            $nout = $ffi->new('int');
            Library::check($ffi->tsr_fn_arity($info['id'], $method, FFI::addr($nin), FFI::addr($nout)), $name);
            $nin = $nin->cdata;
            $nout = $nout->cdata;
            $outs = $nout === 1 ? ['y'] : ($method === 8 ? ['mean', 'var', 'skew', 'kurtosis'] : ['a', 'b']);
            $kernelMethod = $method;
        } else {
            if ($info['kind'] !== 'ufunc') {
                throw new \ValueError("{$name} is not element-wise");
            }
            $nin = $info['nin'];
            $nout = $info['nout'];
            $outs = $info['outs'];
            $kernelMethod = 0;
        }
        $label = self::label($name);
        if (count($args) !== $nin) {
            throw new \ArgumentCountError("{$label}() takes {$nin} argument(s), " . count($args) . ' given');
        }

        // NumPy's integer loop when every integer argument is an integer (PHP int, or an integer array)
        if ($kernelMethod === 0 && isset($info['int_args'])) {
            $ints = true;
            foreach ($info['int_args'] as $k) {
                $v = $args[$k];
                if (! (is_int($v) || is_bool($v) || ($v instanceof NDArray && ($v->dtype()->isInteger() || $v->dtype() === DType::Bool)))) {
                    $ints = false;
                    break;
                }
            }
            if ($ints) {
                $kernelMethod = 1;
            }
        }

        // scalar fast path
        $allScalar = $out === null;
        foreach ($args as $a) {
            if (! (is_int($a) || is_float($a) || is_bool($a))) {
                $allScalar = false;
                break;
            }
        }
        if ($allScalar) {
            $nop = $nin + $nout;
            $vals = $ffi->new("double[{$nop}]");
            $ptrs = $ffi->new("void*[{$nop}]");
            foreach ($args as $k => $a) {
                $vals[$k] = (float) $a;
            }
            for ($k = 0; $k < $nop; $k++) {
                $ptrs[$k] = $ffi->cast('void*', FFI::addr($vals[$k]));
            }
            self::check($ffi->tsr_fn_ufunc($info['id'], $kernelMethod, 0, 0, null, $nop, $ptrs, null), $label);
            if ($nout === 1) {
                return $vals[$nin];
            }
            $r = [];
            foreach ($outs as $k => $o) {
                $r[$o] = $vals[$nin + $k];
            }

            return $r;
        }

        // loop dtype: complex128 when any operand is complex (the kernel runs the ufunc's complex loop if it has
        // one, else raises a DTypeError); otherwise float32 only when every array operand is float32.
        $ops = [];
        $f32 = true;
        $anyArray = false;
        $anyComplex = false;
        foreach ($args as $a) {
            if ($a instanceof NDArray) {
                $anyArray = true;
                if ($a->dtype() === DType::Complex128) {
                    $anyComplex = true;
                } elseif ($a->dtype() !== DType::Float32) {
                    $f32 = false;
                }
            } elseif (is_array($a)) {
                $anyArray = true;
                $f32 = false;
            }
        }
        $dt = $anyComplex ? DType::Complex128 : (($anyArray && $f32) ? DType::Float32 : DType::Float64);
        $shapes = [];
        foreach ($args as $a) {
            $arr = $a instanceof NDArray ? $a : NDArray::array($a, $dt === DType::Complex128 ? DType::Complex128 : DType::Float64);
            $arr = $arr->dtype() === $dt ? $arr : $arr->astype($dt);
            $ops[] = $arr;
            $shapes[] = $arr->shape();
        }
        try {
            $shape = NDArray::broadcastShapes(...$shapes);
        } catch (ShapeError) {
            throw new ShapeError("{$label}(): operands with shapes " . implode(', ', array_map(self::shapeStr(...), $shapes)) . ' cannot be broadcast together');
        }
        $outList = $out === null ? [] : ($out instanceof NDArray ? [$out] : array_values($out));
        if ($outList !== [] && count($outList) !== $nout) {
            throw new \ArgumentCountError("{$label}(): out must hold {$nout} array(s)");
        }
        $targets = [];
        $direct = [];
        for ($k = 0; $k < $nout; $k++) {
            $o = $outList[$k] ?? null;
            if ($o !== null && ! $o->isWritable()) {
                throw new TesseroException("{$label}(): out array is read-only");
            }
            $ok = $o !== null && $o->dtype() === $dt && $o->shape() === $shape && ! self::overlaps($o, $ops);
            $direct[$k] = $ok;
            $targets[$k] = $ok ? $o : NDArray::empty($shape, $dt);
        }
        $size = (int) array_product($shape);
        if ($size > 0) {
            $nop = $nin + $nout;
            $nd = count($shape);
            $ptrs = $ffi->new("void*[{$nop}]");
            $strides = [];
            $keep = [];
            foreach ($ops as $k => $a) {
                $b = $a->shape() === $shape ? $a : $a->broadcastTo($shape);
                $keep[] = $b;
                $ptrs[$k] = $ffi->cast('void*', $b->ptr());
                array_push($strides, ...$b->strides());
            }
            foreach ($targets as $k => $t) {
                $ptrs[$nin + $k] = $ffi->cast('void*', $t->ptr());
                array_push($strides, ...$t->strides());
            }
            self::check($ffi->tsr_fn_ufunc($info['id'], $kernelMethod, $dt->value, $nd, Library::i64($shape), $nop, $ptrs,
                $nd > 0 ? Library::i64($strides) : null), $label);
            unset($keep);
        }
        $res = [];
        foreach ($targets as $k => $t) {
            if (isset($outList[$k]) && ! $direct[$k]) {
                $t = self::castInto($t, $outList[$k], $label);
            }
            $res[$outs[$k]] = $t;
        }

        return $nout === 1 ? $res[$outs[0]] : $res;
    }

    // ------------------------------------------------------------------ generalised ufuncs

    /**
     * Evaluate a generalised ufunc (reductions and other functions over an axis).
     *
     * @param list<mixed> $inputs arrays (PHP arrays or NDArray)
     * @param int|list<int>|null $axis
     * @param array<string, mixed> $params by name; missing ones take their defaults
     */
    public static function gufunc(string $name, array $inputs, int|array|null $axis, bool $keepdims, array $params): NDArray|float|int|bool|array
    {
        $info = self::info($name);
        if ($info['kind'] !== 'gufunc') {
            throw new \ValueError("{$name} is not a generalised ufunc");
        }
        $label = self::label($name);
        $ffi = Library::ffi();
        $nin = $info['nin'];
        if (count($inputs) !== $nin) {
            throw new \ArgumentCountError("{$label}() takes {$nin} array argument(s)");
        }
        $arrs = [];
        foreach ($inputs as $a) {
            if ($a === null) {                                   // an omitted optional input (weights): 1.0
                $a = NDArray::full([], 1.0, DType::Float64);
            }
            $arr = $a instanceof NDArray ? $a : NDArray::array($a);
            if ($arr->dtype() === DType::Complex128) {
                throw new DTypeError("{$label}() does not support complex128 input");
            }
            $arrs[] = $arr;
        }
        if (($info['axis_single'] ?? 0) && is_array($axis)) {
            throw new \TypeError("{$label}(): axis must be an integer or null, not a list of axes");
        }
        if ($keepdims && ! ($info['keepdims'] ?? 1)) {
            throw new \ArgumentCountError("{$label}() does not take keepdims");
        }
        $axisInputs = $info['axis_inputs'];
        $masks = $ffi->new("uint64_t[{$nin}]");
        $metas = $ffi->new("tsr_array[{$nin}]");
        foreach ($arrs as $k => $arr) {
            $nd = $arr->ndim();
            $mask = 0;
            $whole = $k > 0 && ($nd === 0 || ($nd === 1 && $arrs[0]->ndim() > 1));
            if ($k < $axisInputs && $axis !== null && ! $whole) {
                foreach ((array) $axis as $ax) {
                    $a = $ax < 0 ? $ax + $nd : $ax;
                    if ($a < 0 || $a >= $nd) {
                        throw new \Tessero\Exceptions\ShapeError("{$label}(): axis {$ax} is out of bounds for an array of dimension {$nd}");
                    }
                    if ($mask & (1 << $a)) {
                        throw new \ValueError("{$label}(): repeated axis");
                    }
                    $mask |= 1 << $a;
                }
            } else {
                $mask = $nd >= 63 ? -1 : (1 << $nd) - 1;
            }
            $masks[$k] = $mask;
            $m = $arr->meta();
            FFI::memcpy(FFI::addr($metas[$k]), FFI::addr($m), FFI::sizeof($m));
        }
        $pvals = self::params($info, $params, $label);
        $np = max(1, count($pvals));
        $pc = $ffi->new("double[{$np}]");
        foreach ($pvals as $k => $v) {
            $pc[$k] = $v;
        }
        $nout = $info['nout'];
        $ond = $ffi->new("int32_t[{$nout}]");
        $osh = $ffi->new('int64_t[' . (32 * $nout) . ']');
        self::check($ffi->tsr_fn_gufunc_shape($info['id'], $nin, $metas, $masks, $keepdims ? 1 : 0, $pc, $ond, $osh), $label);
        $outs = [];
        $ometa = $ffi->new("tsr_array[{$nout}]");
        for ($o = 0; $o < $nout; $o++) {
            $shape = [];
            for ($d = 0; $d < $ond[$o]; $d++) {
                $shape[] = $osh[$o * 32 + $d];
            }
            $dt = ($info['out_int'] >> $o) & 1 ? DType::Int64 : (($info['out_bool'] >> $o) & 1 ? DType::Bool : DType::Float64);
            if ((($info['out_int_like'] ?? 0) >> $o) & 1 && in_array($arrs[0]->dtype(), [DType::Int64, DType::Int32, DType::UInt8], true)) {
                $dt = DType::Int64;                              // NumPy keeps integer results integer (nansum, nanmin, ptp ...)
            }
            $kp = $info['keep_param'] ?? -1;
            if ($o === 0 && $kp >= 0 && (($info['keep_mask'] >> (int) ($pvals[$kp] ?? 0)) & 1)
                && in_array($arrs[0]->dtype(), [DType::Int64, DType::Int32, DType::UInt8, DType::Bool], true)
                && $arrs[0]->size() > 0) {                       // an empty sample: nanquantile's NaN is float64
                $dt = $arrs[0]->dtype();                         // numpy.quantile's discrete methods return a sample
            }
            $outs[$o] = NDArray::empty($shape, $dt);
            $m = $outs[$o]->meta();
            FFI::memcpy(FFI::addr($ometa[$o]), FFI::addr($m), FFI::sizeof($m));
        }
        self::check($ffi->tsr_fn_gufunc($info['id'], $nin, $metas, $masks, $keepdims ? 1 : 0, $pc, $nout, $ometa), $label);
        unset($arrs);
        $res = [];
        foreach ($outs as $o => $arr) {
            $res[$info['outs'][$o]] = $arr->ndim() === 0 ? $arr->item() : $arr;
        }

        return $nout === 1 ? $res[$info['outs'][0]] : $res;
    }

    // ------------------------------------------------------------------ numpy.random.Generator methods

    /**
     * Draw variates with a Generator method of the registry (kind "random"), NumPy's stream exactly.
     *
     * @param CData $state the generator's PCG64 state (uint64_t[6])
     * @param list<mixed> $inputs the distribution's array arguments (numbers, PHP arrays or NDArray)
     * @param int|list<int>|null $size output shape; null: the broadcast shape of the inputs (a scalar for scalars)
     * @param array<string, mixed> $params enum parameters by name (dtype, method)
     */
    public static function random(string $name, CData $state, array $inputs, int|array|null $size, array $params = []): NDArray|float|int
    {
        $info = self::info($name);
        if ($info['kind'] !== 'random') {
            throw new \ValueError("{$name} is not a Generator method");
        }
        $label = 'Generator::' . explode('.', $name)[1];
        $ffi = Library::ffi();
        $ops = [];
        $shapes = [];
        foreach ($inputs as $a) {
            $arr = $a instanceof NDArray ? $a : NDArray::array($a, DType::Float64);
            if ($arr->dtype() === DType::Complex128) {
                throw new DTypeError("{$label}() does not support complex128 input");
            }
            $arr = $arr->dtype() === DType::Float64 ? $arr : $arr->astype(DType::Float64);
            $ops[] = $arr;
            $shapes[] = $arr->shape();
        }
        try {
            $bshape = $shapes === [] ? [] : NDArray::broadcastShapes(...$shapes);
        } catch (ShapeError) {
            throw new ShapeError("{$label}(): shape mismatch: objects cannot be broadcast to a single shape");
        }
        if ($size === null) {
            $shape = $bshape;
        } else {
            $shape = is_int($size) ? [$size] : array_values($size);
            foreach ($shape as $d) {
                if (! is_int($d) || $d < 0) {
                    throw new \ValueError("{$label}(): negative dimensions are not allowed");
                }
            }
            try {
                $ok = NDArray::broadcastShapes($bshape, $shape) === $shape;
            } catch (ShapeError) {
                $ok = false;
            }
            if (! $ok) {
                throw new \ValueError("{$label}(): size " . self::shapeStr($shape) . ' is not compatible with the parameters\' broadcast shape ' . self::shapeStr($bshape));
            }
        }
        $pvals = self::params($info, $params, $label);
        $f32 = false;
        foreach ($info['params'] as $k => $p) {
            if ($p['name'] === 'dtype') {
                $f32 = $pvals[$k] == 1.0;
            }
        }
        $dt = $info['out_int'] ? DType::Int64 : ($f32 ? DType::Float32 : DType::Float64);
        $out = NDArray::empty($shape, $dt);
        $nin = count($ops);
        $nop = $nin + 1;
        $nd = count($shape);
        $ptrs = $ffi->new("void*[{$nop}]");
        $strides = [];
        $keep = [];
        foreach ($ops as $k => $a) {
            $b = $a->shape() === $shape ? $a : $a->broadcastTo($shape);
            $keep[] = $b;
            $ptrs[$k] = $ffi->cast('void*', $b->ptr());
            array_push($strides, ...$b->strides());
        }
        $ptrs[$nin] = $ffi->cast('void*', $out->ptr());
        array_push($strides, ...$out->strides());
        $pc = $ffi->new('double[' . max(1, count($pvals)) . ']');
        foreach ($pvals as $k => $v) {
            $pc[$k] = $v;
        }
        self::check($ffi->tsr_fn_random($info['id'], $state, $pc, $nd, $nd > 0 ? Library::i64($shape) : null, $nop, $ptrs,
            $nd > 0 ? Library::i64($strides) : null, $dt->value), $label);
        unset($keep);

        return $size === null && $nd === 0 ? $out->item() : $out;
    }

    // ------------------------------------------------------------------ routines

    /**
     * Call a routine (functions whose output sizes depend on the data: unique, histogram, cov, ...).
     * Arguments go to the kernel as numbers, strings, arrays or null; array results are adopted without a copy.
     *
     * @param list<mixed> $args in registry order (defaults already applied by the façade)
     */
    public static function routine(string $name, array $args): mixed
    {
        $info = self::info($name);
        if ($info['kind'] !== 'routine') {
            throw new \ValueError("{$name} is not a routine");
        }
        $label = self::label($name);
        $ffi = Library::ffi();
        $n = count($args);
        $cargs = $ffi->new('tsr_arg[' . max(1, $n) . ']');
        $keep = [];
        $hold = [];
        $seq = array_flip($info['seq'] ?? []);
        foreach ($info['inplace'] ?? [] as $k) {
            // numpy.put, place, copyto ...: the array is modified in place, so it must be a writable NDArray
            $v = array_values($args)[$k] ?? null;
            if (! $v instanceof NDArray) {
                throw new \TypeError("{$label}(): argument \${$info['args'][$k]} must be an NDArray (it is modified in place), " . get_debug_type($v) . ' given');
            }
            if ($v->isReadonly()) {
                throw new \ValueError("{$label}(): assignment destination is read-only");
            }
        }
        foreach (array_values($args) as $k => $v) {
            if (isset($seq[$k]) && is_array($v)) {
                // a sequence argument (numpy's `arrays`, `*args`): one tsr_arg per item
                $items = array_values($v);
                $sub = $ffi->new('tsr_arg[' . max(1, count($items)) . ']');
                $hold[] = $sub;                       // owned by PHP: released with the variable
                foreach ($items as $j => $item) {
                    self::routineArg($ffi, $sub[$j], $item, $keep, $label);
                }
                $cargs[$k]->kind = 5;
                $cargs[$k]->count = count($items);
                $cargs[$k]->items = $ffi->cast('tsr_arg*', FFI::addr($sub[0]));
                continue;
            }
            self::routineArg($ffi, $cargs[$k], $v, $keep, $label);
        }
        $nres = max(1, $info['nout']);
        $res = $ffi->new("tsr_result[{$nres}]");
        $rc = $ffi->tsr_fn_routine($info['id'], $n, $cargs, $nres, $res);
        foreach ($keep as $b) {
            if ($b instanceof CData) {
                FFI::free($b);                        // the unmanaged string buffers
            }
        }
        unset($keep, $hold);
        self::check($rc, $label);
        $out = [];
        for ($k = 0; $k < $nres; $k++) {
            $r = $res[$k];
            if ($r->kind !== 0) {
                $out[$info['outs'][$k] ?? "r{$k}"] = self::resultValue($ffi, $r);
            }
        }

        if ($out === [] && $info['nout'] === 0) {
            return null;                                   // numpy.put, copyto ...: modified in place, returns None
        }

        return count($out) === 1 ? reset($out) : $out;
    }

    /**
     * A routine with a variadic argument (numpy's `*xi`): the positional arguments form that sequence, named
     * arguments (collected by the façade's `...$args`) fill the keyword-only arguments after it.
     *
     * @param array<int|string, mixed> $given
     */
    public static function routineVariadic(string $name, array $given): mixed
    {
        $info = self::info($name);
        $label = self::label($name);
        $var = $info['variadic'];
        $positional = [];
        $named = [];
        foreach ($given as $k => $v) {
            if (is_int($k)) {
                if ($named !== []) {
                    throw new \Error("{$label}(): cannot use positional argument after named argument");
                }
                $positional[] = $v;
            } else {
                $named[$k] = $v;
            }
        }
        $args = [];
        foreach ($info['args'] as $i => $argName) {
            $phpName = self::camelName($argName);
            if ($i < $var) {
                if ($positional === [] && ! array_key_exists($phpName, $named)) {
                    throw new \ArgumentCountError("{$label}(): argument \${$phpName} is required");
                }
                $args[] = array_key_exists($phpName, $named) ? $named[$phpName] : array_shift($positional);
                unset($named[$phpName]);
            } elseif ($i === $var) {
                $args[] = $positional;
                $positional = [];
            } elseif (array_key_exists($phpName, $named)) {
                $args[] = $named[$phpName];
                unset($named[$phpName]);
            } else {
                $args[] = self::defaultValue((string) ($info['defaults'][$argName] ?? 'None'));
            }
        }
        foreach ($named as $k => $_) {
            throw new \Error("Unknown named parameter \${$k}");
        }

        return self::routine($name, $args);
    }

    /** A registry default ("None", "True", "10", "'xy'") as a PHP value. */
    public static function defaultValue(string $d): mixed
    {
        return match (true) {
            $d === 'None' => null,
            $d === 'True' => true,
            $d === 'False' => false,
            $d === 'inf' => INF,
            $d === '-inf' => -INF,
            (bool) preg_match("/^'(.*)'$/", $d, $m) => $m[1],
            is_numeric($d) => str_contains($d, '.') || str_contains(strtolower($d), 'e') ? (float) $d : (int) $d,
            default => $d,
        };
    }

    private static function camelName(string $s): string
    {
        $parts = explode('_', $s);
        $out = array_shift($parts);
        foreach ($parts as $p) {
            $out .= $p === '' ? '_' : ucfirst($p);
        }

        return $out;
    }

    /** Fill one tsr_arg from a PHP value. @param list<mixed> $keep */
    private static function routineArg(FFI $ffi, CData $slot, mixed $v, array &$keep, string $label): void
    {
        if ($v === null) {
            $slot->kind = 0;
        } elseif (is_bool($v)) {
            $slot->kind = 4;
            $slot->num = $v ? 1.0 : 0.0;
        } elseif (is_int($v) || is_float($v)) {
            $slot->kind = 1;
            $slot->num = (float) $v;
            $slot->flags = is_int($v) ? 1 : 0;
            $slot->ival = is_int($v) ? $v : 0;
        } elseif (is_string($v)) {
            $buf = $ffi->new('char[' . (strlen($v) + 1) . ']', false);
            FFI::memcpy($buf, $v, strlen($v));
            $buf[strlen($v)] = "\0";
            $keep[] = $buf;
            $slot->kind = 2;
            $slot->str = $ffi->cast('char*', $buf);
        } else {
            $arr = $v instanceof NDArray ? $v : NDArray::array($v);
            $keep[] = $arr;
            $slot->kind = 3;
            $slot->flags = $v instanceof NDArray ? 0 : 2;     // bit 1: built from a PHP list (a Python sequence)
            $m = $arr->meta();
            FFI::memcpy(FFI::addr($slot->arr), FFI::addr($m), FFI::sizeof($m));
        }
    }

    /** A kernel result as a PHP value (kind 6, a sequence, becomes a list). */
    private static function resultValue(FFI $ffi, CData $r): mixed
    {
        if ($r->kind === 6) {
            $n = $r->arr->shape[0];
            $items = $ffi->cast('tsr_result*', $r->arr->data);
            $list = [];
            $failed = null;
            for ($j = 0; $j < $n; $j++) {
                try {
                    $list[] = self::resultValue($ffi, $items[$j]);
                } catch (\Throwable $e) {
                    $failed ??= $e;
                    if ($items[$j]->kind === 3) {
                        $ffi->tsr_free($items[$j]->arr->data, $items[$j]->bytes);
                    }
                }
            }
            $ffi->tsr_free($r->arr->data, $r->bytes);
            if ($failed !== null) {
                throw $failed;
            }

            return $list;
        }

        if ($r->kind === 2) {                                   // a string result (dtype name)
            $s = FFI::string($ffi->cast('char*', $r->arr->data));
            $ffi->tsr_free($r->arr->data, $r->bytes);

            return $s;
        }

        return match ($r->kind) {
            1 => (float) $r->num,
            4 => $r->num != 0.0,
            5 => (int) $r->ival,
            3 => self::adopt($r),
            default => null,
        };
    }

    /** Take ownership of a result block allocated by the kernel. */
    private static function adopt(CData $r): NDArray
    {
        $a = $r->arr;
        $shape = [];
        for ($d = 0; $d < $a->ndim; $d++) {
            $shape[] = $a->shape[$d];
        }
        $buffer = Buffer::adopt(Library::address($a->data), $r->bytes);

        return new NDArray($buffer, DType::from($a->dtype), $shape);
    }

    /**
     * Parameter values in registry order: numbers as given, booleans 0/1, null NAN, enum names by index.
     *
     * @param array<string, mixed> $info
     * @param array<string, mixed> $given
     * @return list<float>
     */
    public static function params(array $info, array $given, string $label): array
    {
        $vals = [];
        $known = [];
        foreach ($info['params'] as $p) {
            $name = $p['name'];
            $known[$name] = true;
            $has = array_key_exists($name, $given);
            $v = $has ? $given[$name] : null;
            if (isset($p['enum'])) {
                $choice = $has && $v !== null ? (string) $v : $p['enum'][0];
                $idx = array_search($choice, $p['enum'], true);
                if ($idx === false) {
                    throw new \ValueError("{$label}(): {$name} must be one of '" . implode("', '", $p['enum']) . "', got '{$choice}'");
                }
                $vals[] = (float) $idx;
                continue;
            }
            if (! $has) {
                $d = $p['default'] ?? 'None';
                $vals[] = match ($d) {
                    'None' => NAN, 'True' => 1.0, 'False' => 0.0, 'inf' => INF, '-inf' => -INF, default => (float) $d,
                };
                continue;
            }
            $vals[] = match (true) {
                $v === null => NAN,
                is_bool($v) => $v ? 1.0 : 0.0,
                is_int($v) || is_float($v) => (float) $v,
                default => throw new \TypeError("{$label}(): {$name} must be a number, bool or null"),
            };
        }
        foreach ($given as $k => $_) {
            if (! isset($known[$k])) {
                throw new \Error("Unknown named parameter \${$k} for {$label}()");
            }
        }

        return $vals;
    }

    // ------------------------------------------------------------------ helpers

    public static function label(string $name): string
    {
        // the façade's class and method, as the extension labels it (Np::putAlongAxis, Generator::standardNormal)
        $parts = explode('.', $name, 2);
        if (count($parts) < 2) {
            return $name;
        }

        return ($parts[0] === 'random' ? 'Generator' : ucfirst($parts[0])) . '::' . self::camelName($parts[1]);
    }

    private static function check(int $rc, string $what): void
    {
        if ($rc >= 0) {
            return;
        }
        $msg = (string) Library::ffi()->tsr_fn_error();
        if ($msg !== '') {
            throw match ($rc) {
                -6 => new ShapeError("{$what}: {$msg}"),
                -3 => new \Tessero\Exceptions\MemoryError("{$what}: {$msg}"),
                -2 => new \Tessero\Exceptions\IndexError("{$what}: {$msg}"),
                -4 => new DTypeError("{$what}: {$msg}"),
                default => new \ValueError("{$what}: {$msg}"),
            };
        }
        Library::check($rc, $what);
    }

    /** @param list<NDArray> $ops */
    private static function overlaps(NDArray $out, array $ops): bool
    {
        foreach ($ops as $a) {
            if ($a->buffer() === $out->buffer() && ($a->offset() !== $out->offset() || $a->strides() !== $out->strides())) {
                return true;
            }
        }

        return false;
    }

    private static function castInto(NDArray $r, NDArray $out, string $label): NDArray
    {
        $kind = static fn (DType $d): int => match ($d) {
            DType::Bool => 0, DType::UInt8, DType::Int32, DType::Int64 => 1, DType::Float32, DType::Float64 => 2, DType::Complex128 => 3,
        };
        if ($kind($r->dtype()) > $kind($out->dtype())) {
            throw new DTypeError("{$label}(): cannot cast the result from {$r->dtype()->name()} to {$out->dtype()->name()}");
        }
        try {
            $b = $r->shape() === $out->shape() ? $r : $r->broadcastTo($out->shape());
        } catch (ShapeError) {
            throw new ShapeError("{$label}(): result of shape " . self::shapeStr($r->shape()) . ' does not fit an output of shape ' . self::shapeStr($out->shape()));
        }

        return $out->assign($b);
    }

    /** @param list<int> $s */
    private static function shapeStr(array $s): string
    {
        return '(' . implode(', ', $s) . (count($s) === 1 ? ',' : '') . ')';
    }
}
