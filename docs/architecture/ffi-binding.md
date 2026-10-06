# The FFI binding

`tessero/tessero` is pure PHP. It reaches `libtessero` and OpenBLAS through
PHP's FFI extension.

## Loading

`Native\Library::ffi()` returns the `FFI` instance, loaded once per process in
one of two modes:

| Mode | When | How |
|---|---|---|
| `cdef` | `ffi.enable=1`, or the CLI | `FFI::cdef(file_get_contents('tessero.h'), $path)` |
| `scope` | `ffi.enable=preload` under FPM/Apache | `FFI::scope('tessero')`, bound by `resources/preload.php` at server start |

The library path comes from `TESSERO_LIB` or
`lib/<os>[-musl]-<arch>/libtessero.<so|dylib|dll>`. `Library::mode()` and
`Tessero::info()['load_mode']` report which mode is in use.

`Native\Blas` does the same for OpenBLAS + LAPACKE (scope `tessero_blas`). It
searches `TESSERO_BLAS`, the system OpenBLAS and the copy bundled in SciPy's
wheels (whose symbols carry a `scipy_` prefix). It refuses ILP64 builds
(`libopenblas64_`, NumPy's bundled copy), because they take 64-bit integers and
passing 32-bit ones would corrupt memory.

## Why the preload mode exists

`ffi.enable=1` in a web SAPI lets any code call arbitrary C, so PHP's default
is `preload`: FFI only from code compiled at server start. Tessero supports
the default rather than asking users to weaken it. The cost is that every
Tessero class must itself be preloaded, because in this mode PHP checks the
*calling* function, not just the library. The preload script loads every file
under `src/`. `tests/e2e/preload.sh` proves the mode under a real non-CLI SAPI
with `FFI::cdef` forbidden.

## Passing pointers

PHP's FFI has two defects that matter here (see
[Upstream bugs](../project/upstream-bugs.md)):

1. Passing a pointer-arithmetic temporary (`$p + 8`) as an argument breaks
   later arithmetic on `$p`.
2. `FFI::cast('uintptr_t', $voidPtr)` can return a different value from a cast
   through `char*`.

Tessero therefore never does `CData` arithmetic:

- `Buffer` stores the block's address as a PHP **integer**
  (`Library::address()`, which casts through `char*`).
- Each call builds its pointer from `address + byte offset` with
  `FFI::cast('char*', $int)` and then casts to the parameter's type
  (`double*`, `int64_t*`, `uint8_t*`). A `void*` parameter would accept anything,
  but typed parameters need the explicit cast.

## Moving data across the boundary

| Direction | Method | Cost for 1M float64 |
|---|---|---:|
| PHP array → native | `pack('d*', ...$chunk)` in 262 144-element chunks, one `FFI::memcpy` per chunk | 26 ms |
| native → PHP list | `FFI::string` + `unpack` in 4M-element chunks (≤ 32 MiB each) | 65 ms |
| native → JSON | `tsr_array_json` in C (integer fast path; shortest round-trip floats) | 61 ms |
| native → bytes | `FFI::string` | ~1 ms |

The chunk sizes bound the transient memory of a conversion.

## Per-call overhead

A kernel call through FFI costs about 9 µs: argument conversion, the `CData`
casts and the call itself. This is why the design avoids per-element calls,
and why the [native extension](native-extension.md) exists for workloads
made of many small calls.

## Operators

PHP classes cannot overload operators. When `ext-tessero` is loaded,
`Tessero\Internal\OperandBase` becomes an alias of `Tessero\Ext\Operand`, a
class the extension gives a `do_operation` handler. `NDArray` extends
`OperandBase`, so `$a + $b` dispatches to `add()`. Without the extension
`OperandBase` is an empty class, and the methods still work.

## JIT interaction

The FFI package's PHP code (shape arithmetic, argument building, the
optimisers in `Optimize\`) runs under the opcache JIT. Three JIT
miscompilations were found during development and are avoided with plain code
shapes: no compound assignment on array properties in hot loops, no list
destructuring in hot loops, loop counters of very large methods kept in
properties, and `CData` arrays copied out in one call instead of read element
by element. CI runs every suite with the JIT
off, `function` and `tracing`.
