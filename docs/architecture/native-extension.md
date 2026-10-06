# The native extension

`ext/` is a Zend extension that makes the array a first-class PHP object
backed directly by `libtessero`. It is about 3 000 lines of C in four files.

| File | Contents |
|---|---|
| `php_tessero.h` | module globals (threads, memory budget, epsilon), `tsr_obj`, class entries, shared helpers |
| `tessero.c` | module lifecycle (MINIT/RINIT/MINFO), INI entries, exception classes, `Operand`, `Engine` settings |
| `tessero_ndarray.c` | `NDArray`: conversion to/from PHP values, dtype promotion, object handlers, iterator, methods |
| `tessero_solvers.c` | `Engine` solvers: LP/MILP, MDP, FFT, random streams |
| `tessero_ufunc.c` | `Math`: the ufunc registry, typed inner loops and the broadcasting/threading engine ([Ufuncs](../guide/ufuncs.md)) |
| `tessero_memmap.c` | `NDArray::memmap()`, `flush()`, mapping release ([Memmap](../guide/memmap.md)) |
| `tessero_arginfo.h` | argument info for reflection and named arguments |
| `tessero.map` | linker version script: export only `get_module` |
| `libtessero/` | verified copy of `csrc/` (`tools/sync-ext.sh --check` in CI) |

## Object layout

```c
typedef struct {
    tsr_array    a;           /* the kernel's metadata, used directly by every call */
    void        *block;       /* heap owner only (tsr_alloc) */
    int64_t      block_bytes;
    zend_object *base;        /* view → root owner (refcounted) */
    uint32_t     flags;       /* TSR_F_MMAP, TSR_F_READONLY (read on the root) */
    tsr_map     *map;         /* memmap root only: address, length, fd, path, mode */
    zend_object  std;         /* last, as the Zend API requires */
} tsr_obj;
```

An array's memory comes from exactly one of two owners: a `tsr_alloc` block
(`block`) or a file mapping (`map`). Views have neither and point at the root.
`free_obj` releases whichever the object owns: `tsr_free`, or `munmap` + `close`
through `tsr_memmap_release()`. Flags live on the root, so a view of a
read-only memmap is read-only too. Every write path (`$a[...] = `, `assign()`,
`out:` of `Math` and of the element-wise cores) goes through
`tsr_require_writable()`.

Objects are created with `zend_object_alloc(sizeof(tsr_obj), ce)`, and
`Z_TSR_P(zv)` recovers the struct from a zval by offset. Because `tsr_array`
lives inside the object, a kernel call needs no marshalling: the extension
passes `&o->a` or its fields straight to C. That is why a call costs about
0.3 µs instead of 9 µs through FFI.

See [Memory model](memory-model.md) for ownership and views.

## Handlers

| Handler | Behaviour |
|---|---|
| `free_obj` | frees the block (owner) or releases the root owner (view) |
| `clone_obj` | compact deep copy; a clone is always an owner |
| `do_operation` | `+ - * / ** %` with arrays, PHP scalars and PHP arrays on either side, broadcasting and NEP 50 promotion; compound assignment rebinds |
| `compare` | `==`/`!=`: same dtype, same shape and identical element bytes (so `-0.0` ≠ `0.0`, and a NaN equals itself). Use `eq()` for NumPy-style element-wise comparison |
| `read_dimension` | int → element or sub-array view; string → slice via the C parser (`'1:, ::-1'`, `...`, `None`); bool array → mask copy; int array/list → row gather |
| `write_dimension` | the same keys; scalar or array broadcast into the target |
| `has_dimension` / `unset_dimension` | `isset($a[i])` bounds check; `unset` throws (arrays have fixed size) |
| `count_elements` | `count($a)` = length of the first axis |
| `get_iterator` | native iterator over the first axis (views for ≥ 2-D, scalars for 1-D) |
| `get_gc` | reports the root owner to the cycle collector |
| `get_debug_info` | `var_dump` shows dtype, shape, strides, view flag and a preview of the data |

`NDArray` implements `IteratorAggregate`, `Countable` and `JsonSerializable`.
PHP requires a `Traversable` class to implement `Iterator` or
`IteratorAggregate`, so `getIterator()` exists and wraps the internal iterator
(`zend_create_internal_iterator_zval`).

## Conversions

- **PHP → native**: one pass computes shape and the result dtype (int, float,
  bool, NEP 50 weak scalars), a second pass fills the block. Packed PHP
  arrays are read directly from their bucket storage.
- **Native → PHP**: builds packed arrays with preallocated size
  (`zend_new_array` + `ZEND_HASH_FILL_PACKED`).
- **JSON**: `smart_str` with `php_gcvt(value, 17, '.', 'e', buf)` and the same
  post-processing as `json_encode` (`serialize_precision = -1` shortest
  round-trip, `.0` kept for floats). The output is byte-identical to
  `json_encode($a->toList(), JSON_PRESERVE_ZERO_FRACTION)`, which the tests check.
- **serialize**: `__serialize` returns `['v' => 1, 'dtype', 'shape', 'data' => bytes]`.
  `__unserialize` validates the version, dtype, shape and byte length.

## Operator dispatch for userland classes

`Tessero\Ext\Operand` is an empty abstract class whose handlers include
`do_operation`. For an operation involving a subclass instance, the handler
calls the method named after the operator (`add`, `sub`, `mul`, `div`, `pow`,
`mod`) on the left operand, or the reflected method (`rsub`, `rdiv`, `rpow`)
on the right. The FFI package's `NDArray` extends it when the extension is
loaded, which is how FFI arrays get operators.

## Ufunc engine

`Math` has one static method per registry entry (74), all sharing two C
handlers that look up their own name. Kernel ufuncs call the same
`binary_core`/`unary_core` as the operators, extended with an optional
destination (`tsr_binary_into`/`tsr_unary_into`). When the destination's
dtype and shape match and it does not partially overlap an input, the kernel
writes straight into it. Loop ufuncs use typed inner loops,
`loop(char **args, int64_t n, const int64_t *steps)`. The engine resolves a
loop from the promoted dtype, broadcasts, coalesces with `tsr_iter`, and calls
the loop per inner run across OpenMP threads. See [Ufuncs](../guide/ufuncs.md#how-it-runs)
and [ADR-0009](adr/0009-ufunc-engine-and-memmap.md).

## Settings and request lifecycle

- `MINIT` registers classes and INI entries. INI changes call
  `tsr_set_threads`/`tsr_set_budget` immediately.
- `RINIT` re-applies the INI values at every request start, so a value
  changed at runtime (`Engine::setMaxThreads()`) never leaks into the next
  request on the same worker.
- Module globals use the standard `ZEND_MODULE_GLOBALS` mechanism, so they are
  per thread under ZTS. The kernel's thread count and budget are
  process-wide ([ADR-0006](adr/0006-process-wide-kernel-settings.md)).

## Compatibility

PHP 8.2 to 8.5. The one API difference handled is `ZEND_RAW_FENTRY`, which
takes 6 arguments from 8.4 and 4 before (`TSR_FE` macro). Built and tested
here on PHP 8.4.21 NTS; the other versions are CI jobs.
