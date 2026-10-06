# Architecture overview

Tessero is one numerical kernel with three ways in.

```
 ┌──────────────────────────── application ───────────────────────────────┐
 │   Laravel app  ·  Symfony app  ·  CLI job  ·  queue worker             │
 └───────┬─────────────────────────────┬──────────────────────────────────┘
         │                             │
 ┌───────▼────────────────┐            │
 │ tessero/laravel        │            │
 │ ServiceProvider        │            │
 │ TesseroManager ────────┼──── picks ext, else FFI (identical results)
 │ Casts · Rule · Facade  │            │
 └───────┬───────┬────────┘            │
         │       │                     │
 ┌───────▼───┐ ┌─▼────────────────────────────────┐
 │ ext-      │ │ tessero/tessero (PHP)            │
 │ tessero   │ │ NDArray · Linalg · Fft · Random  │
 │ (Zend C)  │ │ Sparse · Optimize · Mdp · Io     │
 │ NDArray   │ │        │ FFI (cdef or preload scope)
 │ Engine    │ │ Native\Library · Buffer · Blas   │
 │ Operand ◄─┼─┤ (NDArray extends Operand when ext loaded → operators)
 └─────┬─────┘ └────────┬─────────────────┬───────┘
       │ compiled in    │ dlopen          │ dlopen
 ┌─────▼────────────────▼──────┐   ┌──────▼──────────────┐
 │ libtessero (C11)            │   │ OpenBLAS + LAPACKE  │
 │ tessero.h — the whole ABI   │   │ (LP64, optional)    │
 │ alloc · array · elementwise │   └─────────────────────┘
 │ reduce · index · sort · fft │
 │ rng · sparse · matmul       │
 │ lp · mdp                    │
 │ OpenMP (optional)           │
 └─────────────────────────────┘
```

## Layers

| Layer | Language | Responsibility | Lines |
|---|---|---|---:|
| `csrc/` libtessero | C11 | every loop over elements; allocator; solvers | ~3 600 |
| `src/` FFI package | PHP 8.2 | shapes, strides, broadcasting rules, dtype promotion, API; LAPACK calls | ~6 300 |
| `ext/` extension | C (Zend API) | the same array model as a native PHP class; operators, dimensions, iteration, JSON, serialisation | ~3 000 |
| `laravel/` bridge | PHP | framework integration; backend selection; per-request settings | ~630 |

## Design rules

1. **One header is the ABI.** `csrc/include/tessero.h` holds declarations only,
   so the FFI package passes it verbatim to `FFI::cdef`/`FFI::load`, and the
   extension includes it. Op codes are mirrored in `Native\Abi`. A unit test
   fails if they drift. See [C kernel](c-kernel.md).
2. **No PHP loops over elements.** Every array operation is one or a few C
   calls. PHP computes only shapes and strides. The same holds for solvers:
   an LP, a MILP or a whole MDP solve is one call.
3. **Strides are bytes.** Views can be reversed (negative stride) or broadcast
   (zero stride) without copying. Kernels never assume contiguity; the
   iterator coalesces dimensions so the contiguous case becomes one flat
   loop.
4. **Ownership is explicit.** Memory blocks are freed exactly once, when the
   last array or view referring to them is destroyed. See [Memory model](memory-model.md).
5. **NumPy is the specification.** Semantics (broadcasting, NEP 50
   promotion, slicing, integer wrap-around, NaN ordering, random streams, JSON
   formatting) are taken from NumPy/SciPy and checked against them by
   generated fixtures. See [Testing and quality](testing-and-quality.md).
6. **Errors are typed and early.** Kernels return negative codes; bindings map
   them to typed exceptions ([Error codes](../reference/error-codes.md)). Inputs
   are validated before C runs.
7. **Determinism.** Results do not depend on thread count. Reductions keep a
   fixed summation order. Random streams are fixed by seed.
8. **Two backends, one kernel.** The extension compiles the same C sources
   (`ext/libtessero` is a verified copy), so numerical behaviour is shared by
   construction. The parity suite checks it anyway.

## Request lifecycle (Laravel, extension backend)

1. Worker boots: the extension's MINIT registers classes; RINIT applies
   `tessero.threads` and `tessero.memory_budget` for this request. The service
   provider applies `config/tessero.php` on top, and does so again on every
   Octane `RequestReceived` or `JobProcessing` event.
2. The controller validates input (`NumericArray`) and builds arrays
   (`Tessero::array`): one pass converts the PHP array into a 64-byte-aligned
   native block, counted against the budget.
3. Operators and methods call kernels directly on the `tsr_array` metadata
   embedded in each PHP object. Views share the block and hold a reference to
   the owner object.
4. `response()->ndarray($result)` writes JSON from native memory.
5. At the end of the request PHP destroys the objects. `free_obj` releases each
   block as its last reference disappears, and the counter returns to baseline.

## Where to read next

- [C kernel](c-kernel.md): iterator, SIMD dispatch, threading, file map
- [Memory model](memory-model.md): allocator, budget, views and ownership in both bindings
- [FFI binding](ffi-binding.md): loading modes, preload, pointer handling, PHP engine defects
- [Native extension](native-extension.md): object layout, handlers, operator dispatch
- [Numerical methods](numerical-methods.md) and [Solvers](solvers.md): algorithms and their accuracy
- [Decision records](adr/index.md): why things are the way they are
- [Prompt-to-code map](prompt-mapping.md): where each part of the original design brief is implemented
