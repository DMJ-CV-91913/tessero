# Security

## Threat model

Tessero is a computation library. It opens no sockets, spawns no processes
(apart from `tessero build`, which runs `make`), reads files only when you call
`Npy::load`/`loadZ`, and evaluates no code from data. The relevant risks are:

1. **Resource exhaustion** from attacker-controlled sizes: huge arrays,
   huge LP/MDP models, many small requests that each allocate.
2. **Memory safety** in the C kernel and the extension: out-of-bounds reads or
   writes from malformed input.
3. **FFI exposure** when `ffi.enable=1` is used in web SAPIs.
4. **Unsafe deserialisation** of arrays from caches, queues or files.

## 1. Resource exhaustion

- **Always set a native memory budget** in web-facing processes
  (`tessero.memory_budget`, `TESSERO_MEMORY_BUDGET`). Without one, a single
  request can allocate until the OOM killer ends the worker.
- **Validate sizes before building arrays.** `NumericArray(maxElements: …)`
  rejects oversized input before any allocation. Its check is linear in the
  input size and allocates nothing.
- **Bound solver work** for user-supplied models: `maxiter`, `nodeLimit`,
  `maxIter`, and cap the number of states, constraints and variables at the
  API boundary. A dense LP tableau grows as m·(n+m).
- `Npy::load` reads the whole file into PHP memory (so `memory_limit`
  applies), then checks the header's declared shape against the data length
  before building the array. Check file sizes before loading untrusted files.

## 2. Memory safety

- Every entry point validates shapes, dtypes and index ranges before C code
  runs. Kernels receive only well-formed metadata.
- Solvers validate their inputs: LP/MILP reject NaN/INF data; MDP input must
  be finite and well-formed CSR (`indptr` monotone,
  ending at `nnz`; indices in range), with rows summing to 1 unless the
  caller passes `validate: false`.
- `unserialize()`, `fromBytes()` and `.npy` loading check, before any
  allocation, that every dimension is non-negative, that the element count
  cannot overflow, and that the byte length matches dtype × shape. A crafted
  payload raises an exception; it can never describe more memory than it
  carries.
- The C test suite, the extension's phpt suite and the parity suite run under
  AddressSanitizer and UndefinedBehaviorSanitizer in CI. The kernel's integer
  arithmetic wraps by definition (NumPy semantics) and is never undefined
  behaviour.
- Coverage-guided fuzzing (libFuzzer, in CI) covers every parser of untrusted
  text or bytes: the slice parser, the `.npy` header parser, memory-map
  argument planning and the JSON writer. NumPy-generated slice fixtures check
  the results as well.
- The [memory-safety review](../project/memory-safety-review.md) lists the
  safeguards, the defects fixed and the residual risks.

## 3. FFI exposure

`ffi.enable=1` in a web SAPI lets **any** PHP code in the application call
arbitrary C functions, which turns any code-injection bug into native code
execution. Use:

- the **extension** (no FFI needed), or
- `ffi.enable=preload` with Tessero's preload script. Only preloaded code can
  use FFI, and the only libraries bound are libtessero and BLAS. The script
  writes its temporary header with an exclusive, private file (`tempnam`,
  0600) and deletes it after loading, so another local user cannot redirect
  which library is loaded. The FFI package also refuses a libtessero binary
  whose version differs from the PHP code.

Keep `ffi.enable=1` to the CLI.

## 4. Memory-mapped files

- `NDArray::memmap()`, `openMemmap()` and `load($path, $mode)`, on both
  backends, resolve paths through `open_basedir` and refuse stream
  wrappers. Do not pass user-supplied paths without validating them against an
  allow-list, as you would for `fopen()`.
- Mode `'r'` maps pages read-only in addition to the library's own checks
  (every write path, including `out:` arguments and `shuffle`), so a
  read-only array cannot be written even through a bug.
- If a mapped file is truncated by another process, touching the missing pages
  raises SIGBUS and kills the worker. Map only files your application controls,
  and never map files that untrusted parties can modify.
- Mapped bytes are outside `memory_budget`. Cap file sizes you accept for
  mapping, as you would cap request sizes.

## 5. Deserialisation

- `serialize()`d arrays contain only dtype, shape and raw bytes, and
  `__unserialize` validates them. They are safe to store in caches and
  queues you control. As with any `unserialize()`, never unserialise
  data from untrusted sources: other classes' gadgets are the risk, not
  Tessero's.
- The `binary` Eloquent cast stores `{dtype, shape, b64}` JSON and validates
  the decoded length.
- `.npy` headers are parsed with a strict grammar. They are never evaluated,
  and object/pickle dtypes are rejected.

## Supply chain

- Prebuilt binaries are produced by the CI `native` job from tagged sources.
  Release notes list SHA-256 checksums (see [Releasing](../project/releasing.md)).
- To avoid prebuilt binaries entirely, build `libtessero` from source
  (`tessero build`) and set `TESSERO_LIB`, or use the extension built from source.
- The ziggurat tables are NumPy's (BSD-3-Clause). There are no other third-party
  sources in the kernel.

## Reporting a vulnerability

Follow `SECURITY.md` at the repository root: report privately, never in a
public issue.
