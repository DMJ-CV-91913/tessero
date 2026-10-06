# 0013. Registry routines take sequences, variadic and in-place arguments

- Status: Accepted
- Date: 2026-09-28
- Amends: [0011](0011-kernel-function-registry.md) (the routine calling convention), [0004](0004-native-allocator-and-budget.md) (allocations without a budget)

## Context

NumPy's array construction and manipulation functions are routines, but their signatures don't fit the
registry's first calling convention. That convention allowed numbers, strings, arrays and null as arguments and
a fixed set of named results:

- `concatenate(arrays, ...)`, `stack`, `select(condlist, choicelist)` and `choose` take a sequence of arrays
  that may differ in shape. A PHP list of such arrays cannot be converted to one NDArray.
- `meshgrid(*xi, indexing=...)`, `atleast_1d(*arys)`, `broadcast_arrays(*args)`, `ix_` and `gradient(f,
  *varargs)` take positional varargs followed by keyword-only options.
- `split`, `nonzero`, `unravel_index`, `meshgrid` and `broadcast_shapes` return a tuple or list whose length
  depends on the data.
- `put`, `place`, `putmask`, `copyto`, `fill_diagonal` and `put_along_axis` modify their first argument and
  return None.
- Some conversions depend on where a value came from. NumPy treats a Python sequence of floats given as
  indices by `int()`, but raises on a float ndarray. A Python int must keep its exact value above 2^53.

## Decision

The routine ABI (`tsr_arg`, `tsr_result` in `csrc/include/tessero.h`) is extended. Old routines are unaffected.

- **Sequence arguments.** `tsr_arg` kind 5 carries `count` items at `items`, each an ordinary `tsr_arg`. In a
  routine's argument list, `name[]` marks a sequence argument. The bindings pass a PHP array given there item
  by item. An NDArray given there is passed as one array, and the kernel iterates it along its first axis, as
  NumPy does.
- **Variadic arguments.** `*name` marks NumPy's `*args`. The façade is `f(mixed ...$name)`. Positional
  arguments form the sequence, and named arguments fill the keyword-only options that follow. The extension
  registers a variadic internal function and reads the options from its extra named parameters.
- **Sequence results.** `tsr_result` kind 6 holds `arr.shape[0]` results in a `tsr_alloc` block. An OUTS list
  starting with `*` declares one. The bindings return a PHP list, and 0-d items become PHP numbers.
- **In-place arguments.** `&name` marks an array the routine writes into. Both bindings pass it only when it is
  an NDArray that is not read-only, and otherwise raise `TypeError` or `ValueError`. The kernel then writes
  through the array's strides. A routine with no results returns `null`.
- **Argument provenance.** For a number, `flags` bit 0 means it was a PHP int, whose exact value is in `ival`.
  For an array, bit 1 means it was built from a PHP list, not passed as an NDArray. This lets the kernel apply
  NumPy's conversions: weak Python scalars under NEP 50, `int()` of a float index list, and `'safe'` casting
  for ndarray values in `putmask` and `place`.
- **Exact integer results.** `tsr_result` kind 5 carries the value in `ival` as well as in `num`, so a 0-d
  integer result (`trace` of an int64 array, `ravel_multi_index`) keeps its exact value above 2^53.
- **Error classes.** A routine's error code selects the PHP class in both bindings: `TSR_EINDEX` is
  `IndexError` (`IndexException` in the extension), `TSR_ETYPE` is `DTypeError` (`DTypeException`),
  `TSR_ESHAPE` is `ShapeError`, `TSR_ENOMEM` is `MemoryError`, and any other code is PHP's `ValueError`. The
  message names the façade method as PHP code calls it (`Np::putAlongAxis`) on both backends.
- **Output dtypes that depend on a parameter.** A generalised ufunc can declare `fn_keep`: output 0 keeps an
  integer or bool input's dtype for some values of an enum parameter. `numpy.quantile` returns a sample for its
  discrete methods (`inverted_cdf`, `closest_observation`, `lower`, `higher`, `nearest`) in the input's dtype,
  and exactly for int64; the other methods raise `TypeError` for a bool input. The registry info carries
  `keep_param` and `keep_mask`, and both bindings allocate the output accordingly.
- **Defaults.** The extension passes the registry's default for every omitted trailing argument, as the FFI
  façades do, so a kernel routine sees the same arguments from both bindings.
- **Copies, not views.** Where NumPy returns a view (`flip`, `swapaxes`, `moveaxis`, `diagonal`,
  `broadcast_to`, the `atleast_*` family), the registry routine returns a fresh array with the same shape,
  dtype and values. The NDArray classes keep their own view-returning methods (ADR 0005).
- **Allocations without a budget.** With no memory budget set, `tsr_alloc` refuses a single block larger than
  the machine's physical memory, and the caller raises `MemoryError`. Without this, an overcommitting
  allocator hands out the block and the process is killed when the block is filled. This is the policy
  `tsd::alloc_count` already applied in the C++ modules.

## Consequences

- About 120 NumPy functions (`csrc/src/np_shape.c`, `np_numeric.c`) are implemented once in C and reach both
  backends with the same names, arguments, dtypes and errors. Fixtures verify them
  (`tools/parity/fixtures_np_shape.py`, `fixtures_np_numeric.py`).
- A caller who relies on NumPy's view semantics, for example writing through the result of `np.flip` to
  change the input, gets a copy from `Np::flip`. For that, the NDArray methods remain.
- The ABI change is additive. The header grows `tsr_arg` by three fields and `tsr_result` by one, and both
  bindings are built from the same header (the FFI binding parses it, the extension compiles it).
- Float32 inputs to the numeric routines are computed in double and rounded to float32. NumPy computes some of
  them in float32 arithmetic, so float32 results can differ in the last bit. The fixtures use float64, int64
  and bool.

## Alternatives considered

- **Pack a sequence into one padded array with a length vector.** This loses each item's dtype and shape and
  needs a second convention for results.
- **Return views that share memory with PHP-owned buffers.** Each binding would need its own lifetime rules
  for kernel-created views. Views stay in the NDArray classes, where those rules exist.
