# 0016. A central input gate before every registry call

- Status: Proposed
- Date: 2026-10-24
- Amends: 0011 (registry entry points)

## Context

Every registry routine validated its own arguments. The 2026-10 fuzz campaigns found dozens of memory faults
whose root cause was one routine missing one check: a negative or overflowing dimension, a NULL data pointer, a
stride that walked off its block, a sequence with NULL items. The registry entry points (`tsr_fn_routine`,
`tsr_fn_ufunc`, `tsr_fn_gufunc`, `tsr_fn_gufunc_shape`, `tsr_fn_random`, `tsr_fn_rvs`) accept these
structures from three callers: the PHP FFI binding, the Zend extension, and native code that links
`libtessero` directly (Talario, Kinetix). The owner's directive: sanitize inputs before code executes.

## Decision

Each entry point runs a structural gate (`csrc/src/fn_gate.c`) before it dispatches. No routine, element loop
or core function runs on input the gate refuses; the call returns `TSR_EARG`, `TSR_EDIM` or `TSR_ETYPE` and
`tsr_fn_error()` names the argument and the problem.

What a call that passes the gate guarantees:

- every `tsr_arg` has a defined kind (0-5) and only the flag bits defined for it (bit 0 on numbers, bit 1 on
  arrays); an integer number's `num` equals `(double)ival`;
- strings are non-NULL and NUL-terminated within 2^31 - 1 bytes (routines keep lengths in `int`);
- arrays have a valid dtype, `0 <= ndim <= 32`, non-negative dimensions, an element count and byte extent that
  fit in int64 (the count at most INT64_MAX / 16), a non-NULL data pointer when they have elements, elements
  aligned to their type (at most 8 bytes), every element at or after the start of its block (`data + offset`
  plus the most negative stride reach is `>= data`), and no wrap of the address space; output arrays and output
  operands have no zero stride on an axis longer than 1 (their elements would alias: parallel workers would race on
  one address, and the work would no longer be bounded by any memory the caller owns). Inputs may broadcast;
- sequences hold at most 2^24 items, nest at most 4 deep, and have non-NULL items when non-empty;
- strided operand sets (ufunc, random, rvs) have non-negative dimensions, non-NULL operands, first elements and
  steps aligned to their type, and stride extents that do not overflow; `tsr_fn_random` also checks its operand count and output dtype;
- `tsr_fn_gufunc_shape` checks shapes only (it reads no data); `tsr_fn_gufunc` also checks every output array.

What stays each routine's job (the gate cannot know it): the meaning of values (domains, integrality of float
parameters, NaN policy), agreement between arguments (matching lengths, broadcastable shapes), and the end of a
caller's block: the ABI carries no block size, so an array whose elements extend past the memory the caller
owns cannot be detected. Per-routine validation stays as defence in depth; none is removed.

The checks are O(nargs + ndim) per call (strings: one bounded scan) and never read array elements.

## Consequences

- A whole class of findings (malformed structures reaching a routine) is closed at one place for every routine,
  present and future, and for native callers as well as the PHP bindings.
- `tsr_fn_rvs` sized its scratch as `(np + 1) * n * 8` without an overflow check; it now uses
  `tsr_bytes_n` and refuses an impossible size with `TSR_ENOMEM`.
- `fuzz/fuzz_abi.c` fuzzes the structures themselves: anything the harness classifies invalid is sent to a
  random registered routine and must be refused, so a routine reached past the gate fails the run.
- Arrays with misaligned elements are refused. NumPy allows unaligned arrays, but Tessero's kernels load through
  typed pointers, so they were undefined behaviour before; both bindings always pass aligned arrays.

## Alternatives considered

- Per-routine validation only (the status quo): every new routine had to repeat every check, and the fuzzers
  kept finding the ones that were missed.
- Declaring per-parameter value rules in the registry metadata and enforcing them centrally: useful, but it
  changes error messages that parity fixtures pin, and it is semantic rather than structural. Left for a later
  record.
