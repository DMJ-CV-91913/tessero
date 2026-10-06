# Architecture decision records

Each record states a decision, the context it was made in, and its
consequences. Records are not edited after acceptance. A changed decision gets
a new record that supersedes the old one.

| # | Decision | Status |
|---|---|---|
| [0001](0001-ffi-first-prebuilt-kernels.md) | FFI-first distribution with prebuilt native kernels | Accepted |
| [0002](0002-numpy-as-specification.md) | NumPy and SciPy are the specification; parity fixtures are generated | Accepted |
| [0003](0003-one-kernel-two-bindings.md) | One C kernel, two bindings (FFI and Zend extension) | Accepted |
| [0004](0004-native-allocator-and-budget.md) | A native allocator with a per-process budget, outside the Zend heap | Accepted |
| [0005](0005-byte-strides-and-views.md) | NumPy memory layout: byte strides, views share memory | Accepted |
| [0006](0006-process-wide-kernel-settings.md) | Kernel settings are process-wide; bindings re-apply them per request | Accepted |
| [0007](0007-solvers-in-c.md) | LP/MILP and MDP solvers in C with a dense simplex and CSR transitions | Accepted |
| [0008](0008-project-name.md) | The project is named Tessero | Accepted |
| [0009](0009-ufunc-engine-and-memmap.md) | A ufunc engine in the extension; memory-mapped arrays outside the budget | Accepted (amended by 0010) |
| [0010](0010-shared-kernels-and-vector-math.md) | Ufuncs, memory maps and `.npy` in the shared kernel; vector exp/log written in-house | Accepted |
| [0011](0011-kernel-function-registry.md) | One function registry in the kernel, interpreted by both bindings | Accepted |
| [0012](0012-pinned-reference-environment.md) | Reference fixtures are generated in a pinned numerical environment | Accepted (amends 0002) |
| [0013](0013-registry-sequences-and-in-place-routines.md) | Registry routines take sequences, variadic and in-place arguments | Accepted (amends 0011, 0004) |

To propose a decision, copy the template below into a new file with the next
number and open a pull request. Discussion happens on the pull request.

```markdown
# NNNN. Title

- Status: Proposed | Accepted | Superseded by NNNN
- Date: YYYY-MM-DD

## Context
## Decision
## Consequences
## Alternatives considered
```
