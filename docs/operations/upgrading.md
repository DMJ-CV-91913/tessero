# Upgrading

Tessero follows [Semantic Versioning](https://semver.org). Before 1.0, minor
versions (0.x → 0.y) may contain breaking changes. Each one is listed here
with the action it needs. Patch versions never break.

The full history is in `CHANGELOG.md` at the repository root.

## General procedure

1. Read this page and the changelog for every version you skip.
2. Upgrade staging first, with the same backend (extension or FFI) as
   production.
3. Run `tessero doctor` in every SAPI.
4. Compare outputs of your key calculations before and after on stored
   inputs. See [RB-4](runbooks.md#rb-4-solver-results-changed-after-an-upgrade)
   for which differences are expected.
5. Upgrade the extension and the Composer packages **together**. Their
   versions must match (`Engine::version()` and `Tessero::VERSION`).
6. FFI under FPM: restart FPM (preloaded code is fixed at start).

## 0.1 → 0.2

### New

- `ext-tessero`, a full native backend (`Tessero\Ext\NDArray`, `Engine`,
  `Operand`). It replaces the 0.1 operator-only extension `tessero_ops`.
- Linear and mixed-integer programming (`Optimize\LinearProgramming`),
  Markov decision processes (`Mdp\MarkovDecisionProcess`).
- `tessero/laravel`: provider, facade, casts, validation rule, response
  macro, Octane-safe settings.
- OpenMP threads (`Tessero::setThreads`, `tessero.threads`).
- `NDArray::toJson()`, `serialize()` support, C slice parser (`sliceNative`).

### Breaking changes and actions

| Change | Action |
|---|---|
| The `tessero_ops` extension is gone. Operators now come from `ext-tessero`. | Remove `extension=tessero_ops`, install `ext-tessero`. Code using operators needs no change. |
| `NDArray` constructor-promoted properties are no longer `readonly` (needed for `__unserialize`). | None, unless you relied on the `Error` thrown when writing them. |
| `NDArray` now extends `Tessero\Internal\OperandBase`. | None, unless you use `get_parent_class()` on arrays. |
| New error codes −6 (shape), −7 (not contiguous), −8 (did not converge) map to `ShapeError`, `TesseroException`, `ConvergenceError`. | Catch `TesseroException` if you matched messages of the old generic error. |
| LP/MILP reject NaN/±INF in `c`, `A`, `b` with `InvalidArgumentException` (0.2.0). | Clean data before solving; infinite bounds remain allowed. |
| MDP constructors reject non-finite rewards and probabilities, and malformed CSR. | As above. |
| LP/MILP workspaces count against the memory budget. | Raise the budget if large LPs now throw `MemoryError`. |
| MILP no longer prunes nodes whose relaxation hit the iteration limit or numerical trouble; the result reports status 1 (limit) instead of a possibly wrong "optimal". | Check `status`/`gap()` as documented. |

### Configuration

New keys only; nothing renamed. Laravel users can publish the config
(`vendor:publish --tag=tessero-config`) or rely on the defaults.
