# 0004. A native allocator with a per-process budget

- Status: Accepted
- Date: 2026-09-26

## Context

`FFI::new` zero-fills memory and has no alignment control. Memory allocated by
C libraries is not counted by PHP's `memory_limit`, so an FPM worker can grow
until the OOM killer ends it. The design brief asked for a userland tracking
budget in the PHP class. The extension brief suggested `emalloc`/`pemalloc`.

## Decision

All array memory, and solver workspaces that grow with the problem, come
from `tsr_alloc`: 64-byte aligned, not zeroed, huge-page-hinted for large
blocks, and counted by an atomic counter. An optional budget is checked
**before** memory is requested. Exceeding it returns `TSR_ENOMEM`, which the
bindings raise as a catchable exception with sizes in the message.

## Consequences

- One mechanism covers both bindings and memory allocated inside C (solver
  tableaux), which a PHP-side counter would miss.
- Arrays are independent of PHP's per-request heap, so they can live across
  requests in Octane workers. That is correct, but it means leaks in
  application code (arrays in statics) persist, so monitoring guidance covers it.
- The counter is per process, not per PHP thread under ZTS
  ([ADR-0006](0006-process-wide-kernel-settings.md)).
- `emalloc` was rejected: it gives no alignment control, allocations would be
  freed by the Zend heap at request end, and the budget would not see them.
