# 0006. Kernel settings are process-wide; bindings re-apply them per request

- Status: Accepted
- Date: 2026-09-26

## Context

The thread count and the memory budget live in the C kernel, which has one
instance per process. PHP runtimes differ. FPM runs one request per process
at a time. Octane and FrankenPHP workers serve many requests in one
long-lived process. ZTS builds run several PHP threads in one process. The
design brief asked for thread-local configuration to avoid cross-request
leakage.

## Decision

- The kernel stores its settings in atomics: process-wide and race-free.
- The extension keeps its INI-derived settings in module globals (per thread
  under ZTS) and **re-applies them to the kernel in RINIT**, at the start of
  every request.
- The Laravel provider re-applies `config/tessero.php` at boot and on every
  Octane request, task and queued job.
- OpenMP's global thread setting is never touched. Each parallel region passes
  `num_threads(n)`.

## Consequences

- A setting changed during a request cannot leak into the next request on the
  same worker, which covers the Octane case the brief was concerned about.
- Under ZTS, concurrent PHP threads share the kernel's thread count and budget.
  This is documented, and the budget should be sized for the whole process.
- True per-thread kernel settings would need the settings passed into every
  kernel call (an ABI change). It is not needed by any supported deployment today.
