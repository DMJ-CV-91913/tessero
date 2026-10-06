# Operations manual

This manual is for the people who install, run and support Tessero in
production: platform engineers, SREs, and developers who own a service end
to end. It assumes you know how PHP runs in your environment (FPM, Octane,
queue workers, CLI).

| Page | Read it when |
|---|---|
| [Deployment](deployment.md) | Putting Tessero on a server for the first time: CLI, FPM with preload, Octane, queue workers |
| [Native extension](native-extension.md) | Building, packaging and rolling out `ext-tessero` |
| [Docker](docker.md) | Building images (Debian and Alpine), multi-stage builds |
| [Configuration reference](configuration-reference.md) | Every INI setting, environment variable and config key in one place |
| [Memory and threads](memory-and-threads.md) | Sizing workers, setting the native memory budget, choosing thread counts |
| [Monitoring](monitoring.md) | What to measure, health checks, logs |
| [Performance tuning](performance-tuning.md) | A workload is slower than expected |
| [Troubleshooting](troubleshooting.md) | An error message or symptom, and what to do about it |
| [Runbooks](runbooks.md) | Step-by-step responses to incidents |
| [Security](security.md) | Hardening, untrusted input, reporting vulnerabilities |
| [Upgrading](upgrading.md) | Moving between versions |
| [Soak testing](soak-testing.md) | Proving that long-lived workers do not accumulate memory, mappings or descriptors |

## The operating model in one page

- **Two memory pools.** PHP's `memory_limit` covers PHP values. Array data
  lives in native memory, allocated by `libtessero` and **not** counted by
  `memory_limit`. Each process has its own counter and optional budget
  (`memory_budget`). Set the budget; it turns a runaway allocation into an
  exception instead of an OOM kill.
- **Memory is freed deterministically.** A native block is freed when the
  last PHP object that references it is destroyed. There is no separate
  garbage collector and no finaliser queue.
- **Threads are opt-in.** Kernels run on one thread unless you raise
  `threads`. Under FPM, parallelism comes from having many workers. Raise
  threads for CLI jobs and dedicated worker pools.
- **Settings are per process.** Threads, budget and tolerance are process-wide.
  The Laravel provider re-applies them per request and per job. Outside
  Laravel, set them at bootstrap.
- **Two backends, identical numbers.** The FFI package needs `ffi.enable` (and
  preload under FPM). The extension needs `extension=tessero`. Either can be
  rolled back to the other without changing results.
- **No network, no files, no background threads of its own.** Tessero does
  not open sockets, does not read files except the ones you pass to
  `Npy::load`, and does not keep threads alive between calls, apart from the
  OpenMP runtime's pool when threads > 1.
