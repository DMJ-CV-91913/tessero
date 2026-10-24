# 0017. OpenMP runtimes: ship with GCC libgomp, analyse races under LLVM libomp + Archer

- Status: Accepted
- Date: 2026-10-24
- Relates to: 0014 (ABI, `-z nodelete`), 0015 (memory ownership)

## Context

The element-wise and ufunc kernels run in parallel with `#pragma omp parallel for`. The production Linux binaries
(glibc x86-64 and aarch64, musl) are built with GCC and so run on GCC's OpenMP runtime, libgomp; macOS builds use
clang with LLVM's libomp, and Windows builds MinGW. Users who build the extension with `phpize` get their system
compiler, which on Linux is usually GCC.

ThreadSanitizer finds data races in our code only if it can see every synchronisation point. libgomp is not built
with TSan instrumentation, so TSan cannot see its barriers: a gcc + libgomp TSan run reported the same 14 races every
time, all of the form "workers read a variable the main thread wrote before the parallel region", which the region's
barrier orders. That lane could never pass and was removed. LLVM's libomp with the Archer tool reports its barriers
and task synchronisation to TSan through OMPT, and that run is clean.

So races are analysed under one runtime while another ships.

## Decision

Keep shipping GCC + libgomp on Linux, and keep race analysis on clang + LLVM libomp + Archer. The split is
deliberate:

- **Race freedom is a property of the source.** Whether two threads touch the same memory without an ordering
  between them follows from our code under the OpenMP rules, whichever runtime implements the barriers. Archer lets
  TSan see those barriers, so its verdict on our loops carries over to libgomp builds. Defects inside libgomp itself
  are outside our code and invisible to TSan in any configuration.
- **Runtime behaviour is tested on the runtime we ship.** Every Linux CI lane builds with GCC + libgomp and runs the
  threaded C tests, the NumPy/SciPy parity suites, Valgrind memcheck and the reproducibility job, which requires
  bit-identical results across thread counts on the libgomp build.
- **Fork safety is tested on libgomp.** The extension starts no parallel region at module startup, so PHP-FPM forks
  its workers before any worker pool exists. A `pthread_atfork` handler resets the thread count, so a child of a
  threaded parent runs serially until the caller sets it again. The C test runs a real parallel region in the parent,
  forks, and requires the child to run the same kernel serially, with the parent's bits, under a deadlock alarm.
- **Unload safety:** the library and the extension link with `-z nodelete`, so libgomp workers never execute
  unmapped code after `dlclose` (ADR 0014). `tsr_feature("nodelete")` reads the flag from the loaded object, and CI
  asserts it on both bindings.
- **No environment overrides from the library.** Tessero does not `setenv` OpenMP variables: that is not
  thread-safe in a running ZTS process, and it would override the operator's configuration. Thread counts are set
  through `tessero.threads` / `Tessero::setThreads` and `tessero.blas_threads`.

## Consequences

- One CI configuration (Archer) carries race analysis, and it is blocking.
- A libgomp-specific behaviour (thread-pool reuse, nested regions) would surface in the GCC-built threaded tests or
  in the reproducibility job, not in TSan. If one ever appears, the response is a targeted libgomp stress lane;
  switching the production toolchain is the last resort.

## Alternatives considered

- **Build production with clang + libomp.** It would unify test and ship, but self-built extensions still use the
  system compiler, libomp is not installed by default on most distributions, and every shipped binary would change
  to close a gap that the source-level argument above already covers.
- **Keep the gcc + libgomp TSan lane as informational.** It could never pass, and a permanently red check hides
  real failures.
- **Suppress the libgomp reports.** A suppression broad enough to silence them would also hide real races in the
  same loops.
