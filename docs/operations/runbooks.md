# Runbooks

Step-by-step responses to the incidents most likely to involve Tessero. Each
starts with how to confirm the cause, because the symptom often has another
cause.

---

## RB-1: Workers killed by the OOM killer / container restarts

**Confirm:** `dmesg | grep -i oom` or the orchestrator's events show kills;
PHP logs show no `Allowed memory size exhausted`. Native memory is invisible
to `memory_limit`, so this is the likely source.

1. Check the configured budget in the affected SAPI:
   `php -r 'var_dump(Tessero\Ext\Engine::memoryBudget());'` (or `Tessero::memoryBudget()`).
   If it is 0 (unlimited), that is the problem.
2. Set a budget ([sizing](memory-and-threads.md#sizing)) and redeploy. Over-budget
   requests now fail with a catchable exception instead of killing the worker.
3. Find what allocates. Log `memoryInUse()`/`peakMemory()` per request or job
   ([Monitoring](monitoring.md)). Check request sizes against validation caps.
4. If `memoryInUse()` does not return to baseline between jobs, look for
   arrays kept in static properties, singletons, or long-lived collections.

**Resolved when:** kills stop, and `MemoryError` rates are known and acceptable.

---

## RB-2: Spike in `MemoryError` / `MemoryException`

**Confirm:** error tracker grouping by exception class.

1. Read the message: it gives the requested size, the amount in use and the budget.
2. If the **requested size** is large, an input grew. Find the endpoint or job,
   and check whether the validation cap (`maxElements`) should reject it.
3. If **in use** is already near the budget before the request, something
   accumulates. Follow RB-1 step 4.
4. If the workload legitimately grew, raise the budget and, if needed, the
   container memory, keeping `max_children × (memory_limit + budget)` within limits.

---

## RB-3: FPM returns 500 after a deploy: "FFI API is restricted"

**Confirm:** the FFI package is in use under FPM with `ffi.enable=preload`,
and the error started with a deploy.

1. Preloaded code is fixed at FPM start. If the deploy changed Tessero, FPM
   must be **restarted**, not reloaded: `systemctl restart php8.4-fpm`.
2. Check `opcache.preload` points at the *current* release path. With
   symlinked releases (`/var/www/current`), preload resolves the symlink at
   start-up. Restart after switching.
3. Check the preload script ran: `php-fpm -i | grep preload` and the FPM
   error log for preload errors (a missing `TESSERO_LIB`, unreadable files).
4. Long-term: install the extension, which has no preload step.

---

## RB-4: Solver results changed after an upgrade

**Confirm:** the same input gives a different policy, plan or number.

1. Check the changelog for solver changes ([Upgrading](upgrading.md)).
2. **LP/MILP with multiple optimal solutions:** a different optimal vertex is
   a correct answer. Compare `fun`, which must match to tolerance. Add a
   tie-breaking term to the objective if you need a unique plan.
3. **MDP ties:** equal-valued actions resolve to the lowest action index. A
   change in the last bits of values can flip a tie. Compare values, not only
   policies.
4. **Floating-point differences in the last digits** after a CPU or OpenBLAS
   change are expected ([Troubleshooting](troubleshooting.md#results-differ-slightly-between-machines)).
5. Anything else is a bug: capture the input (`serialize()` the arrays, or
   `Npy::saveZ`) and open an issue.

---

## RB-5: Requests slow down

1. `tessero doctor` in the production SAPI: is OpenBLAS found? Is the SIMD
   level what you expect (a baseline on a modern x86 host suggests a VM with
   masked CPU flags)?
2. Is `threads > 1` under FPM? Set it to 1 ([Memory and threads](memory-and-threads.md)).
3. Did a code change add per-element PHP loops or `toArray()` round trips?
   See [Performance tuning](performance-tuning.md).
4. Solver slowness: log `iterations`/`nodes`. Growth there means the models
   grew, not that Tessero slowed down.

---

## RB-6: Worker hangs (100 % CPU or 0 % CPU, no progress)

1. `py-spy`/`gdb -p <pid> -batch -ex 'thread apply all bt'` shows where.
2. Frames in `gomp_*` after a `fork`: RB-7.
3. Frames in `tsr_milp`/`tsr_linprog`: a large or degenerate model. Set
   `nodeLimit`/`maxiter`, and run such solves on a queue with a timeout.
4. Frames in `tsr_mdp_*` with γ close to 1: value iteration needs
   log(ε)/log(γ) sweeps. Use policy iteration.

---

## RB-7: Hang after `pcntl_fork` (Horizon, custom supervisors)

1. The parent ran OpenMP kernels (threads > 1) before forking. libgomp does
   not support this.
2. Set `tessero.threads = 1` / `TESSERO_THREADS=1` for the supervisor process,
   or make sure numerical work happens only in children.
3. Or deploy an extension build without OpenMP.

---

## RB-8: Suspected wrong numbers

1. Reproduce with the JIT off (`-d opcache.jit=disable`). If the result
   changes, it is a JIT problem: keep the JIT off for now and report it.
2. Reproduce on the other backend (FFI vs extension). A difference is a bug in
   Tessero.
3. Reproduce in NumPy/SciPy with the same input. Export with `Npy::save`, and use
   `numpy.random.default_rng(seed)` for random data, since the streams are identical.
4. Report with the input files, versions (`tessero info --json`) and the
   expected result.
