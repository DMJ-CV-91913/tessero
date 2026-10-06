# Sanitizer sweep of the function registry

`run.sh` builds libtessero with AddressSanitizer and UBSan. It then calls every registry function through
the FFI package with adversarial arguments, each function in its own PHP process so that one crash does not
hide the others. The arguments are empty, NaN, ±inf, huge and negative values, wrong shapes, invalid strings
and out-of-range axes.

It reports:

- sanitizer findings, attributed to the call that caused them;
- crashes;
- hangs (over `HANG` seconds);
- slow calls;
- results that differ between two identical calls (uninitialised memory).

| File | Role |
|---|---|
| `make_jobs.py` | one job per registry function (`tools/parity/registry.json`) |
| `harness.php` | the argument generators per kind (`special`, `np`, `stats`, `dist`, `gen`) |
| `driver.py` | runs the jobs, resumes after crashes and hangs, writes `results.jsonl` |
| `analyze.py` | groups the findings by root cause |

The parity program runs it at the end of every phase. A finding is fixed in the kernel and pinned in
`tests/Unit/AdversarialInputTest.php`. Where SciPy itself hangs on the same input (a quadratic algorithm, a
loop over n), the case is listed under "Known, not guarded" in `docs/project/reference-deviations.md` instead.

With `allocator_may_return_null=1`, ASan reports an oversized allocation as out of memory. Tessero turns such a
request into a `MemoryError` before allocating, from the memory budget or the machine's RAM. An ASan
out-of-memory report therefore marks an allocation that skipped that check.
