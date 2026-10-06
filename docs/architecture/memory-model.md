# Memory model

## The allocator

`tsr_alloc(bytes)` is the only way array memory is obtained.

- **64-byte alignment**, one cache line and one AVX-512 vector, so SIMD loads
  on contiguous data never split lines.
- **No zeroing.** `FFI::new` and `calloc` zero memory. Most arrays are
  immediately overwritten by a kernel, so zeroing would be a wasted pass.
  `tsr_calloc` zeroes only for `zeros()`.
- **Huge pages for big blocks.** Blocks of 32 MiB or more are 2 MiB aligned and
  marked `MADV_HUGEPAGE` (Linux), which cuts TLB misses on large arrays.
  Smaller blocks skip this, because the extra alignment and madvise cost more
  than they save on short-lived temporaries.
- **Accounting.** An atomic counter tracks bytes in use and the peak. If a
  budget is set, an allocation that would exceed it fails with `TSR_ENOMEM`
  **before** memory is requested, and the counter is unchanged.
- `tsr_free(p, bytes)` takes the size back so accounting is exact without a
  header in front of every block.

The counter and budget are per library instance, and so per process. The FFI
library and the extension each have their own instance.

Solver workspaces that grow with the problem (LP tableau, the `m × n` constraint
copy, dense policy-evaluation matrices, sort buffers) also use `tsr_alloc`
and count against the budget. Small O(n) bookkeeping uses `malloc`.

## Ownership in the FFI package

```
NDArray (view)  ─┐
NDArray (view)  ─┼──►  Buffer  ──►  native block  (tsr_alloc)
NDArray (owner) ─┘       │
                         └─ __destruct → tsr_free (exactly once)
```

- `Native\Buffer` owns the block: pointer, size and dtype. It is the only
  object that frees memory, in `__destruct`. It cannot be cloned.
- Every `NDArray`, including views, holds a reference to its `Buffer`, plus
  its own offset, shape and strides. A view therefore keeps the block alive
  after its parent is gone, and the block is freed when the last array
  referring to it is destroyed.
- `copy()` allocates a new `Buffer`. Views never do.
- Pointers passed to C are recreated from the buffer's integer address and a
  byte offset for each call (`Library::ptr()`), never by `CData` arithmetic
  ([FFI binding](ffi-binding.md)).

## Ownership in the extension

```c
typedef struct {
    tsr_array    a;            /* data, offset, ndim, dtype, shape, strides */
    void        *block;        /* non-NULL only for the owner */
    int64_t      block_bytes;
    zend_object *base;         /* views: the owning object (refcounted) */
    zend_object  std;          /* must be last */
} tsr_obj;
```

- An owner has `block` set and `base == NULL`.
- A view has `block == NULL` and `base` pointing to the **root owner**, not
  its immediate parent, with a reference held (`GC_ADDREF`). Chains of views
  therefore never form long reference chains.
- `free_obj` frees `block` if set, then releases `base`. The owner object, and
  so the block, lives until the last view is destroyed.
- `get_gc` reports `base` to PHP's cycle collector, so a view stored inside an
  object graph that also references the owner is still collected.
- `clone` makes a compact copy (a new owner), never a shared view, so
  `clone $view` never aliases memory.

## Memory-mapped arrays

```
NDArray (view) ─┐
NDArray (view) ─┼──► NDArray (memmap root) ──► tsr_map ──► mmap'd file pages
                                │                     (addr, length, fd, path, mode)
                                └─ free_obj → munmap + close (after the last view)
```

- `NDArray::memmap()` points `a.data` at the address returned by `mmap()`
  (`MapViewOfFile()` on Windows), offset within the first page when the file
  offset is not page-aligned. Nothing is copied or allocated for the data.
- The root owns the mapping. Views reference the root exactly as heap views do,
  so the unmap happens once, after the last view.
- Mappings are **not** counted against the memory budget (they are
  file-backed and evictable). The process total is reported separately
  (`Engine::info()['memory_mapped']`), and the tests check it returns to 0.
- Mode `'r'` maps pages `PROT_READ` and sets `TSR_F_READONLY`. Writes are
  refused before they reach the kernel, and the page protection catches
  anything that would slip past.
- The FFI package follows the same design. A `Buffer` created by
  `Buffer::mapped()` holds the `tsr_mmap` handle, and its `__destruct` calls
  `tsr_mmap_close()` once, then clears the handle so a late `flush()` is a
  no-op. Views share the `Buffer`, so the unmap happens after the last one. A
  read-only mapping freezes the `Buffer`, and every write path
  (`assign`, `put`, `setWhere`, `offsetSet`, `out:` arguments, `shuffle`)
  checks it.

## Safeguards

These are the rules every code path follows, with the test that pins each one.

| Rule | How it is enforced | Test |
|---|---|---|
| Every allocation has exactly one owner that frees it | `Buffer::__destruct` / `free_obj`; `released` flag; `Buffer::__clone` is private | `MemorySafetyTest::testEveryAllocationIsReturned`, `001_lifecycle.phpt`, `014-memory-safety.phpt`, soak |
| Released memory is never handed to C | `NDArray::ptr()` and every `Buffer` accessor throw after release; LAPACK pointers go through the same check | `MemorySafetyTest::testReleasedBuffersAreNeverUsed` |
| Pointers never outlive their array | every C call receives `$var->ptr()` of a named variable, never a temporary expression (audited: no `->copy()->ptr()`-style chains) | code review; ASan runs |
| A shape can never describe more memory than was allocated | element counts are overflow-checked (`checkedSize`, `tsr_shape_size`, `checked_count`) before allocating, in `fromBytes`, in `unserialize`, and in reshape and broadcasting | `MemorySafetyTest::testShapesThatOverflowAreRefused`, `testUnserializeValidatesItsInput`, `test_size_overflow` (C), `014-memory-safety.phpt` |
| Read-only memory is never written from C | writability checked before every write path | `MemorySafetyTest::testReadOnlyMemoryIsNeverWrittenFromC`, `012_memmap.phpt` |
| Untrusted text is parsed only by fuzzed code | slice strings, `.npy` headers, memmap arguments and the JSON writer are libFuzzer targets | `csrc/fuzz/` |
| Runaway allocation fails as an exception, not an OOM kill | the budget is checked before memory is requested | `testBudgetTurnsRunawayAllocationIntoAnException` |
| Binary and header always match | `tsr_version()` must equal `Tessero::VERSION`; a missing symbol fails `FFI::cdef` loudly | loading |
| Preload cannot be redirected to another library | the temporary header is created exclusively (`tempnam`, 0600) and deleted after `FFI::load` | `tests/e2e/preload.sh` |
| Long-lived workers do not accumulate | native bytes and mapped bytes back at baseline after every iteration; heap, RSS and descriptor trends | `tools/soak.php`, nightly |

## Invariants the tests check

- The byte counter returns to its starting value after creating and
  destroying arrays, views of views, masks, clones and serialised copies
  (`tests/Unit/NDArrayTest.php`, `ext/tests/001_lifecycle.phpt`).
- A view outlives its parent and still reads correct data.
- Writes through a view are visible in the parent.
- Failed allocations (over budget) leave the counter unchanged.
- A memmap's views keep the mapping alive after the root variable is unset,
  and the mapped-bytes counter returns to 0 when the last one goes
  (`ext/tests/012_memmap.phpt`).
- The C suite runs under AddressSanitizer: no leaks, no out-of-bounds
  access, no use after free.
- The extension's phpt and parity suites run under ASan + UBSan with
  `USE_ZEND_ALLOC=0`.
