# Memory-mapped arrays

`NDArray::memmap()` turns a file into an array without reading it. The data
pointer is the address the operating system returns from `mmap()`
(`MapViewOfFile()` on Windows). Pages are loaded when they are touched and
dropped under memory pressure, so a PHP process can work on files far larger
than its RAM, and several processes can share one file's pages.

```php
use Tessero\Ext\NDArray as X;
use Tessero\Ext\Math;

// create a 20-year hourly × 1 200-node price cube on disk (1.7 GB)
$cube = X::memmap('/data/lmp.f64', 'w+', [175_320, 1_200], 'float64');
$cube['0:24'] = $firstDay;             // writes go to the file's pages
$cube->flush();                        // schedule write-back (MS_ASYNC), returns at once
$cube->flush(sync: true);              // wait until it is on disk (MS_SYNC + fsync)

// later, in another process: open read-only; only touched pages are read
$lmp = X::memmap('/data/lmp.f64', 'r', [175_320, 1_200]);
$node = $lmp[':, 417'];                // a view into the file: no copy
$node->mean();                         // reads that column's pages
$lmp['-8760:']->max(0);                // last year's maximum per node

// shape inferred from the file size (1-D), with a header offset
$raw = X::memmap('/data/meter.i32', 'r', null, 'int32', offset: 128);
```

Both backends have the same calls and error behaviour (`Tessero\NDArray::memmap()`
in the FFI package): the mapping code is libtessero's `tsr_mmap_*`, shared by
the two bindings. The Laravel facade has the same call:
`Tessero::memmap($file, 'r', [..])`.

## .npy files

NumPy's `.npy` format carries dtype and shape in a header, so a file written
by Python can be opened, or mapped, without passing either:

```php
$a = X::load('/data/prices.npy');              // read (numpy.load)
$m = X::load('/data/prices.npy', 'r');         // memory-map instead (mmap_mode='r'; also 'r+', 'c')
$a->save('/data/out.npy');                     // write (numpy.save), byte-identical to NumPy's file

// create a memory-mapped .npy for results (numpy.lib.format.open_memmap)
$out = X::openMemmap('/data/forecast.npy', 'w+', [8760, 1200], 'float32');
Math::exp($logp, $out);                        // the kernel writes into the file
$out->flush(sync: true);
```

Supported: format versions 1.0, 2.0 and 3.0; little-endian `float64`,
`float32`, `int64`, `int32`, `uint8`, `bool`, `complex128`; C or Fortran
order (Fortran-order files map with Fortran strides, and `load()` returns a
C-order copy). Big-endian and structured dtypes are refused with a dtype
error. The header parser is libtessero's `tsr_npy_header`, which is fuzzed in
CI ([testing](../architecture/testing-and-quality.md)).

## Modes

Modes follow `numpy.memmap`:

| Mode | File | Array | Notes |
|---|---|---|---|
| `'r'` | must exist and be large enough | read-only | writes, `assign`, and use as `out` throw; the pages are mapped read-only as a second line of defence |
| `'r+'` (default) | must exist; extended when shorter than `offset + shape` | read/write | changes reach the file |
| `'w+'` | created, or truncated, then sized to `offset + shape` (zero-filled) | read/write | `shape` is required |
| `'c'` | must exist | read/write in this process only | copy-on-write: the file never changes; `flush()` does nothing |

Parameters: `memmap(string $filename, string $mode = 'r+', ?array $shape =
null, string $dtype = 'float64', int $offset = 0)`. `offset` is a byte offset
into the file. It must be a multiple of the item size, and it need not be
page-aligned: the mapping starts at the page boundary below it. Without
`shape`, the array is 1-D with as many items as fit after `offset` (the byte
count must be a whole number of items).

## What you can do with a memmap

A memmap *is* an `NDArray`, so everything works on it:

- **Views stay in the file.** Slices, transposes, reshapes and row iteration
  point into the mapping, and writes through them change the file.
- **Computations produce ordinary arrays.** `$lmp->mean(0)`, `$lmp * 2` and
  `Math::sqrt($lmp)` return heap arrays, counted against the memory budget.
  Size results accordingly: `$lmp * 2` on a 1.7 GB map allocates 1.7 GB.
- **Write results into another file** with `out`:
  `Math::log($prices, X::memmap('/data/logp.f64', 'w+', $prices->shape()))`.
  The kernel writes straight into the mapping.
- `clone`, `serialize()` and `toBytes()` make in-memory copies.
- `isMemmap()`, `isReadonly()` and `filename()` describe the array. Views
  report their root's values.

## Lifecycle

- The root array owns the mapping. Every view holds a reference to the root,
  so the mapping stays valid while any view exists, even after the variable
  holding the root is gone.
- When the last reference goes, the library unmaps the file and closes its
  descriptor (`munmap` + `close`, or `UnmapViewOfFile` + `CloseHandle`).
  Nothing leaks, and a view can never outlive its pages.
- Unmapping does not lose data: dirty pages of a shared mapping stay in the
  OS page cache and reach the disk on the OS's schedule. Call
  `flush(sync: true)` when you need durability at a known point, for example
  before telling another system the file is complete.

## Memory accounting

Mapped bytes are **not** counted against `tessero.memory_budget`. They are
file-backed, the OS can evict clean pages at any time, and counting a 50 GB
mapping against a 256 MB budget would make memmaps useless. The process total
is `Engine::info()['memory_mapped']` (extension) or
`Tessero::info()['memory_mapped']` (FFI package). Resident pages do appear in the process
RSS, and the OS reclaims them under pressure.

## Things to know

- **Do not shrink a mapped file.** If another process truncates the file while
  it is mapped, touching the missing pages raises SIGBUS, which kills the PHP
  worker. This is how memory mapping works on every OS, not something Tessero
  can catch. Coordinate writers, or copy files before mapping them.
- **Two memmaps of the same file are separate arrays.** Writes through one are
  visible through the other (shared pages), but overlap between them is not
  detected when one is used as `out` for a computation on the other.
- **Local files only.** Stream wrappers (`php://`, `phar://`, `s3://`) have no
  descriptor to map and are refused. Paths go through `open_basedir`.
- **Network file systems** (NFS, SMB) can be mapped, but their coherence
  guarantees are weaker. Prefer local disks for files that several hosts write.
- **Byte order is native** (little-endian on x86-64 and ARM64), the same as
  `toBytes()`. For NumPy `.npy` files, skip the header with `offset` (the
  header length is in the file), or use the FFI package's `Npy::load`.
