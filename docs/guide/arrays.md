# Arrays

`Tessero\NDArray` (FFI) and `Tessero\Ext\NDArray` (extension) are
n-dimensional arrays of one element type, stored in native memory. The layout
is NumPy's: a pointer into a memory block, a shape, and **byte** strides.

## Creating arrays

```php
use Tessero\NDArray;
use function Tessero\{arr, zeros, ones, arange, linspace};

arr([1, 2, 3]);                           // int64
arr([1, 2.5]);                            // float64 (ints and floats promote)
arr([[true, false]]);                     // bool
arr([1, 2], 'float32');                   // explicit dtype
zeros([3, 4]);                            // float64 zeros
ones(5, 'int32');
NDArray::full([2, 2], 7);                 // int64 7s
arange(10);                               // 0..9, int64
arange(0.0, 1.0, 0.25);                   // [0, .25, .5, .75]
linspace(0, 1, 5);                        // [0, .25, .5, .75, 1]
NDArray::eye(3);                          // identity
NDArray::fromBytes($bytes, 'float64', [1000, 3]);   // raw little-endian bytes
```

Input must be rectangular: `arr([[1, 2], [3]])` throws `ShapeError`. Only
numbers and booleans are accepted. Strings throw `DTypeError`, and nothing is
silently converted to 0.

## Element types

| dtype | PHP value | Bytes | Notes |
|---|---|---:|---|
| `float64` | float | 8 | default for floating input |
| `float32` | float | 4 | half the memory; arithmetic stays in float32 |
| `int64` | int | 8 | default for integer input; overflow wraps (two's complement), as in NumPy |
| `int32` | int | 4 | |
| `uint8` | int | 1 | images, masks, byte data |
| `bool` | bool | 1 | result of comparisons; used for masks |
| `complex128` | `[re, im]` | 16 | full support on the FFI package; storage, arithmetic, `real imag conj angle abs` and FFT on the extension |

Type promotion follows NumPy 2 (NEP 50):

- Mixing two arrays promotes to the smallest type that holds both
  (`int32 + float32 → float64`, `uint8 + int32 → int32`).
- PHP scalars are **weak**: `$float32Array->mul(2.5)` stays float32 and
  `$int32Array->add(1)` stays int32.
- True division of integers gives float64. `floorDiv` and `mod` keep integers
  (with Python's sign rules for negative numbers).
- Comparisons give bool.

Convert with `astype('float32')`. `view('int64')` reinterprets the bytes without
converting them.

## Shape and memory

```php
$a = arange(24)->reshape(2, 3, 4);
$a->shape();          // [2, 3, 4]
$a->ndim();           // 3
$a->size();           // 24
$a->strides();        // [96, 32, 8]  bytes
$a->nbytes();         // 192
$a->isContiguous();   // true
```

These return **views**, which share memory and copy nothing:

- slicing ([Indexing](indexing.md))
- `transpose()` and `t()`, `swapAxes()`, `moveAxis()`
- `reshape()` of a contiguous array
- `expandDims()`, `squeeze()`, `broadcastTo()`, `view()`

Every computation returns a new C-contiguous array. `copy()` always copies.
`contiguous()` copies only when the array is not already contiguous.

A view keeps its memory block alive. The block is freed when the last array
or view that uses it is destroyed, and never earlier. This is covered by
tests; see [Memory model](../architecture/memory-model.md).

Writing through a view writes to the original:

```php
$a = zeros(6);
$v = $a['::2'];
$v->assign(1.0);
$a->toList();         // [1, 0, 1, 0, 1, 0]
```

## Getting data out

| Method | Result |
|---|---|
| `toArray()` | nested PHP arrays |
| `toList()` | flat PHP list |
| `item(1, 2)`, `get(1, 2)` | one element as a PHP scalar |
| `toBytes()` | raw little-endian bytes, C order |
| `toJson()` | JSON text written from native memory (see below) |
| `json_encode($a)` | via `JsonSerializable` |
| `serialize($a)` | dtype + shape + bytes (exact); safe for caches and queues |
| `foreach ($a as $i => $row)` | iterates the first axis (rows as views, or scalars for 1-D) |

`toJson()` keeps the zero fraction of floats (`1.0`, not `1`), so a float
array stays a float array when decoded. On the extension it is byte-identical
to `json_encode($a->toList(), JSON_PRESERVE_ZERO_FRACTION)`. The FFI package's
`toJson()` writes the same numbers (shortest round-trip form) but spells
exponents the C way: `1e-05` where `json_encode` writes `1.0e-5`. Both parse to
the same float. `json_encode($a)` goes through `jsonSerialize()` and follows
your flags. NaN and ±INF are written as `null` by `toJson()`.

## Operators (extension)

With `ext-tessero` loaded, both array classes support `+ - * / ** %` and
compound assignment (`$a += 1` rebinds `$a` to a new array). On
`Tessero\Ext\NDArray`, `==` is true when dtype, shape and every element's
bytes match. Use `eq()` for an element-wise comparison. Without the extension
use the methods (`add`, `sub`, `mul`, `div`, `pow`, `mod`).

## NumPy's array functions (`Np`)

`Tessero\Np` (FFI) and `Tessero\Ext\Np` (extension) provide NumPy's array construction and manipulation
functions under their NumPy names in camelCase, with NumPy's arguments as PHP named arguments. Both classes
run the same kernel code ([ADR 0013](../architecture/adr/0013-registry-sequences-and-in-place-routines.md)).

```php
use Tessero\Np;

$a = [[1, 2, 3], [4, 5, 6]];
Np::flip($a, axis: 1);                   // [[3, 2, 1], [6, 5, 4]]
Np::concatenate([$a, [[7, 8, 9]]]);      // arrays of different shapes: a PHP list
Np::split(range(0, 8), 3);               // a PHP list of three arrays
[$xx, $yy] = Np::meshgrid([1, 2, 3], [4, 5], indexing: 'ij');   // NumPy's *xi: positional arguments
Np::pad([1, 2, 3], [2, 1], mode: 'reflect');                    // [3, 2, 1, 2, 3, 2]
Np::interp([0.5, 2.5], [0, 1, 2], [0, 10, 20]);                 // [5.0, 20.0]
```

The functions include:

- the flip family, `rot90`, `roll`, `tile`, `repeat` and `resize`;
- joining (`concatenate`, `stack`, `vstack`, `hstack`, `dstack`, `column_stack`), `unstack`, and splitting
  (`split`, `array_split`, `hsplit`, `vsplit`, `dsplit`);
- axis moves (`expand_dims`, `squeeze`, `swapaxes`, `moveaxis`, `rollaxis`, `matrix_transpose`,
  `atleast_1d/2d/3d`, `broadcast_to`, `broadcast_arrays`, `broadcast_shapes`);
- `append`, `insert`, `delete` and `trim_zeros`;
- diagonals and triangles (`diag`, `diagflat`, `diagonal`, `trace`, `tri`, `tril`, `triu`, and the
  `tril_indices` / `triu_indices` / `diag_indices` families), `identity`, and the `*_like` constructors;
- gathering (`take`, `take_along_axis`, `compress`, `extract`, `nonzero`, `argwhere`, `flatnonzero`,
  `select`, `choose`) and index conversion (`unravel_index`, `ravel_multi_index`, `indices`, `ix_`,
  `meshgrid`);
- products (`outer`, `inner`, `vdot`, `kron`, `cross`, `tensordot`), `diff`, `ediff1d`, `gradient`,
  `trapezoid`, `unwrap`, `interp`, `logspace`, `geomspace`, `vander` and the polynomial helpers (`polyval`,
  `polyadd`, `polysub`, `polymul`, `polyder`, `polyint`, `polydiv`);
- window functions, `sinc`, `i0`, `pad`, `nan_to_num`, and the complex-type predicates.

The [coverage page](../project/numpy-scipy-coverage.md) lists what exists, verified against NumPy on both
backends.

Two differences from NumPy are deliberate:

- **Copies, not views.** Where NumPy returns a view (`flip`, `swapaxes`, `diagonal`, `broadcast_to`, ...),
  `Np` returns a new array with the same values. Use the NDArray methods (`slice`, `transpose`, ...) when you
  need a view.
- **In-place writers.** `put`, `place`, `putmask`, `copyto`, `fill_diagonal` and `put_along_axis` modify
  their first argument, as in NumPy, and return `null`. That argument must be an NDArray (a PHP array cannot
  be modified through a call) and must not be a read-only memory map.
