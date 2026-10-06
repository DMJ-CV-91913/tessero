# Element-wise maths, reductions and sorting

## Binary operations

| Method | Operator (ext) | NumPy |
|---|---|---|
| `add`, `sub`, `mul`, `div` | `+ - * /` | `add subtract multiply true_divide` |
| `pow` | `**` | `power` |
| `mod`, `floorDiv` | `%` | `remainder floor_divide` (Python sign rules) |
| `maximum`, `minimum` | | NaN-propagating |
| `atan2`, `hypot` | | |
| `eq ne lt le gt ge` | | comparisons → bool |
| `logicalAnd logicalOr logicalXor` | | → bool |
| `rsub`, `rdiv`, `rpow` | `2 - $a` etc. | reflected: the scalar on the left |

All of them broadcast and accept PHP scalars, PHP arrays or arrays. On the FFI
package they take an optional `out:` array that receives the result, which
avoids an allocation in hot loops:

```php
$acc = zeros(1_000_000);
foreach ($batches as $b) {
    $acc->add($b, out: $acc);
}
```

Integer overflow wraps (two's complement), exactly like NumPy. It is never
undefined behaviour: the C kernel is tested under UBSan.

## Unary functions

`neg abs square sign sqrt exp log log10 log2 expm1 log1p sin cos tan arcsin
arccos arctan sinh cosh tanh floor ceil rint reciprocal isnan isfinite isinf
invert logicalNot`, plus `round($decimals)` and `clip($min, $max)`. The FFI
package adds `real imag conj angle` for complex arrays.

Integer input to a floating-point function (`sqrt`, `exp`, …) gives float64.

## Reductions

```php
$x = arange(12)->reshape(3, 4)->astype('float64');

$x->sum();                        // 66.0
$x->sum(axis: 0);                 // column sums, shape [4]
$x->mean(axis: -1, keepdims: true);   // shape [3, 1]
$x->sum(axis: [0, 1]);            // several axes (FFI package)
$x->var(ddof: 1);                 // sample variance
$x->std(axis: 0);
$x->min(); $x->max(); $x->argmax();   // argmax: index into the flattened array
$x->any(); $x->all();
$x->cumsum(axis: 1);
$x->diff();                       // FFI package
```

Accuracy:

- **sum and mean use pairwise summation**, like NumPy. The error grows with
  log n rather than n, so a million values sum to within a few ulps.
- **mean, var and std are computed as NumPy computes them**: the mean is the
  pairwise sum divided by n, and the variance is the pairwise sum of squared
  deviations from that mean. This is stable when the mean is large compared with
  the spread (unlike the textbook E[x²]−E[x]² formula), and bit-identical to
  NumPy on contiguous data.
- **Reductions of integers** accumulate in int64 (wrapping). The mean of
  integers is float64.
- NaN propagates through `sum`, `min`, `max`, `mean`, as in NumPy.

## Sorting

```php
arr([3, 1, 2])->sort();            // [1, 2, 3]  (new array)
arr([3, 1, 2])->argsort();         // [1, 2, 0]
$m->sort(axis: 0);                 // along any axis
```

`sort` is a radix sort for every dtype (an MSD pass, then cache-sized LSD
passes, with buckets sorted in parallel when `threads > 1`). `argsort` is a
stable LSD radix sort. NaN sorts last, as in NumPy. On 1M float64 values
`sort` takes about 30 ms against NumPy's 7.5 ms (AVX-512 quicksort), and
`argsort` is on par with NumPy's stable argsort.

## Matrix products

```php
$a->matmul($b);        // or dot(); ext: $a->matmul($b)
```

Batched and broadcast like `numpy.matmul`. The FFI package calls OpenBLAS
`dgemm`/`sgemm` when OpenBLAS is available, and otherwise a blocked C kernel
that uses FMA. The extension always uses the C kernel.

## Threads

Element-wise kernels on arrays of at least 131 072 elements are split
across OpenMP threads when threads > 1 (`Tessero::setThreads()`,
`Engine::setMaxThreads()`, `tessero.threads`). Each thread handles a
contiguous chunk, so results are bit-identical to one thread. Reductions stay
serial to keep the pairwise summation order fixed. See
[Memory and threads](../operations/memory-and-threads.md).
