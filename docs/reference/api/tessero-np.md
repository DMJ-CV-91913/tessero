# Np

`Tessero\Np`

NumPy functions beyond the NDArray methods: statistics, NaN-aware reductions, quantiles, histograms, set operations, searching (numpy.*).

Names are the SciPy/NumPy names in camelCase (log_ndtr -> logNdtr); apply() also takes the original names.
The native extension provides the same class as Tessero\Ext\Np.

## Constants

| Name | Value |
|---|---|
| `FUNCTIONS` | `{"append":"append","argpartition":"argpartition","argwhere":"argwhere","array_equal":"arrayEqual","array_equiv":"arrayEquiv","array_split":"arraySplit","atleast_1d":"atleast1d","atleast_2d":"atleast2d","atleast_3d":"atleast3d","average":"average","bartlett":"bartlett","bincount":"bincount","blackman":"blackman","broadcast_arrays":"broadcastArrays","broadcast_shapes":"broadcastShapes","broadcast_to":"broadcastTo","choose":"choose","column_stack":"columnStack","compress":"compress","concat":"concat","concatenate":"concatenate","convolve":"convolve","copyto":"copyto","corrcoef":"corrcoef","correlate":"correlate","count_nonzero":"countNonzero","cov":"cov","cross":"cross","cumulative_prod":"cumulativeProd","cumulative_sum":"cumulativeSum","delete":"delete","diag":"diag","diag_indices":"diagIndices","diag_indices_from":"diagIndicesFrom","diagflat":"diagflat","diagonal":"diagonal","diff":"diff","digitize":"digitize","dsplit":"dsplit","dstack":"dstack","ediff1d":"ediff1d","empty_like":"emptyLike","expand_dims":"expandDims","extract":"extract","fill_diagonal":"fillDiagonal","flatnonzero":"flatnonzero","flip":"flip","fliplr":"fliplr","flipud":"flipud","full_like":"fullLike","geomspace":"geomspace","gradient":"gradient","hamming":"hamming","hanning":"hanning","histogram":"histogram","histogram_bin_edges":"histogramBinEdges","hsplit":"hsplit","hstack":"hstack","i0":"i0","identity":"identity","indices":"indices","inner":"inner","insert":"insert","interp":"interp","intersect1d":"intersect1d","iscomplex":"iscomplex","iscomplexobj":"iscomplexobj","isin":"isin","isneginf":"isneginf","isposinf":"isposinf","isreal":"isreal","isrealobj":"isrealobj","ix_":"ix_","kaiser":"kaiser","kron":"kron","lexsort":"lexsort","logspace":"logspace","matrix_transpose":"matrixTranspose","median":"median","meshgrid":"meshgrid","moveaxis":"moveaxis","nan_to_num":"nanToNum","nanargmax":"nanargmax","nanargmin":"nanargmin","nancumprod":"nancumprod","nancumsum":"nancumsum","nanmax":"nanmax","nanmean":"nanmean","nanmedian":"nanmedian","nanmin":"nanmin","nanpercentile":"nanpercentile","nanprod":"nanprod","nanquantile":"nanquantile","nanstd":"nanstd","nansum":"nansum","nanvar":"nanvar","ndim":"ndim","nonzero":"nonzero","ones_like":"onesLike","outer":"outer","pad":"pad","partition":"partition","percentile":"percentile","place":"place","polyadd":"polyadd","polyder":"polyder","polydiv":"polydiv","polyint":"polyint","polymul":"polymul","polysub":"polysub","polyval":"polyval","ptp":"ptp","put":"put","put_along_axis":"putAlongAxis","putmask":"putmask","quantile":"quantile","ravel_multi_index":"ravelMultiIndex","real_if_close":"realIfClose","repeat":"repeat","resize":"resize","roll":"roll","rollaxis":"rollaxis","rot90":"rot90","searchsorted":"searchsorted","select":"select","setdiff1d":"setdiff1d","setxor1d":"setxor1d","shape":"shape","sinc":"sinc","size":"size","split":"split","squeeze":"squeeze","stack":"stack","swapaxes":"swapaxes","take":"take","take_along_axis":"takeAlongAxis","tensordot":"tensordot","tile":"tile","trace":"trace","trapezoid":"trapezoid","tri":"tri","tril":"tril","tril_indices":"trilIndices","tril_indices_from":"trilIndicesFrom","trim_zeros":"trimZeros","triu":"triu","triu_indices":"triuIndices","triu_indices_from":"triuIndicesFrom","union1d":"union1d","unique":"unique","unique_all":"uniqueAll","unique_counts":"uniqueCounts","unique_inverse":"uniqueInverse","unique_values":"uniqueValues","unravel_index":"unravelIndex","unstack":"unstack","unwrap":"unwrap","vander":"vander","vdot":"vdot","vsplit":"vsplit","vstack":"vstack","zeros_like":"zerosLike"}` |

## Methods

### append

```php
static append(mixed $arr, mixed $values, mixed $axis = null): mixed
```

Append values to the end of an array (numpy.append).

numpy.append

### apply

```php
static apply(string $name, mixed ...$args): mixed
```

Call a function by its SciPy/NumPy name: Np::apply('log_ndtr', $x).

### argpartition

```php
static argpartition(mixed $a, mixed $kth, ?int $axis = -1): Tessero\NDArray|int|float
```

Indices that partition the array (numpy.argpartition).

numpy.argpartition

### argwhere

```php
static argwhere(mixed $a): mixed
```

Indices of the non-zero elements, one row per element (numpy.argwhere).

numpy.argwhere

### arrayEqual

```php
static arrayEqual(mixed $a1, mixed $a2, mixed $equalNan = false): mixed
```

True when two arrays have the same shape and elements (numpy.array_equal).

numpy.array_equal

### arrayEquiv

```php
static arrayEquiv(mixed $a1, mixed $a2): mixed
```

True when two arrays are shape-consistent and equal (numpy.array_equiv).

numpy.array_equiv

### arraySplit

```php
static arraySplit(mixed $ary, mixed $indicesOrSections, mixed $axis = 0): mixed
```

Split into sections of nearly equal size (numpy.array_split).

numpy.array_split

### atleast1d

```php
static atleast1d(mixed ...$arys): mixed
```

Arrays with at least one dimension; several arguments give a list (numpy.atleast_1d).

numpy.atleast_1d: the positional arguments are numpy's *arys.

### atleast2d

```php
static atleast2d(mixed ...$arys): mixed
```

Arrays with at least two dimensions; several arguments give a list (numpy.atleast_2d).

numpy.atleast_2d: the positional arguments are numpy's *arys.

### atleast3d

```php
static atleast3d(mixed ...$arys): mixed
```

Arrays with at least three dimensions; several arguments give a list (numpy.atleast_3d).

numpy.atleast_3d: the positional arguments are numpy's *arys.

### average

```php
static average(mixed $a, mixed $weights = null, array|int|null $axis = null, bool $keepdims = false): Tessero\NDArray|int|float
```

Weighted average along the given axes (numpy.average).

numpy.average

### bartlett

```php
static bartlett(mixed $M): mixed
```

Bartlett (triangular) window (numpy.bartlett).

numpy.bartlett

### bincount

```php
static bincount(mixed $x, mixed $weights = null, mixed $minlength = 0): mixed
```

Count (or sum weights of) each non-negative integer (numpy.bincount).

numpy.bincount

### blackman

```php
static blackman(mixed $M): mixed
```

Blackman window (numpy.blackman).

numpy.blackman

### broadcastArrays

```php
static broadcastArrays(mixed ...$args): mixed
```

Broadcast arrays against each other (numpy.broadcast_arrays).

numpy.broadcast_arrays: the positional arguments are numpy's *args.

### broadcastShapes

```php
static broadcastShapes(mixed ...$args): mixed
```

The broadcast shape of the given shapes (numpy.broadcast_shapes).

numpy.broadcast_shapes: the positional arguments are numpy's *args.

### broadcastTo

```php
static broadcastTo(mixed $array, mixed $shape): mixed
```

Broadcast an array to a shape (numpy.broadcast_to).

numpy.broadcast_to

### choose

```php
static choose(mixed $a, mixed $choices, mixed $mode = 'raise'): mixed
```

Build an array from an index array and a list of choices (numpy.choose).

numpy.choose

### columnStack

```php
static columnStack(mixed $tup): mixed
```

Stack 1-D arrays as columns of a 2-D array (numpy.column_stack).

numpy.column_stack

### compress

```php
static compress(mixed $condition, mixed $a, mixed $axis = null): mixed
```

Selected slices along an axis (numpy.compress).

numpy.compress

### concat

```php
static concat(mixed $arrays, mixed $axis = 0, mixed $dtype = null, mixed $casting = 'same_kind'): mixed
```

Join arrays along an existing axis (numpy.concat, the Array API name of concatenate).

numpy.concat

### concatenate

```php
static concatenate(mixed $arrays, mixed $axis = 0, mixed $dtype = null, mixed $casting = 'same_kind'): mixed
```

Join arrays along an existing axis; axis=None flattens them first (numpy.concatenate).

numpy.concatenate

### convolve

```php
static convolve(mixed $a, mixed $v, mixed $mode = 'full'): mixed
```

Discrete linear convolution of two 1-D sequences (numpy.convolve).

numpy.convolve

### copyto

```php
static copyto(mixed $dst, mixed $src, mixed $casting = 'same_kind', mixed $where = null): mixed
```

Copy values into an array, broadcasting, in place (numpy.copyto).

numpy.copyto

### corrcoef

```php
static corrcoef(mixed $x, mixed $y = null, mixed $rowvar = true): mixed
```

Pearson correlation coefficients (numpy.corrcoef).

numpy.corrcoef

### correlate

```php
static correlate(mixed $a, mixed $v, mixed $mode = 'valid'): mixed
```

Cross-correlation of two 1-D sequences (numpy.correlate).

numpy.correlate

### countNonzero

```php
static countNonzero(mixed $a, array|int|null $axis = null, bool $keepdims = false): Tessero\NDArray|int|float
```

Number of non-zero values (numpy.count_nonzero).

numpy.count_nonzero

### cov

```php
static cov(mixed $m, mixed $y = null, mixed $rowvar = true, mixed $bias = false, mixed $ddof = null, mixed $fweights = null, mixed $aweights = null): mixed
```

Covariance matrix (numpy.cov).

numpy.cov

### cross

```php
static cross(mixed $a, mixed $b, mixed $axisa = -1, mixed $axisb = -1, mixed $axisc = -1, mixed $axis = null): mixed
```

Cross product of 3- (or 2-) element vectors (numpy.cross).

numpy.cross

### cumulativeProd

```php
static cumulativeProd(mixed $x, ?int $axis = null, int|float|bool|null $includeInitial = false): Tessero\NDArray|int|float
```

Cumulative product, optionally starting with 1 (numpy.cumulative_prod).

numpy.cumulative_prod

### cumulativeSum

```php
static cumulativeSum(mixed $x, ?int $axis = null, int|float|bool|null $includeInitial = false): Tessero\NDArray|int|float
```

Cumulative sum, optionally starting with 0 (numpy.cumulative_sum).

numpy.cumulative_sum

### delete

```php
static delete(mixed $arr, mixed $obj, mixed $axis = null): mixed
```

Remove the given indices along an axis (numpy.delete).

numpy.delete

### diag

```php
static diag(mixed $v, mixed $k = 0): mixed
```

Extract a diagonal, or build a 2-D array with the given diagonal (numpy.diag).

numpy.diag

### diagIndices

```php
static diagIndices(mixed $n, mixed $ndim = 2): mixed
```

Indices of the main diagonal of an n x ... x n array (numpy.diag_indices).

numpy.diag_indices

### diagIndicesFrom

```php
static diagIndicesFrom(mixed $arr): mixed
```

Indices of the main diagonal of an array (numpy.diag_indices_from).

numpy.diag_indices_from

### diagflat

```php
static diagflat(mixed $v, mixed $k = 0): mixed
```

A 2-D array with the flattened input as a diagonal (numpy.diagflat).

numpy.diagflat

### diagonal

```php
static diagonal(mixed $a, mixed $offset = 0, mixed $axis1 = 0, mixed $axis2 = 1): mixed
```

The given diagonal of an array (numpy.diagonal).

numpy.diagonal

### diff

```php
static diff(mixed $a, mixed $n = 1, mixed $axis = -1, mixed $prepend = null, mixed $append = null): mixed
```

n-th discrete difference along an axis (numpy.diff).

numpy.diff

### digitize

```php
static digitize(mixed $x, mixed $bins, int|float|bool|null $right = false): Tessero\NDArray|int|float
```

Indices of the bins to which each value belongs (numpy.digitize).

numpy.digitize

### dsplit

```php
static dsplit(mixed $ary, mixed $indicesOrSections): mixed
```

Split along the third axis (numpy.dsplit).

numpy.dsplit

### dstack

```php
static dstack(mixed $tup): mixed
```

Stack arrays depth-wise, along the third axis (numpy.dstack).

numpy.dstack

### ediff1d

```php
static ediff1d(mixed $ary, mixed $toEnd = null, mixed $toBegin = null): mixed
```

Differences between consecutive elements of the flattened array (numpy.ediff1d).

numpy.ediff1d

### emptyLike

```php
static emptyLike(mixed $a, mixed $dtype = null, mixed $shape = null): mixed
```

An array with the shape and dtype of another; Tessero fills it with zeros (numpy.empty_like).

numpy.empty_like

### expandDims

```php
static expandDims(mixed $a, mixed $axis): mixed
```

Insert length-1 axes (numpy.expand_dims).

numpy.expand_dims

### extract

```php
static extract(mixed $condition, mixed $arr): mixed
```

Elements where the condition holds, flattened (numpy.extract).

numpy.extract

### fillDiagonal

```php
static fillDiagonal(mixed $a, mixed $val, mixed $wrap = false): mixed
```

Fill the main diagonal, in place (numpy.fill_diagonal).

numpy.fill_diagonal

### flatnonzero

```php
static flatnonzero(mixed $a): mixed
```

Indices of the non-zero elements of the flattened array (numpy.flatnonzero).

numpy.flatnonzero

### flip

```php
static flip(mixed $m, mixed $axis = null): mixed
```

Reverse the order of elements along the given axes (numpy.flip).

numpy.flip

### fliplr

```php
static fliplr(mixed $m): mixed
```

Reverse the order along axis 1 (numpy.fliplr).

numpy.fliplr

### flipud

```php
static flipud(mixed $m): mixed
```

Reverse the order along axis 0 (numpy.flipud).

numpy.flipud

### fullLike

```php
static fullLike(mixed $a, mixed $fillValue, mixed $dtype = null, mixed $shape = null): mixed
```

An array of fill_value with the shape and dtype of another (numpy.full_like).

numpy.full_like

### geomspace

```php
static geomspace(mixed $start, mixed $stop, mixed $num = 50, mixed $endpoint = true, mixed $dtype = null, mixed $axis = 0): mixed
```

Numbers spaced evenly on a log scale, a geometric progression (numpy.geomspace).

numpy.geomspace

### gradient

```php
static gradient(mixed ...$varargs): mixed
```

Gradient by central differences, one-sided at the edges (numpy.gradient); several axes give a list.

numpy.gradient: the positional arguments are numpy's *varargs.

Named options: $f, $axis = null, $edgeOrder = 1.

### hamming

```php
static hamming(mixed $M): mixed
```

Hamming window (numpy.hamming).

numpy.hamming

### hanning

```php
static hanning(mixed $M): mixed
```

Hanning window (numpy.hanning).

numpy.hanning

### histogram

```php
static histogram(mixed $a, mixed $bins = 10, mixed $range = null, mixed $density = false, mixed $weights = null): mixed
```

Histogram of the data: counts (or density) and bin edges (numpy.histogram).

numpy.histogram

### histogramBinEdges

```php
static histogramBinEdges(mixed $a, mixed $bins = 10, mixed $range = null, mixed $weights = null): mixed
```

Bin edges numpy.histogram would use (numpy.histogram_bin_edges).

numpy.histogram_bin_edges

### hsplit

```php
static hsplit(mixed $ary, mixed $indicesOrSections): mixed
```

Split horizontally, column-wise (numpy.hsplit).

numpy.hsplit

### hstack

```php
static hstack(mixed $tup, mixed $dtype = null, mixed $casting = 'same_kind'): mixed
```

Stack arrays horizontally, column-wise (numpy.hstack).

numpy.hstack

### i0

```php
static i0(mixed $x): mixed
```

Modified Bessel function of the first kind, order 0 (numpy.i0).

numpy.i0

### identity

```php
static identity(mixed $n, mixed $dtype = null): mixed
```

The identity matrix (numpy.identity).

numpy.identity

### indices

```php
static indices(mixed $dimensions, mixed $dtype = 'int64', mixed $sparse = false): mixed
```

An array of grid indices (numpy.indices); sparse=True gives a list of arrays.

numpy.indices

### inner

```php
static inner(mixed $a, mixed $b): mixed
```

Inner product over the last axes (numpy.inner).

numpy.inner

### insert

```php
static insert(mixed $arr, mixed $obj, mixed $values, mixed $axis = null): mixed
```

Insert values before the given indices (numpy.insert).

numpy.insert

### interp

```php
static interp(mixed $x, mixed $xp, mixed $fp, mixed $left = null, mixed $right = null, mixed $period = null): mixed
```

One-dimensional piecewise linear interpolation (numpy.interp).

numpy.interp

### intersect1d

```php
static intersect1d(mixed $ar1, mixed $ar2, mixed $assumeUnique = false, mixed $returnIndices = false): mixed
```

Sorted unique values in both arrays (numpy.intersect1d).

numpy.intersect1d

### iscomplex

```php
static iscomplex(mixed $x): mixed
```

Element-wise test for a non-zero imaginary part (numpy.iscomplex).

numpy.iscomplex

### iscomplexobj

```php
static iscomplexobj(mixed $x): mixed
```

True for a complex dtype (numpy.iscomplexobj).

numpy.iscomplexobj

### isin

```php
static isin(mixed $element, mixed $testElements, int|float|bool|null $invert = false): Tessero\NDArray|int|float|bool
```

Whether each element is in test_elements (numpy.isin).

numpy.isin

### isneginf

```php
static isneginf(mixed $x): mixed
```

Element-wise test for negative infinity (numpy.isneginf).

numpy.isneginf

### isposinf

```php
static isposinf(mixed $x): mixed
```

Element-wise test for positive infinity (numpy.isposinf).

numpy.isposinf

### isreal

```php
static isreal(mixed $x): mixed
```

Element-wise test for a zero imaginary part (numpy.isreal).

numpy.isreal

### isrealobj

```php
static isrealobj(mixed $x): mixed
```

True for a non-complex dtype (numpy.isrealobj).

numpy.isrealobj

### ix_

```php
static ix_(mixed ...$args): mixed
```

An open mesh from several sequences (numpy.ix_).

numpy.ix_: the positional arguments are numpy's *args.

### kaiser

```php
static kaiser(mixed $M, mixed $beta): mixed
```

Kaiser window (numpy.kaiser).

numpy.kaiser

### kron

```php
static kron(mixed $a, mixed $b): mixed
```

Kronecker product (numpy.kron).

numpy.kron

### lexsort

```php
static lexsort(mixed $keys): mixed
```

Indirect stable sort on several keys, the last key primary (numpy.lexsort).

numpy.lexsort

### logspace

```php
static logspace(mixed $start, mixed $stop, mixed $num = 50, mixed $endpoint = true, mixed $base = 10.0, mixed $dtype = null, mixed $axis = 0): mixed
```

Numbers spaced evenly on a log scale (numpy.logspace).

numpy.logspace

### matrixTranspose

```php
static matrixTranspose(mixed $x): mixed
```

Transpose the last two axes (numpy.matrix_transpose).

numpy.matrix_transpose

### median

```php
static median(mixed $a, array|int|null $axis = null, bool $keepdims = false): Tessero\NDArray|int|float
```

Median along the given axes (numpy.median).

numpy.median

### meshgrid

```php
static meshgrid(mixed ...$xi): mixed
```

Coordinate matrices from coordinate vectors (numpy.meshgrid).

numpy.meshgrid: the positional arguments are numpy's *xi.

Named options: $copy = true, $sparse = false, $indexing = 'xy'.

### moveaxis

```php
static moveaxis(mixed $a, mixed $source, mixed $destination): mixed
```

Move axes to new positions (numpy.moveaxis).

numpy.moveaxis

### nanToNum

```php
static nanToNum(mixed $x, mixed $copy = true, mixed $nan = 0.0, mixed $posinf = null, mixed $neginf = null): mixed
```

Replace NaN and infinities by finite numbers (numpy.nan_to_num).

numpy.nan_to_num

### nanargmax

```php
static nanargmax(mixed $a, ?int $axis = null, bool $keepdims = false): Tessero\NDArray|int|float
```

Index of the maximum ignoring NaN (numpy.nanargmax).

numpy.nanargmax

### nanargmin

```php
static nanargmin(mixed $a, ?int $axis = null, bool $keepdims = false): Tessero\NDArray|int|float
```

Index of the minimum ignoring NaN (numpy.nanargmin).

numpy.nanargmin

### nancumprod

```php
static nancumprod(mixed $a, ?int $axis = null): Tessero\NDArray|int|float
```

Cumulative product treating NaN as one (numpy.nancumprod).

numpy.nancumprod

### nancumsum

```php
static nancumsum(mixed $a, ?int $axis = null): Tessero\NDArray|int|float
```

Cumulative sum treating NaN as zero (numpy.nancumsum).

numpy.nancumsum

### nanmax

```php
static nanmax(mixed $a, array|int|null $axis = null, bool $keepdims = false): Tessero\NDArray|int|float
```

Maximum ignoring NaN; NaN for an all-NaN slice (numpy.nanmax).

numpy.nanmax

### nanmean

```php
static nanmean(mixed $a, array|int|null $axis = null, bool $keepdims = false): Tessero\NDArray|int|float
```

Mean ignoring NaN (numpy.nanmean).

numpy.nanmean

### nanmedian

```php
static nanmedian(mixed $a, array|int|null $axis = null, bool $keepdims = false): Tessero\NDArray|int|float
```

Median ignoring NaNs (numpy.nanmedian).

numpy.nanmedian

### nanmin

```php
static nanmin(mixed $a, array|int|null $axis = null, bool $keepdims = false): Tessero\NDArray|int|float
```

Minimum ignoring NaN; NaN for an all-NaN slice (numpy.nanmin).

numpy.nanmin

### nanpercentile

```php
static nanpercentile(mixed $a, mixed $q, array|int|null $axis = null, string $method = 'linear', bool $keepdims = false): Tessero\NDArray|int|float|bool
```

Percentiles ignoring NaNs (numpy.nanpercentile).

numpy.nanpercentile

### nanprod

```php
static nanprod(mixed $a, array|int|null $axis = null, bool $keepdims = false): Tessero\NDArray|int|float
```

Product treating NaN as one (numpy.nanprod).

numpy.nanprod

### nanquantile

```php
static nanquantile(mixed $a, mixed $q, array|int|null $axis = null, string $method = 'linear', bool $keepdims = false): Tessero\NDArray|int|float|bool
```

Quantiles ignoring NaNs (numpy.nanquantile).

numpy.nanquantile

### nanstd

```php
static nanstd(mixed $a, array|int|null $axis = null, int|float|bool|null $ddof = 0, bool $keepdims = false): Tessero\NDArray|int|float
```

Standard deviation ignoring NaN (numpy.nanstd).

numpy.nanstd

### nansum

```php
static nansum(mixed $a, array|int|null $axis = null, bool $keepdims = false): Tessero\NDArray|int|float
```

Sum treating NaN as zero (numpy.nansum).

numpy.nansum

### nanvar

```php
static nanvar(mixed $a, array|int|null $axis = null, int|float|bool|null $ddof = 0, bool $keepdims = false): Tessero\NDArray|int|float
```

Variance ignoring NaN (numpy.nanvar).

numpy.nanvar

### ndim

```php
static ndim(mixed $a): mixed
```

Number of dimensions (numpy.ndim).

numpy.ndim

### nonzero

```php
static nonzero(mixed $a): mixed
```

Indices of the non-zero elements, one array per axis (numpy.nonzero).

numpy.nonzero

### onesLike

```php
static onesLike(mixed $a, mixed $dtype = null, mixed $shape = null): mixed
```

Ones with the shape and dtype of an array (numpy.ones_like).

numpy.ones_like

### outer

```php
static outer(mixed $a, mixed $b): mixed
```

Outer product of two vectors (numpy.outer).

numpy.outer

### pad

```php
static pad(mixed $array, mixed $padWidth, mixed $mode = 'constant', mixed $constantValues = null, mixed $endValues = null, mixed $statLength = null, mixed $reflectType = null): mixed
```

Pad an array (numpy.pad): constant, edge, linear_ramp, maximum, mean, median, minimum, reflect, symmetric, wrap, empty.

numpy.pad

### partition

```php
static partition(mixed $a, mixed $kth, ?int $axis = -1): Tessero\NDArray|int|float
```

Partially sorted copy: the kth element in its sorted place (numpy.partition).

numpy.partition

### percentile

```php
static percentile(mixed $a, mixed $q, array|int|null $axis = null, string $method = 'linear', bool $keepdims = false): Tessero\NDArray|int|float|bool
```

q-th percentiles along the given axes, q in [0, 100] (numpy.percentile).

numpy.percentile

### place

```php
static place(mixed $arr, mixed $mask, mixed $vals): mixed
```

Write successive values where the mask holds, in place (numpy.place).

numpy.place

### polyadd

```php
static polyadd(mixed $a1, mixed $a2): mixed
```

Sum of two polynomials (numpy.polyadd).

numpy.polyadd

### polyder

```php
static polyder(mixed $p, mixed $m = 1): mixed
```

Derivative of a polynomial (numpy.polyder).

numpy.polyder

### polydiv

```php
static polydiv(mixed $u, mixed $v): mixed
```

Quotient and remainder of polynomial division (numpy.polydiv).

numpy.polydiv

### polyint

```php
static polyint(mixed $p, mixed $m = 1, mixed $k = null): mixed
```

Antiderivative of a polynomial (numpy.polyint).

numpy.polyint

### polymul

```php
static polymul(mixed $a1, mixed $a2): mixed
```

Product of two polynomials (numpy.polymul).

numpy.polymul

### polysub

```php
static polysub(mixed $a1, mixed $a2): mixed
```

Difference of two polynomials (numpy.polysub).

numpy.polysub

### polyval

```php
static polyval(mixed $p, mixed $x): mixed
```

Evaluate a polynomial at x (numpy.polyval).

numpy.polyval

### ptp

```php
static ptp(mixed $a, array|int|null $axis = null, bool $keepdims = false): Tessero\NDArray|int|float
```

Range of values, maximum - minimum (numpy.ptp).

numpy.ptp

### put

```php
static put(mixed $a, mixed $ind, mixed $v, mixed $mode = 'raise'): mixed
```

Write values at flat indices, in place (numpy.put).

numpy.put

### putAlongAxis

```php
static putAlongAxis(mixed $arr, mixed $indices, mixed $values, mixed $axis): mixed
```

Write values at matching 1-D index slices, in place (numpy.put_along_axis).

numpy.put_along_axis

### putmask

```php
static putmask(mixed $a, mixed $mask, mixed $values): mixed
```

Write values where the mask holds, in place (numpy.putmask).

numpy.putmask

### quantile

```php
static quantile(mixed $a, mixed $q, array|int|null $axis = null, string $method = 'linear', bool $keepdims = false): Tessero\NDArray|int|float|bool
```

q-th quantiles along the given axes, q in [0, 1] (numpy.quantile).

numpy.quantile

### ravelMultiIndex

```php
static ravelMultiIndex(mixed $multiIndex, mixed $dims, mixed $mode = 'raise', mixed $order = 'C'): mixed
```

Convert coordinate arrays into flat indices (numpy.ravel_multi_index).

numpy.ravel_multi_index

### realIfClose

```php
static realIfClose(mixed $a, mixed $tol = 100): mixed
```

The real part when every imaginary part is close to zero (numpy.real_if_close).

numpy.real_if_close

### repeat

```php
static repeat(mixed $a, mixed $repeats, mixed $axis = null): mixed
```

Repeat each element (numpy.repeat).

numpy.repeat

### resize

```php
static resize(mixed $a, mixed $newShape): mixed
```

A new array of the given shape, repeating the data cyclically (numpy.resize).

numpy.resize

### roll

```php
static roll(mixed $a, mixed $shift, mixed $axis = null): mixed
```

Roll elements along the given axes; without axis on the flattened array (numpy.roll).

numpy.roll

### rollaxis

```php
static rollaxis(mixed $a, mixed $axis, mixed $start = 0): mixed
```

Roll an axis backwards to a position (numpy.rollaxis).

numpy.rollaxis

### rot90

```php
static rot90(mixed $m, mixed $k = 1, mixed $axes = null): mixed
```

Rotate by 90 degrees in the plane of two axes, default (0, 1) (numpy.rot90).

numpy.rot90

### searchsorted

```php
static searchsorted(mixed $a, mixed $v, string $side = 'left'): Tessero\NDArray|int|float
```

Indices where v would be inserted into the sorted array a (numpy.searchsorted).

numpy.searchsorted

### select

```php
static select(mixed $condlist, mixed $choicelist, mixed $default = 0): mixed
```

Elements from the choices where the conditions hold (numpy.select).

numpy.select

### setdiff1d

```php
static setdiff1d(mixed $ar1, mixed $ar2, mixed $assumeUnique = false): mixed
```

Unique values of ar1 that are not in ar2 (numpy.setdiff1d).

numpy.setdiff1d

### setxor1d

```php
static setxor1d(mixed $ar1, mixed $ar2, mixed $assumeUnique = false): mixed
```

Sorted values in exactly one of the arrays (numpy.setxor1d).

numpy.setxor1d

### shape

```php
static shape(mixed $a): mixed
```

The shape of an array (numpy.shape).

numpy.shape

### sinc

```php
static sinc(mixed $x): mixed
```

Normalised sinc, sin(pi x) / (pi x) (numpy.sinc).

numpy.sinc

### size

```php
static size(mixed $a, mixed $axis = null): mixed
```

Number of elements, in total or along axes (numpy.size).

numpy.size

### split

```php
static split(mixed $ary, mixed $indicesOrSections, mixed $axis = 0): mixed
```

Split into equal sections or at the given indices (numpy.split).

numpy.split

### squeeze

```php
static squeeze(mixed $a, mixed $axis = null): mixed
```

Remove length-1 axes (numpy.squeeze).

numpy.squeeze

### stack

```php
static stack(mixed $arrays, mixed $axis = 0, mixed $dtype = null, mixed $casting = 'same_kind'): mixed
```

Join arrays along a new axis (numpy.stack).

numpy.stack

### swapaxes

```php
static swapaxes(mixed $a, mixed $axis1, mixed $axis2): mixed
```

Interchange two axes (numpy.swapaxes).

numpy.swapaxes

### take

```php
static take(mixed $a, mixed $indices, mixed $axis = null, mixed $mode = 'raise'): mixed
```

Elements at the given indices along an axis (numpy.take).

numpy.take

### takeAlongAxis

```php
static takeAlongAxis(mixed $arr, mixed $indices, mixed $axis = -1): mixed
```

Pick values by matching 1-D index slices (numpy.take_along_axis).

numpy.take_along_axis

### tensordot

```php
static tensordot(mixed $a, mixed $b, mixed $axes = 2): mixed
```

Tensor dot product over the given axes (numpy.tensordot).

numpy.tensordot

### tile

```php
static tile(mixed $A, mixed $reps): mixed
```

Repeat an array the given number of times per axis (numpy.tile).

numpy.tile

### trace

```php
static trace(mixed $a, mixed $offset = 0, mixed $axis1 = 0, mixed $axis2 = 1, mixed $dtype = null): mixed
```

Sum along a diagonal (numpy.trace).

numpy.trace

### trapezoid

```php
static trapezoid(mixed $y, mixed $x = null, mixed $dx = 1.0, mixed $axis = -1): mixed
```

Integral by the composite trapezoidal rule (numpy.trapezoid).

numpy.trapezoid

### tri

```php
static tri(mixed $N, mixed $M = null, mixed $k = 0, mixed $dtype = 'float64'): mixed
```

Ones at and below the k-th diagonal, zeros elsewhere (numpy.tri).

numpy.tri

### tril

```php
static tril(mixed $m, mixed $k = 0): mixed
```

Lower triangle of an array (numpy.tril).

numpy.tril

### trilIndices

```php
static trilIndices(mixed $n, mixed $k = 0, mixed $m = null): mixed
```

Indices of the lower triangle of an (n, m) array (numpy.tril_indices).

numpy.tril_indices

### trilIndicesFrom

```php
static trilIndicesFrom(mixed $arr, mixed $k = 0): mixed
```

Indices of the lower triangle of a 2-D array (numpy.tril_indices_from).

numpy.tril_indices_from

### trimZeros

```php
static trimZeros(mixed $filt, mixed $trim = 'fb', mixed $axis = null): mixed
```

Trim leading and/or trailing zeros (numpy.trim_zeros).

numpy.trim_zeros

### triu

```php
static triu(mixed $m, mixed $k = 0): mixed
```

Upper triangle of an array (numpy.triu).

numpy.triu

### triuIndices

```php
static triuIndices(mixed $n, mixed $k = 0, mixed $m = null): mixed
```

Indices of the upper triangle of an (n, m) array (numpy.triu_indices).

numpy.triu_indices

### triuIndicesFrom

```php
static triuIndicesFrom(mixed $arr, mixed $k = 0): mixed
```

Indices of the upper triangle of a 2-D array (numpy.triu_indices_from).

numpy.triu_indices_from

### union1d

```php
static union1d(mixed $ar1, mixed $ar2): mixed
```

Sorted unique values in either array (numpy.union1d).

numpy.union1d

### unique

```php
static unique(mixed $ar, mixed $returnIndex = false, mixed $returnInverse = false, mixed $returnCounts = false, mixed $equalNan = true): mixed
```

Sorted unique elements, optionally with indices, inverse and counts (numpy.unique).

numpy.unique

### uniqueAll

```php
static uniqueAll(mixed $x): mixed
```

Unique values, first indices, inverse indices and counts (numpy.unique_all).

numpy.unique_all

### uniqueCounts

```php
static uniqueCounts(mixed $x): mixed
```

Unique values and their counts (numpy.unique_counts).

numpy.unique_counts

### uniqueInverse

```php
static uniqueInverse(mixed $x): mixed
```

Unique values and the inverse indices (numpy.unique_inverse).

numpy.unique_inverse

### uniqueValues

```php
static uniqueValues(mixed $x): mixed
```

Unique values; NaNs are not merged (numpy.unique_values).

numpy.unique_values

### unravelIndex

```php
static unravelIndex(mixed $indices, mixed $shape, mixed $order = 'C'): mixed
```

Convert flat indices into a tuple of coordinate arrays (numpy.unravel_index).

numpy.unravel_index

### unstack

```php
static unstack(mixed $x, mixed $axis = 0): mixed
```

Split an array into the arrays along an axis (numpy.unstack).

numpy.unstack

### unwrap

```php
static unwrap(mixed $p, mixed $discont = null, mixed $axis = -1, mixed $period = 6.283185307179586): mixed
```

Unwrap by taking the complement of large jumps with respect to the period (numpy.unwrap).

numpy.unwrap

### vander

```php
static vander(mixed $x, mixed $N = null, mixed $increasing = false): mixed
```

Vandermonde matrix (numpy.vander).

numpy.vander

### vdot

```php
static vdot(mixed $a, mixed $b): mixed
```

Dot product of the flattened arrays (numpy.vdot).

numpy.vdot

### vsplit

```php
static vsplit(mixed $ary, mixed $indicesOrSections): mixed
```

Split vertically, row-wise (numpy.vsplit).

numpy.vsplit

### vstack

```php
static vstack(mixed $tup, mixed $dtype = null, mixed $casting = 'same_kind'): mixed
```

Stack arrays vertically, row-wise (numpy.vstack).

numpy.vstack

### zerosLike

```php
static zerosLike(mixed $a, mixed $dtype = null, mixed $shape = null): mixed
```

Zeros with the shape and dtype of an array (numpy.zeros_like).

numpy.zeros_like
