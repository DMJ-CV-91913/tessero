# Np

`Tessero\Ext\Np` *(native extension)*

## Methods

### append

```php
static append($arr, $values, $axis = null)
```

### argpartition

```php
static argpartition($a, $kth, $axis = -1)
```

### argwhere

```php
static argwhere($a)
```

### arrayEqual

```php
static arrayEqual($a1, $a2, $equalNan = false)
```

### arrayEquiv

```php
static arrayEquiv($a1, $a2)
```

### arraySplit

```php
static arraySplit($ary, $indicesOrSections, $axis = 0)
```

### atleast1d

```php
static atleast1d(...$arys)
```

### atleast2d

```php
static atleast2d(...$arys)
```

### atleast3d

```php
static atleast3d(...$arys)
```

### average

```php
static average($a, $weights = null, $axis = null, $keepdims = false)
```

### bartlett

```php
static bartlett($M)
```

### bincount

```php
static bincount($x, $weights = null, $minlength = 0)
```

### blackman

```php
static blackman($M)
```

### broadcastArrays

```php
static broadcastArrays(...$args)
```

### broadcastShapes

```php
static broadcastShapes(...$args)
```

### broadcastTo

```php
static broadcastTo($array, $shape)
```

### choose

```php
static choose($a, $choices, $mode = 'raise')
```

### columnStack

```php
static columnStack($tup)
```

### compress

```php
static compress($condition, $a, $axis = null)
```

### concat

```php
static concat($arrays, $axis = 0, $dtype = null, $casting = 'same_kind')
```

### concatenate

```php
static concatenate($arrays, $axis = 0, $dtype = null, $casting = 'same_kind')
```

### convolve

```php
static convolve($a, $v, $mode = 'full')
```

### copyto

```php
static copyto($dst, $src, $casting = 'same_kind', $where = null)
```

### corrcoef

```php
static corrcoef($x, $y = null, $rowvar = true)
```

### correlate

```php
static correlate($a, $v, $mode = 'valid')
```

### countNonzero

```php
static countNonzero($a, $axis = null, $keepdims = false)
```

### cov

```php
static cov($m, $y = null, $rowvar = true, $bias = false, $ddof = null, $fweights = null, $aweights = null)
```

### cross

```php
static cross($a, $b, $axisa = -1, $axisb = -1, $axisc = -1, $axis = null)
```

### cumulativeProd

```php
static cumulativeProd($x, $axis = null, $includeInitial = false)
```

### cumulativeSum

```php
static cumulativeSum($x, $axis = null, $includeInitial = false)
```

### delete

```php
static delete($arr, $obj, $axis = null)
```

### diag

```php
static diag($v, $k = 0)
```

### diagIndices

```php
static diagIndices($n, $ndim = 2)
```

### diagIndicesFrom

```php
static diagIndicesFrom($arr)
```

### diagflat

```php
static diagflat($v, $k = 0)
```

### diagonal

```php
static diagonal($a, $offset = 0, $axis1 = 0, $axis2 = 1)
```

### diff

```php
static diff($a, $n = 1, $axis = -1, $prepend = null, $append = null)
```

### digitize

```php
static digitize($x, $bins, $right = false)
```

### dsplit

```php
static dsplit($ary, $indicesOrSections)
```

### dstack

```php
static dstack($tup)
```

### ediff1d

```php
static ediff1d($ary, $toEnd = null, $toBegin = null)
```

### emptyLike

```php
static emptyLike($a, $dtype = null, $shape = null)
```

### expandDims

```php
static expandDims($a, $axis)
```

### extract

```php
static extract($condition, $arr)
```

### fillDiagonal

```php
static fillDiagonal($a, $val, $wrap = false)
```

### flatnonzero

```php
static flatnonzero($a)
```

### flip

```php
static flip($m, $axis = null)
```

### fliplr

```php
static fliplr($m)
```

### flipud

```php
static flipud($m)
```

### fullLike

```php
static fullLike($a, $fillValue, $dtype = null, $shape = null)
```

### geomspace

```php
static geomspace($start, $stop, $num = 50, $endpoint = true, $dtype = null, $axis = 0)
```

### gradient

```php
static gradient($f, ...$varargs)
```

### hamming

```php
static hamming($M)
```

### hanning

```php
static hanning($M)
```

### histogram

```php
static histogram($a, $bins = 10, $range = null, $density = false, $weights = null)
```

### histogramBinEdges

```php
static histogramBinEdges($a, $bins = 10, $range = null, $weights = null)
```

### hsplit

```php
static hsplit($ary, $indicesOrSections)
```

### hstack

```php
static hstack($tup, $dtype = null, $casting = 'same_kind')
```

### i0

```php
static i0($x)
```

### identity

```php
static identity($n, $dtype = null)
```

### indices

```php
static indices($dimensions, $dtype = 'int64', $sparse = false)
```

### inner

```php
static inner($a, $b)
```

### insert

```php
static insert($arr, $obj, $values, $axis = null)
```

### interp

```php
static interp($x, $xp, $fp, $left = null, $right = null, $period = null)
```

### intersect1d

```php
static intersect1d($ar1, $ar2, $assumeUnique = false, $returnIndices = false)
```

### iscomplex

```php
static iscomplex($x)
```

### iscomplexobj

```php
static iscomplexobj($x)
```

### isin

```php
static isin($element, $testElements, $invert = false)
```

### isneginf

```php
static isneginf($x)
```

### isposinf

```php
static isposinf($x)
```

### isreal

```php
static isreal($x)
```

### isrealobj

```php
static isrealobj($x)
```

### ix_

```php
static ix_(...$args)
```

### kaiser

```php
static kaiser($M, $beta)
```

### kron

```php
static kron($a, $b)
```

### lexsort

```php
static lexsort($keys)
```

### logspace

```php
static logspace($start, $stop, $num = 50, $endpoint = true, $base = 10.0, $dtype = null, $axis = 0)
```

### matrixTranspose

```php
static matrixTranspose($x)
```

### median

```php
static median($a, $axis = null, $keepdims = false)
```

### meshgrid

```php
static meshgrid(...$xi)
```

### moveaxis

```php
static moveaxis($a, $source, $destination)
```

### nanToNum

```php
static nanToNum($x, $copy = true, $nan = 0.0, $posinf = null, $neginf = null)
```

### nanargmax

```php
static nanargmax($a, $axis = null, $keepdims = false)
```

### nanargmin

```php
static nanargmin($a, $axis = null, $keepdims = false)
```

### nancumprod

```php
static nancumprod($a, $axis = null)
```

### nancumsum

```php
static nancumsum($a, $axis = null)
```

### nanmax

```php
static nanmax($a, $axis = null, $keepdims = false)
```

### nanmean

```php
static nanmean($a, $axis = null, $keepdims = false)
```

### nanmedian

```php
static nanmedian($a, $axis = null, $keepdims = false)
```

### nanmin

```php
static nanmin($a, $axis = null, $keepdims = false)
```

### nanpercentile

```php
static nanpercentile($a, $q, $axis = null, $method = 'linear', $keepdims = false)
```

### nanprod

```php
static nanprod($a, $axis = null, $keepdims = false)
```

### nanquantile

```php
static nanquantile($a, $q, $axis = null, $method = 'linear', $keepdims = false)
```

### nanstd

```php
static nanstd($a, $axis = null, $ddof = 0, $keepdims = false)
```

### nansum

```php
static nansum($a, $axis = null, $keepdims = false)
```

### nanvar

```php
static nanvar($a, $axis = null, $ddof = 0, $keepdims = false)
```

### ndim

```php
static ndim($a)
```

### nonzero

```php
static nonzero($a)
```

### onesLike

```php
static onesLike($a, $dtype = null, $shape = null)
```

### outer

```php
static outer($a, $b)
```

### pad

```php
static pad($array, $padWidth, $mode = 'constant', $constantValues = null, $endValues = null, $statLength = null, $reflectType = null)
```

### partition

```php
static partition($a, $kth, $axis = -1)
```

### percentile

```php
static percentile($a, $q, $axis = null, $method = 'linear', $keepdims = false)
```

### place

```php
static place($arr, $mask, $vals)
```

### polyadd

```php
static polyadd($a1, $a2)
```

### polyder

```php
static polyder($p, $m = 1)
```

### polydiv

```php
static polydiv($u, $v)
```

### polyint

```php
static polyint($p, $m = 1, $k = null)
```

### polymul

```php
static polymul($a1, $a2)
```

### polysub

```php
static polysub($a1, $a2)
```

### polyval

```php
static polyval($p, $x)
```

### ptp

```php
static ptp($a, $axis = null, $keepdims = false)
```

### put

```php
static put($a, $ind, $v, $mode = 'raise')
```

### putAlongAxis

```php
static putAlongAxis($arr, $indices, $values, $axis)
```

### putmask

```php
static putmask($a, $mask, $values)
```

### quantile

```php
static quantile($a, $q, $axis = null, $method = 'linear', $keepdims = false)
```

### ravelMultiIndex

```php
static ravelMultiIndex($multiIndex, $dims, $mode = 'raise', $order = 'C')
```

### realIfClose

```php
static realIfClose($a, $tol = 100)
```

### repeat

```php
static repeat($a, $repeats, $axis = null)
```

### resize

```php
static resize($a, $newShape)
```

### roll

```php
static roll($a, $shift, $axis = null)
```

### rollaxis

```php
static rollaxis($a, $axis, $start = 0)
```

### rot90

```php
static rot90($m, $k = 1, $axes = null)
```

### searchsorted

```php
static searchsorted($a, $v, $side = 'left')
```

### select

```php
static select($condlist, $choicelist, $default = 0)
```

### setdiff1d

```php
static setdiff1d($ar1, $ar2, $assumeUnique = false)
```

### setxor1d

```php
static setxor1d($ar1, $ar2, $assumeUnique = false)
```

### shape

```php
static shape($a)
```

### sinc

```php
static sinc($x)
```

### size

```php
static size($a, $axis = null)
```

### split

```php
static split($ary, $indicesOrSections, $axis = 0)
```

### squeeze

```php
static squeeze($a, $axis = null)
```

### stack

```php
static stack($arrays, $axis = 0, $dtype = null, $casting = 'same_kind')
```

### swapaxes

```php
static swapaxes($a, $axis1, $axis2)
```

### take

```php
static take($a, $indices, $axis = null, $mode = 'raise')
```

### takeAlongAxis

```php
static takeAlongAxis($arr, $indices, $axis = -1)
```

### tensordot

```php
static tensordot($a, $b, $axes = 2)
```

### tile

```php
static tile($A, $reps)
```

### trace

```php
static trace($a, $offset = 0, $axis1 = 0, $axis2 = 1, $dtype = null)
```

### trapezoid

```php
static trapezoid($y, $x = null, $dx = 1.0, $axis = -1)
```

### tri

```php
static tri($N, $M = null, $k = 0, $dtype = 'float64')
```

### tril

```php
static tril($m, $k = 0)
```

### trilIndices

```php
static trilIndices($n, $k = 0, $m = null)
```

### trilIndicesFrom

```php
static trilIndicesFrom($arr, $k = 0)
```

### trimZeros

```php
static trimZeros($filt, $trim = 'fb', $axis = null)
```

### triu

```php
static triu($m, $k = 0)
```

### triuIndices

```php
static triuIndices($n, $k = 0, $m = null)
```

### triuIndicesFrom

```php
static triuIndicesFrom($arr, $k = 0)
```

### union1d

```php
static union1d($ar1, $ar2)
```

### unique

```php
static unique($ar, $returnIndex = false, $returnInverse = false, $returnCounts = false, $equalNan = true)
```

### uniqueAll

```php
static uniqueAll($x)
```

### uniqueCounts

```php
static uniqueCounts($x)
```

### uniqueInverse

```php
static uniqueInverse($x)
```

### uniqueValues

```php
static uniqueValues($x)
```

### unravelIndex

```php
static unravelIndex($indices, $shape, $order = 'C')
```

### unstack

```php
static unstack($x, $axis = 0)
```

### unwrap

```php
static unwrap($p, $discont = null, $axis = -1, $period = 6.283185307179586)
```

### vander

```php
static vander($x, $N = null, $increasing = false)
```

### vdot

```php
static vdot($a, $b)
```

### vsplit

```php
static vsplit($ary, $indicesOrSections)
```

### vstack

```php
static vstack($tup, $dtype = null, $casting = 'same_kind')
```

### zerosLike

```php
static zerosLike($a, $dtype = null, $shape = null)
```
