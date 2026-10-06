# NDArray

`Tessero\NDArray` extends `Tessero\Ext\Operand` implements `ArrayAccess`, `Countable`, `IteratorAggregate`, `JsonSerializable`, `Traversable`

An n-dimensional, homogeneously typed array in native memory.

Layout is NumPy's: a Buffer, a byte offset, a shape and BYTE strides
(which may be zero for broadcast dimensions or negative for reversed
views). Slicing, transposing, reshaping a contiguous array, broadcasting
and expandDims/squeeze return views that share the Buffer; everything
that computes returns a new C-contiguous array. All loops run in
libtessero - PHP only computes shapes and strides.

## Methods

### allclose

```php
static allclose(mixed $a, mixed $b, float $rtol = 1.0E-5, float $atol = 1.0E-8, bool $equalNan = false): bool
```

### arange

```php
static arange(int|float $start, int|float|null $stop = null, int|float $step = 1, Tessero\DType|string|null $dtype = null): self
```

### array

```php
static array(mixed $data, Tessero\DType|string|null $dtype = null): self
```

Build an array from PHP data: a scalar, a (nested) list, or another NDArray (copied).
Passing $dtype skips type inference, which is the faster path for big inputs.

### asArray

```php
static asArray(mixed $value, Tessero\DType|string|null $dtype = null): self
```

### broadcastShapes

```php
static broadcastShapes(array ...$shapes): array
```

### cStrides

```php
static cStrides(array $shape, int $itemsize): array
```

### complex

```php
static complex(mixed $real, mixed $imag = null): self
```

Complex array from real and imaginary parts (arrays or nested lists of equal shape).

### concatenate

```php
static concatenate(array $arrays, int $axis = 0): self
```

### empty

```php
static empty(array|int $shape, Tessero\DType|string $dtype = Tessero\DType::Float64): self
```

### eye

```php
static eye(int $n, ?int $m = null, int $k = 0, Tessero\DType|string $dtype = Tessero\DType::Float64): self
```

### fromBytes

```php
static fromBytes(string $bytes, Tessero\DType|string $dtype, ?array $shape = null): self
```

Wrap raw little-endian bytes (copied).

### fromFlat

```php
static fromFlat(array $flat, array $shape, Tessero\DType|string $dtype = Tessero\DType::Float64): self
```

### full

```php
static full(array|int $shape, int|float|bool $value, Tessero\DType|string|null $dtype = null): self
```

### isclose

```php
static isclose(mixed $a, mixed $b, float $rtol = 1.0E-5, float $atol = 1.0E-8, bool $equalNan = false): self
```

### linspace

```php
static linspace(float $start, float $stop, int $num = 50, bool $endpoint = true): self
```

### load

```php
static load(string $path, ?string $mmapMode = null): self
```

Read a .npy file (numpy.load); with $mmapMode ('r', 'r+', 'c') the data is memory-mapped instead of read.

### memmap

```php
static memmap(string $filename, string $mode = 'r+', ?array $shape = null, Tessero\DType|string $dtype = Tessero\DType::Float64, int $offset = 0): self
```

An array backed by a memory-mapped file (numpy.memmap): nothing is read up front; the OS pages data
in on access. Modes: 'r' read-only, 'r+' read/write (extends a short file), 'w+' create or truncate,
'c' copy-on-write. Without $shape the file (after $offset) is one 1-D array. Views share the mapping;
results of computations are ordinary arrays. Same arguments and errors as ext-tessero.

### normAxis

```php
static normAxis(int $axis, int $ndim): int
```

### ones

```php
static ones(array|int $shape, Tessero\DType|string $dtype = Tessero\DType::Float64): self
```

### openMemmap

```php
static openMemmap(string $path, string $mode = 'r+', ?array $shape = null, Tessero\DType|string $dtype = Tessero\DType::Float64, bool $fortranOrder = false): self
```

A memory-mapped .npy file (numpy.lib.format.open_memmap): 'w+' creates it with a header for $shape and
$dtype; 'r', 'r+' and 'c' read shape and dtype from the file.

### stack

```php
static stack(array $arrays, int $axis = 0): self
```

### where

```php
static where(mixed $cond, mixed $a, mixed $b): self
```

Element-wise selection: cond ? a : b, with broadcasting.

### zeros

```php
static zeros(array|int $shape, Tessero\DType|string $dtype = Tessero\DType::Float64): self
```

### __construct

```php
__construct(Tessero\Native\Buffer $buffer, Tessero\DType $dtype, array $shape, ?array $strides = null, int $offset = 0)
```

### __serialize

```php
__serialize(): array
```

Queue/cache/session serialisation: dtype, shape and the raw little-endian bytes (C order).

### __toString

```php
__toString(): string
```

### __unserialize

```php
__unserialize(array $data): void
```

### abs

```php
abs(?self $out = null): self
```

### add

```php
add(mixed $b, ?self $out = null): self
```

### all

```php
all(array|int|null $axis = null, bool $keepdims = false): mixed
```

### angle

```php
angle(): self
```

### any

```php
any(array|int|null $axis = null, bool $keepdims = false): mixed
```

### arccos

```php
arccos(?self $out = null): self
```

### arcsin

```php
arcsin(?self $out = null): self
```

### arctan

```php
arctan(?self $out = null): self
```

### argmax

```php
argmax(?int $axis = null, bool $keepdims = false): mixed
```

### argmin

```php
argmin(?int $axis = null, bool $keepdims = false): mixed
```

### argsort

```php
argsort(int $axis = -1): self
```

### assign

```php
assign(mixed $src): self
```

Copy (and cast) the values of $src, broadcast to this array's shape, into this array (which may be a view).

### astype

```php
astype(Tessero\DType|string $dtype, bool $copy = false): self
```

### atan2

```php
atan2(mixed $b, ?self $out = null): self
```

### broadcastTo

```php
broadcastTo(array $shape): self
```

### buffer

```php
buffer(): Tessero\Native\Buffer
```

The native block this array (or view) reads; shared by all views of it. Advanced use only.

### ceil

```php
ceil(?self $out = null): self
```

### clip

```php
clip(int|float|null $min = null, int|float|null $max = null): self
```

### conj

```php
conj(): self
```

### contiguous

```php
contiguous(): self
```

This array if already C-contiguous, else a contiguous copy.

### copy

```php
copy(): self
```

### cos

```php
cos(?self $out = null): self
```

### cosh

```php
cosh(?self $out = null): self
```

### count

```php
count(): int
```

### cumprod

```php
cumprod(?int $axis = null): self
```

### cumsum

```php
cumsum(?int $axis = null): self
```

### diff

```php
diff(int $n = 1, int $axis = -1): self
```

n-th discrete difference along an axis (numpy.diff).

### div

```php
div(mixed $b, ?self $out = null): self
```

### dot

```php
dot(self|array $other): mixed
```

numpy.dot for 1-D/2-D operands (alias of matmul there).

### dtype

```php
dtype(): Tessero\DType
```

### eq

```php
eq(mixed $b): self
```

### exp

```php
exp(?self $out = null): self
```

### expandDims

```php
expandDims(int $axis): self
```

### expm1

```php
expm1(?self $out = null): self
```

### filename

```php
filename(): ?string
```

The mapped file's resolved path, or null.

### filter

```php
filter(self $mask): self
```

Elements where $mask is true, as a 1-D array (a[mask]).

### flatNonzero

```php
flatNonzero(): self
```

Flat indices of non-zero elements (numpy.flatnonzero).

### flatten

```php
flatten(): self
```

### floor

```php
floor(?self $out = null): self
```

### floorDiv

```php
floorDiv(mixed $b, ?self $out = null): self
```

### flush

```php
flush(bool $sync = false): self
```

Write a memory map's dirty pages back to its file: scheduled (default) or synchronous with $sync. No-op for other arrays.

### ge

```php
ge(mixed $b): self
```

### get

```php
get(int ...$index): mixed
```

Element or sub-array at integer indices (scalar when all axes are indexed).

### getIterator

```php
getIterator(): Traversable
```

### gt

```php
gt(mixed $b): self
```

### hypot

```php
hypot(mixed $b, ?self $out = null): self
```

### imag

```php
imag(): self
```

### invert

```php
invert(): self
```

Bitwise NOT for integers, logical NOT for bool (numpy.invert).

### isContiguous

```php
isContiguous(): bool
```

### isFortranContiguous

```php
isFortranContiguous(): bool
```

### isMemmap

```php
isMemmap(): bool
```

True for memory-mapped arrays and views of them.

### isReadonly

```php
isReadonly(): bool
```

True for arrays over read-only memory (memmap mode 'r').

### isWritable

```php
isWritable(): bool
```

False for arrays over read-only memory (a memory map opened with mode 'r').

### isfinite

```php
isfinite(): self
```

### isinf

```php
isinf(): self
```

### isnan

```php
isnan(): self
```

### item

```php
item(int ...$index): array|int|float|bool
```

One element as a PHP scalar ([re, im] for complex).

### jsonSerialize

```php
jsonSerialize(): mixed
```

### le

```php
le(mixed $b): self
```

### log

```php
log(?self $out = null): self
```

### log10

```php
log10(?self $out = null): self
```

### log1p

```php
log1p(?self $out = null): self
```

### log2

```php
log2(?self $out = null): self
```

### logicalAnd

```php
logicalAnd(mixed $b): self
```

### logicalNot

```php
logicalNot(): self
```

### logicalOr

```php
logicalOr(mixed $b): self
```

### logicalXor

```php
logicalXor(mixed $b): self
```

### lt

```php
lt(mixed $b): self
```

### matmul

```php
matmul(self|array $other): mixed
```

Matrix product with NumPy's matmul rules: 1-D operands are promoted to
row/column vectors, >2-D operands are stacks of matrices with broadcast
batch dimensions. float64/float32 use BLAS when one is available.

### max

```php
max(array|int|null $axis = null, bool $keepdims = false): mixed
```

### maximum

```php
maximum(mixed $b, ?self $out = null): self
```

### mean

```php
mean(array|int|null $axis = null, bool $keepdims = false): mixed
```

### meta

```php
meta(): FFI\CData
```

C-side metadata (tsr_array) describing this view, for kernels that take whole arrays.
Like ptr(), the embedded data pointer is valid only while this array is alive.

### min

```php
min(array|int|null $axis = null, bool $keepdims = false): mixed
```

### minimum

```php
minimum(mixed $b, ?self $out = null): self
```

### mod

```php
mod(mixed $b, ?self $out = null): self
```

### moveAxis

```php
moveAxis(int $source, int $destination): self
```

### mul

```php
mul(mixed $b, ?self $out = null): self
```

### nbytes

```php
nbytes(): int
```

### ndim

```php
ndim(): int
```

### ne

```php
ne(mixed $b): self
```

### neg

```php
neg(?self $out = null): self
```

### offset

```php
offset(): int
```

### offsetExists

```php
offsetExists(mixed $offset): bool
```

### offsetGet

```php
offsetGet(mixed $offset): mixed
```

$a[2], $a[-1], $a['1:5'], $a['::2, 3'], $a['..., 0'], $a[$boolMask], $a[[0, 2, 5]], $a[$intArray]

### offsetSet

```php
offsetSet(mixed $offset, mixed $value): void
```

### offsetUnset

```php
offsetUnset(mixed $offset): void
```

### pow

```php
pow(mixed $b, ?self $out = null): self
```

### prod

```php
prod(array|int|null $axis = null, bool $keepdims = false): mixed
```

### ptr

```php
ptr(): FFI\CData
```

char* to the first element, for passing to C in the same statement.

The pointer does not keep the memory alive: keep this array in a variable until the C call returns,
and never store the pointer. Throws if the memory was already released (PHP shutdown ordering).

### put

```php
put(mixed $indices, mixed $values, ?int $axis = null): self
```

Scatter values along an axis in place (numpy.put_along_axis for whole slices).

### ravel

```php
ravel(): self
```

### rdiv

```php
rdiv(mixed $a): self
```

### real

```php
real(): self
```

### reciprocal

```php
reciprocal(?self $out = null): self
```

### reshape

```php
reshape(array|int ...$shape): self
```

### rint

```php
rint(?self $out = null): self
```

### round

```php
round(int $decimals = 0): self
```

### rpow

```php
rpow(mixed $a): self
```

### rsub

```php
rsub(mixed $a): self
```

Reflected forms for scalar - array expressions.

### save

```php
save(string $path): void
```

Write this array to a .npy file (numpy.save), C order.

### setWhere

```php
setWhere(self $mask, mixed $values): self
```

a[mask] = values (a scalar, or one value per true element).

### shape

```php
shape(): array
```

### sign

```php
sign(?self $out = null): self
```

### sin

```php
sin(?self $out = null): self
```

### sinh

```php
sinh(?self $out = null): self
```

### size

```php
size(): int
```

### slice

```php
slice(string|int|null ...$specs): self
```

Basic indexing, one spec per axis:
  int         select (drops the axis; negative counts from the end)
  'a:b:c'     slice, any part optional ('::-1' reverses, ':' keeps)
  '...'       fill the remaining axes with ':'
  null        insert a new axis of length 1 (numpy.newaxis)
Always returns a view.

### sliceNative

```php
sliceNative(string $spec): self
```

Basic indexing with the spec parsed by libtessero (same grammar as offsetGet):
one call into C, no PHP string handling. Returns a view.

### sort

```php
sort(int $axis = -1): self
```

### sqrt

```php
sqrt(?self $out = null): self
```

### square

```php
square(?self $out = null): self
```

### squeeze

```php
squeeze(?int $axis = null): self
```

### std

```php
std(array|int|null $axis = null, int $ddof = 0, bool $keepdims = false): mixed
```

### strides

```php
strides(): array
```

### sub

```php
sub(mixed $b, ?self $out = null): self
```

### sum

```php
sum(array|int|null $axis = null, bool $keepdims = false): mixed
```

### swapAxes

```php
swapAxes(int $a, int $b): self
```

### t

```php
t(): self
```

Matrix transpose (reverses all axes).

### take

```php
take(mixed $indices, ?int $axis = null): self
```

Gather along an axis (numpy.take). Indices may be negative. With $axis
null the array is flattened first.

### tan

```php
tan(?self $out = null): self
```

### tanh

```php
tanh(?self $out = null): self
```

### toArray

```php
toArray(): mixed
```

Nested PHP array (scalar for 0-d). Complex elements become [re, im] pairs.

### toBytes

```php
toBytes(): string
```

Raw little-endian bytes in C order.

### toJson

```php
toJson(): string
```

JSON text written directly by libtessero from the strided memory (no PHP
arrays in between): about 10x faster than json_encode($a->toArray()) on
large arrays. Floats use the shortest round-trip form; NaN/Inf become null.

### toList

```php
toList(): array
```

Flat C-order list of values (complex: interleaved re, im).

### transpose

```php
transpose(?array $axes = null): self
```

### var

```php
var(array|int|null $axis = null, int $ddof = 0, bool $keepdims = false): mixed
```

### view

```php
view(Tessero\DType|string $dtype): self
```

Reinterpret the same bytes as another dtype of equal itemsize (bool <-> uint8).
