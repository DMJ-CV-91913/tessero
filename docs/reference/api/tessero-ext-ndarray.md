# NDArray

`Tessero\Ext\NDArray` implements `IteratorAggregate`, `Traversable`, `Countable`, `JsonSerializable` *(native extension)*

## Methods

### arange

```php
static arange($start, $stop = …, $step = …, ?string $dtype = …)
```

### array

```php
static array($data, ?string $dtype = …)
```

### eye

```php
static eye($n, $m = …, $k = …, ?string $dtype = …)
```

### fromBytes

```php
static fromBytes(string $bytes, string $dtype, $shape = …)
```

### full

```php
static full($shape, $value, ?string $dtype = …)
```

### linspace

```php
static linspace($start, $stop, $num = …, $endpoint = …)
```

### load

```php
static load(string $path, ?string $mmapMode = …)
```

### memmap

```php
static memmap(string $filename, string $mode = …, ?array $shape = …, string $dtype = …, int $offset = …)
```

### ones

```php
static ones($shape, ?string $dtype = …)
```

### openMemmap

```php
static openMemmap(string $path, string $mode = …, ?array $shape = …, string $dtype = …, bool $fortranOrder = …)
```

### where

```php
static where($cond, $x, $y)
```

### zeros

```php
static zeros($shape, ?string $dtype = …)
```

### __serialize

```php
__serialize(): array
```

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
abs()
```

### add

```php
add($value)
```

### all

```php
all(array|int|null $axis = null, bool $keepdims = false)
```

### angle

```php
angle()
```

### any

```php
any(array|int|null $axis = null, bool $keepdims = false)
```

### arccos

```php
arccos()
```

### arcsin

```php
arcsin()
```

### arctan

```php
arctan()
```

### argmax

```php
argmax(?int $axis = null, bool $keepdims = false)
```

### argmin

```php
argmin(?int $axis = null, bool $keepdims = false)
```

### argsort

```php
argsort()
```

### assign

```php
assign($value)
```

### astype

```php
astype(string $dtype)
```

### atan2

```php
atan2($value)
```

### ceil

```php
ceil()
```

### clip

```php
clip($min = …, $max = …)
```

### conj

```php
conj()
```

### copy

```php
copy()
```

### cos

```php
cos()
```

### cosh

```php
cosh()
```

### count

```php
count(): int
```

### cumprod

```php
cumprod(?int $axis = …)
```

### cumsum

```php
cumsum(?int $axis = …)
```

### div

```php
div($value)
```

### dot

```php
dot($value)
```

### dtype

```php
dtype()
```

### eq

```php
eq($value)
```

### exp

```php
exp()
```

### expm1

```php
expm1()
```

### filename

```php
filename()
```

### floor

```php
floor()
```

### floorDiv

```php
floorDiv($value)
```

### flush

```php
flush(bool $sync = …)
```

### ge

```php
ge($value)
```

### getIterator

```php
getIterator(): Iterator
```

### gt

```php
gt($value)
```

### hypot

```php
hypot($value)
```

### imag

```php
imag()
```

### invert

```php
invert()
```

### isContiguous

```php
isContiguous()
```

### isMemmap

```php
isMemmap()
```

### isReadonly

```php
isReadonly()
```

### isView

```php
isView()
```

### isfinite

```php
isfinite()
```

### isinf

```php
isinf()
```

### isnan

```php
isnan()
```

### item

```php
item(...$args)
```

### jsonSerialize

```php
jsonSerialize(): mixed
```

### le

```php
le($value)
```

### log

```php
log()
```

### log10

```php
log10()
```

### log1p

```php
log1p()
```

### log2

```php
log2()
```

### logicalAnd

```php
logicalAnd($value)
```

### logicalNot

```php
logicalNot()
```

### logicalOr

```php
logicalOr($value)
```

### logicalXor

```php
logicalXor($value)
```

### lt

```php
lt($value)
```

### matmul

```php
matmul($value)
```

### max

```php
max(array|int|null $axis = null, bool $keepdims = false)
```

### maximum

```php
maximum($value)
```

### mean

```php
mean(array|int|null $axis = null, bool $keepdims = false)
```

### min

```php
min(array|int|null $axis = null, bool $keepdims = false)
```

### minimum

```php
minimum($value)
```

### mod

```php
mod($value)
```

### mul

```php
mul($value)
```

### ndim

```php
ndim()
```

### ne

```php
ne($value)
```

### neg

```php
neg()
```

### pow

```php
pow($value)
```

### prod

```php
prod(array|int|null $axis = null, bool $keepdims = false)
```

### ravel

```php
ravel()
```

### rdiv

```php
rdiv($value)
```

### real

```php
real()
```

### reciprocal

```php
reciprocal()
```

### reshape

```php
reshape(...$args)
```

### rint

```php
rint()
```

### rpow

```php
rpow($value)
```

### rsub

```php
rsub($value)
```

### save

```php
save(string $path)
```

### shape

```php
shape()
```

### sign

```php
sign()
```

### sin

```php
sin()
```

### sinh

```php
sinh()
```

### size

```php
size()
```

### slice

```php
slice(string $spec)
```

### sort

```php
sort()
```

### sqrt

```php
sqrt()
```

### square

```php
square()
```

### std

```php
std(array|int|null $axis = null, int $ddof = 0, bool $keepdims = false)
```

### strides

```php
strides()
```

### sub

```php
sub($value)
```

### sum

```php
sum(array|int|null $axis = null, bool $keepdims = false)
```

### t

```php
t()
```

### tan

```php
tan()
```

### tanh

```php
tanh()
```

### toArray

```php
toArray()
```

### toBytes

```php
toBytes()
```

### toJson

```php
toJson()
```

### toList

```php
toList()
```

### transpose

```php
transpose(?array $axes = …)
```

### var

```php
var(array|int|null $axis = null, int $ddof = 0, bool $keepdims = false)
```
