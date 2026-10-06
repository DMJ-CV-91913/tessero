# DType

`Tessero\DType`

Element types. The integer values are the libtessero ABI codes.

Promotion follows NumPy 2 (NEP 50): array-array promotion by kind and size;
a PHP int/float scalar is "weak" and adopts the array's dtype unless its
kind is higher (a float scalar with an int array gives float64).

## Cases

- `Float64` = `0`
- `Float32` = `1`
- `Int64` = `2`
- `Int32` = `3`
- `UInt8` = `4`
- `Bool` = `5`
- `Complex128` = `6`

## Methods

### cases

```php
static cases(): array
```

### from

```php
static from(string|int $value): static
```

### fromDescr

```php
static fromDescr(string $descr): self
```

### from_

```php
static from_(Tessero\DType|string $d): self
```

### ofValue

```php
static ofValue(mixed $v): self
```

Smallest dtype that holds a PHP value (for array() inference).

### promote

```php
static promote(self $a, self $b): self
```

### tryFrom

```php
static tryFrom(string|int $value): ?static
```

### descr

```php
descr(): string
```

NumPy .npy descr string.

### isComplex

```php
isComplex(): bool
```

### isFloat

```php
isFloat(): bool
```

### isInteger

```php
isInteger(): bool
```

### itemsize

```php
itemsize(): int
```

### name

```php
name(): string
```

### packFormat

```php
packFormat(): string
```

pack()/unpack() format for one element (little-endian hosts; see Buffer).

### toFloat

```php
toFloat(): self
```

Float dtype used for results of true division, sqrt, exp, ...

### withScalar

```php
withScalar(int|float|bool $scalar): self
```

Result dtype when a PHP scalar (weakly typed) meets an array of this dtype.
