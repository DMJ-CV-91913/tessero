# Npy

`Tessero\Io\Npy`

NumPy .npy (format 1.0 / 2.0 / 3.0) and .npz reading and writing, so arrays
move between PHP and Python without JSON: np.load('x.npy') reads what save()
writes and vice versa. Headers are parsed by libtessero's strict parser
(tsr_npy_header), the same code ext-tessero uses. For memory-mapped .npy
files see NDArray::openMemmap() and NDArray::load($path, $mmapMode).

## Methods

### decode

```php
static decode(string $bytes, string $what = '.npy data'): Tessero\NDArray
```

### encode

```php
static encode(Tessero\NDArray $array): string
```

### load

```php
static load(string $path): Tessero\NDArray
```

### loadZ

```php
static loadZ(string $path): array
```

### save

```php
static save(string $path, Tessero\NDArray $array): void
```

### saveZ

```php
static saveZ(string $path, array $arrays, bool $compress = true): void
```
