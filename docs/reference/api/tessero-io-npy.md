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
static encode(Tessero\NDArray|Tessero\Ext\NDArray $array): string
```

### load

```php
static load(string $path): Tessero\NDArray
```

$path must be a local file (stream wrappers are refused); never build it from request input.

### loadZ

```php
static loadZ(string $path, int $maxBytes = 268435456): array
```

Read every array of a .npz archive. The archive may inflate to at most $maxBytes in total
(default 256 MiB, checked before decompressing) and hold at most IoGuard::MAX_ZIP_ENTRIES members.

### save

```php
static save(string $path, Tessero\NDArray $array): void
```

$path must be a local file (stream wrappers are refused); never build it from request input.

### saveZ

```php
static saveZ(string $path, array $arrays, bool $compress = true): void
```
