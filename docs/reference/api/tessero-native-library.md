# Library

`Tessero\Native\Library`

Loads libtessero once per process.

Resolution order:
  1. FFI::scope('tessero')      - the library was bound by opcache.preload
                                  (required under PHP-FPM/Apache with ffi.enable=preload);
  2. FFI::cdef(header, path)    - CLI, or any SAPI with ffi.enable=1.
The shared object is looked up in $TESSERO_LIB, then the prebuilt binary
shipped for this platform in lib/<os>-<arch>/.

Pointers into native memory are created with ptr($address). Pointer
arithmetic on CData (`$p + $n`) is deliberately never used: on PHP 8.4
passing such a temporary to a function silently breaks later arithmetic
on the source pointer (see docs/project/upstream-bugs.md, issue 2).

## Constants

| Name | Value |
|---|---|
| `SCOPE` | `'tessero'` |

## Methods

### address

```php
static address(FFI\CData $pointer): int
```

Integer address of any pointer CData (void* is routed through char*: a direct void*->uintptr_t cast yields 0 on PHP 8.4).

### available

```php
static available(): bool
```

### check

```php
static check(int $rc, string $what): int
```

Map a negative libtessero return code to an exception.

### ffi

```php
static ffi(): FFI
```

### header

```php
static header(): string
```

### headerPath

```php
static headerPath(): string
```

### i64

```php
static i64(array $values): FFI\CData
```

### locate

```php
static locate(): ?string
```

### mode

```php
static mode(): string
```

'scope' (preloaded), 'cdef' or 'unloaded'.

### path

```php
static path(): ?string
```

### platform

```php
static platform(): string
```

### ptr

```php
static ptr(int $address): FFI\CData
```

char* at an absolute address. The caller keeps the owning Buffer alive.
