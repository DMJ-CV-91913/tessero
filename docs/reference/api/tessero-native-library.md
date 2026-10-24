# Library

`Tessero\Native\Library`

Loads libtessero once per process.

Resolution order:
  1. FFI::scope('tessero')      - the library was bound by opcache.preload
                                  (required under PHP-FPM/Apache with ffi.enable=preload);
  2. FFI::cdef(header, path)    - CLI, or any SAPI with ffi.enable=1.
The shared object is looked up in $TESSERO_LIB, then the prebuilt binary
shipped for this platform in lib/<os>-<arch>/.

Pointers into native memory are created by Buffer::ptr(). Pointer
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

### defaultBlasThreads

```php
static defaultBlasThreads(string $sapi, string|false $env): ?int
```

The BLAS thread cap to apply at load, or null to leave the BLAS alone: TESSERO_BLAS_THREADS=n (n >= 1)
wins, 0 means leave alone, and otherwise web SAPIs get 1 and the CLI keeps the BLAS default.

### defaultBudget

```php
static defaultBudget(string $sapi, string|false $env, string $memoryLimit, int $current): ?int
```

The native memory budget a non-CLI process starts with, or null to leave the kernel's setting alone.
Native buffers do not count toward memory_limit, so outside the CLI the FFI package caps them by
default at max(256 MiB, memory_limit), as the extension does. TESSERO_MEMORY_BUDGET (bytes, process
environment) overrides it; 0 or a negative value opts out (unlimited). memory_limit = -1 also means
unlimited. Tessero::setMemoryBudget() changes the budget at any time; under FPM the default is
re-applied when a request loads the library while the budget is 0.

### ffi

```php
static ffi(): FFI
```

The bound library. Internal: it reaches raw native memory, so application code must not use it.
Under ffi.enable=preload it is a preloaded method, so any request code that calls it gets full FFI
access; PHP cannot restrict that (Reflection reaches private members too), see docs/operations/security.md.

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

The library to load: $TESSERO_LIB when set (read from the process environment only, never from
SAPI/request variables; it must be an absolute path to a file that neither it nor its directory is
group- or world-writable), else the bundled binary.

### mode

```php
static mode(): string
```

'scope' (preloaded), 'cdef' or 'unloaded'.

### path

```php
static path(): ?string
```

### phpPatched

```php
static phpPatched(int $versionId): bool
```

Security floor (owner decision 2026-10-24): only PHP releases that carry the 2026-10-21 security fixes,
8.2.35, 8.3.36, 8.4.27, 8.5.12 or newer on each branch; PHP 8.2 to 8.5 only (8.6 is a pre-release). Mirrors the extension's MINIT check and the
composer.json "php" constraints (tests/Unit/PhpSecurityTest.php keeps the three in step).

### platform

```php
static platform(): string
```

### trustedFile

```php
static trustedFile(string $path, string $var): string
```

A shared library named by an environment variable, checked before anything dlopen()s it: an absolute
path (a relative one resolves against the working directory, the web root under php-cgi), an existing
file, and on POSIX neither the file nor its directory writable by group or others (fileperms & 0022),
so no other account can swap the code PHP is about to run. Returns the resolved path.
