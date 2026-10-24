# Buffer

`Tessero\Native\Buffer`

A block of native memory owned by exactly one PHP object.

Every NDArray (including views) holds a reference to the Buffer it reads,
so the memory lives exactly as long as some array can reach it and is
returned with tsr_free() in __destruct - never twice, never early.
Allocations are 64-byte aligned, not zeroed (unless asked), and counted
against the process-wide budget set with Tessero::setMemoryBudget(), which
is the native counterpart of memory_limit.

## Methods

### adopter

```php
static adopter(): Closure
```

A closure that takes ownership of a tsr_alloc() block (freed with tsr_free($bytes)), handed out to
Tessero\Native\Registry only. PHP has no friend classes, so the caller is checked once here, not on
every call; Registry caches the closure.

### allocate

```php
static allocate(int $bytes, bool $zero = false): self
```

### fromBytes

```php
static fromBytes(string $bytes): self
```

### mapped

```php
static mapped(int $address, int $bytes, int $handle, string $path, string $mode): self
```

A file mapping from tsr_mmap_open(): $address is element 0, $handle is unmapped by tsr_mmap_close()
when the last array over it is gone. Mapped bytes are outside the memory budget.

### __serialize

```php
__serialize(): array
```

Buffers are never serialized. unserialize() creates objects without the constructor, so a forged payload
could otherwise name any address and size, and __destruct would free (or unmap) it. Serialize the NDArray
instead: its __serialize copies the data and its __unserialize validates it.

### __unserialize

```php
__unserialize(array $data): void
```

### flush

```php
flush(bool $sync = false): void
```

Write dirty pages of a mapping back to the file (no-op for heap memory and modes 'r'/'c').

### freeze

```php
freeze(): void
```

Mark the memory read-only (irreversible).

### isMapped

```php
isMapped(): bool
```

### isReadonly

```php
isReadonly(): bool
```

Read-only memory (a memory map opened with mode 'r'): arrays over it refuse writes.

### mappedPath

```php
mappedPath(): ?string
```

Resolved path of a file mapping, else null.

### ptr

```php
ptr(int $offset = 0): FFI\CData
```

char* to byte $offset. Keep this Buffer referenced while the pointer is in use.

### read

```php
read(int $offset, int $length): string
```

### write

```php
write(int $offset, string $bytes): void
```
