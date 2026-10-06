<?php

declare(strict_types=1);

namespace Tessero\Native;

use FFI;
use FFI\CData;
use Tessero\Exceptions\MemoryError;
use Tessero\Exceptions\TesseroException;

/**
 * A block of native memory owned by exactly one PHP object.
 *
 * Every NDArray (including views) holds a reference to the Buffer it reads,
 * so the memory lives exactly as long as some array can reach it and is
 * returned with tsr_free() in __destruct - never twice, never early.
 * Allocations are 64-byte aligned, not zeroed (unless asked), and counted
 * against the process-wide budget set with Tessero::setMemoryBudget(), which
 * is the native counterpart of memory_limit.
 */
final class Buffer
{
    private bool $released = false;

    private bool $readonly = false;

    /** tsr_mmap handle address when this buffer is a file mapping */
    private int $mapHandle = 0;

    private ?string $mapPath = null;

    private string $mapMode = '';

    private function __construct(
        public readonly int $address,
        public readonly int $bytes,
        private readonly bool $owned,
        private readonly mixed $keepAlive = null,
    ) {
    }

    public static function allocate(int $bytes, bool $zero = false): self
    {
        if ($bytes < 0) {
            throw new MemoryError('Could not allocate a negative number of bytes (size overflow).');
        }
        $bytes = max(64, $bytes);
        $ffi = Library::ffi();
        $raw = $zero ? $ffi->tsr_calloc($bytes) : $ffi->tsr_alloc($bytes);
        if ($raw === null || FFI::isNull($raw)) {
            throw new MemoryError(sprintf(
                'Could not allocate %s of native memory (in use: %s, budget: %s).',
                self::human($bytes),
                self::human($ffi->tsr_allocated()),
                $ffi->tsr_budget() > 0 ? self::human($ffi->tsr_budget()) : 'unlimited',
            ));
        }

        return new self(Library::address($raw), $bytes, true);
    }

    /**
     * Take ownership of a tsr_alloc() block the kernel returned (routine results): freed with tsr_free($bytes).
     *
     * @internal
     */
    public static function adopt(int $address, int $bytes): self
    {
        return new self($address, $bytes, true);
    }

    /** Wrap memory owned by someone else (e.g. an FFI array); $keepAlive keeps it reachable. */
    public static function wrap(int $address, int $bytes, mixed $keepAlive): self
    {
        return new self($address, $bytes, false, $keepAlive);
    }

    /**
     * A file mapping from tsr_mmap_open(): $address is element 0, $handle is unmapped by tsr_mmap_close()
     * when the last array over it is gone. Mapped bytes are outside the memory budget.
     *
     * @internal use NDArray::memmap()
     */
    public static function mapped(int $address, int $bytes, int $handle, string $path, string $mode): self
    {
        $b = new self($address, $bytes, false);
        $b->mapHandle = $handle;
        $b->mapPath = $path;
        $b->mapMode = $mode;
        $b->readonly = $mode === 'r';

        return $b;
    }

    public function isMapped(): bool
    {
        return $this->mapHandle !== 0;
    }

    /** Resolved path of a file mapping, else null. */
    public function mappedPath(): ?string
    {
        return $this->mapPath;
    }

    /** Write dirty pages of a mapping back to the file (no-op for heap memory and modes 'r'/'c'). */
    public function flush(bool $sync = false): void
    {
        if ($this->mapHandle === 0 || $this->released) {
            return;
        }
        $ffi = Library::ffi();
        if ($ffi->tsr_mmap_flush(Library::ptr($this->mapHandle), $sync ? 1 : 0) !== 0) {
            throw new TesseroException('memmap: ' . $ffi->tsr_mmap_error() . " ({$this->mapPath})");
        }
    }

    public static function fromBytes(string $bytes): self
    {
        $buffer = self::allocate(strlen($bytes));
        $buffer->write(0, $bytes);

        return $buffer;
    }

    /** char* to byte $offset. Keep this Buffer referenced while the pointer is in use. */
    public function ptr(int $offset = 0): CData
    {
        if ($this->released) {
            throw new TesseroException('Buffer used after release.');
        }

        return Library::ptr($this->address + $offset);
    }

    /** Read-only memory (a memory map opened with mode 'r'): arrays over it refuse writes. */
    public function isReadonly(): bool
    {
        return $this->readonly;
    }

    /** Mark the memory read-only (irreversible). */
    public function freeze(): void
    {
        $this->readonly = true;
    }

    public function write(int $offset, string $bytes): void
    {
        if ($this->readonly) {
            throw new TesseroException('The array is read-only.');
        }
        $len = strlen($bytes);
        if ($len === 0) {
            return;
        }
        if ($offset < 0 || $offset + $len > $this->bytes) {
            throw new TesseroException("Write of {$len} bytes at {$offset} overruns a {$this->bytes}-byte buffer.");
        }
        FFI::memcpy($this->ptr($offset), $bytes, $len);
    }

    public function read(int $offset, int $length): string
    {
        if ($length <= 0) {
            return '';
        }
        if ($offset < 0 || $offset + $length > $this->bytes) {
            throw new TesseroException("Read of {$length} bytes at {$offset} overruns a {$this->bytes}-byte buffer.");
        }

        return FFI::string($this->ptr($offset), $length);
    }

    public function __destruct()
    {
        if ($this->mapHandle !== 0 && ! $this->released) {
            $this->released = true;
            $handle = $this->mapHandle;
            $this->mapHandle = 0;                     // flush() after this point is a no-op, never a use-after-free
            Library::ffi()->tsr_mmap_close(Library::ptr($handle));

            return;
        }
        if ($this->owned && ! $this->released) {
            $this->released = true;
            Library::ffi()->tsr_free(Library::ptr($this->address), $this->bytes);
        }
    }

    /** Buffers are never cloned: two owners would free the same block. Copy the array instead. */
    private function __clone()
    {
    }

    private static function human(int $bytes): string
    {
        $units = ['B', 'KiB', 'MiB', 'GiB', 'TiB'];
        $i = 0;
        $v = (float) $bytes;
        while ($v >= 1024 && $i < 4) {
            $v /= 1024;
            $i++;
        }

        return sprintf('%.1f %s', $v, $units[$i]);
    }
}
