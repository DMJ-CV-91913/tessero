<?php

declare(strict_types=1);

namespace Tessero;

use ArrayAccess;
use Countable;
use FFI;
use FFI\CData;
use IteratorAggregate;
use JsonSerializable;
use Tessero\Exceptions\DTypeError;
use Tessero\Exceptions\IndexError;
use Tessero\Exceptions\ShapeError;
use Tessero\Exceptions\TesseroException;
use Tessero\Native\Abi;
use Tessero\Native\Blas;
use Tessero\Native\Buffer;
use Tessero\Native\Library;
use Traversable;

/**
 * An n-dimensional, homogeneously typed array in native memory.
 *
 * Layout is NumPy's: a Buffer, a byte offset, a shape and BYTE strides
 * (which may be zero for broadcast dimensions or negative for reversed
 * views). Slicing, transposing, reshaping a contiguous array, broadcasting
 * and expandDims/squeeze return views that share the Buffer; everything
 * that computes returns a new C-contiguous array. All loops run in
 * libtessero - PHP only computes shapes and strides.
 *
 * @implements ArrayAccess<int|string|array|NDArray, mixed>
 * @implements IteratorAggregate<int, mixed>
 */
final class NDArray extends Internal\OperandBase implements ArrayAccess, Countable, IteratorAggregate, JsonSerializable
{
    /** Elements packed per pack() call when importing PHP data (bounds the argument list). */
    private const CHUNK = 1 << 18;

    /** Elements unpacked per unpack() call when exporting (bounds the transient byte string to 32 MiB). */
    private const EXPORT_CHUNK = 1 << 22;

    /** @var list<int> */
    private array $shape;

    /** @var list<int> */
    private array $strides;

    private int $size;

    /**
     * @param list<int> $shape
     * @param list<int>|null $strides byte strides; null = C-contiguous
     */
    public function __construct(
        private Buffer $buffer,
        private DType $dtype,
        array $shape,
        ?array $strides = null,
        private int $offset = 0,
    ) {
        $this->shape = array_values($shape);
        $this->size = self::checkedSize($this->shape);
        $this->strides = $strides === null ? self::cStrides($this->shape, $dtype->itemsize()) : array_values($strides);
    }

    // ------------------------------------------------------------------ creation

    /** @param list<int>|int $shape */
    public static function empty(array|int $shape, DType|string $dtype = DType::Float64): self
    {
        $shape = self::shapeArg($shape);
        $dtype = DType::from_($dtype);

        return new self(Buffer::allocate(self::checkedSize($shape) * $dtype->itemsize()), $dtype, $shape);
    }

    /** @param list<int>|int $shape */
    public static function zeros(array|int $shape, DType|string $dtype = DType::Float64): self
    {
        $shape = self::shapeArg($shape);
        $dtype = DType::from_($dtype);

        return new self(Buffer::allocate(self::checkedSize($shape) * $dtype->itemsize(), true), $dtype, $shape);
    }

    /** @param list<int>|int $shape */
    public static function ones(array|int $shape, DType|string $dtype = DType::Float64): self
    {
        return self::full($shape, 1, $dtype);
    }

    /** @param list<int>|int $shape */
    public static function full(array|int $shape, int|float|bool $value, DType|string|null $dtype = null): self
    {
        $dtype = $dtype === null ? DType::ofValue($value) : DType::from_($dtype);
        $out = self::empty($shape, $dtype);
        $scalar = self::scalarBytes($value, $dtype);
        $tmp = Buffer::fromBytes($scalar);
        Library::check(Library::ffi()->tsr_fill($dtype->value, $out->size, $out->ptr(), $tmp->ptr()), 'full');

        return $out;
    }

    public static function arange(int|float $start, int|float|null $stop = null, int|float $step = 1, DType|string|null $dtype = null): self
    {
        if ($stop === null) {
            [$start, $stop] = [0, $start];
        }
        if ($step == 0) {
            throw new TesseroException('arange step must be non-zero.');
        }
        $n = max(0, (int) ceil(($stop - $start) / $step));
        $dtype = $dtype === null
            ? (is_int($start) && is_int($stop) && is_int($step) ? DType::Int64 : DType::Float64)
            : DType::from_($dtype);
        $out = self::empty([$n], $dtype);
        if ($n > 0) {
            Library::check(Library::ffi()->tsr_arange($dtype->value, $n, (float) $start, (float) $step, $out->ptr()), 'arange');
        }

        return $out;
    }

    public static function linspace(float $start, float $stop, int $num = 50, bool $endpoint = true): self
    {
        if ($num < 0) {
            throw new TesseroException('linspace num must be >= 0.');
        }
        $div = $endpoint ? $num - 1 : $num;
        $step = $div > 0 ? ($stop - $start) / $div : 0.0;
        $out = self::empty([$num], DType::Float64);
        if ($num > 0) {
            Library::ffi()->tsr_arange(0, $num, $start, $step, $out->ptr());
            if ($endpoint && $num > 1) {
                $out->buffer->write(($num - 1) * 8, pack('d', $stop));
            }
        }

        return $out;
    }

    public static function eye(int $n, ?int $m = null, int $k = 0, DType|string $dtype = DType::Float64): self
    {
        $m ??= $n;
        $out = self::zeros([$n, $m], $dtype);
        $one = self::scalarBytes(1, $out->dtype);
        for ($i = max(0, -$k); $i < $n && $i + $k < $m; $i++) {
            $out->buffer->write(($i * $m + $i + $k) * $out->dtype->itemsize(), $one);
        }

        return $out;
    }

    /**
     * Build an array from PHP data: a scalar, a (nested) list, or another NDArray (copied).
     * Passing $dtype skips type inference, which is the faster path for big inputs.
     */
    public static function array(mixed $data, DType|string|null $dtype = null): self
    {
        if ($data instanceof self) {
            return $data->astype($dtype === null ? $data->dtype : DType::from_($dtype), true);
        }
        if (! is_array($data)) {
            $dt = $dtype === null ? DType::ofValue($data) : DType::from_($dtype);

            return new self(Buffer::fromBytes(self::scalarBytes($data, $dt)), $dt, []);
        }

        if (self::holdsArrays($data)) {                   // a list of arrays, as numpy.array([a, b]) accepts
            $data = self::unwrapArrays($data);
        }
        $shape = [];
        $probe = $data;
        while (is_array($probe)) {
            $shape[] = count($probe);
            if ($probe === []) {
                break;
            }
            $probe = $probe[array_key_first($probe)];
        }
        $flat = array_values($data);
        for ($d = 1; $d < count($shape); $d++) {
            foreach ($flat as $row) {
                if (! is_array($row) || count($row) !== $shape[$d]) {
                    throw new ShapeError('Ragged nested array: every row must have the same length (expected ' . $shape[$d] . ').');
                }
            }
            $flat = $flat === [] ? [] : array_merge(...array_map(array_values(...), $flat));
        }

        $dt = $dtype === null ? self::inferDType($flat) : DType::from_($dtype);

        return self::fromFlat($flat, $shape, $dt);
    }

    /**
     * @param list<int|float|bool> $flat C-order values
     * @param list<int> $shape
     */
    public static function fromFlat(array $flat, array $shape, DType|string $dtype = DType::Float64): self
    {
        $dtype = DType::from_($dtype);
        $n = self::checkedSize(array_values($shape));
        if (count($flat) !== $n) {
            throw new ShapeError('Got ' . count($flat) . " values for shape (" . implode(', ', $shape) . ').');
        }
        $out = self::empty($shape, $dtype);
        $fmt = $dtype->packFormat() . '*';
        $isz = $dtype->itemsize();
        for ($start = 0; $start < $n; $start += self::CHUNK) {
            $chunk = array_slice($flat, $start, self::CHUNK);
            if ($dtype === DType::Complex128) {
                $pairs = [];
                foreach ($chunk as $v) {
                    $pairs[] = (float) $v;
                    $pairs[] = 0.0;
                }
                $chunk = $pairs;
            } elseif ($dtype === DType::Bool) {
                $chunk = array_map(static fn ($v): int => $v ? 1 : 0, $chunk);
            } elseif ($dtype->isInteger()) {
                $chunk = array_map(intval(...), $chunk);
            }
            $out->buffer->write($start * $isz, pack($fmt, ...$chunk));
        }

        return $out;
    }

    /** Complex array from real and imaginary parts (arrays or nested lists of equal shape). */
    public static function complex(mixed $real, mixed $imag = null): self
    {
        $re = self::asArray($real)->astype(DType::Float64);
        $im = $imag === null ? self::zeros($re->shape) : self::asArray($imag)->astype(DType::Float64);
        $shape = self::broadcastShapes($re->shape, $im->shape);
        $out = self::empty($shape, DType::Complex128);
        // view the complex buffer as float64 (..., 2) and fill the two planes
        $planes = new self($out->buffer, DType::Float64, [...$shape, 2]);
        $planes->slice(...array_fill(0, count($shape), ':'), ...[0])->assign($re);
        $planes->slice(...array_fill(0, count($shape), ':'), ...[1])->assign($im);

        return $out;
    }

    /** Wrap raw little-endian bytes (copied). */
    public static function fromBytes(string $bytes, DType|string $dtype, ?array $shape = null): self
    {
        $dtype = DType::from_($dtype);
        $n = intdiv(strlen($bytes), $dtype->itemsize());
        $shape ??= [$n];
        if (self::checkedSize(array_values($shape)) !== $n || $n * $dtype->itemsize() !== strlen($bytes)) {
            throw new ShapeError('Byte length does not match the shape and dtype.');
        }

        return new self(Buffer::fromBytes($bytes), $dtype, $shape);
    }

    public static function asArray(mixed $value, DType|string|null $dtype = null): self
    {
        if ($value instanceof self) {
            return $dtype === null ? $value : $value->astype(DType::from_($dtype));
        }

        return self::array($value, $dtype);
    }

    /** A clone owns a copy of the data (as in ext-tessero): writing it never changes the original, a memmap or a view. */
    public function __clone()
    {
        $c = $this->copy();
        $this->buffer = $c->buffer;
        $this->strides = $c->strides;
        $this->offset = 0;
    }

    // ------------------------------------------------------------------ files

    /**
     * An array backed by a memory-mapped file (numpy.memmap): nothing is read up front; the OS pages data
     * in on access. Modes: 'r' read-only, 'r+' read/write (extends a short file), 'w+' create or truncate,
     * 'c' copy-on-write. Without $shape the file (after $offset) is one 1-D array. Views share the mapping;
     * results of computations are ordinary arrays. Same arguments and errors as ext-tessero.
     *
     * @param list<int>|null $shape
     */
    public static function memmap(string $filename, string $mode = 'r+', ?array $shape = null, DType|string $dtype = DType::Float64, int $offset = 0): self
    {
        return Io\MemoryMap::memmap($filename, $mode, $shape === null ? null : array_values($shape), DType::from_($dtype), $offset);
    }

    /**
     * A memory-mapped .npy file (numpy.lib.format.open_memmap): 'w+' creates it with a header for $shape and
     * $dtype; 'r', 'r+' and 'c' read shape and dtype from the file.
     *
     * @param list<int>|null $shape
     */
    public static function openMemmap(string $path, string $mode = 'r+', ?array $shape = null, DType|string $dtype = DType::Float64, bool $fortranOrder = false): self
    {
        return Io\MemoryMap::openMemmap($path, $mode, $shape === null ? null : array_values($shape), DType::from_($dtype), $fortranOrder);
    }

    /** Read a .npy file (numpy.load); with $mmapMode ('r', 'r+', 'c') the data is memory-mapped instead of read. */
    public static function load(string $path, ?string $mmapMode = null): self
    {
        if ($mmapMode !== null) {
            if (! in_array($mmapMode, ['r', 'r+', 'c', 'readonly', 'readwrite', 'copyonwrite'], true)) {
                throw new \ValueError("NDArray::load(): Argument #2 (\$mmapMode) must be null, 'r', 'r+' or 'c' (openMemmap() creates files)");
            }

            return Io\MemoryMap::mapNpy($path, $mmapMode === 'r' || $mmapMode === 'readonly' ? 'r' : ($mmapMode === 'c' || $mmapMode === 'copyonwrite' ? 'c' : '+'));
        }

        return Io\Npy::load($path);
    }

    /** Write this array to a .npy file (numpy.save), C order. */
    public function save(string $path): void
    {
        Io\Npy::save($path, $this);
    }

    /** Write a memory map's dirty pages back to its file: scheduled (default) or synchronous with $sync. No-op for other arrays. */
    public function flush(bool $sync = false): self
    {
        $this->buffer->flush($sync);

        return $this;
    }

    /** True for memory-mapped arrays and views of them. */
    public function isMemmap(): bool
    {
        return $this->buffer->isMapped();
    }

    /** True for arrays over read-only memory (memmap mode 'r'). */
    public function isReadonly(): bool
    {
        return $this->buffer->isReadonly();
    }

    /** The mapped file's resolved path, or null. */
    public function filename(): ?string
    {
        return $this->buffer->mappedPath();
    }

    // ------------------------------------------------------------------ introspection

    /** @return list<int> */
    public function shape(): array
    {
        return $this->shape;
    }

    /** @return list<int> byte strides */
    public function strides(): array
    {
        return $this->strides;
    }

    public function dtype(): DType
    {
        return $this->dtype;
    }

    public function ndim(): int
    {
        return count($this->shape);
    }

    public function size(): int
    {
        return $this->size;
    }

    public function nbytes(): int
    {
        return $this->size * $this->dtype->itemsize();
    }

    public function offset(): int
    {
        return $this->offset;
    }

    /** The native block this array (or view) reads; shared by all views of it. Advanced use only. */
    public function buffer(): Buffer
    {
        return $this->buffer;
    }

    /**
     * char* to the first element, for passing to C in the same statement.
     *
     * The pointer does not keep the memory alive: keep this array in a variable until the C call returns,
     * and never store the pointer. Throws if the memory was already released (PHP shutdown ordering).
     */
    public function ptr(): CData
    {
        return $this->buffer->ptr($this->offset);
    }

    public function isContiguous(): bool
    {
        return $this->strides === self::cStrides($this->shape, $this->dtype->itemsize()) || $this->size <= 1
            || self::coalescedContiguous($this->shape, $this->strides, $this->dtype->itemsize());
    }

    public function isFortranContiguous(): bool
    {
        $expected = $this->dtype->itemsize();
        foreach ($this->shape as $d => $len) {
            if ($len !== 1 && $this->strides[$d] !== $expected) {
                return false;
            }
            $expected *= $len;
        }

        return true;
    }

    public function count(): int
    {
        if ($this->shape === []) {
            throw new TesseroException('len() of a 0-d array.');
        }

        return $this->shape[0];
    }

    // ------------------------------------------------------------------ export

    /** Nested PHP array (scalar for 0-d). Complex elements become [re, im] pairs. */
    public function toArray(): mixed
    {
        $flat = $this->toList();
        if ($this->shape === []) {
            return $flat[0];
        }
        if ($this->dtype === DType::Complex128) {
            $flat = array_chunk($flat, 2);
        }
        for ($d = count($this->shape) - 1; $d >= 1; $d--) {
            $flat = $this->shape[$d] === 0 ? array_fill(0, (int) array_product(array_slice($this->shape, 0, $d)), []) : array_chunk($flat, $this->shape[$d]);
        }

        return $flat;
    }

    /** Flat C-order list of values (complex: interleaved re, im). */
    public function toList(): array
    {
        $c = $this->contiguous();
        $fmt = $this->dtype->packFormat() . '*';
        $isz = $this->dtype->itemsize();
        $out = [];
        for ($start = 0; $start < $c->size; $start += self::EXPORT_CHUNK) {
            $count = min(self::EXPORT_CHUNK, $c->size - $start);
            $vals = unpack($fmt, $c->buffer->read($c->offset + $start * $isz, $count * $isz));
            if ($out === [] && $start + $count >= $c->size) {
                $out = array_values($vals);
            } else {
                array_push($out, ...array_values($vals));
            }
        }
        if ($this->dtype === DType::Bool) {
            $out = array_map(static fn (int $v): bool => $v !== 0, $out);
        }

        return $out;
    }

    /** Raw little-endian bytes in C order. */
    public function toBytes(): string
    {
        $c = $this->contiguous();

        return $c->buffer->read($c->offset, $c->nbytes());
    }

    /** One element as a PHP scalar ([re, im] for complex). */
    public function item(int ...$index): int|float|bool|array
    {
        if ($index === [] && $this->size !== 1) {
            throw new TesseroException('item() without an index needs a one-element array.');
        }
        $off = $this->offset;
        if ($index !== []) {
            if (count($index) !== count($this->shape)) {
                throw new IndexError('item() needs ' . count($this->shape) . ' indices.');
            }
            foreach ($index as $d => $i) {
                $off += self::normIndex($i, $this->shape[$d], $d) * $this->strides[$d];
            }
        }
        $bytes = $this->buffer->read($off, $this->dtype->itemsize());
        $vals = array_values(unpack($this->dtype->packFormat() . '*', $bytes));

        return match ($this->dtype) {
            DType::Complex128 => $vals,
            DType::Bool => $vals[0] !== 0,
            default => $vals[0],
        };
    }

    /**
     * JSON text written directly by libtessero from the strided memory (no PHP
     * arrays in between): about 10x faster than json_encode($a->toArray()) on
     * large arrays. Floats use the shortest round-trip form; NaN/Inf become null.
     */
    public function toJson(): string
    {
        $ffi = Library::ffi();
        $meta = $this->meta();
        $need = $ffi->tsr_array_json(FFI::addr($meta), null, 0);
        if ($need === 0) {
            return '';
        }
        $buf = Buffer::allocate($need);
        $ffi->tsr_array_json(FFI::addr($meta), $buf->ptr(), $need);

        return $buf->read(0, $need);
    }

    /**
     * C-side metadata (tsr_array) describing this view, for kernels that take whole arrays.
     * Like ptr(), the embedded data pointer is valid only while this array is alive.
     */
    public function meta(): CData
    {
        $ffi = Library::ffi();
        $m = $ffi->new('tsr_array');
        $m->data = $ffi->cast('void*', $this->buffer->ptr());
        $m->offset = $this->offset;
        $m->ndim = count($this->shape);
        $m->dtype = $this->dtype->value;
        foreach ($this->shape as $d => $len) {
            $m->shape[$d] = $len;
            $m->strides[$d] = $this->strides[$d];
        }

        return $m;
    }

    /**
     * Basic indexing with the spec parsed by libtessero (same grammar as offsetGet):
     * one call into C, no PHP string handling. Returns a view.
     */
    public function sliceNative(string $spec): self
    {
        $ffi = Library::ffi();
        $src = $this->meta();
        $out = $ffi->new('tsr_array');
        $rc = $ffi->tsr_array_slice(FFI::addr($src), $spec, FFI::addr($out));
        if ($rc === -1) {
            throw new IndexError("Invalid index '{$spec}'.");
        }
        Library::check($rc, "index '{$spec}'");
        $shape = [];
        $strides = [];
        for ($d = 0; $d < $out->ndim; $d++) {
            $shape[] = $out->shape[$d];
            $strides[] = $out->strides[$d];
        }

        return new self($this->buffer, $this->dtype, $shape, $strides, (int) $out->offset);
    }

    /** Queue/cache/session serialisation: dtype, shape and the raw little-endian bytes (C order). */
    public function __serialize(): array
    {
        return ['v' => 1, 'dtype' => $this->dtype->name(), 'shape' => $this->shape, 'data' => $this->toBytes()];
    }

    public function __unserialize(array $data): void
    {
        if (! isset($data['dtype'], $data['shape'], $data['data']) || ! is_string($data['data'])) {
            throw new TesseroException('Invalid serialized NDArray.');
        }
        if (! is_array($data['shape']) || ! array_is_list($data['shape'])) {
            throw new TesseroException('Invalid serialized NDArray.');
        }
        // serialized data may come from a cache or a queue an attacker can write: validate before trusting it
        $dtype = DType::from_($data['dtype']);
        $shape = array_map('intval', $data['shape']);
        if (self::checkedSize($shape) * $dtype->itemsize() !== strlen($data['data'])) {
            throw new TesseroException('Serialized NDArray data length mismatch.');
        }
        $this->buffer = Buffer::fromBytes($data['data']);
        $this->dtype = $dtype;
        $this->shape = $shape;
        $this->strides = self::cStrides($shape, $dtype->itemsize());
        $this->offset = 0;
        $this->size = self::checkedSize($shape);
    }

    public function jsonSerialize(): mixed
    {
        return $this->toArray();
    }

    public function __toString(): string
    {
        $head = 'array(';
        $body = $this->size > 1000
            ? self::formatSummary($this)
            : self::formatNested($this->toArray(), $this->dtype, strlen($head));

        return $head . $body . ', dtype=' . $this->dtype->name() . ')';
    }

    // ------------------------------------------------------------------ copies, casts, views

    public function copy(): self
    {
        $out = self::empty($this->shape, $this->dtype);
        $this->copyInto($out);

        return $out;
    }

    /** This array if already C-contiguous, else a contiguous copy. */
    public function contiguous(): self
    {
        return $this->isContiguous() ? $this : $this->copy();
    }

    public function astype(DType|string $dtype, bool $copy = false): self
    {
        $dtype = DType::from_($dtype);
        if ($dtype === $this->dtype && ! $copy) {
            return $this;
        }
        $out = self::empty($this->shape, $dtype);
        $this->copyInto($out);

        return $out;
    }

    /** Reinterpret the same bytes as another dtype of equal itemsize (bool <-> uint8). */
    public function view(DType|string $dtype): self
    {
        $dtype = DType::from_($dtype);
        if ($dtype->itemsize() !== $this->dtype->itemsize()) {
            throw new DTypeError('view() needs a dtype with the same itemsize.');
        }

        return new self($this->buffer, $dtype, $this->shape, $this->strides, $this->offset);
    }

    /** False for arrays over read-only memory (a memory map opened with mode 'r'). */
    public function isWritable(): bool
    {
        return ! $this->buffer->isReadonly();
    }

    private function requireWritable(string $what): void
    {
        if ($this->buffer->isReadonly()) {
            throw new TesseroException("{$what}: the array is read-only.");
        }
    }

    /** Copy (and cast) the values of $src, broadcast to this array's shape, into this array (which may be a view). */
    public function assign(mixed $src): self
    {
        $this->requireWritable('assign');
        if (! $src instanceof self) {
            $src = is_array($src) ? self::array($src) : new self(Buffer::fromBytes(self::scalarBytes($src, $this->dtype)), $this->dtype, []);
        }
        $strides = self::broadcastStrides($src, $this->shape);
        if ($this->size === 0) {
            return $this;
        }
        // overlapping source and destination (e.g. a[1:] = a[:-1]) must go through a temporary
        if ($src->buffer === $this->buffer) {
            $src = $src->copy();
            $strides = self::broadcastStrides($src, $this->shape);
        }
        Library::check(Library::ffi()->tsr_copy(
            $src->dtype->value, $this->dtype->value, count($this->shape), Library::i64($this->shape),
            $src->ptr(), Library::i64($strides), $this->ptr(), Library::i64($this->strides),
        ), 'assign');

        return $this;
    }

    private function copyInto(self $out): void
    {
        if ($this->size === 0) {
            return;
        }
        Library::check(Library::ffi()->tsr_copy(
            $this->dtype->value, $out->dtype->value, count($this->shape), Library::i64($this->shape),
            $this->ptr(), Library::i64($this->strides), $out->ptr(), Library::i64($out->strides),
        ), 'copy');
    }

    /** @param int|list<int> ...$shape one dimension may be -1 */
    public function reshape(int|array ...$shape): self
    {
        if (count($shape) === 1 && is_array($shape[0])) {
            $shape = $shape[0];
        }
        $shape = array_values($shape);
        $unknown = array_keys($shape, -1, true);
        if (count($unknown) > 1) {
            throw new ShapeError('Only one dimension can be -1.');
        }
        if ($unknown !== []) {
            $known = self::checkedSize(array_values(array_filter($shape, static fn (int $d): bool => $d !== -1)));
            if ($known === 0 || $this->size % $known !== 0) {
                throw new ShapeError("Cannot reshape {$this->size} elements into (" . implode(', ', $shape) . ').');
            }
            $shape[$unknown[0]] = intdiv($this->size, $known);
        }
        if (self::checkedSize($shape) !== $this->size) {
            throw new ShapeError("Cannot reshape {$this->size} elements into (" . implode(', ', $shape) . ').');
        }
        $src = $this->contiguous();

        return new self($src->buffer, $this->dtype, $shape, null, $src->offset);
    }

    public function ravel(): self
    {
        return $this->reshape([$this->size]);
    }

    public function flatten(): self
    {
        return $this->copy()->reshape([$this->size]);
    }

    /** @param list<int>|null $axes */
    public function transpose(?array $axes = null): self
    {
        $n = count($this->shape);
        $axes ??= range($n - 1, 0);
        if ($n === 0) {
            return $this;
        }
        $axes = array_map(fn (int $a): int => self::normAxis($a, $n), $axes);
        $sorted = $axes;
        sort($sorted);
        if ($sorted !== range(0, $n - 1)) {
            throw new ShapeError('transpose() axes must be a permutation of 0..' . ($n - 1) . '.');
        }
        $shape = [];
        $strides = [];
        foreach ($axes as $a) {
            $shape[] = $this->shape[$a];
            $strides[] = $this->strides[$a];
        }

        return new self($this->buffer, $this->dtype, $shape, $strides, $this->offset);
    }

    /** Matrix transpose (reverses all axes). */
    public function t(): self
    {
        return $this->transpose();
    }

    public function swapAxes(int $a, int $b): self
    {
        $axes = range(0, count($this->shape) - 1);
        $a = self::normAxis($a, count($this->shape));
        $b = self::normAxis($b, count($this->shape));
        [$axes[$a], $axes[$b]] = [$axes[$b], $axes[$a]];

        return $this->transpose($axes);
    }

    public function moveAxis(int $source, int $destination): self
    {
        $n = count($this->shape);
        $source = self::normAxis($source, $n);
        $destination = self::normAxis($destination, $n);
        $order = array_values(array_diff(range(0, $n - 1), [$source]));
        array_splice($order, $destination, 0, [$source]);

        return $this->transpose($order);
    }

    public function expandDims(int $axis): self
    {
        $n = count($this->shape) + 1;
        $axis = self::normAxis($axis, $n);
        $shape = $this->shape;
        $strides = $this->strides;
        array_splice($shape, $axis, 0, [1]);
        array_splice($strides, $axis, 0, [0]);

        return new self($this->buffer, $this->dtype, $shape, $strides, $this->offset);
    }

    public function squeeze(?int $axis = null): self
    {
        $shape = [];
        $strides = [];
        $target = $axis === null ? null : self::normAxis($axis, count($this->shape));
        if ($target !== null && $this->shape[$target] !== 1) {
            throw new ShapeError("Cannot squeeze axis {$axis} of length {$this->shape[$target]}.");
        }
        foreach ($this->shape as $d => $len) {
            if ($len === 1 && ($target === null || $target === $d)) {
                continue;
            }
            $shape[] = $len;
            $strides[] = $this->strides[$d];
        }

        return new self($this->buffer, $this->dtype, $shape, $strides, $this->offset);
    }

    /** @param list<int> $shape */
    public function broadcastTo(array $shape): self
    {
        if (self::broadcastShapes($this->shape, $shape) !== array_values($shape)) {
            throw new ShapeError('Cannot broadcast (' . implode(', ', $this->shape) . ') to (' . implode(', ', $shape) . ').');
        }

        return new self($this->buffer, $this->dtype, $shape, self::broadcastStrides($this, $shape), $this->offset);
    }

    // ------------------------------------------------------------------ indexing

    /**
     * Basic indexing, one spec per axis:
     *   int         select (drops the axis; negative counts from the end)
     *   'a:b:c'     slice, any part optional ('::-1' reverses, ':' keeps)
     *   '...'       fill the remaining axes with ':'
     *   null        insert a new axis of length 1 (numpy.newaxis)
     * Always returns a view.
     */
    public function slice(int|string|null ...$specs): self
    {
        $n = count($this->shape);
        $consumes = count(array_filter($specs, static fn ($s): bool => $s !== null && $s !== '...'));
        $expanded = [];
        $ellipsis = false;
        foreach ($specs as $s) {
            if ($s === '...') {
                if ($ellipsis) {
                    throw new IndexError('Only one ellipsis is allowed.');
                }
                $ellipsis = true;
                for ($k = 0; $k < $n - $consumes; $k++) {
                    $expanded[] = ':';
                }
            } else {
                $expanded[] = is_string($s) ? trim($s) : $s;
            }
        }
        if ($consumes > $n) {
            throw new IndexError("Too many indices: array is {$n}-dimensional, got {$consumes}.");
        }
        $shape = [];
        $strides = [];
        $offset = $this->offset;
        $axis = 0;
        foreach ($expanded as $s) {
            if ($s === null) {
                $shape[] = 1;
                $strides[] = 0;
                continue;
            }
            $len = $this->shape[$axis];
            $stride = $this->strides[$axis];
            if (is_int($s) || (is_string($s) && ! str_contains($s, ':') && preg_match('/^-?\d+$/', $s))) {
                $offset += self::normIndex((int) $s, $len, $axis) * $stride;
            } else {
                [$start, $count, $step] = self::parseSlice((string) $s, $len);
                $offset += $count > 0 ? $start * $stride : 0;
                $shape[] = $count;
                $strides[] = $count > 1 ? $stride * $step : $stride;       // |step| < len when count > 1: no overflow
            }
            $axis++;
        }
        for (; $axis < $n; $axis++) {
            $shape[] = $this->shape[$axis];
            $strides[] = $this->strides[$axis];
        }

        return new self($this->buffer, $this->dtype, $shape, $strides, $offset);
    }

    /** Element or sub-array at integer indices (scalar when all axes are indexed). */
    public function get(int ...$index): mixed
    {
        if (count($index) === count($this->shape)) {
            return $this->item(...$index);
        }

        return $this->slice(...$index);
    }

    /**
     * Gather along an axis (numpy.take). Indices may be negative. With $axis
     * null the array is flattened first.
     */
    public function take(mixed $indices, ?int $axis = null): self
    {
        $idx = self::asArray($indices, DType::Int64)->contiguous();
        $src = $axis === null ? $this->ravel() : $this->contiguous();
        $axis = $axis === null ? 0 : self::normAxis($axis, count($src->shape));
        $outer = (int) array_product(array_slice($src->shape, 0, $axis));
        $inner = (int) array_product(array_slice($src->shape, $axis + 1));
        $shape = [...array_slice($src->shape, 0, $axis), ...$idx->shape, ...array_slice($src->shape, $axis + 1)];
        $out = self::empty($shape, $this->dtype);
        if ($out->size > 0) {
            Library::check(Library::ffi()->tsr_take(
                $this->dtype->itemsize(), $outer, $src->shape[$axis], $inner, $src->ptr(), Library::ffi()->cast('int64_t*', $idx->ptr()), $idx->size, $out->ptr(),
            ), 'take');
        }

        return $out;
    }

    /** Scatter values along an axis in place (numpy.put_along_axis for whole slices). */
    public function put(mixed $indices, mixed $values, ?int $axis = null): self
    {
        $this->requireWritable('put');
        $idx = self::asArray($indices, DType::Int64)->contiguous();
        $target = $this->isContiguous() ? $this : $this->copy();
        $work = $axis === null ? $target->reshape([$target->size]) : $target;
        $axis = $axis === null ? 0 : self::normAxis($axis, count($work->shape));
        $outer = (int) array_product(array_slice($work->shape, 0, $axis));
        $inner = (int) array_product(array_slice($work->shape, $axis + 1));
        $vshape = [...array_slice($work->shape, 0, $axis), ...$idx->shape, ...array_slice($work->shape, $axis + 1)];
        $vals = self::asArray($values)->astype($this->dtype)->broadcastTo($vshape)->copy();
        if ($idx->size > 0) {
            Library::check(Library::ffi()->tsr_put(
                $this->dtype->itemsize(), $outer, $work->shape[$axis], $inner, $work->ptr(), Library::ffi()->cast('int64_t*', $idx->ptr()), $idx->size, $vals->ptr(),
            ), 'put');
        }
        if ($target !== $this) {
            $this->assign($target);
        }

        return $this;
    }

    /** Elements where $mask is true, as a 1-D array (a[mask]). */
    public function filter(self $mask): self
    {
        $m = self::boolMask($mask, $this->shape);
        $src = $this->contiguous();
        $ffi = Library::ffi();
        $mp = $ffi->cast('uint8_t*', $m->ptr());
        $n = $ffi->tsr_count_true($m->size, $mp);
        $out = self::empty([$n], $this->dtype);
        if ($n > 0) {
            $ffi->tsr_compress($this->dtype->itemsize(), $src->size, $src->ptr(), $mp, $out->ptr());
        }

        return $out;
    }

    /** a[mask] = values (a scalar, or one value per true element). */
    public function setWhere(self $mask, mixed $values): self
    {
        $this->requireWritable('setWhere');
        $m = self::boolMask($mask, $this->shape);
        $target = $this->isContiguous() ? $this : $this->copy();
        $vals = self::asArray($values)->astype($this->dtype)->contiguous();
        Library::check(Library::ffi()->tsr_mask_assign(
            $this->dtype->itemsize(), $target->size, $target->ptr(), Library::ffi()->cast('uint8_t*', $m->ptr()), $vals->ptr(), $vals->size,
        ), 'setWhere');
        if ($target !== $this) {
            $this->assign($target);
        }

        return $this;
    }

    /** Flat indices of non-zero elements (numpy.flatnonzero). */
    public function flatNonzero(): self
    {
        return self::arange($this->size)->filter($this->ravel()->ne(0));
    }

    public function offsetExists(mixed $offset): bool
    {
        try {
            $this->offsetGet($offset);

            return true;
        } catch (IndexError) {
            return false;
        }
    }

    /**
     * $a[2], $a[-1], $a['1:5'], $a['::2, 3'], $a['..., 0'], $a[$boolMask], $a[[0, 2, 5]], $a[$intArray]
     */
    public function offsetGet(mixed $offset): mixed
    {
        if ($offset instanceof self) {
            return $offset->dtype === DType::Bool ? $this->filter($offset) : $this->take($offset, 0);
        }
        if (is_array($offset)) {
            return $this->take($offset, 0);
        }
        if (is_int($offset)) {
            return count($this->shape) === 1 ? $this->item($offset) : $this->slice($offset);
        }
        if (is_string($offset)) {
            $specs = array_map(static function (string $p): int|string|null {
                $p = trim($p);

                return match (true) {
                    $p === 'None' || $p === 'newaxis' => null,
                    preg_match('/^-?\d+$/', $p) === 1 => (int) $p,
                    default => $p,
                };
            }, explode(',', $offset));
            $view = $this->slice(...$specs);

            return $view->shape === [] ? $view->item() : $view;
        }

        throw new IndexError('Unsupported index type ' . get_debug_type($offset) . '.');
    }

    public function offsetSet(mixed $offset, mixed $value): void
    {
        if ($offset === null) {
            throw new IndexError('Arrays have a fixed size; append is not supported.');
        }
        if ($offset instanceof self && $offset->dtype === DType::Bool) {
            $this->setWhere($offset, $value);

            return;
        }
        if ($offset instanceof self || is_array($offset)) {
            $this->put($offset, $value, 0);

            return;
        }
        $view = is_int($offset) ? $this->slice($offset) : $this->viewFor((string) $offset);
        $view->assign($value);
    }

    public function offsetUnset(mixed $offset): void
    {
        throw new IndexError('Array elements cannot be unset.');
    }

    public function getIterator(): Traversable
    {
        if ($this->shape === []) {
            throw new TesseroException('Iteration over a 0-d array.');
        }
        for ($i = 0; $i < $this->shape[0]; $i++) {
            yield $i => count($this->shape) === 1 ? $this->item($i) : $this->slice($i);
        }
    }

    private function viewFor(string $spec): self
    {
        $specs = array_map(static fn (string $p): int|string|null => preg_match('/^\s*-?\d+\s*$/', $p) === 1 ? (int) $p : (trim($p) === 'None' ? null : trim($p)), explode(',', $spec));

        return $this->slice(...$specs);
    }

    // ------------------------------------------------------------------ arithmetic

    public function add(mixed $b, ?self $out = null): self { return $this->binary(Abi::ADD, $b, $out); }
    public function sub(mixed $b, ?self $out = null): self { return $this->binary(Abi::SUB, $b, $out); }
    public function mul(mixed $b, ?self $out = null): self { return $this->binary(Abi::MUL, $b, $out); }
    public function div(mixed $b, ?self $out = null): self { return $this->binary(Abi::DIV, $b, $out); }
    public function pow(mixed $b, ?self $out = null): self { return $this->binary(Abi::POW, $b, $out); }
    public function mod(mixed $b, ?self $out = null): self { return $this->binary(Abi::MOD, $b, $out); }
    public function floorDiv(mixed $b, ?self $out = null): self { return $this->binary(Abi::FLOORDIV, $b, $out); }
    public function maximum(mixed $b, ?self $out = null): self { return $this->binary(Abi::MAX, $b, $out); }
    public function minimum(mixed $b, ?self $out = null): self { return $this->binary(Abi::MIN, $b, $out); }
    public function atan2(mixed $b, ?self $out = null): self { return $this->binary(Abi::ATAN2, $b, $out); }
    public function hypot(mixed $b, ?self $out = null): self { return $this->binary(Abi::HYPOT, $b, $out); }
    public function eq(mixed $b): self { return $this->binary(Abi::EQ, $b); }
    public function ne(mixed $b): self { return $this->binary(Abi::NE, $b); }
    public function lt(mixed $b): self { return $this->binary(Abi::LT, $b); }
    public function le(mixed $b): self { return $this->binary(Abi::LE, $b); }
    public function gt(mixed $b): self { return $this->binary(Abi::GT, $b); }
    public function ge(mixed $b): self { return $this->binary(Abi::GE, $b); }
    public function logicalAnd(mixed $b): self { return $this->binary(Abi::AND, $b); }
    public function logicalOr(mixed $b): self { return $this->binary(Abi::OR, $b); }
    public function logicalXor(mixed $b): self { return $this->binary(Abi::XOR, $b); }

    /** Reflected forms for scalar - array expressions. */
    public function rsub(mixed $a): self { return self::asScalarOrArray($a, $this)->binary(Abi::SUB, $this); }
    public function rdiv(mixed $a): self { return self::asScalarOrArray($a, $this)->binary(Abi::DIV, $this); }
    public function rpow(mixed $a): self { return self::asScalarOrArray($a, $this)->binary(Abi::POW, $this); }

    public function neg(?self $out = null): self { return $this->unary(Abi::NEG, $out); }
    public function abs(?self $out = null): self { return $this->dtype === DType::Complex128 ? $this->unary(Abi::CABS, $out) : $this->unary(Abi::ABS, $out); }
    public function square(?self $out = null): self { return $this->unary(Abi::SQUARE, $out); }
    public function sign(?self $out = null): self { return $this->unary(Abi::SIGN, $out); }
    public function sqrt(?self $out = null): self { return $this->unary(Abi::SQRT, $out); }
    public function exp(?self $out = null): self { return $this->unary(Abi::EXP, $out); }
    public function log(?self $out = null): self { return $this->unary(Abi::LOG, $out); }
    public function log10(?self $out = null): self { return $this->unary(Abi::LOG10, $out); }
    public function log2(?self $out = null): self { return $this->unary(Abi::LOG2, $out); }
    public function expm1(?self $out = null): self { return $this->unary(Abi::EXPM1, $out); }
    public function log1p(?self $out = null): self { return $this->unary(Abi::LOG1P, $out); }
    public function sin(?self $out = null): self { return $this->unary(Abi::SIN, $out); }
    public function cos(?self $out = null): self { return $this->unary(Abi::COS, $out); }
    public function tan(?self $out = null): self { return $this->unary(Abi::TAN, $out); }
    public function arcsin(?self $out = null): self { return $this->unary(Abi::ARCSIN, $out); }
    public function arccos(?self $out = null): self { return $this->unary(Abi::ARCCOS, $out); }
    public function arctan(?self $out = null): self { return $this->unary(Abi::ARCTAN, $out); }
    public function sinh(?self $out = null): self { return $this->unary(Abi::SINH, $out); }
    public function cosh(?self $out = null): self { return $this->unary(Abi::COSH, $out); }
    public function tanh(?self $out = null): self { return $this->unary(Abi::TANH, $out); }
    public function floor(?self $out = null): self { return $this->unary(Abi::FLOOR, $out); }
    public function ceil(?self $out = null): self { return $this->unary(Abi::CEIL, $out); }
    public function rint(?self $out = null): self { return $this->unary(Abi::RINT, $out); }
    public function reciprocal(?self $out = null): self { return $this->unary(Abi::RECIPROCAL, $out); }
    public function isnan(): self { return $this->unary(Abi::ISNAN); }
    public function isfinite(): self { return $this->unary(Abi::ISFINITE); }
    public function isinf(): self { return $this->unary(Abi::ISINF); }

    /** Bitwise NOT for integers, logical NOT for bool (numpy.invert). */
    public function invert(): self { return $this->unary(Abi::NOT); }

    public function logicalNot(): self
    {
        return $this->astype(DType::Bool)->unary(Abi::NOT);
    }

    public function real(): self
    {
        return $this->dtype === DType::Complex128 ? $this->unary(Abi::REAL) : $this->astype($this->dtype->toFloat(), true);
    }

    public function imag(): self
    {
        return $this->dtype === DType::Complex128 ? $this->unary(Abi::IMAG) : self::zeros($this->shape, $this->dtype->toFloat());
    }

    public function conj(): self
    {
        return $this->dtype === DType::Complex128 ? $this->unary(Abi::CONJ) : $this->copy();
    }

    public function angle(): self
    {
        return $this->dtype === DType::Complex128 ? $this->unary(Abi::ANGLE) : self::zeros($this->shape)->atan2(0)->add($this->lt(0)->astype(DType::Float64)->mul(M_PI));
    }

    public function round(int $decimals = 0): self
    {
        if ($decimals === 0) {
            return $this->rint();
        }
        $f = 10.0 ** $decimals;

        return $this->mul($f)->rint()->div($f);
    }

    public function clip(int|float|null $min = null, int|float|null $max = null): self
    {
        $r = $this;
        if ($min !== null) {
            $r = $r->maximum($min);
        }
        if ($max !== null) {
            $r = $r->minimum($max);
        }

        return $r === $this ? $this->copy() : $r;
    }

    private function binary(int $op, mixed $other, ?self $out = null): self
    {
        if ($other instanceof self) {
            $b = $other;
            $dt = DType::promote($this->dtype, $b->dtype);
        } elseif (is_int($other) || is_float($other) || is_bool($other)) {
            $dt = $this->dtype->withScalar($other);
            $b = new self(Buffer::fromBytes(self::scalarBytes($other, $dt)), $dt, []);
        } else {
            $b = self::array($other);
            $dt = DType::promote($this->dtype, $b->dtype);
        }

        $resultDt = $dt;
        $opDt = $dt;
        if (in_array($op, Abi::LOGICAL, true)) {
            $opDt = $resultDt = DType::Bool;
        } elseif (in_array($op, Abi::COMPARISONS, true)) {
            $resultDt = DType::Bool;
            if ($dt === DType::Bool) {
                $opDt = DType::UInt8;
            }
        } elseif ($op === Abi::DIV || $op === Abi::ATAN2 || $op === Abi::HYPOT) {
            $opDt = $resultDt = $dt->toFloat();
        } elseif ($op === Abi::POW && ($dt->isInteger() || $dt === DType::Bool)) {
            // computed in float64 and cast back (exact below 2^53, numpy keeps the int dtype)
            $opDt = DType::Float64;
            $resultDt = $dt === DType::Bool ? DType::Int64 : $dt;
        } elseif ($dt === DType::Bool) {
            $opDt = $resultDt = ($op === Abi::MAX || $op === Abi::MIN) ? DType::Bool : DType::Int64;
            if ($op === Abi::MAX || $op === Abi::MIN) {
                $opDt = DType::UInt8;
            }
        }

        $a = self::castForOp($this, $opDt);
        $b = self::castForOp($b, $opDt);
        $shape = self::broadcastShapes($a->shape, $b->shape);
        $computeDt = in_array($op, Abi::COMPARISONS, true) ? DType::Bool : ($opDt === DType::UInt8 && $resultDt === DType::Bool ? DType::Bool : $opDt);
        if ($out !== null) {
            $out->requireWritable('out');
        }
        $direct = $out !== null && $out->shape === $shape && $out->dtype === $computeDt
            && ! self::clobbers($out, $a, self::broadcastStrides($a, $shape)) && ! self::clobbers($out, $b, self::broadcastStrides($b, $shape));
        $target = $direct ? $out : self::empty($shape, $computeDt);
        if ($target->size > 0) {
            Library::check(Library::ffi()->tsr_binary(
                $op, $opDt->value, count($shape), Library::i64($shape),
                $a->ptr(), Library::i64(self::broadcastStrides($a, $shape)),
                $b->ptr(), Library::i64(self::broadcastStrides($b, $shape)),
                $target->ptr(), Library::i64($target->strides),
            ), self::opName($op));
        }
        if ($computeDt !== $resultDt) {
            $target = $target->astype($resultDt);
        }
        if ($out !== null && ! $direct) {
            return $out->assign($target);
        }

        return $target;
    }

    private function unary(int $op, ?self $out = null): self
    {
        $in = $this;
        $dt = $this->dtype;
        $outDt = $dt;
        if (in_array($op, Abi::FLOAT_UNARY, true)) {
            $dt = $outDt = $dt->toFloat();
            $in = $this->astype($dt);
        } elseif (in_array($op, Abi::BOOL_UNARY, true)) {
            $outDt = DType::Bool;
            if ($dt === DType::Bool) {
                $in = $this->view(DType::UInt8);
                $dt = DType::UInt8;
            }
        } elseif (in_array($op, [Abi::REAL, Abi::IMAG, Abi::CABS, Abi::ANGLE], true)) {
            $outDt = DType::Float64;
        } elseif ($dt === DType::Bool && $op !== Abi::NOT) {
            $dt = $outDt = DType::Int64;
            $in = $this->astype($dt);
        } elseif (in_array($op, [Abi::FLOOR, Abi::CEIL, Abi::RINT], true) && ! $dt->isFloat() && $dt !== DType::Complex128) {
            return $this->copy();
        }
        if ($out !== null) {
            $out->requireWritable('out');
        }
        $direct = $out !== null && $out->shape === $this->shape && $out->dtype === $outDt && ! self::clobbers($out, $in, $in->strides);
        $target = $direct ? $out : self::empty($this->shape, $outDt);
        if ($target->size > 0) {
            Library::check(Library::ffi()->tsr_unary(
                $op, $dt->value, count($this->shape), Library::i64($this->shape),
                $in->ptr(), Library::i64($in->strides), $target->ptr(), Library::i64($target->strides),
            ), self::opName($op, true));
        }
        if ($out !== null && ! $direct) {
            return $out->assign($target);
        }

        return $target;
    }

    // ------------------------------------------------------------------ reductions

    /** @param int|list<int>|null $axis */
    public function sum(int|array|null $axis = null, bool $keepdims = false): mixed { return $this->reduce(Abi::R_SUM, $axis, $keepdims); }
    /** @param int|list<int>|null $axis */
    public function prod(int|array|null $axis = null, bool $keepdims = false): mixed { return $this->reduce(Abi::R_PROD, $axis, $keepdims); }
    /** @param int|list<int>|null $axis */
    public function min(int|array|null $axis = null, bool $keepdims = false): mixed { return $this->reduce(Abi::R_MIN, $axis, $keepdims); }
    /** @param int|list<int>|null $axis */
    public function max(int|array|null $axis = null, bool $keepdims = false): mixed { return $this->reduce(Abi::R_MAX, $axis, $keepdims); }
    /** @param int|list<int>|null $axis */
    public function any(int|array|null $axis = null, bool $keepdims = false): mixed { return $this->reduce(Abi::R_ANY, $axis, $keepdims); }
    /** @param int|list<int>|null $axis */
    public function all(int|array|null $axis = null, bool $keepdims = false): mixed { return $this->reduce(Abi::R_ALL, $axis, $keepdims); }
    public function argmin(?int $axis = null, bool $keepdims = false): mixed { return $this->reduce(Abi::R_ARGMIN, $axis, $keepdims); }
    public function argmax(?int $axis = null, bool $keepdims = false): mixed { return $this->reduce(Abi::R_ARGMAX, $axis, $keepdims); }

    /** @param int|list<int>|null $axis */
    public function mean(int|array|null $axis = null, bool $keepdims = false): mixed
    {
        if ($this->dtype === DType::Complex128) {
            $n = $this->countAlong($axis);
            $s = $this->sum($axis, $keepdims);

            return $s instanceof self ? $s->div($n) : [$s[0] / $n, $s[1] / $n];
        }

        return $this->moments($axis, $keepdims, 0, false);
    }

    /** @param int|list<int>|null $axis */
    public function var(int|array|null $axis = null, int $ddof = 0, bool $keepdims = false): mixed
    {
        return $this->moments($axis, $keepdims, $ddof, true);
    }

    /** @param int|list<int>|null $axis */
    public function std(int|array|null $axis = null, int $ddof = 0, bool $keepdims = false): mixed
    {
        $v = $this->var($axis, $ddof, $keepdims);

        return $v instanceof self ? $v->sqrt() : sqrt($v);
    }

    public function cumsum(?int $axis = null): self
    {
        return $this->cumulative(0, $axis);
    }

    public function cumprod(?int $axis = null): self
    {
        return $this->cumulative(1, $axis);
    }

    /** n-th discrete difference along an axis (numpy.diff). */
    public function diff(int $n = 1, int $axis = -1): self
    {
        $r = $this;
        for ($i = 0; $i < $n; $i++) {
            $ax = self::normAxis($axis, count($r->shape));
            $hi = array_fill(0, count($r->shape), ':');
            $lo = $hi;
            $hi[$ax] = '1:';
            $lo[$ax] = ':-1';
            $r = $r->slice(...$hi)->sub($r->slice(...$lo));
        }

        return $r;
    }

    private function cumulative(int $op, ?int $axis): self
    {
        $src = $axis === null ? $this->ravel() : $this->contiguous();
        $axis = $axis === null ? 0 : self::normAxis($axis, count($src->shape));
        $dt = $src->dtype;
        $outDt = $dt->isFloat() ? $dt : ($dt === DType::Complex128 ? throw new DTypeError('cumsum/cumprod of complex arrays is not supported.') : DType::Int64);
        $out = self::empty($src->shape, $outDt);
        if ($out->size > 0) {
            Library::check(Library::ffi()->tsr_cumulative(
                $op, $dt->value, (int) array_product(array_slice($src->shape, 0, $axis)), $src->shape[$axis],
                (int) array_product(array_slice($src->shape, $axis + 1)), $src->ptr(), $out->ptr(),
            ), $op === 0 ? 'cumsum' : 'cumprod');
        }

        return $out;
    }

    /** @param int|list<int>|null $axis */
    private function reduce(int $op, int|array|null $axis, bool $keepdims): mixed
    {
        $src = $this->dtype === DType::Bool ? $this->view(DType::UInt8) : $this;
        if ($this->dtype === DType::Complex128) {
            if ($op !== Abi::R_SUM) {
                throw new DTypeError('Only sum() is supported for complex arrays (use abs()/real() first).');
            }
            // a complex array is a float64 array with a trailing axis of 2
            $planes = new self($this->buffer, DType::Float64, [...$this->shape, 2], [...$this->strides, 8], $this->offset);
            $axes = $axis === null ? range(0, count($this->shape) - 1) : (array) $axis;
            $r = $planes->reduce(Abi::R_SUM, array_map(fn (int $a): int => self::normAxis($a, count($this->shape)), $axes), $keepdims);
            $r = $r->contiguous();
            $res = new self($r->buffer, DType::Complex128, array_slice($r->shape, 0, -1), null, $r->offset);

            return $axis === null && ! $keepdims ? $res->item() : $res;
        }
        $n = count($this->shape);
        if ($axis === null) {
            $flat = $src->contiguous();
            $flat = new self($flat->buffer, $flat->dtype, [$flat->size], null, $flat->offset);
            $r = $flat->reduceAxis($op, 0);
            $r = $this->restoreBool($op, $r);
            if ($keepdims) {
                return $r->reshape(array_fill(0, $n, 1));
            }

            return $r->item();
        }
        $axes = array_map(fn (int $a): int => self::normAxis($a, $n), (array) $axis);
        if (count(array_unique($axes)) !== count($axes)) {
            throw new ShapeError('Duplicate axis in reduction.');
        }
        if (count($axes) > 1 && ($op === Abi::R_ARGMIN || $op === Abi::R_ARGMAX)) {
            throw new ShapeError('argmin/argmax take a single axis.');
        }
        rsort($axes);
        $r = $src;
        foreach ($axes as $a) {
            $r = $r->reduceAxis($op, $a);
            if ($op === Abi::R_SUM || $op === Abi::R_PROD) {
                // later passes reduce the int64/float partials
            }
        }
        $r = $this->restoreBool($op, $r);
        if ($keepdims) {
            $shape = $this->shape;
            foreach ($axes as $a) {
                $shape[$a] = 1;
            }
            $r = $r->reshape($shape);
        }

        return $r;
    }

    private function restoreBool(int $op, self $r): self
    {
        if ($this->dtype === DType::Bool && ($op === Abi::R_MIN || $op === Abi::R_MAX)) {
            return $r->view(DType::Bool);
        }

        return $r;
    }

    private function reduceAxis(int $op, int $axis): self
    {
        $dt = $this->dtype;
        $outDt = match ($op) {
            Abi::R_SUM, Abi::R_PROD => $dt->isFloat() ? $dt : DType::Int64,
            Abi::R_MIN, Abi::R_MAX => $dt,
            Abi::R_ARGMIN, Abi::R_ARGMAX => DType::Int64,
            default => DType::Bool,
        };
        $shape = $this->shape;
        $len = $shape[$axis];
        array_splice($shape, $axis, 1);
        if ($len === 0 && in_array($op, [Abi::R_MIN, Abi::R_MAX, Abi::R_ARGMIN, Abi::R_ARGMAX], true)) {
            throw new ShapeError('Zero-size array to a reduction operation that has no identity.');
        }
        $out = self::empty($shape, $outDt);
        if ($out->size === 0) {
            return $out;
        }
        $n = count($this->shape);
        if ($axis < $n - 1 && in_array($op, [Abi::R_SUM, Abi::R_PROD, Abi::R_MIN, Abi::R_MAX], true) && $this->isContiguous()) {
            // non-last axis of a contiguous array: combine whole rows (unit stride, vectorised)
            Library::check(Library::ffi()->tsr_reduce_mid(
                $op, $dt->value, (int) array_product(array_slice($this->shape, 0, $axis)), $len,
                (int) array_product(array_slice($this->shape, $axis + 1)), $this->ptr(), $out->ptr(),
            ), 'reduce');

            return $out;
        }
        Library::check(Library::ffi()->tsr_reduce(
            $op, $dt->value, count($this->shape), Library::i64($this->shape), $this->ptr(), Library::i64($this->strides),
            $axis, $outDt->value, $out->ptr(), Library::i64($out->strides),
        ), 'reduce');

        return $out;
    }

    /** @param int|list<int>|null $axis */
    private function moments(int|array|null $axis, bool $keepdims, int $ddof, bool $variance): mixed
    {
        $src = $this->dtype === DType::Bool ? $this->view(DType::UInt8) : $this;
        $n = count($this->shape);
        if ($axis === null) {
            $c = $src->contiguous();
            $src = new self($c->buffer, $c->dtype, [$c->size], null, $c->offset);
            $ax = 0;
        } elseif (is_array($axis) && count($axis) > 1) {
            // move the reduced axes to the end and merge them into one
            $axes = array_map(fn (int $a): int => self::normAxis($a, $n), $axis);
            sort($axes);                                // canonical order: the result must not depend on how axes are listed
            $keep = array_values(array_diff(range(0, $n - 1), $axes));
            $moved = $src->transpose([...$keep, ...$axes])->contiguous();
            $kept = array_map(fn (int $a): int => $this->shape[$a], $keep);
            $src = $moved->reshape([...$kept, -1]);
            $ax = count($kept);
        } else {
            $ax = self::normAxis(is_array($axis) ? $axis[0] : $axis, $n);
        }
        $len = $src->shape[$ax];
        $shape = $src->shape;
        array_splice($shape, $ax, 1);
        $mean = self::empty($shape, DType::Float64);
        $m2 = self::empty($shape, DType::Float64);
        if ($mean->size > 0) {
            Library::check(Library::ffi()->tsr_moments(
                $src->dtype->value, count($src->shape), Library::i64($src->shape), $src->ptr(), Library::i64($src->strides),
                $ax, Library::ffi()->cast('double*', $mean->ptr()), Library::i64($mean->strides),
                Library::ffi()->cast('double*', $m2->ptr()), Library::i64($m2->strides),
            ), 'moments');
        }
        $r = $variance ? $m2->div(max(0, $len - $ddof) ?: NAN) : $mean;
        if ($len === 0) {
            $r = self::full($shape, NAN);
        }
        if ($keepdims) {
            $kshape = $this->shape;
            foreach ($axis === null ? range(0, $n - 1) : (array) $axis as $a) {
                $kshape[self::normAxis($a, $n)] = 1;
            }

            return $r->reshape($kshape);
        }

        return $axis === null ? $r->item() : $r;
    }

    /** @param int|list<int>|null $axis */
    private function countAlong(int|array|null $axis): int
    {
        if ($axis === null) {
            return $this->size;
        }
        $n = 1;
        foreach ((array) $axis as $a) {
            $n *= $this->shape[self::normAxis($a, count($this->shape))];
        }

        return $n;
    }

    // ------------------------------------------------------------------ sorting

    public function sort(int $axis = -1): self
    {
        if ($this->dtype === DType::Complex128) {
            throw new DTypeError('Sorting complex arrays is not supported.');
        }
        if ($this->shape === []) {
            return $this->copy();
        }
        $axis = self::normAxis($axis, count($this->shape));
        $work = $this->moveAxis($axis, -1)->copy();
        $len = $work->shape[count($work->shape) - 1];
        $rows = $len > 0 ? intdiv($work->size, $len) : 0;
        $dt = $this->dtype === DType::Bool ? DType::UInt8 : $this->dtype;
        $ffi = Library::ffi();
        $rowBytes = $len * $dt->itemsize();
        for ($r = 0; $r < $rows; $r++) {
            Library::check($ffi->tsr_sort($dt->value, $len, $work->buffer->ptr($work->offset + $r * $rowBytes)), 'sort');
        }

        return $work->moveAxis(-1, $axis)->contiguous();
    }

    public function argsort(int $axis = -1): self
    {
        if ($this->dtype === DType::Complex128) {
            throw new DTypeError('Sorting complex arrays is not supported.');
        }
        $axis = self::normAxis($axis, max(1, count($this->shape)));
        $work = $this->moveAxis($axis, -1)->copy();
        $len = $work->shape[count($work->shape) - 1];
        $rows = $len > 0 ? intdiv($work->size, $len) : 0;
        $out = self::empty($work->shape, DType::Int64);
        $dt = $this->dtype === DType::Bool ? DType::UInt8 : $this->dtype;
        $ffi = Library::ffi();
        for ($r = 0; $r < $rows; $r++) {
            Library::check($ffi->tsr_argsort(
                $dt->value, $len, $work->buffer->ptr($work->offset + $r * $len * $dt->itemsize()),
                $ffi->cast('int64_t*', $out->buffer->ptr($r * $len * 8)),
            ), 'argsort');
        }

        return $out->moveAxis(-1, $axis)->contiguous();
    }

    // ------------------------------------------------------------------ products

    /**
     * Matrix product with NumPy's matmul rules: 1-D operands are promoted to
     * row/column vectors, >2-D operands are stacks of matrices with broadcast
     * batch dimensions. float64/float32 use BLAS when one is available.
     */
    public function matmul(self|array $other): mixed
    {
        $b = self::asArray($other);
        $a = $this;
        if ($a->shape === [] || $b->shape === []) {
            throw new ShapeError('matmul does not accept 0-d operands; use mul().');
        }
        $dt = DType::promote($a->dtype, $b->dtype);
        if ($dt === DType::Complex128) {
            throw new DTypeError('matmul of complex arrays is not supported yet.');
        }
        $opDt = match ($dt) {
            DType::Float64, DType::Float32, DType::Int64 => $dt,
            default => DType::Int64,
        };
        $a1 = count($a->shape) === 1;
        $b1 = count($b->shape) === 1;
        $A = $a1 ? $a->reshape([1, $a->shape[0]]) : $a;
        $B = $b1 ? $b->reshape([$b->shape[0], 1]) : $b;
        $m = $A->shape[count($A->shape) - 2];
        $k = $A->shape[count($A->shape) - 1];
        $k2 = $B->shape[count($B->shape) - 2];
        $n = $B->shape[count($B->shape) - 1];
        if ($k !== $k2) {
            throw new ShapeError("matmul: inner dimensions differ ({$m}x{$k} @ {$k2}x{$n}).");
        }
        $batch = self::broadcastShapes(array_slice($A->shape, 0, -2), array_slice($B->shape, 0, -2));
        $A = $A->astype($opDt)->broadcastTo([...$batch, $m, $k])->contiguous();
        $B = $B->astype($opDt)->broadcastTo([...$batch, $k, $n])->contiguous();
        $out = self::empty([...$batch, $m, $n], $opDt);
        $count = (int) array_product($batch);
        $isz = $opDt->itemsize();
        for ($i = 0; $i < $count; $i++) {
            self::gemm($opDt, $m, $n, $k,
                $A->buffer, $A->offset + $i * $m * $k * $isz,
                $B->buffer, $B->offset + $i * $k * $n * $isz,
                $out->buffer, $i * $m * $n * $isz);
        }
        if ($opDt !== $dt && $dt !== DType::Bool) {
            $out = $out->astype($dt);
        }
        $shape = $out->shape;
        if ($a1) {
            array_splice($shape, -2, 1);
        }
        if ($b1) {
            array_splice($shape, -1, 1);
        }
        if ($shape === []) {
            return $out->item();
        }

        return $shape === $out->shape ? $out : $out->reshape($shape);
    }

    /** numpy.dot for 1-D/2-D operands (alias of matmul there). */
    public function dot(self|array $other): mixed
    {
        return $this->matmul($other);
    }

    private static function gemm(DType $dt, int $m, int $n, int $k, Buffer $a, int $ao, Buffer $b, int $bo, Buffer $c, int $co): void
    {
        if ($m === 0 || $n === 0) {
            return;
        }
        if ($k === 0) {
            $zero = Buffer::allocate(16, true);
            Library::ffi()->tsr_fill($dt->value, $m * $n, $c->ptr($co), $zero->ptr());

            return;
        }
        if (($dt === DType::Float64 || $dt === DType::Float32) && Blas::available()) {
            Blas::gemm($dt, $m, $n, $k, $a->ptr($ao), $b->ptr($bo), $c->ptr($co));

            return;
        }
        Library::check(Library::ffi()->tsr_matmul($dt->value, $m, $n, $k, $a->ptr($ao), $k, $b->ptr($bo), $n, $c->ptr($co), $n), 'matmul');
    }

    // ------------------------------------------------------------------ helpers (static)

    /**
     * Element-wise selection: cond ? a : b, with broadcasting.
     */
    public static function where(mixed $cond, mixed $a, mixed $b): self
    {
        $c = self::asArray($cond)->astype(DType::Bool);
        $x = $a instanceof self ? $a : null;
        $y = $b instanceof self ? $b : null;
        $dt = match (true) {
            $x !== null && $y !== null => DType::promote($x->dtype, $y->dtype),
            $x !== null => is_array($b) ? DType::promote($x->dtype, self::array($b)->dtype) : $x->dtype->withScalar($b),
            $y !== null => is_array($a) ? DType::promote($y->dtype, self::array($a)->dtype) : $y->dtype->withScalar($a),
            default => DType::promote(self::asArray($a)->dtype, self::asArray($b)->dtype),
        };
        $x = self::asArray($a)->astype($dt);
        $y = self::asArray($b)->astype($dt);
        $shape = self::broadcastShapes($c->shape, $x->shape, $y->shape);
        $out = self::empty($shape, $dt);
        if ($out->size > 0) {
            Library::check(Library::ffi()->tsr_where(
                $dt->value, count($shape), Library::i64($shape),
                $c->ptr(), Library::i64(self::broadcastStrides($c, $shape)),
                $x->ptr(), Library::i64(self::broadcastStrides($x, $shape)),
                $y->ptr(), Library::i64(self::broadcastStrides($y, $shape)),
                $out->ptr(), Library::i64($out->strides),
            ), 'where');
        }

        return $out;
    }

    /** @param list<NDArray|array> $arrays */
    public static function concatenate(array $arrays, int $axis = 0): self
    {
        $arrays = array_map(static fn ($a): self => self::asArray($a), array_values($arrays));
        if ($arrays === []) {
            throw new ShapeError('concatenate needs at least one array.');
        }
        $first = $arrays[0];
        $n = count($first->shape);
        $axis = self::normAxis($axis, $n);
        $dt = $first->dtype;
        $total = 0;
        foreach ($arrays as $arr) {
            $dt = DType::promote($dt, $arr->dtype);
            $s1 = $arr->shape;
            $s0 = $first->shape;
            if (count($s1) !== $n) {
                throw new ShapeError('concatenate: arrays must have the same number of dimensions.');
            }
            $s1[$axis] = $s0[$axis] = 0;
            if ($s1 !== $s0) {
                throw new ShapeError('concatenate: shapes differ outside the concatenation axis.');
            }
            $total += $arr->shape[$axis];
        }
        $shape = $first->shape;
        $shape[$axis] = $total;
        $out = self::empty($shape, $dt);
        $pos = 0;
        foreach ($arrays as $arr) {
            $len = $arr->shape[$axis];
            if ($len > 0) {
                $spec = array_fill(0, $n, ':');
                $spec[$axis] = $pos . ':' . ($pos + $len);
                $out->slice(...$spec)->assign($arr);
            }
            $pos += $len;
        }

        return $out;
    }

    /** @param list<NDArray|array> $arrays */
    public static function stack(array $arrays, int $axis = 0): self
    {
        $arrays = array_map(static fn ($a): self => self::asArray($a), array_values($arrays));
        $n = count($arrays[0]->shape) + 1;
        $axis = self::normAxis($axis, $n);

        return self::concatenate(array_map(static fn (self $a): self => $a->expandDims($axis), $arrays), $axis);
    }

    public static function allclose(mixed $a, mixed $b, float $rtol = 1e-5, float $atol = 1e-8, bool $equalNan = false): bool
    {
        return (bool) self::isclose($a, $b, $rtol, $atol, $equalNan)->all();
    }

    public static function isclose(mixed $a, mixed $b, float $rtol = 1e-5, float $atol = 1e-8, bool $equalNan = false): self
    {
        $x = self::asArray($a)->astype(DType::Float64);
        $y = self::asArray($b)->astype(DType::Float64);
        $close = $x->sub($y)->abs()->le($y->abs()->mul($rtol)->add($atol));
        $sameInf = $x->eq($y);
        $r = $close->logicalOr($sameInf);
        if ($equalNan) {
            $r = $r->logicalOr($x->isnan()->logicalAnd($y->isnan()));
        }

        return $r;
    }

    /**
     * Element count of a shape. Refuses negative dimensions, more than 32 dimensions, and counts whose byte
     * size could overflow (array_product silently turns into a float there, and a wrapped size would let a
     * small buffer masquerade as a huge array).
     *
     * @param list<int> $shape
     */
    private static function checkedSize(array $shape): int
    {
        if (count($shape) > 32) {
            throw new ShapeError('Arrays are limited to 32 dimensions.');
        }
        $n = 1;
        foreach ($shape as $len) {
            if (! is_int($len) || $len < 0) {
                throw new ShapeError('Dimensions must be non-negative integers.');
            }
            if ($len !== 0 && $n > intdiv(PHP_INT_MAX >> 4, $len)) {
                throw new \Tessero\Exceptions\MemoryError('The requested shape is too large (element count overflows).');
            }
            $n *= $len;
        }

        return $n;
    }

    /** @param list<int> ...$shapes @return list<int> */
    public static function broadcastShapes(array ...$shapes): array
    {
        $nd = max(array_map('count', $shapes) ?: [0]);
        $out = array_fill(0, $nd, 1);
        foreach ($shapes as $s) {
            $pad = $nd - count($s);
            foreach ($s as $i => $len) {
                $cur = $out[$pad + $i];
                if ($len === $cur || $len === 1) {
                    continue;
                }
                if ($cur === 1) {
                    $out[$pad + $i] = $len;
                    continue;
                }
                throw new ShapeError('Shapes ' . implode(' and ', array_map(static fn (array $x): string => '(' . implode(', ', $x) . ')', $shapes)) . ' cannot be broadcast together.');
            }
        }

        return $out;
    }

    /** @param list<int> $shape @return list<int> */
    public static function cStrides(array $shape, int $itemsize): array
    {
        $strides = [];
        $acc = $itemsize;
        for ($d = count($shape) - 1; $d >= 0; $d--) {
            $strides[$d] = $acc;
            $acc *= max(1, $shape[$d]);
        }
        ksort($strides);

        return array_values($strides);
    }

    /** @param list<int> $shape @return list<int> */
    private static function broadcastStrides(self $a, array $shape): array
    {
        $pad = count($shape) - count($a->shape);
        if ($pad < 0) {
            throw new ShapeError('Cannot broadcast to fewer dimensions.');
        }
        $out = array_fill(0, count($shape), 0);
        foreach ($a->shape as $i => $len) {
            if ($len === $shape[$pad + $i]) {
                $out[$pad + $i] = $a->strides[$i];
            } elseif ($len !== 1) {
                throw new ShapeError('Cannot broadcast (' . implode(', ', $a->shape) . ') to (' . implode(', ', $shape) . ').');
            }
        }

        return $out;
    }

    /**
     * Would writing $out (element by element) overwrite elements of $in, read with byte strides
     * $inStrides over $out's shape, before they are read? Same memory with the identical layout
     * is fine (in place); any other layout over the same buffer goes through a temporary.
     *
     * @param list<int> $inStrides
     */
    private static function clobbers(self $out, self $in, array $inStrides): bool
    {
        if ($in->buffer !== $out->buffer) {
            return false;
        }
        if ($in->offset !== $out->offset) {
            return true;
        }
        foreach ($inStrides as $d => $s) {
            if ($s !== $out->strides[$d] && $out->shape[$d] > 1) {
                return true;
            }
        }

        return false;
    }

    private static function coalescedContiguous(array $shape, array $strides, int $itemsize): bool
    {
        $expected = $itemsize;
        for ($d = count($shape) - 1; $d >= 0; $d--) {
            if ($shape[$d] === 1) {
                continue;
            }
            if ($strides[$d] !== $expected) {
                return false;
            }
            $expected *= $shape[$d];
        }

        return true;
    }

    private static function castForOp(self $a, DType $dt): self
    {
        if ($a->dtype === $dt) {
            return $a;
        }
        if ($a->dtype->itemsize() === $dt->itemsize() && (($a->dtype === DType::Bool && $dt === DType::UInt8) || ($a->dtype === DType::UInt8 && $dt === DType::Bool))) {
            return $a->view($dt);
        }

        return $a->astype($dt);
    }

    private static function asScalarOrArray(mixed $v, self $like): self
    {
        if ($v instanceof self) {
            return $v;
        }
        if (is_int($v) || is_float($v) || is_bool($v)) {
            $dt = $like->dtype->withScalar($v);

            return new self(Buffer::fromBytes(self::scalarBytes($v, $dt)), $dt, []);
        }

        return self::array($v);
    }

    private static function boolMask(self $mask, array $shape): self
    {
        if ($mask->shape !== $shape) {
            throw new ShapeError('Boolean mask shape (' . implode(', ', $mask->shape) . ') does not match the array (' . implode(', ', $shape) . ').');
        }

        return $mask->astype(DType::Bool)->contiguous();
    }

    /** @param array<mixed> $data */
    private static function holdsArrays(array $data): bool
    {
        foreach ($data as $v) {
            if ($v instanceof self) {
                return true;
            }
        }
        $first = $data === [] ? null : $data[array_key_first($data)];

        return is_array($first) && self::holdsArrays($first);   // rows of arrays: checked along the first row
    }

    private static function unwrapArrays(mixed $v): mixed
    {
        if ($v instanceof self) {
            return $v->toArray();
        }

        return is_array($v) ? array_map(self::unwrapArrays(...), array_values($v)) : $v;
    }

    /** @param list<mixed> $flat */
    private static function inferDType(array $flat): DType
    {
        $allBool = true;
        foreach ($flat as $v) {
            if (is_float($v)) {
                return DType::Float64;
            }
            if (! is_bool($v)) {
                $allBool = false;
                if (! is_int($v)) {
                    throw new DTypeError('Arrays hold numbers or booleans, got ' . get_debug_type($v) . '.');
                }
            }
        }

        return $allBool && $flat !== [] ? DType::Bool : ($flat === [] ? DType::Float64 : DType::Int64);
    }

    private static function scalarBytes(int|float|bool $v, DType $dt): string
    {
        return match ($dt) {
            DType::Complex128 => pack('dd', (float) $v, 0.0),
            DType::Bool => pack('C', $v ? 1 : 0),
            DType::Float64 => pack('d', (float) $v),
            DType::Float32 => pack('g', (float) $v),
            default => pack($dt->packFormat(), (int) $v),
        };
    }

    /** @param list<int>|int $shape @return list<int> */
    private static function shapeArg(array|int $shape): array
    {
        $shape = is_int($shape) ? [$shape] : array_values($shape);
        foreach ($shape as $d) {
            if (! is_int($d) || $d < 0) {
                throw new ShapeError('Dimensions must be non-negative integers.');
            }
        }

        return $shape;
    }

    public static function normAxis(int $axis, int $ndim): int
    {
        $a = $axis < 0 ? $axis + $ndim : $axis;
        if ($a < 0 || $a >= $ndim) {
            throw new ShapeError("Axis {$axis} is out of bounds for an array of dimension {$ndim}.");
        }

        return $a;
    }

    private static function normIndex(int $i, int $len, int $axis): int
    {
        $j = $i < 0 ? $i + $len : $i;
        if ($j < 0 || $j >= $len) {
            throw new IndexError("Index {$i} is out of bounds for axis {$axis} with size {$len}.");
        }

        return $j;
    }

    /** @return array{int, int, int} start, count, step (Python slice semantics) */
    private static function parseSlice(string $spec, int $len): array
    {
        $parts = explode(':', $spec);
        if (count($parts) > 3) {
            throw new IndexError("Invalid slice '{$spec}'.");
        }
        foreach ($parts as $p) {
            if (trim($p) !== '' && preg_match('/^\s*-?\d+\s*$/', $p) !== 1) {
                throw new IndexError("Invalid slice '{$spec}'.");
            }
        }
        $step = isset($parts[2]) && trim($parts[2]) !== '' ? (int) $parts[2] : 1;
        if ($step === 0) {
            throw new IndexError('Slice step cannot be zero.');
        }
        $startRaw = trim($parts[0]);
        $stopRaw = isset($parts[1]) ? trim($parts[1]) : '';
        if ($step > 0) {
            $start = $startRaw === '' ? 0 : (int) $startRaw;
            $stop = $stopRaw === '' ? $len : (int) $stopRaw;
            $start = $start < 0 ? max(0, $start + $len) : min($start, $len);
            $stop = $stop < 0 ? max(0, $stop + $len) : min($stop, $len);
            $count = $stop > $start ? 1 + intdiv($stop - $start - 1, $step) : 0;       // no overflow for huge steps
        } else {
            $start = $startRaw === '' ? $len - 1 : (int) $startRaw;
            $stop = $stopRaw === '' ? -1 : (int) $stopRaw;
            $start = $start < 0 ? max(-1, $start + $len) : min($start, $len - 1);
            if ($stopRaw !== '') {
                $stop = $stop < 0 ? max(-1, $stop + $len) : min($stop, $len - 1);
            }
            $count = $start > $stop ? 1 + intdiv($start - $stop - 1, -$step) : 0;
        }

        return [$start, $count, $step];
    }

    private static function opName(int $op, bool $unary = false): string
    {
        static $bin = null, $un = null;
        if ($bin === null) {
            $ref = new \ReflectionClass(Abi::class);
            foreach ($ref->getConstants() as $name => $v) {
                if (is_int($v)) {
                    // first 20 names are binary ops, next are unary; names overlap in values, so keep two maps
                    if (in_array($name, ['ADD', 'SUB', 'MUL', 'DIV', 'POW', 'MAX', 'MIN', 'ATAN2', 'HYPOT', 'MOD', 'EQ', 'NE', 'LT', 'LE', 'GT', 'GE', 'AND', 'OR', 'XOR', 'FLOORDIV'], true)) {
                        $bin[$v] = strtolower($name);
                    } elseif (! str_starts_with($name, 'R_')) {
                        $un[$v] = strtolower($name);
                    }
                }
            }
        }

        return ($unary ? $un[$op] ?? 'unary' : $bin[$op] ?? 'binary') . ' (' . ($unary ? 'unary' : 'binary') . ')';
    }

    private static function formatSummary(self $a): string
    {
        $flat = $a->ravel();
        $head = $flat->slice('0:3')->toList();
        $tail = $flat->slice('-3:')->toList();
        $fmt = static fn (array $xs): string => implode(', ', array_map(static fn ($v): string => is_float($v) ? self::fmtFloat($v) : var_export($v, true), $xs));

        return '[' . $fmt($head) . ', ..., ' . $fmt($tail) . '], shape=(' . implode(', ', $a->shape) . ')';
    }

    private static function formatNested(mixed $v, DType $dt, int $indent): string
    {
        if (! is_array($v)) {
            return is_float($v) ? self::fmtFloat($v) : var_export($v, true);
        }
        if ($dt === DType::Complex128 && count($v) === 2 && ! is_array($v[0])) {
            return self::fmtFloat($v[0]) . ($v[1] < 0 ? '-' : '+') . self::fmtFloat(abs($v[1])) . 'j';
        }
        $parts = array_map(static fn ($x): string => self::formatNested($x, $dt, $indent + 1), $v);
        $nested = $v !== [] && is_array($v[0]) && ! ($dt === DType::Complex128 && ! is_array($v[0][0] ?? null));

        return '[' . implode($nested ? ",\n" . str_repeat(' ', $indent + 1) : ', ', $parts) . ']';
    }

    private static function fmtFloat(float $f): string
    {
        if (is_nan($f)) {
            return 'nan';
        }
        if (is_infinite($f)) {
            return $f > 0 ? 'inf' : '-inf';
        }
        $s = sprintf('%.8g', $f);
        if (str_contains($s, 'e')) {
            return $s;
        }
        if (str_contains($s, '.')) {
            return rtrim(rtrim($s, '0'), '.') . (str_ends_with(rtrim($s, '0'), '.') ? '.' : '');
        }
        $s .= '.';

        return $s;
    }
}
