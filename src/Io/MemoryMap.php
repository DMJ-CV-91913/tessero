<?php

declare(strict_types=1);

namespace Tessero\Io;

use FFI;
use Tessero\DType;
use Tessero\Exceptions\DTypeError;
use Tessero\Exceptions\ShapeError;
use Tessero\Exceptions\TesseroException;
use Tessero\NDArray;
use Tessero\Native\Buffer;
use Tessero\Native\Library;

/**
 * Memory-mapped arrays for the FFI package: the same libtessero code
 * (tsr_mmap_*, tsr_npy_*) as ext-tessero's NDArray::memmap(), so the two
 * backends accept the same files and arguments and fail the same way.
 *
 * Use it through NDArray::memmap(), NDArray::openMemmap() and
 * NDArray::load($path, $mmapMode).
 *
 * @internal
 */
final class MemoryMap
{
    private const MODES = ['r' => 'r', 'readonly' => 'r', 'r+' => '+', 'readwrite' => '+', 'w+' => 'w', 'write' => 'w',
        'c' => 'c', 'copyonwrite' => 'c'];

    private const KERNEL_MODE = ['r' => 0, '+' => 1, 'w' => 2, 'c' => 3];

    /** @param list<int>|null $shape */
    public static function memmap(string $filename, string $mode, ?array $shape, DType $dtype, int $offset): NDArray
    {
        $m = self::MODES[$mode] ?? throw new \ValueError("NDArray::memmap(): Argument #2 (\$mode) must be one of 'r', 'r+', 'w+' or 'c'");
        if ($offset < 0 || $offset % $dtype->itemsize() !== 0) {
            throw new \ValueError("NDArray::memmap(): Argument #5 (\$offset) must be a non-negative multiple of the item size ({$dtype->itemsize()} bytes)");
        }
        if ($shape !== null) {
            self::checkShape($shape, 3);
        } elseif ($m === 'w') {
            throw new \ValueError("NDArray::memmap(): Argument #3 (\$shape) is required with mode 'w+'");
        }
        $path = self::resolve($filename);

        return self::map($path, $m, $dtype, $shape, $offset, false);
    }

    /** numpy.lib.format.open_memmap: 'w+' writes a header for $shape and $dtype; other modes read it. */
    public static function openMemmap(string $filename, string $mode, ?array $shape, DType $dtype, bool $fortranOrder): NDArray
    {
        $m = self::MODES[$mode] ?? throw new \ValueError("NDArray::openMemmap(): Argument #2 (\$mode) must be one of 'r', 'r+', 'w+' or 'c'");
        if ($m !== 'w') {
            return self::mapNpy($filename, $m);
        }
        if ($shape === null) {
            throw new \ValueError("NDArray::openMemmap(): Argument #3 (\$shape) is required with mode 'w+'");
        }
        self::checkShape($shape, 3);
        $path = self::resolve($filename);
        $ffi = Library::ffi();
        $n = count($shape);
        $buf = $ffi->new('char[4096]');
        $hlen = $ffi->tsr_npy_write_header($dtype->value, $n, Library::i64($shape), $fortranOrder ? 1 : 0, $buf, 4096);
        if ($hlen <= 0 || $hlen > 4096) {
            throw new TesseroException('openMemmap: could not build the .npy header.');
        }
        if (@file_put_contents($path, FFI::string($buf, $hlen)) !== $hlen) {
            throw new TesseroException("npy: cannot write '{$path}'");
        }

        return self::map($path, '+', $dtype, $shape, $hlen, $fortranOrder);
    }

    /** Map an existing .npy file in place (mode 'r', 'r+' or 'c'). */
    public static function mapNpy(string $filename, string $mode): NDArray
    {
        $path = self::resolve($filename);
        $head = @file_get_contents($path, false, null, 0, 65536 + 12);
        if ($head === false) {
            throw new TesseroException("npy: cannot open '{$path}'");
        }
        [$dtype, $shape, $fortran, $offset] = self::header($head, $path);
        if ($offset % $dtype->itemsize() !== 0) {
            throw new TesseroException("npy: the data in '{$path}' is not aligned to its item size; load() it instead");
        }
        if (in_array(0, $shape, true)) {
            return NDArray::zeros($shape, $dtype);
        }

        return self::map($path, $mode, $dtype, $shape, $offset, $fortran);
    }

    /**
     * Parse a .npy preamble with the kernel's strict parser.
     *
     * @return array{DType, list<int>, bool, int} dtype, shape, fortran order, data offset
     */
    public static function header(string $bytes, string $what): array
    {
        $ffi = Library::ffi();
        $dt = $ffi->new('int');
        $fo = $ffi->new('int');
        $nd = $ffi->new('int32_t');
        $off = $ffi->new('int64_t');
        $shape = $ffi->new('int64_t[32]');
        $rc = $ffi->tsr_npy_header($bytes, strlen($bytes), FFI::addr($dt), FFI::addr($nd), $shape, FFI::addr($fo), FFI::addr($off));
        if ($rc === -4) {
            throw new DTypeError("npy: '{$what}' has a dtype or byte order Tessero does not support "
                . '(supported: little-endian float64, float32, int64, int32, uint8, bool, complex128)');
        }
        if ($rc === -5 || $rc === -6) {
            throw new ShapeError("npy: '{$what}' has a shape Tessero cannot hold");
        }
        if ($rc !== 0) {
            throw new TesseroException("npy: '{$what}' is not a valid .npy file");
        }
        $s = [];
        for ($i = 0; $i < $nd->cdata; $i++) {
            $s[] = $shape[$i];
        }

        return [DType::from($dt->cdata), $s, $fo->cdata === 1, $off->cdata];
    }

    /** @param list<int>|null $shape */
    private static function map(string $path, string $mode, DType $dtype, ?array $shape, int $offset, bool $fortran): NDArray
    {
        $ffi = Library::ffi();
        $nd = $ffi->new('int32_t');
        $nd->cdata = $shape === null ? -1 : count($shape);
        $sh = $ffi->new('int64_t[32]');
        foreach ($shape ?? [] as $i => $v) {
            $sh[$i] = $v;
        }
        $handle = $ffi->new('void*');
        $data = $ffi->new('void*');
        $rc = $ffi->tsr_mmap_open($path, self::KERNEL_MODE[$mode], $dtype->value, FFI::addr($nd), $sh, $offset, FFI::addr($handle), FFI::addr($data));
        if ($rc !== 0) {
            $msg = 'memmap: ' . $ffi->tsr_mmap_error();
            throw ($rc === -6 || $rc === -5 || $rc === -3) ? new ShapeError($msg) : new TesseroException($msg);
        }
        $shapeOut = [];
        for ($i = 0; $i < $nd->cdata; $i++) {
            $shapeOut[] = $sh[$i];
        }
        $bytes = (int) array_product($shapeOut) * $dtype->itemsize();
        $buffer = Buffer::mapped(Library::address($data), $bytes, Library::address($handle), $path, $mode);
        $strides = null;
        if ($fortran) {
            $strides = [];
            $acc = $dtype->itemsize();
            foreach ($shapeOut as $len) {
                $strides[] = $acc;
                $acc *= max(1, $len);
            }
        }

        return new NDArray($buffer, $dtype, $shapeOut, $strides);
    }

    /** @param array<mixed> $shape */
    private static function checkShape(array $shape, int $arg): void
    {
        if (count($shape) > 32) {
            throw new ShapeError('memmap: more than 32 dimensions');
        }
        foreach ($shape as $v) {
            if (! is_int($v) || $v < 0) {
                throw new \ValueError("NDArray::memmap(): Argument #{$arg} (\$shape) must be a list of non-negative integers");
            }
        }
    }

    /** Absolute path, refusing stream wrappers and paths outside open_basedir (as ext-tessero does). */
    private static function resolve(string $filename): string
    {
        if ($filename === '' || str_contains($filename, "\0")) {
            throw new \ValueError('NDArray::memmap(): Argument #1 ($filename) cannot be empty');
        }
        if (str_contains($filename, '://')) {
            throw new \ValueError('NDArray::memmap(): Argument #1 ($filename) must be a local file path (stream wrappers cannot be memory-mapped)');
        }
        $basedir = (string) ini_get('open_basedir');
        $real = @realpath($filename);
        if ($real === false) {
            $dir = @realpath(dirname($filename));
            if ($dir === false) {
                throw new TesseroException($basedir !== ''
                    ? "memmap: '{$filename}' is outside the allowed path(s) (open_basedir) or does not exist"
                    : "memmap: cannot resolve path '{$filename}'");
            }
            $real = $dir . DIRECTORY_SEPARATOR . basename($filename);
        }
        if ($basedir !== '') {
            $ok = false;
            foreach (explode(PATH_SEPARATOR, $basedir) as $base) {
                $b = @realpath($base);
                if ($b !== false && ($real === $b || str_starts_with($real, rtrim($b, DIRECTORY_SEPARATOR) . DIRECTORY_SEPARATOR))) {
                    $ok = true;
                    break;
                }
            }
            if (! $ok) {
                throw new TesseroException("memmap: '{$real}' is outside the allowed path(s) (open_basedir)");
            }
        }

        return $real;
    }
}
