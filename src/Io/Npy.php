<?php

declare(strict_types=1);

namespace Tessero\Io;

use Tessero\DType;
use Tessero\Exceptions\TesseroException;
use Tessero\NDArray;
use ZipArchive;

/**
 * NumPy .npy (format 1.0 / 2.0 / 3.0) and .npz reading and writing, so arrays
 * move between PHP and Python without JSON: np.load('x.npy') reads what save()
 * writes and vice versa. Headers are parsed by libtessero's strict parser
 * (tsr_npy_header), the same code ext-tessero uses. For memory-mapped .npy
 * files see NDArray::openMemmap() and NDArray::load($path, $mmapMode).
 */
final class Npy
{
    private const MAGIC = "\x93NUMPY";

    public static function save(string $path, NDArray $array): void
    {
        if (file_put_contents($path, self::encode($array)) === false) {
            throw new TesseroException("Could not write {$path}.");
        }
    }

    public static function load(string $path): NDArray
    {
        $bytes = @file_get_contents($path);
        if ($bytes === false) {
            throw new TesseroException("npy: cannot open '{$path}'");
        }

        return self::decode($bytes, $path);
    }

    public static function encode(NDArray $array): string
    {
        $shape = $array->shape();
        $shapeStr = '(' . implode(', ', $shape) . (count($shape) === 1 ? ',' : '') . ')';
        $dict = "{'descr': '" . $array->dtype()->descr() . "', 'fortran_order': False, 'shape': {$shapeStr}, }";
        $major = 1;
        $prefix = strlen(self::MAGIC) + 2 + 2;
        if (strlen($dict) + $prefix + 1 > 65535) {
            $major = 2;
            $prefix += 2;
        }
        $total = $prefix + strlen($dict) + 1;
        $pad = (64 - $total % 64) % 64;
        $header = $dict . str_repeat(' ', $pad) . "\n";
        $len = $major === 1 ? pack('v', strlen($header)) : pack('V', strlen($header));

        return self::MAGIC . chr($major) . chr(0) . $len . $header . $array->toBytes();
    }

    public static function decode(string $bytes, string $what = '.npy data'): NDArray
    {
        [$dtype, $shape, $fortran, $start] = MemoryMap::header($bytes, $what);
        $expected = (int) array_product($shape) * $dtype->itemsize();
        if (strlen($bytes) - $start < $expected) {
            throw new TesseroException("npy: '{$what}' is truncated (" . max(0, strlen($bytes) - $start) . " data bytes, {$expected} expected)");
        }
        $data = substr($bytes, $start, $expected);
        if ($fortran && count($shape) > 1) {
            return NDArray::fromBytes($data, $dtype, array_reverse($shape))->transpose()->copy();
        }

        return NDArray::fromBytes($data, $dtype, $shape);
    }

    /** @param array<string, NDArray> $arrays */
    public static function saveZ(string $path, array $arrays, bool $compress = true): void
    {
        self::requireZip();
        $zip = new ZipArchive();
        if ($zip->open($path, ZipArchive::CREATE | ZipArchive::OVERWRITE) !== true) {
            throw new TesseroException("Could not create {$path}.");
        }
        foreach ($arrays as $name => $arr) {
            $entry = $name . '.npy';
            $zip->addFromString($entry, self::encode($arr));
            $zip->setCompressionName($entry, $compress ? ZipArchive::CM_DEFLATE : ZipArchive::CM_STORE);
        }
        $zip->close();
    }

    /** @return array<string, NDArray> */
    public static function loadZ(string $path): array
    {
        self::requireZip();
        $zip = new ZipArchive();
        if ($zip->open($path) !== true) {
            throw new TesseroException("Could not open {$path}.");
        }
        $out = [];
        for ($i = 0; $i < $zip->numFiles; $i++) {
            $name = (string) $zip->getNameIndex($i);
            $out[preg_replace('/\.npy$/', '', $name)] = self::decode((string) $zip->getFromIndex($i));
        }
        $zip->close();

        return $out;
    }

    private static function requireZip(): void
    {
        if (! class_exists(ZipArchive::class)) {
            throw new TesseroException('.npz needs ext-zip.');
        }
    }
}
