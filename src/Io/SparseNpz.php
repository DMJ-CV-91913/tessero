<?php

declare(strict_types=1);

namespace Tessero\Io;

use Tessero\DType;
use Tessero\Exceptions\TesseroException;
use Tessero\NDArray;
use Tessero\Sparse\CsrStructure;
use ZipArchive;

/**
 * scipy.sparse.save_npz / load_npz for CSR data: a .npz archive that scipy's
 * scipy.sparse.load_npz reads back as a csr_array (or csr_matrix), and that can
 * read what scipy.sparse.save_npz writes. The archive carries the CSR triple
 * (data / indices / indptr), the matrix shape, a 'format' tag ('csr') and an
 * '_is_array' flag; this is exactly scipy's on-disk layout.
 *
 * The CSR triple is written as standard .npy entries via Io\Npy::encode (indices
 * and indptr as int32, data as float64, matching scipy's small-matrix dtypes).
 * The scalar 'format' entry (a |S3 bytestring) and '_is_array' entry (a |b1 bool)
 * are hand-written/parsed here because Tessero's DType does not model bytestrings
 * and 0-d scalars; the rest routes through the shared Io\Npy code so both backends
 * behave identically.
 *
 * save_npz(path, {data, indices, indptr, shape}, isArray=true): writes the archive.
 * load_npz(path): returns ['data'=>NDArray, 'indices'=>NDArray, 'indptr'=>NDArray,
 *                           'shape'=>[m,n], 'format'=>'csr', 'is_array'=>bool].
 */
final class SparseNpz
{
    private const MAGIC = "\x93NUMPY";

    /**
     * scipy.sparse.save_npz: write a CSR matrix as a scipy-readable .npz.
     *
     * @param array{data:mixed,indices:mixed,indptr:mixed,shape:array<int>} $csr
     */
    public static function save(string $path, array $csr, bool $isArray = true, bool $compress = true): void
    {
        IoGuard::localPath($path, 'save_npz');
        self::requireZip();
        $data = self::asArray($csr['data'], DType::Float64);
        $indices = self::asArray($csr['indices'], DType::Int32);
        $indptr = self::asArray($csr['indptr'], DType::Int32);
        $shape = array_map('intval', array_values($csr['shape']));
        if (count($shape) !== 2) {
            throw new TesseroException('save_npz: shape must be 2-D (m, n).');
        }

        $zip = new ZipArchive();
        if ($zip->open($path, ZipArchive::CREATE | ZipArchive::OVERWRITE) !== true) {
            throw new TesseroException("save_npz: could not create {$path}.");
        }
        // scipy orders entries indices, indptr, format, shape, data, _is_array; order is irrelevant to readers.
        $entries = [
            'indices.npy' => Npy::encode($indices),
            'indptr.npy' => Npy::encode($indptr),
            'format.npy' => self::encodeBytesScalar('csr'),
            'shape.npy' => Npy::encode(NDArray::array($shape, DType::Int64)),
            'data.npy' => Npy::encode($data),
        ];
        if ($isArray) {
            $entries['_is_array.npy'] = self::encodeBoolScalar(true);
        }
        foreach ($entries as $name => $blob) {
            $zip->addFromString($name, $blob);
            $zip->setCompressionName($name, $compress ? ZipArchive::CM_DEFLATE : ZipArchive::CM_STORE);
        }
        $zip->close();
    }

    /**
     * scipy.sparse.load_npz: read a CSR .npz (scipy's or Tessero's own).
     *
     * The archive may inflate to at most $maxBytes (default 256 MiB, checked before decompressing), and the
     * CSR structure is validated (indptr starts at 0, never decreases and ends at nnz; indices inside the
     * matrix), so the arrays are safe to hand to the sparse kernels. $path must be a local file.
     *
     * @return array{data:NDArray,indices:NDArray,indptr:NDArray,shape:array{0:int,1:int},format:string,is_array:bool}
     */
    public static function load(string $path, int $maxBytes = IoGuard::DEFAULT_MAX_BYTES): array
    {
        IoGuard::localPath($path, 'load_npz');
        self::requireZip();
        $raw = [];
        foreach (IoGuard::zipMembers($path, $maxBytes, 'load_npz') as $name => $bytes) {
            $raw[preg_replace('/\.npy$/', '', $name)] = $bytes;
        }

        foreach (['data', 'indices', 'indptr', 'shape', 'format'] as $req) {
            if (! isset($raw[$req])) {
                throw new TesseroException("load_npz: archive is missing '{$req}.npy'.");
            }
        }
        $format = self::decodeBytesScalar($raw['format']);
        if ($format !== 'csr') {
            throw new TesseroException("load_npz: only CSR archives are supported, got '{$format}'.");
        }
        $shapeArr = Npy::decode($raw['shape'], 'shape.npy');
        $shapeList = array_map('intval', $shapeArr->toList());
        if (count($shapeList) !== 2 || $shapeList[0] < 0 || $shapeList[1] < 0) {
            throw new TesseroException('load_npz: shape entry is not 2-D.');
        }
        $data = Npy::decode($raw['data'], 'data.npy');
        $indices = Npy::decode($raw['indices'], 'indices.npy');
        $indptr = Npy::decode($raw['indptr'], 'indptr.npy');
        foreach (['indices' => $indices, 'indptr' => $indptr] as $what => $arr) {
            if (! $arr->dtype()->isInteger() || count($arr->shape()) !== 1) {
                throw new TesseroException("load_npz: '{$what}' must be a 1-D integer array.");
            }
        }
        if (count($data->shape()) !== 1) {
            throw new TesseroException("load_npz: 'data' must be a 1-D array.");
        }
        CsrStructure::validate($indptr->toList(), $indices->toList(), $data->size(), $shapeList[0], $shapeList[1], 'load_npz');

        return [
            'data' => $data,
            'indices' => $indices,
            'indptr' => $indptr,
            'shape' => [$shapeList[0], $shapeList[1]],
            'format' => 'csr',
            'is_array' => isset($raw['_is_array']) ? self::decodeBoolScalar($raw['_is_array']) : false,
        ];
    }

    /** Reconstruct the dense matrix from a loaded CSR archive (for verification / convenience). */
    public static function toDense(string $path): NDArray
    {
        $csr = self::load($path);
        [$m, $n] = $csr['shape'];
        $data = $csr['data']->toList();
        $indices = array_map('intval', $csr['indices']->toList());
        $indptr = array_map('intval', $csr['indptr']->toList());
        $dense = array_fill(0, $m, array_fill(0, $n, 0.0));
        for ($r = 0; $r < $m; $r++) {
            for ($k = $indptr[$r]; $k < $indptr[$r + 1]; $k++) {
                $dense[$r][$indices[$k]] = (float) $data[$k];
            }
        }

        return NDArray::array($dense, DType::Float64);
    }

    private static function asArray(mixed $v, DType $dt): NDArray
    {
        if ($v instanceof NDArray) {
            return $v->astype($dt);
        }
        if (is_object($v) && method_exists($v, 'toArray')) {   // ext NDArray or FFI NDArray facade result
            $v = $v->toArray();
        }

        return NDArray::array($v, $dt);
    }

    /** A 0-d |S<n> bytestring scalar .npy blob (scipy's 'format' entry). */
    private static function encodeBytesScalar(string $s): string
    {
        $descr = '|S' . max(1, strlen($s));
        $payload = $s;

        return self::scalarNpy($descr, $payload);
    }

    /** A 0-d |b1 bool scalar .npy blob (scipy's '_is_array' entry). */
    private static function encodeBoolScalar(bool $b): string
    {
        return self::scalarNpy('|b1', $b ? "\x01" : "\x00");
    }

    private static function scalarNpy(string $descr, string $payload): string
    {
        $dict = "{'descr': '{$descr}', 'fortran_order': False, 'shape': (), }";
        $prefix = strlen(self::MAGIC) + 2 + 2;
        $total = $prefix + strlen($dict) + 1;
        $pad = (64 - $total % 64) % 64;
        $header = $dict . str_repeat(' ', $pad) . "\n";

        return self::MAGIC . "\x01\x00" . pack('v', strlen($header)) . $header . $payload;
    }

    /** Parse a 0-d |S<n> bytestring scalar. */
    private static function decodeBytesScalar(string $bytes): string
    {
        [$descr, $start] = self::scalarHeader($bytes, 'format.npy');
        if ($descr[0] !== '|' || $descr[1] !== 'S') {
            throw new TesseroException("load_npz: 'format' entry has unexpected dtype '{$descr}'.");
        }
        $n = (int) substr($descr, 2);

        return rtrim(substr($bytes, $start, $n), "\x00");
    }

    private static function decodeBoolScalar(string $bytes): bool
    {
        [$descr, $start] = self::scalarHeader($bytes, '_is_array.npy');
        if ($start >= strlen($bytes)) {
            throw new TesseroException("load_npz: '_is_array.npy' is truncated.");
        }

        return $bytes[$start] !== "\x00";
    }

    /** @return array{0:string,1:int} [descr, data-start offset] for a 0-d scalar .npy. */
    private static function scalarHeader(string $bytes, string $what): array
    {
        if (strlen($bytes) < 12 || strncmp($bytes, self::MAGIC, 6) !== 0) {
            throw new TesseroException("load_npz: '{$what}' is not a .npy entry.");
        }
        $major = ord($bytes[6]);
        if ($major === 1) {
            $hlen = unpack('v', substr($bytes, 8, 2))[1];
            $hstart = 10;
        } else {
            $hlen = unpack('V', substr($bytes, 8, 4))[1];
            $hstart = 12;
        }
        if ($hstart + $hlen > strlen($bytes)) {
            throw new TesseroException("load_npz: '{$what}' header is truncated.");
        }
        $header = substr($bytes, $hstart, $hlen);
        if (! preg_match("/'descr'\\s*:\\s*'([^']{2,})'/", $header, $m)) {
            throw new TesseroException("load_npz: cannot parse '{$what}' header.");
        }

        return [$m[1], $hstart + $hlen];
    }

    private static function requireZip(): void
    {
        if (! class_exists(ZipArchive::class)) {
            throw new TesseroException('.npz needs ext-zip.');
        }
    }
}
