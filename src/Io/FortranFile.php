<?php

declare(strict_types=1);

namespace Tessero\Io;

use Tessero\Exceptions\TesseroException;

/**
 * Fortran unformatted sequential files (scipy.io.FortranFile), pure PHP.
 *
 * Fortran's unformatted sequential records are length-prefixed: each record is a leading record-length
 * marker, the raw little-endian data, then a trailing marker equal to the leading one. The marker type
 * is usually 4-byte unsigned (scipy's default `header_dtype='uint32'`); some compilers use 8-byte.
 *
 * readRecords() splits scipy-written (and compiler-written) bytes into records and decodes each as a flat
 *               PHP array for the given element dtype (int32 / int64 / float32 / float64).
 * writeRecord() frames one flat numeric array into a single record with matching leading/trailing markers,
 *               so scipy.io.FortranFile(...).read_record reads it back exactly.
 *
 * Backend-agnostic: pure binary string packing and PHP numeric arrays.
 */
final class FortranFile
{
    private const ITEMSIZE = ['int32' => 4, 'int64' => 8, 'float32' => 4, 'float64' => 8, 'uint32' => 4];

    /**
     * Split a Fortran unformatted file into records, decoding each to a flat PHP array.
     *
     * @param string $dtype       element dtype of every record (int32/int64/float32/float64)
     * @param string $headerDtype marker width: 'uint32' (default) or 'uint64'
     *
     * @return array<int,array<int,int|float>> one flat array per record
     */
    public static function readRecords(string $bytes, string $dtype, string $headerDtype = 'uint32'): array
    {
        $markerSize = $headerDtype === 'uint64' ? 8 : 4;
        $records = [];
        $pos = 0;
        $len = strlen($bytes);
        while ($pos < $len) {
            if ($pos + $markerSize > $len) {
                throw new TesseroException('FortranFile: truncated leading marker');
            }
            $recLen = self::readMarker($bytes, $pos, $markerSize);
            $pos += $markerSize;
            if ($pos + $recLen + $markerSize > $len) {
                throw new TesseroException('FortranFile: record longer than file');
            }
            $payload = substr($bytes, $pos, $recLen);
            $pos += $recLen;
            $trailing = self::readMarker($bytes, $pos, $markerSize);
            $pos += $markerSize;
            if ($trailing !== $recLen) {
                throw new TesseroException("FortranFile: marker mismatch ({$recLen} != {$trailing})");
            }
            $records[] = self::unpackData($payload, $dtype);
        }

        return $records;
    }

    /** Convenience: decode exactly the first record. @return array<int,int|float> */
    public static function readRecord(string $bytes, string $dtype, string $headerDtype = 'uint32'): array
    {
        $recs = self::readRecords($bytes, $dtype, $headerDtype);
        if ($recs === []) {
            throw new TesseroException('FortranFile: no records');
        }

        return $recs[0];
    }

    /**
     * Frame one flat numeric array into a single Fortran unformatted record.
     *
     * @param array<int,int|float> $values
     */
    public static function writeRecord(array $values, string $dtype, string $headerDtype = 'uint32'): string
    {
        $payload = self::packData($values, $dtype);
        $marker = self::packMarker(strlen($payload), $headerDtype === 'uint64' ? 8 : 4);

        return $marker . $payload . $marker;
    }

    /** Frame several records back-to-back. @param array<int,array<int,int|float>> $records */
    public static function writeRecords(array $records, string $dtype, string $headerDtype = 'uint32'): string
    {
        $out = '';
        foreach ($records as $r) {
            $out .= self::writeRecord($r, $dtype, $headerDtype);
        }

        return $out;
    }

    private static function readMarker(string $bytes, int $pos, int $size): int
    {
        $chunk = substr($bytes, $pos, $size);

        return $size === 8 ? unpack('P', $chunk)[1] : unpack('V', $chunk)[1];
    }

    private static function packMarker(int $v, int $size): string
    {
        return $size === 8 ? pack('P', $v) : pack('V', $v);
    }

    /** @return array<int,int|float> */
    private static function unpackData(string $raw, string $dtype): array
    {
        if ($raw === '') {
            return [];
        }

        return match ($dtype) {
            'int32' => self::toSigned(array_values(unpack('V*', $raw)), 32),
            'int64' => array_values(unpack('q*', $raw)),
            'float32' => array_values(unpack('g*', $raw)),
            'float64' => array_values(unpack('e*', $raw)),
            default => throw new TesseroException("FortranFile: unsupported dtype '{$dtype}'"),
        };
    }

    /** @param array<int,int> $u @return array<int,int> */
    private static function toSigned(array $u, int $bits): array
    {
        $mod = 1 << $bits;
        $half = 1 << ($bits - 1);

        return array_map(static fn (int $x) => $x >= $half ? $x - $mod : $x, $u);
    }

    /** @param array<int,int|float> $values */
    private static function packData(array $values, string $dtype): string
    {
        return match ($dtype) {
            'int32' => pack('V*', ...array_map(static fn ($x) => ((int) $x) & 0xFFFFFFFF, $values)),
            'int64' => pack('q*', ...array_map('intval', $values)),
            'float32' => $values === [] ? '' : pack('g*', ...array_map('floatval', $values)),
            'float64' => $values === [] ? '' : pack('e*', ...array_map('floatval', $values)),
            default => throw new TesseroException("FortranFile: unsupported dtype '{$dtype}'"),
        };
    }
}
