<?php

declare(strict_types=1);

namespace Tessero\Io;

use Tessero\Exceptions\TesseroException;

/**
 * IDL "save" files (scipy.io.readsav), pure PHP, read-only.
 *
 * IDL .sav files are a big-endian, XDR-style record stream produced by the Interactive Data
 * Language. The file opens with the signature "SR" followed by a 2-byte record format marker
 * ("\x00\x04" plain, "\x00\x06" zlib-compressed). After that comes a chain of length-linked
 * records; each starts with a 4-byte record type, an 8-byte "next record" offset and 4 skipped
 * bytes. VARIABLE records carry a name, a type descriptor (typecode + flags + optional array /
 * structure descriptors) and the raw value; HEAP_DATA records hold the targets of heap pointers.
 *
 * read() walks the record chain and returns a name => value map (names lower-cased, matching
 * scipy's AttrDict). Each value is {type, shape, data} where `data` is the flat row-major payload:
 *   - numeric scalars/arrays        -> floats/ints in C order (arrays are stored column-major on
 *                                      disk, so they are reshaped the way scipy does);
 *   - complex                       -> interleaved [re, im, ...];
 *   - strings                       -> the raw string.
 *
 * Scope: the numeric scalar types (byte/int16/int32/int64/uint16/uint32/uint64/float32/float64/
 * complex64/complex128) + string scalars, and numeric N-D arrays. Heap pointers, structures and
 * object references are not decoded (scipy's recursive STRUCTDESC / heap resolution); such records
 * are skipped. Compressed files are inflated with zlib when present.
 *
 * Backend-agnostic: pure PHP binary string unpacking, so FFI and ext behave identically.
 */
final class IdlSave
{
    /** IDL typecode => [name, kind]. kind: i=int, u=uint, f=float, c=complex, s=string, x=unsupported. */
    private const TYPECODE = [
        1 => ['uint8', 'u'], 2 => ['int16', 'i'], 3 => ['int32', 'i'], 4 => ['float32', 'f'],
        5 => ['float64', 'f'], 6 => ['complex64', 'c'], 7 => ['string', 's'], 8 => ['struct', 'x'],
        9 => ['complex128', 'c'], 10 => ['pointer', 'x'], 11 => ['object', 'x'],
        12 => ['uint16', 'u'], 13 => ['uint32', 'u'], 14 => ['int64', 'i'], 15 => ['uint64', 'u'],
    ];

    private string $buf;
    private int $pos = 0;
    private int $len;

    private function __construct(string $bytes)
    {
        $this->buf = $bytes;
        $this->len = strlen($bytes);
    }

    /**
     * Parse an IDL .sav byte string.
     *
     * @return array<string,array{type:string,shape:array<int,int>,data:array<int,mixed>}>
     *         name => {type, shape (C order), data (flat row-major; complex interleaved re/im)}
     */
    public static function read(string $bytes, int $maxBytes = IoGuard::DEFAULT_MAX_BYTES): array
    {
        IoGuard::checkLimit($maxBytes, 'maxBytes', 'readsav');
        if (strncmp($bytes, 'SR', 2) !== 0) {
            throw new TesseroException('readsav: invalid signature (expected "SR")');
        }
        $recfmt = substr($bytes, 2, 2);
        if ($recfmt === "\x00\x06") {
            $bytes = self::inflateSav($bytes, $maxBytes);
        } elseif ($recfmt !== "\x00\x04") {
            throw new TesseroException('readsav: invalid record format marker');
        }

        $r = new self($bytes);
        $r->pos = 4;                                        // skip "SR" + recfmt
        $variables = [];
        while (true) {
            $rec = $r->readRecord();
            if ($rec === null || ($rec['end'] ?? false)) {
                break;
            }
            if (($rec['rectype'] ?? '') === 'VARIABLE' && isset($rec['value'])) {
                $variables[strtolower((string) $rec['varname'])] = $rec['value'];
            }
        }

        return $variables;
    }

    /**
     * Inflate a /compress .sav: rebuild a plain stream by zlib-decompressing each record body. The rebuilt
     * stream may be at most $maxBytes long; every record must end after it starts and inside the file.
     */
    private static function inflateSav(string $bytes, int $maxBytes): string
    {
        $r = new self($bytes);
        $r->pos = 4;
        $out = "SR\x00\x04";
        while (true) {
            $start = $r->pos;
            $rectype = $r->readLong();
            $nextrec = $r->readUint32();
            $nextrec += $r->readUint32() * (2 ** 32);
            $unknown = $r->readBytes(4);
            if ($rectype === 6) {                           // END_MARKER: copy verbatim
                $out .= pack('N', $rectype);
                $mod = 2 ** 32;
                $out .= pack('N', (int) fmod($nextrec, $mod));
                $out .= pack('N', (int) (($nextrec - fmod($nextrec, $mod)) / $mod));
                $out .= $unknown;
                break;
            }
            $r->checkNext($start, $nextrec);
            $recString = IoGuard::inflate($r->readBytes((int) $nextrec - $r->pos), $maxBytes - strlen($out) - 16, 'readsav', 'any');
            $newNext = strlen($out) + strlen($recString) + 12;
            $out .= pack('N', $rectype);
            $out .= pack('N', (int) ($newNext % (2 ** 32)));
            $out .= pack('N', (int) (($newNext - ($newNext % (2 ** 32))) / (2 ** 32)));
            $out .= $unknown;
            $out .= $recString;
        }

        return $out;
    }

    /** Read one record; seeks to nextrec afterwards (mirrors scipy's _read_record). */
    private function readRecord(): ?array
    {
        if ($this->pos + 12 > $this->len) {
            return null;
        }
        $start = $this->pos;
        $rectype = $this->readLong();
        $nextrec = $this->readUint32();
        $nextrec += $this->readUint32() * (2 ** 32);
        $this->pos += 4;                                    // skip unknown

        if ($rectype === 6) {                               // END_MARKER
            return ['end' => true];
        }
        $this->checkNext($start, $nextrec);                 // before parsing: the chain must move forward
        $record = [];
        if ($rectype === 2 || $rectype === 16) {            // VARIABLE / HEAP_DATA
            $record['rectype'] = $rectype === 2 ? 'VARIABLE' : 'HEAP_DATA';
            if ($rectype === 2) {
                $record['varname'] = $this->readString();
            } else {
                $this->readLong();                          // heap_index
                $this->pos += 4;
            }
            $td = $this->readTypedesc();
            if ($td['typecode'] === 0) {
                $record['value'] = null;
            } else {
                $varstart = $this->readLong();
                if ($varstart !== 7) {
                    throw new TesseroException('readsav: VARSTART is not 7');
                }
                if ($td['structure']) {
                    $record['value'] = null;                // structures not decoded
                } elseif ($td['array']) {
                    $record['value'] = $this->readArray($td['typecode'], $td['array_desc']);
                } else {
                    $record['value'] = $this->readScalar($td['typecode']);
                }
            }
        }
        // All other record types (TIMESTAMP, VERSION, HEAP_HEADER, ...) carry no variable we need.

        $this->pos = (int) $nextrec;                        // seek to next record

        return $record;
    }

    /** Read a type descriptor (mirrors _read_typedesc). */
    private function readTypedesc(): array
    {
        $typecode = $this->readLong();
        $varflags = $this->readLong();
        if (($varflags & 2) === 2) {
            throw new TesseroException('readsav: system variables not implemented');
        }
        $td = [
            'typecode' => $typecode,
            'array' => ($varflags & 4) === 4,
            'structure' => ($varflags & 32) === 32,
        ];
        if ($td['structure']) {
            $td['array_desc'] = $this->readArraydesc();
            $td['struct_desc'] = null;                      // not decoded; readRecord returns null value
        } elseif ($td['array']) {
            $td['array_desc'] = $this->readArraydesc();
        }

        return $td;
    }

    /** Read an array descriptor (mirrors _read_arraydesc), 32-bit variant. */
    private function readArraydesc(): array
    {
        $arrstart = $this->readLong();
        if ($arrstart === 8) {
            $this->pos += 4;
            $nbytes = $this->readLong();
            $nelements = $this->readLong();
            $ndims = $this->readLong();
            $this->pos += 8;
            $nmax = $this->readLong();
            $dims = [];
            for ($i = 0; $i < $nmax; $i++) {
                $dims[] = $this->readLong();
            }

            return ['nbytes' => $nbytes, 'nelements' => $nelements, 'ndims' => $ndims, 'nmax' => $nmax, 'dims' => $dims];
        }
        if ($arrstart === 18) {                             // experimental 64-bit
            $this->pos += 8;
            $nbytes = $this->readUint64();
            $nelements = $this->readUint64();
            $ndims = $this->readLong();
            $this->pos += 8;
            $dims = [];
            for ($i = 0; $i < 8; $i++) {
                $this->readLong();                          // zero
                $dims[] = $this->readLong();
            }

            return ['nbytes' => $nbytes, 'nelements' => $nelements, 'ndims' => $ndims, 'nmax' => 8, 'dims' => $dims];
        }
        throw new TesseroException("readsav: unknown ARRSTART {$arrstart}");
    }

    /**
     * Read an array of `typecode` (mirrors _read_array), returning {type, shape, data}.
     * On-disk order is column-major (Fortran, fastest-first dims); scipy reverses the dims and
     * reshapes, so the flat we return equals numpy's .ravel(order='C') of the scipy result.
     */
    private function readArray(int $typecode, array $ad): array
    {
        [$typeName, $kind] = self::TYPECODE[$typecode] ?? ['?', 'x'];
        $n = (int) $ad['nelements'];
        $flat = [];

        if ($kind === 'c') {                                // complex: interleaved per element
            for ($i = 0; $i < $n; $i++) {
                $v = $this->readScalarValue($typecode);
                $flat[] = $v[0];
                $flat[] = $v[1];
            }
        } elseif ($typecode === 1) {                        // byte array: leading int32 nbytes, then packed bytes
            $this->readInt32();
            $raw = $this->readBytes((int) $ad['nbytes']);
            foreach (unpack('C*', $raw) as $b) {
                $flat[] = $b;
            }
        } elseif ($typecode === 2 || $typecode === 12) {    // int16/uint16: stored in 4-byte slots, scipy keeps every 2nd 2-byte word
            $raw = $this->readBytes((int) $ad['nbytes'] * 2);
            $words = str_split($raw, 2);
            for ($i = 1; $i < count($words); $i += 2) {
                $flat[] = $typecode === 2 ? self::beInt16($words[$i]) : self::beUint16($words[$i]);
            }
        } elseif ($kind === 'f' || $kind === 'i' || $kind === 'u') {
            $raw = $this->readBytes((int) $ad['nbytes']);
            $flat = self::unpackBlock($typecode, $raw);
        } else {
            for ($i = 0; $i < $n; $i++) {
                $flat[] = $this->readScalarValue($typecode);
            }
        }
        $this->align32();

        // numpy C shape = reverse(IDL dims). Disk order is Fortran over IDL dims, which equals
        // C order over the reversed shape, so the flat list is already in C order (just reshape).
        $ndims = (int) $ad['ndims'];
        $dims = array_slice($ad['dims'], 0, $ndims);
        $shape = array_reverse($dims);

        return ['type' => $typeName, 'shape' => array_map('intval', $shape), 'data' => $flat];
    }

    /** Read a scalar value (mirrors _read_data) and wrap as {type, shape:[], data:[...]}. */
    private function readScalar(int $typecode): array
    {
        [$typeName, $kind] = self::TYPECODE[$typecode] ?? ['?', 'x'];
        $v = $this->readScalarValue($typecode);
        $data = $kind === 'c' ? [$v[0], $v[1]] : [$v];

        return ['type' => $typeName, 'shape' => [], 'data' => $data];
    }

    /** Decode one scalar of `typecode`; complex returns [re, im], others a scalar. */
    private function readScalarValue(int $typecode): mixed
    {
        switch ($typecode) {
            case 1:                                         // byte: int32 flag==1 then a byte in a 4-byte slot
                if ($this->readInt32() !== 1) {
                    throw new TesseroException('readsav: error reading byte variable');
                }
                $b = $this->readBytes(4);

                return ord($b[0]);
            case 2:
                return self::beInt16(substr($this->readBytes(4), 2, 2));
            case 3:
                return $this->readInt32();
            case 4:
                return self::beFloat32($this->readBytes(4));
            case 5:
                return self::beFloat64($this->readBytes(8));
            case 6:
                return [self::beFloat32($this->readBytes(4)), self::beFloat32($this->readBytes(4))];
            case 7:
                return $this->readStringData();
            case 9:
                return [self::beFloat64($this->readBytes(8)), self::beFloat64($this->readBytes(8))];
            case 10:
            case 11:
                return $this->readInt32();                  // pointer index (unresolved)
            case 12:
                return self::beUint16(substr($this->readBytes(4), 2, 2));
            case 13:
                return $this->readUint32();
            case 14:
                return self::beInt64($this->readBytes(8));
            case 15:
                return self::beUint64($this->readBytes(8));
            default:
                throw new TesseroException("readsav: unknown IDL type {$typecode}");
        }
    }

    /** Unpack a contiguous BE block of a numeric typecode into a PHP array. */
    private static function unpackBlock(int $typecode, string $raw): array
    {
        switch ($typecode) {
            case 3:
                return self::signAll(array_values(unpack('N*', $raw)), 32);
            case 4:
                return array_values(unpack('G*', $raw));
            case 5:
                return array_values(unpack('E*', $raw));
            case 13:
                return array_values(unpack('N*', $raw));
            case 14:
                $out = [];
                foreach (str_split($raw, 8) as $c) {
                    $out[] = self::beInt64($c);
                }

                return $out;
            case 15:
                $out = [];
                foreach (str_split($raw, 8) as $c) {
                    $out[] = self::beUint64($c);
                }

                return $out;
        }
        throw new TesseroException("readsav: cannot block-unpack type {$typecode}");
    }

    /** Apply two's-complement sign to each value of a `$bits`-wide unsigned list. */
    private static function signAll(array $vals, int $bits): array
    {
        $half = 2 ** ($bits - 1);
        $mod = 2 ** $bits;
        foreach ($vals as &$v) {
            if ($v >= $half) {
                $v -= $mod;
            }
        }

        return $vals;
    }

    /**
     * A record's "next record" offset must lie after the record's start and inside the file; otherwise a
     * crafted chain seeks backwards (or onto itself) and the reader loops forever.
     */
    private function checkNext(int $start, int|float $nextrec): void
    {
        if (! ($nextrec > $start && $nextrec <= $this->len)) {
            throw new TesseroException("readsav: record at byte {$start} points to an invalid next record ({$nextrec})");
        }
    }

    // ---- primitive readers (advance $this->pos) ----
    private function readBytes(int $n): string
    {
        if ($n < 0) {
            throw new TesseroException('readsav: negative length');
        }
        if ($this->pos + $n > $this->len) {
            throw new TesseroException('readsav: unexpected end of file');
        }
        $s = substr($this->buf, $this->pos, $n);
        $this->pos += $n;

        return $s;
    }

    private function readLong(): int
    {
        return self::beInt32($this->readBytes(4));
    }

    private function readInt32(): int
    {
        return self::beInt32($this->readBytes(4));
    }

    private function readUint32(): int
    {
        return unpack('N', $this->readBytes(4))[1];
    }

    private function readUint64(): int|float
    {
        return self::beUint64($this->readBytes(8));
    }

    private function readString(): string
    {
        $length = $this->readLong();
        if ($length > 0) {
            $chars = $this->readBytes($length);
            $this->align32();

            return $chars;
        }

        return '';
    }

    private function readStringData(): string
    {
        $length = $this->readLong();
        if ($length > 0) {
            $length = $this->readLong();
            $s = $this->readBytes($length);
            $this->align32();

            return $s;
        }

        return '';
    }

    private function align32(): void
    {
        if ($this->pos % 4 !== 0) {
            $this->pos += 4 - $this->pos % 4;
        }
    }

    // ---- big-endian scalar decoders ----
    private static function beInt16(string $b): int
    {
        $v = unpack('n', $b)[1];

        return $v >= 0x8000 ? $v - 0x10000 : $v;
    }

    private static function beUint16(string $b): int
    {
        return unpack('n', $b)[1];
    }

    private static function beInt32(string $b): int
    {
        $v = unpack('N', $b)[1];

        return $v >= 0x80000000 ? $v - 0x100000000 : $v;
    }

    private static function beFloat32(string $b): float
    {
        return unpack('G', $b)[1];
    }

    private static function beFloat64(string $b): float
    {
        return unpack('E', $b)[1];
    }

    private static function beInt64(string $b): int
    {
        return unpack('J', $b)[1];                          // PHP int is 64-bit signed
    }

    private static function beUint64(string $b): int|float
    {
        $v = unpack('J', $b)[1];
        if ($v < 0) {
            return (float) $v + 2.0 ** 64;                  // wrap to unsigned magnitude
        }

        return $v;
    }
}
