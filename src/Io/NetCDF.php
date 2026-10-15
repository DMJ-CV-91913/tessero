<?php

declare(strict_types=1);

namespace Tessero\Io;

use Tessero\Exceptions\TesseroException;

/**
 * netCDF-3 classic reader/writer (scipy.io.netcdf_file / netcdf_variable), pure PHP.
 *
 * netCDF-3 is a self-describing, big-endian, 4-byte-aligned binary container. The layout is
 *   magic 'CDF' + version byte (1 = classic 32-bit offset, 2 = 64-bit offset)
 *   numrecs (int32; number of records for the unlimited dimension)
 *   dim_list   : NC_DIMENSION(0x0A) tag, count, then (name, length) per dim; length 0 = the record dim
 *   gatt_list  : NC_ATTRIBUTE(0x0C) tag, count, then (name, nc_type, nelems, values) global attributes
 *   var_list   : NC_VARIABLE(0x0B) tag, count, then per variable: name, ndims, dimids,
 *                per-variable attribute list, nc_type, vsize, begin-offset
 *   data       : non-record variables stored contiguously at their begin offset; record variables
 *                interleaved record-by-record in the record block.
 * Strings are int32 length + latin1 bytes padded to a 4-byte boundary. Numeric payloads are
 * big-endian and padded to a 4-byte boundary. nc_type codes: NC_BYTE=1, NC_CHAR=2, NC_SHORT=3,
 * NC_INT=4, NC_FLOAT=5, NC_DOUBLE=6; typecodes b/c/h/i/f/d.
 *
 * read()  parses scipy-written (and spec-conforming) bytes into {dimensions, variables, attrs}
 *         where each variable carries {dims, typecode, size, shape, data (flat row-major), attrs}.
 * write() emits bytes scipy.io.netcdf_file reads back, matching scipy's variable ordering
 *         (non-record vars first, by descending shape; record vars last), vsize/recsize maths,
 *         fill-value padding and begin offsets, so a round-trip is exact.
 *
 * Supports netCDF-3 classic (version 1) read+write with fixed dimensions and at most one leading
 * unlimited (record) dimension; version-2 (64-bit offset) files are read. Backend-agnostic: it only
 * packs/unpacks PHP strings and numeric arrays, so FFI and ext behave identically.
 */
final class NetCDF
{
    private const NC_BYTE = 1;
    private const NC_CHAR = 2;
    private const NC_SHORT = 3;
    private const NC_INT = 4;
    private const NC_FLOAT = 5;
    private const NC_DOUBLE = 6;

    private const NC_DIMENSION = 0x0A;
    private const NC_VARIABLE = 0x0B;
    private const NC_ATTRIBUTE = 0x0C;

    /** nc_type => [typecode, itemsize]. */
    private const TYPEMAP = [
        self::NC_BYTE => ['b', 1],
        self::NC_CHAR => ['c', 1],
        self::NC_SHORT => ['h', 2],
        self::NC_INT => ['i', 4],
        self::NC_FLOAT => ['f', 4],
        self::NC_DOUBLE => ['d', 8],
    ];

    /** typecode => nc_type (writer side). */
    private const REVERSE = ['b' => self::NC_BYTE, 'c' => self::NC_CHAR, 'h' => self::NC_SHORT,
        'i' => self::NC_INT, 'f' => self::NC_FLOAT, 'd' => self::NC_DOUBLE];

    /** Default fill values (big-endian bytes) used to pad a variable up to its vsize, per scipy FILLMAP. */
    private const FILLMAP = [
        self::NC_BYTE => "\x81",
        self::NC_CHAR => "\x00",
        self::NC_SHORT => "\x80\x01",
        self::NC_INT => "\x80\x00\x00\x01",
        self::NC_FLOAT => "\x7C\xF0\x00\x00",
        self::NC_DOUBLE => "\x47\x9E\x00\x00\x00\x00\x00\x00",
    ];

    // --------------------------------------------------------------------- read

    /**
     * Parse netCDF-3 bytes.
     *
     * @return array{version:int, numrecs:int, dimensions:array<string,int|null>,
     *               attrs:array<string,mixed>,
     *               variables:array<string,array{dims:array<int,string>, typecode:string, size:int,
     *                   shape:array<int,int>, data:array<int,int|float>|string, attrs:array<string,mixed>}>}
     */
    public static function read(string $bytes): array
    {
        $p = 0;
        $magic = substr($bytes, 0, 3);
        if ($magic !== 'CDF') {
            throw new TesseroException('netcdf: not a valid NetCDF-3 file (bad magic)');
        }
        $version = ord($bytes[3]);
        $p = 4;

        $numrecs = self::unpackInt($bytes, $p);
        [$dimNames, $dimLengths] = self::readDimArray($bytes, $p);
        $gattrs = self::readAttArray($bytes, $p);

        // var_list
        $header = self::unpackInt($bytes, $p);
        if ($header !== 0 && $header !== self::NC_VARIABLE) {
            throw new TesseroException('netcdf: unexpected var header');
        }
        $count = self::unpackInt($bytes, $p);

        $vars = [];
        $recVars = [];                                   // names of record variables, in file order
        $recsize = 0;
        $recBegin = 0;
        for ($vi = 0; $vi < $count; $vi++) {
            $name = self::unpackString($bytes, $p);
            $ndims = self::unpackInt($bytes, $p);
            $dims = [];
            $shape = [];
            for ($d = 0; $d < $ndims; $d++) {
                $dimid = self::unpackInt($bytes, $p);
                $dims[] = $dimNames[$dimid];
                $shape[] = $dimLengths[$dimid];          // null for the record dim
            }
            $vattrs = self::readAttArray($bytes, $p);
            $ncType = self::unpackInt($bytes, $p);
            [$typecode, $size] = self::TYPEMAP[$ncType];
            $vsize = self::unpackInt($bytes, $p);
            $begin = $version === 2 ? self::unpackInt64($bytes, $p) : self::unpackInt($bytes, $p);

            $isRec = $shape !== [] && $shape[0] === null;
            $vars[$name] = ['dims' => $dims, 'typecode' => $typecode, 'size' => $size,
                '_shape' => $shape, '_begin' => $begin, '_vsize' => $vsize,
                'attrs' => $vattrs, 'data' => null];
            if ($isRec) {
                $recVars[] = $name;
                $recsize += $vsize;
                if ($recBegin === 0) {
                    $recBegin = $begin;
                }
            }
        }

        // Non-record variable data: contiguous at begin, big-endian, reshaped row-major.
        foreach ($vars as $name => &$v) {
            if (in_array($name, $recVars, true)) {
                continue;
            }
            $n = 1;
            foreach ($v['_shape'] as $s) {
                $n *= (int) $s;
            }
            $raw = substr($bytes, $v['_begin'], $n * $v['size']);
            $dec = self::decodeValues($raw, $v['typecode'], $n);
            $v['data'] = ($v['typecode'] === 'c') ? implode('', $dec) : $dec;
            $v['shape'] = array_map('intval', $v['_shape']);
        }
        unset($v);

        // Record variables: interleaved per record, each padded to its own vsize within recsize.
        foreach ($recVars as $name) {
            $v = &$vars[$name];
            $perRec = 1;
            foreach (array_slice($v['_shape'], 1) as $s) {
                $perRec *= (int) $s;
            }
            $within = 0;
            foreach ($recVars as $rn) {
                if ($rn === $name) {
                    break;
                }
                $within += $vars[$rn]['_vsize'];
            }
            $flat = [];
            for ($r = 0; $r < $numrecs; $r++) {
                $off = $recBegin + $r * $recsize + $within;
                $raw = substr($bytes, $off, $perRec * $v['size']);
                $flat = array_merge($flat, self::decodeValues($raw, $v['typecode'], $perRec));
            }
            $v['data'] = ($v['typecode'] === 'c') ? implode('', $flat) : $flat;
            $v['shape'] = array_merge([$numrecs], array_map('intval', array_slice($v['_shape'], 1)));
            unset($v);
        }

        // Clean private keys.
        $outVars = [];
        foreach ($vars as $name => $v) {
            $outVars[$name] = ['dims' => $v['dims'], 'typecode' => $v['typecode'], 'size' => $v['size'],
                'shape' => $v['shape'], 'data' => $v['data'], 'attrs' => $v['attrs']];
        }

        $dimsOut = [];
        foreach ($dimNames as $i => $dn) {
            $dimsOut[$dn] = $dimLengths[$i];
        }

        return ['version' => $version, 'numrecs' => $numrecs, 'dimensions' => $dimsOut,
            'attrs' => $gattrs, 'variables' => $outVars];
    }

    /** @return array{0:array<int,string>,1:array<int,int|null>} ordered dim names and lengths */
    private static function readDimArray(string $bytes, int &$p): array
    {
        $header = self::unpackInt($bytes, $p);
        if ($header !== 0 && $header !== self::NC_DIMENSION) {
            throw new TesseroException('netcdf: unexpected dim header');
        }
        $count = self::unpackInt($bytes, $p);
        $names = [];
        $lengths = [];
        for ($i = 0; $i < $count; $i++) {
            $names[] = self::unpackString($bytes, $p);
            $l = self::unpackInt($bytes, $p);
            $lengths[] = $l === 0 ? null : $l;           // 0 marks the record (unlimited) dim
        }

        return [$names, $lengths];
    }

    /** @return array<string,mixed> attribute name => value (string for char, scalar/list for numeric) */
    private static function readAttArray(string $bytes, int &$p): array
    {
        $header = self::unpackInt($bytes, $p);
        if ($header !== 0 && $header !== self::NC_ATTRIBUTE) {
            throw new TesseroException('netcdf: unexpected attr header');
        }
        $count = self::unpackInt($bytes, $p);
        $attrs = [];
        for ($i = 0; $i < $count; $i++) {
            $name = self::unpackString($bytes, $p);
            $ncType = self::unpackInt($bytes, $p);
            $n = self::unpackInt($bytes, $p);
            [$typecode, $size] = self::TYPEMAP[$ncType];
            $count2 = $n * $size;
            $raw = substr($bytes, $p, $count2);
            $p += $count2;
            $p += (4 - ($count2 % 4)) % 4;               // advance past padding
            if ($typecode === 'c') {
                $attrs[$name] = rtrim($raw, "\x00");
            } else {
                $vals = self::decodeValues($raw, $typecode, $n);
                $attrs[$name] = count($vals) === 1 ? $vals[0] : $vals;
            }
        }

        return $attrs;
    }

    /** @return array<int,int|float|string> flat big-endian values; for char, list of single chars */
    private static function decodeValues(string $raw, string $typecode, int $n): array
    {
        $out = [];
        switch ($typecode) {
            case 'd':
                return $n === 0 ? [] : array_values(unpack('E*', $raw));   // big-endian doubles
            case 'f':
                return $n === 0 ? [] : array_values(unpack('G*', $raw));   // big-endian floats
            case 'i':
                if ($n === 0) {
                    return [];
                }
                foreach (unpack('N*', $raw) as $x) {
                    $out[] = $x >= 0x80000000 ? $x - 0x100000000 : $x;     // signed int32
                }

                return $out;
            case 'h':
                if ($n === 0) {
                    return [];
                }
                foreach (unpack('n*', $raw) as $x) {
                    $out[] = $x >= 0x8000 ? $x - 0x10000 : $x;             // signed int16
                }

                return $out;
            case 'b':
                for ($i = 0; $i < $n; $i++) {
                    $x = ord($raw[$i]);
                    $out[] = $x >= 0x80 ? $x - 0x100 : $x;                 // signed int8
                }

                return $out;
            case 'c':
                for ($i = 0; $i < $n; $i++) {
                    $out[] = $raw[$i];
                }

                return $out;
            default:
                throw new TesseroException("netcdf: unknown typecode {$typecode}");
        }
    }

    // --------------------------------------------------------------------- write

    /**
     * Emit netCDF-3 classic (version 1) bytes.
     *
     * $spec = [
     *   'dimensions' => ['name' => length, ...]   // length 0 marks the unlimited/record dim (must be first)
     *   'attrs'      => ['name' => value, ...]    // global attributes (string, int, float or list)
     *   'variables'  => [
     *       'name' => [
     *          'dims'     => ['dim', ...],         // dimension names in order
     *          'typecode' => 'd'|'f'|'i'|'h'|'b'|'c',
     *          'data'     => [flat row-major values]  (string for char),
     *          'attrs'    => ['name' => value, ...]   // optional per-variable attributes
     *       ], ...
     *   ],
     * ]
     *
     * Attribute type inference mirrors scipy: PHP int => NC_INT, float => NC_FLOAT, string => NC_CHAR.
     * To force a numeric type/width, pass ['type' => 'i'|'h'|'b'|'f'|'d', 'values' => [...]].
     */
    public static function write(array $spec): string
    {
        $dimensions = $spec['dimensions'] ?? [];
        $gattrs = $spec['attrs'] ?? [];
        $variables = $spec['variables'] ?? [];

        $dimId = array_flip(array_keys($dimensions));

        // numrecs = max record count among record variables.
        $numrecs = 0;
        foreach ($variables as $v) {
            if (self::isRecVar($v, $dimensions)) {
                $perRec = self::perRecCount($v, $dimensions);
                $cnt = $perRec > 0 ? (int) (count(self::flatData($v)) / $perRec) : 0;
                $numrecs = max($numrecs, $cnt);
            }
        }

        // Variable write order: scipy sorts non-record vars by descending shape, record vars last.
        $order = self::sortVariables($variables, $dimensions);

        $recVarCount = 0;
        foreach ($variables as $v) {
            if (self::isRecVar($v, $dimensions)) {
                $recVarCount++;
            }
        }

        // vsize per variable.
        $vsize = [];
        foreach ($order as $name) {
            $v = $variables[$name];
            $size = self::sizeOf($v['typecode']);
            if (! self::isRecVar($v, $dimensions)) {
                $bytes = self::totalCount($v, $dimensions) * $size;
                $bytes += (4 - ($bytes % 4)) % 4;
            } else {
                $bytes = self::perRecCount($v, $dimensions) * $size;
                if ($recVarCount > 1) {
                    $bytes += (4 - ($bytes % 4)) % 4;
                }
            }
            $vsize[$name] = $bytes;
        }
        $recsize = 0;
        foreach ($order as $name) {
            if (self::isRecVar($variables[$name], $dimensions)) {
                $recsize += $vsize[$name];
            }
        }

        // --- header ---
        $out = 'CDF' . chr(1);
        $out .= self::packInt($numrecs);

        // dim_array
        if ($dimensions) {
            $out .= self::packInt(self::NC_DIMENSION) . self::packInt(count($dimensions));
            foreach ($dimensions as $dn => $dl) {
                $out .= self::packString((string) $dn) . self::packInt((int) $dl);
            }
        } else {
            $out .= str_repeat("\x00", 8);
        }

        // gatt_array
        $out .= self::packAttArray($gattrs);

        // var_array — pass 1: metadata with placeholder begins; then patch begins and append data.
        if ($variables) {
            $out .= self::packInt(self::NC_VARIABLE) . self::packInt(count($variables));
            $beginPos = [];                              // byte offset of each var's begin field
            foreach ($order as $name) {
                $v = $variables[$name];
                $out .= self::packString($name);
                $out .= self::packInt(count($v['dims']));
                foreach ($v['dims'] as $dn) {
                    $out .= self::packInt($dimId[$dn]);
                }
                $out .= self::packAttArray($v['attrs'] ?? []);
                $out .= self::packInt(self::REVERSE[$v['typecode']]);
                $out .= self::packInt($vsize[$name]);
                $beginPos[$name] = strlen($out);
                $out .= self::packInt(0);                // placeholder begin
            }

            // Non-record variable data (in write order).
            foreach ($order as $name) {
                $v = $variables[$name];
                if (self::isRecVar($v, $dimensions)) {
                    continue;
                }
                $begin = strlen($out);
                $out = substr_replace($out, self::packInt($begin), $beginPos[$name], 4);
                $payload = self::encodeValues(self::flatData($v), $v['typecode']);
                $out .= $payload;
                $out .= self::fillPad($v['typecode'], $vsize[$name] - strlen($payload));
            }

            // Record variable block: set begins, then write interleaved records.
            $recOrder = array_values(array_filter($order, static fn ($n) => self::isRecVar($variables[$n], $dimensions)));
            if ($recOrder) {
                $recStart = strlen($out);
                $within = 0;
                foreach ($recOrder as $name) {
                    $out = substr_replace($out, self::packInt($recStart + $within), $beginPos[$name], 4);
                    $within += $vsize[$name];
                }
                $block = str_repeat("\x00", $numrecs * $recsize);
                $within = 0;
                foreach ($recOrder as $name) {
                    $v = $variables[$name];
                    $perRec = self::perRecCount($v, $dimensions);
                    $flat = self::flatData($v);
                    for ($r = 0; $r < $numrecs; $r++) {
                        $slice = array_slice($flat, $r * $perRec, $perRec);
                        $payload = self::encodeValues($slice, $v['typecode']);
                        $payload .= self::fillPad($v['typecode'], $vsize[$name] - strlen($payload));
                        $block = substr_replace($block, $payload, $r * $recsize + $within, strlen($payload));
                    }
                    $within += $vsize[$name];
                }
                $out .= $block;
            }
        } else {
            $out .= str_repeat("\x00", 8);
        }

        return $out;
    }

    /** Variable write order: scipy sorts non-record vars by descending _shape, record vars (key [-1]) last. */
    private static function sortVariables(array $variables, array $dimensions): array
    {
        $names = array_keys($variables);
        usort($names, static function (string $a, string $b) use ($variables, $dimensions): int {
            // scipy: sorted(..., reverse=True) on tuple keys -> compare kb vs ka.
            return self::cmpTuple(self::sortKey($variables[$b], $dimensions),
                self::sortKey($variables[$a], $dimensions));
        });

        return $names;
    }

    /** @return array<int,int> scipy sortkey: record var => [-1]; else the shape tuple */
    private static function sortKey(array $v, array $dimensions): array
    {
        if (self::isRecVar($v, $dimensions)) {
            return [-1];
        }
        $shape = [];
        foreach ($v['dims'] as $dn) {
            $shape[] = (int) $dimensions[$dn];
        }

        return $shape;
    }

    /** Lexicographic tuple compare matching Python tuple ordering. */
    private static function cmpTuple(array $a, array $b): int
    {
        $n = min(count($a), count($b));
        for ($i = 0; $i < $n; $i++) {
            if ($a[$i] !== $b[$i]) {
                return $a[$i] <=> $b[$i];
            }
        }

        return count($a) <=> count($b);
    }

    private static function isRecVar(array $v, array $dimensions): bool
    {
        if (($v['dims'] ?? []) === []) {
            return false;
        }

        return (int) ($dimensions[$v['dims'][0]] ?? -1) === 0;
    }

    private static function perRecCount(array $v, array $dimensions): int
    {
        $n = 1;
        foreach (array_slice($v['dims'], 1) as $dn) {
            $n *= (int) $dimensions[$dn];
        }

        return $n;
    }

    private static function totalCount(array $v, array $dimensions): int
    {
        $n = 1;
        foreach ($v['dims'] as $dn) {
            $n *= (int) $dimensions[$dn];
        }

        return $n;
    }

    /** @return array<int,int|float|string> flat values; char strings are split into single chars */
    private static function flatData(array $v): array
    {
        $d = $v['data'];
        if ($v['typecode'] === 'c' && is_string($d)) {
            return $d === '' ? [] : str_split($d);
        }

        return is_array($d) ? array_values($d) : [$d];
    }

    private static function sizeOf(string $typecode): int
    {
        return self::TYPEMAP[self::REVERSE[$typecode]][1];
    }

    private static function encodeValues(array $flat, string $typecode): string
    {
        $out = '';
        foreach ($flat as $x) {
            $out .= match ($typecode) {
                'd' => pack('E', (float) $x),
                'f' => pack('G', (float) $x),
                'i' => self::packInt((int) $x),
                'h' => pack('n', ((int) $x) & 0xFFFF),
                'b' => chr(((int) $x) & 0xFF),
                'c' => is_string($x) ? $x[0] : chr(((int) $x) & 0xFF),
                default => throw new TesseroException("netcdf: cannot encode typecode {$typecode}"),
            };
        }

        return $out;
    }

    private static function fillPad(string $typecode, int $bytes): string
    {
        if ($bytes <= 0) {
            return '';
        }
        $fill = self::FILLMAP[self::REVERSE[$typecode]];

        return str_repeat($fill, intdiv($bytes, strlen($fill)));
    }

    /** Pack one attribute list: NC_ATTRIBUTE tag, count, then (name, nc_type, nelems, values+pad). */
    private static function packAttArray(array $attrs): string
    {
        if (! $attrs) {
            return str_repeat("\x00", 8);
        }
        $out = self::packInt(self::NC_ATTRIBUTE) . self::packInt(count($attrs));
        foreach ($attrs as $name => $value) {
            $out .= self::packString((string) $name) . self::packAttValue($value);
        }

        return $out;
    }

    private static function packAttValue(mixed $value): string
    {
        // Explicit typing: ['type' => 'i'|'h'|'b'|'f'|'d', 'values' => [...]].
        if (is_array($value) && isset($value['type'], $value['values'])) {
            $tc = $value['type'];
            $vals = array_values($value['values']);
            $payload = self::encodeValues($vals, $tc);

            return self::packInt(self::REVERSE[$tc]) . self::packInt(count($vals)) . $payload
                . str_repeat("\x00", (4 - (strlen($payload) % 4)) % 4);
        }

        if (is_string($value)) {
            return self::packInt(self::NC_CHAR) . self::packInt(strlen($value)) . $value
                . str_repeat("\x00", (4 - (strlen($value) % 4)) % 4);
        }

        $vals = is_array($value) ? array_values($value) : [$value];
        $tc = is_int($vals[0]) ? 'i' : 'f';
        $payload = self::encodeValues($vals, $tc);

        return self::packInt(self::REVERSE[$tc]) . self::packInt(count($vals)) . $payload
            . str_repeat("\x00", (4 - (strlen($payload) % 4)) % 4);
    }

    // --------------------------------------------------------------------- primitives

    private static function packInt(int $v): string
    {
        return pack('N', $v & 0xFFFFFFFF);
    }

    private static function unpackInt(string $bytes, int &$p): int
    {
        $v = unpack('N', substr($bytes, $p, 4))[1];
        $p += 4;

        return $v >= 0x80000000 ? $v - 0x100000000 : $v;
    }

    private static function unpackInt64(string $bytes, int &$p): int
    {
        $v = unpack('J', substr($bytes, $p, 8))[1];
        $p += 8;

        return $v;
    }

    private static function packString(string $s): string
    {
        $count = strlen($s);

        return self::packInt($count) . $s . str_repeat("\x00", (4 - ($count % 4)) % 4);
    }

    private static function unpackString(string $bytes, int &$p): string
    {
        $count = self::unpackInt($bytes, $p);
        $s = rtrim(substr($bytes, $p, $count), "\x00");
        $p += $count + ((4 - ($count % 4)) % 4);

        return $s;
    }
}
