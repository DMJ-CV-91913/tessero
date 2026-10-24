<?php

declare(strict_types=1);

namespace Tessero\Io;

use Tessero\Exceptions\TesseroException;

/**
 * MATLAB level-5 MAT-file format (scipy.io.loadmat / savemat / whosmat), pure PHP.
 *
 * A MAT v5 file is a 128-byte header (116-byte ASCII description, 8-byte subsystem offset, a 2-byte
 * version 0x0100 and the 2-byte endian marker "IM" for little-endian) followed by a flat sequence of
 * data elements, one miMATRIX per top-level variable. Every element is a tag (4-byte LE type, 4-byte LE
 * byte-count) then its data padded to an 8-byte boundary; small (<=4 byte) elements use the compressed
 * "small data element" tag where the byte-count is packed into the high 16 bits of the type word. A
 * miMATRIX nests, in order: array-flags (class + complex/global/logical bits), int32 dimensions
 * (column-major), an int8 name, the real part pr and -- when complex -- the imaginary part pi. Numeric
 * data is stored column-major (Fortran). A top-level element may be miCOMPRESSED, a zlib stream that
 * inflates to an ordinary miMATRIX (scipy's do_compression=True); this reader transparently inflates it.
 *
 * read()  parses scipy-written (and self-written) bytes into name => {shape, class, data, iscomplex}
 *         where `data` is the flat matrix in C/row-major order (complex entries interleaved re, im).
 * write() emits bytes that scipy.io.loadmat accepts, matching scipy's tag layout, small-element packing
 *         and column-major ordering, so a scipy round-trip is exact.
 * info()  (whosmat) reads only the tags/dims/flags of each variable, returning [name, shape, class].
 *
 * Supported classes: double, single, int8/16/32/64, uint8/16/32/64, char and logical, real or complex,
 * 2-D (the common-usage subset). Struct / cell / object / sparse variables are recognised by whosmat but
 * read() raises on them rather than guessing. Backend-agnostic: pure string pack/unpack, no BLAS.
 */
final class MatIO
{
    // MAT data-type codes (the element tag's type field).
    private const MI_INT8 = 1, MI_UINT8 = 2, MI_INT16 = 3, MI_UINT16 = 4, MI_INT32 = 5, MI_UINT32 = 6;
    private const MI_SINGLE = 7, MI_DOUBLE = 9, MI_INT64 = 12, MI_UINT64 = 13;
    private const MI_MATRIX = 14, MI_COMPRESSED = 15, MI_UTF8 = 16, MI_UTF16 = 17, MI_UTF32 = 18;

    // MATLAB array-class codes (the array-flags low byte).
    private const MX_CELL = 1, MX_STRUCT = 2, MX_OBJECT = 3, MX_CHAR = 4, MX_SPARSE = 5;
    private const MX_DOUBLE = 6, MX_SINGLE = 7, MX_INT8 = 8, MX_UINT8 = 9, MX_INT16 = 10, MX_UINT16 = 11;
    private const MX_INT32 = 12, MX_UINT32 = 13, MX_INT64 = 14, MX_UINT64 = 15;

    private const FLAG_COMPLEX = 0x08, FLAG_LOGICAL = 0x02;

    /** MATLAB class code -> scipy whosmat class string. */
    private const CLASS_NAME = [
        self::MX_DOUBLE => 'double', self::MX_SINGLE => 'single',
        self::MX_INT8 => 'int8', self::MX_UINT8 => 'uint8', self::MX_INT16 => 'int16', self::MX_UINT16 => 'uint16',
        self::MX_INT32 => 'int32', self::MX_UINT32 => 'uint32', self::MX_INT64 => 'int64', self::MX_UINT64 => 'uint64',
        self::MX_CHAR => 'char', self::MX_SPARSE => 'sparse', self::MX_CELL => 'cell', self::MX_STRUCT => 'struct',
        self::MX_OBJECT => 'object',
    ];

    /** Numeric scipy class string -> [MATLAB class code, pr element type, itemsize, signed?, is_float?]. */
    private const NUMERIC = [
        'double' => [self::MX_DOUBLE, self::MI_DOUBLE, 8, true, true],
        'single' => [self::MX_SINGLE, self::MI_SINGLE, 4, true, true],
        'int8'   => [self::MX_INT8, self::MI_INT8, 1, true, false],
        'uint8'  => [self::MX_UINT8, self::MI_UINT8, 1, false, false],
        'int16'  => [self::MX_INT16, self::MI_INT16, 2, true, false],
        'uint16' => [self::MX_UINT16, self::MI_UINT16, 2, false, false],
        'int32'  => [self::MX_INT32, self::MI_INT32, 4, true, false],
        'uint32' => [self::MX_UINT32, self::MI_UINT32, 4, false, false],
        'int64'  => [self::MX_INT64, self::MI_INT64, 8, true, false],
        'uint64' => [self::MX_UINT64, self::MI_UINT64, 8, false, false],
    ];

    // ---------------------------------------------------------------- loadmat ---

    /**
     * Parse MAT v5 bytes into variables. Compressed variables inflate to at most $maxBytes in total
     * (default 256 MiB) and at most $maxElements values are decoded (see IoGuard); declared sizes are
     * checked against the bytes present before any loop.
     *
     * @return array<string, array{shape:array<int,int>, class:string, data:array<int,int|float>, iscomplex:bool}>
     *   `data` is the flat matrix in C/row-major order; complex matrices carry interleaved [re, im, ...].
     */
    public static function read(string $bytes, int $maxBytes = IoGuard::DEFAULT_MAX_BYTES, int $maxElements = IoGuard::DEFAULT_MAX_ELEMENTS): array
    {
        IoGuard::checkLimit($maxBytes, 'maxBytes', 'loadmat');
        IoGuard::checkLimit($maxElements, 'maxElements', 'loadmat');
        if (strlen($bytes) < 128) {
            throw new TesseroException('loadmat: truncated header');
        }
        $out = [];
        $pos = 128;
        $len = strlen($bytes);
        $inflated = 0;                                       // compressed variables share one $maxBytes budget
        $elements = 0;                                       // values decoded so far, against $maxElements
        while ($pos + 8 <= $len) {
            [$type, $nbytes, $dataPos, $next] = self::readTag($bytes, $pos, 'loadmat');
            if ($type === self::MI_COMPRESSED) {
                $inner = IoGuard::inflate(substr($bytes, $dataPos, $nbytes), $maxBytes - $inflated, 'loadmat');
                $inflated += strlen($inner);
                [$iType, $iBytes, $iPos] = self::readTag($inner, 0, 'loadmat');   // inflated stream starts with the miMATRIX tag
                if ($iType === self::MI_MATRIX) {
                    [$name, $var] = self::parseMatrix($inner, $iPos, $iPos + $iBytes, $maxElements, $elements);
                    if ($name !== '') {
                        $out[$name] = $var;
                    }
                }
            } elseif ($type === self::MI_MATRIX) {
                [$name, $var] = self::parseMatrix($bytes, $dataPos, $dataPos + $nbytes, $maxElements, $elements);
                if ($name !== '') {
                    $out[$name] = $var;
                }
            }
            $pos = $next;
        }

        return $out;
    }

    /**
     * whosmat: name/shape/class of every variable without decoding data.
     *
     * @return array<int, array{0:string, 1:array<int,int>, 2:string}>
     */
    public static function info(string $bytes, int $maxBytes = IoGuard::DEFAULT_MAX_BYTES): array
    {
        IoGuard::checkLimit($maxBytes, 'maxBytes', 'whosmat');
        if (strlen($bytes) < 128) {
            throw new TesseroException('whosmat: truncated header');
        }
        $out = [];
        $pos = 128;
        $len = strlen($bytes);
        $total = 0;
        while ($pos + 8 <= $len) {
            [$type, $nbytes, $dataPos, $next] = self::readTag($bytes, $pos, 'whosmat');
            $blob = null;
            $base = 0;
            $end = 0;
            if ($type === self::MI_COMPRESSED) {
                $inflated = IoGuard::inflate(substr($bytes, $dataPos, $nbytes), $maxBytes - $total, 'whosmat');
                $total += strlen($inflated);
                [$iType, $iBytes, $iPos] = self::readTag($inflated, 0, 'whosmat');
                if ($iType === self::MI_MATRIX) {
                    $blob = $inflated;
                    $base = $iPos;
                    $end = $iPos + $iBytes;
                }
            } elseif ($type === self::MI_MATRIX) {
                $blob = $bytes;
                $base = $dataPos;
                $end = $dataPos + $nbytes;
            }
            if ($blob !== null) {
                [$name, $shape, $cls] = self::peekMatrix($blob, $base, $end);
                if ($name !== '') {
                    $out[] = [$name, $shape, $cls];
                }
            }
            $pos = $next;
        }

        return $out;
    }

    // ---------------------------------------------------------------- savemat ---

    /**
     * Emit MAT v5 bytes for a dict of variables.
     *
     * @param array<string, array{shape:array<int,int>, class:string, data:array<int,int|float>, iscomplex?:bool}> $vars
     *   Each value: `shape` the 2-D [rows, cols] (1-D is treated as [1, n]); `class` a scipy class string
     *   (double, single, the signed/unsigned int kinds, char or logical); `data` the flat C/row-major values (interleaved re, im when
     *   `iscomplex`). char data may be given as the raw string or as a flat list of code points.
     */
    public static function write(array $vars): string
    {
        $desc = 'MATLAB 5.0 MAT-file, Platform: PHP, Created by Tessero';
        $header = str_pad($desc, 116, ' ');
        $header .= str_repeat("\x00", 8);          // subsystem offset
        $header .= pack('v', 0x0100);              // version
        $header .= 'IM';                           // little-endian marker
        $body = '';
        foreach ($vars as $name => $v) {
            $body .= self::writeMatrix((string) $name, $v);
        }

        return $header . $body;
    }

    // ---------------------------------------------------------------- internals ---

    /**
     * Decode an element tag at $pos.
     *
     * @return array{0:int, 1:int, 2:int, 3:int} [type, nbytes, dataPos, nextPos]
     */
    private static function readTag(string $b, int $pos, string $what = 'loadmat'): array
    {
        $len = strlen($b);
        if ($pos < 0 || $pos + 8 > $len) {          // both tag forms occupy 8 bytes
            throw new TesseroException("{$what}: truncated element tag at byte {$pos}");
        }
        $w = unpack('V', substr($b, $pos, 4))[1];
        if (($w >> 16) !== 0) {                     // small data element: nbytes in high 16 bits
            $type = $w & 0xFFFF;
            $nbytes = ($w >> 16) & 0xFFFF;
            if ($nbytes > 4) {
                throw new TesseroException("{$what}: small data element larger than 4 bytes at byte {$pos}");
            }

            return [$type, $nbytes, $pos + 4, $pos + 8];   // 4-byte tag + 4-byte data slot
        }
        $type = $w;
        $nbytes = unpack('V', substr($b, $pos + 4, 4))[1];
        $dataPos = $pos + 8;
        if ($nbytes > $len - $dataPos) {            // the declared size must be present
            throw new TesseroException("{$what}: element at byte {$pos} declares {$nbytes} bytes, only " . ($len - $dataPos) . ' remain');
        }
        $pad = (8 - ($nbytes % 8)) % 8;             // standard elements pad to 8 bytes

        return [$type, $nbytes, $dataPos, $dataPos + $nbytes + $pad];
    }

    /**
     * Read the array-flags, dims and name sub-elements of a miMATRIX body [$start, $end).
     *
     * @return array{0:int, 1:int, 2:array<int,int>, 3:string, 4:int} [class, flags, dims, name, posAfterName]
     */
    private static function readHead(string $b, int $start, int $end): array
    {
        [, $afBytes, $afPos, $afNext] = self::readTag($b, $start);          // array flags
        if ($afBytes < 4) {
            throw new TesseroException('loadmat: array flags element is too short');
        }
        $flagsWord = unpack('V', substr($b, $afPos, 4))[1];
        $class = $flagsWord & 0xFF;
        $flags = ($flagsWord >> 8) & 0xFF;

        [$dType, $dBytes, $dPos, $dNext] = self::readTag($b, $afNext);   // dimensions
        $dims = array_values(unpack('V*', substr($b, $dPos, $dBytes)));

        [$nType, $nBytes, $nPos, $nNext] = self::readTag($b, $dNext);    // name
        $name = substr($b, $nPos, $nBytes);

        return [$class, $flags, $dims, $name, $nNext];
    }

    /** Lightweight whosmat decode: head only. @return array{0:string,1:array<int,int>,2:string} */
    private static function peekMatrix(string $b, int $start, int $end): array
    {
        [$class, , $dims, $name] = self::readHead($b, $start, $end);
        $cls = self::CLASS_NAME[$class] ?? 'unknown';
        if ($class === self::MX_CHAR && count($dims) > 1) {
            array_pop($dims);                          // scipy reports char shape without the string-length dim
        }

        return [$name, $dims, $cls];
    }

    /**
     * Fully decode one miMATRIX body [$start, $end).
     *
     * @return array{0:string, 1:array{shape:array<int,int>,class:string,data:array<int,int|float>,iscomplex:bool}}
     */
    private static function parseMatrix(string $b, int $start, int $end, int $maxElements, int &$elements): array
    {
        [$class, $flags, $dims, $name, $prStart] = self::readHead($b, $start, $end);
        $complex = ($flags & self::FLAG_COMPLEX) !== 0;

        if (! isset(self::CLASS_NAME[$class])) {
            throw new TesseroException("loadmat: unsupported class code {$class}");
        }
        if (in_array($class, [self::MX_STRUCT, self::MX_CELL, self::MX_OBJECT, self::MX_SPARSE], true)) {
            throw new TesseroException('loadmat: ' . self::CLASS_NAME[$class] . ' variables are not supported');
        }

        // real part. Every count is checked against $maxElements before anything is decoded or looped over.
        [$rType, $rBytes, $rPos, $rNext] = self::readTag($b, $prStart);
        self::spend($elements, intdiv($rBytes, self::itemsize($rType)), $maxElements);
        $re = self::decodeNumeric(substr($b, $rPos, $rBytes), $rType);
        $im = null;
        if ($complex) {
            [$iType, $iBytes, $iPos] = self::readTag($b, $rNext);
            self::spend($elements, intdiv($iBytes, self::itemsize($iType)), $maxElements);
            $im = self::decodeNumeric(substr($b, $iPos, $iBytes), $iType);
        }

        if ($class === self::MX_CHAR) {
            $cls = 'char';
        } else {
            $cls = self::CLASS_NAME[$class];
        }

        // stored column-major; reorder to C/row-major for the flat result.
        $rows = $dims[0] ?? 0;
        $cols = $dims[1] ?? 1;
        $data = [];
        // the declared dims drive the loop below: the values must actually be there (a 0-sized dim skips it)
        $count = $rows * $cols;                     // two uint32: at most ~1.8e19, a float past PHP_INT_MAX
        if ($count > 0 && ($count > count($re) || ($complex && $count > count($im)))) {
            throw new TesseroException("loadmat: variable '{$name}' declares {$rows}x{$cols} values but holds " . count($re));
        }                                            // so the output is no larger than the values already counted
        $toFloat = ($class === self::MX_DOUBLE || $class === self::MX_SINGLE);
        for ($r = 0; $count > 0 && $r < $rows; $r++) {
            for ($c = 0; $c < $cols; $c++) {
                $k = $c * $rows + $r;                 // column-major source index
                if ($complex) {
                    $data[] = $toFloat ? (float) $re[$k] : $re[$k];
                    $data[] = $toFloat ? (float) $im[$k] : $im[$k];
                } else {
                    $data[] = $toFloat ? (float) $re[$k] : $re[$k];
                }
            }
        }
        if ($rows * $cols === 0) {
            $data = [];
        }

        return [$name, ['shape' => $dims, 'class' => $cls, 'data' => $data, 'iscomplex' => $complex]];
    }

    /** Bytes per value of a MAT data type (unknown types are rejected later by decodeNumeric). */
    private static function itemsize(int $type): int
    {
        return match ($type) {
            self::MI_DOUBLE, self::MI_INT64, self::MI_UINT64 => 8,
            self::MI_SINGLE, self::MI_INT32, self::MI_UINT32, self::MI_UTF32 => 4,
            self::MI_INT16, self::MI_UINT16, self::MI_UTF16 => 2,
            default => 1,
        };
    }

    /** Count $n more decoded values against the per-file cap. */
    private static function spend(int &$elements, int $n, int $maxElements): void
    {
        $elements += $n;
        if ($elements > $maxElements) {
            throw new TesseroException("loadmat: the file holds more than {$maxElements} values (\$maxElements)");
        }
    }

    /**
     * Decode a numeric data blob of MAT type $type into a flat PHP array (storage order preserved).
     *
     * @return array<int,int|float>
     */
    private static function decodeNumeric(string $blob, int $type): array
    {
        switch ($type) {
            case self::MI_DOUBLE:
                return array_values(unpack('e*', $blob));
            case self::MI_SINGLE:
                return array_values(unpack('g*', $blob));
            case self::MI_INT8:
                return array_values(unpack('c*', $blob));
            case self::MI_UINT8:
                return array_values(unpack('C*', $blob));
            case self::MI_INT16:
                return array_values(unpack('s*', $blob));
            case self::MI_UINT16:
            case self::MI_UTF16:
                return array_values(unpack('v*', $blob));
            case self::MI_INT32:
                return array_values(unpack('l*', $blob));
            case self::MI_UINT32:
            case self::MI_UTF32:
                return array_values(unpack('V*', $blob));
            case self::MI_UTF8:
                return array_values(unpack('C*', $blob));
            case self::MI_INT64:
                return array_values(unpack('q*', $blob));
            case self::MI_UINT64:
                return array_values(unpack('P*', $blob));
            default:
                throw new TesseroException("loadmat: unsupported element type {$type}");
        }
    }

    /** Encode one variable as a miMATRIX element (standard, uncompressed). */
    private static function writeMatrix(string $name, array $v): string
    {
        $cls = $v['class'];
        $shape = $v['shape'];
        if (count($shape) === 1) {
            $shape = [1, $shape[0]];
        }
        $rows = $shape[0];
        $cols = $shape[1] ?? 1;
        $complex = (bool) ($v['iscomplex'] ?? false);
        $data = $v['data'];

        if ($cls === 'char') {
            return self::writeChar($name, $rows, $cols, $data);
        }
        if (! isset(self::NUMERIC[$cls])) {
            throw new TesseroException("savemat: unsupported class {$cls}");
        }
        [$mxClass, $miType, $itemsize, $signed, $isFloat] = self::NUMERIC[$cls];

        // split interleaved complex into re/im, de-interleave row-major -> column-major.
        if ($complex) {
            $re = $im = [];
            for ($i = 0, $n = count($data); $i < $n; $i += 2) {
                $re[] = $data[$i];
                $im[] = $data[$i + 1];
            }
        } else {
            $re = $data;
            $im = null;
        }
        $reCol = self::toColumnMajor($re, $rows, $cols);
        $prPayload = self::packNumeric($reCol, $miType);
        $sub = self::arrayFlags($mxClass, $complex ? self::FLAG_COMPLEX : 0)
            . self::dimsElement([$rows, $cols])
            . self::nameElement($name)
            . self::element($miType, $prPayload);
        if ($complex) {
            $imCol = self::toColumnMajor($im, $rows, $cols);
            $sub .= self::element($miType, self::packNumeric($imCol, $miType));
        }

        return self::element(self::MI_MATRIX, $sub);
    }

    /** Encode a char matrix (scipy stores char payload as UTF-8 bytes, class mxCHAR). */
    private static function writeChar(string $name, int $rows, int $cols, $data): string
    {
        if (is_string($data)) {
            $codes = array_values(unpack('C*', $data) ?: []);
        } else {
            $codes = array_map('intval', $data);
        }
        $colCodes = self::toColumnMajor($codes, $rows, $cols);
        $payload = '';
        foreach ($colCodes as $c) {
            $payload .= chr($c & 0xFF);
        }
        $sub = self::arrayFlags(self::MX_CHAR, 0)
            . self::dimsElement([$rows, $cols])
            . self::nameElement($name)
            . self::element(self::MI_UTF8, $payload);

        return self::element(self::MI_MATRIX, $sub);
    }

    /** Reorder a flat C/row-major list of $rows x $cols into column-major (Fortran) order. */
    private static function toColumnMajor(array $flat, int $rows, int $cols): array
    {
        if ($rows * $cols === 0) {
            return [];
        }
        $out = [];
        for ($c = 0; $c < $cols; $c++) {
            for ($r = 0; $r < $rows; $r++) {
                $out[] = $flat[$r * $cols + $c];
            }
        }

        return $out;
    }

    private static function packNumeric(array $vals, int $miType): string
    {
        $fmt = [
            self::MI_DOUBLE => 'e', self::MI_SINGLE => 'g', self::MI_INT8 => 'c', self::MI_UINT8 => 'C',
            self::MI_INT16 => 's', self::MI_UINT16 => 'v', self::MI_INT32 => 'l', self::MI_UINT32 => 'V',
            self::MI_INT64 => 'q', self::MI_UINT64 => 'P',
        ][$miType] ?? null;
        if ($fmt === null) {
            throw new TesseroException("savemat: cannot pack type {$miType}");
        }
        $s = '';
        foreach ($vals as $x) {
            $s .= ($miType === self::MI_DOUBLE || $miType === self::MI_SINGLE) ? pack($fmt, (float) $x) : pack($fmt, (int) $x);
        }

        return $s;
    }

    private static function arrayFlags(int $mxClass, int $flags): string
    {
        $word = ($flags << 8) | ($mxClass & 0xFF);

        return self::element(self::MI_UINT32, pack('VV', $word, 0));
    }

    private static function dimsElement(array $dims): string
    {
        $s = '';
        foreach ($dims as $d) {
            $s .= pack('l', $d);
        }

        return self::element(self::MI_INT32, $s);
    }

    private static function nameElement(string $name): string
    {
        return self::element(self::MI_INT8, $name);
    }

    /** Build an element (tag + data), choosing the small-data-element form when the payload is <=4 bytes. */
    private static function element(int $type, string $data): string
    {
        $n = strlen($data);
        if ($n <= 4 && $n > 0) {                       // small data element
            $tag = pack('V', ($n << 16) | ($type & 0xFFFF));

            return $tag . str_pad($data, 4, "\x00");
        }
        $tag = pack('VV', $type, $n);
        $pad = (8 - ($n % 8)) % 8;

        return $tag . $data . str_repeat("\x00", $pad);
    }
}
