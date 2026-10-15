<?php

declare(strict_types=1);

namespace Tessero\Io;

use Tessero\Exceptions\TesseroException;

/**
 * Harwell-Boeing exchange format (scipy.io.hb_read / hb_write), pure PHP.
 *
 * Harwell-Boeing is a fixed-column, Fortran-formatted text container for a single sparse matrix stored
 * in compressed sparse column (CSC) form. scipy supports the common subset: assembled, non-symmetric
 * ("unsymmetric"), real (or integer) matrices with exponential-format values and integer pointers/indices.
 * The file is four header lines followed by three Fortran-formatted blocks (column pointers, row indices,
 * values), all 1-indexed (Fortran one-based).
 *
 *   Line 1: title (72 chars) + key (8 chars)
 *   Line 2: total_nlines, pointer_nlines, indices_nlines, values_nlines   (each %14d)
 *   Line 3: mxtype (ljust 14) + nrows + ncols + nnz + 0                    (mxtype + 4 x %14d)
 *   Line 4: pointer-format (ljust 16) + indices-format (ljust 16) + values-format (ljust 20)
 *
 * The Fortran formats are generated exactly as scipy does: integer blocks use "(<repeat>I<width>)" with
 * width = digits + 1 (one extra for sign/space) and repeat = 80 // width; float blocks always resolve to
 * "(3E25.16)" / "%24.16E" for float64. Fortran (and C/Python) emit a two-digit minimum exponent (E+00);
 * PHP %E emits a one-digit minimum (E+0), so the exponent is re-padded to two digits here.
 *
 * read()  parses an HB file into {nrows, ncols, nnz, indptr[], indices[], data[]} (0-indexed CSC).
 * write() emits bytes scipy.io.hb_read reads back identically (and matches scipy hb_write byte-for-byte
 *         for real/integer unsymmetric assembled matrices).
 *
 * Backend-agnostic: pure string formatting and numeric arrays, so FFI and ext behave identically.
 */
final class HarwellBoeing
{
    /**
     * Parse an HB-format string into a 0-indexed CSC description.
     *
     * @return array{nrows:int, ncols:int, nnz:int, indptr:array<int,int>, indices:array<int,int>,
     *               data:array<int,float|int>, mxtype:string, title:string, key:string}
     */
    public static function read(string $text): array
    {
        $lines = preg_split('/\r\n|\r|\n/', $text);
        if ($lines === false || count($lines) < 4) {
            throw new TesseroException('hb_read: expected at least 4 header lines');
        }

        $l1 = $lines[0];
        if (strlen(rtrim($l1, "\n")) < 72) {
            throw new TesseroException('hb_read: first line shorter than 72 characters');
        }
        $title = rtrim(substr($l1, 0, 72));
        $key = rtrim(substr($l1, 72));

        $l2 = $lines[1];
        $totalNlines = (int) trim(substr($l2, 0, 14));
        $pointerNlines = (int) trim(substr($l2, 14, 14));
        $indicesNlines = (int) trim(substr($l2, 28, 14));
        $valuesNlines = (int) trim(substr($l2, 42, 14));
        $rhs = trim(substr($l2, 56, 16));
        if ($rhs !== '' && (int) $rhs !== 0) {
            throw new TesseroException('hb_read: right-hand sides not supported');
        }

        $l3 = $lines[2];
        $mxtype = strtoupper(substr($l3, 0, 3));
        if ($mxtype[0] !== 'R' && $mxtype[0] !== 'I') {
            throw new TesseroException("hb_read: only real/integer matrices supported (got {$mxtype})");
        }
        if ($mxtype[1] !== 'U') {
            throw new TesseroException("hb_read: only unsymmetric matrices supported (got {$mxtype})");
        }
        if ($mxtype[2] !== 'A') {
            throw new TesseroException("hb_read: only assembled matrices supported (got {$mxtype})");
        }
        $nrows = (int) trim(substr($l3, 14, 14));
        $ncols = (int) trim(substr($l3, 28, 14));
        $nnz = (int) trim(substr($l3, 42, 14));

        $formats = preg_split('/\s+/', trim($lines[3]));
        if ($formats === false || count($formats) !== 3) {
            throw new TesseroException('hb_read: expected 3 fortran formats on line 4');
        }
        $ptrFmt = self::parseFortran($formats[0]);
        $indFmt = self::parseFortran($formats[1]);
        $valFmt = self::parseFortran($formats[2]);
        if ($ptrFmt['kind'] !== 'int' || $indFmt['kind'] !== 'int') {
            throw new TesseroException('hb_read: pointer/indices must be integer format');
        }

        $pos = 4;
        [$indptr, $pos] = self::readBlock($lines, $pos, $pointerNlines, false);
        [$indices, $pos] = self::readBlock($lines, $pos, $indicesNlines, false);
        [$data, $pos] = self::readBlock($lines, $pos, $valuesNlines, $valFmt['kind'] === 'exp');

        $indptr = array_map(static fn ($v) => (int) $v - 1, $indptr);
        $indices = array_map(static fn ($v) => (int) $v - 1, $indices);
        if ($valFmt['kind'] === 'int') {
            $data = array_map(static fn ($v) => (int) $v, $data);
        }

        return [
            'nrows' => $nrows, 'ncols' => $ncols, 'nnz' => $nnz,
            'indptr' => $indptr, 'indices' => $indices, 'data' => $data,
            'mxtype' => $mxtype, 'title' => $title, 'key' => $key,
        ];
    }

    /**
     * @return array{0:array<int,float|int|string>, 1:int}
     */
    private static function readBlock(array $lines, int $pos, int $nlines, bool $float): array
    {
        $out = [];
        for ($k = 0; $k < $nlines; $k++) {
            if (! isset($lines[$pos])) {
                throw new TesseroException('hb_read: truncated data block');
            }
            $toks = preg_split('/\s+/', trim($lines[$pos]));
            foreach ($toks as $t) {
                if ($t === '') {
                    continue;
                }
                $out[] = $float ? (float) $t : $t;
            }
            $pos++;
        }

        return [$out, $pos];
    }

    /**
     * @return array{kind:string, repeat:int, width:int, significand:int}
     */
    private static function parseFortran(string $s): array
    {
        if (preg_match('/^\((\d*)I(\d+)(?:\.(\d+))?\)$/i', $s, $m)) {
            return ['kind' => 'int', 'repeat' => $m[1] === '' ? 1 : (int) $m[1], 'width' => (int) $m[2], 'significand' => 0];
        }
        if (preg_match('/^\((\d*)E(\d+)\.(\d+)(?:E\d+)?\)$/i', $s, $m)) {
            return ['kind' => 'exp', 'repeat' => $m[1] === '' ? 1 : (int) $m[1], 'width' => (int) $m[2], 'significand' => (int) $m[3]];
        }
        throw new TesseroException("hb_read: unsupported fortran format {$s}");
    }

    /**
     * Write a 0-indexed CSC matrix as HB-format text matching scipy.io.hb_write.
     *
     * @param array<int,int>       $indptr  column pointers, length ncols+1 (0-indexed)
     * @param array<int,int>       $indices row indices (0-indexed)
     * @param array<int,float|int> $data    stored values
     */
    public static function write(int $nrows, int $ncols, array $indptr, array $indices, array $data, string $title = 'Default title', string $key = '0'): string
    {
        $nnz = count($data);

        $isInt = true;
        foreach ($data as $v) {
            if (is_float($v) || (is_string($v) && ! ctype_digit(ltrim((string) $v, '-')))) {
                $isInt = false;
                break;
            }
        }

        $ptrMax = 0;
        foreach ($indptr as $p) {
            $ptrMax = max($ptrMax, $p + 1);
        }
        $indMax = 0;
        foreach ($indices as $i) {
            $indMax = max($indMax, $i + 1);
        }
        $ptrFmt = self::intFormat($ptrMax);
        $indFmt = self::intFormat($indMax);

        if ($isInt) {
            $absMax = 0;
            foreach ($data as $v) {
                $absMax = max($absMax, abs((int) $v));
            }
            $valFmt = self::intFormat(-$absMax);
            $mxtype = 'IUA';
        } else {
            $valFmt = ['repeat' => 3, 'width' => 25, 'significand' => 16, 'kind' => 'exp'];
            $mxtype = 'RUA';
        }

        $ptrNlines = self::nlines(count($indptr), $ptrFmt['repeat']);
        $indNlines = self::nlines(count($indices), $indFmt['repeat']);
        $valNlines = self::nlines(count($data), $valFmt['repeat']);
        $totalNlines = $ptrNlines + $indNlines + $valNlines;

        $header = str_pad(substr($title, 0, 72), 72) . str_pad(substr($key, 0, 8), 8);
        $header .= "\n" . sprintf('%14d%14d%14d%14d', $totalNlines, $ptrNlines, $indNlines, $valNlines);
        $header .= "\n" . str_pad($mxtype, 14) . sprintf('%14d%14d%14d%14d', $nrows, $ncols, $nnz, 0);
        $header .= "\n" . str_pad(self::fortranStr($ptrFmt), 16) . str_pad(self::fortranStr($indFmt), 16) . str_pad(self::fortranStr($valFmt), 20);

        $body = "\n";
        $body .= self::writeIntBlock(array_map(static fn ($v) => (int) $v + 1, $indptr), $ptrNlines, $ptrFmt);
        $body .= self::writeIntBlock(array_map(static fn ($v) => (int) $v + 1, $indices), $indNlines, $indFmt);
        if ($isInt) {
            $body .= self::writeIntBlock(array_map(static fn ($v) => (int) $v, $data), $valNlines, $valFmt);
        } else {
            // scipy prints each value with python_format "%<width-1>.<sig>E" (24 chars for E25.16).
            $body .= self::writeExpBlock(array_map(static fn ($v) => (float) $v, $data), $valNlines, $valFmt);
        }

        return $header . $body;
    }

    private static function intFormat(int $n): array
    {
        $digits = $n === 0 ? 1 : (int) floor(log10(abs($n))) + 1;
        $width = $digits + 1;
        if ($n < 0) {
            $width += 1;
        }
        $repeat = intdiv(80, $width);

        return ['kind' => 'int', 'repeat' => $repeat, 'width' => $width, 'significand' => 0];
    }

    private static function nlines(int $size, int $repeat): int
    {
        $n = intdiv($size, $repeat);
        if ($n * $repeat !== $size) {
            $n++;
        }

        return $n;
    }

    private static function fortranStr(array $fmt): string
    {
        if ($fmt['kind'] === 'int') {
            return '(' . $fmt['repeat'] . 'I' . $fmt['width'] . ')';
        }

        return '(' . $fmt['repeat'] . 'E' . $fmt['width'] . '.' . $fmt['significand'] . ')';
    }

    /** @param array<int,int> $ar */
    private static function writeIntBlock(array $ar, int $nlines, array $fmt): string
    {
        $repeat = $fmt['repeat'];
        $width = $fmt['width'];
        $out = '';
        $size = count($ar);
        $full = ($nlines - 1) * $repeat;
        for ($k = 0; $k < $full; $k += $repeat) {
            $line = '';
            for ($j = 0; $j < $repeat; $j++) {
                $line .= sprintf('%' . $width . 'd', $ar[$k + $j]);
            }
            $out .= $line . "\n";
        }
        if ($size > $full) {
            $line = '';
            for ($k = $full; $k < $size; $k++) {
                $line .= sprintf('%' . $width . 'd', $ar[$k]);
            }
            $out .= $line . "\n";
        }

        return $out;
    }

    /** @param array<int,float> $ar */
    private static function writeExpBlock(array $ar, int $nlines, array $fmt): string
    {
        $repeat = $fmt['repeat'];
        $out = '';
        $size = count($ar);
        $full = ($nlines - 1) * $repeat;
        for ($k = 0; $k < $full; $k += $repeat) {
            $line = '';
            for ($j = 0; $j < $repeat; $j++) {
                $line .= self::expValue($ar[$k + $j], $fmt['width'], $fmt['significand']);
            }
            $out .= $line . "\n";
        }
        if ($size > $full) {
            $line = '';
            for ($k = $full; $k < $size; $k++) {
                $line .= self::expValue($ar[$k], $fmt['width'], $fmt['significand']);
            }
            $out .= $line . "\n";
        }

        return $out;
    }

    private static function expValue(float $v, int $width, int $sig): string
    {
        $s = sprintf('%.' . $sig . 'E', $v);
        $s = preg_replace_callback('/E([+-])(\d+)$/', static function ($m) {
            $exp = $m[2];
            if (strlen($exp) < 2) {
                $exp = str_pad($exp, 2, '0', STR_PAD_LEFT);
            }

            return 'E' . $m[1] . $exp;
        }, $s);

        // scipy's python_format is "%<width-1>.<sig>E", i.e. the printed field is width-1 chars.
        return str_pad($s, $width - 1, ' ', STR_PAD_LEFT);
    }
}
