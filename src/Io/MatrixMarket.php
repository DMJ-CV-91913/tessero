<?php

declare(strict_types=1);

namespace Tessero\Io;

use Tessero\Exceptions\TesseroException;

/**
 * Matrix Market exchange format (scipy.io.mmread / mmwrite / mminfo), pure PHP.
 *
 * The NIST Matrix Market format is a text container for a single matrix. The first line is the banner
 *   %%MatrixMarket matrix <array|coordinate> <real|integer|complex|pattern> <symmetry>
 * followed by optional `%` comment lines, a size line, and the data. Two storage shapes:
 *   - array      dense, column-major: one scalar (or `re im` pair for complex) per line, M*N values.
 *   - coordinate sparse: a size line `M N NNZ` then NNZ lines `i j value` (1-indexed row/col).
 * Symmetry (symmetric / skew-symmetric / hermitian) mirrors the stored lower triangle onto the upper one.
 *
 * read()  parses scipy-written (and hand-written) files into {rows, cols, format, field, symmetry, data}
 *         where `data` is the dense matrix as a row-major PHP array of rows (complex entries are [re, im]).
 * write() emits bytes scipy.io.mmread accepts, matching scipy's banner, size line and column-major /
 *         coordinate ordering, so a round-trip is exact.
 *
 * Backend-agnostic: it only manipulates PHP strings and numeric arrays, so FFI and ext behave identically.
 */
final class MatrixMarket
{
    /**
     * Parse Matrix Market text. The dense result may hold at most $maxElements entries (see IoGuard);
     * the size line is checked against that cap and against the data present before allocating.
     *
     * @return array{rows:int, cols:int, format:string, field:string, symmetry:string, data:array<int,array<int,float|int|array{0:float,1:float}>>}
     */
    public static function read(string $text, int $maxElements = IoGuard::DEFAULT_MAX_ELEMENTS): array
    {
        IoGuard::checkLimit($maxElements, 'maxElements', 'mmread');
        $lines = preg_split('/\r\n|\r|\n/', $text);
        if ($lines === false || $lines === []) {
            throw new TesseroException('mmread: empty input');
        }
        $banner = array_shift($lines);
        if (! str_starts_with(strtolower(trim((string) $banner)), '%%matrixmarket')) {
            throw new TesseroException('mmread: missing %%MatrixMarket banner');
        }
        $tok = preg_split('/\s+/', trim((string) $banner));
        $format = strtolower($tok[2] ?? 'array');       // %%MatrixMarket matrix <format> <field> <symmetry>
        $field = strtolower($tok[3] ?? 'real');
        $symmetry = strtolower($tok[4] ?? 'general');

        $i = 0;
        $n = count($lines);
        while ($i < $n && (trim($lines[$i]) === '' || $lines[$i][0] === '%')) {   // skip comments/blank
            $i++;
        }
        if ($i >= $n) {
            throw new TesseroException('mmread: missing size line');
        }
        $size = preg_split('/\s+/', trim($lines[$i]));
        $i++;
        $rows = (int) $size[0];
        $cols = (int) ($size[1] ?? -1);
        // The dense result is rows x cols whatever the format, so the declared size is checked before
        // anything is allocated: against the cap, and (array format) against the lines actually present.
        if ($rows < 0 || $cols < 0) {
            throw new TesseroException('mmread: invalid size line');
        }
        if ($rows > 0 && $cols > intdiv($maxElements, $rows)) {
            throw new TesseroException("mmread: a {$rows}x{$cols} matrix exceeds {$maxElements} elements (\$maxElements)");
        }
        if ($format === 'array' && $rows * $cols > $n - $i) {
            throw new TesseroException('mmread: truncated array data');
        }
        if ($format === 'coordinate') {
            $nnz = (int) ($size[2] ?? 0);
            if ($nnz < 0 || $nnz > $n - $i) {
                throw new TesseroException('mmread: truncated coordinate data');
            }
        }

        $isComplex = $field === 'complex';
        $parseVal = static function (array $t, int $off) use ($isComplex, $field): float|int|array {
            if ($isComplex) {
                return [(float) $t[$off], (float) $t[$off + 1]];
            }

            return $field === 'integer' ? (int) $t[$off] : (float) $t[$off];
        };
        $zero = $isComplex ? [0.0, 0.0] : ($field === 'integer' ? 0 : 0.0);

        $data = [];                                      // dense output, row-major
        for ($r = 0; $r < $rows; $r++) {
            $data[$r] = array_fill(0, $cols, $zero);
        }

        if ($format === 'array') {
            for ($c = 0; $c < $cols; $c++) {             // column-major fill
                for ($r = 0; $r < $rows; $r++) {
                    while ($i < $n && trim($lines[$i]) === '') {
                        $i++;
                    }
                    if ($i >= $n) {
                        throw new TesseroException('mmread: truncated array data');
                    }
                    $t = preg_split('/\s+/', trim($lines[$i]));
                    $i++;
                    $v = $parseVal($t, 0);
                    $data[$r][$c] = $v;
                    if ($symmetry !== 'general' && $r > $c) {
                        $data[$c][$r] = self::conjugate($v, $symmetry, $isComplex);
                    }
                }
            }
        } elseif ($format === 'coordinate') {
            for ($k = 0; $k < $nnz; $k++) {
                while ($i < $n && trim($lines[$i]) === '') {
                    $i++;
                }
                if ($i >= $n) {
                    throw new TesseroException('mmread: truncated coordinate data');
                }
                $t = preg_split('/\s+/', trim($lines[$i]));
                $i++;
                $r = (int) $t[0] - 1;
                $c = (int) ($t[1] ?? 0) - 1;
                if ($r < 0 || $r >= $rows || $c < 0 || $c >= $cols) {
                    throw new TesseroException('mmread: coordinate entry outside the matrix');
                }
                $v = $field === 'pattern' ? ($field === 'integer' ? 1 : 1.0) : $parseVal($t, 2);
                $data[$r][$c] = $v;
                if ($symmetry !== 'general' && $r !== $c) {
                    $data[$c][$r] = self::conjugate($v, $symmetry, $isComplex);
                }
            }
        } else {
            throw new TesseroException("mmread: unknown format '{$format}'");
        }

        return ['rows' => $rows, 'cols' => $cols, 'format' => $format,
            'field' => $field, 'symmetry' => $symmetry, 'data' => $data];
    }

    /** The value stored at the mirror position for a non-general symmetry. */
    private static function conjugate(float|int|array $v, string $symmetry, bool $isComplex): float|int|array
    {
        return match ($symmetry) {
            'skew-symmetric' => $isComplex ? [-$v[0], -$v[1]] : -$v,
            'hermitian' => $isComplex ? [$v[0], -$v[1]] : $v,
            default => $v,                                // symmetric
        };
    }

    /**
     * mminfo: inspect the header/size without reading the body.
     *
     * @return array{0:int,1:int,2:int,3:string,4:string,5:string} rows, cols, entries, format, field, symmetry
     */
    public static function info(string $text): array
    {
        $lines = preg_split('/\r\n|\r|\n/', $text);
        $banner = trim((string) array_shift($lines));
        $tok = preg_split('/\s+/', $banner);
        $format = strtolower($tok[2] ?? 'array');
        $field = strtolower($tok[3] ?? 'real');
        $symmetry = strtolower($tok[4] ?? 'general');
        $i = 0;
        $n = count($lines);
        while ($i < $n && (trim($lines[$i]) === '' || ($lines[$i][0] ?? '') === '%')) {
            $i++;
        }
        $size = preg_split('/\s+/', trim($lines[$i] ?? ''));
        $rows = (int) ($size[0] ?? 0);
        $cols = (int) ($size[1] ?? 0);
        $entries = $format === 'coordinate' ? (int) ($size[2] ?? 0) : $rows * $cols;

        return [$rows, $cols, $entries, $format, $field, $symmetry];
    }

    /**
     * Serialise a dense row-major matrix to Matrix Market text, matching scipy.io.mmwrite's layout so
     * scipy.io.mmread reads it back exactly.
     *
     * @param array<int,array<int,float|int|array{0:float,1:float}>> $data row-major rows (complex entries [re, im])
     */
    public static function write(array $data, string $format = 'array', string $field = 'real', string $symmetry = 'general'): string
    {
        $rows = count($data);
        $cols = $rows > 0 ? count($data[0]) : 0;
        $isComplex = $field === 'complex';

        $fmtScalar = static function (float|int|array $v) use ($isComplex, $field): string {
            if ($isComplex) {
                return self::num((float) $v[0]) . ' ' . self::num((float) $v[1]);
            }
            if ($field === 'integer') {
                return (string) (int) $v;
            }

            return self::num((float) $v);
        };

        $out = "%%MatrixMarket matrix {$format} {$field} {$symmetry}\n%\n";
        if ($format === 'array') {
            $out .= "{$rows} {$cols}\n";
            for ($c = 0; $c < $cols; $c++) {             // column-major
                for ($r = 0; $r < $rows; $r++) {
                    $out .= $fmtScalar($data[$r][$c]) . "\n";
                }
            }
        } elseif ($format === 'coordinate') {
            $entries = [];
            for ($r = 0; $r < $rows; $r++) {
                for ($c = 0; $c < $cols; $c++) {
                    $v = $data[$r][$c];
                    $nz = $isComplex ? ($v[0] != 0.0 || $v[1] != 0.0) : ($v != 0);
                    if ($nz) {
                        $entries[] = [$r, $c, $v];
                    }
                }
            }
            $out .= "{$rows} {$cols} " . count($entries) . "\n";
            foreach ($entries as [$r, $c, $v]) {
                $out .= ($r + 1) . ' ' . ($c + 1) . ' ' . $fmtScalar($v) . "\n";
            }
        } else {
            throw new TesseroException("mmwrite: unknown format '{$format}'");
        }

        return $out;
    }

    /** Format a float the way scipy writes Matrix Market values ("%.*g" repr, e.g. 3 not 3.0). */
    private static function num(float $v): string
    {
        if (is_nan($v)) {
            return 'nan';
        }
        if (is_infinite($v)) {
            return $v > 0 ? 'inf' : '-inf';
        }
        $s = sprintf('%.16g', $v);

        return $s === '' ? '0' : $s;
    }
}
