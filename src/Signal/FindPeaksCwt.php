<?php

declare(strict_types=1);

namespace Tessero\Signal;

/**
 * scipy.signal.find_peaks_cwt: continuous-wavelet-transform peak detection.
 *
 * Pure PHP, array-in / int[]-out, deterministic (no RNG, no kernel call). Reproduces the scipy 1.17
 * pipeline exactly:
 *   1. _cwt: for each width w, convolve the signal with the (reversed) Ricker wavelet of
 *      N = min(10*w, len(data)) points via direct convolution, mode='same' (scipy.signal.convolve
 *      selects the direct method for these sizes, so this is bit-faithful).
 *   2. _identify_ridge_lines: relative maxima per row (strict greater, order 1, 'clip' edges),
 *      linked across decreasing rows under max_distances with a gap counter capped at gap_thresh.
 *   3. _filter_ridge_lines: drop ridge lines shorter than min_length or with SNR below min_snr,
 *      where the noise floor is scipy.stats.scoreatpercentile(window, noise_perc) over a window of
 *      row 0 around each column (fraction interpolation).
 *
 * Defaults mirror scipy: gap_thresh=ceil(widths[0]), max_distances=widths/4, wavelet=Ricker,
 * min_length=ceil(nrows/4), window_size=ceil(ncols/20), min_snr=1, noise_perc=10.
 *
 * All arithmetic is reimplemented here so one class serves both the FFI and ext backends.
 */
final class FindPeaksCwt
{
    /**
     * scipy.signal.find_peaks_cwt(vector, widths, ...). Returns the sorted peak indices.
     *
     * @param  float[]       $vector
     * @param  float[]       $widths
     * @param  float[]|null  $maxDistances
     * @return int[]
     */
    public static function find(
        array $vector,
        array $widths,
        ?array $maxDistances = null,
        ?float $gapThresh = null,
        ?int $minLength = null,
        float $minSnr = 1.0,
        float $noisePerc = 10.0,
        ?int $windowSize = null,
    ): array {
        $vector = array_values(array_map('floatval', $vector));
        $widths = array_values(array_map('floatval', $widths));

        if ($gapThresh === null) {
            $gapThresh = ceil($widths[0]);
        }
        if ($maxDistances === null) {
            $maxDistances = array_map(static fn ($w) => $w / 4.0, $widths);
        } else {
            $maxDistances = array_values(array_map('floatval', $maxDistances));
        }

        $cwt = self::cwt($vector, $widths);
        $ridgeLines = self::identifyRidgeLines($cwt, $maxDistances, $gapThresh);
        $filtered = self::filterRidgeLines($cwt, $ridgeLines, $windowSize, $minLength, $minSnr, $noisePerc);

        $maxLocs = [];
        foreach ($filtered as $line) {
            $maxLocs[] = $line[1][0];   // columns[0] of each surviving ridge
        }
        sort($maxLocs, SORT_NUMERIC);

        return array_map('intval', $maxLocs);
    }

    /**
     * Ricker ("mexican hat") wavelet of `points` samples at scale `a`; matches scipy._peak_finding._ricker.
     *
     * @return float[]
     */
    private static function ricker(int $points, float $a): array
    {
        $A = 2.0 / (sqrt(3.0 * $a) * pow(M_PI, 0.25));
        $wsq = $a * $a;
        $out = [];
        $center = ($points - 1.0) / 2.0;
        for ($i = 0; $i < $points; $i++) {
            $x = (float) $i - $center;
            $xsq = $x * $x;
            $mod = 1.0 - $xsq / $wsq;
            $gauss = exp(-$xsq / (2.0 * $wsq));
            $out[] = $A * $mod * $gauss;
        }

        return $out;
    }

    /**
     * CWT matrix: one row per width. Each row is convolve(data, conj(ricker(N,w))[::-1], 'same').
     * The Ricker wavelet is real, so conj is a no-op and conj(...)[::-1] is just the reversal.
     *
     * @param  float[]  $data
     * @param  float[]  $widths
     * @return float[][]
     */
    private static function cwt(array $data, array $widths): array
    {
        $n = count($data);
        $out = [];
        foreach ($widths as $width) {
            $N = (int) min(10.0 * $width, (float) $n);
            $wavelet = array_reverse(self::ricker($N, $width));
            $out[] = self::convolveSame($data, $wavelet);
        }

        return $out;
    }

    /**
     * Direct linear convolution, mode='same': the centered len(a) slice of the full convolution
     * (length len(a)+len(v)-1), start = (L - len(a)) // 2. Matches scipy.signal.convolve's direct path.
     *
     * @param  float[]  $a
     * @param  float[]  $v
     * @return float[]
     */
    private static function convolveSame(array $a, array $v): array
    {
        $na = count($a);
        $nv = count($v);
        $L = $na + $nv - 1;
        $full = array_fill(0, $L, 0.0);
        for ($i = 0; $i < $na; $i++) {
            $ai = $a[$i];
            for ($j = 0; $j < $nv; $j++) {
                $full[$i + $j] += $ai * $v[$j];
            }
        }
        $start = intdiv($L - $na, 2);

        return array_slice($full, $start, $na);
    }

    /**
     * Boolean "strict relative maximum along the row" with order 1 and 'clip' edges: a column c is a
     * relative max iff data[c] > data[c-1] and data[c] > data[c+1], where out-of-range neighbours are
     * clipped to the edge value (so endpoints compare against themselves -> never strictly greater).
     *
     * @param  float[]  $row
     * @return bool[]
     */
    private static function boolRelExtremaRow(array $row): array
    {
        $n = count($row);
        $res = array_fill(0, $n, true);
        for ($i = 0; $i < $n; $i++) {
            $main = $row[$i];
            $plus = $row[min($i + 1, $n - 1)];
            $minus = $row[max($i - 1, 0)];
            $res[$i] = ($main > $plus) && ($main > $minus);
        }

        return $res;
    }

    /**
     * scipy._peak_finding._identify_ridge_lines. Returns a list of [rowsAsc, colsAsc] pairs.
     *
     * @param  float[][]  $matr
     * @param  float[]    $maxDistances
     * @return array<int, array{0: int[], 1: int[]}>
     */
    private static function identifyRidgeLines(array $matr, array $maxDistances, float $gapThresh): array
    {
        $nrows = count($matr);
        if ($nrows === 0) {
            return [];
        }
        $ncols = count($matr[0]);

        // all_max_cols[row] = bool[] per column
        $allMaxCols = [];
        for ($r = 0; $r < $nrows; $r++) {
            $allMaxCols[$r] = self::boolRelExtremaRow($matr[$r]);
        }

        // Highest row index with any relative maximum.
        $startRow = -1;
        for ($r = $nrows - 1; $r >= 0; $r--) {
            if (in_array(true, $allMaxCols[$r], true)) {
                $startRow = $r;
                break;
            }
        }
        if ($startRow === -1) {
            return [];
        }

        // Seed one ridge line per relative-max column of the start row.
        // Each line = [rows, cols, gap].
        $ridgeLines = [];
        for ($c = 0; $c < $ncols; $c++) {
            if ($allMaxCols[$startRow][$c]) {
                $ridgeLines[] = [[$startRow], [$c], 0];
            }
        }
        $finalLines = [];

        for ($row = $startRow - 1; $row >= 0; $row--) {
            $thisMaxCols = [];
            for ($c = 0; $c < $ncols; $c++) {
                if ($allMaxCols[$row][$c]) {
                    $thisMaxCols[] = $c;
                }
            }

            foreach ($ridgeLines as &$line) {
                $line[2]++;
            }
            unset($line);

            // Snapshot the current tail columns (scipy reads prev_ridge_cols once per row).
            $prevRidgeCols = [];
            foreach ($ridgeLines as $line) {
                $cols = $line[1];
                $prevRidgeCols[] = $cols[count($cols) - 1];
            }

            foreach ($thisMaxCols as $col) {
                $matchIdx = -1;
                if (count($prevRidgeCols) > 0) {
                    // np.argmin of |col - prev|: first index of the minimum.
                    $best = -1;
                    $bestDiff = INF;
                    foreach ($prevRidgeCols as $k => $pc) {
                        $d = abs($col - $pc);
                        if ($d < $bestDiff) {
                            $bestDiff = $d;
                            $best = $k;
                        }
                    }
                    if ($bestDiff <= $maxDistances[$row]) {
                        $matchIdx = $best;
                    }
                }
                if ($matchIdx >= 0) {
                    $ridgeLines[$matchIdx][1][] = $col;
                    $ridgeLines[$matchIdx][0][] = $row;
                    $ridgeLines[$matchIdx][2] = 0;
                } else {
                    $ridgeLines[] = [[$row], [$col], 0];
                }
            }

            // Retire ridge lines whose gap exceeds gap_thresh (iterate backwards).
            for ($ind = count($ridgeLines) - 1; $ind >= 0; $ind--) {
                if ($ridgeLines[$ind][2] > $gapThresh) {
                    $finalLines[] = $ridgeLines[$ind];
                    array_splice($ridgeLines, $ind, 1);
                }
            }
        }

        $outLines = [];
        foreach (array_merge($finalLines, $ridgeLines) as $line) {
            // Sort by row ascending (rows are appended descending; stable by original order on ties).
            $rows = $line[0];
            $cols = $line[1];
            $idx = range(0, count($rows) - 1);
            usort($idx, static function ($i, $j) use ($rows) {
                return $rows[$i] <=> $rows[$j] ?: $i <=> $j;
            });
            $sortedRows = [];
            $sortedCols = [];
            foreach ($idx as $i) {
                $sortedRows[] = $rows[$i];
                $sortedCols[] = $cols[$i];
            }
            $outLines[] = [$sortedRows, $sortedCols];
        }

        return $outLines;
    }

    /**
     * scipy._peak_finding._filter_ridge_lines.
     *
     * @param  float[][]                              $cwt
     * @param  array<int, array{0: int[], 1: int[]}>  $ridgeLines
     * @return array<int, array{0: int[], 1: int[]}>
     */
    private static function filterRidgeLines(
        array $cwt,
        array $ridgeLines,
        ?int $windowSize,
        ?int $minLength,
        float $minSnr,
        float $noisePerc,
    ): array {
        $nrows = count($cwt);
        $numPoints = count($cwt[0]);
        if ($minLength === null) {
            $minLength = (int) ceil($nrows / 4.0);
        }
        if ($windowSize === null) {
            $windowSize = (int) ceil($numPoints / 20.0);
        }
        $windowSize = (int) $windowSize;
        $hfWindow = intdiv($windowSize, 2);
        $odd = $windowSize % 2;

        $rowOne = $cwt[0];
        $noises = [];
        for ($ind = 0; $ind < $numPoints; $ind++) {
            $start = max($ind - $hfWindow, 0);
            $end = min($ind + $hfWindow + $odd, $numPoints);
            $window = array_slice($rowOne, $start, $end - $start);
            $noises[$ind] = self::scoreAtPercentile($window, $noisePerc);
        }

        $out = [];
        foreach ($ridgeLines as $line) {
            if (count($line[0]) < $minLength) {
                continue;
            }
            $r0 = $line[0][0];
            $c0 = $line[1][0];
            $snr = abs($cwt[$r0][$c0] / $noises[$c0]);
            if ($snr < $minSnr) {
                continue;
            }
            $out[] = $line;
        }

        return $out;
    }

    /**
     * scipy.stats.scoreatpercentile with interpolation_method='fraction' over a 1-D array.
     *
     * @param  float[]  $a
     */
    private static function scoreAtPercentile(array $a, float $per): float
    {
        $sorted = $a;
        sort($sorted, SORT_NUMERIC);
        $n = count($sorted);
        $idx = $per / 100.0 * ($n - 1);
        $i = (int) $idx;        // truncates toward zero; idx >= 0 here
        if ((float) $i === $idx) {
            return $sorted[$i];
        }
        $j = $i + 1;
        $wLo = $j - $idx;
        $wHi = $idx - $i;
        $sumval = $wLo + $wHi;

        return ($sorted[$i] * $wLo + $sorted[$j] * $wHi) / $sumval;
    }
}
