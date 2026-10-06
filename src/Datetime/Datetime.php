<?php

declare(strict_types=1);

namespace Tessero\Datetime;

use Tessero\Exceptions\TesseroException;

/**
 * numpy datetime64 / timedelta64 helpers. Tessero has no datetime dtype, so values are represented exactly as
 * NumPy stores them: a signed int64 count of the given unit since the 1970-01-01 epoch (NaT = INT64_MIN). The
 * calendar maths (business days, parsing, formatting) matches NumPy. Pure PHP, so backend-neutral.
 */
final class Datetime
{
    private const NAT = PHP_INT_MIN;
    private const US = ['D' => 86400000000, 'h' => 3600000000, 'm' => 60000000, 's' => 1000000, 'ms' => 1000, 'us' => 1];

    /** Microseconds since the epoch for an ISO string (UTC, naive). */
    private static function epochMicros(string $s): int
    {
        $d = new \DateTimeImmutable($s, new \DateTimeZone('UTC'));
        return $d->getTimestamp() * 1000000 + (int) $d->format('u');
    }

    /** One datetime64 count at $unit for an ISO string (or NaT). */
    private static function one64(string $s, string $unit): int
    {
        if ($s === 'NaT' || $s === 'nat') {
            return self::NAT;
        }
        if ($unit === 'Y' || $unit === 'M') {
            $d = new \DateTimeImmutable($s, new \DateTimeZone('UTC'));
            $y = (int) $d->format('Y');
            $mo = (int) $d->format('n');
            return $unit === 'Y' ? $y - 1970 : ($y - 1970) * 12 + ($mo - 1);
        }
        $us = self::epochMicros($s);
        if ($unit === 'ns') {
            return $us * 1000;
        }
        if (! isset(self::US[$unit])) {
            throw new TesseroException("datetime64: unsupported unit '{$unit}'.");
        }
        return intdiv($us, self::US[$unit]);
    }

    /**
     * numpy.datetime64: parse an ISO date/time (or array of them) to int64 counts at $unit (default 'D').
     *
     * @return int|list<int>
     */
    public static function datetime64(string|array $value, string $unit = 'D'): int|array
    {
        if (is_array($value)) {
            return array_map(static fn ($v): int => self::one64((string) $v, $unit), array_values($value));
        }
        return self::one64($value, $unit);
    }

    /** numpy.timedelta64: the magnitude as an int64 count of the unit (NaT passes through). */
    public static function timedelta64(int|string $value, string $unit = 's'): int
    {
        if ($value === 'NaT' || $value === 'nat') {
            return self::NAT;
        }
        return (int) $value;
    }

    /**
     * numpy.datetime_data: the (base unit, step) of a datetime64 unit spec such as 's', '10m' or
     * 'datetime64[10m]'.
     *
     * @return array{0: string, 1: int}
     */
    public static function datetimeData(string $dtype): array
    {
        if (preg_match('/\[?(\d*)([A-Za-z]+)\]?$/', $dtype, $m)) {
            return [$m[2], $m[1] === '' ? 1 : (int) $m[1]];
        }
        throw new TesseroException("datetime_data: cannot parse '{$dtype}'.");
    }

    /**
     * numpy.isnat: which counts are NaT (INT64_MIN).
     *
     * @param int|list<int> $counts
     * @return bool|list<bool>
     */
    public static function isnat(int|array $counts): bool|array
    {
        if (is_array($counts)) {
            return array_map(static fn ($v): bool => (int) $v === self::NAT, array_values($counts));
        }
        return $counts === self::NAT;
    }

    /** Day number (datetime64[D]) for a date string. */
    private static function dayNum(string $s): int
    {
        return self::one64($s, 'D');
    }

    /** Monday = 0 .. Sunday = 6 for a day number (1970-01-01 was a Thursday). */
    private static function weekday(int $day): int
    {
        return (($day + 3) % 7 + 7) % 7;
    }

    /** @param list<int> $holidayDays */
    private static function isBusinessDay(int $day, string $weekmask, array $holidayDays): bool
    {
        return $weekmask[self::weekday($day)] === '1' && ! in_array($day, $holidayDays, true);
    }

    /** @param list<string> $holidays */
    private static function holidayDays(array $holidays): array
    {
        return array_map(static fn ($h): int => self::dayNum((string) $h), $holidays);
    }

    /**
     * numpy.is_busday.
     *
     * @param string|list<string> $dates
     * @param list<string> $holidays
     * @return bool|list<bool>
     */
    public static function isBusday(string|array $dates, string $weekmask = '1111100', array $holidays = []): bool|array
    {
        $hd = self::holidayDays($holidays);
        $one = static fn (string $s): bool => self::isBusinessDay(self::dayNum($s), $weekmask, $hd);
        if (is_array($dates)) {
            return array_map(static fn ($d): bool => $one((string) $d), array_values($dates));
        }
        return $one($dates);
    }

    /**
     * numpy.busday_count: business days in the half-open interval [begin, end).
     *
     * @param list<string> $holidays
     */
    public static function busdayCount(string $begin, string $end, string $weekmask = '1111100', array $holidays = []): int
    {
        $a = self::dayNum($begin);
        $b = self::dayNum($end);
        $hd = self::holidayDays($holidays);
        $sign = $a <= $b ? 1 : -1;
        [$lo, $hi] = $a <= $b ? [$a, $b] : [$b, $a];
        $count = 0;
        for ($d = $lo; $d < $hi; $d++) {
            if (self::isBusinessDay($d, $weekmask, $hd)) {
                $count++;
            }
        }
        return $sign * $count;
    }

    /**
     * numpy.busday_offset: offset a date by a number of business days, returning the day number.
     *
     * @param list<string> $holidays
     */
    public static function busdayOffset(string $date, int $offset, string $roll = 'raise', string $weekmask = '1111100', array $holidays = []): int
    {
        $hd = self::holidayDays($holidays);
        $day = self::dayNum($date);
        if (! self::isBusinessDay($day, $weekmask, $hd)) {   // roll to a valid business day first
            $dir = match ($roll) {
                'forward', 'following' => 1,
                'backward', 'preceding' => -1,
                default => throw new TesseroException("busday_offset: {$date} is not a business day (roll='{$roll}')."),
            };
            while (! self::isBusinessDay($day, $weekmask, $hd)) {
                $day += $dir;
            }
        }
        $step = $offset > 0 ? 1 : -1;
        $remaining = abs($offset);
        while ($remaining > 0) {
            $day += $step;
            if (self::isBusinessDay($day, $weekmask, $hd)) {
                $remaining--;
            }
        }
        return $day;
    }

    /**
     * numpy.datetime_as_string: format int64 counts at $unit back to ISO strings.
     *
     * @param int|list<int> $counts
     * @return string|list<string>
     */
    public static function datetimeAsString(int|array $counts, string $unit = 'D'): string|array
    {
        $fmt = static function (int $c) use ($unit): string {
            if ($c === self::NAT) {
                return 'NaT';
            }
            if ($unit === 'D') {
                return (new \DateTimeImmutable('@' . $c * 86400, new \DateTimeZone('UTC')))->format('Y-m-d');
            }
            if (! isset(self::US[$unit]) && $unit !== 'ns') {
                throw new TesseroException("datetime_as_string: unsupported unit '{$unit}'.");
            }
            $us = $unit === 'ns' ? intdiv($c, 1000) : $c * self::US[$unit];
            $sec = intdiv($us, 1000000);
            $d = new \DateTimeImmutable('@' . $sec, new \DateTimeZone('UTC'));
            return $d->format('Y-m-d\TH:i:s');
        };
        if (is_array($counts)) {
            return array_map(static fn ($c): string => $fmt((int) $c), array_values($counts));
        }
        return $fmt($counts);
    }
}
