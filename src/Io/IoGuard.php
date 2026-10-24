<?php

declare(strict_types=1);

namespace Tessero\Io;

use Tessero\Exceptions\TesseroException;
use ZipArchive;

/**
 * Limits shared by the file readers in Tessero\Io.
 *
 * Every reader treats its input as untrusted: a compressed member or zlib stream is inflated only up to
 * $maxBytes (default DEFAULT_MAX_BYTES, 256 MiB), counts declared in a header are checked against the
 * bytes actually present before any loop or allocation, and the number of decoded elements is capped
 * (DEFAULT_MAX_ELEMENTS). Raise the limits per call for files you trust; never derive them from request
 * input.
 *
 * Paths given to the loaders and savers must be local files. Stream wrappers (http://, phar://,
 * php://filter, data:, ...) are refused, as for memory maps. Never build a path from request input.
 *
 * @internal
 */
final class IoGuard
{
    /** Default cap on decompressed bytes (a whole .npz, one MAT/IDL file). */
    public const DEFAULT_MAX_BYTES = 256 * 1024 * 1024;

    /** Default cap on the number of values a pure-PHP reader decodes into PHP arrays. */
    public const DEFAULT_MAX_ELEMENTS = 64 * 1024 * 1024;

    /** Members read from one .npz archive at most. */
    public const MAX_ZIP_ENTRIES = 4096;

    private function __construct()
    {
    }

    /**
     * Refuse stream wrappers and malformed paths; returns the path unchanged.
     *
     * @throws \ValueError
     */
    public static function localPath(string $path, string $what): string
    {
        if ($path === '' || str_contains($path, "\0")) {
            throw new \ValueError("{$what}: the path cannot be empty or contain NUL bytes");
        }
        if (self::isStreamWrapper($path)) {
            throw new \ValueError("{$what}: '{$path}' must be a local file path (stream wrappers are refused)");
        }

        return $path;
    }

    /** True for "scheme://..." and for "scheme:..." when scheme is a registered stream wrapper (data:). */
    public static function isStreamWrapper(string $path): bool
    {
        if (str_contains($path, '://')) {
            return true;
        }
        if (preg_match('/^([A-Za-z][A-Za-z0-9+.\-]+):/', $path, $m) === 1) {   // 2+ chars: not a drive letter
            return in_array(strtolower($m[1]), array_map('strtolower', stream_get_wrappers()), true);
        }

        return false;
    }

    public static function checkLimit(int $limit, string $name, string $what): void
    {
        if ($limit < 1) {
            throw new \ValueError("{$what}: \${$name} must be at least 1");
        }
    }

    /**
     * Every member of a zip archive as name => bytes, with the member count capped and the total
     * uncompressed size checked against $maxBytes BEFORE anything is inflated.
     *
     * @return array<string, string>
     */
    public static function zipMembers(string $path, int $maxBytes, string $what): array
    {
        self::checkLimit($maxBytes, 'maxBytes', $what);
        $zip = new ZipArchive();
        if ($zip->open($path, ZipArchive::RDONLY) !== true) {
            throw new TesseroException("{$what}: could not open {$path}.");
        }
        try {
            $n = $zip->numFiles;
            if ($n > self::MAX_ZIP_ENTRIES) {
                throw new TesseroException("{$what}: archive has {$n} members (at most " . self::MAX_ZIP_ENTRIES . ' are read).');
            }
            $sizes = [];
            $total = 0;
            for ($i = 0; $i < $n; $i++) {
                $st = $zip->statIndex($i);
                if ($st === false || ! is_int($st['size'] ?? null) || $st['size'] < 0) {
                    throw new TesseroException("{$what}: cannot read member {$i} of {$path}.");
                }
                $total += $st['size'];
                if ($total > $maxBytes) {
                    throw new TesseroException("{$what}: archive inflates to more than {$maxBytes} bytes (\$maxBytes); refusing to decompress it.");
                }
                $sizes[$i] = [(string) $st['name'], $st['size']];
            }
            $out = [];
            foreach ($sizes as $i => [$name, $size]) {
                // read at most the declared size: a member that inflates further is cut off and rejected below
                $data = $size === 0 ? '' : $zip->getFromIndex($i, $size);
                if (! is_string($data) || strlen($data) !== $size) {
                    throw new TesseroException("{$what}: member '{$name}' of {$path} is corrupt (size mismatch).");
                }
                $out[$name] = $data;
            }

            return $out;
        } finally {
            $zip->close();
        }
    }

    /**
     * zlib-inflate at most $maxBytes (what is left of the caller's budget). $format: 'zlib' (gzuncompress,
     * RFC 1950) or 'any' (zlib_decode: zlib, gzip or raw deflate).
     */
    public static function inflate(string $data, int $maxBytes, string $what, string $format = 'zlib'): string
    {
        if ($maxBytes < 1) {
            throw new TesseroException("{$what}: compressed data inflates to more than the \$maxBytes limit.");
        }
        error_clear_last();
        $out = $format === 'zlib' ? @gzuncompress($data, $maxBytes) : @zlib_decode($data, $maxBytes);
        if ($out === false) {
            $err = error_get_last()['message'] ?? '';
            if (str_contains($err, 'insufficient memory')) {
                throw new TesseroException("{$what}: compressed data inflates to more than {$maxBytes} bytes (\$maxBytes).");
            }
            throw new TesseroException("{$what}: zlib inflate failed");
        }
        if (strlen($out) > $maxBytes) {                         // zlib_decode may overshoot its limit slightly
            throw new TesseroException("{$what}: compressed data inflates to more than {$maxBytes} bytes (\$maxBytes).");
        }

        return $out;
    }
}
