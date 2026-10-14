<?php

declare(strict_types=1);

namespace Tessero\Io;

use Tessero\Exceptions\TesseroException;

/**
 * RIFF/WAVE audio files (scipy.io.wavfile.read / write), pure PHP.
 *
 * A WAV file is a RIFF container: "RIFF" <size> "WAVE" followed by chunks. The two that matter are
 *   fmt   the format: audio-format tag (1 = integer PCM, 3 = IEEE float), channels, sample rate,
 *         byte rate, block align and bits-per-sample (plus a 2-byte extension size for float);
 *   data  the interleaved samples (little-endian), uint8 / int16 / int32 PCM or float32 / float64.
 * Float files additionally carry a `fact` chunk (sample count), exactly as scipy writes.
 *
 * read()  parses scipy-written (and standard) WAV bytes into {rate, data, channels, dtype, samples}
 *         where `data` is a flat row-major PHP array of samples (interleaved by channel as on disk).
 * write() emits bytes scipy.io.wavfile.read accepts, choosing the PCM / float tag, chunk sizes and the
 *         fact chunk exactly as scipy does, so a round-trip is bit-exact.
 *
 * Backend-agnostic: it only packs/unpacks binary strings and PHP numeric arrays.
 */
final class Wav
{
    private const TYPES = [
        'uint8' => [1, 8], 'int16' => [1, 16], 'int32' => [1, 32],
        'float32' => [3, 32], 'float64' => [3, 64],
    ];

    /**
     * Parse WAV bytes.
     *
     * @return array{rate:int, data:array<int,int|float>, channels:int, dtype:string, samples:int}
     */
    public static function read(string $bytes): array
    {
        if (strlen($bytes) < 12 || substr($bytes, 0, 4) !== 'RIFF' || substr($bytes, 8, 4) !== 'WAVE') {
            throw new TesseroException('wavfile: not a RIFF/WAVE file');
        }
        $pos = 12;
        $len = strlen($bytes);
        $tag = $channels = $rate = $bits = 0;
        $dataBytes = null;

        while ($pos + 8 <= $len) {
            $cid = substr($bytes, $pos, 4);
            $csize = unpack('V', substr($bytes, $pos + 4, 4))[1];
            $body = $pos + 8;
            if ($cid === 'fmt ') {
                $f = unpack('vtag/vch/Vrate/Vbyterate/vblock/vbits', substr($bytes, $body, 16));
                $tag = $f['tag'];
                $channels = $f['ch'];
                $rate = $f['rate'];
                $bits = $f['bits'];
            } elseif ($cid === 'data') {
                $dataBytes = substr($bytes, $body, $csize);
            }
            $pos = $body + $csize + ($csize & 1);          // chunks are word-aligned (pad to even)
        }

        if ($dataBytes === null) {
            throw new TesseroException('wavfile: no data chunk');
        }
        $dtype = self::dtypeFor($tag, $bits);
        $data = self::unpackSamples($dataBytes, $dtype);
        $samples = $channels > 0 ? intdiv(count($data), $channels) : count($data);

        return ['rate' => $rate, 'data' => $data, 'channels' => $channels, 'dtype' => $dtype, 'samples' => $samples];
    }

    /**
     * Serialise interleaved samples to WAV bytes matching scipy.io.wavfile.write.
     *
     * @param array<int,int|float> $data flat interleaved samples (channel-major per frame)
     */
    public static function write(int $rate, array $data, string $dtype, int $channels = 1): string
    {
        if (! isset(self::TYPES[$dtype])) {
            throw new TesseroException("wavfile: unsupported dtype '{$dtype}'");
        }
        [$tag, $bits] = self::TYPES[$dtype];
        $bytesPerSample = intdiv($bits, 8);
        $blockAlign = $bytesPerSample * $channels;
        $byteRate = $rate * $blockAlign;
        $raw = self::packSamples($data, $dtype);
        $isFloat = $tag === 3;

        // fmt chunk: 16 bytes PCM, 18 for float (adds a 2-byte cbSize = 0)
        $fmtBody = pack('vvVVvv', $tag, $channels, $rate, $byteRate, $blockAlign, $bits);
        if ($isFloat) {
            $fmtBody .= pack('v', 0);
        }
        $fmt = 'fmt ' . pack('V', strlen($fmtBody)) . $fmtBody;

        $fact = '';
        if ($isFloat) {                                    // scipy writes a fact chunk for float data
            $nsamples = $channels > 0 ? intdiv(count($data), $channels) : count($data);
            $fact = 'fact' . pack('V', 4) . pack('V', $nsamples);
        }

        $data = 'data' . pack('V', strlen($raw)) . $raw;
        $payload = 'WAVE' . $fmt . $fact . $data;
        $riffSize = strlen($payload);

        return 'RIFF' . pack('V', $riffSize) . $payload;
    }

    private static function dtypeFor(int $tag, int $bits): string
    {
        if ($tag === 3) {
            return $bits === 64 ? 'float64' : 'float32';
        }
        // integer PCM: 8-bit is unsigned, 16/32 are signed
        return match ($bits) {
            8 => 'uint8',
            16 => 'int16',
            32 => 'int32',
            default => throw new TesseroException("wavfile: unsupported PCM width {$bits}"),
        };
    }

    /** @return array<int,int|float> */
    private static function unpackSamples(string $raw, string $dtype): array
    {
        if ($raw === '') {
            return [];
        }

        return match ($dtype) {
            'uint8' => array_values(unpack('C*', $raw)),
            'int16' => self::toSigned(array_values(unpack('v*', $raw)), 16),
            'int32' => self::toSigned(array_values(unpack('V*', $raw)), 32),
            'float32' => array_values(unpack('g*', $raw)),
            'float64' => array_values(unpack('e*', $raw)),
            default => throw new TesseroException("wavfile: cannot unpack {$dtype}"),
        };
    }

    /** @param array<int,int> $u @return array<int,int> */
    private static function toSigned(array $u, int $bits): array
    {
        $mod = 1 << $bits;
        $half = 1 << ($bits - 1);

        return array_map(static fn (int $x) => $x >= $half ? $x - $mod : $x, $u);
    }

    /** @param array<int,int|float> $data */
    private static function packSamples(array $data, string $dtype): string
    {
        return match ($dtype) {
            'uint8' => pack('C*', ...array_map('intval', $data)),
            'int16' => pack('v*', ...array_map(static fn ($x) => ((int) $x) & 0xFFFF, $data)),
            'int32' => pack('V*', ...array_map(static fn ($x) => ((int) $x) & 0xFFFFFFFF, $data)),
            'float32' => $data === [] ? '' : pack('g*', ...array_map('floatval', $data)),
            'float64' => $data === [] ? '' : pack('e*', ...array_map('floatval', $data)),
            default => throw new TesseroException("wavfile: cannot pack {$dtype}"),
        };
    }
}
