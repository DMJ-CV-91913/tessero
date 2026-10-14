<?php

declare(strict_types=1);

namespace Tessero\Stats;

use FFI\CData;
use Tessero\Native\Library;
use Tessero\Random\Generator;

/**
 * numpy's next_uint32 draws plus random_interval (masked-rejection bounded integer) over a seeded
 * PCG64 {@see Generator}, used by {@see RandomTable} (Boyett / rcont1).
 *
 * numpy's PCG64 next_uint32 is BUFFERED: it returns the low 32 bits of a fresh uint64 and stashes the
 * high 32 bits for the following call. The kernel's raw primitive tsr_pcg64_uint64 draws full uint64
 * words WITHOUT touching that buffer, so this class reproduces the buffer in PHP on top of raw uint64
 * draws — yielding the identical next_uint32 sequence numpy would, and therefore advancing the stream
 * exactly. (A freshly seeded generator starts with an empty uint32 buffer, and rcont1 is the sole
 * consumer, so the PHP-side buffer is authoritative.)
 *
 * random_interval(max) returns a uniform integer in [0, max] via numpy's smallest-mask rejection. For
 * contingency tables max = ntot - 1 always fits in 32 bits, so numpy takes the next_uint32 path, which
 * is what this class implements.
 *
 * @internal
 */
final class Pcg64Uint32Stream
{
    private const UINT32_MASK = 0xFFFFFFFF;

    private CData $state;

    private CData $buf;

    private bool $hasBuffered = false;

    private int $buffered = 0;

    public function __construct(Generator $gen)
    {
        $this->state = $gen->stateHandle();
        $this->buf = Library::ffi()->new('uint64_t[1]');
    }

    /** The next 32-bit word (numpy next_uint32), reproducing PCG64's low-then-high buffering. */
    public function next32(): int
    {
        if ($this->hasBuffered) {
            $this->hasBuffered = false;

            return $this->buffered & self::UINT32_MASK;
        }

        Library::ffi()->tsr_pcg64_uint64($this->state, 1, $this->buf);
        $v = $this->buf[0]; // PHP int carrying the uint64 bit pattern

        // Low 32 bits now; stash the high 32 for the next call. Use logical shift via masking so the
        // sign of PHP's 64-bit int does not leak into the high word.
        $low = $v & self::UINT32_MASK;
        $this->buffered = ($v >> 32) & self::UINT32_MASK;
        $this->hasBuffered = true;

        return $low;
    }

    /**
     * numpy random_interval: a uniform integer in [0, $max] by masked rejection. The mask is the
     * smallest 2^b - 1 that is >= $max; values above $max are rejected and redrawn.
     */
    public function randomInterval(int $max): int
    {
        if ($max === 0) {
            return 0;
        }
        $mask = $max;
        $mask |= $mask >> 1;
        $mask |= $mask >> 2;
        $mask |= $mask >> 4;
        $mask |= $mask >> 8;
        $mask |= $mask >> 16;
        // (max fits in 32 bits for contingency tables, so no >> 32 step is needed on the uint32 path.)

        while (true) {
            $value = $this->next32() & $mask;
            if ($value <= $max) {
                return $value;
            }
        }
    }
}
