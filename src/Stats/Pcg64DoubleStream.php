<?php

declare(strict_types=1);

namespace Tessero\Stats;

use FFI\CData;
use Tessero\Native\Library;
use Tessero\Random\Generator;

/**
 * A thin iterator over numpy's next_double draws (random_standard_uniform) from a seeded PCG64
 * {@see Generator}, used by {@see RandomTable} (Patefield / rcont2). Each call advances the SAME
 * stream numpy advances, via the exported kernel primitive tsr_pcg64_random, so the uniforms match
 *   np.random.default_rng(seed).random()
 * exactly on both backends. Because rcont2 draws an unknown number of uniforms (the inner
 * sum-probability renormalisation may redraw), values are pulled one at a time rather than in a
 * single batch — tsr_pcg64_random is incremental, so n=1 calls continue the stream identically.
 *
 * @internal
 */
final class Pcg64DoubleStream
{
    private CData $state;

    private CData $buf;

    public function __construct(Generator $gen)
    {
        $this->state = $gen->stateHandle();
        $this->buf = Library::ffi()->new('double[1]');
    }

    /** The next uniform in [0, 1) (numpy next_double). */
    public function next(): float
    {
        Library::ffi()->tsr_pcg64_random($this->state, 1, $this->buf);

        return (float) $this->buf[0];
    }
}
