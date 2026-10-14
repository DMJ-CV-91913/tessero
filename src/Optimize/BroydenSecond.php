<?php

declare(strict_types=1);

namespace Tessero\Optimize;

/**
 * scipy.optimize._nonlin.BroydenSecond ("bad" Broyden): same low-rank machinery
 * as {@see BroydenFirst} but with the second-method update direction.
 *
 * @internal
 */
final class BroydenSecond extends BroydenFirst
{
    protected function doUpdate(array $x, array $f, array $dx, array $df, float $dxNorm, float $dfNorm): void
    {
        $this->Gm->restartReduce($this->maxRank - 1);
        $gmdf = $this->Gm->matvec($df);
        $c = [];
        foreach ($dx as $i => $dxi) {
            $c[] = $dxi - $gmdf[$i];
        }
        $inv = 1.0 / ($dfNorm ** 2);
        $d = [];
        foreach ($df as $dfi) {
            $d[] = $dfi * $inv;
        }
        $this->Gm->append($c, $d);
    }
}
