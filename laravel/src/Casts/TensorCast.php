<?php

declare(strict_types=1);

namespace Tessero\Laravel\Casts;

use Illuminate\Contracts\Database\Eloquent\Castable;
use Illuminate\Contracts\Database\Eloquent\CastsAttributes;

/**
 * Alias of AsNDArray under the name used in the original design notes:
 *   protected $casts = ['transition_matrix' => TensorCast::class];
 */
final class TensorCast implements Castable
{
    public static function castUsing(array $arguments): CastsAttributes
    {
        return AsNDArray::castUsing($arguments);
    }
}
