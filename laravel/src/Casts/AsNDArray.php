<?php

declare(strict_types=1);

namespace Tessero\Laravel\Casts;

use Illuminate\Contracts\Database\Eloquent\Castable;
use Illuminate\Contracts\Database\Eloquent\CastsAttributes;

/**
 * Eloquent cast between a column and an NDArray.
 *
 *   protected $casts = [
 *       'weights'  => AsNDArray::class,                  // JSON nested arrays (default)
 *       'profile'  => AsNDArray::class.':binary',        // exact bytes, base64, compact
 *       'features' => AsNDArray::class.':binary,float32',
 *   ];
 *
 * Binary storage is {"dtype":"float64","shape":[24],"b64":"..."}: exact to
 * the bit, about 2.7x smaller than JSON for float64 and decodes without any
 * float parsing. Either format can be read back regardless of the setting.
 */
final class AsNDArray implements Castable
{
    /** @param list<string> $arguments format (json|binary) and/or dtype */
    public static function castUsing(array $arguments): CastsAttributes
    {
        $format = null;
        $dtype = null;
        foreach ($arguments as $arg) {
            $arg = strtolower(trim((string) $arg));
            if (in_array($arg, ['json', 'binary'], true)) {
                $format = $arg;
            } elseif ($arg !== '') {
                $dtype = $arg;
            }
        }

        return new NDArrayCaster($format, $dtype);
    }
}
