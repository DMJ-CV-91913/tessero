<?php

declare(strict_types=1);

namespace Tessero\Laravel\Casts;

use Illuminate\Contracts\Database\Eloquent\CastsAttributes;
use Illuminate\Contracts\Database\Eloquent\SerializesCastableAttributes;
use InvalidArgumentException;
use Tessero\Ext\NDArray as ExtArray;
use Tessero\Laravel\TesseroManager;
use Tessero\NDArray as FfiArray;

/** Caster built by AsNDArray::castUsing(). */
final class NDArrayCaster implements CastsAttributes, SerializesCastableAttributes
{
    public function __construct(private readonly ?string $format, private readonly ?string $dtype)
    {
    }

    public function get($model, string $key, mixed $value, array $attributes): ExtArray|FfiArray|null
    {
        if ($value === null || $value === '') {
            return null;
        }
        $decoded = is_string($value) ? json_decode($value, true, 512, JSON_THROW_ON_ERROR) : $value;
        $manager = self::manager();
        if (is_array($decoded) && isset($decoded['b64'], $decoded['dtype'], $decoded['shape'])) {
            $bytes = base64_decode((string) $decoded['b64'], true);
            if ($bytes === false) {
                throw new InvalidArgumentException("Column {$key}: invalid base64 array payload.");
            }
            $arr = $manager->fromBytes($bytes, (string) $decoded['dtype'], array_map('intval', $decoded['shape']));

            return $this->dtype !== null && $arr->dtype() !== $this->dtype ? $arr->astype($this->dtype) : $arr;
        }

        return $manager->array($decoded, $this->dtype);
    }

    public function set($model, string $key, mixed $value, array $attributes): ?string
    {
        if ($value === null) {
            return null;
        }
        $arr = $value instanceof ExtArray || $value instanceof FfiArray ? $value : self::manager()->array($value, $this->dtype);
        if ($this->dtype !== null && self::dtypeName($arr) !== $this->dtype) {
            $arr = $arr->astype($this->dtype);
        }
        $format = $this->format ?? (string) (self::manager()->config()['cast_format'] ?? 'json');
        if ($format === 'binary') {
            return json_encode(['dtype' => self::dtypeName($arr), 'shape' => $arr->shape(), 'b64' => base64_encode($arr->toBytes())], JSON_THROW_ON_ERROR);
        }

        return $arr->toJson();
    }

    /** $model->toArray() / API resources: nested PHP arrays. */
    public function serialize($model, string $key, mixed $value, array $attributes): mixed
    {
        return $value === null ? null : $value->toArray();
    }

    private static function dtypeName(ExtArray|FfiArray $a): string
    {
        $d = $a->dtype();

        return is_string($d) ? $d : $d->name();
    }

    private static function manager(): TesseroManager
    {
        if (function_exists('app')) {
            try {
                return app(TesseroManager::class);
            } catch (\Throwable) {
                // outside a booted application (tests, scripts)
            }
        }

        return new TesseroManager(['backend' => 'auto', 'cast_format' => 'json']);
    }
}
