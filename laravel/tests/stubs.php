<?php

// Minimal Illuminate contracts so the bridge's casts, rule and manager can be
// unit-tested without installing Laravel (CI also runs them inside a real app).
namespace Illuminate\Contracts\Database\Eloquent {
    if (! interface_exists(CastsAttributes::class)) {
        interface CastsAttributes
        {
            public function get($model, string $key, mixed $value, array $attributes);
            public function set($model, string $key, mixed $value, array $attributes);
        }
        interface Castable
        {
            public static function castUsing(array $arguments);
        }
        interface SerializesCastableAttributes
        {
            public function serialize($model, string $key, mixed $value, array $attributes);
        }
    }
}

namespace Illuminate\Contracts\Validation {
    if (! interface_exists(ValidationRule::class)) {
        interface ValidationRule
        {
            public function validate(string $attribute, mixed $value, \Closure $fail): void;
        }
    }
}
