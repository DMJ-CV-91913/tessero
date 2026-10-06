<?php

declare(strict_types=1);

namespace Tessero;

use Tessero\Exceptions\DTypeError;

/**
 * Element types. The integer values are the libtessero ABI codes.
 *
 * Promotion follows NumPy 2 (NEP 50): array-array promotion by kind and size;
 * a PHP int/float scalar is "weak" and adopts the array's dtype unless its
 * kind is higher (a float scalar with an int array gives float64).
 */
enum DType: int
{
    case Float64 = 0;
    case Float32 = 1;
    case Int64 = 2;
    case Int32 = 3;
    case UInt8 = 4;
    case Bool = 5;
    case Complex128 = 6;

    public static function from_(DType|string $d): self
    {
        if ($d instanceof self) {
            return $d;
        }

        return match (strtolower($d)) {
            'float64', 'f8', 'double', 'float', 'd' => self::Float64,
            'float32', 'f4', 'single', 'f' => self::Float32,
            'int64', 'i8', 'int', 'q' => self::Int64,
            'int32', 'i4', 'l' => self::Int32,
            'uint8', 'u1', 'b' => self::UInt8,
            'bool', '?' => self::Bool,
            'complex128', 'c16', 'complex' => self::Complex128,
            default => throw new DTypeError("Unknown dtype '{$d}'."),
        };
    }

    public function itemsize(): int
    {
        return match ($this) {
            self::Float64, self::Int64 => 8,
            self::Float32, self::Int32 => 4,
            self::UInt8, self::Bool => 1,
            self::Complex128 => 16,
        };
    }

    /** pack()/unpack() format for one element (little-endian hosts; see Buffer). */
    public function packFormat(): string
    {
        return match ($this) {
            self::Float64, self::Complex128 => 'd',
            self::Float32 => 'g',
            self::Int64 => 'q',
            self::Int32 => 'l',
            self::UInt8, self::Bool => 'C',
        };
    }

    /** NumPy .npy descr string. */
    public function descr(): string
    {
        return match ($this) {
            self::Float64 => '<f8',
            self::Float32 => '<f4',
            self::Int64 => '<i8',
            self::Int32 => '<i4',
            self::UInt8 => '|u1',
            self::Bool => '|b1',
            self::Complex128 => '<c16',
        };
    }

    public static function fromDescr(string $descr): self
    {
        return match ($descr) {
            '<f8', '=f8', 'f8' => self::Float64,
            '<f4', '=f4', 'f4' => self::Float32,
            '<i8', '=i8', 'i8' => self::Int64,
            '<i4', '=i4', 'i4' => self::Int32,
            '|u1', 'u1' => self::UInt8,
            '|b1', 'b1', '?' => self::Bool,
            '<c16', '=c16', 'c16' => self::Complex128,
            default => throw new DTypeError("Unsupported .npy dtype '{$descr}' (big-endian and exotic types are not supported)."),
        };
    }

    public function name(): string
    {
        return strtolower($this->name);
    }

    public function isFloat(): bool
    {
        return $this === self::Float64 || $this === self::Float32;
    }

    public function isInteger(): bool
    {
        return $this === self::Int64 || $this === self::Int32 || $this === self::UInt8;
    }

    public function isComplex(): bool
    {
        return $this === self::Complex128;
    }

    /** kind rank: bool < unsigned/int < float < complex */
    private function kind(): int
    {
        return match ($this) {
            self::Bool => 0,
            self::UInt8, self::Int32, self::Int64 => 1,
            self::Float32, self::Float64 => 2,
            self::Complex128 => 3,
        };
    }

    public static function promote(self $a, self $b): self
    {
        if ($a === $b) {
            return $a;
        }
        $ka = $a->kind();
        $kb = $b->kind();
        if ($ka === 3 || $kb === 3) {
            return self::Complex128;
        }
        if ($ka === 2 || $kb === 2) {
            // float32 only survives against bool / uint8 (int16 would too, not modelled)
            $other = $ka === 2 ? $b : $a;
            $float = $ka === 2 ? $a : $b;
            if ($ka === 2 && $kb === 2) {
                return self::Float64;
            }
            if ($float === self::Float32 && ($other === self::Bool || $other === self::UInt8)) {
                return self::Float32;
            }

            return self::Float64;
        }
        if ($ka === 0) {
            return $b;
        }
        if ($kb === 0) {
            return $a;
        }
        // integers: widest wins (uint8 fits in both int32 and int64)
        return $a->itemsize() >= $b->itemsize() ? $a : $b;
    }

    /** Result dtype when a PHP scalar (weakly typed) meets an array of this dtype. */
    public function withScalar(int|float|bool $scalar): self
    {
        if (is_bool($scalar)) {
            return $this;
        }
        if (is_float($scalar)) {
            return $this->kind() >= 2 ? $this : self::Float64;
        }

        return $this === self::Bool ? self::Int64 : $this;
    }

    /** Smallest dtype that holds a PHP value (for array() inference). */
    public static function ofValue(mixed $v): self
    {
        return match (true) {
            is_bool($v) => self::Bool,
            is_int($v) => self::Int64,
            is_float($v) => self::Float64,
            default => throw new DTypeError('Arrays hold numbers or booleans, got ' . get_debug_type($v) . '.'),
        };
    }

    /** Float dtype used for results of true division, sqrt, exp, ... */
    public function toFloat(): self
    {
        return match ($this) {
            self::Float32, self::Float64, self::Complex128 => $this,
            default => self::Float64,
        };
    }
}
