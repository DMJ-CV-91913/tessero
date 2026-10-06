<?php

declare(strict_types=1);

namespace Tessero\Native;

/** Operation codes, mirrored from csrc/src/ops.h. AbiTest checks they stay in sync. */
final class Abi
{
    // binary
    public const ADD = 0, SUB = 1, MUL = 2, DIV = 3, POW = 4, MAX = 5, MIN = 6, ATAN2 = 7, HYPOT = 8, MOD = 9;
    public const EQ = 10, NE = 11, LT = 12, LE = 13, GT = 14, GE = 15, AND = 16, OR = 17, XOR = 18, FLOORDIV = 19;

    // unary
    public const NEG = 0, ABS = 1, SQUARE = 2, SIGN = 3, SQRT = 4, EXP = 5, LOG = 6, LOG10 = 7, LOG2 = 8;
    public const SIN = 9, COS = 10, TAN = 11, ARCSIN = 12, ARCCOS = 13, ARCTAN = 14, SINH = 15, COSH = 16, TANH = 17;
    public const FLOOR = 18, CEIL = 19, RINT = 20, EXPM1 = 21, LOG1P = 22, RECIPROCAL = 23;
    public const ISNAN = 24, ISFINITE = 25, ISINF = 26, NOT = 27, REAL = 28, IMAG = 29, CABS = 30, ANGLE = 31, CONJ = 32;

    // reductions
    public const R_SUM = 0, R_PROD = 1, R_MIN = 2, R_MAX = 3, R_ARGMIN = 4, R_ARGMAX = 5, R_ANY = 6, R_ALL = 7;

    /** Unary ops whose result is float even for integer input. */
    public const FLOAT_UNARY = [
        self::SQRT, self::EXP, self::LOG, self::LOG10, self::LOG2, self::SIN, self::COS, self::TAN, self::ARCSIN,
        self::ARCCOS, self::ARCTAN, self::SINH, self::COSH, self::TANH, self::EXPM1, self::LOG1P, self::RECIPROCAL,
    ];

    /** Unary ops that return bool. */
    public const BOOL_UNARY = [self::ISNAN, self::ISFINITE, self::ISINF];

    /** Binary ops that return bool. */
    public const COMPARISONS = [self::EQ, self::NE, self::LT, self::LE, self::GT, self::GE];

    public const LOGICAL = [self::AND, self::OR, self::XOR];
}
