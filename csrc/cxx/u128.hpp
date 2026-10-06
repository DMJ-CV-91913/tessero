/*
 * 128-bit integers for the exact rational arithmetic a few ports need (irwinhall moments, the Cramér–von Mises
 * and Kendall tau variance terms), where SciPy uses Python's unbounded integers.
 *
 * GCC and Clang have __int128. MSVC, which builds the extension on Windows, does not, so it gets small structs
 * with the same operators the ports use. Conversion to double rounds correctly in both cases, as Python's
 * int-to-float conversion and GCC's (double)__int128 do.
 */
#pragma once

#include <cmath>
#include <cstdint>

namespace tsr128 {

#if defined(__SIZEOF_INT128__) && !defined(TSR_PORTABLE_U128)   /* TSR_PORTABLE_U128: test the MSVC path */

typedef unsigned __int128 u128;
typedef __int128 i128;
inline bool mul_overflow(u128 a, u128 b, u128 *r) { return __builtin_mul_overflow(a, b, r); }
inline bool add_overflow(u128 a, u128 b, u128 *r) { return __builtin_add_overflow(a, b, r); }
inline double to_double(u128 v) { return (double)v; }
inline double to_double(i128 v) { return (double)v; }

#else

struct u128 {
    uint64_t hi = 0, lo = 0;
    u128() = default;
    u128(uint64_t v) : hi(0), lo(v) {}                       /* NOLINT: implicit, as the built-in type */
    u128(int v) : hi(v < 0 ? ~0ull : 0), lo((uint64_t)(int64_t)v) {}
    u128(int64_t v) : hi(v < 0 ? ~0ull : 0), lo((uint64_t)v) {}
    static u128 make(uint64_t h, uint64_t l) { u128 r; r.hi = h; r.lo = l; return r; }
    explicit operator bool() const { return hi || lo; }
    explicit operator uint64_t() const { return lo; }
    explicit operator unsigned() const { return (unsigned)lo; }
    friend bool operator==(u128 a, u128 b) { return a.hi == b.hi && a.lo == b.lo; }
    friend bool operator!=(u128 a, u128 b) { return !(a == b); }
    friend bool operator<(u128 a, u128 b) { return a.hi < b.hi || (a.hi == b.hi && a.lo < b.lo); }
    friend bool operator>(u128 a, u128 b) { return b < a; }
    friend bool operator<=(u128 a, u128 b) { return !(b < a); }
    friend bool operator>=(u128 a, u128 b) { return !(a < b); }
    friend u128 operator+(u128 a, u128 b) { u128 r = make(a.hi + b.hi, a.lo + b.lo); if (r.lo < a.lo) r.hi++; return r; }
    friend u128 operator-(u128 a, u128 b) { u128 r = make(a.hi - b.hi, a.lo - b.lo); if (a.lo < b.lo) r.hi--; return r; }
    friend u128 operator&(u128 a, u128 b) { return make(a.hi & b.hi, a.lo & b.lo); }
    friend u128 operator|(u128 a, u128 b) { return make(a.hi | b.hi, a.lo | b.lo); }
    u128 &operator|=(u128 b) { hi |= b.hi; lo |= b.lo; return *this; }
    u128 &operator-=(u128 b) { *this = *this - b; return *this; }
    friend u128 operator<<(u128 a, int s)
    {
        if (s <= 0) return a;
        if (s >= 128) return u128();
        if (s >= 64) return make(a.lo << (s - 64), 0);
        return make((a.hi << s) | (a.lo >> (64 - s)), a.lo << s);
    }
    friend u128 operator>>(u128 a, int s)
    {
        if (s <= 0) return a;
        if (s >= 128) return u128();
        if (s >= 64) return make(0, a.hi >> (s - 64));
        return make(a.hi >> s, (a.lo >> s) | (a.hi << (64 - s)));
    }
    u128 &operator<<=(int s) { *this = *this << s; return *this; }
    u128 &operator>>=(int s) { *this = *this >> s; return *this; }
    static void mul64(uint64_t a, uint64_t b, uint64_t &h, uint64_t &l)
    {
        const uint64_t a0 = a & 0xffffffffu, a1 = a >> 32, b0 = b & 0xffffffffu, b1 = b >> 32;
        const uint64_t p00 = a0 * b0, p01 = a0 * b1, p10 = a1 * b0, p11 = a1 * b1;
        const uint64_t mid = (p00 >> 32) + (p01 & 0xffffffffu) + (p10 & 0xffffffffu);
        l = (p00 & 0xffffffffu) | (mid << 32);
        h = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
    }
    friend u128 operator*(u128 a, u128 b)
    {
        uint64_t h, l;
        mul64(a.lo, b.lo, h, l);
        return make(h + a.hi * b.lo + a.lo * b.hi, l);
    }
    static void divmod(u128 n, u128 d, u128 &q, u128 &r)
    {
        q = u128();
        r = u128();
        for (int i = 127; i >= 0; i--) {
            r = (r << 1) | ((n >> i) & u128(1));
            if (r >= d) { r = r - d; q |= (u128(1) << i); }
        }
    }
    friend u128 operator/(u128 a, u128 b) { u128 q, r; divmod(a, b, q, r); return q; }
    friend u128 operator%(u128 a, u128 b) { u128 q, r; divmod(a, b, q, r); return r; }
};

/* two's complement on top of u128: the operations stats_fn_corr uses */
struct i128 {
    u128 v;
    i128() = default;
    i128(int64_t x) : v(x) {}                               /* NOLINT */
    i128(int x) : v(x) {}                                   /* NOLINT */
    static i128 raw(u128 u) { i128 r; r.v = u; return r; }
    friend i128 operator*(i128 a, i128 b) { return raw(a.v * b.v); }
    friend i128 operator-(i128 a, i128 b) { return raw(a.v - b.v); }
    friend i128 operator+(i128 a, i128 b) { return raw(a.v + b.v); }
    bool negative() const { return (v.hi >> 63) != 0; }
};

inline bool mul_overflow(u128 a, u128 b, u128 *r)
{
    /* overflow when the full product needs more than 128 bits */
    if (a.hi && b.hi) { *r = a * b; return true; }
    const u128 big = a.hi ? a : b, small = a.hi ? b : a;     /* at most one operand above 64 bits */
    uint64_t h1, l1, h2, l2;
    u128::mul64(big.lo, small.lo, h1, l1);
    u128::mul64(big.hi, small.lo, h2, l2);
    *r = a * b;
    return h2 != 0 || (l2 + h1) < l2;
}
inline bool add_overflow(u128 a, u128 b, u128 *r) { *r = a + b; return *r < a; }

inline double to_double(u128 v)
{
    if (!v.hi) return (double)v.lo;
    int p = 127;                                            /* the leading bit */
    while (!((v >> p) & u128(1))) p--;
    const int shift = p - 63;                               /* keep 64 bits; the rest folds into a sticky bit */
    const u128 kept = v >> shift;
    const bool sticky = (v & ((u128(1) << shift) - u128(1))) != u128();
    return std::ldexp((double)(kept.lo | (sticky ? 1u : 0u)), shift);
}
inline double to_double(i128 x)
{
    if (!x.negative()) return to_double(x.v);
    return -to_double(u128() - x.v);
}
/* (i128)some_double, for the round trip the ports use (|d| < 2^127, integer-valued after truncation) */
inline i128 from_double(double d)
{
    const bool neg = d < 0;
    double a = std::fabs(std::trunc(d));
    const double two64 = 18446744073709551616.0;
    const uint64_t h = (uint64_t)(a / two64), l = (uint64_t)std::fmod(a, two64);
    const u128 u = u128::make(h, l);
    return i128::raw(neg ? u128() - u : u);
}

#endif

#if defined(__SIZEOF_INT128__) && !defined(TSR_PORTABLE_U128)   /* TSR_PORTABLE_U128: test the MSVC path */
inline i128 from_double(double d) { return (i128)d; }
#endif

}  // namespace tsr128
