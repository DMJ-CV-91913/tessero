/*
 * PCG64 (XSL-RR 128/64) seeded through NumPy's SeedSequence, so that
 *
 *     Tessero\Random::default(seed)->random(n)   ==  numpy.random.default_rng(seed).random(n)
 *     ->integers(lo, hi, n)                        ==  .integers(lo, hi, n)
 *     ->normal(mu, sd, n)                          ==  .normal(mu, sd, n)
 *
 * bit for bit. The state vector is 6 x uint64:
 *   [0] state hi  [1] state lo  [2] inc hi  [3] inc lo  [4] has_uint32  [5] buffered uint32
 * Words 4-5 mirror numpy's bit-generator 32-bit buffer (next_uint32 hands out the
 * low half of a 64-bit draw, then the high half), which integers() consumes for
 * ranges below 2^32.
 */
#include "internal.h"
#include "ziggurat.h"
#include "numpy/random/bitgen.h"
#include <math.h>

typedef unsigned __int128 u128;

#define U128(hi, lo) (((u128)(hi) << 64) | (u128)(lo))
static const u128 PCG_MULT = U128(2549297995355413924ULL, 4865540595714422341ULL);

/* ---------------- SeedSequence (numpy/random/bit_generator.pyx) ---------------- */
#define SS_POOL 4
#define INIT_A 0x43b0d7e5u
#define MULT_A 0x931e8875u
#define INIT_B 0x8b51f9ddu
#define MULT_B 0x58f38dedu
#define MIX_MULT_L 0xca01f9ddu
#define MIX_MULT_R 0x4973f715u
#define XSHIFT 16

static uint32_t hashmix(uint32_t value, uint32_t *hash_const)
{
    value ^= *hash_const;
    *hash_const *= MULT_A;
    value *= *hash_const;
    value ^= value >> XSHIFT;
    return value;
}

static uint32_t mix(uint32_t x, uint32_t y)
{
    uint32_t r = MIX_MULT_L * x - MIX_MULT_R * y;
    r ^= r >> XSHIFT;
    return r;
}

void tsr_seed_sequence(const uint32_t *entropy, int64_t n_entropy, uint64_t *out_words, int64_t n_words)
{
    uint32_t pool[SS_POOL];
    uint32_t hc = INIT_A;
    for (int i = 0; i < SS_POOL; i++)
        pool[i] = hashmix(i < n_entropy ? entropy[i] : 0u, &hc);
    for (int s = 0; s < SS_POOL; s++)
        for (int d = 0; d < SS_POOL; d++)
            if (s != d) pool[d] = mix(pool[d], hashmix(pool[s], &hc));
    for (int64_t s = SS_POOL; s < n_entropy; s++)
        for (int d = 0; d < SS_POOL; d++)
            pool[d] = mix(pool[d], hashmix(entropy[s], &hc));

    uint32_t hb = INIT_B;
    for (int64_t w = 0; w < n_words; w++) {
        uint32_t half[2];
        for (int h = 0; h < 2; h++) {
            uint32_t v = pool[(2 * w + h) % SS_POOL];
            v ^= hb;
            hb *= MULT_B;
            v *= hb;
            v ^= v >> XSHIFT;
            half[h] = v;
        }
        out_words[w] = (uint64_t)half[0] | ((uint64_t)half[1] << 32);
    }
}

/* ---------------- PCG64 core ---------------- */
static inline u128 st_get(const uint64_t *s) { return U128(s[0], s[1]); }
static inline u128 inc_get(const uint64_t *s) { return U128(s[2], s[3]); }
static inline void st_set(uint64_t *s, u128 v) { s[0] = (uint64_t)(v >> 64); s[1] = (uint64_t)v; }

static inline uint64_t next64(uint64_t *s)
{
    u128 st = st_get(s) * PCG_MULT + inc_get(s);
    st_set(s, st);
    uint64_t hi = (uint64_t)(st >> 64), lo = (uint64_t)st;
    uint64_t x = hi ^ lo;
    unsigned rot = (unsigned)(hi >> 58);
    return (x >> rot) | (x << ((64 - rot) & 63));
}

static inline uint32_t next32(uint64_t *s)
{
    if (s[4]) {
        s[4] = 0;
        return (uint32_t)s[5];
    }
    uint64_t v = next64(s);
    s[4] = 1;
    s[5] = v >> 32;
    return (uint32_t)(v & 0xffffffffu);
}

static inline double next_double(uint64_t *s) { return (double)(next64(s) >> 11) * (1.0 / 9007199254740992.0); }

/* numpy's bit-generator interface over this state, for the vendored numpy/random distributions */
static uint64_t bg_next64(void *s) { return next64((uint64_t *)s); }
static uint32_t bg_next32(void *s) { return next32((uint64_t *)s); }
static double bg_next_double(void *s) { return next_double((uint64_t *)s); }

void tsr_pcg64_bitgen(uint64_t *state, struct bitgen *bg)
{
    bg->state = state;
    bg->next_uint64 = bg_next64;
    bg->next_uint32 = bg_next32;
    bg->next_double = bg_next_double;
    bg->next_raw = bg_next64;
}

void tsr_pcg64_seed(uint64_t *state, uint64_t state_hi, uint64_t state_lo, uint64_t inc_hi, uint64_t inc_lo)
{
    u128 initstate = U128(state_hi, state_lo), initseq = U128(inc_hi, inc_lo);
    u128 inc = (initseq << 1) | 1u;
    state[2] = (uint64_t)(inc >> 64);
    state[3] = (uint64_t)inc;
    st_set(state, 0);
    next64(state);
    st_set(state, st_get(state) + initstate);
    next64(state);
    state[4] = 0;
    state[5] = 0;
}

void tsr_pcg64_uint64(uint64_t *state, int64_t n, uint64_t *out)
{
    for (int64_t i = 0; i < n; i++) out[i] = next64(state);
}

void tsr_pcg64_random(uint64_t *state, int64_t n, double *out)
{
    for (int64_t i = 0; i < n; i++) out[i] = next_double(state);
}

static double std_normal(uint64_t *s)
{
    for (;;) {
        uint64_t r = next64(s);
        int idx = (int)(r & 0xff);
        r >>= 8;
        int sign = (int)(r & 1);
        uint64_t rabs = (r >> 1) & 0x000fffffffffffffULL;
        double x = (double)rabs * tsr_zig_wi[idx];
        if (sign) x = -x;
        if (rabs < tsr_zig_ki[idx]) return x;
        if (idx == 0) {
            for (;;) {
                double xx = -tsr_zig_nor_inv_r * log1p(-next_double(s));
                double yy = -log1p(-next_double(s));
                if (yy + yy > xx * xx)
                    return ((rabs >> 8) & 1) ? -(tsr_zig_nor_r + xx) : tsr_zig_nor_r + xx;
            }
        }
        if (((tsr_zig_fi[idx - 1] - tsr_zig_fi[idx]) * next_double(s) + tsr_zig_fi[idx]) < exp(-0.5 * x * x))
            return x;
    }
}

void tsr_pcg64_normal(uint64_t *state, int64_t n, double mean, double sd, double *out)
{
    for (int64_t i = 0; i < n; i++) out[i] = mean + sd * std_normal(state);
}

/* Lemire's nearly-divisionless bounded integers, exactly as numpy's random_bounded_uint64_fill (use_masked=False). */
void tsr_pcg64_integers(uint64_t *state, int64_t n, int64_t low, int64_t high, int64_t *out)
{
    if (n <= 0) return;
    uint64_t rng = (uint64_t)high - (uint64_t)low - 1u; /* inclusive range */
    uint64_t off = (uint64_t)low;
    if (high <= low) rng = 0;
    if (rng == 0) {
        for (int64_t i = 0; i < n; i++) out[i] = low;
    } else if (rng <= 0xffffffffULL) {
        if (rng == 0xffffffffULL) {
            for (int64_t i = 0; i < n; i++) out[i] = (int64_t)(off + next32(state));
            return;
        }
        uint32_t r32 = (uint32_t)rng, excl = r32 + 1u;
        for (int64_t i = 0; i < n; i++) {
            uint64_t m = (uint64_t)next32(state) * excl;
            uint32_t left = (uint32_t)m;
            if (left < excl) {
                uint32_t threshold = (UINT32_MAX - r32) % excl;
                while (left < threshold) {
                    m = (uint64_t)next32(state) * excl;
                    left = (uint32_t)m;
                }
            }
            out[i] = (int64_t)(off + (m >> 32));
        }
    } else if (rng == UINT64_MAX) {
        for (int64_t i = 0; i < n; i++) out[i] = (int64_t)(off + next64(state));
    } else {
        uint64_t excl = rng + 1u;
        for (int64_t i = 0; i < n; i++) {
            u128 m = (u128)next64(state) * excl;
            uint64_t left = (uint64_t)m;
            if (left < excl) {
                uint64_t threshold = (UINT64_MAX - rng) % excl;
                while (left < threshold) {
                    m = (u128)next64(state) * excl;
                    left = (uint64_t)m;
                }
            }
            out[i] = (int64_t)(off + (uint64_t)(m >> 64));
        }
    }
}

/* numpy's random_interval: uniform integer in [0, max] by masked rejection. */
static uint64_t interval(uint64_t *s, uint64_t max)
{
    if (max == 0) return 0;
    uint64_t mask = max, v;
    mask |= mask >> 1; mask |= mask >> 2; mask |= mask >> 4;
    mask |= mask >> 8; mask |= mask >> 16; mask |= mask >> 32;
    if (max <= 0xffffffffULL) {
        while ((v = (next32(s) & mask)) > max) {}
    } else {
        while ((v = (next64(s) & mask)) > max) {}
    }
    return v;
}

/* Fisher-Yates over n items of itemsize bytes; identical to Generator.shuffle / permutation. */
int tsr_pcg64_shuffle(uint64_t *state, int64_t n, int64_t itemsize, void *data)
{
    if (n < 0 || itemsize <= 0) return TSR_EARG;
    char *d = (char *)data;
    char tmp[256];
    char *buf = itemsize <= (int64_t)sizeof(tmp) ? tmp : (char *)tsr_alloc(itemsize);
    if (!buf) return TSR_ENOMEM;
    for (int64_t i = n - 1; i >= 1; i--) {
        int64_t j = (int64_t)interval(state, (uint64_t)i);
        if (j != i) {
            memcpy(buf, d + i * itemsize, (size_t)itemsize);
            memcpy(d + i * itemsize, d + j * itemsize, (size_t)itemsize);
            memcpy(d + j * itemsize, buf, (size_t)itemsize);
        }
    }
    if (buf != tmp) tsr_free(buf, itemsize);
    return TSR_OK;
}
