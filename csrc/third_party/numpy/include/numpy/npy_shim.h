/* Tessera: the few numpy/npy_common.h and npy_math.h definitions numpy's random distributions use, so that
   numpy/random/src/distributions*.c compile without Python or NumPy headers (local modification, see
   THIRD_PARTY_NOTICES.md). */
#ifndef TSR_NPY_SHIM_H
#define TSR_NPY_SHIM_H
#include <math.h>
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
typedef ptrdiff_t npy_intp;
typedef unsigned char npy_bool;
typedef uint64_t npy_uint64;
typedef uint32_t npy_uint32;
typedef uint16_t npy_uint16;
typedef uint8_t npy_uint8;
typedef int64_t npy_int64;
#define NPY_INLINE inline
#define NPY_NAN NAN
#define NPY_INFINITY INFINITY
#define npy_log1p log1p
#define npy_log1pf log1pf
#define npy_isnan isnan
#define npy_isfinite isfinite
#define npy_isinf isinf
#define npy_expm1 expm1
#define npy_exp exp
#define npy_log log
#define npy_sqrt sqrt
#define npy_floor floor
#define npy_ceil ceil
#endif
