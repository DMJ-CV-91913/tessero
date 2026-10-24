/*
 * tessero_native.h -- the public ABI for native C and C++ consumers (e.g. a Zend extension linking libtessero).
 *
 * tessero.h itself contains declarations only (PHP's FFI::cdef parses it verbatim, so it cannot carry includes,
 * macros or an include guard). Native code includes this header instead: it supplies the standard types tessero.h
 * uses, an include guard, and C linkage for C++.
 */
#ifndef TESSERO_NATIVE_H
#define TESSERO_NATIVE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#include "tessero.h"

#ifdef __cplusplus
}
#endif

#endif
