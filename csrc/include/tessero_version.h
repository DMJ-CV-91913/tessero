/*
 * libtessero version: the SINGLE source of the release and ABI numbers (ADR 0014).
 *
 * Read by csrc/src/alloc.c (tsr_version, tsr_abi_version), by csrc/Makefile (the soname,
 * libtessero.so.<TSR_ABI_VERSION>) and by the Zend extension (PHP_TESSERO_VERSION in ext/php_tessero.h).
 * This header carries macros, so it is NOT part of the FFI-parsed tessero.h.
 *
 * TSR_ABI_VERSION is bumped on every INCOMPATIBLE change to the C ABI: a removed or renamed function, a changed
 * signature, a changed struct layout (tsr_array, tsr_arg, tsr_result), a changed constant, or a changed
 * documented meaning. Compatible additions (a new function, a new feature name for tsr_feature) do not bump it.
 * The soname follows it, so a consumer linked against one ABI never loads another.
 */
#ifndef TESSERO_VERSION_H
#define TESSERO_VERSION_H

#define TSR_VERSION_MAJOR 0
#define TSR_VERSION_MINOR 2
#define TSR_VERSION_PATCH 0
#define TSR_VERSION_STRING "0.2.0"
#define TSR_ABI_VERSION 2

#endif
