/* Tessera: stand-in for SciPy's sf_error.h. SciPy reports special-function errors as Python warnings;
   the kernel returns the same values without the warning. */
#pragma once
#include <xsf/error.h>
static inline void sf_error(const char *, sf_error_t, const char *, ...) {}
