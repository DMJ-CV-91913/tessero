/*
 * How Boost.Math's throw_on_error handlers format the value in their message.
 *
 * SciPy's Boost wrappers (third_party/scipy/boost_special_functions.h, SpecialPolicy) use Boost's default
 * throw_on_error policies and catch the exceptions: where the exception is thrown decides the result, so the
 * throw stays exactly as in SciPy. Before throwing, Boost formats the offending value with std::stringstream
 * (detail::prec_format). That pulls the statically linked libstdc++'s locale and codecvt machinery into every
 * error path, and under Valgrind it read address 0 and crashed (betainccinv: the Temme root finder raising
 * evaluation_error). These specialisations (floating-point and integer values) format with snprintf instead: the
 * same text, since std::ostream with precision p and the default float field prints exactly "%.{p}g" (p is
 * Boost's own digits formula) and prints integers in full.
 *
 * Included before any other Boost header (special_common.hpp, the one translation unit that includes Boost), so
 * every use of prec_format sees them.
 */
#pragma once

#include <cstdio>
#include <limits>
#include <string>

#include "boost/math/policies/error_handling.hpp"

namespace boost { namespace math { namespace policies { namespace detail {

/* Boost's precision for T (prec_format with the default policy): 2 + digits * log10(2) */
template <class T> constexpr int tsr_prec() { return 2 + (std::numeric_limits<T>::digits * 30103) / 100000; }

template <> inline std::string prec_format<float>(const float &val)
{
    char buf[64];
    std::snprintf(buf, sizeof buf, "%.*g", tsr_prec<float>(), static_cast<double>(val));
    return std::string(buf);
}
template <> inline std::string prec_format<double>(const double &val)
{
    char buf[64];
    std::snprintf(buf, sizeof buf, "%.*g", tsr_prec<double>(), val);
    return std::string(buf);
}
template <> inline std::string prec_format<long double>(const long double &val)
{
    char buf[96];
    std::snprintf(buf, sizeof buf, "%.*Lg", tsr_prec<long double>(), val);
    return std::string(buf);
}
/* integer values (iteration counts in the root finders' messages): std::ostream prints them in full, as %lld does */
#define TSR_PREC_FORMAT_INT(T, CAST, FMT)                                     \
    template <> inline std::string prec_format<T>(const T &val)               \
    {                                                                         \
        char buf[32];                                                         \
        std::snprintf(buf, sizeof buf, FMT, static_cast<CAST>(val));          \
        return std::string(buf);                                              \
    }
TSR_PREC_FORMAT_INT(int, long long, "%lld")
TSR_PREC_FORMAT_INT(long, long long, "%lld")
TSR_PREC_FORMAT_INT(long long, long long, "%lld")
TSR_PREC_FORMAT_INT(unsigned int, unsigned long long, "%llu")
TSR_PREC_FORMAT_INT(unsigned long, unsigned long long, "%llu")
TSR_PREC_FORMAT_INT(unsigned long long, unsigned long long, "%llu")
#undef TSR_PREC_FORMAT_INT

}}}}
