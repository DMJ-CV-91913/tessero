/*
 * The language boundary between the C function-registry core (fn_core.c) and the C++ implementations
 * (the cxx/ sources): every gufunc core and routine is called through these, so no C++ exception unwinds through C
 * frames (undefined, and in practice std::terminate, which takes the PHP process down). An exception becomes
 * an error code and message, the way SciPy's would become a Python exception:
 *   std::bad_alloc, std::length_error  -> TSR_ENOMEM ("Unable to allocate ...", SciPy's MemoryError)
 *   any other exception                -> TSR_EARG with its message
 */
#include <new>
#include <stdexcept>

extern "C" {
#include "../src/fn.h"
}

namespace {
template <class F> int guard(F &&f)
{
    try {
        (void)dist_take_error();                 /* a failure recorded during this call only */
        int rc = f();
        /* a special function SciPy would have raised from inside (a Boost overflow: OverflowError) */
        const char *why = dist_take_error();
        if (why && rc == TSR_OK) { fn_set_error("%s", why); rc = TSR_EARG; }
        return rc;
    } catch (const std::bad_alloc &) {
        fn_set_error("Unable to allocate memory for the result");
        return TSR_ENOMEM;
    } catch (const std::length_error &) {
        fn_set_error("Maximum allowed size exceeded");
        return TSR_ENOMEM;
    } catch (const std::exception &e) {
        fn_set_error("%s", e.what());
        return TSR_EARG;
    } catch (...) {
        fn_set_error("internal error in a numerical routine");
        return TSR_EARG;
    }
}
}  // namespace

extern "C" int fn_call_core(fn_core f, const void *ctx, const double *const *x, const int64_t *n, const double *p,
                            double *const *out, const int64_t *out_n, void *scratch, unsigned flags)
{
    return guard([&] { return f(ctx, x, n, p, out, out_n, scratch, flags); });
}

extern "C" int fn_call_routine(fn_routine f, const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    return guard([&] { return f(ctx, args, nargs, res, nres); });
}
