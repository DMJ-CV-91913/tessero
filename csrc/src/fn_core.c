/*
 * The function registry (ADR 0011): lookup, self-description and the two engines that run every entry.
 *
 *   tsr_fn_ufunc    element-wise over already broadcast operands (all float64 or all float32), any number of
 *                   inputs and outputs; also the methods of a distribution (pdf, cdf, ppf, stats, ...).
 *   tsr_fn_gufunc   generalised ufunc: gathers each input's core axes into a contiguous float64 buffer,
 *                   calls the core function once per loop position, scatters the output cores. Loop axes of
 *                   the inputs broadcast; keepdims keeps the core axes as length-1 dimensions.
 *   tsr_fn_rvs      random variates of a distribution from a PCG64 state (serial, C order: the draw order is
 *                   the one NumPy uses, so seeded streams are reproducible element for element).
 *
 * Results never depend on the thread count: each element (or loop position) is computed by the same code
 * whichever thread runs it.
 */
#include "fn.h"
#include "numpy/random/bitgen.h"

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#if defined(_MSC_VER)
#  define FN_TLS __declspec(thread)
#else
#  define FN_TLS _Thread_local
#endif

static FN_TLS char g_err[512];

void fn_set_error(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(g_err, sizeof g_err, fmt, ap);
    va_end(ap);
}

const char *tsr_fn_error(void) { return g_err; }

/* ================================================================ lookup */

static int n_static(void)
{
    int n = 0;
    for (int t = 0; t < TSR_FN_NTABLES; t++) if (TSR_FN_TABLES[t]) n += TSR_FN_TABLES[t]->n;
    return n;
}

static int n_dists(void)
{
    int n = 0;
    for (int t = 0; t < TSR_DIST_NTABLES; t++) n += TSR_DIST_TABLES[t]->n;
    return n;
}

int tsr_fn_count(void) { return n_static() + n_dists(); }

static const fn_def *def_of(int id)
{
    if (id < 0) return NULL;
    for (int t = 0; t < TSR_FN_NTABLES; t++) {
        if (!TSR_FN_TABLES[t]) continue;                  /* a module not linked in (weak, see fn_tables.c) */
        if (id < TSR_FN_TABLES[t]->n) return &TSR_FN_TABLES[t]->defs[id];
        id -= TSR_FN_TABLES[t]->n;
    }
    return NULL;
}

static const fn_dist *dist_of(int id)
{
    id -= n_static();
    if (id < 0) return NULL;
    for (int t = 0; t < TSR_DIST_NTABLES; t++) {
        if (id < TSR_DIST_TABLES[t]->n) return &TSR_DIST_TABLES[t]->dists[id];
        id -= TSR_DIST_TABLES[t]->n;
    }
    return NULL;
}

/* distribution names are "stats.<name>"; built once into a static table (names are short and fixed) */
#define DIST_NAME_MAX 40
#define DIST_MAX 256
static char g_dist_names[DIST_MAX][DIST_NAME_MAX];

const char *tsr_fn_name(int id)
{
    const fn_def *d = def_of(id);
    if (d) return d->name;
    const fn_dist *ds = dist_of(id);
    if (!ds) return NULL;
    const int k = id - n_static();
    if (k >= DIST_MAX) return NULL;
    if (g_dist_names[k][0] == '\0') {
        /* benign race: every writer writes the same bytes, and the terminator last */
        char tmp[DIST_NAME_MAX];
        snprintf(tmp, sizeof tmp, "stats.%s", ds->name);
        memcpy(g_dist_names[k] + 1, tmp + 1, strlen(tmp));
        g_dist_names[k][0] = tmp[0];
    }
    return g_dist_names[k];
}

int tsr_fn_find(const char *name)
{
    if (!name) return TSR_EARG;
    const int ns = n_static(), nt = tsr_fn_count();
    for (int i = 0; i < ns; i++)
        if (strcmp(def_of(i)->name, name) == 0) return i;
    if (strncmp(name, "stats.", 6) == 0)
        for (int i = ns; i < nt; i++)
            if (strcmp(dist_of(i)->name, name + 6) == 0) return i;
    return TSR_EARG;
}

/* ================================================================ self-description (JSON) */

typedef struct { char *buf; int64_t cap, len; } jbuf;

static void jraw(jbuf *j, const char *s, size_t n)
{
    if (j->len + (int64_t)n < j->cap) memcpy(j->buf + j->len, s, n);
    j->len += (int64_t)n;
}
static void jlit(jbuf *j, const char *s) { jraw(j, s, strlen(s)); }
static void jstr(jbuf *j, const char *s, size_t n)
{
    jraw(j, "\"", 1);
    for (size_t i = 0; i < n; i++) {
        const char c = s[i];
        if (c == '"' || c == '\\') { jraw(j, "\\", 1); jraw(j, &c, 1); }
        else if (c == '\n') jraw(j, "\\n", 2);
        else if ((unsigned char)c < 0x20) jraw(j, " ", 1);
        else jraw(j, &c, 1);
    }
    jraw(j, "\"", 1);
}
static void jint(jbuf *j, long long v) { char t[32]; snprintf(t, sizeof t, "%lld", v); jlit(j, t); }

static const char *skip_ws(const char *p) { while (*p == ' ') p++; return p; }

/* the bare name of one entry of an argument list: "*xi" (variadic) and "arrays[]" (a sequence) -> xi, arrays */
static void name_span(const char **p, const char **t)
{
    if (**p == '*' || **p == '&') (*p)++;
    if (*t - *p >= 2 && (*t)[-2] == '[' && (*t)[-1] == ']') *t -= 2;
}

/* "a, x=1, w?" -> ["a","x","w"] (defaults and the optional mark are reported separately) */
static void jnames(jbuf *j, const char *list)
{
    jlit(j, "[");
    int first = 1;
    const char *p = list ? list : "";
    while (*(p = skip_ws(p))) {
        const char *e = p;
        while (*e && *e != ',') e++;
        const char *t = p;
        while (t < e && *t != '=' && *t != '?') t++;
        while (t > p && t[-1] == ' ') t--;
        const char *s = p;
        name_span(&s, &t);
        if (!first) jlit(j, ",");
        jstr(j, s, (size_t)(t - s));
        first = 0;
        p = *e ? e + 1 : e;
    }
    jlit(j, "]");
}

/* indices of the list entries marked "&a": arrays the routine modifies in place (numpy.put, place, copyto ...) */
static void jinplace(jbuf *j, const char *list)
{
    jlit(j, "[");
    int first = 1, k = 0;
    const char *p = list ? list : "";
    while (*(p = skip_ws(p))) {
        const char *e = p;
        while (*e && *e != ',') e++;
        if (*p == '&') { if (!first) jlit(j, ","); jint(j, k); first = 0; }
        k++;
        p = *e ? e + 1 : e;
    }
    jlit(j, "]");
}

/* indices of the list entries marked as sequences ("arrays[]", and the variadic "*xi"); *variadic gets the
   index of the "*" entry or -1 */
static void jseq(jbuf *j, const char *list, int *variadic)
{
    jlit(j, "[");
    int first = 1, k = 0;
    *variadic = -1;
    const char *p = list ? list : "";
    while (*(p = skip_ws(p))) {
        const char *e = p;
        while (*e && *e != ',') e++;
        const char *t = p;
        while (t < e && *t != '=' && *t != '?') t++;
        while (t > p && t[-1] == ' ') t--;
        const int var = *p == '*', seq = var || (t - p >= 2 && t[-2] == '[' && t[-1] == ']');
        if (var) *variadic = k;
        if (seq) { if (!first) jlit(j, ","); jint(j, k); first = 0; }
        k++;
        p = *e ? e + 1 : e;
    }
    jlit(j, "]");
}

/* {"x":"1"} for "a, x=1"; optional inputs "w?" get the default "None" */
static void jdefaults(jbuf *j, const char *list)
{
    jlit(j, "{");
    int first = 1;
    const char *p = list ? list : "";
    while (*(p = skip_ws(p))) {
        const char *e = p;
        while (*e && *e != ',') e++;
        const char *t = p;
        while (t < e && *t != '=' && *t != '?') t++;
        if (t < e) {
            const char *ne = t, *s = p;
            while (ne > p && ne[-1] == ' ') ne--;
            name_span(&s, &ne);
            if (!first) jlit(j, ",");
            first = 0;
            jstr(j, s, (size_t)(ne - s));
            jlit(j, ":");
            if (*t == '?') jlit(j, "\"None\"");
            else {
                const char *v = skip_ws(t + 1), *ve = e;
                while (ve > v && ve[-1] == ' ') ve--;
                jstr(j, v, (size_t)(ve - v));
            }
        }
        p = *e ? e + 1 : e;
    }
    jlit(j, "}");
}

/* "ddof=0, method=linear|lower" -> [{"name":"ddof","default":"0"},{"name":"method","enum":["linear","lower"]}] */
static void jparams(jbuf *j, const char *list, int nanpolicy)
{
    jlit(j, "[");
    int first = 1;
    const char *p = list ? list : "";
    while (*(p = skip_ws(p))) {
        const char *e = p;
        while (*e && *e != ',') e++;
        const char *eq = p;
        while (eq < e && *eq != '=') eq++;
        const char *ne = eq;
        while (ne > p && ne[-1] == ' ') ne--;
        if (!first) jlit(j, ",");
        first = 0;
        jlit(j, "{\"name\":");
        jstr(j, p, (size_t)(ne - p));
        if (eq < e) {
            const char *v = skip_ws(eq + 1), *ve = e;
            while (ve > v && ve[-1] == ' ') ve--;
            const char *bar = v;
            while (bar < ve && *bar != '|') bar++;
            if (bar < ve) {
                jlit(j, ",\"enum\":[");
                const char *q = v;
                int f2 = 1;
                while (q < ve) {
                    const char *qe = q;
                    while (qe < ve && *qe != '|') qe++;
                    if (!f2) jlit(j, ",");
                    jstr(j, q, (size_t)(qe - q));
                    f2 = 0;
                    q = qe < ve ? qe + 1 : qe;
                }
                jlit(j, "]");
            } else {
                jlit(j, ",\"default\":");
                jstr(j, v, (size_t)(ve - v));
            }
        }
        jlit(j, "}");
        p = *e ? e + 1 : e;
    }
    if (nanpolicy) {
        if (!first) jlit(j, ",");
        jlit(j, "{\"name\":\"nan_policy\",\"enum\":[\"propagate\",\"omit\",\"raise\"]}");
    }
    jlit(j, "]");
}

static const char *const DM_NAMES[DM_COUNT] = {"pdf", "logpdf", "cdf", "logcdf", "sf", "logsf", "ppf", "isf",
                                               "stats", "entropy", "support", "moment"};

int64_t tsr_fn_info(int id, char *buf, int64_t cap)
{
    jbuf j = {buf, buf ? cap : 0, 0};
    const fn_def *d = def_of(id);
    if (d) {
        static const char *const KINDS[] = {"", "ufunc", "gufunc", "dist", "routine", "random"};
        jlit(&j, "{\"name\":");
        jstr(&j, d->name, strlen(d->name));
        jlit(&j, ",\"kind\":\""); jlit(&j, KINDS[d->kind]); jlit(&j, "\"");
        jlit(&j, ",\"nin\":"); jint(&j, d->nin);
        jlit(&j, ",\"nout\":"); jint(&j, d->nout);
        jlit(&j, ",\"args\":"); jnames(&j, d->args);
        jlit(&j, ",\"defaults\":"); jdefaults(&j, d->args);
        jlit(&j, ",\"outs\":"); jnames(&j, d->outs);
        jlit(&j, ",\"params\":"); jparams(&j, d->params, d->nanpolicy);
        if (d->kind == FN_ROUTINE) {
            int var;
            jlit(&j, ",\"seq\":"); jseq(&j, d->args, &var);
            jlit(&j, ",\"variadic\":"); jint(&j, var);
            jlit(&j, ",\"out_seq\":"); jint(&j, d->outs && skip_ws(d->outs)[0] == '*');
            jlit(&j, ",\"inplace\":"); jinplace(&j, d->args);
        }
        if (d->kind == FN_RANDOM) {
            jlit(&j, ",\"out_int\":"); jint(&j, d->rdraw_i != NULL);
            jlit(&j, ",\"float32\":"); jint(&j, d->rdraw_f != NULL);
        }
        if (d->kind == FN_UFUNC && d->elem_int) {
            jlit(&j, ",\"int_args\":[");
            int f = 1;
            for (int k = 0; k < d->nin; k++)
                if (d->int_args & (1u << k)) { if (!f) jlit(&j, ","); jint(&j, k); f = 0; }
            jlit(&j, "]");
        }
        if (d->kind == FN_GUFUNC) {
            jlit(&j, ",\"axis\":");
            jstr(&j, d->axis_default ? d->axis_default : "None", strlen(d->axis_default ? d->axis_default : "None"));
            jlit(&j, ",\"axis_inputs\":"); jint(&j, d->ninputs_core ? d->ninputs_core : d->nin);
            jlit(&j, ",\"out_int\":"); jint(&j, d->out_int);
            jlit(&j, ",\"out_bool\":"); jint(&j, d->out_bool);
            int il = 0;
            for (int o = 0; o < d->nout && o < 4; o++) if (d->out_core[o] & CORE_INTLIKE) il |= 1 << o;
            jlit(&j, ",\"out_int_like\":"); jint(&j, il);
            jlit(&j, ",\"keep_param\":"); jint(&j, d->keep ? d->keep->param : -1);
            jlit(&j, ",\"keep_mask\":"); jint(&j, d->keep ? (int)d->keep->mask : 0);
            jlit(&j, ",\"axis_single\":"); jint(&j, (d->axis_rules & AX_SINGLE) != 0);
            jlit(&j, ",\"keepdims\":"); jint(&j, (d->axis_rules & AX_NOKEEP) == 0);
            jlit(&j, ",\"core_first\":");
            int cf = 0;
            for (int o = 0; o < d->nout && o < 4; o++) if (d->out_core[o] & CORE_FIRST) cf |= 1 << o;
            jint(&j, cf);
        }
        jlit(&j, ",\"doc\":");
        jstr(&j, d->doc ? d->doc : "", strlen(d->doc ? d->doc : ""));
        jlit(&j, "}");
        if (buf && cap > 0) buf[j.len < cap ? j.len : cap - 1] = '\0';
        return j.len;
    }
    const fn_dist *ds = dist_of(id);
    if (!ds) return TSR_EARG;
    jlit(&j, "{\"name\":\"stats.");
    jlit(&j, ds->name);
    jlit(&j, "\",\"kind\":\"dist\",\"discrete\":");
    jlit(&j, ds->discrete ? "true" : "false");
    jlit(&j, ",\"shapes\":"); jnames(&j, ds->shapes);
    jlit(&j, ",\"methods\":[");
    for (int m = 0; m < DM_COUNT; m++) { if (m) jlit(&j, ","); jlit(&j, "\""); jlit(&j, DM_NAMES[m]); jlit(&j, "\""); }
    jlit(&j, "],\"doc\":");
    jstr(&j, ds->doc ? ds->doc : "", strlen(ds->doc ? ds->doc : ""));
    jlit(&j, "}");
    if (buf && cap > 0) buf[j.len < cap ? j.len : cap - 1] = '\0';
    return j.len;
}

/* ================================================================ n-operand element loop */

typedef struct {
    int nop, nd;
    int64_t shape[TSR_MAXDIM];
    int64_t st[FN_MAXOP][TSR_MAXDIM];
    int64_t size;
} fn_iter;

/* drop size-1 dims, merge dims contiguous for every operand */
static int fn_iter_init(fn_iter *it, int nop, int32_t ndim, const int64_t *shape, const int64_t *strides)
{
    it->nop = nop;
    it->nd = 0;
    it->size = tsr_shape_size(ndim, shape);
    if (it->size < 0) return TSR_ENOMEM;
    for (int32_t d = 0; d < ndim; d++) {
        if (shape[d] == 1) continue;
        if (it->nd > 0) {
            int merge = 1;
            for (int o = 0; o < nop; o++)
                if (it->st[o][it->nd - 1] != strides[o * ndim + d] * shape[d]) { merge = 0; break; }
            if (merge) {
                it->shape[it->nd - 1] *= shape[d];
                for (int o = 0; o < nop; o++) it->st[o][it->nd - 1] = strides[o * ndim + d];
                continue;
            }
        }
        it->shape[it->nd] = shape[d];
        for (int o = 0; o < nop; o++) it->st[o][it->nd] = strides[o * ndim + d];
        it->nd++;
    }
    if (it->nd == 0) {
        it->nd = 1;
        it->shape[0] = 1;
        for (int o = 0; o < nop; o++) it->st[o][0] = 0;
    }
    return TSR_OK;
}

typedef struct {
    const fn_def *def;          /* FN_UFUNC */
    fn_elem fn;                 /* def->elem or def->elem_int */
    const fn_dist *dist;        /* FN_DIST */
    int method;
    int nin, nout;
    int f32;
    int cplx;                   /* complex128 loop: operands are interleaved (re, im) */
    uint64_t *state;            /* rvs */
} elem_job;

static void dist_elem(const fn_dist *d, int method, const double *in, double *out);

static inline void run_elems(const elem_job *job, char **p, const int64_t *steps, int64_t n)
{
    double in[FN_MAXOP], out[4];
    const int nin = job->nin, nout = job->nout;
    if (job->cplx) {                                     /* complex128 operands: 2 interleaved doubles each */
        for (int64_t i = 0; i < n; i++) {
            for (int k = 0; k < nin; k++) { const double *z = (const double *)(p[k] + i * steps[k]); in[2 * k] = z[0]; in[2 * k + 1] = z[1]; }
            job->fn(job->def->ctx, in, out);
            for (int k = 0; k < nout; k++) { double *z = (double *)(p[nin + k] + i * steps[nin + k]); z[0] = out[2 * k]; z[1] = out[2 * k + 1]; }
        }
        return;
    }
    for (int64_t i = 0; i < n; i++) {
        if (job->f32) for (int k = 0; k < nin; k++) in[k] = (double)*(const float *)(p[k] + i * steps[k]);
        else for (int k = 0; k < nin; k++) in[k] = *(const double *)(p[k] + i * steps[k]);
        if (job->dist) dist_elem(job->dist, job->method, in, out);
        else job->fn(job->def->ctx, in, out);
        if (job->f32) for (int k = 0; k < nout; k++) *(float *)(p[nin + k] + i * steps[nin + k]) = (float)out[k];
        else for (int k = 0; k < nout; k++) *(double *)(p[nin + k] + i * steps[nin + k]) = out[k];
    }
}

static int run_elementwise(const elem_job *job, int32_t ndim, const int64_t *shape, void **data, const int64_t *strides,
                           int parallel)
{
    const int nop = job->nin + job->nout;
    fn_iter it;
    int r = fn_iter_init(&it, nop, ndim, shape, strides);
    if (r != TSR_OK) return r;
    if (it.size == 0) return TSR_OK;
    const int64_t n = it.shape[it.nd - 1];
    const int64_t outer = it.size / n;
    int64_t steps[FN_MAXOP];
    for (int o = 0; o < nop; o++) steps[o] = it.st[o][it.nd - 1];
    int threads = parallel ? tsr_get_threads() : 1;
    if (it.size < TSR_PAR_MIN / 16) threads = 1;     /* special functions cost ~100 ns per element */

    if (outer == 1) {
        if (threads <= 1) { run_elems(job, (char **)data, steps, n); return TSR_OK; }
        const int64_t chunk = (n + threads - 1) / threads;
#if defined(_OPENMP)
#pragma omp parallel for num_threads(threads) schedule(static)
#endif
        for (int t = 0; t < threads; t++) {
            const int64_t lo = (int64_t)t * chunk, len = lo >= n ? 0 : (n - lo < chunk ? n - lo : chunk);
            if (len <= 0) continue;
            char *p[FN_MAXOP];
            for (int o = 0; o < nop; o++) p[o] = (char *)data[o] + lo * steps[o];
            run_elems(job, p, steps, len);
        }
        return TSR_OK;
    }
#if defined(_OPENMP)
#pragma omp parallel for num_threads(threads) schedule(static) if (threads > 1)
#endif
    for (int64_t row = 0; row < outer; row++) {
        char *p[FN_MAXOP];
        for (int o = 0; o < nop; o++) p[o] = (char *)data[o];
        int64_t rem = row;
        for (int d = it.nd - 2; d >= 0; d--) {
            const int64_t i = rem % it.shape[d];
            rem /= it.shape[d];
            for (int o = 0; o < nop; o++) p[o] += i * it.st[o][d];
        }
        run_elems(job, p, steps, n);
    }
    return TSR_OK;
}

static int dist_arity(const fn_dist *d, int method, int *nin, int *nout)
{
    const int np = d->nshape + (d->discrete ? 1 : 2);        /* shapes, loc[, scale] */
    switch (method) {
    case DM_STATS: *nin = 1 + np; *nout = 4; return TSR_OK;       /* the moments mask first */
    case DM_ENTROPY: *nin = np; *nout = 1; return TSR_OK;
    case DM_SUPPORT: *nin = np; *nout = 2; return TSR_OK;
    case DM_MOMENT: *nin = 1 + np; *nout = 1; return TSR_OK;      /* the order first */
    default:
        if (method < 0 || method >= DM_COUNT) return TSR_EARG;
        *nin = 1 + np; *nout = 1; return TSR_OK;
    }
}

int tsr_fn_arity(int id, int method, int *nin, int *nout)
{
    const fn_def *d = def_of(id);
    if (d) { *nin = d->nin; *nout = d->nout; return TSR_OK; }
    const fn_dist *ds = dist_of(id);
    if (!ds) return TSR_EARG;
    return dist_arity(ds, method, nin, nout);
}

int tsr_fn_ufunc(int id, int method, int dtype, int32_t ndim, const int64_t *shape, int nop, void **data,
                 const int64_t *strides)
{
    if (ndim < 0 || ndim > TSR_MAXDIM || !data || (ndim > 0 && (!shape || !strides))) return TSR_EARG;
    if (dtype != TSR_F64 && dtype != TSR_F32 && dtype != TSR_C128) return TSR_ETYPE;
    elem_job job = {0};
    job.f32 = dtype == TSR_F32;
    job.cplx = dtype == TSR_C128;
    const fn_def *d = def_of(id);
    if (d) {
        if (d->kind != FN_UFUNC) return TSR_EARG;
        if (job.cplx && !d->elem_cplx) return TSR_ETYPE;   /* this ufunc has no complex128 loop */
        job.def = d;
        job.fn = job.cplx ? d->elem_cplx : ((method == 1 && d->elem_int) ? d->elem_int : d->elem);
        job.nin = d->nin;
        job.nout = d->nout;
    } else {
        const fn_dist *ds = dist_of(id);
        if (!ds || job.cplx) return TSR_ETYPE;
        job.dist = ds;
        job.method = method;
        if (dist_arity(ds, method, &job.nin, &job.nout) != TSR_OK) return TSR_EARG;
    }
    if (nop != job.nin + job.nout || nop > FN_MAXOP) return TSR_EARG;
    for (int o = 0; o < nop; o++) if (!data[o]) return TSR_EARG;
    static const int64_t zero[FN_MAXOP * 1] = {0};
    int rc;
    (void)dist_take_error();                      /* no stale failure from an earlier call */
    if (ndim == 0) {
        int64_t one = 1;
        rc = run_elementwise(&job, 1, &one, data, zero, 0);
    } else {
        rc = run_elementwise(&job, ndim, shape, data, strides, 1);
    }
    /* a failure SciPy raises for the whole call: a generic ppf that cannot bracket the root, or a Boost overflow
       (SciPy's user_overflow_error sets OverflowError; third_party/scipy/boost_special_functions.h) */
    const char *why = dist_take_error();
    if (why && rc == TSR_OK) { fn_set_error("%s", why); rc = TSR_EARG; }
    return rc;
}

int tsr_random_run(const fn_def *d, uint64_t *state, const double *params, int32_t ndim, const int64_t *shape,
                   int nop, void **data, const int64_t *strides, int out_dtype);

int tsr_fn_random(int id, uint64_t *state, const double *params, int32_t ndim, const int64_t *shape, int nop,
                  void **data, const int64_t *strides, int out_dtype)
{
    const fn_def *d = def_of(id);
    if (!d || d->kind != FN_RANDOM || !state || !data) return TSR_EARG;
    if (ndim < 0 || ndim > TSR_MAXDIM || (ndim > 0 && (!shape || !strides))) return TSR_EARG;
    g_err[0] = '\0';
    return tsr_random_run(d, state, params, ndim, shape, nop, data, strides, out_dtype);
}

int tsr_fn_rvs(int id, uint64_t *state, int32_t ndim, const int64_t *shape, int nop, void **data,
               const int64_t *strides)
{
    const fn_dist *ds = dist_of(id);
    if (!ds || !state || !data) return TSR_EARG;
    if (ndim < 0 || ndim > TSR_MAXDIM || (ndim > 0 && (!shape || !strides))) return TSR_EARG;
    const int np = ds->nshape + (ds->discrete ? 1 : 2);
    if (nop != np + 1 || np > 16) return TSR_EARG;
    g_err[0] = '\0';
    const int64_t size = tsr_shape_size(ndim, shape);
    if (size < 0) return TSR_ENOMEM;
    if (size == 0) return TSR_OK;
    /* the parameters broadcast to the output, contiguous in C order (SciPy's _rvs draws whole arrays: a
       distribution may draw all of one variate before the next, so the kernel hands it every element) */
    double *buf = (double *)tsr_alloc((size_t)(np + 1) * (size_t)size * sizeof(double));
    if (!buf) return TSR_ENOMEM;
    const double *params[16];
    for (int k = 0; k < np; k++) params[k] = buf + (size_t)k * (size_t)size;
    double *out = buf + (size_t)np * (size_t)size;
    int64_t idx[TSR_MAXDIM] = {0};
    for (int64_t e = 0; e < size; e++) {
        for (int k = 0; k < np; k++) {
            const char *p = (const char *)data[k];
            for (int32_t d = 0; d < ndim; d++) p += idx[d] * strides[k * ndim + d];
            ((double *)params[k])[e] = *(const double *)p;
        }
        for (int32_t d = ndim - 1; d >= 0; d--) { if (++idx[d] < shape[d]) break; idx[d] = 0; }
    }
    struct bitgen bg;
    tsr_pcg64_bitgen(state, &bg);
    int rc = dist_rvs_array(ds, &bg, size, params, out);
    const char *why = dist_take_error();
    if (rc != 0 || why) {
        fn_set_error("%s", why ? why : "rvs failed");
        tsr_free(buf, (size_t)(np + 1) * (size_t)size * sizeof(double));
        return TSR_EARG;
    }
    memset(idx, 0, sizeof idx);
    for (int64_t e = 0; e < size; e++) {
        char *p = (char *)data[np];
        for (int32_t d = 0; d < ndim; d++) p += idx[d] * strides[np * ndim + d];
        *(double *)p = out[e];
        for (int32_t d = ndim - 1; d >= 0; d--) { if (++idx[d] < shape[d]) break; idx[d] = 0; }
    }
    tsr_free(buf, (size_t)(np + 1) * (size_t)size * sizeof(double));
    return TSR_OK;
}

/* ---------------------------------------------------------------- distribution element functions */

static void dist_elem(const fn_dist *d, int method, const double *in, double *out)
{
    dist_method(d, method, in, out);
}

/* ================================================================ generalised ufuncs */


static inline double load_as_double(int dtype, const char *p)
{
    switch (dtype) {
    case TSR_F64: return *(const double *)p;
    case TSR_F32: return (double)*(const float *)p;
    case TSR_I64: return (double)*(const int64_t *)p;
    case TSR_I32: return (double)*(const int32_t *)p;
    case TSR_U8: return (double)*(const uint8_t *)p;
    case TSR_BOOL: return *(const uint8_t *)p ? 1.0 : 0.0;
    default: return FN_NAN;
    }
}

static inline void store_double(int dtype, char *p, double v)
{
    switch (dtype) {
    case TSR_F64: *(double *)p = v; break;
    case TSR_F32: *(float *)p = (float)v; break;
    /* a float outside the target range is INT_MIN (what x86 and NumPy produce there), never C's undefined cast */
    case TSR_I64: *(int64_t *)p = (v > -9223372036854775808.0 && v < 9223372036854775808.0) ? (int64_t)v : INT64_MIN; break;
    case TSR_I32: *(int32_t *)p = (v > -2147483649.0 && v < 2147483648.0) ? (int32_t)v : INT32_MIN; break;
    case TSR_U8: *(uint8_t *)p = (v > -9223372036854775808.0 && v < 9223372036854775808.0) ? (uint8_t)(int64_t)v : 0; break;
    case TSR_BOOL: *(uint8_t *)p = v != 0.0; break;
    default: break;
    }
}

typedef struct {
    int nin, nout;
    int32_t loop_nd;
    int64_t loop_shape[TSR_MAXDIM];
    int64_t in_loop_st[4][TSR_MAXDIM];       /* stride of each input along each loop dim (0 = broadcast) */
    int32_t core_nd[4];
    int64_t core_shape[4][TSR_MAXDIM];
    int64_t core_st[4][TSR_MAXDIM];
    int64_t core_n[4];
    int32_t out_core_nd[4];
    int64_t out_core_shape[4][TSR_MAXDIM];
    int64_t out_core_n[4];
    int core_axis0;                          /* input 0's core axis when it has exactly one, else -1 */
    int in0_ndim;
    int keepdims;
} gplan;

static int core_len_rule(const fn_def *d, int o, const gplan *g, const double *params, int32_t *nd, int64_t *shape)
{
    const int code = d->out_core[o];
    const int rule = code & 0xff;
    if (rule == CORE_SCALAR) { *nd = 0; return TSR_OK; }
    if (rule >= CORE_LIKE0 && rule <= CORE_LIKE2) {
        const int i = rule - CORE_LIKE0;
        if (i >= g->nin) return TSR_EARG;
        *nd = g->core_nd[i];
        for (int k = 0; k < *nd; k++) shape[k] = g->core_shape[i][k];
    } else if (rule >= CORE_FIXED) {
        *nd = 1;
        shape[0] = rule - CORE_FIXED;
    } else if (rule >= CORE_PARAM) {
        const double v = params ? params[rule - CORE_PARAM] : FN_NAN;
        if (!(v >= 0) || v > 1e15 || v != floor(v)) { fn_set_error("invalid output length %g", v); return TSR_EARG; }
        *nd = 1;
        shape[0] = (int64_t)v;
    } else {
        return TSR_EARG;
    }
    if (code & CORE_FLAT) {
        int64_t n = 1;
        for (int k = 0; k < *nd; k++) n *= shape[k];
        *nd = 1;
        shape[0] = n;
    }
    /* adjustments apply to the last core dimension (1-D cores) */
    if (*nd >= 1) {
        const int add = (code >> 8) & 0xf;
        if (add) {
            /* a boolean parameter (include_initial): one extra element when true, as NumPy's truthiness; the
               core writes exactly one, so the length must never follow the parameter's numeric value */
            const double v = params ? params[add - 1] : 0;
            shape[*nd - 1] += v != 0 ? 1 : 0;      /* NaN is truthy, as bool(nan) */
        }
        if (code & CORE_MINUS1) shape[*nd - 1] = shape[*nd - 1] > 0 ? shape[*nd - 1] - 1 : 0;
    }
    return TSR_OK;
}

static int gplan_make(const fn_def *d, int nin, const tsr_array *in, const uint64_t *axmask, int keepdims,
                      const double *params, gplan *g)
{
    if (nin != d->nin || nin > 4 || d->nout > 4) return TSR_EARG;
    memset(g, 0, sizeof *g);
    g->nin = nin;
    g->nout = d->nout;
    g->core_axis0 = -1;
    g->in0_ndim = nin > 0 ? in[0].ndim : 0;
    g->keepdims = keepdims;
    if (keepdims && (d->axis_rules & AX_NOKEEP)) { fn_set_error("keepdims is not supported"); return TSR_EARG; }
    /* each input: loop view (core axes removed, or kept as length 1) + core description */
    int32_t lnd[4];
    int64_t lshape[4][TSR_MAXDIM], lst[4][TSR_MAXDIM];
    for (int i = 0; i < nin; i++) {
        const tsr_array *a = &in[i];
        if (a->ndim < 0 || a->ndim > TSR_MAXDIM) return TSR_EDIM;
        const uint64_t m = axmask[i];
        if (a->ndim < 64 && (m >> a->ndim) != 0) { fn_set_error("axis out of range for an array of dimension %d", a->ndim); return TSR_EARG; }
        lnd[i] = 0;
        g->core_nd[i] = 0;
        g->core_n[i] = 1;
        for (int32_t k = 0; k < a->ndim; k++) {
            if (m & ((uint64_t)1 << k)) {
                g->core_shape[i][g->core_nd[i]] = a->shape[k];
                g->core_st[i][g->core_nd[i]] = a->strides[k];
                g->core_nd[i]++;
                g->core_n[i] *= a->shape[k];
                if (keepdims) { lshape[i][lnd[i]] = 1; lst[i][lnd[i]] = 0; lnd[i]++; }
            } else {
                lshape[i][lnd[i]] = a->shape[k];
                lst[i][lnd[i]] = a->strides[k];
                lnd[i]++;
            }
        }
        if (g->core_n[i] < 0 || tsr_shape_size(g->core_nd[i], g->core_shape[i]) < 0) return TSR_ENOMEM;
        if (i == 0 && g->core_nd[0] == 1)
            for (int32_t k = 0; k < a->ndim; k++) if (m & ((uint64_t)1 << k)) g->core_axis0 = k;
    }
    /* numpy.average's _weights_are_valid: weights of a's shape, or exactly a's shape along the reduced axes */
    if ((d->axis_rules & AX_WEIGHTS) && nin > 1 && in[1].ndim > 0) {
        int same = in[1].ndim == in[0].ndim;
        for (int32_t k = 0; same && k < in[0].ndim; k++) same = in[1].shape[k] == in[0].shape[k];
        if (!same) {
            int ok = in[1].ndim == g->core_nd[0];
            for (int32_t k = 0; ok && k < g->core_nd[0]; k++) ok = in[1].shape[k] == g->core_shape[0][k];
            if (!ok) { fn_set_error("Shape of weights must be consistent with shape of a along specified axis."); return TSR_EARG; }
        }
    }
    if (d->keep && nin > 0 && in[0].dtype == TSR_BOOL && tsr_shape_size(in[0].ndim, in[0].shape) > 0) {
        const double pv = params ? params[d->keep->param] : 0;
        if (!(pv >= 0 && pv < 32 && ((d->keep->mask >> (int)pv) & 1u))) {
            fn_set_error("numpy boolean subtract, the `-` operator, is not supported, use the bitwise_xor, the `^` operator, or the logical_xor function instead.");
            return TSR_ETYPE;
        }
    }
    if ((d->axis_rules & AX_IN0_1D) && nin > 0 && in[0].ndim != 1) {
        fn_set_error(in[0].ndim == 0 ? "object of too small depth for desired array" : "object too deep for desired array");
        return TSR_EARG;
    }
    if ((d->axis_rules & AX_NONE_1D) && nin > 0 && in[0].ndim > 1 && g->core_nd[0] == in[0].ndim) {
        fn_set_error("axis must be specified when the input has more than one dimension");
        return TSR_EARG;
    }
    /* broadcast loop shapes (right-aligned) */
    int32_t nd = 0;
    for (int i = 0; i < nin; i++) if (lnd[i] > nd) nd = lnd[i];
    g->loop_nd = nd;
    for (int32_t k = 0; k < nd; k++) {
        int64_t len = 1;
        for (int i = 0; i < nin; i++) {
            const int32_t kk = k - (nd - lnd[i]);
            if (kk < 0) continue;
            const int64_t l = lshape[i][kk];
            if (l == 1) continue;
            if (len != 1 && len != l) {
                fn_set_error("operands could not be broadcast together");
                return TSR_ESHAPE;
            }
            len = l;
        }
        g->loop_shape[k] = len;
        for (int i = 0; i < nin; i++) {
            const int32_t kk = k - (nd - lnd[i]);
            g->in_loop_st[i][k] = (kk < 0 || lshape[i][kk] == 1) ? 0 : lst[i][kk];
        }
    }
    if (tsr_shape_size(nd, g->loop_shape) < 0) return TSR_ENOMEM;
    for (int o = 0; o < g->nout; o++) {
        const int r = core_len_rule(d, o, g, params, &g->out_core_nd[o], g->out_core_shape[o]);
        if (r != TSR_OK) return r;
        g->out_core_n[o] = tsr_shape_size(g->out_core_nd[o], g->out_core_shape[o]);
        if (g->out_core_n[o] < 0) return TSR_ENOMEM;
    }
    return TSR_OK;
}

/* where output o's dimensions go: pos_loop[c] for loop dimension c, pos_core[c] for core dimension c */
static void out_layout(const fn_def *d, const gplan *g, int o, int *pos_loop, int *pos_core)
{
    const int cn = g->out_core_nd[o];
    const int code = d->out_core[o];
    if ((code & CORE_INPLACE) && cn == 1 && g->core_axis0 >= 0 && !g->keepdims && g->loop_nd == g->in0_ndim - 1) {
        const int p = g->core_axis0;
        pos_core[0] = p;
        for (int c = 0; c < g->loop_nd; c++) pos_loop[c] = c < p ? c : c + 1;
        return;
    }
    const int first = (code & CORE_FIRST) != 0;
    for (int c = 0; c < cn; c++) pos_core[c] = first ? c : g->loop_nd + c;
    for (int c = 0; c < g->loop_nd; c++) pos_loop[c] = first ? cn + c : c;
}

int tsr_fn_gufunc_shape(int id, int nin, const tsr_array *in, const uint64_t *axmask, int keepdims,
                        const double *params, int32_t *out_ndim, int64_t *out_shape)
{
    const fn_def *d = def_of(id);
    if (!d || d->kind != FN_GUFUNC || !in || !axmask || !out_ndim || !out_shape) return TSR_EARG;
    gplan g;
    const int r = gplan_make(d, nin, in, axmask, keepdims, params, &g);
    if (r != TSR_OK) return r;
    for (int o = 0; o < d->nout; o++) {
        const int32_t n = g.loop_nd + g.out_core_nd[o];
        if (n > TSR_MAXDIM) return TSR_EDIM;
        out_ndim[o] = n;
        int64_t *s = out_shape + (int64_t)o * TSR_MAXDIM;
        int pl[TSR_MAXDIM], pc[TSR_MAXDIM];
        out_layout(d, &g, o, pl, pc);
        for (int c = 0; c < g.out_core_nd[o]; c++) s[pc[c]] = g.out_core_shape[o][c];
        for (int c = 0; c < g.loop_nd; c++) s[pl[c]] = g.loop_shape[c];
    }
    return TSR_OK;
}

/* gather the core of input i at byte address base into dst (C order over the core axes) */
static inline int64_t load_as_int64(int dtype, const char *p)
{
    switch (dtype) {
    case TSR_I64: return *(const int64_t *)p;
    case TSR_I32: return (int64_t)*(const int32_t *)p;
    case TSR_U8: return (int64_t)*(const uint8_t *)p;
    case TSR_BOOL: return *(const uint8_t *)p ? 1 : 0;
    default: return 0;
    }
}

/* CORE_INT64 mode: the core as int64 values (dst is the same 8-byte scratch, reinterpreted) */
static void gather_i64(const gplan *g, int i, int dtype, const char *base, int64_t *dst)
{
    const int32_t cn = g->core_nd[i];
    if (cn == 0) { dst[0] = load_as_int64(dtype, base); return; }
    const int64_t total = g->core_n[i];
    if (total == 0) return;
    const int64_t inner = g->core_shape[i][cn - 1], ist = g->core_st[i][cn - 1];
    int64_t idx[TSR_MAXDIM] = {0};
    int64_t k = 0;
    const char *p = base;
    for (;;) {
        for (int64_t t = 0; t < inner; t++) dst[k++] = load_as_int64(dtype, p + t * ist);
        int d = cn - 2;
        for (; d >= 0; d--) {
            if (++idx[d] < g->core_shape[i][d]) { p += g->core_st[i][d]; break; }
            p -= g->core_st[i][d] * (g->core_shape[i][d] - 1);
            idx[d] = 0;
        }
        if (d < 0) break;
    }
}

static void gather(const gplan *g, int i, int dtype, const char *base, double *dst)
{
    const int32_t cn = g->core_nd[i];
    if (cn == 0) { dst[0] = load_as_double(dtype, base); return; }
    const int64_t total = g->core_n[i];
    if (total == 0) return;
    const int64_t inner = g->core_shape[i][cn - 1], ist = g->core_st[i][cn - 1];
    int64_t idx[TSR_MAXDIM] = {0};
    int64_t k = 0;
    const char *p = base;
    for (;;) {
        if (dtype == TSR_F64) for (int64_t t = 0; t < inner; t++) dst[k++] = *(const double *)(p + t * ist);
        else for (int64_t t = 0; t < inner; t++) dst[k++] = load_as_double(dtype, p + t * ist);
        int d = cn - 2;
        for (; d >= 0; d--) {
            if (++idx[d] < g->core_shape[i][d]) { p += g->core_st[i][d]; break; }
            p -= g->core_st[i][d] * (g->core_shape[i][d] - 1);
            idx[d] = 0;
        }
        if (d < 0) break;
    }
}

int tsr_fn_gufunc(int id, int nin, const tsr_array *in, const uint64_t *axmask, int keepdims, const double *params,
                  int nout, const tsr_array *out)
{
    const fn_def *d = def_of(id);
    if (!d || d->kind != FN_GUFUNC || !in || !axmask || !out || nout != d->nout) return TSR_EARG;
    g_err[0] = '\0';
    gplan g;
    int r = gplan_make(d, nin, in, axmask, keepdims, params, &g);
    if (r != TSR_OK) return r;
    for (int i = 0; i < nin; i++)
        if (in[i].dtype == TSR_C128 || in[i].dtype < 0 || in[i].dtype >= TSR_NDTYPES) return TSR_ETYPE;
    /* output strides: loop part and core part */
    int64_t out_loop_st[4][TSR_MAXDIM], out_core_st[4][TSR_MAXDIM];
    for (int o = 0; o < nout; o++) {
        if (out[o].ndim != g.loop_nd + g.out_core_nd[o]) return TSR_ESHAPE;
        const int cn = g.out_core_nd[o];
        int pl[TSR_MAXDIM], pc[TSR_MAXDIM];
        out_layout(d, &g, o, pl, pc);
        for (int c = 0; c < g.loop_nd; c++) {
            if (out[o].shape[pl[c]] != g.loop_shape[c]) return TSR_ESHAPE;
            out_loop_st[o][c] = out[o].strides[pl[c]];
        }
        for (int c = 0; c < cn; c++) {
            if (out[o].shape[pc[c]] != g.out_core_shape[o][c]) return TSR_ESHAPE;
            out_core_st[o][c] = out[o].strides[pc[c]];
        }
    }
    const int64_t loops = tsr_shape_size(g.loop_nd, g.loop_shape);
    if (loops == 0) return TSR_OK;

    /* nan_policy (last parameter when supported): 0 propagate, 1 omit, 2 raise */
    int nanpol = 0, nparams = 0;
    if (d->nanpolicy) {
        const char *p = d->params;
        if (p && *p) { nparams = 1; for (; *p; p++) if (*p == ',') nparams++; }
        nanpol = params ? (int)params[nparams] : 0;
    }

    /* scratch: gathered cores + output cores + the core's own scratch */
    int64_t in_total = 0, out_total = 0;
    for (int i = 0; i < nin; i++) in_total += g.core_n[i];
    for (int o = 0; o < nout; o++) out_total += g.out_core_n[o] ? g.out_core_n[o] : 1;
    int64_t own = d->scratch ? d->scratch(g.core_n, params) : 0;
    if (own < 0) return TSR_ENOMEM;
    const int64_t bytes = (in_total + out_total) * 8 + own + 64;
    char *mem = (char *)tsr_alloc(bytes);
    if (!mem) return TSR_ENOMEM;
    double *xs[4], *os[4];
    double *q = (double *)mem;
    for (int i = 0; i < nin; i++) { xs[i] = q; q += g.core_n[i]; }
    for (int o = 0; o < nout; o++) { os[o] = q; q += g.out_core_n[o] ? g.out_core_n[o] : 1; }
    void *own_scratch = own ? (void *)(((uintptr_t)q + 63) & ~(uintptr_t)63) : NULL;

    /* NumPy reduces pairwise only when the reduced axis is the innermost (smallest-stride) axis of input 0 */
    unsigned flags = 0;
    if (nin > 0 && g.core_nd[0] > 0 && in[0].ndim > 1) {
        int best = -1;
        int64_t bs = 0;
        for (int32_t k = 0; k < in[0].ndim; k++) {
            if (in[0].shape[k] <= 1) continue;
            const int64_t s = in[0].strides[k] < 0 ? -in[0].strides[k] : in[0].strides[k];
            if (best < 0 || s < bs) { best = k; bs = s; }
        }
        if (best >= 0 && !(axmask[0] & ((uint64_t)1 << best))) flags |= FN_SEQUENTIAL;
    }
    if ((d->out_core[0] & CORE_NOBOOL) && nin >= 1 && in[0].dtype == TSR_BOOL) {
        fn_set_error("numpy boolean subtract, the `-` operator, is not supported, use the bitwise_xor, the `^` operator, or the logical_xor function instead.");
        tsr_free(mem, bytes);
        return TSR_ETYPE;
    }
    /* CORE_INT64: exact integer arithmetic for integer input 0 (output 0 is then int64, CORE_INTLIKE) */
    const int int_mode = (d->out_core[0] & CORE_INT64) && nin >= 1 && out[0].dtype == TSR_I64
        && (in[0].dtype == TSR_I64 || in[0].dtype == TSR_I32 || in[0].dtype == TSR_U8 || in[0].dtype == TSR_BOOL);
    if (int_mode) flags |= FN_INT64;
    int64_t idx[TSR_MAXDIM] = {0};
    r = TSR_OK;
    for (int64_t L = 0; L < loops && r == TSR_OK; L++) {
        int64_t n[4];
        for (int i = 0; i < nin; i++) {
            const char *base = (const char *)in[i].data + in[i].offset;
            for (int c = 0; c < g.loop_nd; c++) base += idx[c] * g.in_loop_st[i][c];
            if (int_mode && i == 0) gather_i64(&g, i, in[i].dtype, base, (int64_t *)xs[i]);
            else gather(&g, i, in[i].dtype, base, xs[i]);
            n[i] = g.core_n[i];
        }
        int has_nan = 0;
        if (d->nanpolicy) {
            for (int i = 0; i < nin && !has_nan; i++)
                for (int64_t t = 0; t < n[i]; t++) if (isnan(xs[i][t])) { has_nan = 1; break; }
            if (has_nan && nanpol == 2) { fn_set_error("The input contains nan values"); r = TSR_EARG; break; }
            if (has_nan && nanpol == 1) {
                if (d->nanpolicy == 2 && nin >= 2) {      /* paired samples: drop a position if any is NaN */
                    int64_t k = 0;
                    for (int64_t t = 0; t < n[0]; t++) {
                        int bad = 0;
                        for (int i = 0; i < nin; i++) if (isnan(xs[i][t])) bad = 1;
                        if (!bad) { for (int i = 0; i < nin; i++) xs[i][k] = xs[i][t]; k++; }
                    }
                    for (int i = 0; i < nin; i++) n[i] = k;
                } else {
                    for (int i = 0; i < nin; i++) {
                        int64_t k = 0;
                        for (int64_t t = 0; t < n[i]; t++) if (!isnan(xs[i][t])) xs[i][k++] = xs[i][t];
                        n[i] = k;
                    }
                }
                has_nan = 0;
            }
        }
        if (has_nan) {
            for (int o = 0; o < nout; o++) for (int64_t t = 0; t < (g.out_core_n[o] ? g.out_core_n[o] : 1); t++) os[o][t] = FN_NAN;
        } else {
            r = fn_call_core(d->core, d->ctx, (const double *const *)xs, n, params, os, g.out_core_n, own_scratch, flags);
            if (r != TSR_OK) break;
        }
        /* scatter */
        for (int o = 0; o < nout; o++) {
            char *base = (char *)out[o].data + out[o].offset;
            for (int c = 0; c < g.loop_nd; c++) base += idx[c] * out_loop_st[o][c];
            const int cn = g.out_core_nd[o];
            const int as_int = int_mode && o == 0;          /* the core wrote int64 values into these slots */
            if (cn == 0) {
                if (as_int) memcpy(base, &os[o][0], 8);
                else store_double(out[o].dtype, base, os[o][0]);
                continue;
            }
            int64_t cidx[TSR_MAXDIM] = {0};
            for (int64_t t = 0; t < g.out_core_n[o]; t++) {
                char *p = base;
                for (int c = 0; c < cn; c++) p += cidx[c] * out_core_st[o][c];
                if (as_int) memcpy(p, &os[o][t], 8);
                else store_double(out[o].dtype, p, os[o][t]);
                for (int c = cn - 1; c >= 0; c--) { if (++cidx[c] < g.out_core_shape[o][c]) break; cidx[c] = 0; }
            }
        }
        for (int c = g.loop_nd - 1; c >= 0; c--) { if (++idx[c] < g.loop_shape[c]) break; idx[c] = 0; }
    }
    tsr_free(mem, bytes);
    return r;
}

/* ================================================================ routines */

void *fn_result_array(tsr_result *r, int dtype, int32_t ndim, const int64_t *shape)
{
    const int64_t n = tsr_shape_size(ndim, shape);
    if (n < 0 || ndim > TSR_MAXDIM) return NULL;
    int64_t bytes = n * tsr_itemsize(dtype);
    if (bytes < 64) bytes = 64;
    void *p = tsr_alloc(bytes);
    if (!p) return NULL;
    memset(r, 0, sizeof *r);
    r->kind = 3;
    r->bytes = bytes;
    tsr_array_init(&r->arr, dtype, ndim, shape, p);
    return p;
}

void fn_result_num(tsr_result *r, double v) { memset(r, 0, sizeof *r); r->kind = 1; r->num = v; }
void fn_result_int(tsr_result *r, int64_t v) { memset(r, 0, sizeof *r); r->kind = 5; r->num = (double)v; r->ival = v; }

/* a string result (kind 2): a NUL-terminated copy in arr.data (bytes = allocation size). The bindings read it
   as a PHP string. Used for dtype-returning functions (numpy.result_type, promote_types). */
int fn_result_str(tsr_result *r, const char *s)
{
    const int64_t len = (int64_t)strlen(s) + 1;
    const int64_t bytes = len > 64 ? len : 64;
    char *p = (char *)tsr_alloc(bytes);
    if (!p) return TSR_ENOMEM;
    memcpy(p, s, (size_t)len);
    memset(r, 0, sizeof *r);
    r->kind = 2;
    r->bytes = bytes;
    r->arr.data = p;
    r->arr.ndim = 1;
    r->arr.shape[0] = len - 1;
    return TSR_OK;
}

void fn_results_free(tsr_result *res, int n)
{
    for (int k = 0; k < n; k++) {
        if ((res[k].kind == 3 || res[k].kind == 2) && res[k].arr.data) tsr_free(res[k].arr.data, res[k].bytes);
        if (res[k].kind == 6 && res[k].arr.data) {
            fn_results_free((tsr_result *)res[k].arr.data, (int)res[k].arr.shape[0]);
            tsr_free(res[k].arr.data, res[k].bytes);
        }
        memset(&res[k], 0, sizeof res[k]);
    }
}

tsr_result *fn_result_seq(tsr_result *r, int64_t n)
{
    if (n < 0 || n > ((int64_t)1 << 40) / (int64_t)sizeof(tsr_result)) { fn_set_error("Maximum allowed size exceeded"); return NULL; }
    const int64_t bytes = (n > 0 ? n : 1) * (int64_t)sizeof(tsr_result);
    tsr_result *items = (tsr_result *)tsr_calloc(bytes);
    if (!items) return NULL;
    memset(r, 0, sizeof *r);
    r->kind = 6;
    r->bytes = bytes;
    r->arr.data = items;
    r->arr.ndim = 1;
    r->arr.shape[0] = n;
    r->arr.strides[0] = (int64_t)sizeof(tsr_result);
    return items;
}

double *fn_arg_doubles(const tsr_arg *a, int64_t *n)
{
    if (!a || a->kind != 3) return NULL;
    const tsr_array *x = &a->arr;
    if (x->dtype == TSR_C128 || x->dtype < 0 || x->dtype >= TSR_NDTYPES) return NULL;
    const int64_t size = tsr_shape_size(x->ndim, x->shape);
    if (size < 0) return NULL;
    double *out = (double *)tsr_alloc(size > 8 ? size * 8 : 64);
    if (!out) return NULL;
    int64_t idx[TSR_MAXDIM] = {0};
    for (int64_t k = 0; k < size; k++) {
        const char *p = (const char *)x->data + x->offset;
        for (int32_t d = 0; d < x->ndim; d++) p += idx[d] * x->strides[d];
        out[k] = load_as_double(x->dtype, p);
        for (int32_t d = x->ndim - 1; d >= 0; d--) { if (++idx[d] < x->shape[d]) break; idx[d] = 0; }
    }
    *n = size;
    return out;
}

void fn_free_doubles(double *p, int64_t n) { if (p) tsr_free(p, n > 8 ? n * 8 : 64); }

int tsr_fn_routine(int id, int nargs, const tsr_arg *args, int nres, tsr_result *res)
{
    const fn_def *d = def_of(id);
    if (!d || d->kind != FN_ROUTINE || !res || nres < d->nout || (nargs > 0 && !args)) return TSR_EARG;
    g_err[0] = '\0';
    for (int k = 0; k < nres; k++) memset(&res[k], 0, sizeof res[k]);
    const int rc = fn_call_routine(d->routine, d->ctx, args, nargs, res, nres);
    if (rc < 0) fn_results_free(res, nres);
    return rc;
}
