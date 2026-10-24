/* numpy.loadtxt: read a whitespace/delimiter-separated text file into a float64 array (numpy.loadtxt).
 * Supports delimiter, skiprows and comments; usecols and non-float dtypes are not handled here. The result is
 * squeezed as NumPy does: a single value -> scalar, a single row or column -> 1-D, otherwise 2-D. */
#include "fn.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
/* getline() with POSIX semantics (returns the line length, -1 at end of file or on error, grows *line as needed).
   POSIX systems use the C library's; Windows (MinGW, MSVC) has none, so it is implemented here. */
#ifdef _WIN32
#if defined(_MSC_VER) && !defined(ssize_t)
typedef long long ssize_t;
#endif
static long long tsr_getline(char **line, size_t *cap, FILE *fp)
{
    if (!line || !cap || !fp) return -1;
    if (!*line || *cap == 0) {
        size_t c = 128;
        char *p = (char *)realloc(*line, c);
        if (!p) return -1;
        *line = p;
        *cap = c;
    }
    size_t n = 0;
    int ch;
    while ((ch = fgetc(fp)) != EOF) {
        if (n + 2 > *cap) {
            if (*cap > ((size_t)-1) / 2) return -1;
            size_t c = *cap * 2;
            char *p = (char *)realloc(*line, c);
            if (!p) return -1;
            *line = p;
            *cap = c;
        }
        (*line)[n++] = (char)ch;
        if (ch == '\n') break;
    }
    if (n == 0) return -1;
    (*line)[n] = '\0';
    return (long long)n;
}
#else
#define tsr_getline getline
#endif

static int push(double **d, int64_t *cap, int64_t *cnt, double v)
{
    if (*cnt >= *cap) {
        const int64_t nc = *cap ? *cap * 2 : 128;
        double *nd = (double *)realloc(*d, (size_t)nc * sizeof(double));
        if (!nd) return -1;
        *d = nd; *cap = nc;
    }
    (*d)[(*cnt)++] = v;
    return 0;
}

static int r_loadtxt(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (args[0].kind != 2 || !args[0].str) { fn_set_error("loadtxt: the file name must be a string"); return TSR_EARG; }
    const char *fname = args[0].str;
    const char *delim = (nargs > 1 && args[1].kind == 2 && args[1].str && args[1].str[0]) ? args[1].str : NULL;
    int64_t skiprows = 0;
    if (nargs > 2 && args[2].kind == 1) skiprows = (args[2].flags & 1) ? args[2].ival : (int64_t)args[2].num;
    const char *comments = (nargs > 3 && args[3].kind == 2 && args[3].str) ? args[3].str : "#";

    FILE *fp = fopen(fname, "r");
    if (!fp) { fn_set_error("loadtxt: could not open '%s'", fname); return TSR_EARG; }
    double *data = NULL; int64_t cap = 0, cnt = 0, nrows = 0, ncols = -1;
    char *line = NULL; size_t lcap = 0; ssize_t len; int64_t lineno = 0; int rc = TSR_OK;
    while ((len = tsr_getline(&line, &lcap, fp)) != -1) {
        if (lineno++ < skiprows) continue;
        if (comments && comments[0]) { char *h = strstr(line, comments); if (h) *h = '\0'; }
        int64_t rowcols = 0;
        if (delim) {
            char *s = line;
            for (;;) {
                char *d = strchr(s, delim[0]);
                char *end = d ? d : s + strlen(s);
                while (s < end && (*s == ' ' || *s == '\t')) s++;                 /* trim field */
                char *fe = end; while (fe > s && (fe[-1] == ' ' || fe[-1] == '\t' || fe[-1] == '\n' || fe[-1] == '\r')) fe--;
                if (fe > s) { char *ep; double v = strtod(s, &ep); if (ep == s) { rc = TSR_EARG; break; } if (push(&data, &cap, &cnt, v) < 0) { rc = TSR_ENOMEM; break; } rowcols++; }
                else if (d) { if (push(&data, &cap, &cnt, 0.0) < 0) { rc = TSR_ENOMEM; break; } rowcols++; }  /* empty field */
                if (!d) break;
                s = d + 1;
            }
        } else {
            char *s = line, *ep;
            for (;;) {
                while (*s == ' ' || *s == '\t' || *s == '\n' || *s == '\r') s++;
                if (*s == '\0') break;
                double v = strtod(s, &ep);
                if (ep == s) { rc = TSR_EARG; break; }
                if (push(&data, &cap, &cnt, v) < 0) { rc = TSR_ENOMEM; break; }
                rowcols++; s = ep;
            }
        }
        if (rc != TSR_OK) break;
        if (rowcols == 0) continue;                       /* blank / comment-only line */
        if (ncols < 0) ncols = rowcols;
        else if (rowcols != ncols) { fn_set_error("loadtxt: row %lld has %lld columns, expected %lld", (long long)lineno, (long long)rowcols, (long long)ncols); rc = TSR_EARG; break; }
        nrows++;
    }
    free(line);
    fclose(fp);
    if (rc != TSR_OK) { free(data); return rc; }
    if (ncols < 0) { ncols = 0; }                         /* empty file -> (0,) */

    int32_t ond; int64_t osh[2];
    if (nrows == 1 && ncols == 1) {                       /* a single value -> scalar */
        const double v = data[0]; free(data); fn_result_num(&res[0], v); return TSR_OK;
    } else if (nrows == 1) { ond = 1; osh[0] = ncols; }
    else if (ncols == 1) { ond = 1; osh[0] = nrows; }
    else { ond = 2; osh[0] = nrows; osh[1] = ncols; }
    double *out = (double *)fn_result_array(&res[0], TSR_F64, ond, osh);
    if (!out) { free(data); return TSR_ENOMEM; }
    memcpy(out, data, sizeof(double) * (size_t)(nrows * ncols));
    free(data);
    return TSR_OK;
}

/* a subset of NumPy dtype names Tessero can represent, with byte sizes */
static int io_dtype(const char *s, int *dt, int *isz)
{
    const struct { const char *n; int d; int z; } T[] = {
        {"float64", TSR_F64, 8}, {"float32", TSR_F32, 4}, {"int64", TSR_I64, 8}, {"int32", TSR_I32, 4},
        {"uint8", TSR_U8, 1}, {"complex128", TSR_C128, 16},
        {"f8", TSR_F64, 8}, {"f4", TSR_F32, 4}, {"i8", TSR_I64, 8}, {"i4", TSR_I32, 4}, {"u1", TSR_U8, 1}, {"d", TSR_F64, 8},
    };
    for (size_t i = 0; i < sizeof T / sizeof T[0]; i++)
        if (strcmp(s, T[i].n) == 0) { *dt = T[i].d; *isz = T[i].z; return TSR_OK; }
    fn_set_error("fromfile: unsupported dtype '%s'", s);
    return TSR_EARG;
}

static void io_store(void *buf, int64_t i, int dt, double v)
{
    switch (dt) {
    case TSR_F64: ((double *)buf)[i] = v; break;
    case TSR_F32: ((float *)buf)[i] = (float)v; break;
    case TSR_I64: ((int64_t *)buf)[i] = (int64_t)v; break;
    case TSR_I32: ((int32_t *)buf)[i] = (int32_t)v; break;
    case TSR_U8: ((uint8_t *)buf)[i] = (uint8_t)v; break;
    default: ((double *)buf)[i] = v; break;
    }
}

/* fromfile(fname, dtype='float64', count=-1, sep=None, offset=0): read a binary (sep=None) or text file into a
   1-D array of the given dtype (numpy.fromfile). */
static int r_fromfile(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (args[0].kind != 2 || !args[0].str) { fn_set_error("fromfile: the file name must be a string"); return TSR_EARG; }
    const char *fname = args[0].str;
    int dt = TSR_F64, isz = 8, rc;
    if (nargs > 1 && args[1].kind == 2 && args[1].str && (rc = io_dtype(args[1].str, &dt, &isz)) < 0) return rc;
    int64_t count = -1;
    if (nargs > 2 && args[2].kind == 1) count = (args[2].flags & 1) ? args[2].ival : (int64_t)args[2].num;
    const char *sep = (nargs > 3 && args[3].kind == 2 && args[3].str && args[3].str[0]) ? args[3].str : NULL;
    int64_t offset = 0;
    if (nargs > 4 && args[4].kind == 1) offset = (args[4].flags & 1) ? args[4].ival : (int64_t)args[4].num;

    if (!sep) {                                               /* binary */
        FILE *fp = fopen(fname, "rb");
        if (!fp) { fn_set_error("fromfile: could not open '%s'", fname); return TSR_EARG; }
        fseek(fp, 0, SEEK_END);
        const long fsz = ftell(fp);
        int64_t avail = ((int64_t)fsz - offset) / isz;
        if (avail < 0) avail = 0;
        int64_t n = (count < 0) ? avail : (count < avail ? count : avail);
        fseek(fp, (long)offset, SEEK_SET);
        void *out = fn_result_array(&res[0], dt, 1, (int64_t[]){n});
        if (!out) { fclose(fp); return TSR_ENOMEM; }
        if (n > 0 && fread(out, (size_t)isz, (size_t)n, fp) != (size_t)n) { fclose(fp); fn_set_error("fromfile: short read"); return TSR_EARG; }
        fclose(fp);
        return TSR_OK;
    }
    /* text: parse numbers separated by sep (whitespace if sep is whitespace), up to count */
    FILE *fp = fopen(fname, "r");
    if (!fp) { fn_set_error("fromfile: could not open '%s'", fname); return TSR_EARG; }
    fseek(fp, 0, SEEK_END); long fsz = ftell(fp); fseek(fp, 0, SEEK_SET);
    char *text = (char *)malloc((size_t)(fsz > 0 ? fsz : 0) + 1);
    if (!text) { fclose(fp); return TSR_ENOMEM; }
    size_t got = fread(text, 1, (size_t)(fsz > 0 ? fsz : 0), fp); text[got] = '\0'; fclose(fp);
    int ws = 1; for (const char *p = sep; *p; p++) if (*p != ' ' && *p != '\t' && *p != '\n') { ws = 0; break; }
    double *vals = NULL; int64_t cap = 0, cnt = 0;
    char *s = text;
    while (*s && (count < 0 || cnt < count)) {
        while (*s == ' ' || *s == '\t' || *s == '\n' || *s == '\r' || (!ws && *s == sep[0])) s++;
        if (!*s) break;
        char *ep; double v = strtod(s, &ep);
        if (ep == s) break;
        if (push(&vals, &cap, &cnt, v) < 0) { free(text); free(vals); return TSR_ENOMEM; }
        s = ep;
    }
    free(text);
    void *out = fn_result_array(&res[0], dt, 1, (int64_t[]){cnt});
    if (!out) { free(vals); return TSR_ENOMEM; }
    for (int64_t i = 0; i < cnt; i++) io_store(out, i, dt, vals[i]);
    free(vals);
    return TSR_OK;
}


/* numpy.genfromtxt: like loadtxt but missing/empty fields become NaN, with skip_header/skip_footer.
   Float64 output, squeezed as NumPy does. usecols and structured dtypes are not handled. */
static int r_genfromtxt(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (args[0].kind != 2 || !args[0].str) { fn_set_error("genfromtxt: the file name must be a string"); return TSR_EARG; }
    const char *fname = args[0].str;
    const char *delim = (nargs > 1 && args[1].kind == 2 && args[1].str && args[1].str[0]) ? args[1].str : NULL;
    int64_t skip_header = 0, skip_footer = 0;
    if (nargs > 2 && args[2].kind == 1) skip_header = (args[2].flags & 1) ? args[2].ival : (int64_t)args[2].num;
    if (nargs > 3 && args[3].kind == 1) skip_footer = (args[3].flags & 1) ? args[3].ival : (int64_t)args[3].num;
    const char *comments = (nargs > 4 && args[4].kind == 2 && args[4].str) ? args[4].str : "#";

    FILE *fp = fopen(fname, "r");
    if (!fp) { fn_set_error("genfromtxt: could not open '%s'", fname); return TSR_EARG; }
    double *data = NULL; int64_t cap = 0, cnt = 0, nrows = 0, ncols = -1;
    char *line = NULL; size_t lcap = 0; ssize_t len; int64_t lineno = 0; int rc = TSR_OK;
    while ((len = tsr_getline(&line, &lcap, fp)) != -1) {
        (void)len;
        if (lineno++ < skip_header) continue;
        if (comments && comments[0]) { char *h = strstr(line, comments); if (h) *h = '\0'; }
        int64_t rowcols = 0;
        int hascontent = 0;
        for (char *p = line; *p; p++) if (*p != ' ' && *p != '\t' && *p != '\n' && *p != '\r') { hascontent = 1; break; }
        if (!hascontent) {
            /* blank line: no fields */
        } else if (delim) {
            char *eol = line + strlen(line);
            while (eol > line && (eol[-1] == '\n' || eol[-1] == '\r')) *--eol = '\0';
            char *s = line;                       /* split into (ndelim + 1) fields; empty field -> NaN */
            for (;;) {
                char *d = strchr(s, delim[0]);
                char *end = d ? d : eol;
                char *fs = s; while (fs < end && (*fs == ' ' || *fs == '\t')) fs++;
                char *fe = end; while (fe > fs && (fe[-1] == ' ' || fe[-1] == '\t')) fe--;
                double v = NAN;
                if (fe > fs) { char *ep; v = strtod(fs, &ep); if (ep == fs) v = NAN; }
                if (push(&data, &cap, &cnt, v) < 0) { rc = TSR_ENOMEM; break; }
                rowcols++;
                if (!d) break;
                s = d + 1;
            }
        } else {
            char *s = line, *ep;
            for (;;) {
                while (*s == ' ' || *s == '\t' || *s == '\n' || *s == '\r') s++;
                if (*s == '\0') break;
                double v = strtod(s, &ep);
                if (ep == s) { v = NAN; while (*s && *s != ' ' && *s != '\t' && *s != '\n' && *s != '\r') s++; }
                else s = ep;
                if (push(&data, &cap, &cnt, v) < 0) { rc = TSR_ENOMEM; break; }
                rowcols++;
            }
        }
        if (rc != TSR_OK) break;
        if (rowcols == 0) continue;
        if (ncols < 0) ncols = rowcols;
        else if (rowcols != ncols) { fn_set_error("genfromtxt: row %lld has %lld columns, expected %lld", (long long)lineno, (long long)rowcols, (long long)ncols); rc = TSR_EARG; break; }
        nrows++;
    }
    free(line);
    fclose(fp);
    if (rc != TSR_OK) { free(data); return rc; }
    if (skip_footer > 0 && nrows > 0) {
        int64_t drop = skip_footer < nrows ? skip_footer : nrows;
        nrows -= drop; cnt -= drop * ncols;
    }
    if (ncols < 0) ncols = 0;
    int32_t ond; int64_t osh[2];
    if (nrows == 1 && ncols == 1) { const double v = data[0]; free(data); fn_result_num(&res[0], v); return TSR_OK; }
    else if (nrows == 1) { ond = 1; osh[0] = ncols; }
    else if (ncols == 1) { ond = 1; osh[0] = nrows; }
    else { ond = 2; osh[0] = nrows; osh[1] = ncols; }
    double *out = (double *)fn_result_array(&res[0], TSR_F64, ond, osh);
    if (!out) { free(data); return TSR_ENOMEM; }
    memcpy(out, data, sizeof(double) * (size_t)(nrows * ncols));
    free(data);
    return TSR_OK;
}


/* numpy.savetxt: write a 1-D or 2-D array to a text file. Supports fmt (a single printf float conversion),
   delimiter, newline, header, footer and comments. Returns the number of bytes written. */
static int st_valid_fmt(const char *fmt)
{
    int specs = 0;
    for (const char *p = fmt; *p; p++) {
        if (*p != '%') continue;
        if (p[1] == '%') { p++; continue; }
        const char *q = p + 1;
        while (*q && !strchr("eEfFgGdiu", *q)) {
            if (*q == 'n' || *q == 's' || *q == 'p') return 0;      /* reject dangerous conversions */
            q++;
        }
        if (!*q || !strchr("eEfFgG", *q)) return 0;                 /* only float conversions supported */
        specs++;
        p = q;
    }
    return specs == 1;
}

static int r_savetxt(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (args[0].kind != 2 || !args[0].str) { fn_set_error("savetxt: the file name must be a string"); return TSR_EARG; }
    if (args[1].kind != 3) { fn_set_error("savetxt: X must be an array"); return TSR_EARG; }
    const char *fname = args[0].str;
    const char *fmt = (nargs > 2 && args[2].kind == 2 && args[2].str && args[2].str[0]) ? args[2].str : "%.18e";
    const char *delim = (nargs > 3 && args[3].kind == 2 && args[3].str) ? args[3].str : " ";
    const char *newline = (nargs > 4 && args[4].kind == 2 && args[4].str) ? args[4].str : "\n";
    const char *header = (nargs > 5 && args[5].kind == 2 && args[5].str) ? args[5].str : "";
    const char *footer = (nargs > 6 && args[6].kind == 2 && args[6].str) ? args[6].str : "";
    const char *comments = (nargs > 7 && args[7].kind == 2 && args[7].str) ? args[7].str : "# ";
    if (!st_valid_fmt(fmt)) { fn_set_error("savetxt: unsupported fmt '%s' (one float conversion expected)", fmt); return TSR_EARG; }
    const int ndim = args[1].arr.ndim;
    if (ndim > 2) { fn_set_error("savetxt: X must be 1-D or 2-D"); return TSR_EARG; }
    int64_t n;
    double *a = fn_arg_doubles(&args[1], &n);
    if (!a) return TSR_ENOMEM;
    int64_t rows = ndim <= 1 ? n : args[1].arr.shape[0];
    int64_t cols = ndim <= 1 ? 1 : args[1].arr.shape[1];

    FILE *fp = fopen(fname, "w");
    if (!fp) { fn_free_doubles(a, n); fn_set_error("savetxt: could not open '%s'", fname); return TSR_EARG; }
    long written = 0;
    char buf[64];
    if (header[0]) { written += fprintf(fp, "%s%s%s", comments, header, newline); }
    for (int64_t r = 0; r < rows; r++) {
        for (int64_t c = 0; c < cols; c++) {
            snprintf(buf, sizeof buf, fmt, a[r * cols + c]);
            written += (long)fputs(buf, fp) >= 0 ? (long)strlen(buf) : 0;
            if (c + 1 < cols) { fputs(delim, fp); written += (long)strlen(delim); }
        }
        fputs(newline, fp); written += (long)strlen(newline);
    }
    if (footer[0]) { written += fprintf(fp, "%s%s%s", comments, footer, newline); }
    fclose(fp);
    fn_free_doubles(a, n);
    fn_result_num(&res[0], (double)written);
    return TSR_OK;
}

static const fn_def DEFS[] = {
    ROUTINE("np.loadtxt", 1, "fname, delimiter=None, skiprows=0, comments=None, usecols=None", "out", r_loadtxt, NULL, "Load a text file of numbers into a float64 array (numpy.loadtxt)."),
    ROUTINE("np.genfromtxt", 1, "fname, delimiter=None, skip_header=0, skip_footer=0, comments=None, usecols=None", "out", r_genfromtxt, NULL, "Load a text file of numbers into a float64 array, missing fields as NaN (numpy.genfromtxt)."),
    ROUTINE("np.savetxt", 1, "fname, X, fmt=None, delimiter=None, newline=None, header=None, footer=None, comments=None", "nbytes", r_savetxt, NULL, "Save an array to a text file (numpy.savetxt)."),
    ROUTINE("np.fromfile", 1, "fname, dtype=None, count=-1, sep=None, offset=0", "out", r_fromfile, NULL, "Read a binary or text file into a 1-D array of the given dtype (numpy.fromfile)."),
};

const fn_table TSR_NP_IO_TABLE = {DEFS, (int)(sizeof DEFS / sizeof DEFS[0])};
