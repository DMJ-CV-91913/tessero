# C ABI (libtessero)

`csrc/include/tessero.h` is the complete ABI. It is parsed verbatim by `FFI::cdef()`/`FFI::load()` and compiled into ext-tessero, so it holds declarations only. Strides are in bytes; every kernel returns 0 or a negative error code (see [error codes](../error-codes.md)).

## Conventions and version

```c
/*
 * libtessero - native kernels for the Tessero PHP array library.
 *
 * This header is the complete public ABI. It is parsed verbatim by PHP's
 * FFI::cdef()/FFI::load(), so it contains declarations only: no macros,
 * no includes, no inline code. Constants (dtypes, ops, error codes) are
 * mirrored in PHP (Tessero\Native\Abi).
 *
 * Conventions
 *   - dtypes: 0 float64, 1 float32, 2 int64, 3 int32, 4 uint8, 5 bool, 6 complex128
 *   - strides are in BYTES and may be negative or zero (broadcast)
 *   - every kernel returns 0 on success or a negative error code:
 *       -1 bad argument, -2 index out of bounds, -3 out of memory / budget,
 *       -4 operation not supported for this dtype, -5 too many dimensions (max 32),
 *       -6 shapes cannot be broadcast / reshaped, -7 view impossible (not contiguous),
 *       -8 iterative method did not converge, -9 operating-system I/O error (tsr_mmap_error() has the text)
 */

const char *tsr_version(void);
int tsr_simd_level(void);                 /* 0 baseline, 1 AVX2 (x86-64-v3), 2 AVX-512 (x86-64-v4), 3 NEON */
```

## Memory: 64-byte aligned, NOT zero-filled, counted against a budget

```c
void *tsr_alloc(int64_t bytes);           /* NULL when out of memory, over budget, or (no budget) larger than the RAM */
void *tsr_calloc(int64_t bytes);          /* zero-filled variant */
void tsr_free(void *p, int64_t bytes);
int64_t tsr_allocated(void);
int64_t tsr_peak(void);
void tsr_set_budget(int64_t bytes);       /* 0 = unlimited */
int64_t tsr_budget(void);
```

## Threads (OpenMP builds; 1 = serial, the default)

```c
void tsr_set_threads(int n);
int tsr_get_threads(void);
int tsr_openmp(void);                     /* 1 if built with OpenMP */
```

## Array metadata: a strided view of memory it does not own

```c
typedef struct tsr_array {
    void *data;                           /* start of the owning block */
    int64_t offset;                       /* byte offset of element [0, ..., 0] */
    int32_t ndim;
    int32_t dtype;
    int64_t shape[32];
    int64_t strides[32];                  /* bytes; 0 = broadcast, negative = reversed */
} tsr_array;
int tsr_array_init(tsr_array *a, int dtype, int32_t ndim, const int64_t *shape, void *data);  /* C-contiguous */
int64_t tsr_array_size(const tsr_array *a);
int tsr_array_is_contiguous(const tsr_array *a);
int tsr_broadcast_shape(int32_t nda, const int64_t *sa, int32_t ndb, const int64_t *sb, int32_t *out_nd, int64_t *out_shape);
int tsr_broadcast_strides(const tsr_array *a, int32_t nd, const int64_t *shape, int64_t *out_strides);
int tsr_array_slice(const tsr_array *src, const char *spec, tsr_array *out);   /* "1:10:2, ::-1", "..., None, 0" */
int tsr_array_transpose(const tsr_array *src, const int32_t *axes, tsr_array *out); /* axes NULL = reverse */
int tsr_array_reshape(const tsr_array *src, int32_t ndim, const int64_t *shape, tsr_array *out); /* -7 if a copy is needed */
void *tsr_array_at(const tsr_array *a, const int64_t *index);                 /* NULL when out of bounds */
int tsr_array_binary(int op, const tsr_array *a, const tsr_array *b, tsr_array *out); /* out has the broadcast shape */
int64_t tsr_array_json(const tsr_array *a, char *buf, int64_t cap);            /* returns the length needed */
```

## Element-wise

```c
int tsr_copy(int dtype_in, int dtype_out, int32_t ndim, const int64_t *shape,
             const void *in, const int64_t *sin, void *out, const int64_t *sout);   /* also casts */
int tsr_fill(int dtype, int64_t n, void *out, const void *value);
int tsr_arange(int dtype, int64_t n, double start, double step, void *out);
int tsr_binary(int op, int dtype, int32_t ndim, const int64_t *shape,
               const void *a, const int64_t *sa, const void *b, const int64_t *sb,
               void *out, const int64_t *sout);
int tsr_unary(int op, int dtype, int32_t ndim, const int64_t *shape,
              const void *in, const int64_t *sin, void *out, const int64_t *sout);
int tsr_where(int dtype, int32_t ndim, const int64_t *shape, const void *cond, const int64_t *sc,
              const void *a, const int64_t *sa, const void *b, const int64_t *sb, void *out, const int64_t *sout);
```

## Loop ufuncs (cbrt, erf, gamma, logaddexp, fmax, fmod, ...): ids are 0 .. tsr_ufunc_count()-1

```c
tsr_ufunc_resolve() maps the promoted input dtype to the loop dtype (cast the inputs to it) and the output
   dtype. Returns 0: call tsr_ufunc; 1: identity on this dtype (copy the input); 2: use tsr_binary(native_op);
   < 0: not supported. Operands are already broadcast to `shape` (zero strides); out may be an input with the
   identical layout but must not partially overlap one. */
int tsr_ufunc_count(void);
const char *tsr_ufunc_name(int id);                 /* NULL when out of range */
const char *tsr_ufunc_summary(int id);
int tsr_ufunc_nin(int id);                          /* 1 or 2 */
int tsr_ufunc_find(const char *name);               /* -1 when unknown */
int tsr_ufunc_resolve(int id, int dtype, int *loop_dtype, int *out_dtype, int *native_op);
int tsr_ufunc(int id, int loop_dtype, int32_t ndim, const int64_t *shape,
              const void *a, const int64_t *sa, const void *b, const int64_t *sb,
              void *out, const int64_t *sout);    /* b/sb ignored for one-input ufuncs */
```

## Memory-mapped files; mode 0 'r', 1 'r+', 2 'w+', 3 'c' (copy-on-write)

```c
*ndim = -1 derives a 1-D shape from the file size. tsr_mmap_plan is the pure validation step (no I/O).
   The data pointer stays valid until tsr_mmap_close(handle). Mapped bytes are not counted by tsr_allocated. */
int tsr_mmap_plan(int dtype, int mode, int32_t *ndim, int64_t *shape, int64_t offset, int64_t file_size,
                  int64_t granularity, int64_t *bytes, int64_t *map_start, int64_t *map_length, int64_t *new_size);
int tsr_mmap_open(const char *path, int mode, int dtype, int32_t *ndim, int64_t *shape, int64_t offset,
                  void **handle, void **data);
int tsr_mmap_flush(void *handle, int sync);         /* sync 0: schedule write-back; 1: MS_SYNC + fsync */
void tsr_mmap_close(void *handle);
int64_t tsr_mmap_length(void *handle);
int64_t tsr_mmap_bytes(void);                       /* all mappings in the process */
int64_t tsr_mmap_granularity(void);
const char *tsr_mmap_error(void);                   /* message for the last failure on this thread */
```

## .npy format (v1, v2, v3; little-endian dtypes)

```c
tsr_npy_header parses the preamble of buf (at least the first len bytes of the file); returns 0 and
   the data offset, -1 malformed, -4 unsupported dtype or byte order, -5 more than 32 dimensions, -6 too large.
   tsr_npy_write_header returns the header length (a multiple of 64); with buf NULL or cap too small it only
   reports the length. */
int tsr_npy_header(const char *buf, int64_t len, int *dtype, int32_t *ndim, int64_t *shape, int *fortran,
                   int64_t *data_offset);
int64_t tsr_npy_write_header(int dtype, int32_t ndim, const int64_t *shape, int fortran, char *buf, int64_t cap);
```

## Reductions along one axis; out has the shape without that axis

```c
int tsr_reduce(int op, int dtype, int32_t ndim, const int64_t *shape, const void *in, const int64_t *sin,
               int32_t axis, int out_dtype, void *out, const int64_t *sout);
int tsr_moments(int dtype, int32_t ndim, const int64_t *shape, const void *in, const int64_t *sin,
                int32_t axis, double *mean, const int64_t *smean, double *m2, const int64_t *sm2);
```

## Indexing (contiguous sources; itemsize in bytes)

```c
int tsr_take(int64_t itemsize, int64_t outer, int64_t axis_len, int64_t inner, const void *src,
             const int64_t *idx, int64_t nidx, void *dst);
int tsr_put(int64_t itemsize, int64_t outer, int64_t axis_len, int64_t inner, void *dst,
            const int64_t *idx, int64_t nidx, const void *values);
int64_t tsr_count_true(int64_t n, const uint8_t *mask);
int64_t tsr_compress(int64_t itemsize, int64_t n, const void *src, const uint8_t *mask, void *dst);
int tsr_mask_assign(int64_t itemsize, int64_t n, void *dst, const uint8_t *mask, const void *values, int64_t values_n);
int tsr_reduce_mid(int op, int dtype, int64_t outer, int64_t len, int64_t inner, const void *in, void *out); /* SUM/PROD/MIN/MAX, contiguous */
int tsr_cumulative(int op, int dtype, int64_t outer, int64_t len, int64_t inner, const void *in, void *out); /* op 0 sum, 1 prod; ints -> int64 */
int tsr_sort(int dtype, int64_t n, void *data);                       /* ascending, NaN last */
int tsr_argsort(int dtype, int64_t n, const void *data, int64_t *out); /* stable */
```

## Dense matrix product fallback (row-major, used when no BLAS is loaded)

```c
int tsr_matmul(int dtype, int64_t m, int64_t n, int64_t k, const void *a, int64_t lda,
               const void *b, int64_t ldb, void *c, int64_t ldc);
double tsr_dot_f64(int64_t n, const double *x, int64_t incx, const double *y, int64_t incy);
```

## FFT on complex128 rows (interleaved re, im), any length; inverse scales by 1/n; in may equal out

```c
int tsr_fft(int64_t n, int64_t rows, int inverse, const double *in, double *out);
/* real input: rows x n float64 -> rows x (n/2 + 1) complex128 (half the work of tsr_fft); irfft is the inverse
   (rows x (n/2 + 1) complex -> rows x n real, scaled by 1/n, imaginary parts of the DC/Nyquist bins ignored) */
int tsr_rfft(int64_t n, int64_t rows, const double *in, double *out);
int tsr_irfft(int64_t n, int64_t rows, const double *in, double *out);
```

## PCG64 random numbers, numpy-exact (state: 6 x uint64 = state hi/lo, inc hi/lo, has_uint32, uint32 buffer)

```c
void tsr_seed_sequence(const uint32_t *entropy, int64_t n_entropy, uint64_t *out_words, int64_t n_words);
void tsr_pcg64_seed(uint64_t *state, uint64_t state_hi, uint64_t state_lo, uint64_t inc_hi, uint64_t inc_lo);
void tsr_pcg64_uint64(uint64_t *state, int64_t n, uint64_t *out);
void tsr_pcg64_random(uint64_t *state, int64_t n, double *out);      /* [0, 1), same stream as numpy */
void tsr_pcg64_normal(uint64_t *state, int64_t n, double mean, double sd, double *out);
void tsr_pcg64_integers(uint64_t *state, int64_t n, int64_t low, int64_t high, int64_t *out); /* [low, high) */
int tsr_pcg64_shuffle(uint64_t *state, int64_t n, int64_t itemsize, void *data);                 /* Generator.shuffle */
```

## Sparse CSR (float64)

```c
int tsr_csr_matvec(int64_t rows, const int64_t *indptr, const int64_t *indices, const double *data,
                   const double *x, double *y);
int tsr_csr_matmat(int64_t rows, int64_t k, const int64_t *indptr, const int64_t *indices, const double *data,
                   const double *b, double *c);       /* C (rows x k) = A (rows x cols) * B (cols x k), row-major */
int64_t tsr_dense_nnz(int64_t rows, int64_t cols, const double *a);
int tsr_dense_to_csr(int64_t rows, int64_t cols, const double *a, int64_t *indptr, int64_t *indices, double *data);
int tsr_csr_transpose(int64_t rows, int64_t cols, const int64_t *indptr, const int64_t *indices, const double *data,
                      int64_t *t_indptr, int64_t *t_indices, double *t_data);
/* Krylov solvers: x = initial guess in, solution out. Returns iterations (>= 0) on convergence,
   -iterations when maxiter was hit, TSR_ENOMEM (-3) on allocation failure. */
int64_t tsr_csr_cg(int64_t n, const int64_t *indptr, const int64_t *indices, const double *data, const double *b,
                   double *x, double tol, double atol, int64_t maxiter, double *resid);
int64_t tsr_csr_bicgstab(int64_t n, const int64_t *indptr, const int64_t *indices, const double *data, const double *b,
                         double *x, double tol, double atol, int64_t maxiter, double *resid);
```

## Linear / mixed-integer programming: min c.x, A_ub x <= b_ub, A_eq x = b_eq, lb <= x <= ub

```c
Row-major dense matrices; lb/ub may be NULL ([0, inf)) and hold -INFINITY/INFINITY.
   Returns 0 optimal, 1 iteration/node limit, 2 infeasible, 3 unbounded, 4 numerical; < 0 argument errors.
   duals_* = d objective / d b (SciPy marginals); any output pointer except x/fun may be NULL. */
int tsr_linprog(int64_t n, const double *c,
                int64_t m_ub, const double *A_ub, const double *b_ub,
                int64_t m_eq, const double *A_eq, const double *b_eq,
                const double *lb, const double *ub, int64_t maxiter, double tol,
                double *x, double *fun, double *duals_ub, double *duals_eq, double *reduced, int64_t *nit);
int tsr_milp(int64_t n, const double *c,
             int64_t m_ub, const double *A_ub, const double *b_ub,
             int64_t m_eq, const double *A_eq, const double *b_eq,
             const double *lb, const double *ub, const uint8_t *integrality,
             int64_t node_limit, double mip_rel_gap, double tol,
             double *x, double *fun, int64_t *nodes, double *best_bound);
```

## Markov decision processes: P as CSR with A*S rows (row a*S+s = P(.|s,a)), R dense (S, A)

```c
int tsr_mdp_bellman(int64_t S, int64_t A, const int64_t *indptr, const int64_t *indices, const double *data,
                    const double *R, double gamma, const double *V, double *Q, double *V_out, int64_t *policy);
int tsr_mdp_value_iteration(int64_t S, int64_t A, const int64_t *indptr, const int64_t *indices, const double *data,
                            const double *R, double gamma, double epsilon, int64_t max_iter,
                            double *V, int64_t *policy, int64_t *iterations, double *delta);
int tsr_mdp_policy_iteration(int64_t S, int64_t A, const int64_t *indptr, const int64_t *indices, const double *data,
                             const double *R, double gamma, int64_t eval_sweeps, double epsilon, int64_t max_iter,
                             double *V, int64_t *policy, int64_t *iterations);
int tsr_mdp_policy_eval(int64_t S, int64_t A, const int64_t *indptr, const int64_t *indices, const double *data,
                        const double *R, double gamma, const int64_t *policy, double *V);
int tsr_mdp_finite_horizon(int64_t S, int64_t A, const int64_t *indptr, const int64_t *indices, const double *data,
                           const double *R, double gamma, int64_t T, const double *V_terminal,
                           double *V_out, int64_t *policy);
```

## Function registry (ADR 0011): special functions, statistics and other SciPy-style routines

```c
Entries are numbered 0 .. tsr_fn_count()-1; tsr_fn_info() describes one as JSON (kind "ufunc", "gufunc" or
   "dist"; argument, output and parameter names; defaults; enums) and returns the length needed (buf may be NULL).
   tsr_fn_ufunc: element-wise over operands already broadcast to `shape`, all float64 (dtype 0) or all float32
   (dtype 1); data[] = inputs then outputs, strides = nop x ndim bytes. For a distribution, `method` selects
   pdf 0, logpdf 1, cdf 2, logcdf 3, sf 4, logsf 5, ppf 6, isf 7 (inputs x, shapes..., loc[, scale]),
   stats 8 (outputs mean, var, skew, kurtosis), entropy 9, support 10 (inputs shapes..., loc[, scale]).
   tsr_fn_gufunc: axmask[i] selects the core axes of input i; outputs are allocated by the caller with the
   shapes from tsr_fn_gufunc_shape (out_shape is nout x 32). params follow the order of info "params".
   tsr_fn_routine: arguments of any kind (numbers, strings, arrays, null, sequences); results the caller adopts
   (arrays are tsr_alloc blocks of `bytes` bytes). A sequence argument (info "seq"/"variadic": numpy's
   `arrays` list or `*args`) is kind 5 with `count` items at `items`; a sequence result (info "out_seq":
   numpy's list/tuple of arrays) is kind 6 whose arr.data is a tsr_alloc block of arr.shape[0] tsr_result
   (`bytes` bytes), each owned by the caller like a top-level result. tsr_fn_error() describes the last
   failure on this thread. */
typedef struct tsr_arg {
    int32_t kind;                          /* 0 null, 1 number, 2 string, 3 array, 4 bool, 5 sequence */
    int32_t flags;                         /* kind 1: bit 0 set when the number was an integer, exactly in ival;
                                              kind 3: bit 1 set when the array was built from a PHP list (NumPy
                                              treats a Python sequence of floats as indices by int(), an ndarray not) */
    double num;
    const char *str;
    tsr_array arr;
    const struct tsr_arg *items;           /* kind 5 */
    int64_t count;                         /* kind 5 */
    int64_t ival;                          /* kind 1 with flags bit 0 */
} tsr_arg;
typedef struct tsr_result {
    int32_t kind;                          /* 0 none, 1 float, 3 array (arr.data from tsr_alloc: the caller owns it), 4 bool, 5 int,
                                              6 sequence (arr.data: arr.shape[0] tsr_result) */
    int32_t flags;
    double num;
    tsr_array arr;
    int64_t bytes;                         /* size of the arr.data block, for tsr_free */
    int64_t ival;                          /* kind 5: the exact value (num holds it rounded to double) */
} tsr_result;
int tsr_fn_count(void);
const char *tsr_fn_name(int id);
int tsr_fn_find(const char *name);
int64_t tsr_fn_info(int id, char *buf, int64_t cap);
int tsr_fn_arity(int id, int method, int *nin, int *nout);
const char *tsr_fn_error(void);
int tsr_fn_ufunc(int id, int method, int dtype, int32_t ndim, const int64_t *shape, int nop, void **data,
                 const int64_t *strides);
/* numpy.random.Generator methods (kind "random"): one variate per element of the output (the last operand,
   float64, int64 or float32 as out_dtype says), in C order, from the PCG64 state (6 x uint64, tsr_pcg64_seed);
   the array arguments are broadcast to the output shape by the caller (stride 0); params holds the enum
   parameters. Every element's parameters are checked before anything is drawn. */
int tsr_fn_random(int id, uint64_t *state, const double *params, int32_t ndim, const int64_t *shape, int nop,
                  void **data, const int64_t *strides, int out_dtype);
int tsr_fn_rvs(int id, uint64_t *state, int32_t ndim, const int64_t *shape, int nop, void **data,
               const int64_t *strides);
int tsr_fn_gufunc_shape(int id, int nin, const tsr_array *in, const uint64_t *axmask, int keepdims,
                        const double *params, int32_t *out_ndim, int64_t *out_shape);
int tsr_fn_gufunc(int id, int nin, const tsr_array *in, const uint64_t *axmask, int keepdims, const double *params,
                  int nout, const tsr_array *out);
int tsr_fn_routine(int id, int nargs, const tsr_arg *args, int nres, tsr_result *res);   /* kind "routine" */
```

