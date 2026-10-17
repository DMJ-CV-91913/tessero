# Kickoff: reproducible f32 transcendental primitives in Tessero

Read this, then read this repo's `CLAUDE.md` and follow its house rules. One hard style
rule carried from the owner: **no em dashes or en dashes anywhere** (code, comments, docs,
commits). Use commas, parentheses, or a plain spaced hyphen.

## Why you are here (context in one paragraph)

A cross-silicon determinism study (2026-10-17, six NVIDIA GPUs + CPU, both GCP and
lambda.ai) proved two tiers. The EXACT-op tier (anything built from IEEE +, *, FMA:
matmul/linear/bmm/add/conv2d/gather/embedding/segment_sum, forward and gradients) is
BIT-IDENTICAL across every GPU architecture, both clouds, two NVIDIA driver families, AND
the Tessero CPU. The TRANSCENDENTAL tier (gelu/softmax/rmsnorm/layernorm and the exp /
rsqrt / erf behind them) is NOT: it varies, and the variation tracks the NVIDIA DRIVER /
shader-compiler version (595 vs 580), not the silicon. Root cause: IEEE-754 pins a single
result for +, *, and FMA, but NOT for exp/rsqrt/erf, so today those lower to the GPU SFU
plus a driver-chosen refinement and the last bit moves. The fix, and your task, is to
compute these transcendentals from a FIXED polynomial evaluated only in correctly-rounded
f32 FMA, in a fixed order, so they reduce to the one thing IEEE pins and become
bit-identical on every vendor, every driver, and the CPU, by construction.

Full design + rationale (engine repo, sibling checkout, readable):
`C:\xampp\htdocs\talario\docs\papers\fma-only-transcendentals-design.md`
Strategy it feeds (reproducibility registry): `...\docs\strategy\nacco-silicon-registry.md`
Evidence: `C:\xampp\htdocs\talario\tests\attest\evidence\attestation-ext-NVIDIA-*.json`

## Your task

Build a NEW reproducible-f32 primitive surface in Tessero, DISTINCT from the existing
accuracy-first ufuncs (`erf`/`gamma`/... which are f64, boost/scipy-backed - do NOT reuse
or modify them; different goal). These primitives are a Tessero feature (reproducible
NumPy-for-PHP) and the CPU canonical that the Talario GPU shaders will mirror bit-for-bit.

### API (match Tessero conventions: `tsr_` prefix, int return 0 / negative error, f32)
Declare in `csrc/include/tessero.h`, implement in a new `csrc/src/repro_f32.c`, add to the
csrc Makefile. Contiguous f32 arrays, length n:

```c
int tsr_repro_rsqrt_f32(int64_t n, const float *x, float *y);  /* 1/sqrt(x) */
int tsr_repro_exp2_f32 (int64_t n, const float *x, float *y);  /* 2^x        */
int tsr_repro_exp_f32  (int64_t n, const float *x, float *y);  /* e^x        */
int tsr_repro_erf_f32  (int64_t n, const float *x, float *y);  /* erf(x)     */
```

### The two rules that make it reproducible (non-negotiable)
1. **Only correctly-rounded f32 ops**: `+ - *` and `fmaf(a,b,c)` (one rounding). NO libm
   `expf`/`exp2f`/`erff`, NO SVML / vectorized-libm, NO `inversesqrt`/reciprocal-estimate,
   NO `powf`. (Exception: `sqrtf` and `/` ARE IEEE-correctly-rounded - see rsqrt below.)
2. **Fixed evaluation order + exact range reduction**: fixed Horner order, constants as
   exact f32 bit patterns, range reduction in integer + `fmaf` (two-part `ln2`/`log2e`).
   No reassociation, no order that depends on SIMD lane or thread.

### Build constraints that make it SIMD- and thread-invariant
- Per-element primitives are embarrassingly parallel: each lane/thread runs the SAME fixed
  `fmaf` sequence, so the output is bit-identical across `tsr_simd_level` (scalar / AVX2 /
  AVX-512 / NEON) and across thread counts BY CONSTRUCTION - IF you use explicit `fmaf`
  everywhere, no SVML, no `-ffast-math`, and compile `repro_f32.c` with **`-ffp-contract=off`**
  (so the compiler cannot contract or reassociate the non-fma arithmetic). Confirm this TU's
  flags override any inherited ones in the Makefile.
- Strict f32 only: no f64 or long-double intermediates, no x87. Keep every intermediate in
  `float` so the CPU rounds at the same steps the GPU will.
- Reductions (later, in the composed ops) must use a FIXED-ORDER reduction, never an OpenMP
  reduction clause. For now your four primitives are per-element, so this does not bite yet.

## Build order (do them in this order)

1. **`tsr_repro_rsqrt_f32` FIRST - likely almost free.** `y = 1.0f / sqrtf(x)`. Both `sqrtf`
   and `/` are IEEE-correctly-rounded, so this is already reproducible with no polynomial.
   Handle x <= 0 (-> NaN per IEEE), +Inf (-> 0), NaN (-> NaN), denormals. This single
   function is what makes rmsnorm and layernorm bit-identical on the GPU side later, so it
   is the fastest proof of the whole thesis. Do it, test it, commit it.
2. **`tsr_repro_exp2_f32`** - the workhorse. Reduce x = k + r with k = round-to-nearest-even
   integer, r in [-0.5, 0.5]; evaluate 2^r by a fixed minimax polynomial in `fmaf` (Horner,
   fixed order); reconstruct 2^k by exact exponent-field construction (or `ldexpf` if it is
   exact on target). Handle overflow (-> +Inf), underflow (-> 0), NaN. Document the exact
   coefficients (as hex floats) and the order - the GLSL shader must mirror them EXACTLY.
3. **`tsr_repro_exp_f32`** = exp2 of (x * log2e) using a two-part `log2e` so the multiply
   stays exact (fmaf), or its own two-part `ln2` reduction. Same discipline.
4. **`tsr_repro_erf_f32`** - fixed minimax (rational or polynomial) in `fmaf`, odd function
   (compute on |x|, apply sign), or build on exp2. Freeze the form.

You may use core-math / RLIBM / a minimax tool to CHOOSE coefficients, but then FREEZE them
as exact f32 constants in the source. Goal is REPRODUCIBLE (one defined bit-result), plus a
stated accuracy bound (target <= 1 ULP vs an mpfr/f64 reference); correctly-rounded is a
later, stronger tier, not required for v1.

## Definition of done (per primitive)

Add a determinism test (Tessero's test harness, or a small standalone) that:
1. Runs the primitive on a FIXED seeded input vector (magnitude spread: tiny, normal, large,
   plus the edge inputs: 0, -0, +Inf, -Inf, NaN, denormal, and values that over/underflow).
2. Confirms the output is **bit-identical across `tsr_simd_level`** (force scalar vs AVX2 vs
   AVX-512 if the build permits, else across `-O1/-O2/-O3` and thread counts) - same bytes.
3. Prints the **SHA-256 of the raw f32 output bytes**. That hash is the GOLDEN for this
   primitive and this version. Record it.
4. Writes the CANONICAL SPEC (the exact constants as hex floats + the evaluation order) into
   a short doc/comment, because the Talario GLSL shader has to reproduce it bit-for-bit. CPU
   and GPU bit-identity depends on both sides running the IDENTICAL fmaf sequence.

## Out of scope for this session (do NOT do)
- No GPU / GLSL / shader code - that is the Talario half (sibling repo). You own the CPU
  canonical and the golden hashes.
- Do not touch or "improve" the existing accuracy-first ufuncs.
- Do not wire into Talario here; the seam is `talario/src/talario_tessero_adapter.c` (today
  binds only `tsr_gemm`) and gets extended on the Talario side later.
- Do not run the cross-GPU attestation - that happens on the Talario side with a cheap
  by-the-hour GPU once the shader mirrors your spec.

## Handback when done
Commit `repro_f32.c` + header decls + Makefile + the determinism test, and report: the four
golden SHA-256 hashes, the canonical spec (constants + order) for each, the measured
accuracy bound vs f64, and the determinism-across-SIMD test result. The Talario side then
mirrors the spec in the shaders and runs CPU==GPU on real silicon to close the loop.
