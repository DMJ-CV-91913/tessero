# Reproducible-f32 golden vectors

Golden SHA-256 hashes and the canonical spec for the reproducible-f32 transcendental
primitives (`tsr_repro_rsqrt_f32` / `tsr_repro_exp2_f32` / `tsr_repro_exp_f32` /
`tsr_repro_erf_f32` / `tsr_repro_recip_f32` / `tsr_repro_tanh_f32`),
defined once in `csrc/include/tessero_repro_f32.h`, exported by `csrc/src/repro_f32.c`,
declared in `csrc/include/tessero.h`. A hash here is the single defined f32 bit-result for the
primitive on a fixed input. Every conformant implementation (Tessero at any SIMD level,
any thread count, any compiler, and the Talario GPU shader) must reproduce it. See
`REPRO_F32_KICKOFF.md` and `talario/docs/papers/fma-only-transcendentals-design.md`.

## How to reproduce

```
bash csrc/repro/run_determinism.sh
```

It builds `csrc/src/repro_f32.c` + `csrc/repro/determinism_test.c` three ways, all with
`-fno-fast-math -ffp-contract=off` (never fast-math, which would swap in the
non-reproducible approximations): scalar (no vectorization), AVX2+FMA, and AVX-512 if the
host has it. For each primitive it hashes the raw f32 output and confirms it is
byte-identical across SIMD level AND across 1 vs 4 OpenMP threads. Equal bytes == bit
identical == reproducible, and that hash is the golden.

## Input vectors

- N = 2^20 (1,048,576) f32, generated as a pure function of the index (no RNG, no time),
  so the golden is stable for everyone. Generators are in `determinism_test.c`.
- `rsqrt`: strictly positive, magnitude spread ~1e-4 .. 1e4 (identical to the original
  proof-of-concept `demo_rsqrt.c`, so the wired ABI reproduces the first proven golden).
- `exp2`: linear spread over [-30, 30]. `exp`: [-20, 20]. `erf`: [-4, 4].
- The golden vector is finite and in-domain on purpose. NaN payloads and the exact
  subnormal/overflow boundary are not guaranteed identical across vendors, so edge inputs
  are verified by CLASSIFICATION (the `edges` mode), not folded into the hash.

## Goldens (first confirmed 2026-10-17, GCP n2 x86-64, AVX2+FMA, gcc 13.3, no AVX-512)

| primitive | GOLDEN(f32, N=2^20) SHA-256 | determinism | accuracy vs f64 ref |
|-----------|------------------------------|-------------|---------------------|
| rsqrt | `a55f20e5934645f1a554a0fddfa094531b13aa8b95b360ed8a44a24dc220e53f` | scalar == avx2 == 4-thread | <= 2 ULP |
| exp2  | `d7b57bb03e2c26b20642f0697daf64a366bedd9d7a3ee5edd79d8dae502aabba` | scalar == avx2 == 4-thread | <= 1 ULP |
| exp   | `7c0748f832cf0c6ea64d0382af76615033014602c90e882437dac5108d0aa336` | scalar == avx2 == 4-thread | <= 1 ULP |
| erf   | `aebea57c4f009254b1dfd67953c209441bad55acb561ef3ca5f1f3b75e5f8797` | scalar == avx2 == 4-thread | <= 5 ULP, max abs err 2.9e-7 |

The rsqrt golden is byte-for-byte the same hash the original proof-of-concept produced,
which confirms the wired ABI function matches it exactly.

## Canonical spec (exact f32 bit patterns)

Constants are exact f32 hex floats (printed by `determinism_test spec`). The evaluation
order is part of the spec: a conformant GPU shader MUST use these exact values and this
exact fmaf order, because CPU/GPU bit-identity depends on both sides running the identical
sequence.

Only the PORTABLY-correctly-rounded f32 op set is used: `+ - *` and `fmaf` (one rounding).
This is deliberate. A portable GPU API guarantees correct rounding (0.5 ULP) for add, sub,
mul, and fma, but NOT for divide or sqrt: under Vulkan, `OpFDiv` is allowed 2.5 ULP and
`OpSqrt` 3 ULP, and real NVIDIA silicon uses those tolerances, so any primitive that calls
`/` or `sqrtf` cannot be bit-identical CPU==GPU. Therefore NO primitive here uses `/` or
`sqrtf`. The two that mathematically need a reciprocal or an inverse-square-root (erf's A&S
term and rsqrt) obtain it from a DETERMINISTIC integer bit-hack seed refined by Newton-Raphson
steps evaluated purely in `fmaf` (see the rsqrt and erf specs). The integer seed uses only
exact bit reinterpretation (`floatBitsToUint`/`uintBitsToFloat`) and integer subtraction/shift,
which are bit-exact on every conformant target. No libm exp/exp2/erf, no SVML, no hardware
reciprocal/rsqrt estimate (`OpFDiv`/`inversesqrt`), no powf, no f64 or x87 intermediates,
no reassociation.

### scale2(y, n) = y * 2^n  (shared by exp2 and exp)
```
n1 = n / 2 (C integer truncation);  n2 = n - n1
f1 = bits_to_f32((uint32)(n1 + 127) << 23)      // exact power of two
f2 = bits_to_f32((uint32)(n2 + 127) << 23)
return (y * f1) * f2                             // left-associated, fixed order
```
Split so each factor is a normal power of two; the first multiply is exact, so the second
is a single correctly-rounded step even into the subnormal range. Valid for n in [-150, 128].

### rsqrt(x) = 1/sqrt(x)
```
seed  = 0x5f3759df - (bits_of(x) >> 1)              // fast inverse-sqrt seed, exact integer
r     = f32_from_bits(seed)                         // approx 1/sqrt(x), ~3.4% rel err
h     = 0.5f * x
r = r * fmaf(-h, r*r, 1.5f)                         // Newton step 1  (r *= 1.5 - h*r*r)
r = r * fmaf(-h, r*r, 1.5f)                         // step 2
r = r * fmaf(-h, r*r, 1.5f)                         // step 3
r = r * fmaf(-h, r*r, 1.5f)                         // step 4
r = r * fmaf(-h, r*r, 1.5f)                         // step 5
return r
```
No `sqrtf` and no `/`: the classic Quake integer-bit-hack seed (Lomont/Carmack constant
`0x5f3759df`) gives a deterministic starting approximation, and five Newton-Raphson iterations
of the inverse-sqrt map `r <- r*(1.5 - (x/2)*r*r)`, each evaluated as a single `fmaf` plus
multiplies, converge to the fixed f32 result. Error decays quadratically; five steps saturate
f32 and leave `<= 2 ULP` vs the f64 reference (the old `1.0f/sqrtf(x)` form was `<= 1 ULP` but
used the non-portable divide and sqrt, so it could not be reproduced on GPU; this form can).
The seed and every step are bit-identical on CPU and on any target with correctly-rounded fma.
Edges: the seed arithmetic makes x <= 0, Inf, and NaN produce deterministic (not IEEE-natural)
outputs; the primitive is specified for finite x > 0 (its gate/attention use), and the golden
is taken over the positive generator `gen_rsqrt`. Out-of-domain inputs are defined only as
"whatever this fixed sequence yields," identically on both sides.

### exp2(x) = 2^x
```
if x >= 128   -> +Inf        (also +Inf input)
if x <= -150  -> 0           (also -Inf input)
k = floorf(x + 0.5f);  r = x - k;  n = (int)k      // r in [-0.5, 0.5]
2^r = fmaf(fmaf(fmaf(fmaf(fmaf(C0,r,C1),r,C2),r,C3),r,C4),r,C5) then fmaf(.,r,1)
return scale2(2^r, n)
```
| const | hex float | decimal |
|-------|-----------|---------|
| C0 | `0x1.41fbbcp-13` | 0.00015353362 |
| C1 | `0x1.5f3e52p-10` | 0.00133988739 |
| C2 | `0x1.3b2d4cp-7`  | 0.00961843692 |
| C3 | `0x1.c6aee8p-5`  | 0.0555033237 |
| C4 | `0x1.ebfbdcp-3`  | 0.240226477 |
| C5 | `0x1.62e43p-1`   | 0.693147182 |

### exp(x) = e^x
```
if x >= 88.72283905  -> +Inf     (also +Inf input)
if x <= -103.972084  -> 0        (also -Inf input)
k = floorf(fmaf(x, LOG2E, 0.5f))                   // round(x*log2e)
r = fmaf(-k, LN2_HI, x);  r = fmaf(-k, LN2_LO, r)  // r = x - k*ln2, two-part ln2
z = r*r;  P = fmaf(fmaf(fmaf(fmaf(fmaf(Q0,r,Q1),r,Q2),r,Q3),r,Q4),r,Q5)
e^r = fmaf(P, z, r) + 1
return scale2(e^r, (int)k)
```
| const | hex float | decimal |
|-------|-----------|---------|
| LOG2E  | `0x1.715476p+0`   | 1.44269502 |
| LN2_HI | `0x1.63p-1`       | 0.693359375 |
| LN2_LO | `-0x1.bd0106p-13` | -0.000212194442 |
| Q0 | `0x1.a0d2cep-13` | 0.000198756912 |
| Q1 | `0x1.6e879cp-10` | 0.00139819994 |
| Q2 | `0x1.11121p-7`   | 0.00833345205 |
| Q3 | `0x1.555382p-5`  | 0.0416657962 |
| Q4 | `0x1.555554p-3`  | 0.166666657 |
| Q5 | `0x1p-1`         | 0.5 |

### erf(x)
Odd function, two frozen forms by magnitude (the branch is a deterministic function of x,
so it stays bit-reproducible):
```
ax = |x|
if ax <= 0.5:        // Maclaurin series, avoids the 1 - P*exp cancellation near 0
    w = x*x
    erf = x * (fmaf(fmaf(fmaf(fmaf(fmaf(B5,w,B4),w,B3),w,B2),w,B1),w,B0))
else:                // Abramowitz & Stegun 7.1.26
    s = sign(x)
    if ax >= 4 -> s                               // saturate; +-Inf -> +-1
    b = fmaf(P, ax, 1.0f)                         // denominator 1 + P*ax
    // t = 1/b WITHOUT OpFDiv: integer seed + fma Newton, so CPU==GPU
    seed = 0x7EF127EA - bits_of(b)                // reciprocal bit-hack seed
    t = f32_from_bits(seed)
    t = t * fmaf(-b, t, 2.0f)                      // Newton step 1  (t *= 2 - b*t)
    t = t * fmaf(-b, t, 2.0f)                      // step 2
    t = t * fmaf(-b, t, 2.0f)                      // step 3
    t = t * fmaf(-b, t, 2.0f)                      // step 4
    t = t * fmaf(-b, t, 2.0f)                      // step 5
    poly = fmaf(fmaf(fmaf(fmaf(A5,t,A4),t,A3),t,A2),t,A1) * t
    e = exp(-(ax*ax))                             // the reproducible exp above
    erf = s * fmaf(-poly, e, 1.0f)
```
The reciprocal `t = 1/b` uses the same no-divide construction as rsqrt: an exact integer seed
(`0x7EF127EA - bits_of(b)`) refined by five Newton steps of the reciprocal map `t <- t*(2 - b*t)`,
each a single `fmaf`. For the erf input domain (`b = 1 + P*ax` with ax in (0.5, 4), so b well
away from 0) five steps reach the fixed f32 reciprocal; erf accuracy is unchanged (~2.9e-7 abs,
<= 5 ULP in the A&S region). The old `1.0f / fmaf(...)` form used `OpFDiv` and could not be
reproduced on GPU; this form is bit-identical CPU==GPU.
| const | hex float | decimal |
|-------|-----------|---------|
| B0 | `0x1.20dd76p+0`   | 1.12837923 |
| B1 | `-0x1.812746p-2`  | -0.376126379 |
| B2 | `0x1.ce2f22p-4`   | 0.112837918 |
| B3 | `-0x1.b82ce4p-6`  | -0.0268661715 |
| B4 | `0x1.565bcep-8`   | 0.00522397785 |
| B5 | `-0x1.c02db4p-11` | -0.000854832702 |
| P  | `0x1.4f740ap-2`   | 0.327591091 |
| A1 | `0x1.04f20cp-2`   | 0.254829586 |
| A2 | `-0x1.23531cp-2`  | -0.284496725 |
| A3 | `0x1.6be1c6p+0`   | 1.42141378 |
| A4 | `-0x1.7401c6p+0`  | -1.45315206 |
| A5 | `0x1.0fb844p+0`   | 1.06140542 |

### Single definition and the canonicalization rule (2026-10-23)

The element-level definitions now live in ONE header-only file,
`csrc/include/tessero_repro_f32.h` (valid C11 and C++20). `csrc/src/repro_f32.c` wraps it
as the exported array ABI; Talario compiles a byte-identical vendored copy for its CPU
twins and mirrors it on the GPU in `shaders/repro_f32.glsl`, whose constant block must be
byte-identical to the header's (Talario's build fails on any drift).

Device float-mode dependence is removed by one helper, `tsr__canon_f32`: a subnormal
becomes a signed zero and any NaN becomes the canonical quiet NaN `0x7FC00000`, using
integer bit operations only, so a flush-to-zero device and a subnormal-preserving one
read the same bits. Every primitive canonicalizes its input, and exp2/exp/recip
canonicalize any output that can land in the subnormal range. recip and rsqrt rescale
arguments >= 2^64 by an exact power of two (bit-identical to the unscaled path where both
are normal) so no intermediate goes subnormal; rsqrt of a negative input returns the
canonical NaN. Canonicalization is the identity on every normal value, so the four
goldens above are unchanged (re-verified 2026-10-23: scalar == AVX2 == 4-thread, same
four hashes). Edge behaviour changed only where it used to depend on the device: for
example exp(-100) is now +0 (was a subnormal), and rsqrt(+Inf) is +0.

### recip(b) = 1/b   (exported as tsr_repro_recip_f32, any f32 input)
```
b = canon(b); NaN -> canonical NaN; +-0 -> +-Inf; +-Inf -> +-0
a = |b|;  r = (a >= 2^64) ? rrecip(a * 2^-64) * 2^-64 : rrecip(a)
rrecip(v): t = f32_from_bits(0x7EF127EA - bits_of(v)); 5x  t = t * fmaf(-v, t, 2)
return canon(r) with the sign of b
```

### tanh(u)   (exported as tsr_repro_tanh_f32)
```
u = canon(u); NaN -> canonical NaN;  a = |u|
a <  2^-12 : u                                  (the correctly rounded f32 tanh)
a >= 9.1   : copysign(1, u)                     (f32 tanh rounds to +-1 there)
a <= 0.25  : w = u*u; p = fmaf(fmaf(fmaf(fmaf(C11,w,C9),w,C7),w,C5),w,C3)
             return fmaf(u*w, p, u)            (odd Taylor series, Horner in fmaf)
otherwise  : e2 = exp(2a); t = (e2 - 1) * rrecip(e2 + 1); sign of u restored by bit
```
Exactly odd. Saturation also removes the e^2u overflow that drove the old
exp-form tanh to -Inf/NaN for u >= 44.36.

| const | hex float | decimal |
|-------|-----------|---------|
| C3  | `-0x1.555556p-2` | -0.333333343 |
| C5  | `0x1.111112p-3`  | 0.13333334 |
| C7  | `-0x1.ba1ba2p-5` | -0.0539682545 |
| C9  | `0x1.664f48p-6`  | 0.0218694881 |
| C11 | `-0x1.226e36p-7` | -0.00886323582 |
| SAT | `0x1.233334p+3`  | 9.10000038 |

Input vectors: `recip` = the rsqrt magnitude spread with alternating sign; `tanh` = a
signed spread over [-10, 10] at scales 1, 1e-2 and 1e-5 so every frozen form is hit.

**Goldens (confirmed 2026-10-23 on two independent hosts).** Captured first on a dev box
(x86-64, gcc 13.3) and then reproduced byte-for-byte on the build server (x86-64, AVX2+FMA,
no AVX-512), each scalar == AVX2 == 4-thread, with identical edge-input classifications. The
four goldens above re-verified unchanged on both hosts in the same runs.

| primitive | GOLDEN(f32, N=2^20) SHA-256 | determinism | accuracy vs f64 ref |
|-----------|--------------------------------|-------------|---------------------|
| recip | `f7f9ec19f8589f559977abefc0ba506e261ac1af84cbcaec32c0f4cf4979613d` | scalar == avx2 == 4-thread | <= 1 ULP |
| tanh  | `5b43f3d6d5167193e3a18b9be0f55b3f99a650661b9635fac15aaa6ad310a694` | scalar == avx2 == 4-thread | <= 3 ULP, max abs err 2.1e-7 |

### Composites (2026-10-23): one definition of the composition

`tsr__sigmoid_f32`, `tsr__gelu_tanh_f32`, `tsr__softmax_row_f32`, `tsr__rmsnorm_row_f32` and
`tsr__layernorm_row_f32` (header) build only from the primitives above in one fixed operation order (canonicalized
loads/stores, sequential index-order f32 sums, divides as products with the guarded reciprocal, inverse sqrt via
rsqrt). Exported as `tsr_repro_{sigmoid,gelu_tanh,softmax,rmsnorm,layernorm}_f32`. Talario's CPU backend calls them
directly and its GPU shaders mirror them; on NVIDIA A10/A100 the Talario attestation of softmax / rmsnorm /
layernorm is bit-identical CPU==GPU (talario run 7), and moving the CPU side onto these functions left every
Talario CPU digest unchanged. Constants `TSR_GELU_C` = sqrt(2/pi), `TSR_GELU_A` = 0.044715 joined the shared block.

Inputs: sigmoid / gelu_tanh linear spread over [-12, 12]; softmax / rmsnorm / layernorm a 1024 x 1024 row-major
matrix from a linear spread over [-8, 8], gamma[j] = 0.5 + (j mod 17) / 16, beta[j] = (j mod 5) - 2, eps = 1e-5.

**CANDIDATE goldens (dev box, x86-64, gcc 13.3, scalar == AVX2 == 4-thread); approve on the reference host:**

| composite | CANDIDATE SHA-256 |
|-----------|-------------------|
| sigmoid   | `4b218f084f039c8c1df22f33c7281db188a9b982a13351bde704ff385e207dbf` |
| gelu_tanh | `e913256381ad13735e34acb246c451685c7abb6a02e88c1ed02438e021b5d0be` |
| softmax   | `fc2c51599eef4eb2aed8f183785ee1a5df8db5445b482768b9c6d13bfc14e917` |
| rmsnorm   | `cee10b51cf07b010ce97a888ff1e8ac4eec2c56a6ad29b381225cfa8aeba5254` |
| layernorm | `94a4721637fd2be28cd8e27f1b1315874a0b4b71ad0e80a56af40ecf6d16405f` |

## What this PROVES

On x86 with AVX2+FMA, all four primitives are bit-identical across SIMD width (scalar ==
vectorized) and across thread count (1 == 4), by construction: each element runs the
identical `fmaf` sequence with no reduction and no SIMD/thread-dependent order. This is the
CPU canonical the Talario GPU shader mirrors bit-for-bit to collapse the transcendental
tier (gelu/softmax/rmsnorm/layernorm) into the single bit-identical family, the same way
the EXACT-op tier (matmul/add/FMA) already is.

## Open cells (honest scope)

- CPU == GPU: CONFIRMED (2026-10-18) on a real NVIDIA Tesla V100 (GCP, Vulkan compute).
  All four Talario GLSL/SPIR-V shaders compute each primitive with these exact constants and
  fmaf order and reproduce these goldens byte-for-byte (rsqrt a55f20e5, exp2 d7b57bb0,
  exp 7c0748f8, erf aebea57c). This is what forced the no-divide/no-sqrt reformulation above:
  Vulkan guarantees correct rounding only for `OpFAdd/OpFSub/OpFMul/OpFMA`, so the original
  `OpFDiv`/`OpSqrt` forms of rsqrt and erf diverged on silicon until reduced to the fma-only
  seed+Newton construction. exp2 and exp (already fma-only) matched from the first run. Harness
  and run script: Talario `tests/attest/repro_gpu_validate.c` + `repro_gpu_run.sh` (separate repo).
- Cross-CPU-ISA: CONFIRMED (2026-10-17). All four goldens reproduce byte-for-byte on ARM64
  Neoverse V2 (GCP c4a Axion) across scalar/NEON/SVE. See csrc/repro/ARCH_MATRIX.md.
- AVX-512: CONFIRMED (2026-10-17). scalar == AVX2 == AVX-512 on Intel Emerald Rapids (GCP
  c4), all four primitives. See csrc/repro/ARCH_MATRIX.md.
- erf accuracy is ~2.9e-7 abs / <= 5 ULP in the A&S region (ax > 0.5); the near-zero
  region (gelu-critical) is tight via the series. Correctly-rounded erf is a later tier.

## Provenance and originality

The reproducible-f32 primitives (tsr_repro_rsqrt/exp2/exp/erf_f32) are a NEW, ORIGINAL
Tessero contribution and are NOT part of the ported NumPy/SciPy surface. NumPy/SciPy
transcendentals are accuracy-first and defer to the system libm / vendor math, which is NOT
bit-reproducible across hardware, drivers, or SIMD width. Tessero's ~1,845 parity-gated
functions match NumPy/SciPy exactly and are untouched by this work.

What is original is the REPRODUCIBLE-BY-CONSTRUCTION discipline: every primitive reduces to
the PORTABLY-correctly-rounded f32 op set only (+, -, *, fmaf), in a fixed evaluation order,
with exact integer range reduction, an exact exponent-bit scale2, and -- for the reciprocal
and inverse-square-root -- a deterministic integer bit-hack seed refined by fma-only Newton
steps in place of the non-portable `/` and `sqrtf`. So the output is ONE defined f32 bit-value
on every SIMD width, thread count, and CPU ISA, and (confirmed on silicon) on the GPU. That
construction -- in particular reducing transcendentals to the exact op set a portable GPU API
guarantees, so divide and sqrt never appear -- and the CPU==GPU / cross-ISA bit-identity it
enables, is the contribution.

Attribution: the minimax polynomial COEFFICIENTS are standard published values - Cephes
(S. Moshier, exp2f/expf; public domain) and Abramowitz & Stegun 7.1.26 (erf; public domain) -
used as source constants and cited in csrc/src/repro_f32.c. Using well-known coefficients is
routine; the novel element is the fixed-order fma construction and the resulting bit-identity,
not the coefficients.
