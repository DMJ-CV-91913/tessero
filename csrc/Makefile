# libtessero build. `make` -> ../lib/<platform>/libtessero.<ext>
#   make test       C unit tests (optimised build)
#   make sanitize   C unit tests under AddressSanitizer + UBSan
#   make portable   build without target_clones (musl, clang, cross builds)

UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)
LIBNAME := libtessero
# the C ABI generation, from the single version source (include/tessero_version.h, ADR 0014): the soname
# libtessero.so.<abi> changes only when the ABI does
TSR_ABI := $(shell sed -n 's/^\#define TSR_ABI_VERSION \([0-9][0-9]*\).*/\1/p' include/tessero_version.h)
# LAPACKE (row-major C LAPACK) and cblas back numpy.linalg / scipy.linalg and the float matmul in the kernel.
#   Linux:   liblapacke + the system BLAS (OpenBLAS via the alternatives system).
#   macOS:   Homebrew OpenBLAS, which ships LAPACK, LAPACKE (lapacke.h) and cblas in one library (Apple's
#            Accelerate has LAPACK but no LAPACKE). `brew install openblas`; OPENBLAS_PREFIX overrides the path.
#   Windows: MSYS2 MINGW64 OpenBLAS (`pacman -S mingw-w64-x86_64-openblas`), which also carries LAPACKE.
# Set LAPACK (link flags) and LAPACK_INC (include flags) to use any other LAPACKE provider.
LAPACK ?=
LAPACK_INC ?=
ifeq ($(UNAME_S),Darwin)
  PLATFORM := darwin-$(UNAME_M)
  EXT := dylib
  SHARED := -dynamiclib -install_name @rpath/libtessero.dylib
  OPENMP ?=
  ifeq ($(LAPACK),)
    OPENBLAS_PREFIX ?= $(shell brew --prefix openblas 2>/dev/null)
    ifneq ($(OPENBLAS_PREFIX),)
      LAPACK_INC := -I$(OPENBLAS_PREFIX)/include
      LAPACK := -L$(OPENBLAS_PREFIX)/lib -lopenblas -Wl,-rpath,$(OPENBLAS_PREFIX)/lib
    endif
  endif
else ifneq (,$(findstring MINGW,$(UNAME_S))$(findstring MSYS,$(UNAME_S)))
  PLATFORM := windows-$(UNAME_M)
  EXT := dll
  LIBNAME := tessero
  SHARED := -shared -static-libgcc
  OPENMP ?=
  # MinGW hides M_PI, M_LN2, ... under strict -std=c++17 (__STRICT_ANSI__); MSVC needs the same define (config.w32)
  PLATFORM_DEFS := -D_USE_MATH_DEFINES
  ifeq ($(LAPACK),)
    MINGW_PREFIX ?= /mingw64
    LAPACK := -lopenblas
    LAPACK_INC := $(if $(wildcard $(MINGW_PREFIX)/include/openblas/lapacke.h),-I$(MINGW_PREFIX)/include/openblas)
  endif
else
  MUSL := $(shell ldd --version 2>&1 | grep -qi musl && echo -musl)
  PLATFORM := linux$(MUSL)-$(UNAME_M)
  EXT := so
  # the file keeps its name (the FFI loader opens lib/<platform>/libtessero.so by path); the soname carries the ABI
  SONAME := libtessero.so.$(TSR_ABI)
  SHARED := -shared -Wl,-soname,$(SONAME)
  LAPACK := -llapacke -lblas
endif

CC ?= cc
CXX ?= c++
BUILD ?= ./build
OUT := ../lib/$(PLATFORM)
# numpy/random distributions (vendored, for numpy.random.Generator and scipy.stats rvs)
NPRAND := third_party/numpy/random/src/distributions.c third_party/numpy/random/src/logfactorial.c \
  third_party/numpy/random/src/random_hypergeometric.c third_party/numpy/random/src/random_mvhg_count.c \
  third_party/numpy/random/src/random_mvhg_marginals.c
# SciPy's QUADPACK translation and brentq (scipy.stats generic methods)
FITPACKC := $(wildcard third_party/scipy/fitpack/*.c)
SCIPYC := third_party/scipy/quadpack/quadpack.c third_party/scipy/zeros/brentq.c third_party/scipy/_cosine.c $(FITPACKC)
SRC := $(wildcard src/*.c) third_party/scipy/cdflib.c $(NPRAND) $(SCIPYC)
# C++: the special-function libraries SciPy uses (xsf, Boost.Math, SciPy's glue), ADR 0011
SRC_CXX := $(wildcard cxx/*.cpp) third_party/scipy/wright.cc $(EXTRA_CXX)
# EXTRA_CXX: work-in-progress sources in cxx/wip/ (make EXTRA_CXX=cxx/wip/dist_x.cpp), built only when named
TP_INC := -Ithird_party -Ithird_party/xsf -Ithird_party/scipy -Ithird_party/boost -Ithird_party/numpy/include
CXXFLAGS ?= -O2 -std=c++17 -fPIC -fno-math-errno -fvisibility=hidden -DBOOST_MATH_STANDALONE $(OPENMP) -Wall -Wno-unused-parameter -Wno-unused-function -Wno-unused-variable -Wno-sign-compare -Wno-maybe-uninitialized -Wno-unused-but-set-variable -Wno-strict-aliasing -Wno-deprecated-declarations
OBJ_CXX = $(patsubst %,$(BUILD)/cxx/%.o,$(notdir $(SRC_CXX)))
ifeq ($(UNAME_S),Darwin)
  CXXLINK :=
else
  CXXLINK := -static-libstdc++ -static-libgcc
endif
WARN := -Wall -Wextra -Wno-unused-parameter -Wshadow -Wno-sign-compare -Wno-type-limits -Wno-missing-field-initializers
OPENMP ?= -fopenmp
CFLAGS ?= -O3 -std=gnu11 -fPIC -fno-math-errno -fno-trapping-math $(OPENMP) $(WARN)
# Hidden by default: only the declarations of include/tessero.h are given default visibility (TSR_EXPORT_ABI,
# src/internal.h), and the version script below exports exactly the tsr_ ABI. Vendored code (FITPACK, numpy
# random, cdflib), internal helpers and the statically linked C++ runtime stay private, so they can neither
# interpose on nor be interposed by another copy in the process (ADR 0014). Appended in the compile rules, so a
# CFLAGS given on the command line keeps it.
VIS := -fvisibility=hidden -DTSR_EXPORT_ABI
LDLIBS := -lm $(LAPACK)
CFLAGS += $(LAPACK_INC) $(PLATFORM_DEFS)
CXXFLAGS += $(LAPACK_INC) $(PLATFORM_DEFS)

# ---- compiler and linker hardening (OpenSSF "Compiler Options Hardening Guide for C and C++"). Every flag is
#      probed, so a toolchain that lacks one drops it instead of failing. None changes floating-point semantics;
#      the parity gate verifies results stay bit-exact. HARDEN=0 turns it off. The sanitizer and fuzz builds keep
#      their own flags (FORTIFY conflicts with the ASan interceptors, zero-init would hide uninitialised reads).
HARDEN ?= 1
comma := ,
try_cc = $(shell printf 'int main(void){return 0;}\n' | $(CC) -Werror -O2 $(1) -x c - -o /dev/null >/dev/null 2>&1 && echo "$(1)")
try_cxx = $(shell printf 'int main(void){return 0;}\n' | $(CXX) -Werror -O2 $(1) -x c++ - -o /dev/null >/dev/null 2>&1 && echo "$(1)")
try_so = $(shell printf 'int tsr_probe(void){return 0;}\n' | $(CC) -Werror -fPIC -shared $(1) -x c - -o /dev/null >/dev/null 2>&1 && echo "$(1)")
# -mbranch-protection is skipped on macOS: Apple's arm64 ABI (not arm64e) unwinder does not follow the pac-ret
# unwind state the flag adds, so a C++ exception could not unwind through those frames and reached terminate even
# under a catch(...) (betainccinv's Boost evaluation_error in CI). Pointer authentication on Apple is arm64e.
ifeq ($(HARDEN),1)
  HARDEN_CC := $(call try_cc,-U_FORTIFY_SOURCE -D_FORTIFY_SOURCE=3) $(call try_cc,-fstack-protector-strong) \
    $(call try_cc,-fstack-clash-protection) $(call try_cc,-fcf-protection=full) \
    $(if $(filter Darwin,$(UNAME_S)),,$(call try_cc,-mbranch-protection=standard)) $(call try_cc,-ftrivial-auto-var-init=zero) \
    $(call try_cc,-fno-delete-null-pointer-checks)
  HARDEN_CXX := $(HARDEN_CC) $(call try_cxx,-D_GLIBCXX_ASSERTIONS)
  HARDEN_LD := $(call try_so,-Wl$(comma)-z$(comma)relro) $(call try_so,-Wl$(comma)-z$(comma)now) \
    $(call try_so,-Wl$(comma)-z$(comma)noexecstack)
  CFLAGS += $(HARDEN_CC) $(call try_cc,-Wformat=2) $(call try_cc,-Werror=format-security)
  CXXFLAGS += $(HARDEN_CXX)
  LDLIBS += $(HARDEN_LD)
endif
# reproducible builds: __FILE__ (assert messages) and any debug info name sources relative to csrc/, so the
# library is byte-identical whatever directory it is built in (probed: GCC >= 8, clang >= 10)
CFLAGS += $(call try_cc,-ffile-prefix-map=$(CURDIR)=.)
CXXFLAGS += $(call try_cxx,-ffile-prefix-map=$(CURDIR)=.)
# code placement (probed): every function starts on a 64-byte boundary, so a hot loop's offset within its cache
# lines no longer depends on how much code the linker placed before it. Without it the hardening flags above
# (more code) moved linprog's inner loop and cost +22% from layout alone (ABBA, A/A -0.1%); with it linprog is
# back to the unhardened time and no benchmark case regresses. Layout only: instructions and results are unchanged.
CFLAGS += $(call try_cc,-falign-functions=64)
CXXFLAGS += $(call try_cxx,-falign-functions=64)

# exports: the version script (GNU ld, gold, lld; probed, so another linker just skips it) keeps every symbol
# that is not tsr_* local, plus the tsr_ helpers listed in it that are internal
VERSION_SCRIPT := $(call try_so,-Wl$(comma)--version-script=$(CURDIR)/libtessero.map)
# The library's own calls to its exported functions (tsr_alloc, tsr_free, tsr_get_threads, ...) bind inside it.
# Without this a second libtessero already in the global scope (Talario's, loaded by its PHP extension) captured
# them: arrays came from the other instance's allocator and budget, and errors went to its slots.
VERSION_SCRIPT += $(call try_so,-Wl$(comma)-Bsymbolic-functions)
# -z nodelete (probed; ELF): never unmapped by dlclose(). Two reasons: (1) PHP 8.2 FFI closes its libraries at
# the end of every request, so each request re-mapped the library and re-ran its initialisers, and the
# process-lifetime state set up once (OpenMP and BLAS thread pools, the static libstdc++ exception emergency
# pool) was reported by LeakSanitizer as leaked; a later FFI::cdef reuses the mapped library. (2) Unmapping while
# libgomp worker threads are alive could make them execute unmapped code at process exit (preventive, not
# reproduction-tested; ADR 0014). tsr_feature("nodelete") reports whether the linker applied it.
VERSION_SCRIPT += $(call try_so,-Wl$(comma)-z$(comma)nodelete)
LDLIBS += $(call try_so,-ldl)

all: $(OUT)/$(LIBNAME).$(EXT)

$(BUILD)/cxx/%.cpp.o: cxx/%.cpp cxx/*.hpp src/fn.h
	@mkdir -p $(BUILD)/cxx
	$(CXX) $(CXXFLAGS) $(VIS) $(TP_INC) -MMD -MP -c -o $@ $<

$(BUILD)/cxx/%.cpp.o: cxx/wip/%.cpp cxx/*.hpp src/fn.h
	@mkdir -p $(BUILD)/cxx
	$(CXX) $(CXXFLAGS) $(VIS) $(TP_INC) -Icxx -MMD -MP -c -o $@ $<

$(BUILD)/cxx/wright.cc.o: third_party/scipy/wright.cc
	@mkdir -p $(BUILD)/cxx
	$(CXX) $(CXXFLAGS) $(VIS) $(TP_INC) -MMD -MP -c -o $@ $<

OBJ_C = $(patsubst %.c,$(BUILD)/c/%.o,$(notdir $(SRC)))
$(BUILD)/c/%.o: src/%.c src/*.h include/tessero.h include/tessero_version.h
	@mkdir -p $(BUILD)/c
	$(CC) $(CFLAGS) $(VIS) -Ithird_party/numpy/include -c -o $@ $<
# repro_f32.c: the reproducible-f32 primitives MUST compile with contraction and fast-math OFF so
# the compiler cannot contract or reassociate the non-fma arithmetic. This is what keeps the result
# bit-identical across SIMD level, thread count and compiler (see csrc/repro/GOLDENS.md). The flags
# follow $(CFLAGS) so they override anything inherited. This explicit target beats the generic rule.
$(BUILD)/c/repro_f32.o: src/repro_f32.c src/*.h include/tessero.h include/tessero_repro_f32.h
	@mkdir -p $(BUILD)/c
	$(CC) $(CFLAGS) -fno-fast-math -ffp-contract=off -fvisibility=default -Ithird_party/numpy/include -c -o $@ $<
# numpy/random builds with contraction off, so a*b+c (random_uniform's lower + range*x, random_normal's
# loc + scale*z, ...) rounds twice, as in NumPy's x86-64 wheels, on every target: aarch64 GCC and Apple clang
# contract by default, which moved those draws 1 ulp off the x86 reference. x86-64 builds without -mfma never
# contract, so there it changes nothing. ext/config.m4 compiles these sources with the same flag.
$(BUILD)/c/%.o: third_party/numpy/random/src/%.c
	@mkdir -p $(BUILD)/c
	$(CC) $(CFLAGS) -ffp-contract=off $(VIS) -Ithird_party/numpy/include -w -MMD -MP -c -o $@ $<
$(BUILD)/c/%.o: third_party/scipy/quadpack/%.c
	@mkdir -p $(BUILD)/c
	$(CC) $(CFLAGS) $(VIS) -w -MMD -MP -c -o $@ $<
$(BUILD)/c/%.o: third_party/scipy/fitpack/%.c
	@mkdir -p $(BUILD)/c
	$(CC) $(CFLAGS) $(VIS) -w -MMD -MP -c -o $@ $<
$(BUILD)/c/%.o: third_party/scipy/zeros/%.c
	@mkdir -p $(BUILD)/c
	$(CC) $(CFLAGS) $(VIS) -w -MMD -MP -c -o $@ $<
$(BUILD)/c/_cosine.o: third_party/scipy/_cosine.c
	@mkdir -p $(BUILD)/c
	$(CC) $(CFLAGS) $(VIS) -w -MMD -MP -c -o $@ $<
$(BUILD)/c/cdflib.o: third_party/scipy/cdflib.c
	@mkdir -p $(BUILD)/c
	$(CC) $(CFLAGS) $(VIS) -w -MMD -MP -c -o $@ $<

$(OUT)/$(LIBNAME).$(EXT): $(OBJ_C) $(OBJ_CXX) libtessero.map
	@mkdir -p $(OUT)
	$(CXX) $(CFLAGS) $(OBJ_C) $(OBJ_CXX) $(SHARED) $(VERSION_SCRIPT) $(CXXLINK) -o $@ $(LDLIBS)
ifneq ($(SONAME),)
	@# programs linked with -ltessero record the soname: give the loader a file of that name next to the library
	ln -sf $(LIBNAME).$(EXT) $(OUT)/$(SONAME)
endif

# The exported dynamic symbols must be exactly the functions declared in include/tessero.h (ifunc resolvers of
# exported multiversioned functions aside). Fails on any extra or missing export.
abi-check: $(OUT)/$(LIBNAME).$(EXT)
	@mkdir -p $(BUILD)
	@$(CC) -E -P -x c include/tessero.h | grep -oE '\btsr_[a-z0-9_]+[[:space:]]*\(' | sed 's/[[:space:]]*(//' | sort -u > $(BUILD)/abi.header
	@nm -D --defined-only $< | awk '{print $$3}' | sed 's/@.*//' | grep -v -e '^TESSERO_' -e '\.resolver$$' | sort -u > $(BUILD)/abi.exported
	@diff -u $(BUILD)/abi.header $(BUILD)/abi.exported && echo "abi-check: $$(wc -l < $(BUILD)/abi.exported) exported symbols == tessero.h declarations" \
	    && readelf -d $< | grep -q 'SONAME.*$(SONAME)' && echo "abi-check: soname $(SONAME)"
	@# a second copy already in the global scope (Talario's) must not capture this library's internal calls
	@$(CC) -O1 -o $(BUILD)/abi_interpose tests/abi_interpose.c $(call try_so,-ldl)
	@cp $< $(BUILD)/libtessero-other-copy.$(EXT)
	@$(BUILD)/abi_interpose $(abspath $(BUILD))/libtessero-other-copy.$(EXT) $(abspath $<)

# header dependencies of the C++ and vendored objects (-MMD)
-include $(wildcard $(BUILD)/cxx/*.d $(BUILD)/c/*.d)

portable:
	$(MAKE) CFLAGS="$(CFLAGS) -DTSR_NO_CLONES"

$(BUILD)/test_opt: $(OBJ_C) tests/test_main.c $(OBJ_CXX)
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) -c -o $(BUILD)/test_main.o tests/test_main.c
	$(CXX) $(CFLAGS) $(OBJ_C) $(BUILD)/test_main.o $(OBJ_CXX) $(CXXLINK) -o $@ $(LDLIBS)

# OWNERSHIP_CHECK=1: any build checks memory ownership (ADR 0015); never the release default
OWNERSHIP_CHECK ?= 0
ifeq ($(OWNERSHIP_CHECK),1)
  CFLAGS += -DTSR_OWNERSHIP_CHECK
endif
# the sanitizer builds always check memory ownership (ADR 0015): tsr_free aborts on a foreign pointer, a double free
# or a wrong size, and free() of a tsr_alloc pointer is an invalid free
SAN := -fsanitize=address,undefined -fno-sanitize-recover=all -fno-omit-frame-pointer -DTSR_OWNERSHIP_CHECK
OBJ_CXX_SAN = $(patsubst %,$(BUILD)/cxx_san/%.o,$(notdir $(SRC_CXX)))

$(BUILD)/cxx_san/%.cpp.o: cxx/%.cpp cxx/*.hpp src/fn.h
	@mkdir -p $(BUILD)/cxx_san
	$(CXX) -O1 -g -std=c++17 -DBOOST_MATH_STANDALONE $(OPENMP) $(SAN) $(TP_INC) -c -o $@ $<

$(BUILD)/cxx_san/wright.cc.o: third_party/scipy/wright.cc
	@mkdir -p $(BUILD)/cxx_san
	$(CXX) -O1 -g -std=c++17 $(OPENMP) $(SAN) $(TP_INC) -c -o $@ $<

$(BUILD)/test_san: $(SRC) tests/test_main.c $(OBJ_CXX_SAN)
	@mkdir -p $(BUILD)/c_san
	cd $(BUILD)/c_san && $(CC) -O1 -g -std=gnu11 -fno-math-errno -fno-trapping-math $(OPENMP) $(SAN) \
	    -DTSR_NO_CLONES $(WARN) -w -I$(abspath third_party/numpy/include) -c $(abspath $(SRC)) $(abspath tests/test_main.c)
	$(CXX) $(SAN) $(OPENMP) $(BUILD)/c_san/*.o $(OBJ_CXX_SAN) -o $@ $(LDLIBS)

test: $(BUILD)/test_opt
	$(BUILD)/test_opt

sanitize: $(BUILD)/test_san
	ASAN_OPTIONS=detect_leaks=1 $(BUILD)/test_san

clean:
	rm -rf $(BUILD)

.PHONY: all portable test sanitize clean fuzz fuzz-standalone abi-check

# ---- fuzzing (see csrc/fuzz/): coverage-guided with libFuzzer (clang + compiler-rt), or the standalone
#      mutation driver under ASan + UBSan anywhere. Corpora live in fuzz/corpus/<target>.
FUZZ_TARGETS := slice npy mmap_plan json npy_write
# the fuzzed parsers do not need the function registry (C++): build without it
FUZZ_SRC := $(filter-out src/gen_%_table.c src/np_%.c src/scipy_%.c third_party/scipy/cdflib.c $(NPRAND) $(SCIPYC),$(SRC))
# registry targets drive public routines (tsr_fn_routine): they link the whole kernel (C + C++ + LAPACK),
# compiled once per mode into $(BUILD)/fuzzobj_lf (libFuzzer, clang) or $(BUILD)/fuzzobj_sa (standalone, $(CC))
FUZZ_REG_TARGETS := einsum hist textio dtype routines
FUZZ_TIME ?= 60
FUZZ_RUNS ?= 200000
FUZZ_TIMEOUT ?= 25
# FUZZ_DEFS=-DFUZZ_SKIP_KNOWN steers the registry targets around open findings (the FUZZ_SKIP_KNOWN blocks in
# fuzz/fuzz_*.c, and fuzz/known_ubsan.supp) so a long run keeps exploring; the default (empty) reports them
FUZZ_DEFS ?=
FUZZ_UBSAN := halt_on_error=1:abort_on_error=1:print_stacktrace=1$(if $(findstring FUZZ_SKIP_KNOWN,$(FUZZ_DEFS)),:suppressions=$(abspath fuzz/known_ubsan.supp))
FUZZ_FLAGS := -O1 -g -std=gnu11 -Ithird_party/numpy/include -fno-math-errno -fno-trapping-math -fno-omit-frame-pointer -DTSR_NO_CLONES -DTSR_OWNERSHIP_CHECK $(LAPACK_INC) $(WARN)
FUZZ_CXXFLAGS := -O1 -g -std=c++17 -DBOOST_MATH_STANDALONE -fno-omit-frame-pointer $(TP_INC) $(LAPACK_INC) -Icxx
FUZZ_CLANG ?= clang
FUZZ_CLANGXX ?= clang++
# clang's -fsanitize=undefined includes float-cast-overflow, gcc's does not: name it so both modes report the same.
# UBSan checks are built recoverable and made fatal at run time (UBSAN_OPTIONS halt_on_error=1, FUZZ_UBSAN): the
# same stop-at-first-report behaviour as -fno-sanitize-recover, but a suppressions file can then exempt a finding.
FUZZ_SAN_LF := -fsanitize=address,undefined -fno-sanitize-recover=all -fsanitize-recover=undefined
FUZZ_SAN_SA := -fsanitize=address,undefined,float-cast-overflow -fno-sanitize-recover=all -fsanitize-recover=undefined,float-cast-overflow
FUZZ_OBJ_LF = $(patsubst %,$(BUILD)/fuzzobj_lf/%.o,$(basename $(notdir $(SRC) $(SRC_CXX))))
FUZZ_OBJ_SA = $(patsubst %,$(BUILD)/fuzzobj_sa/%.o,$(basename $(notdir $(SRC) $(SRC_CXX))))

define FUZZ_C_RULES
$(BUILD)/fuzzobj_lf/$(basename $(notdir $(1))).o: $(1) src/*.h include/tessero.h
	@mkdir -p $$(@D)
	$(FUZZ_CLANG) $(FUZZ_FLAGS) -w -fsanitize=fuzzer-no-link $(FUZZ_SAN_LF) -c -o $$@ $$<
$(BUILD)/fuzzobj_sa/$(basename $(notdir $(1))).o: $(1) src/*.h include/tessero.h
	@mkdir -p $$(@D)
	$(CC) $(FUZZ_FLAGS) -w $(FUZZ_SAN_SA) -c -o $$@ $$<
endef
define FUZZ_CXX_RULES
$(BUILD)/fuzzobj_lf/$(basename $(notdir $(1))).o: $(1) cxx/*.hpp src/fn.h
	@mkdir -p $$(@D)
	$(FUZZ_CLANGXX) $(FUZZ_CXXFLAGS) -w -fsanitize=fuzzer-no-link $(FUZZ_SAN_LF) -c -o $$@ $$<
$(BUILD)/fuzzobj_sa/$(basename $(notdir $(1))).o: $(1) cxx/*.hpp src/fn.h
	@mkdir -p $$(@D)
	$(CXX) $(FUZZ_CXXFLAGS) -w $(FUZZ_SAN_SA) -c -o $$@ $$<
endef
$(foreach s,$(SRC),$(eval $(call FUZZ_C_RULES,$(s))))
$(foreach s,$(SRC_CXX),$(eval $(call FUZZ_CXX_RULES,$(s))))

fuzz: $(if $(FUZZ_REG_TARGETS),$(FUZZ_OBJ_LF))
	@mkdir -p $(BUILD)
	@for t in $(FUZZ_TARGETS); do \
	    clang $(FUZZ_FLAGS) -fsanitize=fuzzer,address,undefined -fno-sanitize-recover=all \
	        -DTSR_NO_REGISTRY -o $(BUILD)/fuzz_$$t $(FUZZ_SRC) fuzz/fuzz_$$t.c $(LDLIBS) || exit 1; \
	    mkdir -p $(BUILD)/corpus_$$t; \
	    $(BUILD)/fuzz_$$t -max_total_time=$(FUZZ_TIME) -max_len=4096 -seed=$${FUZZ_SEED:-1} $(BUILD)/corpus_$$t fuzz/corpus/$$t || exit 1; \
	done
	@for t in $(FUZZ_REG_TARGETS); do \
	    $(FUZZ_CLANG) $(FUZZ_FLAGS) -fsanitize=fuzzer $(FUZZ_SAN_LF) $(FUZZ_DEFS) -c -o $(BUILD)/fuzzobj_lf/target_$$t.o fuzz/fuzz_$$t.c || exit 1; \
	    $(FUZZ_CLANGXX) -fsanitize=fuzzer $(FUZZ_SAN_LF) -o $(BUILD)/fuzz_$$t $(BUILD)/fuzzobj_lf/target_$$t.o $(FUZZ_OBJ_LF) $(LDLIBS) || exit 1; \
	    mkdir -p $(BUILD)/corpus_$$t; \
	    UBSAN_OPTIONS=$(FUZZ_UBSAN) $(BUILD)/fuzz_$$t -max_total_time=$(FUZZ_TIME) -timeout=$(FUZZ_TIMEOUT) -max_len=4096 -seed=$${FUZZ_SEED:-1} \
	        $(BUILD)/corpus_$$t fuzz/corpus/$$t || exit 1; \
	done

fuzz-standalone: $(if $(FUZZ_REG_TARGETS),$(FUZZ_OBJ_SA))
	@mkdir -p $(BUILD)
	@for t in $(FUZZ_TARGETS); do \
	    $(CC) $(FUZZ_FLAGS) -fsanitize=address,undefined -fno-sanitize-recover=all \
	        -DTSR_NO_REGISTRY -o $(BUILD)/fuzzsa_$$t $(FUZZ_SRC) fuzz/fuzz_$$t.c fuzz/driver.c $(LDLIBS) || exit 1; \
	    ASAN_OPTIONS=detect_leaks=1 UBSAN_OPTIONS=abort_on_error=1:print_stacktrace=1 $(BUILD)/fuzzsa_$$t -runs=$(FUZZ_RUNS) -seed=$${FUZZ_SEED:-1} fuzz/corpus/$$t || exit 1; \
	done
	@for t in $(FUZZ_REG_TARGETS); do \
	    $(CC) $(FUZZ_FLAGS) $(FUZZ_SAN_SA) $(FUZZ_DEFS) -c -o $(BUILD)/fuzzobj_sa/target_$$t.o fuzz/fuzz_$$t.c || exit 1; \
	    $(CC) $(FUZZ_FLAGS) $(FUZZ_SAN_SA) -c -o $(BUILD)/fuzzobj_sa/target_driver.o fuzz/driver.c || exit 1; \
	    $(CXX) $(FUZZ_SAN_SA) -o $(BUILD)/fuzzsa_$$t $(BUILD)/fuzzobj_sa/target_$$t.o $(BUILD)/fuzzobj_sa/target_driver.o \
	        $(FUZZ_OBJ_SA) $(LDLIBS) || exit 1; \
	    ASAN_OPTIONS=detect_leaks=1 UBSAN_OPTIONS=$(FUZZ_UBSAN) $(BUILD)/fuzzsa_$$t -runs=$(FUZZ_RUNS) -seed=$${FUZZ_SEED:-1} -timeout=$(FUZZ_TIMEOUT) \
	        -artifact_prefix=$(BUILD)/fuzzsa_$$t- fuzz/corpus/$$t || exit 1; \
	done
