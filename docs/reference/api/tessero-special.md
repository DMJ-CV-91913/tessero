# Special

`Tessero\Special`

scipy.special: special functions (gamma, Bessel, error functions, orthogonal polynomials, ...). Element-wise: arguments broadcast as in NumPy, all-scalar calls return floats.

Names are the SciPy/NumPy names in camelCase (log_ndtr -> logNdtr); apply() also takes the original names.
The native extension provides the same class as Tessero\Ext\Special.

## Constants

| Name | Value |
|---|---|
| `FUNCTIONS` | `{"agm":"agm","airy":"airy","airye":"airye","bdtr":"bdtr","bdtrc":"bdtrc","bdtri":"bdtri","bdtrik":"bdtrik","bdtrin":"bdtrin","bei":"bei","beip":"beip","ber":"ber","berp":"berp","besselpoly":"besselpoly","beta":"beta","betainc":"betainc","betaincc":"betaincc","betainccinv":"betainccinv","betaincinv":"betaincinv","betaln":"betaln","binom":"binom","boxcox":"boxcox","boxcox1p":"boxcox1p","btdtria":"btdtria","btdtrib":"btdtrib","cbrt":"cbrt","chdtr":"chdtr","chdtrc":"chdtrc","chdtri":"chdtri","chdtriv":"chdtriv","chndtr":"chndtr","chndtridf":"chndtridf","chndtrinc":"chndtrinc","chndtrix":"chndtrix","cosdg":"cosdg","cosm1":"cosm1","cotdg":"cotdg","dawsn":"dawsn","digamma":"digamma","ellipe":"ellipe","ellipeinc":"ellipeinc","ellipj":"ellipj","ellipk":"ellipk","ellipkinc":"ellipkinc","ellipkm1":"ellipkm1","elliprc":"elliprc","elliprd":"elliprd","elliprf":"elliprf","elliprg":"elliprg","elliprj":"elliprj","entr":"entr","erf":"erf","erfc":"erfc","erfcinv":"erfcinv","erfcx":"erfcx","erfi":"erfi","erfinv":"erfinv","eval_chebyc":"evalChebyc","eval_chebys":"evalChebys","eval_chebyt":"evalChebyt","eval_chebyu":"evalChebyu","eval_gegenbauer":"evalGegenbauer","eval_genlaguerre":"evalGenlaguerre","eval_hermite":"evalHermite","eval_hermitenorm":"evalHermitenorm","eval_jacobi":"evalJacobi","eval_laguerre":"evalLaguerre","eval_legendre":"evalLegendre","eval_sh_chebyt":"evalShChebyt","eval_sh_chebyu":"evalShChebyu","eval_sh_jacobi":"evalShJacobi","eval_sh_legendre":"evalShLegendre","exp1":"exp1","exp10":"exp10","exp2":"exp2","expi":"expi","expit":"expit","expm1":"expm1","expn":"expn","exprel":"exprel","fdtr":"fdtr","fdtrc":"fdtrc","fdtri":"fdtri","fdtridfd":"fdtridfd","fresnel":"fresnel","gamma":"gamma","gammainc":"gammainc","gammaincc":"gammaincc","gammainccinv":"gammainccinv","gammaincinv":"gammaincinv","gammaln":"gammaln","gammasgn":"gammasgn","gdtr":"gdtr","gdtrc":"gdtrc","gdtria":"gdtria","gdtrib":"gdtrib","gdtrix":"gdtrix","huber":"huber","hyp0f1":"hyp0f1","hyp1f1":"hyp1f1","hyp2f1":"hyp2f1","hyperu":"hyperu","i0":"i0","i0e":"i0e","i1":"i1","i1e":"i1e","inv_boxcox":"invBoxcox","inv_boxcox1p":"invBoxcox1p","it2i0k0":"it2i0k0","it2j0y0":"it2j0y0","it2struve0":"it2struve0","itairy":"itairy","iti0k0":"iti0k0","itj0y0":"itj0y0","itmodstruve0":"itmodstruve0","itstruve0":"itstruve0","iv":"iv","ive":"ive","j0":"j0","j1":"j1","jn":"jn","jv":"jv","jve":"jve","k0":"k0","k0e":"k0e","k1":"k1","k1e":"k1e","kei":"kei","keip":"keip","ker":"ker","kerp":"kerp","kl_div":"klDiv","kn":"kn","kolmogi":"kolmogi","kolmogorov":"kolmogorov","kv":"kv","kve":"kve","log1p":"log1p","log_expit":"logExpit","log_ndtr":"logNdtr","log_wright_bessel":"logWrightBessel","loggamma":"loggamma","logit":"logit","lpmv":"lpmv","mathieu_a":"mathieuA","mathieu_b":"mathieuB","mathieu_cem":"mathieuCem","mathieu_modcem1":"mathieuModcem1","mathieu_modcem2":"mathieuModcem2","mathieu_modsem1":"mathieuModsem1","mathieu_modsem2":"mathieuModsem2","mathieu_sem":"mathieuSem","modstruve":"modstruve","nbdtr":"nbdtr","nbdtrc":"nbdtrc","nbdtri":"nbdtri","nbdtrik":"nbdtrik","nbdtrin":"nbdtrin","ncfdtr":"ncfdtr","ncfdtri":"ncfdtri","ncfdtridfd":"ncfdtridfd","ncfdtridfn":"ncfdtridfn","ncfdtrinc":"ncfdtrinc","nctdtr":"nctdtr","nctdtridf":"nctdtridf","nctdtrinc":"nctdtrinc","nctdtrit":"nctdtrit","ndtr":"ndtr","ndtri":"ndtri","ndtri_exp":"ndtriExp","nrdtrimn":"nrdtrimn","nrdtrisd":"nrdtrisd","obl_ang1":"oblAng1","obl_ang1_cv":"oblAng1Cv","obl_cv":"oblCv","obl_rad1":"oblRad1","obl_rad1_cv":"oblRad1Cv","obl_rad2":"oblRad2","obl_rad2_cv":"oblRad2Cv","owens_t":"owensT","pbdv":"pbdv","pbvv":"pbvv","pbwa":"pbwa","pdtr":"pdtr","pdtrc":"pdtrc","pdtri":"pdtri","pdtrik":"pdtrik","poch":"poch","powm1":"powm1","pro_ang1":"proAng1","pro_ang1_cv":"proAng1Cv","pro_cv":"proCv","pro_rad1":"proRad1","pro_rad1_cv":"proRad1Cv","pro_rad2":"proRad2","pro_rad2_cv":"proRad2Cv","pseudo_huber":"pseudoHuber","psi":"psi","radian":"radian","rel_entr":"relEntr","rgamma":"rgamma","round":"round","shichi":"shichi","sici":"sici","sindg":"sindg","smirnov":"smirnov","smirnovi":"smirnovi","spence":"spence","stdtr":"stdtr","stdtridf":"stdtridf","stdtrit":"stdtrit","struve":"struve","tandg":"tandg","tklmbda":"tklmbda","voigt_profile":"voigtProfile","wright_bessel":"wrightBessel","wrightomega":"wrightomega","xlog1py":"xlog1py","xlogy":"xlogy","y0":"y0","y1":"y1","yn":"yn","yv":"yv","yve":"yve","zetac":"zetac"}` |

## Methods

### agm

```php
static agm(mixed $a, mixed $b, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Compute the arithmetic-geometric mean of `a` and `b`.

scipy.special.agm

### airy

```php
static airy(mixed $z, ?array $out = null): array
```

Airy functions and their derivatives.

scipy.special.airy

### airye

```php
static airye(mixed $z, ?array $out = null): array
```

Exponentially scaled Airy functions and their derivatives.

scipy.special.airye

### apply

```php
static apply(string $name, mixed ...$args): mixed
```

Call a function by its SciPy/NumPy name: Special::apply('log_ndtr', $x).

### bdtr

```php
static bdtr(mixed $k, mixed $n, mixed $p, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Binomial distribution cumulative distribution function.

scipy.special.bdtr

### bdtrc

```php
static bdtrc(mixed $k, mixed $n, mixed $p, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Binomial distribution survival function.

scipy.special.bdtrc

### bdtri

```php
static bdtri(mixed $k, mixed $n, mixed $y, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Inverse function to `bdtr` with respect to `p`.

scipy.special.bdtri

### bdtrik

```php
static bdtrik(mixed $y, mixed $n, mixed $p, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Inverse function to `bdtr` with respect to `k`.

scipy.special.bdtrik

### bdtrin

```php
static bdtrin(mixed $k, mixed $y, mixed $p, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Inverse function to `bdtr` with respect to `n`.

scipy.special.bdtrin

### bei

```php
static bei(mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Kelvin function bei.

scipy.special.bei

### beip

```php
static beip(mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Derivative of the Kelvin function bei.

scipy.special.beip

### ber

```php
static ber(mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Kelvin function ber.

scipy.special.ber

### berp

```php
static berp(mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Derivative of the Kelvin function ber.

scipy.special.berp

### besselpoly

```php
static besselpoly(mixed $a, mixed $lmb, mixed $nu, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Weighted integral of the Bessel function of the first kind.

scipy.special.besselpoly

### beta

```php
static beta(mixed $a, mixed $b, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Beta function.

scipy.special.beta

### betainc

```php
static betainc(mixed $a, mixed $b, mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Regularized incomplete beta function.

scipy.special.betainc

### betaincc

```php
static betaincc(mixed $a, mixed $b, mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Complement of the regularized incomplete beta function.

scipy.special.betaincc

### betainccinv

```php
static betainccinv(mixed $a, mixed $b, mixed $y, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Inverse of the complemented regularized incomplete beta function.

scipy.special.betainccinv

### betaincinv

```php
static betaincinv(mixed $a, mixed $b, mixed $y, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Inverse of the regularized incomplete beta function.

scipy.special.betaincinv

### betaln

```php
static betaln(mixed $a, mixed $b, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Natural logarithm of absolute value of beta function.

scipy.special.betaln

### binom

```php
static binom(mixed $x, mixed $y, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Binomial coefficient considered as a function of two real variables.

scipy.special.binom

### boxcox

```php
static boxcox(mixed $x, mixed $lmbda, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Compute the Box-Cox transformation.

scipy.special.boxcox

### boxcox1p

```php
static boxcox1p(mixed $x, mixed $lmbda, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Compute the Box-Cox transformation of 1 + `x`.

scipy.special.boxcox1p

### btdtria

```php
static btdtria(mixed $p, mixed $b, mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Inverse of `betainc` with respect to `a`.

scipy.special.btdtria

### btdtrib

```php
static btdtrib(mixed $x0, mixed $x1, mixed $x2, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

btdtria(a, p, x, out=None).

scipy.special.btdtrib

### cbrt

```php
static cbrt(mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Element-wise cube root of `x`.

scipy.special.cbrt

### chdtr

```php
static chdtr(mixed $v, mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Chi square cumulative distribution function.

scipy.special.chdtr

### chdtrc

```php
static chdtrc(mixed $v, mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Chi square survival function.

scipy.special.chdtrc

### chdtri

```php
static chdtri(mixed $v, mixed $p, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Inverse to `chdtrc` with respect to `x`.

scipy.special.chdtri

### chdtriv

```php
static chdtriv(mixed $p, mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Inverse to `chdtr` with respect to `v`.

scipy.special.chdtriv

### chndtr

```php
static chndtr(mixed $x, mixed $df, mixed $nc, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Non-central chi square cumulative distribution function.

scipy.special.chndtr

### chndtridf

```php
static chndtridf(mixed $x, mixed $p, mixed $nc, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Inverse to `chndtr` vs `df`.

scipy.special.chndtridf

### chndtrinc

```php
static chndtrinc(mixed $x, mixed $df, mixed $p, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Inverse to `chndtr` vs `nc`.

scipy.special.chndtrinc

### chndtrix

```php
static chndtrix(mixed $p, mixed $df, mixed $nc, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Inverse to `chndtr` vs `x`.

scipy.special.chndtrix

### cosdg

```php
static cosdg(mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Cosine of the angle `x` given in degrees.

scipy.special.cosdg

### cosm1

```php
static cosm1(mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

cos(x) - 1 for use when `x` is near zero.

scipy.special.cosm1

### cotdg

```php
static cotdg(mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Cotangent of the angle `x` given in degrees.

scipy.special.cotdg

### dawsn

```php
static dawsn(mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Dawson's integral.

scipy.special.dawsn

### digamma

```php
static digamma(mixed $z, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

The digamma function (alias of psi).

scipy.special.digamma

### ellipe

```php
static ellipe(mixed $m, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Complete elliptic integral of the second kind.

scipy.special.ellipe

### ellipeinc

```php
static ellipeinc(mixed $phi, mixed $m, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Incomplete elliptic integral of the second kind.

scipy.special.ellipeinc

### ellipj

```php
static ellipj(mixed $u, mixed $m, ?array $out = null): array
```

Jacobi elliptic functions.

scipy.special.ellipj

### ellipk

```php
static ellipk(mixed $m, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Complete elliptic integral of the first kind.

scipy.special.ellipk

### ellipkinc

```php
static ellipkinc(mixed $phi, mixed $m, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Incomplete elliptic integral of the first kind.

scipy.special.ellipkinc

### ellipkm1

```php
static ellipkm1(mixed $p, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Complete elliptic integral of the first kind around `m` = 1.

scipy.special.ellipkm1

### elliprc

```php
static elliprc(mixed $x, mixed $y, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Degenerate symmetric elliptic integral.

scipy.special.elliprc

### elliprd

```php
static elliprd(mixed $x, mixed $y, mixed $z, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Symmetric elliptic integral of the second kind.

scipy.special.elliprd

### elliprf

```php
static elliprf(mixed $x, mixed $y, mixed $z, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Completely-symmetric elliptic integral of the first kind.

scipy.special.elliprf

### elliprg

```php
static elliprg(mixed $x, mixed $y, mixed $z, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Completely-symmetric elliptic integral of the second kind.

scipy.special.elliprg

### elliprj

```php
static elliprj(mixed $x, mixed $y, mixed $z, mixed $p, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Symmetric elliptic integral of the third kind.

scipy.special.elliprj

### entr

```php
static entr(mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Elementwise function for computing entropy.

scipy.special.entr

### erf

```php
static erf(mixed $z, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Returns the error function of complex argument.

scipy.special.erf

### erfc

```php
static erfc(mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Complementary error function, ``1 - erf(x)``.

scipy.special.erfc

### erfcinv

```php
static erfcinv(mixed $y, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Inverse of the complementary error function.

scipy.special.erfcinv

### erfcx

```php
static erfcx(mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Scaled complementary error function, ``exp(x**2) * erfc(x)``.

scipy.special.erfcx

### erfi

```php
static erfi(mixed $z, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Imaginary error function, ``-i erf(i z)``.

scipy.special.erfi

### erfinv

```php
static erfinv(mixed $y, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Inverse of the error function.

scipy.special.erfinv

### evalChebyc

```php
static evalChebyc(mixed $n, mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Evaluate Chebyshev polynomial of the first kind on [-2, 2] at a point.

scipy.special.eval_chebyc

### evalChebys

```php
static evalChebys(mixed $n, mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Evaluate Chebyshev polynomial of the second kind on [-2, 2] at a point.

scipy.special.eval_chebys

### evalChebyt

```php
static evalChebyt(mixed $n, mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Evaluate Chebyshev polynomial of the first kind at a point.

scipy.special.eval_chebyt

### evalChebyu

```php
static evalChebyu(mixed $n, mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Evaluate Chebyshev polynomial of the second kind at a point.

scipy.special.eval_chebyu

### evalGegenbauer

```php
static evalGegenbauer(mixed $n, mixed $alpha, mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Evaluate Gegenbauer polynomial at a point.

scipy.special.eval_gegenbauer

### evalGenlaguerre

```php
static evalGenlaguerre(mixed $n, mixed $alpha, mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Evaluate generalized Laguerre polynomial at a point.

scipy.special.eval_genlaguerre

### evalHermite

```php
static evalHermite(mixed $n, mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Evaluate physicist's Hermite polynomial at a point.

scipy.special.eval_hermite

### evalHermitenorm

```php
static evalHermitenorm(mixed $n, mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Evaluate probabilist's (normalized) Hermite polynomial at a point.

scipy.special.eval_hermitenorm

### evalJacobi

```php
static evalJacobi(mixed $n, mixed $alpha, mixed $beta, mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Evaluate Jacobi polynomial at a point.

scipy.special.eval_jacobi

### evalLaguerre

```php
static evalLaguerre(mixed $n, mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Evaluate Laguerre polynomial at a point.

scipy.special.eval_laguerre

### evalLegendre

```php
static evalLegendre(mixed $n, mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Evaluate Legendre polynomial at a point.

scipy.special.eval_legendre

### evalShChebyt

```php
static evalShChebyt(mixed $n, mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Evaluate shifted Chebyshev polynomial of the first kind at a point.

scipy.special.eval_sh_chebyt

### evalShChebyu

```php
static evalShChebyu(mixed $n, mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Evaluate shifted Chebyshev polynomial of the second kind at a point.

scipy.special.eval_sh_chebyu

### evalShJacobi

```php
static evalShJacobi(mixed $n, mixed $p, mixed $q, mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Evaluate shifted Jacobi polynomial at a point.

scipy.special.eval_sh_jacobi

### evalShLegendre

```php
static evalShLegendre(mixed $n, mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Evaluate shifted Legendre polynomial at a point.

scipy.special.eval_sh_legendre

### exp1

```php
static exp1(mixed $z, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Exponential integral E1.

scipy.special.exp1

### exp10

```php
static exp10(mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Compute ``10**x`` element-wise.

scipy.special.exp10

### exp2

```php
static exp2(mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Compute ``2**x`` element-wise.

scipy.special.exp2

### expi

```php
static expi(mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Exponential integral Ei.

scipy.special.expi

### expit

```php
static expit(mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Expit (a.k.a. logistic sigmoid) ufunc for ndarrays.

scipy.special.expit

### expm1

```php
static expm1(mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Compute ``exp(x) - 1``.

scipy.special.expm1

### expn

```php
static expn(mixed $n, mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Generalized exponential integral En.

scipy.special.expn

### exprel

```php
static exprel(mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Relative error exponential, ``(exp(x) - 1)/x``.

scipy.special.exprel

### fdtr

```php
static fdtr(mixed $dfn, mixed $dfd, mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

F cumulative distribution function.

scipy.special.fdtr

### fdtrc

```php
static fdtrc(mixed $dfn, mixed $dfd, mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

F survival function.

scipy.special.fdtrc

### fdtri

```php
static fdtri(mixed $dfn, mixed $dfd, mixed $p, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

The `p`-th quantile of the F-distribution.

scipy.special.fdtri

### fdtridfd

```php
static fdtridfd(mixed $dfn, mixed $p, mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Inverse to `fdtr` vs dfd.

scipy.special.fdtridfd

### fresnel

```php
static fresnel(mixed $z, ?array $out = null): array
```

Fresnel integrals.

scipy.special.fresnel

### gamma

```php
static gamma(mixed $z, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

gamma function.

scipy.special.gamma

### gammainc

```php
static gammainc(mixed $a, mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Regularized lower incomplete gamma function.

scipy.special.gammainc

### gammaincc

```php
static gammaincc(mixed $a, mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Regularized upper incomplete gamma function.

scipy.special.gammaincc

### gammainccinv

```php
static gammainccinv(mixed $a, mixed $y, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Inverse of the regularized upper incomplete gamma function.

scipy.special.gammainccinv

### gammaincinv

```php
static gammaincinv(mixed $a, mixed $y, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Inverse to the regularized lower incomplete gamma function.

scipy.special.gammaincinv

### gammaln

```php
static gammaln(mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Logarithm of the absolute value of the gamma function.

scipy.special.gammaln

### gammasgn

```php
static gammasgn(mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Sign of the gamma function.

scipy.special.gammasgn

### gdtr

```php
static gdtr(mixed $a, mixed $b, mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Gamma distribution cumulative distribution function.

scipy.special.gdtr

### gdtrc

```php
static gdtrc(mixed $a, mixed $b, mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Gamma distribution survival function.

scipy.special.gdtrc

### gdtria

```php
static gdtria(mixed $p, mixed $b, mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Inverse of `gdtr` vs a.

scipy.special.gdtria

### gdtrib

```php
static gdtrib(mixed $a, mixed $p, mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Inverse of `gdtr` vs b.

scipy.special.gdtrib

### gdtrix

```php
static gdtrix(mixed $a, mixed $b, mixed $p, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Inverse of `gdtr` vs x.

scipy.special.gdtrix

### huber

```php
static huber(mixed $delta, mixed $r, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Huber loss function.

scipy.special.huber

### hyp0f1

```php
static hyp0f1(mixed $v, mixed $z, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Confluent hypergeometric limit function 0F1.

scipy.special.hyp0f1

### hyp1f1

```php
static hyp1f1(mixed $a, mixed $b, mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Confluent hypergeometric function 1F1.

scipy.special.hyp1f1

### hyp2f1

```php
static hyp2f1(mixed $a, mixed $b, mixed $c, mixed $z, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Gauss hypergeometric function 2F1(a, b; c; z).

scipy.special.hyp2f1

### hyperu

```php
static hyperu(mixed $a, mixed $b, mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Confluent hypergeometric function U.

scipy.special.hyperu

### i0

```php
static i0(mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Modified Bessel function of order 0.

scipy.special.i0

### i0e

```php
static i0e(mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Exponentially scaled modified Bessel function of order 0.

scipy.special.i0e

### i1

```php
static i1(mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Modified Bessel function of order 1.

scipy.special.i1

### i1e

```php
static i1e(mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Exponentially scaled modified Bessel function of order 1.

scipy.special.i1e

### invBoxcox

```php
static invBoxcox(mixed $y, mixed $lmbda, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Compute the inverse of the Box-Cox transformation.

scipy.special.inv_boxcox

### invBoxcox1p

```php
static invBoxcox1p(mixed $y, mixed $lmbda, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Compute the inverse of the Box-Cox transformation.

scipy.special.inv_boxcox1p

### it2i0k0

```php
static it2i0k0(mixed $x, ?array $out = null): array
```

Integrals related to modified Bessel functions of order 0.

scipy.special.it2i0k0

### it2j0y0

```php
static it2j0y0(mixed $x, ?array $out = null): array
```

Integrals related to Bessel functions of the first kind of order 0.

scipy.special.it2j0y0

### it2struve0

```php
static it2struve0(mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Integral related to the Struve function of order 0.

scipy.special.it2struve0

### itairy

```php
static itairy(mixed $x, ?array $out = null): array
```

Integrals of Airy functions.

scipy.special.itairy

### iti0k0

```php
static iti0k0(mixed $x, ?array $out = null): array
```

Integrals of modified Bessel functions of order 0.

scipy.special.iti0k0

### itj0y0

```php
static itj0y0(mixed $x, ?array $out = null): array
```

Integrals of Bessel functions of the first kind of order 0.

scipy.special.itj0y0

### itmodstruve0

```php
static itmodstruve0(mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Integral of the modified Struve function of order 0.

scipy.special.itmodstruve0

### itstruve0

```php
static itstruve0(mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Integral of the Struve function of order 0.

scipy.special.itstruve0

### iv

```php
static iv(mixed $v, mixed $z, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Modified Bessel function of the first kind of real order.

scipy.special.iv

### ive

```php
static ive(mixed $v, mixed $z, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Exponentially scaled modified Bessel function of the first kind.

scipy.special.ive

### j0

```php
static j0(mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Bessel function of the first kind of order 0.

scipy.special.j0

### j1

```php
static j1(mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Bessel function of the first kind of order 1.

scipy.special.j1

### jn

```php
static jn(mixed $n, mixed $z, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Bessel function of the first kind of real order and complex argument (alias of jv).

scipy.special.jn

### jv

```php
static jv(mixed $v, mixed $z, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Bessel function of the first kind of real order and complex argument.

scipy.special.jv

### jve

```php
static jve(mixed $v, mixed $z, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Exponentially scaled Bessel function of the first kind of order `v`.

scipy.special.jve

### k0

```php
static k0(mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Modified Bessel function of the second kind of order 0, :math:`K_0`.

scipy.special.k0

### k0e

```php
static k0e(mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Exponentially scaled modified Bessel function K of order 0.

scipy.special.k0e

### k1

```php
static k1(mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Modified Bessel function of the second kind of order 1, :math:`K_1(x)`.

scipy.special.k1

### k1e

```php
static k1e(mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Exponentially scaled modified Bessel function K of order 1.

scipy.special.k1e

### kei

```php
static kei(mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Kelvin function kei.

scipy.special.kei

### keip

```php
static keip(mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Derivative of the Kelvin function kei.

scipy.special.keip

### ker

```php
static ker(mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Kelvin function ker.

scipy.special.ker

### kerp

```php
static kerp(mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Derivative of the Kelvin function ker.

scipy.special.kerp

### klDiv

```php
static klDiv(mixed $x, mixed $y, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Elementwise function for computing Kullback-Leibler divergence.

scipy.special.kl_div

### kn

```php
static kn(mixed $n, mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Modified Bessel function of the second kind of integer order `n`.

scipy.special.kn

### kolmogi

```php
static kolmogi(mixed $p, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Inverse Survival Function of Kolmogorov distribution.

scipy.special.kolmogi

### kolmogorov

```php
static kolmogorov(mixed $y, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Complementary cumulative distribution (Survival Function) function of Kolmogorov distribution.

scipy.special.kolmogorov

### kv

```php
static kv(mixed $v, mixed $z, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Modified Bessel function of the second kind of real order `v`.

scipy.special.kv

### kve

```php
static kve(mixed $v, mixed $z, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Exponentially scaled modified Bessel function of the second kind.

scipy.special.kve

### log1p

```php
static log1p(mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Calculates log(1 + x) for use when `x` is near zero.

scipy.special.log1p

### logExpit

```php
static logExpit(mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Logarithm of the logistic sigmoid function.

scipy.special.log_expit

### logNdtr

```php
static logNdtr(mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Logarithm of Gaussian cumulative distribution function.

scipy.special.log_ndtr

### logWrightBessel

```php
static logWrightBessel(mixed $a, mixed $b, mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Natural logarithm of Wright's generalized Bessel function, see `wright_bessel`. This function comes in handy in particular for large values of x.

scipy.special.log_wright_bessel

### loggamma

```php
static loggamma(mixed $z, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Principal branch of the logarithm of the gamma function.

scipy.special.loggamma

### logit

```php
static logit(mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Logit ufunc for ndarrays.

scipy.special.logit

### lpmv

```php
static lpmv(mixed $m, mixed $v, mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Associated Legendre function of integer order and real degree.

scipy.special.lpmv

### mathieuA

```php
static mathieuA(mixed $m, mixed $q, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Characteristic value of even Mathieu functions.

scipy.special.mathieu_a

### mathieuB

```php
static mathieuB(mixed $m, mixed $q, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Characteristic value of odd Mathieu functions.

scipy.special.mathieu_b

### mathieuCem

```php
static mathieuCem(mixed $m, mixed $q, mixed $x, ?array $out = null): array
```

Even Mathieu function and its derivative.

scipy.special.mathieu_cem

### mathieuModcem1

```php
static mathieuModcem1(mixed $m, mixed $q, mixed $x, ?array $out = null): array
```

Even modified Mathieu function of the first kind and its derivative.

scipy.special.mathieu_modcem1

### mathieuModcem2

```php
static mathieuModcem2(mixed $m, mixed $q, mixed $x, ?array $out = null): array
```

Even modified Mathieu function of the second kind and its derivative.

scipy.special.mathieu_modcem2

### mathieuModsem1

```php
static mathieuModsem1(mixed $m, mixed $q, mixed $x, ?array $out = null): array
```

Odd modified Mathieu function of the first kind and its derivative.

scipy.special.mathieu_modsem1

### mathieuModsem2

```php
static mathieuModsem2(mixed $m, mixed $q, mixed $x, ?array $out = null): array
```

Odd modified Mathieu function of the second kind and its derivative.

scipy.special.mathieu_modsem2

### mathieuSem

```php
static mathieuSem(mixed $m, mixed $q, mixed $x, ?array $out = null): array
```

Odd Mathieu function and its derivative.

scipy.special.mathieu_sem

### modstruve

```php
static modstruve(mixed $v, mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Modified Struve function.

scipy.special.modstruve

### nbdtr

```php
static nbdtr(mixed $k, mixed $n, mixed $p, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Negative binomial cumulative distribution function.

scipy.special.nbdtr

### nbdtrc

```php
static nbdtrc(mixed $k, mixed $n, mixed $p, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Negative binomial survival function.

scipy.special.nbdtrc

### nbdtri

```php
static nbdtri(mixed $k, mixed $n, mixed $y, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Returns the inverse with respect to the parameter `p` of ``y = nbdtr(k, n, p)``, the negative binomial cumulative distribution function.

scipy.special.nbdtri

### nbdtrik

```php
static nbdtrik(mixed $y, mixed $n, mixed $p, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Negative binomial percentile function.

scipy.special.nbdtrik

### nbdtrin

```php
static nbdtrin(mixed $k, mixed $y, mixed $p, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Inverse of `nbdtr` vs `n`.

scipy.special.nbdtrin

### ncfdtr

```php
static ncfdtr(mixed $dfn, mixed $dfd, mixed $nc, mixed $f, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Cumulative distribution function of the non-central F distribution.

scipy.special.ncfdtr

### ncfdtri

```php
static ncfdtri(mixed $dfn, mixed $dfd, mixed $nc, mixed $p, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Inverse with respect to `f` of the CDF of the non-central F distribution.

scipy.special.ncfdtri

### ncfdtridfd

```php
static ncfdtridfd(mixed $dfn, mixed $p, mixed $nc, mixed $f, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Calculate degrees of freedom (denominator) for the noncentral F-distribution.

scipy.special.ncfdtridfd

### ncfdtridfn

```php
static ncfdtridfn(mixed $p, mixed $dfd, mixed $nc, mixed $f, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Calculate degrees of freedom (numerator) for the noncentral F-distribution.

scipy.special.ncfdtridfn

### ncfdtrinc

```php
static ncfdtrinc(mixed $dfn, mixed $dfd, mixed $p, mixed $f, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Calculate non-centrality parameter for non-central F distribution.

scipy.special.ncfdtrinc

### nctdtr

```php
static nctdtr(mixed $df, mixed $nc, mixed $t, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Cumulative distribution function of the non-central `t` distribution.

scipy.special.nctdtr

### nctdtridf

```php
static nctdtridf(mixed $p, mixed $nc, mixed $t, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Calculate degrees of freedom for non-central t distribution.

scipy.special.nctdtridf

### nctdtrinc

```php
static nctdtrinc(mixed $df, mixed $p, mixed $t, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Calculate non-centrality parameter for non-central t distribution.

scipy.special.nctdtrinc

### nctdtrit

```php
static nctdtrit(mixed $df, mixed $nc, mixed $p, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Inverse cumulative distribution function of the non-central t distribution.

scipy.special.nctdtrit

### ndtr

```php
static ndtr(mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Cumulative distribution of the standard normal distribution.

scipy.special.ndtr

### ndtri

```php
static ndtri(mixed $y, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Inverse of `ndtr` vs x.

scipy.special.ndtri

### ndtriExp

```php
static ndtriExp(mixed $y, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Inverse of `log_ndtr` vs x. Allows for greater precision than `ndtri` composed with `numpy.exp` for very small values of y and for y close to 0.

scipy.special.ndtri_exp

### nrdtrimn

```php
static nrdtrimn(mixed $p, mixed $std, mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Calculate mean of normal distribution given other params.

scipy.special.nrdtrimn

### nrdtrisd

```php
static nrdtrisd(mixed $mn, mixed $p, mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Calculate standard deviation of normal distribution given other params.

scipy.special.nrdtrisd

### oblAng1

```php
static oblAng1(mixed $m, mixed $n, mixed $c, mixed $x, ?array $out = null): array
```

Oblate spheroidal angular function of the first kind and its derivative.

scipy.special.obl_ang1

### oblAng1Cv

```php
static oblAng1Cv(mixed $m, mixed $n, mixed $c, mixed $cv, mixed $x, ?array $out = null): array
```

Oblate spheroidal angular function obl_ang1 for precomputed characteristic value.

scipy.special.obl_ang1_cv

### oblCv

```php
static oblCv(mixed $m, mixed $n, mixed $c, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Characteristic value of oblate spheroidal function.

scipy.special.obl_cv

### oblRad1

```php
static oblRad1(mixed $m, mixed $n, mixed $c, mixed $x, ?array $out = null): array
```

Oblate spheroidal radial function of the first kind and its derivative.

scipy.special.obl_rad1

### oblRad1Cv

```php
static oblRad1Cv(mixed $m, mixed $n, mixed $c, mixed $cv, mixed $x, ?array $out = null): array
```

Oblate spheroidal radial function obl_rad1 for precomputed characteristic value.

scipy.special.obl_rad1_cv

### oblRad2

```php
static oblRad2(mixed $m, mixed $n, mixed $c, mixed $x, ?array $out = null): array
```

Oblate spheroidal radial function of the second kind and its derivative.

scipy.special.obl_rad2

### oblRad2Cv

```php
static oblRad2Cv(mixed $m, mixed $n, mixed $c, mixed $cv, mixed $x, ?array $out = null): array
```

Oblate spheroidal radial function obl_rad2 for precomputed characteristic value.

scipy.special.obl_rad2_cv

### owensT

```php
static owensT(mixed $h, mixed $a, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Owen's T Function.

scipy.special.owens_t

### pbdv

```php
static pbdv(mixed $v, mixed $x, ?array $out = null): array
```

Parabolic cylinder function D.

scipy.special.pbdv

### pbvv

```php
static pbvv(mixed $v, mixed $x, ?array $out = null): array
```

Parabolic cylinder function V.

scipy.special.pbvv

### pbwa

```php
static pbwa(mixed $a, mixed $x, ?array $out = null): array
```

Parabolic cylinder function W.

scipy.special.pbwa

### pdtr

```php
static pdtr(mixed $k, mixed $m, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Poisson cumulative distribution function.

scipy.special.pdtr

### pdtrc

```php
static pdtrc(mixed $k, mixed $m, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Poisson survival function.

scipy.special.pdtrc

### pdtri

```php
static pdtri(mixed $k, mixed $y, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Inverse to `pdtr` vs m.

scipy.special.pdtri

### pdtrik

```php
static pdtrik(mixed $p, mixed $m, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Inverse to `pdtr` vs `k`.

scipy.special.pdtrik

### poch

```php
static poch(mixed $z, mixed $m, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Pochhammer symbol.

scipy.special.poch

### powm1

```php
static powm1(mixed $x, mixed $y, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Computes ``x**y - 1``.

scipy.special.powm1

### proAng1

```php
static proAng1(mixed $m, mixed $n, mixed $c, mixed $x, ?array $out = null): array
```

Prolate spheroidal angular function of the first kind and its derivative.

scipy.special.pro_ang1

### proAng1Cv

```php
static proAng1Cv(mixed $m, mixed $n, mixed $c, mixed $cv, mixed $x, ?array $out = null): array
```

Prolate spheroidal angular function pro_ang1 for precomputed characteristic value.

scipy.special.pro_ang1_cv

### proCv

```php
static proCv(mixed $m, mixed $n, mixed $c, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Characteristic value of prolate spheroidal function.

scipy.special.pro_cv

### proRad1

```php
static proRad1(mixed $m, mixed $n, mixed $c, mixed $x, ?array $out = null): array
```

Prolate spheroidal radial function of the first kind and its derivative.

scipy.special.pro_rad1

### proRad1Cv

```php
static proRad1Cv(mixed $m, mixed $n, mixed $c, mixed $cv, mixed $x, ?array $out = null): array
```

Prolate spheroidal radial function pro_rad1 for precomputed characteristic value.

scipy.special.pro_rad1_cv

### proRad2

```php
static proRad2(mixed $m, mixed $n, mixed $c, mixed $x, ?array $out = null): array
```

Prolate spheroidal radial function of the second kind and its derivative.

scipy.special.pro_rad2

### proRad2Cv

```php
static proRad2Cv(mixed $m, mixed $n, mixed $c, mixed $cv, mixed $x, ?array $out = null): array
```

Prolate spheroidal radial function pro_rad2 for precomputed characteristic value.

scipy.special.pro_rad2_cv

### pseudoHuber

```php
static pseudoHuber(mixed $delta, mixed $r, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Pseudo-Huber loss function.

scipy.special.pseudo_huber

### psi

```php
static psi(mixed $z, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

The digamma function.

scipy.special.psi

### radian

```php
static radian(mixed $d, mixed $m, mixed $s, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Convert from degrees to radians.

scipy.special.radian

### relEntr

```php
static relEntr(mixed $x, mixed $y, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Elementwise function for computing relative entropy.

scipy.special.rel_entr

### rgamma

```php
static rgamma(mixed $z, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Reciprocal of the gamma function.

scipy.special.rgamma

### round

```php
static round(mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Round to the nearest integer.

scipy.special.round

### shichi

```php
static shichi(mixed $x, ?array $out = null): array
```

Hyperbolic sine and cosine integrals.

scipy.special.shichi

### sici

```php
static sici(mixed $x, ?array $out = null): array
```

Sine and cosine integrals.

scipy.special.sici

### sindg

```php
static sindg(mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Sine of the angle `x` given in degrees.

scipy.special.sindg

### smirnov

```php
static smirnov(mixed $n, mixed $d, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Kolmogorov-Smirnov complementary cumulative distribution function.

scipy.special.smirnov

### smirnovi

```php
static smirnovi(mixed $n, mixed $p, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Inverse to `smirnov`.

scipy.special.smirnovi

### spence

```php
static spence(mixed $z, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Spence's function, also known as the dilogarithm.

scipy.special.spence

### stdtr

```php
static stdtr(mixed $df, mixed $t, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Student t distribution cumulative distribution function.

scipy.special.stdtr

### stdtridf

```php
static stdtridf(mixed $p, mixed $t, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Inverse of `stdtr` vs df.

scipy.special.stdtridf

### stdtrit

```php
static stdtrit(mixed $df, mixed $p, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

The `p`-th quantile of the student t distribution.

scipy.special.stdtrit

### struve

```php
static struve(mixed $v, mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Struve function.

scipy.special.struve

### tandg

```php
static tandg(mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Tangent of angle `x` given in degrees.

scipy.special.tandg

### tklmbda

```php
static tklmbda(mixed $x, mixed $lmbda, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Cumulative distribution function of the Tukey lambda distribution.

scipy.special.tklmbda

### voigtProfile

```php
static voigtProfile(mixed $x, mixed $sigma, mixed $gamma, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Voigt profile.

scipy.special.voigt_profile

### wrightBessel

```php
static wrightBessel(mixed $a, mixed $b, mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Wright's generalized Bessel function.

scipy.special.wright_bessel

### wrightomega

```php
static wrightomega(mixed $z, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Wright Omega function.

scipy.special.wrightomega

### xlog1py

```php
static xlog1py(mixed $x, mixed $y, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Compute ``x*log1p(y)`` so that the result is 0 if ``x = 0``.

scipy.special.xlog1py

### xlogy

```php
static xlogy(mixed $x, mixed $y, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Compute ``x*log(y)`` so that the result is 0 if ``x = 0``.

scipy.special.xlogy

### y0

```php
static y0(mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Bessel function of the second kind of order 0.

scipy.special.y0

### y1

```php
static y1(mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Bessel function of the second kind of order 1.

scipy.special.y1

### yn

```php
static yn(mixed $n, mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Bessel function of the second kind of integer order and real argument.

scipy.special.yn

### yv

```php
static yv(mixed $v, mixed $z, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Bessel function of the second kind of real order and complex argument.

scipy.special.yv

### yve

```php
static yve(mixed $v, mixed $z, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Exponentially scaled Bessel function of the second kind of real order.

scipy.special.yve

### zetac

```php
static zetac(mixed $x, ?Tessero\NDArray $out = null): Tessero\NDArray|float
```

Riemann zeta function minus 1.

scipy.special.zetac
