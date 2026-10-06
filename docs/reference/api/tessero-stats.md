# Stats

`Tessero\Stats`

scipy.stats: probability distributions and statistical functions.

Names are the SciPy/NumPy names in camelCase (log_ndtr -> logNdtr); apply() also takes the original names.
The native extension provides the same class as Tessero\Ext\Stats.

## Constants

| Name | Value |
|---|---|
| `FUNCTIONS` | `{"alexandergovern":"alexandergovern","alpha":"alpha","anderson":"anderson","anderson_ksamp":"andersonKsamp","anglit":"anglit","ansari":"ansari","arcsine":"arcsine","argus":"argus","bartlett":"bartlett","bernoulli":"bernoulli","beta":"beta","betabinom":"betabinom","betanbinom":"betanbinom","betaprime":"betaprime","binned_statistic":"binnedStatistic","binned_statistic_2d":"binnedStatistic2d","binned_statistic_dd":"binnedStatisticDd","binom":"binom","binomtest":"binomtest","boltzmann":"boltzmann","boxcox":"boxcox","boxcox_llf":"boxcoxLlf","bradford":"bradford","brunnermunzel":"brunnermunzel","burr":"burr","burr12":"burr12","bws_test":"bwsTest","cauchy":"cauchy","chatterjeexi":"chatterjeexi","chi":"chi","chi2":"chi2","chi2_contingency":"chi2Contingency","chisquare":"chisquare","circmean":"circmean","circstd":"circstd","circvar":"circvar","combine_pvalues":"combinePvalues","cosine":"cosine","cramervonmises":"cramervonmises","cramervonmises_2samp":"cramervonmises2samp","crystalball":"crystalball","cumfreq":"cumfreq","describe":"describe","dgamma":"dgamma","differential_entropy":"differentialEntropy","dlaplace":"dlaplace","dpareto_lognorm":"dparetoLognorm","dweibull":"dweibull","energy_distance":"energyDistance","entropy":"entropy","epps_singleton_2samp":"eppsSingleton2samp","erlang":"erlang","expectile":"expectile","expon":"expon","exponnorm":"exponnorm","exponpow":"exponpow","exponweib":"exponweib","f":"f","f_oneway":"fOneway","false_discovery_control":"falseDiscoveryControl","fatiguelife":"fatiguelife","fisher_exact":"fisherExact","fisk":"fisk","fligner":"fligner","foldcauchy":"foldcauchy","foldnorm":"foldnorm","friedmanchisquare":"friedmanchisquare","gamma":"gamma","gausshyper":"gausshyper","genexpon":"genexpon","genextreme":"genextreme","gengamma":"gengamma","genhalflogistic":"genhalflogistic","genhyperbolic":"genhyperbolic","geninvgauss":"geninvgauss","genlogistic":"genlogistic","gennorm":"gennorm","genpareto":"genpareto","geom":"geom","gibrat":"gibrat","gmean":"gmean","gompertz":"gompertz","gstd":"gstd","gumbel_l":"gumbelL","gumbel_r":"gumbelR","gzscore":"gzscore","halfcauchy":"halfcauchy","halfgennorm":"halfgennorm","halflogistic":"halflogistic","halfnorm":"halfnorm","hmean":"hmean","hypergeom":"hypergeom","hypsecant":"hypsecant","invgamma":"invgamma","invgauss":"invgauss","invweibull":"invweibull","iqr":"iqr","irwinhall":"irwinhall","jarque_bera":"jarqueBera","jf_skew_t":"jfSkewT","johnsonsb":"johnsonsb","johnsonsu":"johnsonsu","kappa3":"kappa3","kappa4":"kappa4","kendalltau":"kendalltau","kruskal":"kruskal","ks_1samp":"ks1samp","ks_2samp":"ks2samp","ksone":"ksone","kstat":"kstat","kstatvar":"kstatvar","kstest":"kstest","kstwo":"kstwo","kstwobign":"kstwobign","kurtosis":"kurtosis","kurtosistest":"kurtosistest","landau":"landau","laplace":"laplace","laplace_asymmetric":"laplaceAsymmetric","levene":"levene","levy":"levy","levy_l":"levyL","levy_stable":"levyStable","linregress":"linregress","lmoment":"lmoment","loggamma":"loggamma","logistic":"logistic","loglaplace":"loglaplace","lognorm":"lognorm","logser":"logser","loguniform":"loguniform","lomax":"lomax","mannwhitneyu":"mannwhitneyu","maxwell":"maxwell","median_abs_deviation":"medianAbsDeviation","median_test":"medianTest","mielke":"mielke","mode":"mode","moment":"moment","mood":"mood","moyal":"moyal","nakagami":"nakagami","nbinom":"nbinom","ncf":"ncf","nchypergeom_fisher":"nchypergeomFisher","nchypergeom_wallenius":"nchypergeomWallenius","nct":"nct","ncx2":"ncx2","nhypergeom":"nhypergeom","norm":"norm","normaltest":"normaltest","norminvgauss":"norminvgauss","obrientransform":"obrientransform","page_trend_test":"pageTrendTest","pareto":"pareto","pearson3":"pearson3","pearsonr":"pearsonr","percentileofscore":"percentileofscore","planck":"planck","pmean":"pmean","pointbiserialr":"pointbiserialr","poisson":"poisson","poisson_means_test":"poissonMeansTest","power_divergence":"powerDivergence","powerlaw":"powerlaw","powerlognorm":"powerlognorm","powernorm":"powernorm","quantile":"quantile","quantile_test":"quantileTest","randint":"randint","rankdata":"rankdata","ranksums":"ranksums","rayleigh":"rayleigh","rdist":"rdist","recipinvgauss":"recipinvgauss","reciprocal":"reciprocal","rel_breitwigner":"relBreitwigner","relfreq":"relfreq","rice":"rice","scoreatpercentile":"scoreatpercentile","sem":"sem","semicircular":"semicircular","shapiro":"shapiro","siegelslopes":"siegelslopes","sigmaclip":"sigmaclip","skellam":"skellam","skew":"skew","skewcauchy":"skewcauchy","skewnorm":"skewnorm","skewtest":"skewtest","somersd":"somersd","spearmanr":"spearmanr","spearmanrho":"spearmanrho","studentized_range":"studentizedRange","t":"t","theilslopes":"theilslopes","tiecorrect":"tiecorrect","tmax":"tmax","tmean":"tmean","tmin":"tmin","trapezoid":"trapezoid","triang":"triang","trim1":"trim1","trim_mean":"trimMean","trimboth":"trimboth","truncexpon":"truncexpon","truncnorm":"truncnorm","truncpareto":"truncpareto","truncweibull_min":"truncweibullMin","tsem":"tsem","tstd":"tstd","ttest_1samp":"ttest1samp","ttest_ind":"ttestInd","ttest_ind_from_stats":"ttestIndFromStats","ttest_rel":"ttestRel","tukeylambda":"tukeylambda","tvar":"tvar","uniform":"uniform","variation":"variation","vonmises":"vonmises","vonmises_line":"vonmisesLine","wald":"wald","wasserstein_distance":"wassersteinDistance","weibull_max":"weibullMax","weibull_min":"weibullMin","weightedtau":"weightedtau","wilcoxon":"wilcoxon","wrapcauchy":"wrapcauchy","yeojohnson":"yeojohnson","yeojohnson_llf":"yeojohnsonLlf","yulesimon":"yulesimon","zipf":"zipf","zipfian":"zipfian","zmap":"zmap","zscore":"zscore"}` |

## Methods

### alexandergovern

```php
static alexandergovern(mixed $sample1, mixed $sample2, mixed $sample3 = null, mixed $sample4 = null, mixed $sample5 = null, mixed $sample6 = null, mixed $sample7 = null, mixed $sample8 = null, mixed $sample9 = null, mixed $sample10 = null, mixed $nanPolicy = 'propagate', mixed $axis = 0, mixed $keepdims = false): mixed
```

Alexander-Govern test (scipy.stats.alexandergovern); the samples are sample1, sample2, ... (up to 10).

scipy.stats.alexandergovern

### alpha

```php
static alpha(mixed $a, mixed $loc = 0, mixed $scale = 1): Tessero\Stats\Distribution
```

An alpha continuous random variable.

scipy.stats.alpha (frozen)

### anderson

```php
static anderson(mixed $x, mixed $dist = 'norm', mixed $method = null): mixed
```

Anderson-Darling test (scipy.stats.anderson) for norm, expon, logistic, gumbel_r, gumbel_l (gumbel, extreme1); method='interpolate' gives a p-value.

scipy.stats.anderson

### andersonKsamp

```php
static andersonKsamp(mixed $samples, mixed $midrank = null, mixed $variant = null, mixed $method = null): mixed
```

k-sample Anderson-Darling test (scipy.stats.anderson_ksamp); samples: one sample per row of a 2-D array.

scipy.stats.anderson_ksamp

### anglit

```php
static anglit(mixed $loc = 0, mixed $scale = 1): Tessero\Stats\Distribution
```

An anglit continuous random variable.

scipy.stats.anglit (frozen)

### ansari

```php
static ansari(mixed $x, mixed $y, mixed $alternative = 'two-sided', mixed $axis = 0, mixed $nanPolicy = 'propagate', mixed $keepdims = false): mixed
```

Ansari-Bradley test for equal scale parameters (scipy.stats.ansari).

scipy.stats.ansari

### apply

```php
static apply(string $name, mixed ...$args): mixed
```

Call a function by its SciPy/NumPy name: Stats::apply('log_ndtr', $x).

### arcsine

```php
static arcsine(mixed $loc = 0, mixed $scale = 1): Tessero\Stats\Distribution
```

An arcsine continuous random variable.

scipy.stats.arcsine (frozen)

### argus

```php
static argus(mixed $chi, mixed $loc = 0, mixed $scale = 1): Tessero\Stats\Distribution
```

Argus distribution

scipy.stats.argus (frozen)

### bartlett

```php
static bartlett(mixed $sample1, mixed $sample2 = null, mixed $sample3 = null, mixed $sample4 = null, mixed $sample5 = null, mixed $sample6 = null, mixed $sample7 = null, mixed $sample8 = null, mixed $sample9 = null, mixed $sample10 = null, mixed $sample11 = null, mixed $sample12 = null, mixed $axis = 0, mixed $nanPolicy = 'propagate', mixed $keepdims = false): mixed
```

Bartlett's test for equal variances (scipy.stats.bartlett).

scipy.stats.bartlett

### bernoulli

```php
static bernoulli(mixed $p, mixed $loc = 0): Tessero\Stats\Distribution
```

A Bernoulli discrete random variable.

scipy.stats.bernoulli (frozen)

### beta

```php
static beta(mixed $a, mixed $b, mixed $loc = 0, mixed $scale = 1): Tessero\Stats\Distribution
```

A beta continuous random variable.

scipy.stats.beta (frozen)

### betabinom

```php
static betabinom(mixed $n, mixed $a, mixed $b, mixed $loc = 0): Tessero\Stats\Distribution
```

A beta-binomial discrete random variable.

scipy.stats.betabinom (frozen)

### betanbinom

```php
static betanbinom(mixed $n, mixed $a, mixed $b, mixed $loc = 0): Tessero\Stats\Distribution
```

A beta-negative-binomial discrete random variable.

scipy.stats.betanbinom (frozen)

### betaprime

```php
static betaprime(mixed $a, mixed $b, mixed $loc = 0, mixed $scale = 1): Tessero\Stats\Distribution
```

A beta prime continuous random variable.

scipy.stats.betaprime (frozen)

### binnedStatistic

```php
static binnedStatistic(mixed $x, mixed $values, mixed $statistic = 'mean', mixed $bins = 10, mixed $range = null): mixed
```

A statistic of the values in each bin of x (scipy.stats.binned_statistic; named statistics only).

scipy.stats.binned_statistic

### binnedStatistic2d

```php
static binnedStatistic2d(mixed $x, mixed $y, mixed $values, mixed $statistic = 'mean', mixed $bins = 10, mixed $range = null, mixed $expandBinnumbers = false): mixed
```

A two-dimensional binned statistic (scipy.stats.binned_statistic_2d; named statistics only).

scipy.stats.binned_statistic_2d

### binnedStatisticDd

```php
static binnedStatisticDd(mixed $sample, mixed $values, mixed $statistic = 'mean', mixed $bins = 10, mixed $range = null, mixed $expandBinnumbers = false, mixed $binnedStatisticResult = null): mixed
```

A multidimensional binned statistic (scipy.stats.binned_statistic_dd; named statistics only; bin_edges as one (D, k) array).

scipy.stats.binned_statistic_dd

### binom

```php
static binom(mixed $n, mixed $p, mixed $loc = 0): Tessero\Stats\Distribution
```

A binomial discrete random variable.

scipy.stats.binom (frozen)

### binomtest

```php
static binomtest(mixed $k, mixed $n, mixed $p = 0.5, mixed $alternative = 'two-sided'): mixed
```

Test that the probability of success is p (scipy.stats.binomtest).

scipy.stats.binomtest

### boltzmann

```php
static boltzmann(mixed $lambda_, mixed $N, mixed $loc = 0): Tessero\Stats\Distribution
```

A Boltzmann (Truncated Discrete Exponential) random variable.

scipy.stats.boltzmann (frozen)

### boxcox

```php
static boxcox(mixed $x, mixed $lmbda = null, mixed $alpha = null, mixed $optimizer = null): mixed
```

The Box-Cox power transformation with a given lambda (scipy.stats.boxcox; lmbda required).

scipy.stats.boxcox

### boxcoxLlf

```php
static boxcoxLlf(mixed $lmb, mixed $data, mixed $axis = 0, mixed $keepdims = false, mixed $nanPolicy = 'propagate'): mixed
```

Box-Cox log-likelihood (scipy.stats.boxcox_llf).

scipy.stats.boxcox_llf

### bradford

```php
static bradford(mixed $c, mixed $loc = 0, mixed $scale = 1): Tessero\Stats\Distribution
```

A Bradford continuous random variable.

scipy.stats.bradford (frozen)

### brunnermunzel

```php
static brunnermunzel(mixed $x, mixed $y, mixed $alternative = 'two-sided', mixed $distribution = 't', mixed $nanPolicy = 'propagate', mixed $axis = 0, mixed $keepdims = false): mixed
```

Brunner-Munzel test on samples x and y (scipy.stats.brunnermunzel).

scipy.stats.brunnermunzel

### burr

```php
static burr(mixed $c, mixed $d, mixed $loc = 0, mixed $scale = 1): Tessero\Stats\Distribution
```

A Burr (Type III) continuous random variable.

scipy.stats.burr (frozen)

### burr12

```php
static burr12(mixed $c, mixed $d, mixed $loc = 0, mixed $scale = 1): Tessero\Stats\Distribution
```

A Burr (Type XII) continuous random variable.

scipy.stats.burr12 (frozen)

### bwsTest

```php
static bwsTest(mixed $x, mixed $y, mixed $alternative = 'two-sided', mixed $method = null): mixed
```

Baumgartner-Weiss-Schindler test (scipy.stats.bws_test) with the exact permutation distribution (at most 9999 partitions).

scipy.stats.bws_test

### cauchy

```php
static cauchy(mixed $loc = 0, mixed $scale = 1): Tessero\Stats\Distribution
```

A Cauchy continuous random variable.

scipy.stats.cauchy (frozen)

### chatterjeexi

```php
static chatterjeexi(mixed $x, mixed $y, array|int|null $axis = 0, int|float|bool|null $yContinuous = false, string $nanPolicy = 'propagate', bool $keepdims = false): array
```

Chatterjee's xi correlation with the asymptotic p-value (scipy.stats.chatterjeexi; method='asymptotic').

scipy.stats.chatterjeexi

### chi

```php
static chi(mixed $df, mixed $loc = 0, mixed $scale = 1): Tessero\Stats\Distribution
```

A chi continuous random variable.

scipy.stats.chi (frozen)

### chi2

```php
static chi2(mixed $df, mixed $loc = 0, mixed $scale = 1): Tessero\Stats\Distribution
```

A chi-squared continuous random variable.

scipy.stats.chi2 (frozen)

### chi2Contingency

```php
static chi2Contingency(mixed $observed, mixed $correction = true, mixed $lambda_ = null, mixed $method = null): mixed
```

Chi-square test of independence of variables in a contingency table (scipy.stats.chi2_contingency).

scipy.stats.chi2_contingency

### chisquare

```php
static chisquare(mixed $fObs, mixed $fExp = null, mixed $ddof = 0, mixed $axis = 0, mixed $sumCheck = true, mixed $nanPolicy = 'propagate', mixed $keepdims = false): mixed
```

Pearson's chi-squared test (scipy.stats.chisquare).

scipy.stats.chisquare

### circmean

```php
static circmean(mixed $samples, mixed $high = 6.283185307179586, mixed $low = 0, array|int|null $axis = null, string $nanPolicy = 'propagate', bool $keepdims = false): Tessero\NDArray|int|float
```

Circular mean of samples in [low, high] (scipy.stats.circmean).

scipy.stats.circmean

### circstd

```php
static circstd(mixed $samples, mixed $high = 6.283185307179586, mixed $low = 0, array|int|null $axis = null, string $nanPolicy = 'propagate', int|float|bool|null $normalize = false, bool $keepdims = false): Tessero\NDArray|int|float
```

Circular standard deviation, sqrt(-2 log R) (scipy.stats.circstd).

scipy.stats.circstd

### circvar

```php
static circvar(mixed $samples, mixed $high = 6.283185307179586, mixed $low = 0, array|int|null $axis = null, string $nanPolicy = 'propagate', bool $keepdims = false): Tessero\NDArray|int|float
```

Circular variance, 1 - R (scipy.stats.circvar).

scipy.stats.circvar

### combinePvalues

```php
static combinePvalues(mixed $pvalues, mixed $method = 'fisher', mixed $weights = null, mixed $axis = 0, mixed $nanPolicy = 'propagate', mixed $keepdims = false): mixed
```

Combine p-values of independent tests (scipy.stats.combine_pvalues).

scipy.stats.combine_pvalues

### cosine

```php
static cosine(mixed $loc = 0, mixed $scale = 1): Tessero\Stats\Distribution
```

A cosine continuous random variable.

scipy.stats.cosine (frozen)

### cramervonmises

```php
static cramervonmises(mixed $rvs, mixed $cdf, mixed $args = null, mixed $axis = 0, mixed $nanPolicy = 'propagate', mixed $keepdims = false): mixed
```

One-sample Cramer-von Mises test (scipy.stats.cramervonmises); cdf is a distribution name (norm, expon, uniform, gamma, beta, t, chi2, f), args its parameters.

scipy.stats.cramervonmises

### cramervonmises2samp

```php
static cramervonmises2samp(mixed $x, mixed $y, mixed $method = 'auto', mixed $axis = 0, mixed $nanPolicy = 'propagate', mixed $keepdims = false): mixed
```

Two-sample Cramer-von Mises test (scipy.stats.cramervonmises_2samp).

scipy.stats.cramervonmises_2samp

### crystalball

```php
static crystalball(mixed $beta, mixed $m, mixed $loc = 0, mixed $scale = 1): Tessero\Stats\Distribution
```

Crystalball distribution

scipy.stats.crystalball (frozen)

### cumfreq

```php
static cumfreq(mixed $a, mixed $numbins = 10, mixed $defaultreallimits = null, mixed $weights = null): mixed
```

Cumulative frequency histogram (scipy.stats.cumfreq).

scipy.stats.cumfreq

### describe

```php
static describe(mixed $a, mixed $axis = 0, mixed $ddof = 1, mixed $bias = true, mixed $nanPolicy = 'propagate'): mixed
```

Descriptive statistics; minmax is returned as one array stacking (min, max) (scipy.stats.describe).

scipy.stats.describe

### dgamma

```php
static dgamma(mixed $a, mixed $loc = 0, mixed $scale = 1): Tessero\Stats\Distribution
```

A double gamma continuous random variable.

scipy.stats.dgamma (frozen)

### differentialEntropy

```php
static differentialEntropy(mixed $values, array|int|null $axis = 0, int|float|bool|null $windowLength = null, int|float|bool|null $base = null, string $method = 'auto', string $nanPolicy = 'propagate', bool $keepdims = false): Tessero\NDArray|int|float
```

Differential entropy estimated from a sample (scipy.stats.differential_entropy).

scipy.stats.differential_entropy

### dlaplace

```php
static dlaplace(mixed $a, mixed $loc = 0): Tessero\Stats\Distribution
```

A  Laplacian discrete random variable.

scipy.stats.dlaplace (frozen)

### dparetoLognorm

```php
static dparetoLognorm(mixed $u, mixed $s, mixed $a, mixed $b, mixed $loc = 0, mixed $scale = 1): Tessero\Stats\Distribution
```

A double Pareto lognormal continuous random variable.

scipy.stats.dpareto_lognorm (frozen)

### dweibull

```php
static dweibull(mixed $c, mixed $loc = 0, mixed $scale = 1): Tessero\Stats\Distribution
```

A double Weibull continuous random variable.

scipy.stats.dweibull (frozen)

### energyDistance

```php
static energyDistance(mixed $uValues, mixed $vValues, mixed $uWeights = null, mixed $vWeights = null): mixed
```

Energy distance between two 1-D distributions (scipy.stats.energy_distance).

scipy.stats.energy_distance

### entropy

```php
static entropy(mixed $pk, mixed $qk = null, array|int|null $axis = 0, int|float|bool|null $base = null, string $nanPolicy = 'propagate', bool $keepdims = false): Tessero\NDArray|int|float
```

Shannon entropy of pk, or the relative entropy D(pk||qk) (scipy.stats.entropy).

scipy.stats.entropy

### eppsSingleton2samp

```php
static eppsSingleton2samp(mixed $x, mixed $y, mixed $t = null, mixed $axis = 0, mixed $nanPolicy = 'propagate', mixed $keepdims = false): mixed
```

Epps-Singleton two-sample test (scipy.stats.epps_singleton_2samp); t defaults to (0.4, 0.8).

scipy.stats.epps_singleton_2samp

### erlang

```php
static erlang(mixed $a, mixed $loc = 0, mixed $scale = 1): Tessero\Stats\Distribution
```

An Erlang continuous random variable.

scipy.stats.erlang (frozen)

### expectile

```php
static expectile(mixed $a, mixed $alpha = 0.5, mixed $weights = null): mixed
```

Expectile of the data at level alpha (scipy.stats.expectile).

scipy.stats.expectile

### expon

```php
static expon(mixed $loc = 0, mixed $scale = 1): Tessero\Stats\Distribution
```

An exponential continuous random variable.

scipy.stats.expon (frozen)

### exponnorm

```php
static exponnorm(mixed $K, mixed $loc = 0, mixed $scale = 1): Tessero\Stats\Distribution
```

An exponentially modified Normal continuous random variable.

scipy.stats.exponnorm (frozen)

### exponpow

```php
static exponpow(mixed $b, mixed $loc = 0, mixed $scale = 1): Tessero\Stats\Distribution
```

An exponential power continuous random variable.

scipy.stats.exponpow (frozen)

### exponweib

```php
static exponweib(mixed $a, mixed $c, mixed $loc = 0, mixed $scale = 1): Tessero\Stats\Distribution
```

An exponentiated Weibull continuous random variable.

scipy.stats.exponweib (frozen)

### f

```php
static f(mixed $dfn, mixed $dfd, mixed $loc = 0, mixed $scale = 1): Tessero\Stats\Distribution
```

An F continuous random variable.

scipy.stats.f (frozen)

### fOneway

```php
static fOneway(mixed $sample1, mixed $sample2 = null, mixed $sample3 = null, mixed $sample4 = null, mixed $sample5 = null, mixed $sample6 = null, mixed $sample7 = null, mixed $sample8 = null, mixed $sample9 = null, mixed $sample10 = null, mixed $sample11 = null, mixed $sample12 = null, mixed $axis = 0, mixed $equalVar = true, mixed $nanPolicy = 'propagate', mixed $keepdims = false): mixed
```

One-way ANOVA; Welch's ANOVA with equal_var=False (scipy.stats.f_oneway).

scipy.stats.f_oneway

### falseDiscoveryControl

```php
static falseDiscoveryControl(mixed $ps, ?int $axis = 0, string $method = 'bh'): Tessero\NDArray|int|float
```

Benjamini-Hochberg / Benjamini-Yekutieli adjusted p-values (scipy.stats.false_discovery_control).

scipy.stats.false_discovery_control

### fatiguelife

```php
static fatiguelife(mixed $c, mixed $loc = 0, mixed $scale = 1): Tessero\Stats\Distribution
```

A fatigue-life (Birnbaum-Saunders) continuous random variable.

scipy.stats.fatiguelife (frozen)

### fisherExact

```php
static fisherExact(mixed $table, mixed $alternative = null, mixed $method = null): mixed
```

Fisher exact test on a 2x2 contingency table (scipy.stats.fisher_exact).

scipy.stats.fisher_exact

### fisk

```php
static fisk(mixed $c, mixed $loc = 0, mixed $scale = 1): Tessero\Stats\Distribution
```

A Fisk continuous random variable.

scipy.stats.fisk (frozen)

### fligner

```php
static fligner(mixed $sample1, mixed $sample2 = null, mixed $sample3 = null, mixed $sample4 = null, mixed $sample5 = null, mixed $sample6 = null, mixed $sample7 = null, mixed $sample8 = null, mixed $sample9 = null, mixed $sample10 = null, mixed $sample11 = null, mixed $sample12 = null, mixed $center = 'median', mixed $proportiontocut = 0.05, mixed $axis = 0, mixed $nanPolicy = 'propagate', mixed $keepdims = false): mixed
```

Fligner-Killeen test for equality of variance (scipy.stats.fligner).

scipy.stats.fligner

### foldcauchy

```php
static foldcauchy(mixed $c, mixed $loc = 0, mixed $scale = 1): Tessero\Stats\Distribution
```

A folded Cauchy continuous random variable.

scipy.stats.foldcauchy (frozen)

### foldnorm

```php
static foldnorm(mixed $c, mixed $loc = 0, mixed $scale = 1): Tessero\Stats\Distribution
```

A folded normal continuous random variable.

scipy.stats.foldnorm (frozen)

### friedmanchisquare

```php
static friedmanchisquare(mixed $sample1, mixed $sample2 = null, mixed $sample3 = null, mixed $sample4 = null, mixed $sample5 = null, mixed $sample6 = null, mixed $sample7 = null, mixed $sample8 = null, mixed $sample9 = null, mixed $sample10 = null, mixed $sample11 = null, mixed $sample12 = null, mixed $axis = 0, mixed $nanPolicy = 'propagate', mixed $keepdims = false): mixed
```

Friedman test for repeated samples (scipy.stats.friedmanchisquare).

scipy.stats.friedmanchisquare

### gamma

```php
static gamma(mixed $a, mixed $loc = 0, mixed $scale = 1): Tessero\Stats\Distribution
```

A gamma continuous random variable.

scipy.stats.gamma (frozen)

### gausshyper

```php
static gausshyper(mixed $a, mixed $b, mixed $c, mixed $z, mixed $loc = 0, mixed $scale = 1): Tessero\Stats\Distribution
```

A Gauss hypergeometric continuous random variable.

scipy.stats.gausshyper (frozen)

### genexpon

```php
static genexpon(mixed $a, mixed $b, mixed $c, mixed $loc = 0, mixed $scale = 1): Tessero\Stats\Distribution
```

A generalized exponential continuous random variable.

scipy.stats.genexpon (frozen)

### genextreme

```php
static genextreme(mixed $c, mixed $loc = 0, mixed $scale = 1): Tessero\Stats\Distribution
```

A generalized extreme value continuous random variable.

scipy.stats.genextreme (frozen)

### gengamma

```php
static gengamma(mixed $a, mixed $c, mixed $loc = 0, mixed $scale = 1): Tessero\Stats\Distribution
```

A generalized gamma continuous random variable.

scipy.stats.gengamma (frozen)

### genhalflogistic

```php
static genhalflogistic(mixed $c, mixed $loc = 0, mixed $scale = 1): Tessero\Stats\Distribution
```

A generalized half-logistic continuous random variable.

scipy.stats.genhalflogistic (frozen)

### genhyperbolic

```php
static genhyperbolic(mixed $p, mixed $a, mixed $b, mixed $loc = 0, mixed $scale = 1): Tessero\Stats\Distribution
```

A generalized hyperbolic continuous random variable.

scipy.stats.genhyperbolic (frozen)

### geninvgauss

```php
static geninvgauss(mixed $p, mixed $b, mixed $loc = 0, mixed $scale = 1): Tessero\Stats\Distribution
```

A Generalized Inverse Gaussian continuous random variable.

scipy.stats.geninvgauss (frozen)

### genlogistic

```php
static genlogistic(mixed $c, mixed $loc = 0, mixed $scale = 1): Tessero\Stats\Distribution
```

A generalized logistic continuous random variable.

scipy.stats.genlogistic (frozen)

### gennorm

```php
static gennorm(mixed $beta, mixed $loc = 0, mixed $scale = 1): Tessero\Stats\Distribution
```

A generalized normal continuous random variable.

scipy.stats.gennorm (frozen)

### genpareto

```php
static genpareto(mixed $c, mixed $loc = 0, mixed $scale = 1): Tessero\Stats\Distribution
```

A generalized Pareto continuous random variable.

scipy.stats.genpareto (frozen)

### geom

```php
static geom(mixed $p, mixed $loc = 0): Tessero\Stats\Distribution
```

A geometric discrete random variable.

scipy.stats.geom (frozen)

### gibrat

```php
static gibrat(mixed $loc = 0, mixed $scale = 1): Tessero\Stats\Distribution
```

A Gibrat continuous random variable.

scipy.stats.gibrat (frozen)

### gmean

```php
static gmean(mixed $a, mixed $weights = null, array|int|null $axis = 0, string $nanPolicy = 'propagate', bool $keepdims = false): Tessero\NDArray|int|float
```

Weighted geometric mean along an axis (scipy.stats.gmean; dtype is not supported).

scipy.stats.gmean

### gompertz

```php
static gompertz(mixed $c, mixed $loc = 0, mixed $scale = 1): Tessero\Stats\Distribution
```

A Gompertz (or truncated Gumbel) continuous random variable.

scipy.stats.gompertz (frozen)

### gstd

```php
static gstd(mixed $a, array|int|null $axis = 0, int|float|bool|null $ddof = 1, string $nanPolicy = 'propagate', bool $keepdims = false): Tessero\NDArray|int|float
```

Geometric standard deviation (scipy.stats.gstd).

scipy.stats.gstd

### gumbelL

```php
static gumbelL(mixed $loc = 0, mixed $scale = 1): Tessero\Stats\Distribution
```

A left-skewed Gumbel continuous random variable.

scipy.stats.gumbel_l (frozen)

### gumbelR

```php
static gumbelR(mixed $loc = 0, mixed $scale = 1): Tessero\Stats\Distribution
```

A right-skewed Gumbel continuous random variable.

scipy.stats.gumbel_r (frozen)

### gzscore

```php
static gzscore(mixed $a, ?int $axis = 0, int|float|bool|null $ddof = 0, string $nanPolicy = 'propagate'): Tessero\NDArray|int|float
```

Geometric z-scores: z-scores of log(a) (scipy.stats.gzscore).

scipy.stats.gzscore

### halfcauchy

```php
static halfcauchy(mixed $loc = 0, mixed $scale = 1): Tessero\Stats\Distribution
```

A Half-Cauchy continuous random variable.

scipy.stats.halfcauchy (frozen)

### halfgennorm

```php
static halfgennorm(mixed $beta, mixed $loc = 0, mixed $scale = 1): Tessero\Stats\Distribution
```

The upper half of a generalized normal continuous random variable.

scipy.stats.halfgennorm (frozen)

### halflogistic

```php
static halflogistic(mixed $loc = 0, mixed $scale = 1): Tessero\Stats\Distribution
```

A half-logistic continuous random variable.

scipy.stats.halflogistic (frozen)

### halfnorm

```php
static halfnorm(mixed $loc = 0, mixed $scale = 1): Tessero\Stats\Distribution
```

A half-normal continuous random variable.

scipy.stats.halfnorm (frozen)

### hmean

```php
static hmean(mixed $a, mixed $weights = null, array|int|null $axis = 0, string $nanPolicy = 'propagate', bool $keepdims = false): Tessero\NDArray|int|float
```

Weighted harmonic mean along an axis (scipy.stats.hmean; dtype is not supported).

scipy.stats.hmean

### hypergeom

```php
static hypergeom(mixed $M, mixed $n, mixed $N, mixed $loc = 0): Tessero\Stats\Distribution
```

A hypergeometric discrete random variable.

scipy.stats.hypergeom (frozen)

### hypsecant

```php
static hypsecant(mixed $loc = 0, mixed $scale = 1): Tessero\Stats\Distribution
```

A hyperbolic secant continuous random variable.

scipy.stats.hypsecant (frozen)

### invgamma

```php
static invgamma(mixed $a, mixed $loc = 0, mixed $scale = 1): Tessero\Stats\Distribution
```

An inverted gamma continuous random variable.

scipy.stats.invgamma (frozen)

### invgauss

```php
static invgauss(mixed $mu, mixed $loc = 0, mixed $scale = 1): Tessero\Stats\Distribution
```

An inverse Gaussian continuous random variable.

scipy.stats.invgauss (frozen)

### invweibull

```php
static invweibull(mixed $c, mixed $loc = 0, mixed $scale = 1): Tessero\Stats\Distribution
```

An inverted Weibull continuous random variable.

scipy.stats.invweibull (frozen)

### iqr

```php
static iqr(mixed $x, mixed $axis = null, mixed $rng = null, mixed $scale = 1.0, mixed $nanPolicy = 'propagate', mixed $interpolation = 'linear', mixed $keepdims = false): mixed
```

Interquartile range; rng=None means (25, 75) (scipy.stats.iqr).

scipy.stats.iqr

### irwinhall

```php
static irwinhall(mixed $n, mixed $loc = 0, mixed $scale = 1): Tessero\Stats\Distribution
```

An Irwin-Hall (Uniform Sum) continuous random variable.

scipy.stats.irwinhall (frozen)

### jarqueBera

```php
static jarqueBera(mixed $x, mixed $axis = null, mixed $nanPolicy = 'propagate', mixed $keepdims = false): mixed
```

Jarque-Bera goodness of fit test on sample data (scipy.stats.jarque_bera).

scipy.stats.jarque_bera

### jfSkewT

```php
static jfSkewT(mixed $a, mixed $b, mixed $loc = 0, mixed $scale = 1): Tessero\Stats\Distribution
```

Jones and Faddy skew-t distribution.

scipy.stats.jf_skew_t (frozen)

### johnsonsb

```php
static johnsonsb(mixed $a, mixed $b, mixed $loc = 0, mixed $scale = 1): Tessero\Stats\Distribution
```

A Johnson SB continuous random variable.

scipy.stats.johnsonsb (frozen)

### johnsonsu

```php
static johnsonsu(mixed $a, mixed $b, mixed $loc = 0, mixed $scale = 1): Tessero\Stats\Distribution
```

A Johnson SU continuous random variable.

scipy.stats.johnsonsu (frozen)

### kappa3

```php
static kappa3(mixed $a, mixed $loc = 0, mixed $scale = 1): Tessero\Stats\Distribution
```

Kappa 3 parameter distribution.

scipy.stats.kappa3 (frozen)

### kappa4

```php
static kappa4(mixed $h, mixed $k, mixed $loc = 0, mixed $scale = 1): Tessero\Stats\Distribution
```

Kappa 4 parameter distribution.

scipy.stats.kappa4 (frozen)

### kendalltau

```php
static kendalltau(mixed $x, mixed $y, array|int|null $axis = null, string $method = 'auto', string $variant = 'b', string $alternative = 'two-sided', string $nanPolicy = 'propagate', bool $keepdims = false): array
```

Kendall's tau-b or tau-c with an exact or asymptotic p-value (scipy.stats.kendalltau).

scipy.stats.kendalltau

### kruskal

```php
static kruskal(mixed $sample1, mixed $sample2 = null, mixed $sample3 = null, mixed $sample4 = null, mixed $sample5 = null, mixed $sample6 = null, mixed $sample7 = null, mixed $sample8 = null, mixed $sample9 = null, mixed $sample10 = null, mixed $sample11 = null, mixed $sample12 = null, mixed $nanPolicy = 'propagate', mixed $axis = 0, mixed $keepdims = false): mixed
```

Kruskal-Wallis H-test for independent samples (scipy.stats.kruskal).

scipy.stats.kruskal

### ks1samp

```php
static ks1samp(mixed $x, mixed $cdf, mixed $args = null, mixed $alternative = 'two-sided', mixed $method = 'auto', mixed $axis = 0, mixed $nanPolicy = 'propagate', mixed $keepdims = false): mixed
```

One-sample Kolmogorov-Smirnov test (scipy.stats.ks_1samp); cdf is a distribution name, args its parameters.

scipy.stats.ks_1samp

### ks2samp

```php
static ks2samp(mixed $data1, mixed $data2, mixed $alternative = 'two-sided', mixed $method = 'auto', mixed $axis = 0, mixed $nanPolicy = 'propagate', mixed $keepdims = false): mixed
```

Two-sample Kolmogorov-Smirnov test (scipy.stats.ks_2samp).

scipy.stats.ks_2samp

### ksone

```php
static ksone(mixed $n, mixed $loc = 0, mixed $scale = 1): Tessero\Stats\Distribution
```

Kolmogorov-Smirnov one-sided test statistic distribution.

scipy.stats.ksone (frozen)

### kstat

```php
static kstat(mixed $data, mixed $n = 2, array|int|null $axis = null, string $nanPolicy = 'propagate', bool $keepdims = false): Tessero\NDArray|int|float
```

n-th k-statistic, the unbiased estimator of the n-th cumulant, 1 <= n <= 4 (scipy.stats.kstat).

scipy.stats.kstat

### kstatvar

```php
static kstatvar(mixed $data, mixed $n = 2, array|int|null $axis = null, string $nanPolicy = 'propagate', bool $keepdims = false): Tessero\NDArray|int|float
```

Unbiased estimator of the variance of the k-statistic, n = 1 or 2 (scipy.stats.kstatvar).

scipy.stats.kstatvar

### kstest

```php
static kstest(mixed $rvs, mixed $cdf, mixed $args = null, mixed $N = 20, mixed $alternative = 'two-sided', mixed $method = 'auto', mixed $axis = 0, mixed $nanPolicy = 'propagate', mixed $keepdims = false): mixed
```

Kolmogorov-Smirnov test (scipy.stats.kstest): cdf is a distribution name (one-sample) or a second sample.

scipy.stats.kstest

### kstwo

```php
static kstwo(mixed $n, mixed $loc = 0, mixed $scale = 1): Tessero\Stats\Distribution
```

Kolmogorov-Smirnov two-sided test statistic distribution.

scipy.stats.kstwo (frozen)

### kstwobign

```php
static kstwobign(mixed $loc = 0, mixed $scale = 1): Tessero\Stats\Distribution
```

Limiting distribution of scaled Kolmogorov-Smirnov two-sided test statistic.

scipy.stats.kstwobign (frozen)

### kurtosis

```php
static kurtosis(mixed $a, array|int|null $axis = 0, int|float|bool|null $fisher = true, int|float|bool|null $bias = true, string $nanPolicy = 'propagate', bool $keepdims = false): Tessero\NDArray|int|float
```

Kurtosis (Fisher or Pearson) of a data set (scipy.stats.kurtosis).

scipy.stats.kurtosis

### kurtosistest

```php
static kurtosistest(mixed $a, mixed $axis = 0, mixed $nanPolicy = 'propagate', mixed $alternative = 'two-sided', mixed $keepdims = false): mixed
```

Test whether a dataset has normal kurtosis (scipy.stats.kurtosistest).

scipy.stats.kurtosistest

### landau

```php
static landau(mixed $loc = 0, mixed $scale = 1): Tessero\Stats\Distribution
```

A Landau continuous random variable.

scipy.stats.landau (frozen)

### laplace

```php
static laplace(mixed $loc = 0, mixed $scale = 1): Tessero\Stats\Distribution
```

A Laplace continuous random variable.

scipy.stats.laplace (frozen)

### laplaceAsymmetric

```php
static laplaceAsymmetric(mixed $kappa, mixed $loc = 0, mixed $scale = 1): Tessero\Stats\Distribution
```

An asymmetric Laplace continuous random variable.

scipy.stats.laplace_asymmetric (frozen)

### levene

```php
static levene(mixed $sample1, mixed $sample2 = null, mixed $sample3 = null, mixed $sample4 = null, mixed $sample5 = null, mixed $sample6 = null, mixed $sample7 = null, mixed $sample8 = null, mixed $sample9 = null, mixed $sample10 = null, mixed $sample11 = null, mixed $sample12 = null, mixed $center = 'median', mixed $proportiontocut = 0.05, mixed $axis = 0, mixed $nanPolicy = 'propagate', mixed $keepdims = false): mixed
```

Levene test for equal variances (scipy.stats.levene).

scipy.stats.levene

### levy

```php
static levy(mixed $loc = 0, mixed $scale = 1): Tessero\Stats\Distribution
```

A Levy continuous random variable.

scipy.stats.levy (frozen)

### levyL

```php
static levyL(mixed $loc = 0, mixed $scale = 1): Tessero\Stats\Distribution
```

A left-skewed Levy continuous random variable.

scipy.stats.levy_l (frozen)

### levyStable

```php
static levyStable(mixed $alpha, mixed $beta, mixed $loc = 0, mixed $scale = 1): Tessero\Stats\Distribution
```

A Levy-stable continuous random variable.

scipy.stats.levy_stable (frozen)

### linregress

```php
static linregress(mixed $x, mixed $y, mixed $alternative = 'two-sided', mixed $axis = 0, mixed $nanPolicy = 'propagate', mixed $keepdims = false): mixed
```

Least-squares regression line of two sets of measurements (scipy.stats.linregress).

scipy.stats.linregress

### lmoment

```php
static lmoment(mixed $sample, mixed $order = null, mixed $axis = 0, mixed $sorted = false, mixed $standardize = true, mixed $nanPolicy = 'propagate', mixed $keepdims = false): mixed
```

Sample L-moments, standardized as L-moment ratios by default (scipy.stats.lmoment).

scipy.stats.lmoment

### loggamma

```php
static loggamma(mixed $c, mixed $loc = 0, mixed $scale = 1): Tessero\Stats\Distribution
```

A log gamma continuous random variable.

scipy.stats.loggamma (frozen)

### logistic

```php
static logistic(mixed $loc = 0, mixed $scale = 1): Tessero\Stats\Distribution
```

A logistic (or Sech-squared) continuous random variable.

scipy.stats.logistic (frozen)

### loglaplace

```php
static loglaplace(mixed $c, mixed $loc = 0, mixed $scale = 1): Tessero\Stats\Distribution
```

A log-Laplace continuous random variable.

scipy.stats.loglaplace (frozen)

### lognorm

```php
static lognorm(mixed $s, mixed $loc = 0, mixed $scale = 1): Tessero\Stats\Distribution
```

A lognormal continuous random variable.

scipy.stats.lognorm (frozen)

### logser

```php
static logser(mixed $p, mixed $loc = 0): Tessero\Stats\Distribution
```

A Logarithmic (Log-Series, Series) discrete random variable.

scipy.stats.logser (frozen)

### loguniform

```php
static loguniform(mixed $a, mixed $b, mixed $loc = 0, mixed $scale = 1): Tessero\Stats\Distribution
```

A loguniform or reciprocal continuous random variable.

scipy.stats.loguniform (frozen)

### lomax

```php
static lomax(mixed $c, mixed $loc = 0, mixed $scale = 1): Tessero\Stats\Distribution
```

A Lomax (Pareto of the second kind) continuous random variable.

scipy.stats.lomax (frozen)

### mannwhitneyu

```php
static mannwhitneyu(mixed $x, mixed $y, mixed $useContinuity = true, mixed $alternative = 'two-sided', mixed $axis = 0, mixed $method = 'auto', mixed $nanPolicy = 'propagate', mixed $keepdims = false): mixed
```

Mann-Whitney U rank test on two independent samples (scipy.stats.mannwhitneyu).

scipy.stats.mannwhitneyu

### maxwell

```php
static maxwell(mixed $loc = 0, mixed $scale = 1): Tessero\Stats\Distribution
```

A Maxwell continuous random variable.

scipy.stats.maxwell (frozen)

### medianAbsDeviation

```php
static medianAbsDeviation(mixed $x, mixed $axis = 0, mixed $center = null, mixed $scale = 1.0, mixed $nanPolicy = 'propagate', mixed $keepdims = false): mixed
```

Median absolute deviation; center must be None (the median) (scipy.stats.median_abs_deviation).

scipy.stats.median_abs_deviation

### medianTest

```php
static medianTest(mixed $sample1, mixed $sample2 = null, mixed $sample3 = null, mixed $sample4 = null, mixed $sample5 = null, mixed $sample6 = null, mixed $sample7 = null, mixed $sample8 = null, mixed $sample9 = null, mixed $sample10 = null, mixed $sample11 = null, mixed $sample12 = null, mixed $ties = 'below', mixed $correction = true, mixed $lambda_ = 1, mixed $nanPolicy = 'propagate'): mixed
```

Mood's median test (scipy.stats.median_test).

scipy.stats.median_test

### mielke

```php
static mielke(mixed $k, mixed $s, mixed $loc = 0, mixed $scale = 1): Tessero\Stats\Distribution
```

A Mielke Beta-Kappa / Dagum continuous random variable.

scipy.stats.mielke (frozen)

### mode

```php
static mode(mixed $a, mixed $axis = 0, mixed $nanPolicy = 'propagate', mixed $keepdims = false): mixed
```

Modal (most common) value and its count; ties give the smallest value (scipy.stats.mode).

scipy.stats.mode

### moment

```php
static moment(mixed $a, mixed $order = 1, array|int|null $axis = 0, string $nanPolicy = 'propagate', int|float|bool|null $center = null, bool $keepdims = false): Tessero\NDArray|int|float
```

n-th central moment (or about center) of a sample; an array of orders gives one result each (scipy.stats.moment).

scipy.stats.moment

### mood

```php
static mood(mixed $x, mixed $y, mixed $axis = 0, mixed $alternative = 'two-sided', mixed $nanPolicy = 'propagate', mixed $keepdims = false): mixed
```

Mood's test for equal scale parameters (scipy.stats.mood).

scipy.stats.mood

### moyal

```php
static moyal(mixed $loc = 0, mixed $scale = 1): Tessero\Stats\Distribution
```

A Moyal continuous random variable.

scipy.stats.moyal (frozen)

### nakagami

```php
static nakagami(mixed $nu, mixed $loc = 0, mixed $scale = 1): Tessero\Stats\Distribution
```

A Nakagami continuous random variable.

scipy.stats.nakagami (frozen)

### nbinom

```php
static nbinom(mixed $n, mixed $p, mixed $loc = 0): Tessero\Stats\Distribution
```

A negative binomial discrete random variable.

scipy.stats.nbinom (frozen)

### ncf

```php
static ncf(mixed $dfn, mixed $dfd, mixed $nc, mixed $loc = 0, mixed $scale = 1): Tessero\Stats\Distribution
```

A non-central F distribution continuous random variable.

scipy.stats.ncf (frozen)

### nchypergeomFisher

```php
static nchypergeomFisher(mixed $M, mixed $n, mixed $N, mixed $odds, mixed $loc = 0): Tessero\Stats\Distribution
```

A Fisher's noncentral hypergeometric discrete random variable.

scipy.stats.nchypergeom_fisher (frozen)

### nchypergeomWallenius

```php
static nchypergeomWallenius(mixed $M, mixed $n, mixed $N, mixed $odds, mixed $loc = 0): Tessero\Stats\Distribution
```

A Wallenius' noncentral hypergeometric discrete random variable.

scipy.stats.nchypergeom_wallenius (frozen)

### nct

```php
static nct(mixed $df, mixed $nc, mixed $loc = 0, mixed $scale = 1): Tessero\Stats\Distribution
```

A non-central Student's t continuous random variable.

scipy.stats.nct (frozen)

### ncx2

```php
static ncx2(mixed $df, mixed $nc, mixed $loc = 0, mixed $scale = 1): Tessero\Stats\Distribution
```

A non-central chi-squared continuous random variable.

scipy.stats.ncx2 (frozen)

### nhypergeom

```php
static nhypergeom(mixed $M, mixed $n, mixed $r, mixed $loc = 0): Tessero\Stats\Distribution
```

A negative hypergeometric discrete random variable.

scipy.stats.nhypergeom (frozen)

### norm

```php
static norm(mixed $loc = 0, mixed $scale = 1): Tessero\Stats\Distribution
```

A normal continuous random variable.

scipy.stats.norm (frozen)

### normaltest

```php
static normaltest(mixed $a, mixed $axis = 0, mixed $nanPolicy = 'propagate', mixed $keepdims = false): mixed
```

D'Agostino and Pearson's test for normality (scipy.stats.normaltest).

scipy.stats.normaltest

### norminvgauss

```php
static norminvgauss(mixed $a, mixed $b, mixed $loc = 0, mixed $scale = 1): Tessero\Stats\Distribution
```

A Normal Inverse Gaussian continuous random variable.

scipy.stats.norminvgauss (frozen)

### obrientransform

```php
static obrientransform(mixed $sample1, mixed $sample2 = null, mixed $sample3 = null, mixed $sample4 = null, mixed $sample5 = null, mixed $sample6 = null, mixed $sample7 = null, mixed $sample8 = null): mixed
```

O'Brien transform of one or more equal-length samples (scipy.stats.obrientransform).

scipy.stats.obrientransform

### pageTrendTest

```php
static pageTrendTest(mixed $data, mixed $ranked = false, mixed $predictedRanks = null, mixed $method = 'auto'): mixed
```

Page's test for ordered alternatives (scipy.stats.page_trend_test); the method used is not returned.

scipy.stats.page_trend_test

### pareto

```php
static pareto(mixed $b, mixed $loc = 0, mixed $scale = 1): Tessero\Stats\Distribution
```

A Pareto continuous random variable.

scipy.stats.pareto (frozen)

### pearson3

```php
static pearson3(mixed $skew, mixed $loc = 0, mixed $scale = 1): Tessero\Stats\Distribution
```

A pearson type III continuous random variable.

scipy.stats.pearson3 (frozen)

### pearsonr

```php
static pearsonr(mixed $x, mixed $y, ?int $axis = 0, string $alternative = 'two-sided'): array
```

Pearson correlation coefficient and p-value (scipy.stats.pearsonr; method=None only, the confidence_interval method is not provided).

scipy.stats.pearsonr

### percentileofscore

```php
static percentileofscore(mixed $a, mixed $score, mixed $kind = 'rank', mixed $nanPolicy = 'propagate'): mixed
```

Percentile rank of score(s) relative to a (scipy.stats.percentileofscore).

scipy.stats.percentileofscore

### planck

```php
static planck(mixed $lambda_, mixed $loc = 0): Tessero\Stats\Distribution
```

A Planck discrete exponential random variable.

scipy.stats.planck (frozen)

### pmean

```php
static pmean(mixed $a, mixed $p, mixed $weights = null, array|int|null $axis = 0, string $nanPolicy = 'propagate', bool $keepdims = false): Tessero\NDArray|int|float
```

Weighted power mean along an axis (scipy.stats.pmean; dtype is not supported).

scipy.stats.pmean

### pointbiserialr

```php
static pointbiserialr(mixed $x, mixed $y, array|int|null $axis = 0, string $nanPolicy = 'propagate', bool $keepdims = false): array
```

Point-biserial correlation coefficient (scipy.stats.pointbiserialr).

scipy.stats.pointbiserialr

### poisson

```php
static poisson(mixed $mu, mixed $loc = 0): Tessero\Stats\Distribution
```

A Poisson discrete random variable.

scipy.stats.poisson (frozen)

### poissonMeansTest

```php
static poissonMeansTest(mixed $k1, mixed $n1, mixed $k2, mixed $n2, mixed $diff = 0, mixed $alternative = 'two-sided'): mixed
```

Poisson means test, the E-test (scipy.stats.poisson_means_test).

scipy.stats.poisson_means_test

### powerDivergence

```php
static powerDivergence(mixed $fObs, mixed $fExp = null, mixed $ddof = 0, mixed $axis = 0, mixed $lambda_ = null, mixed $nanPolicy = 'propagate', mixed $keepdims = false): mixed
```

Cressie-Read power divergence statistic and goodness of fit test (scipy.stats.power_divergence).

scipy.stats.power_divergence

### powerlaw

```php
static powerlaw(mixed $a, mixed $loc = 0, mixed $scale = 1): Tessero\Stats\Distribution
```

A power-function continuous random variable.

scipy.stats.powerlaw (frozen)

### powerlognorm

```php
static powerlognorm(mixed $c, mixed $s, mixed $loc = 0, mixed $scale = 1): Tessero\Stats\Distribution
```

A power log-normal continuous random variable.

scipy.stats.powerlognorm (frozen)

### powernorm

```php
static powernorm(mixed $c, mixed $loc = 0, mixed $scale = 1): Tessero\Stats\Distribution
```

A power normal continuous random variable.

scipy.stats.powernorm (frozen)

### quantile

```php
static quantile(mixed $x, mixed $p, mixed $method = 'linear', mixed $axis = 0, mixed $nanPolicy = 'propagate', mixed $keepdims = null, mixed $weights = null): mixed
```

Quantiles of the data along an axis (scipy.stats.quantile).

scipy.stats.quantile

### quantileTest

```php
static quantileTest(mixed $x, mixed $q = 0, mixed $p = 0.5, mixed $alternative = 'two-sided'): mixed
```

Test that the p-th population quantile is q (scipy.stats.quantile_test).

scipy.stats.quantile_test

### randint

```php
static randint(mixed $low, mixed $high, mixed $loc = 0): Tessero\Stats\Distribution
```

A uniform discrete random variable.

scipy.stats.randint (frozen)

### rankdata

```php
static rankdata(mixed $a, mixed $method = 'average', mixed $axis = null, mixed $nanPolicy = 'propagate'): mixed
```

Ranks of the data, ties handled by method (scipy.stats.rankdata).

scipy.stats.rankdata

### ranksums

```php
static ranksums(mixed $x, mixed $y, mixed $alternative = 'two-sided', mixed $axis = 0, mixed $nanPolicy = 'propagate', mixed $keepdims = false): mixed
```

Wilcoxon rank-sum statistic for two samples (scipy.stats.ranksums).

scipy.stats.ranksums

### rayleigh

```php
static rayleigh(mixed $loc = 0, mixed $scale = 1): Tessero\Stats\Distribution
```

A Rayleigh continuous random variable.

scipy.stats.rayleigh (frozen)

### rdist

```php
static rdist(mixed $c, mixed $loc = 0, mixed $scale = 1): Tessero\Stats\Distribution
```

An R-distributed (symmetric beta) continuous random variable.

scipy.stats.rdist (frozen)

### recipinvgauss

```php
static recipinvgauss(mixed $mu, mixed $loc = 0, mixed $scale = 1): Tessero\Stats\Distribution
```

A reciprocal inverse Gaussian continuous random variable.

scipy.stats.recipinvgauss (frozen)

### reciprocal

```php
static reciprocal(mixed $a, mixed $b, mixed $loc = 0, mixed $scale = 1): Tessero\Stats\Distribution
```

A loguniform or reciprocal continuous random variable.

scipy.stats.reciprocal (frozen)

### relBreitwigner

```php
static relBreitwigner(mixed $rho, mixed $loc = 0, mixed $scale = 1): Tessero\Stats\Distribution
```

A relativistic Breit-Wigner random variable.

scipy.stats.rel_breitwigner (frozen)

### relfreq

```php
static relfreq(mixed $a, mixed $numbins = 10, mixed $defaultreallimits = null, mixed $weights = null): mixed
```

Relative frequency histogram (scipy.stats.relfreq).

scipy.stats.relfreq

### rice

```php
static rice(mixed $b, mixed $loc = 0, mixed $scale = 1): Tessero\Stats\Distribution
```

A Rice continuous random variable.

scipy.stats.rice (frozen)

### scoreatpercentile

```php
static scoreatpercentile(mixed $a, mixed $per, mixed $limit = null, mixed $interpolationMethod = 'fraction', mixed $axis = null): mixed
```

Score at the given percentile(s) of the data (scipy.stats.scoreatpercentile).

scipy.stats.scoreatpercentile

### sem

```php
static sem(mixed $a, array|int|null $axis = 0, int|float|bool|null $ddof = 1, string $nanPolicy = 'propagate', bool $keepdims = false): Tessero\NDArray|int|float
```

Standard error of the mean (scipy.stats.sem).

scipy.stats.sem

### semicircular

```php
static semicircular(mixed $loc = 0, mixed $scale = 1): Tessero\Stats\Distribution
```

A semicircular continuous random variable.

scipy.stats.semicircular (frozen)

### shapiro

```php
static shapiro(mixed $x, mixed $axis = null, mixed $nanPolicy = 'propagate', mixed $keepdims = false): mixed
```

Shapiro-Wilk test for normality (scipy.stats.shapiro).

scipy.stats.shapiro

### siegelslopes

```php
static siegelslopes(mixed $y, mixed $x = null, mixed $method = 'hierarchical', mixed $axis = null, mixed $nanPolicy = 'propagate', mixed $keepdims = false): mixed
```

Siegel's repeated-medians regression line (scipy.stats.siegelslopes).

scipy.stats.siegelslopes

### sigmaclip

```php
static sigmaclip(mixed $a, mixed $low = 4.0, mixed $high = 4.0): mixed
```

Iterative sigma-clipping of the data (scipy.stats.sigmaclip).

scipy.stats.sigmaclip

### skellam

```php
static skellam(mixed $mu1, mixed $mu2, mixed $loc = 0): Tessero\Stats\Distribution
```

A  Skellam discrete random variable.

scipy.stats.skellam (frozen)

### skew

```php
static skew(mixed $a, array|int|null $axis = 0, int|float|bool|null $bias = true, string $nanPolicy = 'propagate', bool $keepdims = false): Tessero\NDArray|int|float
```

Sample skewness of a data set (scipy.stats.skew).

scipy.stats.skew

### skewcauchy

```php
static skewcauchy(mixed $a, mixed $loc = 0, mixed $scale = 1): Tessero\Stats\Distribution
```

A skewed Cauchy random variable.

scipy.stats.skewcauchy (frozen)

### skewnorm

```php
static skewnorm(mixed $a, mixed $loc = 0, mixed $scale = 1): Tessero\Stats\Distribution
```

A skew-normal random variable.

scipy.stats.skewnorm (frozen)

### skewtest

```php
static skewtest(mixed $a, mixed $axis = 0, mixed $nanPolicy = 'propagate', mixed $alternative = 'two-sided', mixed $keepdims = false): mixed
```

Test whether the skew is different from the normal distribution (scipy.stats.skewtest).

scipy.stats.skewtest

### somersd

```php
static somersd(mixed $x, mixed $y = null, mixed $alternative = 'two-sided'): mixed
```

Somers' D of two rankings or of a contingency table (scipy.stats.somersd).

scipy.stats.somersd

### spearmanr

```php
static spearmanr(mixed $a, mixed $b = null, mixed $axis = 0, mixed $nanPolicy = 'propagate', mixed $alternative = 'two-sided'): mixed
```

Spearman rank-order correlation coefficient(s) and p-value(s) (scipy.stats.spearmanr).

scipy.stats.spearmanr

### spearmanrho

```php
static spearmanrho(mixed $x, mixed $y, array|int|null $axis = 0, string $alternative = 'two-sided', string $nanPolicy = 'propagate', bool $keepdims = false): array
```

Spearman's rank correlation (scipy.stats.spearmanrho; method=None only).

scipy.stats.spearmanrho

### studentizedRange

```php
static studentizedRange(mixed $k, mixed $df, mixed $loc = 0, mixed $scale = 1): Tessero\Stats\Distribution
```

A studentized range continuous random variable.

scipy.stats.studentized_range (frozen)

### t

```php
static t(mixed $df, mixed $loc = 0, mixed $scale = 1): Tessero\Stats\Distribution
```

A Student's t continuous random variable.

scipy.stats.t (frozen)

### theilslopes

```php
static theilslopes(mixed $y, mixed $x = null, mixed $alpha = 0.95, mixed $method = 'separate', mixed $axis = null, mixed $nanPolicy = 'propagate', mixed $keepdims = false): mixed
```

Theil-Sen estimator of a regression line with a confidence interval for the slope (scipy.stats.theilslopes).

scipy.stats.theilslopes

### tiecorrect

```php
static tiecorrect(mixed $rankvals): mixed
```

Tie correction factor for Mann-Whitney U and Kruskal-Wallis H (scipy.stats.tiecorrect).

scipy.stats.tiecorrect

### tmax

```php
static tmax(mixed $a, mixed $upperlimit = null, mixed $axis = 0, mixed $inclusive = true, mixed $nanPolicy = 'propagate', mixed $keepdims = false): mixed
```

Trimmed maximum (scipy.stats.tmax).

scipy.stats.tmax

### tmean

```php
static tmean(mixed $a, mixed $limits = null, mixed $inclusive = null, mixed $axis = null, mixed $nanPolicy = 'propagate', mixed $keepdims = false): mixed
```

Trimmed mean; inclusive=None means (True, True) (scipy.stats.tmean).

scipy.stats.tmean

### tmin

```php
static tmin(mixed $a, mixed $lowerlimit = null, mixed $axis = 0, mixed $inclusive = true, mixed $nanPolicy = 'propagate', mixed $keepdims = false): mixed
```

Trimmed minimum (scipy.stats.tmin).

scipy.stats.tmin

### trapezoid

```php
static trapezoid(mixed $c, mixed $d, mixed $loc = 0, mixed $scale = 1): Tessero\Stats\Distribution
```

A trapezoidal continuous random variable.

scipy.stats.trapezoid (frozen)

### triang

```php
static triang(mixed $c, mixed $loc = 0, mixed $scale = 1): Tessero\Stats\Distribution
```

A triangular continuous random variable.

scipy.stats.triang (frozen)

### trim1

```php
static trim1(mixed $a, mixed $proportiontocut, mixed $tail = 'right', mixed $axis = 0): mixed
```

Slice off a proportion from one end; the kept values come sorted (scipy.stats.trim1).

scipy.stats.trim1

### trimMean

```php
static trimMean(mixed $a, mixed $proportiontocut, array|int|null $axis = 0, string $nanPolicy = 'propagate', bool $keepdims = false): Tessero\NDArray|int|float
```

Mean after trimming a proportion of the smallest and largest values (scipy.stats.trim_mean).

scipy.stats.trim_mean

### trimboth

```php
static trimboth(mixed $a, mixed $proportiontocut, mixed $axis = 0): mixed
```

Slice off a proportion of items from both ends; the kept values come sorted (scipy.stats.trimboth).

scipy.stats.trimboth

### truncexpon

```php
static truncexpon(mixed $b, mixed $loc = 0, mixed $scale = 1): Tessero\Stats\Distribution
```

A truncated exponential continuous random variable.

scipy.stats.truncexpon (frozen)

### truncnorm

```php
static truncnorm(mixed $a, mixed $b, mixed $loc = 0, mixed $scale = 1): Tessero\Stats\Distribution
```

A truncated normal continuous random variable.

scipy.stats.truncnorm (frozen)

### truncpareto

```php
static truncpareto(mixed $b, mixed $c, mixed $loc = 0, mixed $scale = 1): Tessero\Stats\Distribution
```

An upper truncated Pareto continuous random variable.

scipy.stats.truncpareto (frozen)

### truncweibullMin

```php
static truncweibullMin(mixed $c, mixed $a, mixed $b, mixed $loc = 0, mixed $scale = 1): Tessero\Stats\Distribution
```

A doubly truncated Weibull minimum continuous random variable.

scipy.stats.truncweibull_min (frozen)

### tsem

```php
static tsem(mixed $a, mixed $limits = null, mixed $inclusive = null, mixed $axis = 0, mixed $ddof = 1, mixed $nanPolicy = 'propagate', mixed $keepdims = false): mixed
```

Trimmed standard error of the mean; inclusive=None means (True, True) (scipy.stats.tsem).

scipy.stats.tsem

### tstd

```php
static tstd(mixed $a, mixed $limits = null, mixed $inclusive = null, mixed $axis = 0, mixed $ddof = 1, mixed $nanPolicy = 'propagate', mixed $keepdims = false): mixed
```

Trimmed standard deviation; inclusive=None means (True, True) (scipy.stats.tstd).

scipy.stats.tstd

### ttest1samp

```php
static ttest1samp(mixed $a, mixed $popmean, mixed $axis = 0, mixed $nanPolicy = 'propagate', mixed $alternative = 'two-sided', mixed $keepdims = false): mixed
```

T-test for the mean of one group of scores (scipy.stats.ttest_1samp).

scipy.stats.ttest_1samp

### ttestInd

```php
static ttestInd(mixed $a, mixed $b, mixed $axis = 0, mixed $equalVar = true, mixed $nanPolicy = 'propagate', mixed $alternative = 'two-sided', mixed $trim = 0, mixed $method = null, mixed $keepdims = false): mixed
```

T-test for the means of two independent samples; Welch's test with equal_var=False, Yuen's with trim (scipy.stats.ttest_ind).

scipy.stats.ttest_ind

### ttestIndFromStats

```php
static ttestIndFromStats(mixed $mean1, mixed $std1, mixed $nobs1, mixed $mean2, mixed $std2, mixed $nobs2, mixed $equalVar = true, mixed $alternative = 'two-sided'): mixed
```

T-test for the means of two independent samples from descriptive statistics (scipy.stats.ttest_ind_from_stats).

scipy.stats.ttest_ind_from_stats

### ttestRel

```php
static ttestRel(mixed $a, mixed $b, mixed $axis = 0, mixed $nanPolicy = 'propagate', mixed $alternative = 'two-sided', mixed $keepdims = false): mixed
```

T-test on two related samples (scipy.stats.ttest_rel).

scipy.stats.ttest_rel

### tukeylambda

```php
static tukeylambda(mixed $lam, mixed $loc = 0, mixed $scale = 1): Tessero\Stats\Distribution
```

A Tukey-Lamdba continuous random variable.

scipy.stats.tukeylambda (frozen)

### tvar

```php
static tvar(mixed $a, mixed $limits = null, mixed $inclusive = null, mixed $axis = 0, mixed $ddof = 1, mixed $nanPolicy = 'propagate', mixed $keepdims = false): mixed
```

Trimmed variance; inclusive=None means (True, True) (scipy.stats.tvar).

scipy.stats.tvar

### uniform

```php
static uniform(mixed $loc = 0, mixed $scale = 1): Tessero\Stats\Distribution
```

A uniform continuous random variable.

scipy.stats.uniform (frozen)

### variation

```php
static variation(mixed $a, array|int|null $axis = 0, string $nanPolicy = 'propagate', int|float|bool|null $ddof = 0, bool $keepdims = false): Tessero\NDArray|int|float
```

Coefficient of variation, std / mean (scipy.stats.variation).

scipy.stats.variation

### vonmises

```php
static vonmises(mixed $kappa, mixed $loc = 0, mixed $scale = 1): Tessero\Stats\Distribution
```

A Von Mises continuous random variable.

scipy.stats.vonmises (frozen)

### vonmisesLine

```php
static vonmisesLine(mixed $kappa, mixed $loc = 0, mixed $scale = 1): Tessero\Stats\Distribution
```

A Von Mises continuous random variable.

scipy.stats.vonmises_line (frozen)

### wald

```php
static wald(mixed $loc = 0, mixed $scale = 1): Tessero\Stats\Distribution
```

A Wald continuous random variable.

scipy.stats.wald (frozen)

### wassersteinDistance

```php
static wassersteinDistance(mixed $uValues, mixed $vValues, mixed $uWeights = null, mixed $vWeights = null): mixed
```

First Wasserstein distance between two 1-D distributions (scipy.stats.wasserstein_distance).

scipy.stats.wasserstein_distance

### weibullMax

```php
static weibullMax(mixed $c, mixed $loc = 0, mixed $scale = 1): Tessero\Stats\Distribution
```

Weibull maximum continuous random variable.

scipy.stats.weibull_max (frozen)

### weibullMin

```php
static weibullMin(mixed $c, mixed $loc = 0, mixed $scale = 1): Tessero\Stats\Distribution
```

Weibull minimum continuous random variable.

scipy.stats.weibull_min (frozen)

### weightedtau

```php
static weightedtau(mixed $x, mixed $y, mixed $rank = true, mixed $weigher = null, mixed $additive = true, mixed $axis = null, mixed $nanPolicy = 'propagate', mixed $keepdims = false): mixed
```

Weighted Kendall's tau with the hyperbolic weigher (scipy.stats.weightedtau; weigher=None only).

scipy.stats.weightedtau

### wilcoxon

```php
static wilcoxon(mixed $x, mixed $y = null, mixed $zeroMethod = 'wilcox', mixed $correction = false, mixed $alternative = 'two-sided', mixed $method = 'auto', mixed $axis = 0, mixed $nanPolicy = 'propagate', mixed $keepdims = false): mixed
```

Wilcoxon signed-rank test; zstatistic with method='asymptotic' / 'approx' (scipy.stats.wilcoxon).

scipy.stats.wilcoxon

### wrapcauchy

```php
static wrapcauchy(mixed $c, mixed $loc = 0, mixed $scale = 1): Tessero\Stats\Distribution
```

A wrapped Cauchy continuous random variable.

scipy.stats.wrapcauchy (frozen)

### yeojohnson

```php
static yeojohnson(mixed $x, mixed $lmbda = null): mixed
```

The Yeo-Johnson power transformation with a given lambda (scipy.stats.yeojohnson; lmbda required).

scipy.stats.yeojohnson

### yeojohnsonLlf

```php
static yeojohnsonLlf(mixed $lmb, mixed $data, mixed $axis = 0, mixed $nanPolicy = 'propagate', mixed $keepdims = false): mixed
```

Yeo-Johnson log-likelihood (scipy.stats.yeojohnson_llf).

scipy.stats.yeojohnson_llf

### yulesimon

```php
static yulesimon(mixed $alpha, mixed $loc = 0): Tessero\Stats\Distribution
```

A Yule-Simon discrete random variable.

scipy.stats.yulesimon (frozen)

### zipf

```php
static zipf(mixed $a, mixed $loc = 0): Tessero\Stats\Distribution
```

A Zipf (Zeta) discrete random variable.

scipy.stats.zipf (frozen)

### zipfian

```php
static zipfian(mixed $a, mixed $n, mixed $loc = 0): Tessero\Stats\Distribution
```

A Zipfian discrete random variable.

scipy.stats.zipfian (frozen)

### zmap

```php
static zmap(mixed $scores, mixed $compare, ?int $axis = 0, int|float|bool|null $ddof = 0, string $nanPolicy = 'propagate'): Tessero\NDArray|int|float
```

z-scores of scores relative to the mean and standard deviation of compare (scipy.stats.zmap).

scipy.stats.zmap

### zscore

```php
static zscore(mixed $a, ?int $axis = 0, int|float|bool|null $ddof = 0, string $nanPolicy = 'propagate'): Tessero\NDArray|int|float
```

z-scores of each value relative to the sample mean and standard deviation (scipy.stats.zscore).

scipy.stats.zscore
