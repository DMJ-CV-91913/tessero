# Stats

`Tessero\Ext\Stats` *(native extension)*

## Methods

### alexandergovern

```php
static alexandergovern($sample1, $sample2, $sample3 = null, $sample4 = null, $sample5 = null, $sample6 = null, $sample7 = null, $sample8 = null, $sample9 = null, $sample10 = null, $nanPolicy = 'propagate', $axis = 0, $keepdims = false)
```

### alpha

```php
static alpha($a, $loc = 0, $scale = 1)
```

### anderson

```php
static anderson($x, $dist = 'norm', $method = null)
```

### andersonKsamp

```php
static andersonKsamp($samples, $midrank = null, $variant = null, $method = null)
```

### anglit

```php
static anglit($loc = 0, $scale = 1)
```

### ansari

```php
static ansari($x, $y, $alternative = 'two-sided', $axis = 0, $nanPolicy = 'propagate', $keepdims = false)
```

### arcsine

```php
static arcsine($loc = 0, $scale = 1)
```

### argus

```php
static argus($chi, $loc = 0, $scale = 1)
```

### bartlett

```php
static bartlett($sample1, $sample2 = null, $sample3 = null, $sample4 = null, $sample5 = null, $sample6 = null, $sample7 = null, $sample8 = null, $sample9 = null, $sample10 = null, $sample11 = null, $sample12 = null, $axis = 0, $nanPolicy = 'propagate', $keepdims = false)
```

### bernoulli

```php
static bernoulli($p, $loc = 0)
```

### beta

```php
static beta($a, $b, $loc = 0, $scale = 1)
```

### betabinom

```php
static betabinom($n, $a, $b, $loc = 0)
```

### betanbinom

```php
static betanbinom($n, $a, $b, $loc = 0)
```

### betaprime

```php
static betaprime($a, $b, $loc = 0, $scale = 1)
```

### binnedStatistic

```php
static binnedStatistic($x, $values, $statistic = 'mean', $bins = 10, $range = null)
```

### binnedStatistic2d

```php
static binnedStatistic2d($x, $y, $values, $statistic = 'mean', $bins = 10, $range = null, $expandBinnumbers = false)
```

### binnedStatisticDd

```php
static binnedStatisticDd($sample, $values, $statistic = 'mean', $bins = 10, $range = null, $expandBinnumbers = false, $binnedStatisticResult = null)
```

### binom

```php
static binom($n, $p, $loc = 0)
```

### binomtest

```php
static binomtest($k, $n, $p = 0.5, $alternative = 'two-sided')
```

### boltzmann

```php
static boltzmann($lambda_, $N, $loc = 0)
```

### boxcox

```php
static boxcox($x, $lmbda = null, $alpha = null, $optimizer = null)
```

### boxcoxLlf

```php
static boxcoxLlf($lmb, $data, $axis = 0, $keepdims = false, $nanPolicy = 'propagate')
```

### bradford

```php
static bradford($c, $loc = 0, $scale = 1)
```

### brunnermunzel

```php
static brunnermunzel($x, $y, $alternative = 'two-sided', $distribution = 't', $nanPolicy = 'propagate', $axis = 0, $keepdims = false)
```

### burr

```php
static burr($c, $d, $loc = 0, $scale = 1)
```

### burr12

```php
static burr12($c, $d, $loc = 0, $scale = 1)
```

### bwsTest

```php
static bwsTest($x, $y, $alternative = 'two-sided', $method = null)
```

### cauchy

```php
static cauchy($loc = 0, $scale = 1)
```

### chatterjeexi

```php
static chatterjeexi($x, $y, $axis = 0, $yContinuous = false, $nanPolicy = 'propagate', $keepdims = false)
```

### chi

```php
static chi($df, $loc = 0, $scale = 1)
```

### chi2

```php
static chi2($df, $loc = 0, $scale = 1)
```

### chi2Contingency

```php
static chi2Contingency($observed, $correction = true, $lambda_ = null, $method = null)
```

### chisquare

```php
static chisquare($fObs, $fExp = null, $ddof = 0, $axis = 0, $sumCheck = true, $nanPolicy = 'propagate', $keepdims = false)
```

### circmean

```php
static circmean($samples, $high = 6.283185307179586, $low = 0, $axis = null, $nanPolicy = 'propagate', $keepdims = false)
```

### circstd

```php
static circstd($samples, $high = 6.283185307179586, $low = 0, $axis = null, $nanPolicy = 'propagate', $normalize = false, $keepdims = false)
```

### circvar

```php
static circvar($samples, $high = 6.283185307179586, $low = 0, $axis = null, $nanPolicy = 'propagate', $keepdims = false)
```

### combinePvalues

```php
static combinePvalues($pvalues, $method = 'fisher', $weights = null, $axis = 0, $nanPolicy = 'propagate', $keepdims = false)
```

### cosine

```php
static cosine($loc = 0, $scale = 1)
```

### cramervonmises

```php
static cramervonmises($rvs, $cdf, $args = null, $axis = 0, $nanPolicy = 'propagate', $keepdims = false)
```

### cramervonmises2samp

```php
static cramervonmises2samp($x, $y, $method = 'auto', $axis = 0, $nanPolicy = 'propagate', $keepdims = false)
```

### crystalball

```php
static crystalball($beta, $m, $loc = 0, $scale = 1)
```

### cumfreq

```php
static cumfreq($a, $numbins = 10, $defaultreallimits = null, $weights = null)
```

### describe

```php
static describe($a, $axis = 0, $ddof = 1, $bias = true, $nanPolicy = 'propagate')
```

### dgamma

```php
static dgamma($a, $loc = 0, $scale = 1)
```

### differentialEntropy

```php
static differentialEntropy($values, $axis = 0, $windowLength = null, $base = null, $method = 'auto', $nanPolicy = 'propagate', $keepdims = false)
```

### dlaplace

```php
static dlaplace($a, $loc = 0)
```

### dparetoLognorm

```php
static dparetoLognorm($u, $s, $a, $b, $loc = 0, $scale = 1)
```

### dweibull

```php
static dweibull($c, $loc = 0, $scale = 1)
```

### energyDistance

```php
static energyDistance($uValues, $vValues, $uWeights = null, $vWeights = null)
```

### entropy

```php
static entropy($pk, $qk = null, $axis = 0, $base = null, $nanPolicy = 'propagate', $keepdims = false)
```

### eppsSingleton2samp

```php
static eppsSingleton2samp($x, $y, $t = null, $axis = 0, $nanPolicy = 'propagate', $keepdims = false)
```

### erlang

```php
static erlang($a, $loc = 0, $scale = 1)
```

### expectile

```php
static expectile($a, $alpha = 0.5, $weights = null)
```

### expon

```php
static expon($loc = 0, $scale = 1)
```

### exponnorm

```php
static exponnorm($K, $loc = 0, $scale = 1)
```

### exponpow

```php
static exponpow($b, $loc = 0, $scale = 1)
```

### exponweib

```php
static exponweib($a, $c, $loc = 0, $scale = 1)
```

### f

```php
static f($dfn, $dfd, $loc = 0, $scale = 1)
```

### fOneway

```php
static fOneway($sample1, $sample2 = null, $sample3 = null, $sample4 = null, $sample5 = null, $sample6 = null, $sample7 = null, $sample8 = null, $sample9 = null, $sample10 = null, $sample11 = null, $sample12 = null, $axis = 0, $equalVar = true, $nanPolicy = 'propagate', $keepdims = false)
```

### falseDiscoveryControl

```php
static falseDiscoveryControl($ps, $axis = 0, $method = 'bh')
```

### fatiguelife

```php
static fatiguelife($c, $loc = 0, $scale = 1)
```

### fisherExact

```php
static fisherExact($table, $alternative = null, $method = null)
```

### fisk

```php
static fisk($c, $loc = 0, $scale = 1)
```

### fligner

```php
static fligner($sample1, $sample2 = null, $sample3 = null, $sample4 = null, $sample5 = null, $sample6 = null, $sample7 = null, $sample8 = null, $sample9 = null, $sample10 = null, $sample11 = null, $sample12 = null, $center = 'median', $proportiontocut = 0.05, $axis = 0, $nanPolicy = 'propagate', $keepdims = false)
```

### foldcauchy

```php
static foldcauchy($c, $loc = 0, $scale = 1)
```

### foldnorm

```php
static foldnorm($c, $loc = 0, $scale = 1)
```

### friedmanchisquare

```php
static friedmanchisquare($sample1, $sample2 = null, $sample3 = null, $sample4 = null, $sample5 = null, $sample6 = null, $sample7 = null, $sample8 = null, $sample9 = null, $sample10 = null, $sample11 = null, $sample12 = null, $axis = 0, $nanPolicy = 'propagate', $keepdims = false)
```

### gamma

```php
static gamma($a, $loc = 0, $scale = 1)
```

### gausshyper

```php
static gausshyper($a, $b, $c, $z, $loc = 0, $scale = 1)
```

### genexpon

```php
static genexpon($a, $b, $c, $loc = 0, $scale = 1)
```

### genextreme

```php
static genextreme($c, $loc = 0, $scale = 1)
```

### gengamma

```php
static gengamma($a, $c, $loc = 0, $scale = 1)
```

### genhalflogistic

```php
static genhalflogistic($c, $loc = 0, $scale = 1)
```

### genhyperbolic

```php
static genhyperbolic($p, $a, $b, $loc = 0, $scale = 1)
```

### geninvgauss

```php
static geninvgauss($p, $b, $loc = 0, $scale = 1)
```

### genlogistic

```php
static genlogistic($c, $loc = 0, $scale = 1)
```

### gennorm

```php
static gennorm($beta, $loc = 0, $scale = 1)
```

### genpareto

```php
static genpareto($c, $loc = 0, $scale = 1)
```

### geom

```php
static geom($p, $loc = 0)
```

### gibrat

```php
static gibrat($loc = 0, $scale = 1)
```

### gmean

```php
static gmean($a, $weights = null, $axis = 0, $nanPolicy = 'propagate', $keepdims = false)
```

### gompertz

```php
static gompertz($c, $loc = 0, $scale = 1)
```

### gstd

```php
static gstd($a, $axis = 0, $ddof = 1, $nanPolicy = 'propagate', $keepdims = false)
```

### gumbelL

```php
static gumbelL($loc = 0, $scale = 1)
```

### gumbelR

```php
static gumbelR($loc = 0, $scale = 1)
```

### gzscore

```php
static gzscore($a, $axis = 0, $ddof = 0, $nanPolicy = 'propagate')
```

### halfcauchy

```php
static halfcauchy($loc = 0, $scale = 1)
```

### halfgennorm

```php
static halfgennorm($beta, $loc = 0, $scale = 1)
```

### halflogistic

```php
static halflogistic($loc = 0, $scale = 1)
```

### halfnorm

```php
static halfnorm($loc = 0, $scale = 1)
```

### hmean

```php
static hmean($a, $weights = null, $axis = 0, $nanPolicy = 'propagate', $keepdims = false)
```

### hypergeom

```php
static hypergeom($M, $n, $N, $loc = 0)
```

### hypsecant

```php
static hypsecant($loc = 0, $scale = 1)
```

### invgamma

```php
static invgamma($a, $loc = 0, $scale = 1)
```

### invgauss

```php
static invgauss($mu, $loc = 0, $scale = 1)
```

### invweibull

```php
static invweibull($c, $loc = 0, $scale = 1)
```

### iqr

```php
static iqr($x, $axis = null, $rng = null, $scale = 1.0, $nanPolicy = 'propagate', $interpolation = 'linear', $keepdims = false)
```

### irwinhall

```php
static irwinhall($n, $loc = 0, $scale = 1)
```

### jarqueBera

```php
static jarqueBera($x, $axis = null, $nanPolicy = 'propagate', $keepdims = false)
```

### jfSkewT

```php
static jfSkewT($a, $b, $loc = 0, $scale = 1)
```

### johnsonsb

```php
static johnsonsb($a, $b, $loc = 0, $scale = 1)
```

### johnsonsu

```php
static johnsonsu($a, $b, $loc = 0, $scale = 1)
```

### kappa3

```php
static kappa3($a, $loc = 0, $scale = 1)
```

### kappa4

```php
static kappa4($h, $k, $loc = 0, $scale = 1)
```

### kendalltau

```php
static kendalltau($x, $y, $axis = null, $method = 'auto', $variant = 'b', $alternative = 'two-sided', $nanPolicy = 'propagate', $keepdims = false)
```

### kruskal

```php
static kruskal($sample1, $sample2 = null, $sample3 = null, $sample4 = null, $sample5 = null, $sample6 = null, $sample7 = null, $sample8 = null, $sample9 = null, $sample10 = null, $sample11 = null, $sample12 = null, $nanPolicy = 'propagate', $axis = 0, $keepdims = false)
```

### ks1samp

```php
static ks1samp($x, $cdf, $args = null, $alternative = 'two-sided', $method = 'auto', $axis = 0, $nanPolicy = 'propagate', $keepdims = false)
```

### ks2samp

```php
static ks2samp($data1, $data2, $alternative = 'two-sided', $method = 'auto', $axis = 0, $nanPolicy = 'propagate', $keepdims = false)
```

### ksone

```php
static ksone($n, $loc = 0, $scale = 1)
```

### kstat

```php
static kstat($data, $n = 2, $axis = null, $nanPolicy = 'propagate', $keepdims = false)
```

### kstatvar

```php
static kstatvar($data, $n = 2, $axis = null, $nanPolicy = 'propagate', $keepdims = false)
```

### kstest

```php
static kstest($rvs, $cdf, $args = null, $N = 20, $alternative = 'two-sided', $method = 'auto', $axis = 0, $nanPolicy = 'propagate', $keepdims = false)
```

### kstwo

```php
static kstwo($n, $loc = 0, $scale = 1)
```

### kstwobign

```php
static kstwobign($loc = 0, $scale = 1)
```

### kurtosis

```php
static kurtosis($a, $axis = 0, $fisher = true, $bias = true, $nanPolicy = 'propagate', $keepdims = false)
```

### kurtosistest

```php
static kurtosistest($a, $axis = 0, $nanPolicy = 'propagate', $alternative = 'two-sided', $keepdims = false)
```

### landau

```php
static landau($loc = 0, $scale = 1)
```

### laplace

```php
static laplace($loc = 0, $scale = 1)
```

### laplaceAsymmetric

```php
static laplaceAsymmetric($kappa, $loc = 0, $scale = 1)
```

### levene

```php
static levene($sample1, $sample2 = null, $sample3 = null, $sample4 = null, $sample5 = null, $sample6 = null, $sample7 = null, $sample8 = null, $sample9 = null, $sample10 = null, $sample11 = null, $sample12 = null, $center = 'median', $proportiontocut = 0.05, $axis = 0, $nanPolicy = 'propagate', $keepdims = false)
```

### levy

```php
static levy($loc = 0, $scale = 1)
```

### levyL

```php
static levyL($loc = 0, $scale = 1)
```

### levyStable

```php
static levyStable($alpha, $beta, $loc = 0, $scale = 1)
```

### linregress

```php
static linregress($x, $y, $alternative = 'two-sided', $axis = 0, $nanPolicy = 'propagate', $keepdims = false)
```

### lmoment

```php
static lmoment($sample, $order = null, $axis = 0, $sorted = false, $standardize = true, $nanPolicy = 'propagate', $keepdims = false)
```

### loggamma

```php
static loggamma($c, $loc = 0, $scale = 1)
```

### logistic

```php
static logistic($loc = 0, $scale = 1)
```

### loglaplace

```php
static loglaplace($c, $loc = 0, $scale = 1)
```

### lognorm

```php
static lognorm($s, $loc = 0, $scale = 1)
```

### logser

```php
static logser($p, $loc = 0)
```

### loguniform

```php
static loguniform($a, $b, $loc = 0, $scale = 1)
```

### lomax

```php
static lomax($c, $loc = 0, $scale = 1)
```

### mannwhitneyu

```php
static mannwhitneyu($x, $y, $useContinuity = true, $alternative = 'two-sided', $axis = 0, $method = 'auto', $nanPolicy = 'propagate', $keepdims = false)
```

### maxwell

```php
static maxwell($loc = 0, $scale = 1)
```

### medianAbsDeviation

```php
static medianAbsDeviation($x, $axis = 0, $center = null, $scale = 1.0, $nanPolicy = 'propagate', $keepdims = false)
```

### medianTest

```php
static medianTest($sample1, $sample2 = null, $sample3 = null, $sample4 = null, $sample5 = null, $sample6 = null, $sample7 = null, $sample8 = null, $sample9 = null, $sample10 = null, $sample11 = null, $sample12 = null, $ties = 'below', $correction = true, $lambda_ = 1, $nanPolicy = 'propagate')
```

### mielke

```php
static mielke($k, $s, $loc = 0, $scale = 1)
```

### mode

```php
static mode($a, $axis = 0, $nanPolicy = 'propagate', $keepdims = false)
```

### moment

```php
static moment($a, $order = 1, $axis = 0, $nanPolicy = 'propagate', $center = null, $keepdims = false)
```

### mood

```php
static mood($x, $y, $axis = 0, $alternative = 'two-sided', $nanPolicy = 'propagate', $keepdims = false)
```

### moyal

```php
static moyal($loc = 0, $scale = 1)
```

### nakagami

```php
static nakagami($nu, $loc = 0, $scale = 1)
```

### nbinom

```php
static nbinom($n, $p, $loc = 0)
```

### ncf

```php
static ncf($dfn, $dfd, $nc, $loc = 0, $scale = 1)
```

### nchypergeomFisher

```php
static nchypergeomFisher($M, $n, $N, $odds, $loc = 0)
```

### nchypergeomWallenius

```php
static nchypergeomWallenius($M, $n, $N, $odds, $loc = 0)
```

### nct

```php
static nct($df, $nc, $loc = 0, $scale = 1)
```

### ncx2

```php
static ncx2($df, $nc, $loc = 0, $scale = 1)
```

### nhypergeom

```php
static nhypergeom($M, $n, $r, $loc = 0)
```

### norm

```php
static norm($loc = 0, $scale = 1)
```

### normaltest

```php
static normaltest($a, $axis = 0, $nanPolicy = 'propagate', $keepdims = false)
```

### norminvgauss

```php
static norminvgauss($a, $b, $loc = 0, $scale = 1)
```

### obrientransform

```php
static obrientransform($sample1, $sample2 = null, $sample3 = null, $sample4 = null, $sample5 = null, $sample6 = null, $sample7 = null, $sample8 = null)
```

### pageTrendTest

```php
static pageTrendTest($data, $ranked = false, $predictedRanks = null, $method = 'auto')
```

### pareto

```php
static pareto($b, $loc = 0, $scale = 1)
```

### pearson3

```php
static pearson3($skew, $loc = 0, $scale = 1)
```

### pearsonr

```php
static pearsonr($x, $y, $axis = 0, $alternative = 'two-sided')
```

### percentileofscore

```php
static percentileofscore($a, $score, $kind = 'rank', $nanPolicy = 'propagate')
```

### planck

```php
static planck($lambda_, $loc = 0)
```

### pmean

```php
static pmean($a, $p, $weights = null, $axis = 0, $nanPolicy = 'propagate', $keepdims = false)
```

### pointbiserialr

```php
static pointbiserialr($x, $y, $axis = 0, $nanPolicy = 'propagate', $keepdims = false)
```

### poisson

```php
static poisson($mu, $loc = 0)
```

### poissonMeansTest

```php
static poissonMeansTest($k1, $n1, $k2, $n2, $diff = 0, $alternative = 'two-sided')
```

### powerDivergence

```php
static powerDivergence($fObs, $fExp = null, $ddof = 0, $axis = 0, $lambda_ = null, $nanPolicy = 'propagate', $keepdims = false)
```

### powerlaw

```php
static powerlaw($a, $loc = 0, $scale = 1)
```

### powerlognorm

```php
static powerlognorm($c, $s, $loc = 0, $scale = 1)
```

### powernorm

```php
static powernorm($c, $loc = 0, $scale = 1)
```

### quantile

```php
static quantile($x, $p, $method = 'linear', $axis = 0, $nanPolicy = 'propagate', $keepdims = null, $weights = null)
```

### quantileTest

```php
static quantileTest($x, $q = 0, $p = 0.5, $alternative = 'two-sided')
```

### randint

```php
static randint($low, $high, $loc = 0)
```

### rankdata

```php
static rankdata($a, $method = 'average', $axis = null, $nanPolicy = 'propagate')
```

### ranksums

```php
static ranksums($x, $y, $alternative = 'two-sided', $axis = 0, $nanPolicy = 'propagate', $keepdims = false)
```

### rayleigh

```php
static rayleigh($loc = 0, $scale = 1)
```

### rdist

```php
static rdist($c, $loc = 0, $scale = 1)
```

### recipinvgauss

```php
static recipinvgauss($mu, $loc = 0, $scale = 1)
```

### reciprocal

```php
static reciprocal($a, $b, $loc = 0, $scale = 1)
```

### relBreitwigner

```php
static relBreitwigner($rho, $loc = 0, $scale = 1)
```

### relfreq

```php
static relfreq($a, $numbins = 10, $defaultreallimits = null, $weights = null)
```

### rice

```php
static rice($b, $loc = 0, $scale = 1)
```

### scoreatpercentile

```php
static scoreatpercentile($a, $per, $limit = null, $interpolationMethod = 'fraction', $axis = null)
```

### sem

```php
static sem($a, $axis = 0, $ddof = 1, $nanPolicy = 'propagate', $keepdims = false)
```

### semicircular

```php
static semicircular($loc = 0, $scale = 1)
```

### shapiro

```php
static shapiro($x, $axis = null, $nanPolicy = 'propagate', $keepdims = false)
```

### siegelslopes

```php
static siegelslopes($y, $x = null, $method = 'hierarchical', $axis = null, $nanPolicy = 'propagate', $keepdims = false)
```

### sigmaclip

```php
static sigmaclip($a, $low = 4.0, $high = 4.0)
```

### skellam

```php
static skellam($mu1, $mu2, $loc = 0)
```

### skew

```php
static skew($a, $axis = 0, $bias = true, $nanPolicy = 'propagate', $keepdims = false)
```

### skewcauchy

```php
static skewcauchy($a, $loc = 0, $scale = 1)
```

### skewnorm

```php
static skewnorm($a, $loc = 0, $scale = 1)
```

### skewtest

```php
static skewtest($a, $axis = 0, $nanPolicy = 'propagate', $alternative = 'two-sided', $keepdims = false)
```

### somersd

```php
static somersd($x, $y = null, $alternative = 'two-sided')
```

### spearmanr

```php
static spearmanr($a, $b = null, $axis = 0, $nanPolicy = 'propagate', $alternative = 'two-sided')
```

### spearmanrho

```php
static spearmanrho($x, $y, $axis = 0, $alternative = 'two-sided', $nanPolicy = 'propagate', $keepdims = false)
```

### studentizedRange

```php
static studentizedRange($k, $df, $loc = 0, $scale = 1)
```

### t

```php
static t($df, $loc = 0, $scale = 1)
```

### theilslopes

```php
static theilslopes($y, $x = null, $alpha = 0.95, $method = 'separate', $axis = null, $nanPolicy = 'propagate', $keepdims = false)
```

### tiecorrect

```php
static tiecorrect($rankvals)
```

### tmax

```php
static tmax($a, $upperlimit = null, $axis = 0, $inclusive = true, $nanPolicy = 'propagate', $keepdims = false)
```

### tmean

```php
static tmean($a, $limits = null, $inclusive = null, $axis = null, $nanPolicy = 'propagate', $keepdims = false)
```

### tmin

```php
static tmin($a, $lowerlimit = null, $axis = 0, $inclusive = true, $nanPolicy = 'propagate', $keepdims = false)
```

### trapezoid

```php
static trapezoid($c, $d, $loc = 0, $scale = 1)
```

### triang

```php
static triang($c, $loc = 0, $scale = 1)
```

### trim1

```php
static trim1($a, $proportiontocut, $tail = 'right', $axis = 0)
```

### trimMean

```php
static trimMean($a, $proportiontocut, $axis = 0, $nanPolicy = 'propagate', $keepdims = false)
```

### trimboth

```php
static trimboth($a, $proportiontocut, $axis = 0)
```

### truncexpon

```php
static truncexpon($b, $loc = 0, $scale = 1)
```

### truncnorm

```php
static truncnorm($a, $b, $loc = 0, $scale = 1)
```

### truncpareto

```php
static truncpareto($b, $c, $loc = 0, $scale = 1)
```

### truncweibullMin

```php
static truncweibullMin($c, $a, $b, $loc = 0, $scale = 1)
```

### tsem

```php
static tsem($a, $limits = null, $inclusive = null, $axis = 0, $ddof = 1, $nanPolicy = 'propagate', $keepdims = false)
```

### tstd

```php
static tstd($a, $limits = null, $inclusive = null, $axis = 0, $ddof = 1, $nanPolicy = 'propagate', $keepdims = false)
```

### ttest1samp

```php
static ttest1samp($a, $popmean, $axis = 0, $nanPolicy = 'propagate', $alternative = 'two-sided', $keepdims = false)
```

### ttestInd

```php
static ttestInd($a, $b, $axis = 0, $equalVar = true, $nanPolicy = 'propagate', $alternative = 'two-sided', $trim = 0, $method = null, $keepdims = false)
```

### ttestIndFromStats

```php
static ttestIndFromStats($mean1, $std1, $nobs1, $mean2, $std2, $nobs2, $equalVar = true, $alternative = 'two-sided')
```

### ttestRel

```php
static ttestRel($a, $b, $axis = 0, $nanPolicy = 'propagate', $alternative = 'two-sided', $keepdims = false)
```

### tukeylambda

```php
static tukeylambda($lam, $loc = 0, $scale = 1)
```

### tvar

```php
static tvar($a, $limits = null, $inclusive = null, $axis = 0, $ddof = 1, $nanPolicy = 'propagate', $keepdims = false)
```

### uniform

```php
static uniform($loc = 0, $scale = 1)
```

### variation

```php
static variation($a, $axis = 0, $nanPolicy = 'propagate', $ddof = 0, $keepdims = false)
```

### vonmises

```php
static vonmises($kappa, $loc = 0, $scale = 1)
```

### vonmisesLine

```php
static vonmisesLine($kappa, $loc = 0, $scale = 1)
```

### wald

```php
static wald($loc = 0, $scale = 1)
```

### wassersteinDistance

```php
static wassersteinDistance($uValues, $vValues, $uWeights = null, $vWeights = null)
```

### weibullMax

```php
static weibullMax($c, $loc = 0, $scale = 1)
```

### weibullMin

```php
static weibullMin($c, $loc = 0, $scale = 1)
```

### weightedtau

```php
static weightedtau($x, $y, $rank = true, $weigher = null, $additive = true, $axis = null, $nanPolicy = 'propagate', $keepdims = false)
```

### wilcoxon

```php
static wilcoxon($x, $y = null, $zeroMethod = 'wilcox', $correction = false, $alternative = 'two-sided', $method = 'auto', $axis = 0, $nanPolicy = 'propagate', $keepdims = false)
```

### wrapcauchy

```php
static wrapcauchy($c, $loc = 0, $scale = 1)
```

### yeojohnson

```php
static yeojohnson($x, $lmbda = null)
```

### yeojohnsonLlf

```php
static yeojohnsonLlf($lmb, $data, $axis = 0, $nanPolicy = 'propagate', $keepdims = false)
```

### yulesimon

```php
static yulesimon($alpha, $loc = 0)
```

### zipf

```php
static zipf($a, $loc = 0)
```

### zipfian

```php
static zipfian($a, $n, $loc = 0)
```

### zmap

```php
static zmap($scores, $compare, $axis = 0, $ddof = 0, $nanPolicy = 'propagate')
```

### zscore

```php
static zscore($a, $axis = 0, $ddof = 0, $nanPolicy = 'propagate')
```
