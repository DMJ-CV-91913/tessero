/*
 * scipy.stats discrete distributions (scipy/stats/_discrete_distns.py, SciPy 1.17.1), private method by private
 * method: bernoulli, betabinom, betanbinom, boltzmann, dlaplace, geom, hypergeom, logser, nbinom, nhypergeom,
 * planck, randint, skellam, yulesimon, zipf, zipfian, nchypergeom_fisher and nchypergeom_wallenius (the latter two
 * through SciPy's BiasedUrn C++ library, ported below as SciPy builds it). See stats_dist.hpp.
 */
#include "stats_dist.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace tsd {
namespace {

inline bool isintegral(double x) { return x == std::nearbyint(x); }   /* x == np.round(x) (half to even) */

/* the rvs shape parameters are all the same for the n elements (SciPy's scalar-argument path) */
bool same_shapes(int64_t n, int nshape, const double *const *sa)
{
    for (int k = 0; k < nshape; k++)
        for (int64_t i = 1; i < n; i++)
            if (!(sa[k][i] == sa[k][0])) return false;
    return true;
}

/* scipy.special.logsumexp of a 1-d array (real, no weights) */
double logsumexp(const std::vector<double> &a)
{
    const int64_t n = (int64_t)a.size();
    if (n == 0) return -INF;
    std::vector<double> e((size_t)n);
    for (int64_t i = 0; i < n; i++) e[(size_t)i] = std::exp(a[(size_t)i]);
    const double out_inf = std::log(tsr_psum(e.data(), n));
    double a_max = a[0];
    for (int64_t i = 0; i < n; i++) {                 /* np.max: NaN propagates */
        if (std::isnan(a[(size_t)i])) { a_max = a[(size_t)i]; break; }
        if (a[(size_t)i] > a_max) a_max = a[(size_t)i];
    }
    std::vector<double> mk((size_t)n);
    for (int64_t i = 0; i < n; i++) {
        const bool is_max = a[(size_t)i] == a_max;
        mk[(size_t)i] = is_max ? 1.0 : 0.0;
        e[(size_t)i] = std::exp((is_max ? -INF : a[(size_t)i]) - a_max);
    }
    double m = tsr_psum(mk.data(), n);
    double s = tsr_psum(e.data(), n);
    s = s == 0 ? s : s / m;
    auto sign = [](double x) { return std::isnan(x) ? x : (x > 0 ? 1.0 : (x < 0 ? -1.0 : 0.0)); };
    const double sgn = sign(s + 1) * sign(m);
    s = s < -1 ? -s - 2 : s;
    m = std::fabs(m);
    double out = std::log1p(s) + std::log(m) + a_max;
    if (sgn < 0) out = NaN;
    return std::isfinite(out) ? out : out_inf;
}

/* ================================================================ BiasedUrn (scipy/stats/biasedurn, Agner Fog,
   released under SciPy's license; see scipy/stats/biasedurn/license.txt), built as SciPy builds it (-DR_BUILD):
   stocc.h (univariate classes), stoc1.cpp (R_BUILD part), stoc3.cpp (univariate generators), fnchyppr.cpp and
   wnchyppr.cpp (univariate classes), erfres.cpp, stocR.h and impls.cpp. The multivariate classes are omitted. */
namespace biasedurn {

void FatalError(const char *msg) { throw std::runtime_error(msg); }

/* stocR.h: the uniform and normal generators are the caller's bitgen_t (_biasedurn.pyx next_double, next_normal) */
thread_local bitgen_t *glob_rng = nullptr;
double next_double() { return glob_rng->next_double(glob_rng->state); }
double next_normal(const double m, const double s) { return random_normal(glob_rng, m, s); }
struct StocRBase {

  double(*next_double)();
  double(*next_normal)(const double m, const double s);

  StocRBase() : next_double(NULL), next_normal(NULL) {}
  StocRBase(int seed) : next_double(NULL), next_normal(NULL) {}

  double Random() {
    return next_double();
  }

  double Normal(double m, double s) {
    // Also see impls.cpp for the StochasticLib1 implementation
    // (should be identical to this)
    return next_normal(m, s);
  }
};

/***********************************************************************
         Other simple functions
***********************************************************************/

double LnFac(int32_t n);               // log factorial (stoc1.cpp)
double LnFacr(double x);               // log factorial of non-integer (wnchyppr.cpp)
double FallingFactorial(double a, double b); // Falling factorial (wnchyppr.cpp)
double Erf (double x);                 // error function (wnchyppr.cpp)
int32_t FloorLog2(float x);            // floor(log2(x)) for x > 0 (wnchyppr.cpp)
int NumSD (double accuracy);           // used internally for determining summation interval


/***********************************************************************
         Constants and tables
***********************************************************************/

// constant for LnFac function:
static const int FAK_LEN = 1024;       // length of factorial table

// The following tables are tables of residues of a certain expansion
// of the error function. These tables are used in the Laplace method
// for calculating Wallenius' noncentral hypergeometric distribution.
// There are ERFRES_N tables covering desired precisions from
// 2^(-ERFRES_B) to 2^(-ERFRES_E). Only the table that matches the
// desired precision is used. The tables are defined in erfres.h which
// is included in wnchyppr.cpp.

// constants for ErfRes tables:
static const int ERFRES_B = 16;        // begin: -log2 of lowest precision
static const int ERFRES_E = 40;        // end:   -log2 of highest precision
static const int ERFRES_S =  2;        // step size from begin to end
static const int ERFRES_N = (ERFRES_E-ERFRES_B)/ERFRES_S+1; // number of tables
static const int ERFRES_L = 48;        // length of each table


/***********************************************************************
         Class StochasticLib1
***********************************************************************/

class StochasticLib1 : public StocRBase {
   // This class encapsulates the random variate generating functions.
   // May be derived from any of the random number generators.
public:
   StochasticLib1 (int seed);          // Constructor
   int Bernoulli(double p);            // Bernoulli distribution
   double Normal(double m, double s);  // Normal distribution
   double NormalTrunc(double m, double s, double limit); // Truncated normal distribution
   int32_t Poisson (double L);         // Poisson distribution
   int32_t Binomial (int32_t n, double p); // Binomial distribution
   int32_t Hypergeometric (int32_t n, int32_t m, int32_t N); // Hypergeometric distribution
   void Multinomial (int32_t * destination, double * source, int32_t n, int colors); // Multinomial distribution
   void Multinomial (int32_t * destination, int32_t * source, int32_t n, int colors);// Multinomial distribution
   void MultiHypergeometric (int32_t * destination, int32_t * source, int32_t n, int colors); // Multivariate hypergeometric distribution
   void Shuffle(int * list, int min, int n); // Shuffle integers

   // functions used internally
protected:
   static double fc_lnpk(int32_t k, int32_t N_Mn, int32_t M, int32_t n); // used by Hypergeometric

   // subfunctions for each approximation method
   int32_t PoissonInver(double L);                         // poisson by inversion
   int32_t PoissonRatioUniforms(double L);                 // poisson by ratio of uniforms
   int32_t PoissonLow(double L);                           // poisson for extremely low L
   int32_t BinomialInver (int32_t n, double p);            // binomial by inversion
   int32_t BinomialRatioOfUniforms (int32_t n, double p);  // binomial by ratio of uniforms
   int32_t HypInversionMod (int32_t n, int32_t M, int32_t N);  // hypergeometric by inversion searching from mode
   int32_t HypRatioOfUnifoms (int32_t n, int32_t M, int32_t N);// hypergeometric by ratio of uniforms method

   // Variables specific to each distribution:
   // Variables used by Normal distribution
   double normal_x2;  int normal_x2_valid;

   // Variables used by Hypergeometric distribution
   int32_t  hyp_n_last, hyp_m_last, hyp_N_last;            // Last values of parameters
   int32_t  hyp_mode, hyp_mp;                              // Mode, mode+1
   int32_t  hyp_bound;                                     // Safety upper bound
   double hyp_a;                                           // hat center
   double hyp_h;                                           // hat width
   double hyp_fm;                                          // Value at mode

   // Variables used by Poisson distribution
   double pois_L_last;                                     // previous value of L
   double pois_f0;                                         // value at x=0 or at mode
   double pois_a;                                          // hat center
   double pois_h;                                          // hat width
   double pois_g;                                          // ln(L)
   int32_t  pois_bound;                                    // upper bound

   // Variables used by Binomial distribution
   int32_t bino_n_last;                                    // last n
   double bino_p_last;                                     // last p
   int32_t bino_mode;                                      // mode
   int32_t bino_bound;                                     // upper bound
   double bino_a;                                          // hat center
   double bino_h;                                          // hat width
   double bino_g;                                          // value at mode
   double bino_r1;                                         // p/(1-p) or ln(p/(1-p))
};
/***********************************************************************
Class StochasticLib3
***********************************************************************/

class StochasticLib3 : public StochasticLib1 {
   // This class can be derived from either StochasticLib1 or StochasticLib2.
   // Adds more probability distributions
public:
   StochasticLib3(int seed);           // Constructor
   void SetAccuracy(double accur);     // Define accuracy of calculations
   int32_t WalleniusNCHyp (int32_t n, int32_t m, int32_t N, double odds); // Wallenius noncentral hypergeometric distribution
   int32_t FishersNCHyp (int32_t n, int32_t m, int32_t N, double odds); // Fisher's noncentral hypergeometric distribution
   void MultiWalleniusNCHyp (int32_t * destination, int32_t * source, double * weights, int32_t n, int colors); // Multivariate Wallenius noncentral hypergeometric distribution
   void MultiComplWalleniusNCHyp (int32_t * destination, int32_t * source, double * weights, int32_t n, int colors); // Multivariate complementary Wallenius noncentral hypergeometric distribution
   void MultiFishersNCHyp (int32_t * destination, int32_t * source, double * weights, int32_t n, int colors); // Multivariate Fisher's noncentral hypergeometric distribution
   // subfunctions for each approximation method
protected:
   int32_t WalleniusNCHypUrn (int32_t n, int32_t m, int32_t N, double odds); // WalleniusNCHyp by urn model
   int32_t WalleniusNCHypInversion (int32_t n, int32_t m, int32_t N, double odds); // WalleniusNCHyp by inversion method
   int32_t WalleniusNCHypTable (int32_t n, int32_t m, int32_t N, double odds); // WalleniusNCHyp by table method
   int32_t WalleniusNCHypRatioOfUnifoms (int32_t n, int32_t m, int32_t N, double odds); // WalleniusNCHyp by ratio-of-uniforms
   int32_t FishersNCHypInversion (int32_t n, int32_t m, int32_t N, double odds); // FishersNCHyp by inversion
   int32_t FishersNCHypRatioOfUnifoms (int32_t n, int32_t m, int32_t N, double odds); // FishersNCHyp by ratio-of-uniforms

   // variables
   double accuracy;                                        // desired accuracy of calculations

   // Variables for Fisher
   int32_t fnc_n_last, fnc_m_last, fnc_N_last;             // last values of parameters
   int32_t fnc_bound;                                      // upper bound
   double fnc_o_last;
   double fnc_f0, fnc_scale;
   double fnc_a;                                           // hat center
   double fnc_h;                                           // hat width
   double fnc_lfm;                                         // ln(f(mode))
   double fnc_logb;                                        // ln(odds)

   // variables for Wallenius
   int32_t wnc_n_last, wnc_m_last, wnc_N_last;             // previous parameters
   double wnc_o_last;
   int32_t wnc_bound1, wnc_bound2;                         // lower and upper bound
   int32_t wnc_mode;                                       // mode
   double wnc_a;                                           // hat center
   double wnc_h;                                           // hat width
   double wnc_k;                                           // probability value at mode
   int UseChopDown;                                        // use chop down inversion instead
   #define WALL_TABLELENGTH  512                           // max length of table
   double wall_ytable[WALL_TABLELENGTH];                   // table of probability values
   int32_t wall_tablen;                                    // length of table
   int32_t wall_x1;                                        // lower x limit for table
};


/***********************************************************************
Class CWalleniusNCHypergeometric
***********************************************************************/

class CWalleniusNCHypergeometric {
   // This class contains methods for calculating the univariate
   // Wallenius' noncentral hypergeometric probability function
public:
   CWalleniusNCHypergeometric(int32_t n, int32_t m, int32_t N, double odds, double accuracy=1.E-8); // constructor
   void SetParameters(int32_t n, int32_t m, int32_t N, double odds); // change parameters
   double probability(int32_t x);                          // calculate probability function
   int32_t MakeTable(double * table, int32_t MaxLength, int32_t * xfirst, int32_t * xlast, double cutoff = 0.); // make table of probabilities
   double mean(void);                                      // approximate mean
   double variance(void);                                  // approximate variance (poor approximation)
   int32_t mode(void);                                     // calculate mode
   double moments(double * mean, double * var);            // calculate exact mean and variance
   int BernouilliH(int32_t x, double h, double rh, StochasticLib1 *sto); // used by rejection method

   // implementations of different calculation methods
protected:
   double recursive(void);                                 // recursive calculation
   double binoexpand(void);                                // binomial expansion of integrand
   double laplace(void);                                   // Laplace's method with narrow integration interval
   double integrate(void);                                 // numerical integration

   // other subfunctions
   double lnbico(void);                                    // natural log of binomial coefficients
   void findpars(void);                                    // calculate r, w, E
   double integrate_step(double a, double b);              // used by integrate()
   double search_inflect(double t_from, double t_to);      // used by integrate()

   // parameters
   double omega;                                           // Odds
   int32_t n, m, N, x;                                     // Parameters
   int32_t xmin, xmax;                                     // Minimum and maximum x
   double accuracy;                                        // Desired precision
   // parameters used by lnbico
   int32_t xLastBico;
   double bico, mFac, xFac;
   // parameters generated by findpars and used by probability, laplace, integrate:
   double r, rd, w, wr, E, phi2d;
   int32_t xLastFindpars;
};
/***********************************************************************
Class CFishersNCHypergeometric
***********************************************************************/

class CFishersNCHypergeometric {
   // This class contains methods for calculating the univariate Fisher's
   // noncentral hypergeometric probability function
public:
   CFishersNCHypergeometric(int32_t n, int32_t m, int32_t N, double odds, double accuracy = 1E-8); // constructor
   double probability(int32_t x);                          // calculate probability function
   double probabilityRatio(int32_t x, int32_t x0);         // calculate probability f(x)/f(x0)
   double MakeTable(double * table, int32_t MaxLength, int32_t * xfirst, int32_t * xlast, double cutoff = 0.); // make table of probabilities
   double mean(void);                                      // calculate approximate mean
   double variance(void);                                  // approximate variance
   int32_t mode(void);                                     // calculate mode (exact)
   double moments(double * mean, double * var);            // calculate exact mean and variance

protected:
   double lng(int32_t x);                                  // natural log of proportional function

   // parameters
   double odds;                                            // odds ratio
   double logodds;                                         // ln odds ratio
   double accuracy;                                        // accuracy
   int32_t n, m, N;                                        // Parameters
   int32_t xmin, xmax;                                     // minimum and maximum of x

   // parameters used by subfunctions
   int32_t xLast;
   double mFac, xFac;                                      // log factorials
   double scale;                                           // scale to apply to lng function
   double rsum;                                            // reciprocal sum of proportional function
   int ParametersChanged;
};

/*************************** erfres.cpp **********************************
* Author:        Agner Fog
* Date created:  2004-07-10
* Last modified: 2008-12-12
* Project:       stocc.zip
* Source URL:    www.agner.org/random
*
* Description:
Table of residues of a certain expansion of the error function.  
These tables are used in the Laplace method for calculating Wallenius noncentral
hypergeometric distribution. Used in CWalleniusNCHypergeometric::laplace() and
CMultiWalleniusNCHypergeometric::laplace().

This file is generated by ERFRESMK.CPP. Please see the file ERFRESMK.CPP for a
detailed description. You must re-run ERFRESMK.CPP if the constants in STOCC.H
are changed.

The following constants have been used for making the tables below:
ERFRES_B =   16    (-log2 of lowest precision)
ERFRES_E =   40    (-log2 of highest precision)
ERFRES_S =    2    (step size from begin to end)
ERFRES_N =   13    (number of tables)
ERFRES_L =   48    (length of each table)

* Copyright 2004-2008 by Agner Fog. 
* Released under SciPy's license with permission of Agner Fog; see license.txt
*****************************************************************************/

   //number of standard deviations to integrate
   double NumSDev[ERFRES_N] = {
   4.324919041, 4.621231001, 4.900964208, 5.16657812, 5.419983175, 5.662697617, 5.895951217, 6.120756286, 6.337957755, 6.548269368, 6.752300431, 6.950575948, 7.143552034};

   //tables of error function residues
   double ErfRes[ERFRES_N][ERFRES_L] = {
   // 0: precision 1.53E-05
   {1.77242680540608204400E+00, 4.42974050453076994800E-01, 5.52683719287987914000E-02, 4.57346771067359261300E-03, 
   2.80459064155823224600E-04, 1.34636065677244878500E-05, 5.21352785817798300800E-07, 1.65832271688171705300E-08, 
   4.38865717471213472100E-10, 9.76518286165874680600E-12, 1.84433013221606645200E-13, 2.98319658966723379900E-15, 
   4.16751049288581722800E-17, 5.06844293411881381200E-19, 5.40629927341885830200E-21, 5.09268600245963099700E-23, 
   4.26365286677037947600E-25, 3.19120961809492396300E-27, 2.14691825888024309100E-29, 1.30473994083903636000E-31, 
   7.19567933922698314600E-34, 3.61655672748362805300E-36, 1.66299275803871018000E-38, 7.02143932105206679000E-41, 
   2.73122271211734530800E-43, 9.81824938600123102500E-46, 3.27125155121613401700E-48, 1.01290491600297417870E-50, 
   2.92208589554240568800E-53, 7.87247562929246970200E-56, 1.98510836143160618600E-58, 4.69476368999432417500E-61, 
   1.04339442450396263710E-63, 2.18317315734482557700E-66, 4.30811606197931495800E-69, 8.03081062303437395000E-72, 
   1.41637813978528824300E-74, 2.36693694351427741600E-77, 3.75309000199992425400E-80, 5.65409397708564003600E-83, 
   8.10322084538751956300E-86, 1.10610328893385430400E-88, 1.43971150303803736000E-91, 1.78884532267880002700E-94, 
   2.12393968173898899400E-97, 2.41222807417272408400E-100, 2.62311608532487946600E-103, 2.73362126618952541200E-106},
   // 1: precision 3.81E-06
   {1.77244708953065753100E+00, 4.43074113723358004800E-01, 5.53507546366094128100E-02, 4.60063583541917741200E-03, 
   2.85265530531727983900E-04, 1.39934570721569428400E-05, 5.61234181715130108200E-07, 1.87635216633109792000E-08, 
   5.29386567604284238200E-10, 1.27170893476994027400E-11, 2.62062404027629145800E-13, 4.66479837413316034000E-15, 
   7.22069968938298529400E-17, 9.78297384753513147400E-19, 1.16744590415498861200E-20, 1.23448081765041655900E-22, 
   1.16327347874717650400E-24, 9.82084801488552519700E-27, 7.46543820883360082800E-29, 5.13361419796185362400E-31, 
   3.20726459674397306300E-33, 1.82784782995019591600E-35, 9.53819678596992509200E-38, 4.57327699736894183000E-40, 
   2.02131302843758583500E-42, 8.26035836048709995200E-45, 3.13004443753993537100E-47, 1.10264466279388735400E-49, 
   3.62016356599029098800E-52, 1.11028768672354227000E-54, 3.18789098809699663200E-57, 8.58660896411902915800E-60, 
   2.17384332055877431800E-62, 5.18219413865915035000E-65, 1.16526530012222654600E-67, 2.47552943408735877700E-70, 
   4.97637013794934320200E-73, 9.47966949394160838200E-76, 1.71361124212171341900E-78, 2.94335699587741039100E-81, 
   4.80983789654609513600E-84, 7.48676877660738410200E-87, 1.11129798477201315100E-89, 1.57475145101473103400E-92, 
   2.13251069867015016100E-95, 2.76249093386952224300E-98, 3.42653604413897348900E-101, 4.07334940102519697800E-104},
   // 2: precision 9.54E-07
   {1.77245216056180140300E+00, 4.43102496776356791100E-01, 5.53772601883593673800E-02, 4.61054749828262358400E-03, 
   2.87253302758514987700E-04, 1.42417784632842086400E-05, 5.82408831964509309600E-07, 2.00745450404117050700E-08, 
   5.91011604093749423400E-10, 1.49916022838813094600E-11, 3.29741365965300606900E-13, 6.32307780683001018100E-15, 
   1.06252674842175897800E-16, 1.57257431560311360800E-18, 2.06034642322747725700E-20, 2.40159615347654528000E-22, 
   2.50271435589313449400E-24, 2.34271631492982176000E-26, 1.97869636045309031700E-28, 1.51440731538936707000E-30, 
   1.05452976534458622500E-32, 6.70612854853490875900E-35, 3.90863249061728208500E-37, 2.09490406980039604000E-39, 
   1.03572639732910843160E-41, 4.73737271771599553200E-44, 2.01016799853191990700E-46, 7.93316727009805559200E-49, 
   2.91896910080597410900E-51, 1.00361556207253403120E-53, 3.23138481735358914000E-56, 9.76266225260763484100E-59, 
   2.77288342251948021500E-61, 7.41751660051554639600E-64, 1.87191699537047863600E-66, 4.46389809367038823800E-69, 
   1.00740435367143552990E-71, 2.15468537440631290200E-74, 4.37372804933525238000E-77, 8.43676369508201162800E-80, 
   1.54845094802349484100E-82, 2.70727577941653793200E-85, 4.51412388960109772800E-88, 7.18605932463221426200E-91, 
   1.09328719452457957600E-93, 1.59123500193816486400E-96, 2.21770259794482485600E-99, 2.96235081914900644200E-102},
   // 3: precision 2.38E-07
   {1.77245342831958737100E+00, 4.43110438095780200600E-01, 5.53855581791170228000E-02, 4.61401880234106439000E-03, 
   2.88031928895194049600E-04, 1.43505456256023050800E-05, 5.92777558091362167400E-07, 2.07920891418090254000E-08, 
   6.28701715960960909000E-10, 1.65457546101845217200E-11, 3.81394501062348919800E-13, 7.73640169798996619200E-15, 
   1.38648618664047143200E-16, 2.20377376795474051600E-18, 3.11871105901085320300E-20, 3.94509797765438339700E-22, 
   4.47871054279593642800E-24, 4.58134444141001287500E-26, 4.23915369932833545200E-28, 3.56174643985755223000E-30, 
   2.72729562179570597400E-32, 1.90986605998546816600E-34, 1.22720072734085613700E-36, 7.25829034260272865500E-39, 
   3.96321699645874596800E-41, 2.00342049456074966200E-43, 9.40055798441764717800E-46, 4.10462275003981738400E-48, 
   1.67166813346582579800E-50, 6.36422340874443565900E-53, 2.26969100679582421400E-55, 7.59750937838053600600E-58, 
   2.39149482673471882600E-60, 7.09134153544718378800E-63, 1.98415128824311335000E-65, 5.24683837588056156800E-68, 
   1.31326161465641387500E-70, 3.11571024962460536800E-73, 7.01627137211411880000E-76, 1.50162731270605666400E-78, 
   3.05816530510335364700E-81, 5.93355048535012188600E-84, 1.09802441010335521600E-86, 1.94008240128183308800E-89, 
   3.27631821921541675800E-92, 5.29343480369738200400E-95, 8.19001419434114020600E-98, 1.21456436757992622700E-100},
   // 4: precision 5.96E-08
   {1.77245374525903386300E+00, 4.43112635580628681700E-01, 5.53880993417431935600E-02, 4.61519508177347361400E-03, 
   2.88323830371235781500E-04, 1.43956506488931199600E-05, 5.97533121516696046900E-07, 2.11560073234896927000E-08, 
   6.49836113541376862800E-10, 1.75091216044688314800E-11, 4.16782737060155846600E-13, 8.80643257335436424800E-15, 
   1.65748420791207225100E-16, 2.78707349086274968000E-18, 4.19899868515935354900E-20, 5.68498078698629510200E-22, 
   6.93816222596422139400E-24, 7.65747618996655475200E-26, 7.66779861336649418200E-28, 6.98905143723583695400E-30, 
   5.81737537190421990800E-32, 4.43568540037466870600E-34, 3.10768227888207447300E-36, 2.00640852664381818400E-38, 
   1.19706367104711013300E-40, 6.61729939738396217600E-43, 3.39784063694262711800E-45, 1.62450416252839296200E-47, 
   7.24798161653719932800E-50, 3.02428684730111423300E-52, 1.18255348374176440700E-54, 4.34156802253088795200E-57, 
   1.49931575039307549400E-59, 4.87879082698754128200E-62, 1.49836511723882777600E-64, 4.34998243416684050900E-67, 
   1.19554618884894856000E-69, 3.11506828608539767000E-72, 7.70504604851319512900E-75, 1.81153231245726529100E-77, 
   4.05332288179748454100E-80, 8.64127160751002389800E-83, 1.75723563299790750600E-85, 3.41217779987510142000E-88, 
   6.33324341504830543600E-91, 1.12470466360665277900E-93, 1.91282818505057981800E-96, 3.11838272111119088500E-99},
   // 5: precision 1.49E-08
   {1.77245382449389548700E+00, 4.43113238150016054000E-01, 5.53888635367372804600E-02, 4.61558298326459057200E-03, 
   2.88429374592283566800E-04, 1.44135302457832808700E-05, 5.99599530816354110000E-07, 2.13293263207088596800E-08, 
   6.60866899904610148200E-10, 1.80600922150303605400E-11, 4.38957621672449876700E-13, 9.54096365498724593600E-15, 
   1.86125270560486321400E-16, 3.26743200260750243300E-18, 5.17322947745786073000E-20, 7.40303709577309752000E-22, 
   9.59703297362487960100E-24, 1.12979041959758568400E-25, 1.21090586780714120800E-27, 1.18477600671972569200E-29, 
   1.06110784945102789800E-31, 8.72301430014194580800E-34, 6.59978694597213862400E-36, 4.60782503988683505400E-38, 
   2.97629996764696360400E-40, 1.78296967476668997800E-42, 9.92947813649120231300E-45, 5.15238281451496107200E-47, 
   2.49648080941516617600E-49, 1.13183145876711695200E-51, 4.81083885812771760200E-54, 1.92068525483444959800E-56, 
   7.21538203720691761200E-59, 2.55484244329461795400E-61, 8.54021947322263940200E-64, 2.69922457940407460300E-66, 
   8.07806757099831088400E-69, 2.29233505413233278200E-71, 6.17627451352383776600E-74, 1.58198519435517862400E-76, 
   3.85682833066898009900E-79, 8.96007783937447061800E-82, 1.98575880907873828900E-84, 4.20275001914011054200E-87, 
   8.50301055680340658200E-90, 1.64613519849643900900E-92, 3.05222294684008316300E-95, 5.42516704506242119200E-98},
   // 6: precision 3.73E-09
   {1.77245384430261089200E+00, 4.43113402125597019200E-01, 5.53890898808651020700E-02, 4.61570802060252211600E-03, 
   2.88466397094702578100E-04, 1.44203545983349722400E-05, 6.00457657669759309400E-07, 2.14076280553580130200E-08, 
   6.66287908992827087900E-10, 1.83546080772263722600E-11, 4.51849203153760888400E-13, 1.00053478654150626250E-14, 
   2.00133542358651377800E-16, 3.62647881190865840300E-18, 5.96489800325831839200E-20, 8.92069144951359438200E-22, 
   1.21499978844978062400E-23, 1.50969159775091919100E-25, 1.71458470816131592700E-27, 1.78354149193378771000E-29, 
   1.70298947555869630200E-31, 1.49600537831395400600E-33, 1.21186208172570666700E-35, 9.07362642179266008600E-38, 
   6.29382543478586469600E-40, 4.05352760000606626000E-42, 2.42933889358226154400E-44, 1.35768914148821438100E-46, 
   7.09017160688256911600E-49, 3.46664168532600651800E-51, 1.58991153690202909500E-53, 6.85218984466549798200E-56, 
   2.77986852228382907500E-58, 1.06333492956411188200E-60, 3.84102521375678317000E-63, 1.31221496031384552800E-65, 
   4.24584095965170648000E-68, 1.30291378525223696900E-70, 3.79687911940099574200E-73, 1.05205378465263412500E-75, 
   2.77502269989758744900E-78, 6.97601832816401403200E-81, 1.67315109709482392200E-83, 3.83268665565667928900E-86, 
   8.39358376033290752000E-89, 1.75907817494562062400E-91, 3.53115954806899335200E-94, 6.79562013989671425000E-97},
   // 7: precision 9.31E-10
   {1.77245384925478974400E+00, 4.43113446460012284000E-01, 5.53891560601252504200E-02, 4.61574755288994634700E-03, 
   2.88479053368568788400E-04, 1.44228769021976818600E-05, 6.00800544645992949800E-07, 2.14414502554089331400E-08, 
   6.68819005926294320800E-10, 1.85032367193584636900E-11, 4.58880445172944815400E-13, 1.02790650461108873560E-14, 
   2.09055796622121955200E-16, 3.87357904265687446300E-18, 6.55355746022352119400E-20, 1.01398465283490267200E-21, 
   1.43654532753298842400E-23, 1.86580454392148962200E-25, 2.22454554378132065200E-27, 2.43828788210971585600E-29, 
   2.46099438567553070000E-31, 2.29136593939231572900E-33, 1.97178483051357608300E-35, 1.57129911859150760300E-37, 
   1.16187715309016251400E-39, 7.98791034830625946600E-42, 5.11610271388176540200E-44, 3.05861085454619325800E-46, 
   1.71006575230074253400E-48, 8.95787473757552059200E-51, 4.40426750636187741200E-53, 2.03593329808165663200E-55, 
   8.86319619094250260800E-58, 3.63949556302483252000E-60, 1.41180525527432472100E-62, 5.18110448656726197600E-65, 
   1.80130976146235507900E-67, 5.94089489436009998000E-70, 1.86108901096460881000E-72, 5.54453617603266634800E-75, 
   1.57273231131712670500E-77, 4.25229555550383344000E-80, 1.09708064410784368000E-82, 2.70363777400980301400E-85, 
   6.37064773173804957600E-88, 1.43666982549400138800E-90, 3.10359876850474266200E-93, 6.42822304267944541900E-96},
   // 8: precision 2.33E-10
   {1.77245385049283445600E+00, 4.43113458380306853400E-01, 5.53891751960330686200E-02, 4.61575984524613369300E-03, 
   2.88483285115404915700E-04, 1.44237837119469849000E-05, 6.00933085215778545800E-07, 2.14555059613473259000E-08, 
   6.69949807134525424700E-10, 1.85746173246056176400E-11, 4.62510251141501895600E-13, 1.04309449728125451550E-14, 
   2.14376794695367282400E-16, 4.03195345507914206800E-18, 6.95901230873262760600E-20, 1.10422005968960415700E-21, 
   1.61274044622451622200E-23, 2.17010646570190394600E-25, 2.69272585719737993500E-27, 3.08406442023150341400E-29, 
   3.26412756902204044100E-31, 3.19659762892894327800E-33, 2.90079234489442113000E-35, 2.44307440922101839900E-37, 
   1.91280099578638699700E-39, 1.39463784147443818800E-41, 9.48568383329895892700E-44, 6.02906080392955580400E-46, 
   3.58720420688290561300E-48, 2.00136767763554841800E-50, 1.04877885428425423540E-52, 5.17045929753308956200E-55, 
   2.40183088534749939500E-57, 1.05288434613857573000E-59, 4.36191374659545444200E-62, 1.71017740178796946700E-64, 
   6.35417287308090154000E-67, 2.24023617204667066100E-69, 7.50388817892399787300E-72, 2.39087016939309798700E-74, 
   7.25439736654156264700E-77, 2.09846227207024494800E-79, 5.79315651373498761100E-82, 1.52786617607871741100E-84, 
   3.85332605389629328300E-87, 9.30196261538477647000E-90, 2.15126632809118648300E-92, 4.77058936290696223500E-95},
   // 9: precision 5.82E-11
   {1.77245385080234563500E+00, 4.43113461569894215700E-01, 5.53891806760746538300E-02, 4.61576361260268991600E-03, 
   2.88484673044866409200E-04, 1.44241019771415521500E-05, 6.00982861902849871600E-07, 2.14611541966231908200E-08, 
   6.70435999307504633400E-10, 1.86074527008731886600E-11, 4.64296589104966284700E-13, 1.05109058078120195880E-14, 
   2.17373506425627932200E-16, 4.12736258800510237200E-18, 7.22027572389545573000E-20, 1.16641031427122158000E-21, 
   1.74261574594878846800E-23, 2.40999131874158664000E-25, 3.08741471404781296800E-27, 3.66622899027160893300E-29, 
   4.03832398444680182100E-31, 4.12964092806000764200E-33, 3.92459969957984993300E-35, 3.47023698321199047400E-37, 
   2.85870037656881575800E-39, 2.19701222983622897200E-41, 1.57757442199878062800E-43, 1.05998290283581317870E-45, 
   6.67461794578944750100E-48, 3.94493775265477963400E-50, 2.19180590286711897200E-52, 1.14647284342367091100E-54, 
   5.65409064942635909000E-57, 2.63281413190197920300E-59, 1.15914855705146421000E-61, 4.83173813806023163900E-64, 
   1.90931412007029721900E-66, 7.16152712238209948300E-69, 2.55277823724126351900E-71, 8.65775632882397637500E-74, 
   2.79685049229469435800E-76, 8.61535752145576873700E-79, 2.53319381071928112300E-81, 7.11686161831786026200E-84, 
   1.91227899461300469000E-86, 4.91879425560043181900E-89, 1.21226578717106016000E-91, 2.86511260628508142200E-94},
   // 10: precision 1.46E-11
   {1.77245385087972342800E+00, 4.43113462419744630200E-01, 5.53891822321947835700E-02, 4.61576475266972634100E-03, 
   2.88485120632836570100E-04, 1.44242113476668549100E-05, 6.01001089101483108200E-07, 2.14633579957941871400E-08, 
   6.70638121912630560800E-10, 1.86219965341716152100E-11, 4.65139560168398521100E-13, 1.05511053035457485150E-14, 
   2.18978467579008781700E-16, 4.18179627467181890600E-18, 7.37905600609363562400E-20, 1.20666925770415139000E-21, 
   1.83216676939141016100E-23, 2.58616160243870388400E-25, 3.39612594393133643000E-27, 4.15117456105401982300E-29, 
   4.72512355800254106200E-31, 5.01108411105699264300E-33, 4.95452692086540934200E-35, 4.57052259669118191500E-37, 
   3.93757613394119041600E-39, 3.17143225730425447800E-41, 2.39087136989889684400E-43, 1.68918677399352864600E-45, 
   1.11992962513487784300E-47, 6.97720003652956407000E-50, 4.09017183052803247800E-52, 2.25925194899934230000E-54, 
   1.17743902383784437300E-56, 5.79751618317805258800E-59, 2.70049127204827368400E-61, 1.19150157862632851000E-63, 
   4.98581510751975724600E-66, 1.98102566456273457700E-68, 7.48277410614888503600E-71, 2.68994458637406843000E-73, 
   9.21308680313745922900E-76, 3.00957175301701607000E-78, 9.38604174484261857600E-81, 2.79745691952436047200E-83, 
   7.97548757616816228000E-86, 2.17700350714256603000E-88, 5.69442820814374326200E-91, 1.42855756885812751800E-93},
   // 11: precision 3.64E-12
   {1.77245385089906787700E+00, 4.43113462645337308000E-01, 5.53891826707801996000E-02, 4.61576509382801447000E-03, 
   2.88485262834342722100E-04, 1.44242482379506758200E-05, 6.01007615943023924400E-07, 2.14641957411498484200E-08, 
   6.70719685646245707700E-10, 1.86282265411023575000E-11, 4.65522856702499667400E-13, 1.05705070352080171380E-14, 
   2.19800647930093079100E-16, 4.21139261151871749000E-18, 7.47068213693802656400E-20, 1.23132525686457329000E-21, 
   1.89037080673535316000E-23, 2.70767450402634975900E-25, 3.62208731605653583200E-27, 4.52783644780645903400E-29, 
   5.29116794891083221600E-31, 5.78191926529856774600E-33, 5.91019131357709915300E-35, 5.65375339320520942200E-37, 
   5.06448494950527399600E-39, 4.25125004489814020300E-41, 3.34702040997479327500E-43, 2.47392597585772167100E-45, 
   1.71856809642179370600E-47, 1.12329116466680264100E-49, 6.91635006957699099400E-52, 4.01648185933072044700E-54, 
   2.20256743728563483200E-56, 1.14197705850825122000E-58, 5.60474946818590333800E-61, 2.60701847612354797700E-63, 
   1.15061401831998511400E-65, 4.82402847794291118400E-68, 1.92339714685666953300E-70, 7.30092195189691915600E-73, 
   2.64114863236683700200E-75, 9.11500639536260716600E-78, 3.00399043312000082200E-80, 9.46306767642663343000E-83, 
   2.85205432245625504600E-85, 8.23120145271503093200E-88, 2.27678649791096140000E-90, 6.04082678746563674000E-93},
   // 12: precision 9.09E-13
   {1.77245385090390399000E+00, 4.43113462705021723200E-01, 5.53891827935733966800E-02, 4.61576519490408572200E-03, 
   2.88485307416075940900E-04, 1.44242604760223605000E-05, 6.01009907022372119900E-07, 2.14645068933581115800E-08, 
   6.70751738699247757000E-10, 1.86308168994678478700E-11, 4.65691470353760117700E-13, 1.05795367138350319200E-14, 
   2.20205466324054638500E-16, 4.22680889851439179400E-18, 7.52117118137557251000E-20, 1.24569747014608843200E-21, 
   1.92626007811754286900E-23, 2.78693040917777943300E-25, 3.77798094465194860200E-27, 4.80270052176922369800E-29, 
   5.72806202403284098500E-31, 6.41118455649104110000E-33, 6.73530071235990996000E-35, 6.64287180769401900600E-37, 
   6.15272463485746774200E-39, 5.35401292372264035500E-41, 4.37964050507321407500E-43, 3.37013878900376065400E-45, 
   2.44151902553507999600E-47, 1.66674472552984171500E-49, 1.07324838386391679300E-51, 6.52532932562465070600E-54, 
   3.75007759408864456600E-56, 2.03933010598440151000E-58, 1.05056269424470639500E-60, 5.13240427502016103000E-63, 
   2.38044205354512290600E-65, 1.04929890842558070320E-67, 4.40052237815903136000E-70, 1.75760526644875492000E-72, 
   6.69249991110777975200E-75, 2.43182093294000139800E-77, 8.44044451319186471300E-80, 2.80086205952805676200E-82, 
   8.89407469263960473600E-85, 2.70501913533005623200E-87, 7.88617413146613817400E-90, 2.20568290007963387700E-92}
   };



/***********************************************************************
constants
***********************************************************************/
const double SHAT1 = 2.943035529371538573;    // 8/e
const double SHAT2 = 0.8989161620588987408;   // 3-sqrt(12/e)


/***********************************************************************
Log factorial function
***********************************************************************/
double LnFac(int32_t n) {
   // log factorial function. gives natural logarithm of n!

   // define constants
   static const double                 // coefficients in Stirling approximation     
      C0 =  0.918938533204672722,      // ln(sqrt(2*pi))
      C1 =  1./12., 
      C3 = -1./360.;
   // C5 =  1./1260.,                  // use r^5 term if FAK_LEN < 50
   // C7 = -1./1680.;                  // use r^7 term if FAK_LEN < 20
   // static variables
   static double fac_table[FAK_LEN];   // table of ln(n!):
   static int initialized = 0;         // remember if fac_table has been initialized

   if (n < FAK_LEN) {
      if (n <= 1) {
         if (n < 0) FatalError("Parameter negative in LnFac function");  
         return 0;
      }
      if (!initialized) {              // first time. Must initialize table
         // make table of ln(n!)
         double sum = fac_table[0] = 0.;
         for (int i=1; i<FAK_LEN; i++) {
            sum += log(double(i));
            fac_table[i] = sum;
         }
         initialized = 1;
      }
      return fac_table[n];
   }
   // not found in table. use Stirling approximation
   double  n1, r;
   n1 = n;  r  = 1. / n1;
   return (n1 + 0.5)*log(n1) - n1 + C0 + r*(C1 + r*r*C3);
}


/***********************************************************************
Constructor
***********************************************************************/
StochasticLib1::StochasticLib1 (int seed)
: StocRBase(seed) {
   // Initialize variables for various distributions
   normal_x2_valid = 0;
   hyp_n_last = hyp_m_last = hyp_N_last = -1; // Last values of hypergeometric parameters
   pois_L_last = -1.;                         // Last values of Poisson parameters
   bino_n_last = -1;  bino_p_last = -1.;      // Last values of binomial parameters
}


/***********************************************************************
Hypergeometric distribution
***********************************************************************/
int32_t StochasticLib1::Hypergeometric (int32_t n, int32_t m, int32_t N) {
   /*
   This function generates a random variate with the hypergeometric
   distribution. This is the distribution you get when drawing balls without 
   replacement from an urn with two colors. n is the number of balls you take,
   m is the number of red balls in the urn, N is the total number of balls in 
   the urn, and the return value is the number of red balls you get.

   This function uses inversion by chop-down search from the mode when
   parameters are small, and the ratio-of-uniforms method when the former
   method would be too slow or would give overflow.
   */   

   int32_t fak, addd;                    // used for undoing transformations
   int32_t x;                            // result

   // check if parameters are valid
   if (n > N || m > N || n < 0 || m < 0) {
      FatalError("Parameter out of range in hypergeometric function");}

   // symmetry transformations
   fak = 1;  addd = 0;
   if (m > N/2) {
      // invert m
      m = N - m;
      fak = -1;  addd = n;
   }    
   if (n > N/2) {
      // invert n
      n = N - n;
      addd += fak * m;  fak = - fak;
   }    
   if (n > m) {
      // swap n and m
      x = n;  n = m;  m = x;
   }    
   // cases with only one possible result end here
   if (n == 0)  return addd;

   //------------------------------------------------------------------
   //                 choose method
   //------------------------------------------------------------------
   if (N > 680 || n > 70) {
      // use ratio-of-uniforms method
      x = HypRatioOfUnifoms (n, m, N);
   }
   else {
      // inversion method, using chop-down search from mode
      x = HypInversionMod (n, m, N);
   }
   // undo symmetry transformations  
   return x * fak + addd;
}


/***********************************************************************
Subfunctions used by hypergeometric
***********************************************************************/

int32_t StochasticLib1::HypInversionMod (int32_t n, int32_t m, int32_t N) {
   /* 
   Subfunction for Hypergeometric distribution. Assumes 0 <= n <= m <= N/2.
   Overflow protection is needed when N > 680 or n > 75.

   Hypergeometric distribution by inversion method, using down-up 
   search starting at the mode using the chop-down technique.

   This method is faster than the rejection method when the variance is low.
   */

   // Sampling 
   int32_t       I;                    // Loop counter
   int32_t       L = N - m - n;        // Parameter
   double        modef;                // mode, float
   double        Mp, np;               // m + 1, n + 1
   double        p;                    // temporary
   double        U;                    // uniform random
   double        c, d;                 // factors in iteration
   double        divisor;              // divisor, eliminated by scaling
   double        k1, k2;               // float version of loop counter
   double        L1 = L;               // float version of L

   Mp = (double)(m + 1);
   np = (double)(n + 1);

   if (N != hyp_N_last || m != hyp_m_last || n != hyp_n_last) {
      // set-up when parameters have changed
      hyp_N_last = N;  hyp_m_last = m;  hyp_n_last = n;

      p  = Mp / (N + 2.);
      modef = np * p;                       // mode, real
      hyp_mode = (int32_t)modef;            // mode, integer
      if (hyp_mode == modef && p == 0.5) {   
         hyp_mp = hyp_mode--;
      }
      else {
         hyp_mp = hyp_mode + 1;
      }
      // mode probability, using log factorial function
      // (may read directly from fac_table if N < FAK_LEN)
      hyp_fm = exp(LnFac(N-m) - LnFac(L+hyp_mode) - LnFac(n-hyp_mode)
         + LnFac(m)   - LnFac(m-hyp_mode) - LnFac(hyp_mode)
         - LnFac(N)   + LnFac(N-n)      + LnFac(n)        );

      // safety bound - guarantees at least 17 significant decimal digits
      // bound = min(n, (int32_t)(modef + k*c'))
      hyp_bound = (int32_t)(modef + 11. * sqrt(modef * (1.-p) * (1.-n/(double)N)+1.));
      if (hyp_bound > n) hyp_bound = n;
   }

   // loop until accepted
   while(1) {
      U = Random();                    // uniform random number to be converted

      // start chop-down search at mode
      if ((U -= hyp_fm) <= 0.) return(hyp_mode);
      c = d = hyp_fm;

      // alternating down- and upward search from the mode
      k1 = hyp_mp - 1;  k2 = hyp_mode + 1;
      for (I = 1; I <= hyp_mode; I++, k1--, k2++) {
         // Downward search from k1 = hyp_mp - 1
         divisor = (np - k1)*(Mp - k1);
         // Instead of dividing c with divisor, we multiply U and d because 
         // multiplication is faster. This will give overflow if N > 800
         U *= divisor;  d *= divisor;
         c *= k1 * (L1 + k1);
         if ((U -= c) <= 0.)  return(hyp_mp - I - 1); // = k1 - 1

         // Upward search from k2 = hyp_mode + 1
         divisor = k2 * (L1 + k2);
         // re-scale parameters to avoid time-consuming division
         U *= divisor;  c *= divisor; 
         d *= (np - k2) * (Mp - k2);
         if ((U -= d) <= 0.)  return(hyp_mode + I);  // = k2
         // Values of n > 75 or N > 680 may give overflow if you leave out this..
         // overflow protection
         // if (U > 1.E100) {U *= 1.E-100; c *= 1.E-100; d *= 1.E-100;}
      }

      // Upward search from k2 = 2*mode + 1 to bound
      for (k2 = I = hyp_mp + hyp_mode; I <= hyp_bound; I++, k2++) {
         divisor = k2 * (L1 + k2);
         U *= divisor;
         d *= (np - k2) * (Mp - k2);
         if ((U -= d) <= 0.)  return(I);
         // more overflow protection
         // if (U > 1.E100) {U *= 1.E-100; d *= 1.E-100;}
      }
   }
}


int32_t StochasticLib1::HypRatioOfUnifoms (int32_t n, int32_t m, int32_t N) {
   /*
   Subfunction for Hypergeometric distribution using the ratio-of-uniforms
   rejection method.

   This code is valid for 0 < n <= m <= N/2.

   The computation time hardly depends on the parameters, except that it matters
   a lot whether parameters are within the range where the LnFac function is
   tabulated.

   Reference: E. Stadlober: "The ratio of uniforms approach for generating
   discrete random variates". Journal of Computational and Applied Mathematics,
   vol. 31, no. 1, 1990, pp. 181-189.
   */
   int32_t L;                          // N-m-n
   int32_t mode;                       // mode
   int32_t k;                          // integer sample
   double x;                           // real sample
   double rNN;                         // 1/(N*(N+2))
   double my;                          // mean
   double var;                         // variance
   double u;                           // uniform random
   double lf;                          // ln(f(x))

   L = N - m - n;
   if (hyp_N_last != N || hyp_m_last != m || hyp_n_last != n) {
      hyp_N_last = N;  hyp_m_last = m;  hyp_n_last = n;         // Set-up
      rNN = 1. / ((double)N*(N+2));                             // make two divisions in one
      my = (double)n * m * rNN * (N+2);                         // mean = n*m/N
      mode = (int32_t)(double(n+1) * double(m+1) * rNN * N);    // mode = floor((n+1)*(m+1)/(N+2))
      var = (double)n * m * (N-m) * (N-n) / ((double)N*N*(N-1));// variance
      hyp_h = sqrt(SHAT1 * (var+0.5)) + SHAT2;                  // hat width
      hyp_a = my + 0.5;                                         // hat center
      hyp_fm = fc_lnpk(mode, L, m, n);                          // maximum
      hyp_bound = (int32_t)(hyp_a + 4.0 * hyp_h);               // safety-bound
      if (hyp_bound > n) hyp_bound = n;
   }    
   while(1) {
      u = Random();                              // uniform random number
      if (u == 0) continue;                      // avoid division by 0
      x = hyp_a + hyp_h * (Random()-0.5) / u;    // generate hat distribution
      if (x < 0. || x > 2E9) continue;           // reject, avoid overflow
      k = (int32_t)x;
      if (k > hyp_bound) continue;               // reject if outside range
      lf = hyp_fm - fc_lnpk(k,L,m,n);            // ln(f(k))
      if (u * (4.0 - u) - 3.0 <= lf) break;      // lower squeeze accept
      if (u * (u-lf) > 1.0) continue;            // upper squeeze reject
      if (2.0 * log(u) <= lf) break;             // final acceptance
   }
   return k;
}


double StochasticLib1::fc_lnpk(int32_t k, int32_t L, int32_t m, int32_t n) {
   // subfunction used by hypergeometric and Fisher's noncentral hypergeometric distribution
   return(LnFac(k) + LnFac(m - k) + LnFac(n - k) + LnFac(L + k));
}



/* impls.cpp */
double StochasticLib1::Normal(double m, double s) { return next_normal(m, s); }


/***********************************************************************
constants
***********************************************************************/
static const double LN2 = 0.693147180559945309417; // log(2)


/***********************************************************************
Log and Exp functions with special care for small x
***********************************************************************/
// These are functions that involve expressions of the types log(1+x)
// and exp(x)-1. These functions need special care when x is small to
// avoid loss of precision. There are three versions of these functions:
// (1) Assembly version in library randomaXX.lib
// (2) Use library functions log1p and expm1 if available
// (3) Use Taylor expansion if none of the above are available

#if defined(__GNUC__) || defined (__INTEL_COMPILER) || defined(HAVE_EXPM1)
// Functions log1p(x) = log(1+x) and expm1(x) = exp(x)-1 are available
// in the math libraries of Gnu and Intel compilers
// and in R.DLL (www.r-project.org).

double pow2_1(double q, double * y0 = 0) {
   // calculate 2^q and (1-2^q) without loss of precision.
   // return value is (1-2^q). 2^q is returned in *y0
   double y, y1;
   q *= LN2;
   if (fabs(q) > 0.1) {
      y = exp(q);                      // 2^q
      y1 = 1. - y;                     // 1-2^q
   }
   else { // Use expm1
      y1 = expm1(q);                   // 2^q-1
      y = y1 + 1;                      // 2^q
      y1 = -y1;                        // 1-2^q
   }
   if (y0) *y0 = y;                    // Return y if not void pointer
   return y1;                          // Return y1
}

double log1mx(double x, double x1) {
   // Calculate log(1-x) without loss of precision when x is small.
   // Parameter x1 must be = 1-x.
   if (fabs(x) > 0.03) {
      return log(x1);
   }
   else { // use log1p(x) = log(1+x)
      return log1p(-x);
   }
}

double log1pow(double q, double x) {
   // calculate log((1-e^q)^x) without loss of precision.
   // Combines the methods of the above two functions.
   double y, y1;

   if (fabs(q) > 0.1) {
      y = exp(q);                      // e^q
      y1 = 1. - y;                     // 1-e^q
   }
   else { // Use expm1
      y1 = expm1(q);                   // e^q-1
      y = y1 + 1;                      // e^q
      y1 = -y1;                        // 1-e^q
   }

   if (y > 0.1) { // (1-y)^x calculated without problem
      return x * log(y1);
   }
   else { // Use log1p
      return x * log1p(-y);
   }
}

#else
// (3)
// Functions log1p and expm1 are not available in MS and Borland compiler
// libraries. Use explicit Taylor expansion when needed.

double pow2_1(double q, double * y0 = 0) {
   // calculate 2^q and (1-2^q) without loss of precision.
   // return value is (1-2^q). 2^q is returned in *y0
   double y, y1, y2, qn, i, ifac;
   q *= LN2;
   if (fabs(q) > 0.1) {
      y = exp(q);
      y1 = 1. - y;
   }
   else { // expand 1-e^q = -summa(q^n/n!) to avoid loss of precision
      y1 = 0;  qn = i = ifac = 1;
      do {
         y2 = y1;
         qn *= q;  ifac *= i++;
         y1 -= qn / ifac;
      }
      while (y1 != y2);
      y = 1.-y1;
   }
   if (y0) *y0 = y;
   return y1;
}

double log1mx(double x, double x1) {
   // Calculate log(1-x) without loss of precision when x is small.
   // Parameter x1 must be = 1-x.
   if (fabs(x) > 0.03) {
      return log(x1);
   }
   else { // expand ln(1-x) = -summa(x^n/n)
      double y, z1, z2, i;
      y = i = 1.;  z1 = 0;
      do {
         z2 = z1;
         y *= x;
         z1 -= y / i++;
      }
      while (z1 != z2);
      return z1;
   }
}

double log1pow(double q, double x) {
   // calculate log((1-e^q)^x) without loss of precision
   // Uses various Taylor expansions to avoid loss of precision
   double y, y1, y2, z1, z2, qn, i, ifac;

   if (fabs(q) > 0.1) {
      y = exp(q);  y1 = 1. - y;
   }
   else { // expand 1-e^q = -summa(q^n/n!) to avoid loss of precision
      y1 = 0;  qn = i = ifac = 1;
      do {
         y2 = y1;
         qn *= q;  ifac *= i++;
         y1 -= qn / ifac;
      }
      while (y1 != y2);
      y = 1. - y1;
   }
   if (y > 0.1) { // (1-y)^x calculated without problem
      return x * log(y1);
   }
   else { // expand ln(1-y) = -summa(y^n/n)
      y1 = i = 1.;  z1 = 0;
      do {
         z2 = z1;
         y1 *= y;
         z1 -= y1 / i++;
      }
      while (z1 != z2);
      return x * z1;
   }
}

#endif

/***********************************************************************
Other shared functions
***********************************************************************/

double LnFacr(double x) {
   // log factorial of non-integer x
   int32_t ix = (int32_t)(x);
   if (x == ix) return LnFac(ix);      // x is integer
   double r, r2, D = 1., f;
   static const double             
      C0 =  0.918938533204672722,      // ln(sqrt(2*pi))
      C1 =  1./12.,
      C3 = -1./360.,
      C5 =  1./1260.,
      C7 = -1./1680.;
   if (x < 6.) {
      if (x == 0 || x == 1) return 0;
      while (x < 6) D *= ++x;
   }
   r  = 1. / x;  r2 = r*r;
   f = (x + 0.5)*log(x) - x + C0 + r*(C1 + r2*(C3 + r2*(C5 + r2*C7)));
   if (D != 1.) f -= log(D);
   return f;
}


double FallingFactorial(double a, double b) {
   // calculates ln(a*(a-1)*(a-2)* ... * (a-b+1))

   if (b < 30 && int(b) == b && a < 1E10) {
      // direct calculation
      double f = 1.;
      for (int i = 0; i < b; i++) f *= a--;
      return log(f);
   }

   if (a > 100.*b && b > 1.) {
      // combine Stirling formulas for a and (a-b) to avoid loss of precision
      double ar = 1./a;
      double cr = 1./(a-b);
      // calculate -log(1-b/a) by Taylor expansion
      double s = 0., lasts, n = 1., ba = b*ar, f = ba;
      do {
         lasts = s;
         s += f/n;
         f *= ba;
         n++;
      }
      while (s != lasts);
      return (a+0.5)*s + b*log(a-b) - b + (1./12.)*(ar-cr)    
         //- (1./360.)*(ar*ar*ar-cr*cr*cr)
         ;
   }
   // use LnFacr function
   return LnFacr(a)-LnFacr(a-b);
}

double Erf (double x) {
   // Calculates the error function erf(x) as a series expansion or
   // continued fraction expansion.
   // This function may be available in math libraries as erf(x)
   static const double rsqrtpi  = 0.564189583547756286948; // 1/sqrt(pi)
   static const double rsqrtpi2 = 1.12837916709551257390;  // 2/sqrt(pi)
   if (x < 0.) return -Erf(-x);
   if (x > 6.) return 1.;
   if (x < 2.4) {
      // use series expansion
      double term;                     // term in summation
      double j21;                      // 2j+1
      double sum = 0;                  // summation
      double xx2 = x*x*2.;             // 2x^2
      int j;  
      term = x;  j21 = 1.;
      for (j=0; j<=50; j++) {          // summation loop
         sum += term;
         if (term <= 1.E-13) break;
         j21 += 2.;
         term *= xx2 / j21;
      }
      return exp(-x*x) * sum * rsqrtpi2;
   }
   else {
      // use continued fraction expansion
      double a, f;
      int n = int(2.25f*x*x - 23.4f*x + 60.84f); // predict expansion degree
      if (n < 1) n = 1;
      a = 0.5 * n;  f = x;
      for (; n > 0; n--) {             // continued fraction loop
         f = x + a / f;
         a -= 0.5;
      }
      return 1. - exp(-x*x) * rsqrtpi / f;
   }
}


int32_t FloorLog2(float x) {
   // This function calculates floor(log2(x)) for positive x.
   // The return value is <= -127 for x <= 0.

   union UfloatInt {  // Union for extracting bits from a float
      float   f;
      int32_t i;
      UfloatInt(float ff) {f = ff;}  // constructor
   };

#if defined(_M_IX86) || defined(__INTEL__) || defined(_M_X64) || defined(__IA64__) || defined(__POWERPC__)
   // Running on a platform known to use IEEE-754 floating point format
   //int32_t n = *(int32_t*)&x;
   int32_t n = UfloatInt(x).i;
   return (n >> 23) - 0x7F;
#else
   // Check if floating point format is IEEE-754
   static const UfloatInt check(1.0f);
   if (check.i == 0x3F800000) {
      // We have the standard IEEE floating point format
      int32_t n = UfloatInt(x).i;
      return (n >> 23) - 0x7F;
   }
   else {
      // Unknown floating point format
      if (x <= 0.f) return -127;
      return (int32_t)floor(log(x)*(1./LN2));
   }
#endif
}


int NumSD (double accuracy) {
   // Gives the length of the integration interval necessary to achieve
   // the desired accuracy when integrating/summating a probability 
   // function, relative to the standard deviation
   // Returns an integer approximation to 2*NormalDistrFractile(accuracy/2)
   static const double fract[] = {
      2.699796e-03, 4.652582e-04, 6.334248e-05, 6.795346e-06, 5.733031e-07,
      3.797912e-08, 1.973175e-09, 8.032001e-11, 2.559625e-12, 6.381783e-14
   };
   int i;
   for (i = 0; i < (int)(sizeof(fract)/sizeof(*fract)); i++) {
      if (accuracy >= fract[i]) break;
   }
   return i + 6;
}


/***********************************************************************
Methods for class CWalleniusNCHypergeometric
***********************************************************************/

CWalleniusNCHypergeometric::CWalleniusNCHypergeometric(int32_t n_, int32_t m_, int32_t N_, double odds_, double accuracy_) {
   // constructor
   accuracy = accuracy_;
   SetParameters(n_, m_, N_, odds_);
}


void CWalleniusNCHypergeometric::SetParameters(int32_t n_, int32_t m_, int32_t N_, double odds) {
   // change parameters
   if (n_ < 0 || n_ > N_ || m_ < 0 || m_ > N_ || odds < 0) FatalError("Parameter out of range in CWalleniusNCHypergeometric");
   n = n_; m = m_; N = N_; omega = odds;          // set parameters
   xmin = m + n - N;  if (xmin < 0) xmin = 0;     // calculate xmin
   xmax = n;  if (xmax > m) xmax = m;             // calculate xmax
   xLastBico = xLastFindpars = -99;               // indicate last x is invalid
   r = 1.;                                        // initialize
}


double CWalleniusNCHypergeometric::mean(void) {
   // find approximate mean
   int iter;                            // number of iterations
   double a, b;                         // temporaries in calculation of first guess
   double mean, mean1;                  // iteration value of mean
   double m1r, m2r;                     // 1/m, 1/m2
   double e1, e2;                       // temporaries
   double g;                            // function to find root of
   double gd;                           // derivative of g
   double omegar;                       // 1/omega

   if (omega == 1.) { // simple hypergeometric
      return (double)(m)*n/N;
   }

   if (omega == 0.) {
      if (n > N-m) FatalError("Not enough items with nonzero weight in CWalleniusNCHypergeometric::mean");
      return 0.;
   }

   if (xmin == xmax) return xmin;

   // calculate Cornfield mean of Fisher noncentral hypergeometric distribution as first guess
   a = (m+n)*omega + (N-m-n); 
   b = a*a - 4.*omega*(omega-1.)*m*n;
   b = b > 0. ? sqrt(b) : 0.;
   mean = (a-b)/(2.*(omega-1.));
   if (mean < xmin) mean = xmin;
   if (mean > xmax) mean = xmax;

   m1r = 1./m;  m2r = 1./(N-m);
   iter = 0;

   if (omega > 1.) {
      do { // Newton Raphson iteration
         mean1 = mean;
         e1 = 1.-(n-mean)*m2r;
         if (e1 < 1E-14) {
            e2 = 0.;     // avoid underflow
         }
         else {
            e2 = pow(e1,omega-1.);
         }
         g = e2*e1 + (mean-m)*m1r;
         gd = e2*omega*m2r + m1r;
         mean -= g / gd;
         if (mean < xmin) mean = xmin;
         if (mean > xmax) mean = xmax;
         if (++iter > 40) {
            FatalError("Search for mean failed in function CWalleniusNCHypergeometric::mean");
         }
      }
      while (fabs(mean1 - mean) > 2E-6);
   }
   else { // omega < 1
      omegar = 1./omega;
      do { // Newton Raphson iteration
         mean1 = mean;
         e1 = 1.-mean*m1r;
         if (e1 < 1E-14) {
            e2 = 0.;   // avoid underflow
         }
         else {
            e2 = pow(e1,omegar-1.);
         }
         g = 1.-(n-mean)*m2r-e2*e1;
         gd = e2*omegar*m1r + m2r;
         mean -= g / gd;
         if (mean < xmin) mean = xmin;
         if (mean > xmax) mean = xmax;
         if (++iter > 40) {
            FatalError("Search for mean failed in function CWalleniusNCHypergeometric::mean");
         }
      }
      while (fabs(mean1 - mean) > 2E-6);
   }
   return mean;
}


double CWalleniusNCHypergeometric::variance(void) {
   // find approximate variance (poor approximation)    
   double my = mean(); // approximate mean
   // find approximate variance from Fisher's noncentral hypergeometric approximation
   double r1 = my * (m-my); double r2 = (n-my)*(my+N-n-m);
   if (r1 <= 0. || r2 <= 0.) return 0.;
   double var = N*r1*r2/((N-1)*(m*r2+(N-m)*r1));
   if (var < 0.) var = 0.;
   return var;
}


double CWalleniusNCHypergeometric::moments(double * mean_, double * var_) {
   // calculate exact mean and variance
   // return value = sum of f(x), expected = 1.
   double y, sy=0, sxy=0, sxxy=0, me1;
   int32_t x, xm, x1;
   const double accuracy = 1E-10f;  // accuracy of calculation
   xm = (int32_t)mean();  // approximation to mean
   for (x = xm; x <= xmax; x++) {
      y = probability(x);
      x1 = x - xm;  // subtract approximate mean to avoid loss of precision in sums
      sy += y; sxy += x1 * y; sxxy += x1 * x1 * y;
      if (y < accuracy && x != xm) break;
   }
   for (x = xm-1; x >= xmin; x--) {
      y = probability(x);
      x1 = x - xm;  // subtract approximate mean to avoid loss of precision in sums
      sy += y; sxy += x1 * y; sxxy += x1 * x1 * y;
      if (y < accuracy) break;
   }

   me1 = sxy / sy;
   *mean_ = me1 + xm;
   y = sxxy / sy - me1 * me1;
   if (y < 0) y=0;
   *var_ = y;
   return sy;
}


int32_t CWalleniusNCHypergeometric::mode(void) {
   // find mode
   int32_t Mode;                       // mode

   if (omega == 1.) { 
      // simple hypergeometric
      int32_t L  = m + n - N;
      int32_t m1 = m + 1, n1 = n + 1;
      Mode = int32_t((double)m1*n1*omega/((m1+n1)*omega-L));
   }
   else {
      // find mode
      double f, f2 = 0.; // f2 = -1.; 
      int32_t xi, x2;
      int32_t xmin = m + n - N;  if (xmin < 0) xmin = 0;  // calculate xmin
      int32_t xmax = n;  if (xmax > m) xmax = m;          // calculate xmax

      Mode = (int32_t)mean();                             // floor(mean)
      if (omega < 1.) {
        if (Mode < xmax) Mode++;                        // ceil(mean)
        x2 = xmin;                                      // lower limit
        if (omega > 0.294 && N <= 10000000) {
          x2 = Mode - 1;}                    // search for mode can be limited
        for (xi = Mode; xi >= x2; xi--) {
          f = probability(xi);
          if (f <= f2) break;
          Mode = xi;  f2 = f;
        }
      }
      else {
        if (Mode < xmin) Mode++; 
        x2 = xmax;                           // upper limit
        if (omega < 3.4 && N <= 10000000) {
          x2 = Mode + 1;}                    // search for mode can be limited
        for (xi = Mode; xi <= x2; xi++) {
          f = probability(xi);
          if (f <= f2) break;
          Mode = xi; f2 = f;
        }
      }
   }
   return Mode;
}


double CWalleniusNCHypergeometric::lnbico() {
   // natural log of binomial coefficients.
   // returns lambda = log(m!*x!/(m-x)!*m2!*x2!/(m2-x2)!)
   int32_t x2 = n-x, m2 = N-m;
   if (xLastBico < 0) { // m, n, N have changed
      mFac = LnFac(m) + LnFac(m2);
   }
   if (m < FAK_LEN && m2 < FAK_LEN)  goto DEFLT;
   switch (x - xLastBico) {
  case 0: // x unchanged
     break;
  case 1: // x incremented. calculate from previous value
     xFac += log (double(x) * (m2-x2) / (double(x2+1)*(m-x+1)));
     break;
  case -1: // x decremented. calculate from previous value
     xFac += log (double(x2) * (m-x) / (double(x+1)*(m2-x2+1)));
     break;
  default: DEFLT: // calculate all
     xFac = LnFac(x) + LnFac(x2) + LnFac(m-x) + LnFac(m2-x2);
   }
   xLastBico = x;
   return bico = mFac - xFac;
}


void CWalleniusNCHypergeometric::findpars() {
   // calculate d, E, r, w
   if (x == xLastFindpars) {
      return;    // all values are unchanged since last call
   }

   // find r to center peak of integrand at 0.5
   double dd, d1, z, zd, rr, lastr, rrc, rt, r2, r21, a, b, dummy;
   double oo[2];
   double xx[2] = {static_cast<double>(x), static_cast<double>(n-x)};
   int i, j = 0;
   if (omega > 1.) { // make both omegas <= 1 to avoid overflow
      oo[0] = 1.;  oo[1] = 1./omega;
   }
   else {
      oo[0] = omega;  oo[1] = 1.;
   }
   dd = oo[0]*(m-x) + oo[1]*(N-m-xx[1]);
   d1 = 1./dd;
   E = (oo[0]*m + oo[1]*(N-m)) * d1;
   rr = r;
   if (rr <= d1) rr = 1.2*d1;           // initial guess
   // Newton-Raphson iteration to find r
   do {
      lastr = rr;
      rrc = 1. / rr;
      z = dd - rrc;
      zd = rrc * rrc;
      for (i=0; i<2; i++) {
         rt = rr * oo[i];
         if (rt < 100.) {                  // avoid overflow if rt big
            r21 = pow2_1(rt, &r2);         // r2=2^r, r21=1.-2^r
            a = oo[i] / r21;               // omegai/(1.-2^r)
            b = xx[i] * a;                 // x*omegai/(1.-2^r)
            z  += b;
            zd += b * a * LN2 * r2;
         }
      }
      if (zd == 0) FatalError("can't find r in function CWalleniusNCHypergeometric::findpars");
      rr -= z / zd;
      if (rr <= d1) rr = lastr * 0.125 + d1*0.875;
      if (++j == 70) FatalError("convergence problem searching for r in function CWalleniusNCHypergeometric::findpars");
   }
   while (fabs(rr-lastr) > rr * 1.E-6);
   if (omega > 1) {
      dd *= omega;  rr *= oo[1];
   }
   r = rr;  rd = rr * dd;

   // find peak width
   double ro, k1, k2;
   ro = r * omega;
   if (ro < 300) {                      // avoid overflow
      k1 = pow2_1(ro, &dummy);
      k1 = -1. / k1;
      k1 = omega*omega*(k1+k1*k1);
   }
   else k1 = 0.;
   if (r < 300) {                       // avoid overflow
      k2 = pow2_1(r, &dummy);
      k2 = -1. / k2;
      k2 = (k2+k2*k2);
   }
   else k2 = 0.;
   phi2d = -4.*r*r*(x*k1 + (n-x)*k2);
   if (phi2d >= 0.) {
      FatalError("peak width undefined in function CWalleniusNCHypergeometric::findpars");
      /* wr = r = 0.; */ 
   }
   else {
      wr = sqrt(-phi2d); w = 1./wr;
   }
   xLastFindpars = x;
}


int CWalleniusNCHypergeometric::BernouilliH(int32_t x_, double h, double rh, StochasticLib1 *sto) {
   // This function generates a Bernouilli variate with probability proportional
   // to the univariate Wallenius' noncentral hypergeometric distribution.
   // The return value will be 1 with probability f(x_)/h and 0 with probability
   // 1-f(x_)/h.
   // This is equivalent to calling sto->Bernouilli(probability(x_)/h),
   // but this method is faster. The method used here avoids calculating the
   // Wallenius probability by sampling in the t-domain.
   // rh is a uniform random number in the interval 0 <= rh < h. The function
   // uses additional random numbers generated from sto.
   // This function is intended for use in rejection methods for sampling from
   // the Wallenius distribution. It is called from 
   // StochasticLib3::WalleniusNCHypRatioOfUnifoms in the file stoc3.cpp
   double f0;                      // Lambda*Phi(.5)
   double phideri0;                // phi(.5)/rd
   double qi;                      // 2^(-r*omega[i])
   double qi1;                     // 1-qi
   double omegai[2] = {omega,1.};  // weights for each color
   double romegi;                  // r*omega[i]
   double xi[2] =                  // number of each color sampled
       {static_cast<double>(x_), static_cast<double>(n-x_)};
   double k;                       // adjusted width for majorizing function Ypsilon(t)
   double erfk;                    // erf correction
   double rdm1;                    // rd - 1
   double G_integral;              // integral of majorizing function Ypsilon(t)
   double ts;                      // t sampled from Ypsilon(t) distribution
   double logts;                   // log(ts)
   double rlogts;                  // r*log(ts)
   double fts;                     // Phi(ts)/rd
   double rgts;                    // 1/(Ypsilon(ts)/rd)
   double t2;                      // temporary in calculation of Ypsilon(ts)
   int i, j;                       // loop counters
   static const double rsqrt8 = 0.3535533905932737622; // 1/sqrt(8)
   static const double sqrt2pi = 2.506628274631000454; // sqrt(2*pi)

   x = x_;                         // save x in class object
   lnbico();                       // calculate bico = log(Lambda)
   findpars();                     // calculate r, d, rd, w, E
   if (E > 0.) {
      k = log(E);                  // correction for majorizing function
      k = 1. + 0.0271 * (k * sqrt(k));
   }
   else k = 1.;
   k *= w;                         // w * k   
   rdm1 = rd - 1.;

   // calculate phi(.5)/rd
   phideri0 = -LN2 * rdm1;
   for (i=0; i<2; i++) {
      romegi = r * omegai[i];
      if (romegi > 40.) {
         qi=0.;  qi1 = 1.;           // avoid underflow
      }
      else {
         qi1 = pow2_1(-romegi, &qi);
      }
      phideri0 += xi[i] * log1mx(qi, qi1);
   }

   erfk = Erf(rsqrt8 / k);
   f0 = rd * exp(phideri0 + bico);
   G_integral = f0 * sqrt2pi * k * erfk;

   if (G_integral <= h) {          // G fits under h-hat
      do {
         ts = sto->Normal(0,k);      // sample ts from normal distribution
      }
      while (fabs(ts) >= 0.5);      // reject values outside interval, and avoid ts = 0
      ts += 0.5;                    // ts = normal distributed in interval (0,1)

      for (fts=0., j=0; j<2; j++) { // calculate (Phi(ts)+Phi(1-ts))/2
         logts = log(ts);  rlogts = r * logts; // (ts = 0 avoided above)
         fts += exp(log1pow(rlogts*omega,xi[0]) + log1pow(rlogts,xi[1]) + rdm1*logts + bico);
         ts = 1. - ts;
      }
      fts *= 0.5;

      t2 = (ts-0.5) / k;            // calculate 1/Ypsilon(ts)
      rgts = exp(-(phideri0 + bico - 0.5 * t2*t2));
      return rh < G_integral * fts * rgts;   // Bernouilli variate
   }

   else { // G > h: can't use sampling in t-domain
      return rh < probability(x);
   }
}


/***********************************************************************
methods for calculating probability in class CWalleniusNCHypergeometric
***********************************************************************/

double CWalleniusNCHypergeometric::recursive() {
   // recursive calculation
   // Wallenius noncentral hypergeometric distribution by recursion formula
   // Approximate by ignoring probabilities < accuracy and minimize storage requirement
   const int BUFSIZE = 512;            // buffer size
   double p[BUFSIZE+2];                // probabilities
   double * p1, * p2;                  // offset into p
   double mxo;                         // (m-x)*omega
   double Nmnx;                        // N-m-nu+x
   double y, y1;                       // save old p[x] before it is overwritten
   double d1, d2, dcom;                // divisors in probability formula
   double accuracya;                   // absolute accuracy
   int32_t xi, nu;                     // xi, nu = recursion values of x, n
   int32_t x1, x2;                     // xi_min, xi_max

   accuracya = 0.005f * accuracy;      // absolute accuracy
   p1 = p2 = p + 1;                    // make space for p1[-1]
   p1[-1] = 0.;  p1[0]  = 1.;          // initialize for recursion
   x1 = x2 = 0;
   for (nu=1; nu<=n; nu++) {
      if (n - nu < x - x1 || p1[x1] < accuracya) {
         x1++;                        // increase lower limit when breakpoint passed or probability negligible
         p2--;                        // compensate buffer offset in order to reduce storage space
      }
      if (x2 < x && p1[x2] >= accuracya) {
         x2++;  y1 = 0.;               // increase upper limit until x has been reached
      }
      else {
         y1 = p1[x2];
      }
      if (x1 > x2) return 0.;
      if (p2+x2-p > BUFSIZE) FatalError("buffer overrun in function CWalleniusNCHypergeometric::recursive");

      mxo = (m-x2)*omega;
      Nmnx = N-m-nu+x2+1;
      for (xi = x2; xi >= x1; xi--) {  // backwards loop
         d2 = mxo + Nmnx;
         mxo += omega; Nmnx--;
         d1 = mxo + Nmnx;
         dcom = 1. / (d1 * d2);        // save a division by making common divisor
         y  = p1[xi-1]*mxo*d2*dcom + y1*(Nmnx+1)*d1*dcom;
         y1 = p1[xi-1];                // (warning: pointer alias, can't swap instruction order)
         p2[xi] = y;
      }
      p1 = p2;
   }

   if (x < x1 || x > x2) return 0.;
   return p1[x];
}


double CWalleniusNCHypergeometric::binoexpand() {
   // calculate by binomial expansion of integrand
   // only for x < 2 or n-x < 2 (not implemented for higher x because of loss of precision)
   int32_t x1, m1, m2;
   double o;
   if (x > n/2) { // invert
      x1 = n-x; m1 = N-m; m2 = m; o = 1./omega;
   }
   else {
      x1 = x; m1 = m; m2 = N-m; o = omega;
   }
   if (x1 == 0) {
      return exp(FallingFactorial(m2,n) - FallingFactorial(m2+o*m1,n));
   }    
   if (x1 == 1) {
      double d, e, q, q0, q1;
      q = FallingFactorial(m2,n-1);
      e = o*m1+m2;
      q1 = q - FallingFactorial(e,n);
      e -= o;
      q0 = q - FallingFactorial(e,n);
      d = e - (n-1);
      return m1*d*(exp(q0) - exp(q1));
   }

   FatalError("x > 1 not supported by function CWalleniusNCHypergeometric::binoexpand");
   return 0;
}


double CWalleniusNCHypergeometric::laplace() {
   // Laplace's method with narrow integration interval, 
   // using error function residues table, defined in erfres.cpp
   // Note that this function can only be used when the integrand peak is narrow.
   // findpars() must be called before this function.

   const int COLORS = 2;         // number of colors
   const int MAXDEG = 40;        // arraysize, maximum expansion degree
   int degree;                   // max expansion degree
   double accur;                 // stop expansion when terms below this threshold
   double omegai[COLORS] = {omega, 1.}; // weights for each color
   double xi[COLORS] =           // number of each color sampled
       {static_cast<double>(x), static_cast<double>(n-x)};
   double f0;                    // factor outside integral
   double rho[COLORS];           // r*omegai
   double qi;                    // 2^(-rho)
   double qi1;                   // 1-qi
   double qq[COLORS];            // qi / qi1
   double eta[COLORS+1][MAXDEG+1]; // eta coefficients
   double phideri[MAXDEG+1];     // derivatives of phi
   double PSIderi[MAXDEG+1];     // derivatives of PSI
   double * erfresp;             // pointer to table of error function residues

   // variables in asymptotic summation
   static const double sqrt8  = 2.828427124746190098; // sqrt(8)
   double qqpow;                 // qq^j
   double pow2k;                 // 2^k
   double bino;                  // binomial coefficient  
   double vr;                    // 1/v, v = integration interval
   double v2m2;                  // (2*v)^(-2)
   double v2mk1;                 // (2*v)^(-k-1)
   double s;                     // summation term
   double sum;                   // Taylor sum

   int i;                        // loop counter for color
   int j;                        // loop counter for derivative
   int k;                        // loop counter for expansion degree
   int ll;                       // k/2
   int converg = 0;              // number of consequtive terms below accuracy
   int PrecisionIndex;           // index into ErfRes table according to desired precision

   // initialize
   for (k = 0; k <= 2; k++)  phideri[k] = PSIderi[k] = 0;

   // find rho[i], qq[i], first eta coefficients, and zero'th derivative of phi
   for (i = 0; i < COLORS; i++) {
      rho[i] = r * omegai[i];
      if (rho[i] > 40.) {
         qi=0.;  qi1 = 1.;}                 // avoid underflow
      else {
         qi1 = pow2_1(-rho[i], &qi);}       // qi=2^(-rho), qi1=1.-2^(-rho)
      qq[i] = qi / qi1;                     // 2^(-r*omegai)/(1.-2^(-r*omegai))
      // peak = zero'th derivative
      phideri[0] += xi[i] * log1mx(qi, qi1);
      // eta coefficients
      eta[i][0] = 0.;
      eta[i][1] = eta[i][2] = rho[i]*rho[i];
   }

   // r, rd, and w must be calculated by findpars()
   // zero'th derivative
   phideri[0] -= (rd - 1.) * LN2;
   // scaled factor outside integral
   f0 = rd * exp(phideri[0] + lnbico());

   vr = sqrt8 * w;
   phideri[2] = phi2d;

   // get table according to desired precision
   PrecisionIndex = (-FloorLog2((float)accuracy) - ERFRES_B + ERFRES_S - 1) / ERFRES_S;
   if (PrecisionIndex < 0) PrecisionIndex = 0;
   if (PrecisionIndex > ERFRES_N-1) PrecisionIndex = ERFRES_N-1;
   while (w * NumSDev[PrecisionIndex] > 0.3) { 
      // check if integration interval is too wide
      if (PrecisionIndex == 0) {
         FatalError("Laplace method failed. Peak width too high in function CWalleniusNCHypergeometric::laplace");
         break;}
      PrecisionIndex--;                     // reduce precision to keep integration interval narrow
   }
   erfresp = ErfRes[PrecisionIndex];        // choose desired table

   degree = MAXDEG;                         // max expansion degree
   if (degree >= ERFRES_L*2) degree = ERFRES_L*2-2;

   // set up for starting loop at k=3
   v2m2 = 0.25 * vr * vr;                   // (2*v)^(-2)
   PSIderi[0] = 1.;
   pow2k = 8.;
   sum = 0.5 * vr * erfresp[0];
   v2mk1 = 0.5 * vr * v2m2 * v2m2;
   accur = accuracy * sum;

   // summation loop
   for (k = 3; k <= degree; k++) {
      phideri[k] = 0.;

      // loop for all (2) colors
      for (i = 0; i < COLORS; i++) {
         eta[i][k] = 0.;
         // backward loop for all powers
         for (j = k; j > 0; j--) {
            // find coefficients recursively from previous coefficients
            eta[i][j]  =  eta[i][j]*(j*rho[i]-(k-2)) +  eta[i][j-1]*rho[i]*(j-1);
         }
         qqpow = 1.;
         // forward loop for all powers
         for (j=1; j<=k; j++) {
            qqpow *= qq[i];                 // qq^j
            // contribution to derivative
            phideri[k] += xi[i] * eta[i][j] * qqpow;
         }
      }

      // finish calculation of derivatives
      phideri[k] = -pow2k*phideri[k] + 2*(1-k)*phideri[k-1];

      pow2k *= 2.;    // 2^k

      // loop to calculate derivatives of PSI from derivatives of psi.
      // terms # 0, 1, 2, k-2, and k-1 are zero and not included in loop.
      // The j'th derivatives of psi are identical to the derivatives of phi for j>2, and
      // zero for j=1,2. Hence we are using phideri[j] for j>2 here.
      PSIderi[k] = phideri[k];              // this is term # k
      bino = 0.5 * (k-1) * (k-2);           // binomial coefficient for term # 3
      for (j = 3; j < k-2; j++) {           // loop for remaining nonzero terms (if k>5)
         PSIderi[k] += PSIderi[k-j] * phideri[j] * bino;
         bino *= double(k-j)/double(j);
      }
      if ((k & 1) == 0) {                   // only for even k
         ll = k/2;
         s = PSIderi[k] * v2mk1 * erfresp[ll];
         sum += s;

         // check for convergence of Taylor expansion
         if (fabs(s) < accur) converg++; else converg = 0;
         if (converg > 1) break;

         // update recursive expressions
         v2mk1 *= v2m2;
      }
   }
   // multiply by terms outside integral  
   return f0 * sum;
}


double CWalleniusNCHypergeometric::integrate() {
   // Wallenius non-central hypergeometric distribution function
   // calculation by numerical integration with variable-length steps
   // NOTE: findpars() must be called before this function.
   double s;                           // result of integration step
   double sum;                         // integral
   double ta, tb;                      // subinterval for integration step

   lnbico();                           // compute log of binomial coefficients

   // choose method:
   if (w < 0.02 || (w < 0.1 && (x==m || n-x==N-m) && accuracy > 1E-6)) {
      // normal method. Step length determined by peak width w
      double delta, s1;
      s1 = accuracy < 1E-9 ? 0.5 : 1.;
      delta = s1 * w;                       // integration steplength
      ta = 0.5 + 0.5 * delta;
      sum = integrate_step(1.-ta, ta);      // first integration step around center peak
      do {
         tb = ta + delta;
         if (tb > 1.) tb = 1.;
         s  = integrate_step(ta, tb);       // integration step to the right of peak
         s += integrate_step(1.-tb,1.-ta);  // integration step to the left of peak
         sum += s;
         if (s < accuracy * sum) break;     // stop before interval finished if accuracy reached
         ta = tb;
         if (tb > 0.5 + w) delta *= 2.;     // increase step length far from peak
      }
      while (tb < 1.);
   }
   else {
      // difficult situation. Step length determined by inflection points
      double t1, t2, tinf, delta, delta1;
      sum = 0.;
      // do left and right half of integration interval separately:
      for (t1=0., t2=0.5; t1 < 1.; t1+=0.5, t2+=0.5) { 
         // integrate from 0 to 0.5 or from 0.5 to 1
         tinf = search_inflect(t1, t2);     // find inflection point
         delta = tinf - t1; if (delta > t2 - tinf) delta = t2 - tinf; // distance to nearest endpoint
         delta *= 1./7.;                    // 1/7 will give 3 steps to nearest endpoint
         if (delta < 1E-4) delta = 1E-4;
         delta1 = delta;
         // integrate from tinf forwards to t2
         ta = tinf;
         do {
            tb = ta + delta1;
            if (tb > t2 - 0.25*delta1) tb = t2; // last step of this subinterval
            s = integrate_step(ta, tb);         // integration step
            sum += s;
            delta1 *= 2;                        // double steplength
            if (s < sum * 1E-4) delta1 *= 8.;   // large step when s small
            ta = tb;
         }
         while (tb < t2);
         if (tinf) {
            // integrate from tinf backwards to t1
            tb = tinf;
            do {
               ta = tb - delta;
               if (ta < t1 + 0.25*delta) ta = t1; // last step of this subinterval
               s = integrate_step(ta, tb);        // integration step
               sum += s;
               delta *= 2;                        // double steplength
               if (s < sum * 1E-4) delta *= 8.;   // large step when s small
               tb = ta;}
            while (ta > t1);
         }
      }
   }
   return sum * rd;
}


double CWalleniusNCHypergeometric::integrate_step(double ta, double tb) {
   // integration subprocedure used by integrate()
   // makes one integration step from ta to tb using Gauss-Legendre method.
   // result is scaled by multiplication with exp(bico)
   double ab, delta, tau, ltau, y, sum, taur, rdm1;
   int i;

   // define constants for Gauss-Legendre integration with IPOINTS points
#define IPOINTS  8  // number of points in each integration step

#if   IPOINTS == 3
   static const double xval[3]    = {-.774596669241,0,0.774596668241};
   static const double weights[3] = {.5555555555555555,.88888888888888888,.55555555555555};
#elif IPOINTS == 4
   static const double xval[4]    = {-0.861136311594,-0.339981043585,0.339981043585,0.861136311594},
      static const double weights[4] = {0.347854845137,0.652145154863,0.652145154863,0.347854845137};
#elif IPOINTS == 5
   static const double xval[5]    = {-0.906179845939,-0.538469310106,0,0.538469310106,0.906179845939};
   static const double weights[5] = {0.236926885056,0.478628670499,0.568888888889,0.478628670499,0.236926885056};
#elif IPOINTS == 6
   static const double xval[6]    = {-0.932469514203,-0.661209386466,-0.238619186083,0.238619186083,0.661209386466,0.932469514203};
   static const double weights[6] = {0.171324492379,0.360761573048,0.467913934573,0.467913934573,0.360761573048,0.171324492379};
#elif IPOINTS == 8
   static const double xval[8]    = {-0.960289856498,-0.796666477414,-0.525532409916,-0.183434642496,0.183434642496,0.525532409916,0.796666477414,0.960289856498};
   static const double weights[8] = {0.10122853629,0.222381034453,0.313706645878,0.362683783378,0.362683783378,0.313706645878,0.222381034453,0.10122853629};
#elif IPOINTS == 12
   static const double xval[12]   = {-0.981560634247,-0.90411725637,-0.769902674194,-0.587317954287,-0.367831498998,-0.125233408511,0.125233408511,0.367831498998,0.587317954287,0.769902674194,0.90411725637,0.981560634247};
   static const double weights[12]= {0.0471753363866,0.106939325995,0.160078328543,0.203167426723,0.233492536538,0.249147045813,0.249147045813,0.233492536538,0.203167426723,0.160078328543,0.106939325995,0.0471753363866};
#elif IPOINTS == 16
   static const double xval[16]   = {-0.989400934992,-0.944575023073,-0.865631202388,-0.755404408355,-0.617876244403,-0.458016777657,-0.281603550779,-0.0950125098376,0.0950125098376,0.281603550779,0.458016777657,0.617876244403,0.755404408355,0.865631202388,0.944575023073,0.989400934992};
   static const double weights[16]= {0.027152459411,0.0622535239372,0.0951585116838,0.124628971256,0.149595988817,0.169156519395,0.182603415045,0.189450610455,0.189450610455,0.182603415045,0.169156519395,0.149595988817,0.124628971256,0.0951585116838,0.0622535239372,0.027152459411};
#else
#error // IPOINTS must be a value for which the tables are defined
#endif

   delta = 0.5 * (tb - ta);
   ab = 0.5 * (ta + tb);
   rdm1 = rd - 1.;
   sum = 0;

   for (i = 0; i < IPOINTS; i++) {
      tau = ab + delta * xval[i];
      ltau = log(tau);
      taur = r * ltau;
      // possible loss of precision due to subtraction here:
      y = log1pow(taur*omega,x) + log1pow(taur,n-x) + rdm1*ltau + bico;
      if (y > -50.) sum += weights[i] * exp(y);
   }
   return delta * sum;
}


double CWalleniusNCHypergeometric::search_inflect(double t_from, double t_to) {
   // search for an inflection point of the integrand PHI(t) in the interval
   // t_from < t < t_to
   const int COLORS = 2;                // number of colors
   double t, t1;                        // independent variable
   double rho[COLORS];                  // r*omega[i]
   double q;                            // t^rho[i] / (1-t^rho[i])
   double q1;                           // 1-t^rho[i]
   double xx[COLORS];                   // x[i]
   double zeta[COLORS][4][4];           // zeta[i,j,k] coefficients
   double phi[4];                       // derivatives of phi(t) = log PHI(t)
   double Z2;                           // PHI''(t)/PHI(t)
   double Zd;                           // derivative in Newton Raphson iteration
   double rdm1;                         // r * d - 1
   double tr;                           // 1/t
   double log2t;                        // log2(t)
   double method;                       // 0 for z2'(t) method, 1 for z3(t) method
   int i;                               // color
   int iter;                            // count iterations

   rdm1 = rd - 1.;
   if (t_from == 0 && rdm1 <= 1.) return 0.;//no inflection point
   rho[0] = r*omega;  rho[1] = r;
   xx[0] = x;  xx[1] = n - x;
   t = 0.5 * (t_from + t_to);
   for (i = 0; i < COLORS; i++) {           // calculate zeta coefficients
      zeta[i][1][1] = rho[i];
      zeta[i][1][2] = rho[i] * (rho[i] - 1.);
      zeta[i][2][2] = rho[i] * rho[i];
      zeta[i][1][3] = zeta[i][1][2] * (rho[i] - 2.);
      zeta[i][2][3] = zeta[i][1][2] * rho[i] * 3.;
      zeta[i][3][3] = zeta[i][2][2] * rho[i] * 2.;
   }
   iter = 0;

   do {
      t1 = t;
      tr = 1. / t;
      log2t = log(t)*(1./LN2);
      phi[1] = phi[2] = phi[3] = 0.;
      for (i=0; i<COLORS; i++) {            // calculate first 3 derivatives of phi(t)
         q1 = pow2_1(rho[i]*log2t,&q);
         q /= q1;
         phi[1] -= xx[i] * zeta[i][1][1] * q;
         phi[2] -= xx[i] * q * (zeta[i][1][2] + q * zeta[i][2][2]);
         phi[3] -= xx[i] * q * (zeta[i][1][3] + q * (zeta[i][2][3] + q * zeta[i][3][3]));
      }
      phi[1] += rdm1;
      phi[2] -= rdm1;
      phi[3] += 2. * rdm1;
      phi[1] *= tr;
      phi[2] *= tr * tr;
      phi[3] *= tr * tr * tr;
      method = (iter & 2) >> 1;        // alternate between the two methods
      Z2 = phi[1]*phi[1] + phi[2];
      Zd = method*phi[1]*phi[1]*phi[1] + (2.+method)*phi[1]*phi[2] + phi[3];

      if (t < 0.5) {
         if (Z2 > 0) {
            t_from = t;
         }
         else {
            t_to = t;
         }
         if (Zd >= 0) { 
            // use binary search if Newton-Raphson iteration makes problems
            t = (t_from ? 0.5 : 0.2) * (t_from + t_to);
         }
         else {
            // Newton-Raphson iteration
            t -= Z2 / Zd;
         }
      }
      else {
         if (Z2 < 0) {
            t_from = t;
         }
         else {
            t_to = t;
         }
         if (Zd <= 0) {
            // use binary search if Newton-Raphson iteration makes problems
            t = 0.5 * (t_from + t_to);
         }
         else {
            // Newton-Raphson iteration
            t -= Z2 / Zd;
         }
      }
      if (t >= t_to) t = (t1 + t_to) * 0.5;
      if (t <= t_from) t = (t1 + t_from) * 0.5;
      if (++iter > 20) FatalError("Search for inflection point failed in function CWalleniusNCHypergeometric::search_inflect");
   }
   while (fabs(t - t1) > 1E-5);
   return t;
}


double CWalleniusNCHypergeometric::probability(int32_t x_) {
   // calculate probability function. choosing best method
   x = x_;
   if (x < xmin || x > xmax) return 0.;
   if (xmin == xmax) return 1.;

   if (omega == 1.) { // hypergeometric
      return exp(lnbico() + LnFac(n) + LnFac(N-n) - LnFac(N));
   }

   if (omega == 0.) {
      if (n > N-m) FatalError("Not enough items with nonzero weight in CWalleniusNCHypergeometric::probability");
      return x == 0;
   }

   int32_t x2 = n - x;
   int32_t x0 = x < x2 ? x : x2;
   int em = (x == m || x2 == N-m);

   if (x0 == 0 && n > 500) {
      return binoexpand();
   }

   if (double(n)*x0 < 1000 || (double(n)*x0 < 10000 && (N > 1000.*n || em))) {
      return recursive();
   }

   if (x0 <= 1 && N-n <= 1) {
      return binoexpand();
   }

   findpars();

   if (w < 0.04 && E < 10 && (!em || w > 0.004)) {
      return laplace();
   }

   return integrate();
}


int32_t CWalleniusNCHypergeometric::MakeTable(double * table, int32_t MaxLength, int32_t * xfirst, int32_t * xlast, double cutoff) {
   // Makes a table of Wallenius noncentral hypergeometric probabilities 
   // table must point to an array of length MaxLength. 
   // The function returns 1 if table is long enough. Otherwise it fills
   // the table with as many correct values as possible and returns 0.
   // The tails are cut off where the values are < cutoff, so that 
   // *xfirst may be > xmin and *xlast may be < xmax.
   // The value of cutoff will be 0.01 * accuracy if not specified.
   // The first and last x value represented in the table are returned in 
   // *xfirst and *xlast. The resulting probability values are returned in 
   // the first (*xfirst - *xlast + 1) positions of table. Any unused part
   // of table may be overwritten with garbage.
   //
   // The function will return the following information when MaxLength = 0:
   // The return value is the desired length of table.
   // *xfirst is 1 if it will be more efficient to call MakeTable than to call
   // probability repeatedly, even if only some of the table values are needed.
   // *xfirst is 0 if it is more efficient to call probability repeatedly.

   double * p1, * p2;                  // offset into p
   double mxo;                         // (m-x)*omega
   double Nmnx;                        // N-m-nu+x
   double y, y1;                       // probability. Save old p[x] before it is overwritten
   double d1, d2, dcom;                // divisors in probability formula
   double area;                        // estimate of area needed for recursion method
   int32_t xi, nu;                     // xi, nu = recursion values of x, n
   int32_t x1, x2;                     // lowest and highest x or xi
   int32_t i1, i2;                     // index into table
   int32_t UseTable;                   // 1 if table method used
   int32_t LengthNeeded;               // Necessary table length

   // special cases
   if (n == 0 || m == 0) {x1 = 0; goto DETERMINISTIC;}
   if (n == N)           {x1 = m; goto DETERMINISTIC;}
   if (m == N)           {x1 = n; goto DETERMINISTIC;}
   if (omega <= 0.) {
      if (n > N-m) FatalError("Not enough items with nonzero weight in  CWalleniusNCHypergeometric::MakeTable");
      x1 = 0;
      DETERMINISTIC:
      if (MaxLength == 0) {
         if (xfirst) *xfirst = 1;
         return 1;
      }
      *xfirst = *xlast = x1;
      *table = 1.;
      return 1;
   }

   if (cutoff <= 0. || cutoff > 0.1) cutoff = 0.01 * accuracy;

   LengthNeeded = N - m;               // m2
   if (m < LengthNeeded) LengthNeeded = m;
   if (n < LengthNeeded) LengthNeeded = n; // LengthNeeded = min(m1,m2,n)
   area = double(n)*LengthNeeded;      // Estimate calculation time for table method
   UseTable = area < 5000. || (area < 10000. && N > 1000. * n);

   if (MaxLength <= 0) {
      // Return UseTable and LengthNeeded
      if (xfirst) *xfirst = UseTable;
      i1 = LengthNeeded + 2;           // Necessary table length
      if (!UseTable && i1 > 200) {
         // Calculate necessary table length from standard deviation
         double sd = sqrt(variance()); // calculate approximate standard deviation
         // estimate number of standard deviations to include from normal distribution
         i2 = (int32_t)(NumSD(accuracy) * sd + 0.5);
         if (i1 > i2) i1 = i2;
      }
      return i1;
   }

   if (UseTable && MaxLength > LengthNeeded) {
      // use recursion table method
      p1 = p2 = table + 1;             // make space for p1[-1]
      p1[-1] = 0.;  p1[0] = 1.;        // initialize for recursion
      x1 = x2 = 0;
      for (nu = 1; nu <= n; nu++) {
         if (n - nu < xmin - x1 || p1[x1] < cutoff) {
            x1++;                      // increase lower limit when breakpoint passed or probability negligible
            p2--;                      // compensate buffer offset in order to reduce storage space
         }
         if (x2 < xmax && p1[x2] >= cutoff) {
            x2++;  y1 = 0.;            // increase upper limit until x has been reached
         }
         else {
            y1 = p1[x2];
         }
         if (p2 - table + x2 >= MaxLength || x1 > x2) {
            goto ONE_BY_ONE;           // Error: table length exceeded. Use other method
         }

         mxo = (m-x2)*omega;
         Nmnx = N-m-nu+x2+1;
         for (xi = x2; xi >= x1; xi--) { // backwards loop
            d2 = mxo + Nmnx;
            mxo += omega; Nmnx--;
            d1 = mxo + Nmnx;
            dcom = 1. / (d1 * d2);     // save a division by making common divisor
            y  = p1[xi-1]*mxo*d2*dcom + y1*(Nmnx+1)*d1*dcom;
            y1 = p1[xi-1];             // (warning: pointer alias, can't swap instruction order)
            p2[xi] = y;
         }
         p1 = p2;
      }

      // return results
      i1 = i2 = x2 - x1 + 1;              // desired table length
      if (i2 > MaxLength) i2 = MaxLength; // limit table length
      *xfirst = x1;  *xlast = x1 + i2 - 1;
      if (i2 > 0) memmove(table, table+1, i2*sizeof(table[0]));// copy to start of table
      return i1 == i2;                    // true if table size not reduced
   }

   else {
      // Recursion method would take too much time
      // Calculate values one by one
      ONE_BY_ONE:

      // Start to fill table from the end and down. start with x = floor(mean)
      x2 = (int32_t)mean();
      x1 = x2 + 1;  i1 = MaxLength;
      while (x1 > xmin) {              // loop for left tail
         x1--;  i1--;
         y = probability(x1);
         table[i1] = y;
         if (y < cutoff) break;
         if (i1 == 0) break;
      }
      *xfirst = x1;
      i2 = x2 - x1 + 1; 
      if (i1 > 0 && i2 > 0) { // move numbers down to beginning of table
         memmove(table, table+i1, i2*sizeof(table[0]));
      }
      // Fill rest of table from mean and up
      i2--;
      while (x2 < xmax) {              // loop for right tail
         if (i2 == MaxLength-1) {
            *xlast = x2; return 0;     // table full
         }
         x2++;  i2++;
         y = probability(x2);
         table[i2] = y;
         if (y < cutoff) break;
      }
      *xlast = x2;
      return 1;
   }
}





/***********************************************************************
Methods for class CFishersNCHypergeometric
***********************************************************************/

CFishersNCHypergeometric::CFishersNCHypergeometric(int32_t n, int32_t m, int32_t N, double odds, double accuracy) {
   // constructor
   // set parameters
   this->n = n;  this->m = m;  this->N = N;
   this->odds = odds;  this->accuracy = accuracy;

   // check validity of parameters
   if (n < 0 || m < 0 || N < 0 || odds < 0. || n > N || m > N) {
      FatalError("Parameter out of range in class CFishersNCHypergeometric");
   }
   if (accuracy < 0) accuracy = 0;
   if (accuracy > 1) accuracy = 1;
   // initialize
   logodds = log(odds);  scale = rsum = 0.;
   ParametersChanged = 1;
   // calculate xmin and xmax
   xmin = m + n - N;  if (xmin < 0) xmin = 0;
   xmax = n;  if (xmax > m) xmax = m;
}


int32_t CFishersNCHypergeometric::mode(void) {
   // Find mode (exact)
   // Uses the method of Liao and Rosen, The American Statistician, vol 55,
   // no 4, 2001, p. 366-369.
   // Note that there is an error in Liao and Rosen's formula. 
   // Replace sgn(b) with -1 in Liao and Rosen's formula. 

   double A, B, C, D;                  // coefficients for quadratic equation
   double x;                           // mode
   int32_t L = m + n - N;
   int32_t m1 = m+1, n1 = n+1;

   if (odds == 1.) { 
      // simple hypergeometric
      x = (m + 1.) * (n + 1.) / (N + 2.);
   }
   else {
      // calculate analogously to Cornfield mean
      A = 1. - odds;
      B = (m1+n1)*odds - L; 
      C = -(double)m1*n1*odds;
      D = B*B -4*A*C;
      D = D > 0. ? sqrt(D) : 0.;
      x = (D - B)/(A+A);
   }
   return (int32_t)x;
}


double CFishersNCHypergeometric::mean(void) {
   // Find approximate mean
   // Calculation analogous with mode
   double a, b;                        // temporaries in calculation
   double mean;                        // mean

   if (odds == 1.) {                   // simple hypergeometric
      return double(m)*n/N;
   }
   // calculate Cornfield mean
   a = (m+n)*odds + (N-m-n); 
   b = a*a - 4.*odds*(odds-1.)*m*n;
   b = b > 0. ? sqrt(b) : 0.;
   mean = (a-b)/(2.*(odds-1.));
   return mean;
}


double CFishersNCHypergeometric::variance(void) {
   // find approximate variance (poor approximation)    
   double my = mean(); // approximate mean
   // find approximate variance from Fisher's noncentral hypergeometric approximation
   double r1 = my * (m-my); double r2 = (n-my)*(my+N-n-m);
   if (r1 <= 0. || r2 <= 0.) return 0.;
   double var = N*r1*r2/((N-1)*(m*r2+(N-m)*r1));
   if (var < 0.) var = 0.;
   return var;
}

double CFishersNCHypergeometric::moments(double * mean_, double * var_) {
   // calculate exact mean and variance
   // return value = sum of f(x), expected = 1.
   double y, sy=0, sxy=0, sxxy=0, me1;
   int32_t x, xm, x1;
   const double accur = 0.1 * accuracy;     // accuracy of calculation
   xm = (int32_t)mean();                      // approximation to mean
   for (x=xm; x<=xmax; x++) {
      y = probability(x);
      x1 = x - xm;  // subtract approximate mean to avoid loss of precision in sums
      sy += y; sxy += x1 * y; sxxy += x1 * x1 * y;
      if (y < accur && x != xm) break;
   }
   for (x=xm-1; x>=xmin; x--) {
      y = probability(x);
      x1 = x - xm;  // subtract approximate mean to avoid loss of precision in sums
      sy += y; sxy += x1 * y; sxxy += x1 * x1 * y;
      if (y < accur) break;
   }
   me1 = sxy / sy;
   *mean_ = me1 + xm;
   y = sxxy / sy - me1 * me1;
   if (y < 0) y=0;
   *var_ = y;
   return sy;
}


double CFishersNCHypergeometric::probability(int32_t x) {
   // calculate probability function
   const double accur = accuracy * 0.1;// accuracy of calculation

   if (x < xmin || x > xmax) return 0;

   if (n == 0) return 1.;

   if (odds == 1.) {
      // central hypergeometric
      return exp(
         LnFac(m)   - LnFac(x)   - LnFac(m-x) +
         LnFac(N-m) - LnFac(n-x) - LnFac((N-m)-(n-x)) -
        (LnFac(N)   - LnFac(n)   - LnFac(N-n)));
   }

   if (odds == 0.) {
      if (n > N-m) FatalError("Not enough items with nonzero weight in CFishersNCHypergeometric::probability");
      return x == 0;
   }

   if (!rsum) {
      // first time. calculate rsum = reciprocal of sum of proportional 
      // function over all probable x values
      int32_t x1, x2;                    // x loop
      double y;                        // value of proportional function
      x1 = (int32_t)mean();              // start at mean
      if (x1 < xmin) x1 = xmin;
      x2 = x1 + 1;
      scale = 0.; scale = lng(x1);     // calculate scale to avoid overflow
      rsum = 1.;                       // = exp(lng(x1)) with this scale
      for (x1--; x1 >= xmin; x1--) {
         rsum += y = exp(lng(x1));     // sum from x1 and down 
         if (y < accur) break;         // until value becomes negligible
      }
      for (; x2 <= xmax; x2++) {       // sum from x2 and up
         rsum += y = exp(lng(x2));
         if (y < accur) break;         // until value becomes negligible
      }
      rsum = 1. / rsum;                // save reciprocal sum
   }
   return exp(lng(x)) * rsum;          // function value
}

double CFishersNCHypergeometric::probabilityRatio(int32_t x, int32_t x0) {
   // Calculate probability ratio f(x)/f(x0)
   // This is much faster than calculating a single probability because
   // rsum is not needed
   double a1, a2, a3, a4, f1, f2, f3, f4;
   int32_t y, dx = x - x0;
   int invert = 0;

   if (x < xmin || x > xmax) return 0.;
   if (x0 < xmin || x0 > xmax) {
      FatalError("Infinity in CFishersNCHypergeometric::probabilityRatio");
   }
   if (dx == 0.) return 1.;
   if (dx < 0.) {
      invert = 1;
      dx = -dx;
      y = x;  x = x0;  x0 = y;
   }
   a1 = m - x0;  a2 = n - x0;  a3 = x;  a4 = N - m - n + x;
   if (dx <= 28 && x <= 100000) {      // avoid overflow
      // direct calculation
      f1 = f2 = 1.;
      // compute ratio of binomials
      for (y = 0; y < dx; y++) {
         f1 *= a1-- * a2--;
         f2 *= a3-- * a4--;
      }
      // compute odds^dx
      f3 = 1.;  f4 = odds;  y = dx;
      while (y) {
         if (f4 < 1.E-100) {
            f3 = 0.;  break;           // avoid underflow
         }
         if (y & 1) f3 *= f4;
         f4 *= f4;
         y = (unsigned long)(y) >> 1;
      }
      f1 = f3 * f1 / f2;
      if (invert) f1 = 1. / f1;
   }
   else {
      // use logarithms
      f1 = FallingFactorial(a1,dx) + FallingFactorial(a2,dx) -
           FallingFactorial(a3,dx) - FallingFactorial(a4,dx) +
           dx * log(odds);
      if (invert) f1 = -f1;
      f1 = exp(f1);
   }
   return f1;
}

double CFishersNCHypergeometric::MakeTable(double * table, int32_t MaxLength, int32_t * xfirst, int32_t * xlast, double cutoff) {
   // Makes a table of Fisher's noncentral hypergeometric probabilities.
   // Results are returned in the array table of size MaxLength.
   // The values are scaled so that the highest value is 1. The return value
   // is the sum, s, of all the values in the table. The normalized
   // probabilities are obtained by multiplying all values in the table by
   // 1/s.
   // The tails are cut off where the values are < cutoff, so that 
   // *xfirst may be > xmin and *xlast may be < xmax.
   // The value of cutoff will be 0.01 * accuracy if not specified.
   // The first and last x value represented in the table are returned in 
   // *xfirst and *xlast. The resulting probability values are returned in the 
   // first (*xlast - *xfirst + 1) positions of table. If this would require
   // more than MaxLength values then the table is filled with as many 
   // correct values as possible.
   //
   // The function will return the desired length of table when MaxLength = 0.

   double f;                           // probability function value
   double sum;                         // sum of table values
   double a1, a2, b1, b2;              // factors in recursive calculation of f(x)
   int32_t x;                          // x value
   int32_t x1, x2;                     // lowest and highest x
   int32_t i, i0, i1, i2;              // table index
   int32_t mode = this->mode();        // mode
   int32_t L = n + m - N;              // parameter
   int32_t DesiredLength;              // desired length of table

   // limits for x
   x1 = (L > 0) ? L : 0;               // xmin
   x2 = (n < m) ? n : m;               // xmax

   // special cases
   if (x1 == x2) goto DETERMINISTIC;
   if (odds <= 0.) {
      if (n > N-m) FatalError("Not enough items with nonzero weight in  CWalleniusNCHypergeometric::MakeTable");
      x1 = 0;
      DETERMINISTIC:
      if (MaxLength == 0) {
         if (xfirst) *xfirst = 1;
         return 1;
      }
      *xfirst = *xlast = x1;
      *table = 1.;
      return 1;
   }

   if (MaxLength <= 0) {
      // Return UseTable and LengthNeeded
      DesiredLength = x2 - x1 + 1;     // max length of table
      if (DesiredLength > 200) {
         double sd = sqrt(variance()); // calculate approximate standard deviation
         // estimate number of standard deviations to include from normal distribution
         i = (int32_t)(NumSD(accuracy) * sd + 0.5);
         if (DesiredLength > i) DesiredLength = i;
      }
      if (xfirst) *xfirst = 1;         // for analogy with CWalleniusNCHypergeometric::MakeTable
      return DesiredLength;
   }

   // place mode in the table
   if (mode - x1 <= MaxLength/2) {
      // There is enough space for left tail
      i0 = mode - x1;
   }
   else if (x2 - mode <= MaxLength/2) {
      // There is enough space for right tail
      i0 = MaxLength - x2 + mode - 1;
      if (i0 < 0) i0 = 0;
   }
   else {
      // There is not enough space for any of the tails. Place mode in middle of table
      i0 = MaxLength/2;
   }
   // Table start index
   i1 = i0 - mode + x1;  if (i1 < 0) i1 = 0;

   // Table end index
   i2 = i0 + x2 - mode;  if (i2 > MaxLength-1) i2 = MaxLength-1;

   // make center
   table[i0] = sum = f = 1.;

   // make left tail
   x = mode;
   a1 = m + 1 - x;  a2 = n + 1 - x;
   b1 = x;  b2 = x - L;
   for (i = i0 - 1; i >= i1; i--) {
      f *= b1 * b2 / (a1 * a2 * odds); // recursive formula
      a1++;  a2++;  b1--;  b2--;
      sum += table[i] = f;
      if (f < cutoff) {
         i1 = i;  break;               // cut off tail if < accuracy
      }
   }
   if (i1 > 0) {
      // move table down for cut-off left tail
      memcpy(table, table+i1, (i0-i1+1)*sizeof(*table));
      // adjust indices
      i0 -= i1;  i2 -= i1;  i1 = 0;
   }
   // make right tail
   x = mode + 1;
   a1 = m + 1 - x;  a2 = n + 1 - x;
   b1 = x;  b2 = x - L;
   f = 1.;
   for (i = i0 + 1; i <= i2; i++) {
      f *= a1 * a2 * odds / (b1 * b2); // recursive formula
      a1--;  a2--;  b1++;  b2++;
      sum += table[i] = f;
      if (f < cutoff) {
         i2 = i;  break;               // cut off tail if < accuracy
      }
   }
   // x limits
   *xfirst = mode - (i0 - i1);
   *xlast  = mode + (i2 - i0);

   return sum;
}


double CFishersNCHypergeometric::lng(int32_t x) {
   // natural log of proportional function
   // returns lambda = log(m!*x!/(m-x)!*m2!*x2!/(m2-x2)!*odds^x)
   int32_t x2 = n - x,  m2 = N - m;
   if (ParametersChanged) {
      mFac = LnFac(m) + LnFac(m2);
      xLast = -99; ParametersChanged = 0;
   }
   if (m < FAK_LEN && m2 < FAK_LEN)  goto DEFLT;
   switch (x - xLast) {
  case 0:   // x unchanged
     break;
  case 1:   // x incremented. calculate from previous value
     xFac += log (double(x) * (m2-x2) / (double(x2+1)*(m-x+1)));
     break;
  case -1:  // x decremented. calculate from previous value
     xFac += log (double(x2) * (m-x) / (double(x+1)*(m2-x2+1)));
     break;
  default: DEFLT: // calculate all
     xFac = LnFac(x) + LnFac(x2) + LnFac(m-x) + LnFac(m2-x2);
   }
   xLast = x;
   return mFac - xFac + x * logodds - scale;
}



/******************************************************************************
Methods for class StochasticLib3
******************************************************************************/


/***********************************************************************
Constructor
***********************************************************************/
StochasticLib3::StochasticLib3(int seed) : StochasticLib1(seed) {
   SetAccuracy(1.E-8);                  // set default accuracy
   // Initialize variables
   fnc_n_last = -1, fnc_m_last = -1, fnc_N_last = -1;
   fnc_o_last = -1.;
   wnc_n_last = -1, wnc_m_last = -1, wnc_N_last = -1;
   wnc_o_last = -1.;
}


/***********************************************************************
SetAccuracy
***********************************************************************/
void StochasticLib3::SetAccuracy(double accur) {
   // define accuracy of calculations for 
   // WalleniusNCHyp and MultiWalleniusNCHyp
   if (accur < 0.) accur = 0.;
   if (accur > 0.01) accur = 0.01;
   accuracy = accur;
}


/***********************************************************************
Wallenius Non-central Hypergeometric distribution
***********************************************************************/

int32_t StochasticLib3::WalleniusNCHyp (int32_t n, int32_t m, int32_t N, double odds) {
   /*
   This function generates a random variate with Wallenius noncentral 
   hypergeometric distribution.

   Wallenius noncentral hypergeometric distribution is the distribution you 
   get when drawing balls without replacement from an urn containing red and
   white balls, with bias.

   We define the weight of the balls so that the probability of taking a
   particular ball is proportional to its weight. The value of odds is the
   normalized odds ratio: odds = weight(red) / weight(white).
   If all balls have the same weight, i.e. odds = 1, then we get the
   hypergeometric distribution.

   n is the number of balls you take,
   m is the number of red balls in the urn,
   N is the total number of balls in the urn, 
   odds is the odds ratio,
   and the return value is the number of red balls you get.

   Four different calculation methods are implemented. This function decides
   which method to use, based on the parameters. 
   */   

   // check parameters
   if (n >= N || m >= N || n <= 0 || m <= 0 || odds <= 0.) {
      // trivial cases
      if (n == 0 || m == 0) return 0;
      if (m == N) return n;
      if (n == N) return m;
      if (odds == 0.) {
         if (n > N-m) FatalError("Not enough items with nonzero weight in function WalleniusNCHyp");
         return 0;}
      // illegal parameter    
      FatalError("Parameter out of range in function WalleniusNCHyp");}

   if (odds == 1.) {
      // use hypergeometric function if odds == 1
      return Hypergeometric(n, m, N);}

   if (n < 30) {
      return WalleniusNCHypUrn(n, m, N, odds);}

   if (double(n)*N < 10000) {
      return WalleniusNCHypTable(n, m, N, odds);}

   return WalleniusNCHypRatioOfUnifoms(n, m, N, odds);
   // the decision to use NoncentralHypergeometricInversion is
   // taken inside WalleniusNCHypRatioOfUnifoms based
   // on the calculated variance.
}


/***********************************************************************
Subfunctions for WalleniusNCHyp
***********************************************************************/

int32_t StochasticLib3::WalleniusNCHypUrn (int32_t n, int32_t m, int32_t N, double odds) {
   // sampling from Wallenius noncentral hypergeometric distribution 
   // by simulating urn model
   int32_t x;                           // sample
   int32_t m2;                          // items of color 2 in urn
   double mw1, mw2;                     // total weight of balls of color 1 or 2
   x = 0;  m2 = N - m;
   mw1 = m * odds;  mw2 = m2;
   do {
      if (Random() * (mw1 + mw2) < mw1) {
         x++;  m--;
         if (m == 0) break;
         mw1 = m * odds;
      }
      else {
         m2--;
         if (m2 == 0) {
            x += n-1; break;
         }
         mw2 = m2;
      }
   }
   while (--n);
   return x;
}


int32_t StochasticLib3::WalleniusNCHypTable (int32_t n, int32_t m, int32_t N, double odds) {
   // Sampling from Wallenius noncentral hypergeometric distribution 
   // using chop-down search from a table created by recursive calculation.
   // This method is fast when n is low or when called repeatedly with
   // the same parameters.

   int32_t x2;                          // upper x limit for table
   int32_t x;                           // sample
   double u;                            // uniform random number
   int success;                         // table long enough

   if (n != wnc_n_last || m != wnc_m_last || N != wnc_N_last || odds != wnc_o_last) {
      // set-up: This is done only when parameters have changed
      wnc_n_last = n;  wnc_m_last = m;  wnc_N_last = N;  wnc_o_last = odds;

      CWalleniusNCHypergeometric wnch(n,m,N,odds);   // make object for calculation
      success = wnch.MakeTable(wall_ytable, WALL_TABLELENGTH, &wall_x1, &x2); // make table of probability values
      if (success) {
         wall_tablen = x2 - wall_x1 + 1;         // table long enough. remember length
      }
      else {
         wall_tablen = 0;                        // remember failure
      }
   }

   if (wall_tablen == 0) {
      // table not long enough. Use another method
      return WalleniusNCHypRatioOfUnifoms(n,m,N,odds);
   }

   while (1) {                                   // repeat in the rare case of failure
      u = Random();                              // uniform variate to convert
      for (x=0; x<wall_tablen; x++) {            // chop-down search
         u -= wall_ytable[x];
         if (u < 0.) return x + wall_x1;         // value found
      }
   }
}


int32_t StochasticLib3::WalleniusNCHypRatioOfUnifoms (int32_t n, int32_t m, int32_t N, double odds) {
   // sampling from Wallenius noncentral hypergeometric distribution 
   // using ratio-of-uniforms rejection method.
   int32_t xmin, xmax;                  // x limits
   double mean;                         // mean
   double variance;                     // variance
   double x;                            // real sample
   int32_t xi;                          // integer sample
   int32_t x2;                          // limit when searching for mode
   double u;                            // uniform random
   double f, f2;                        // probability function value
   double s123;                         // components 1,2,3 of hat width
   double s4;                           // component 4 of hat width
   double r1, r2;                       // temporaries
   static const double rsqrt2pi = 0.3989422804014326857; // 1/sqrt(2*pi)

   // Make object for calculating mean and probability.
   CWalleniusNCHypergeometric wnch(n, m, N, odds, accuracy);

   xmin = m+n-N; if (xmin < 0) xmin = 0;  // calculate limits
   xmax = n;     if (xmax > m) xmax = m;

   if (n != wnc_n_last || m != wnc_m_last || N != wnc_N_last || odds != wnc_o_last) {
      // set-up: This is done only when parameters have changed
      wnc_n_last = n;  wnc_m_last = m;  wnc_N_last = N;  wnc_o_last = odds;

      // find approximate mean
      mean = wnch.mean();

      // find approximate variance from Fisher's noncentral hypergeometric approximation
      r1 = mean * (m-mean); r2 = (n-mean)*(mean+N-n-m);
      variance = N*r1*r2/((N-1)*(m*r2+(N-m)*r1));
      UseChopDown = variance < 4.;       // use chop-down method if variance is low

      if (!UseChopDown) {
         // find mode (same code in CWalleniusNCHypergeometric::mode)
         wnc_mode = (int32_t)(mean);  f2 = 0.;
         if (odds < 1.) {
            if (wnc_mode < xmax) wnc_mode++;
            x2 = xmin;
            if (odds > 0.294 && N <= 10000000) {
               x2 = wnc_mode - 1;}                    // search for mode can be limited
            for (xi = wnc_mode; xi >= x2; xi--) {
               f = wnch.probability(xi);
               if (f <= f2) break;
               wnc_mode = xi; f2 = f;
            }
         }
         else {
            if (wnc_mode < xmin) wnc_mode++; 
            x2 = xmax;
            if (odds < 3.4 && N <= 10000000) {
               x2 = wnc_mode + 1;}                    // search for mode can be limited
            for (xi = wnc_mode; xi <= x2; xi++) {
               f = wnch.probability(xi);
               if (f <= f2) break;
               wnc_mode = xi; f2 = f;
            }
         }
         wnc_k = f2;                                // value at mode

         // find approximate variance from normal distribution approximation
         variance = rsqrt2pi / wnc_k;  variance *= variance;

         // find center and width of hat function
         wnc_a = mean + 0.5;
         s123 = 0.40 + 0.8579*sqrt(variance+0.5) + 0.4*fabs(mean-wnc_mode);
         s4 = 0.;
         r1 = xmax - mean - s123;  r2 = mean - s123 - xmin;
         if (r1 > r2) r1 = r2;
         if ((odds>5. || odds<0.2) && r1>=-0.5 && r1<=8.) {
            // s4 correction needed
            if (r1 < 1.) r1 = 1.;
            s4 = 0.029 * pow(double(N),0.23) / (r1*r1);
         }
         wnc_h = 2. * (s123 + s4);

         // find safety bounds
         wnc_bound1 = (int32_t)(mean - 4. * wnc_h);
         if (wnc_bound1 < xmin) wnc_bound1 = xmin;
         wnc_bound2 = (int32_t)(mean + 4. * wnc_h);
         if (wnc_bound2 > xmax) wnc_bound2 = xmax;
      }
   }

   if (UseChopDown) { // for small variance, use chop down inversion
      return WalleniusNCHypInversion(n,m,N,odds);
   }

   // use ratio-of-uniforms rejection method
   while(1) {                                    // rejection loop
      u = Random();
      if (u == 0.) continue;                     // avoid division by 0
      x = wnc_a + wnc_h * (Random()-0.5)/u;
      if (x < 0. || x > 2E9) continue;           // reject, avoid overflow
      xi = (int32_t)(x);                         // truncate
      if (xi < wnc_bound1 || xi > wnc_bound2) {
         continue;                               // reject if outside safety bounds
      }
#if 0 // use rejection in x-domain
      if (xi == wnc_mode) break;                 // accept      
      f = wnch.probability(xi);                  // function value
      if (f > wnc_k * u * u) {
         break;                                  // acceptance
      }
#else // use rejection in t-domain (this is faster)
      double hx, s2, xma2;                       // compute h(x)
      s2 = wnc_h * 0.5;  s2 *= s2; 
      xma2 = xi - (wnc_a-0.5);
      xma2 *= xma2;
      hx = (s2 >= xma2) ? 1. : s2 / xma2;
      // rejection in t-domain implemented in CWalleniusNCHypergeometric::BernouilliH
      if (wnch.BernouilliH(xi, hx * wnc_k * 1.01, u * u * wnc_k  * 1.01, this)) {
         break;                                  // acceptance
      }
#endif      
   }                                             // rejection
   return xi;
}


int32_t StochasticLib3::WalleniusNCHypInversion (int32_t n, int32_t m, int32_t N, double odds) {
   // sampling from Wallenius noncentral hypergeometric distribution 
   // using down-up search starting at the mean using the chop-down technique.
   // This method is faster than the rejection method when the variance is low.
   int32_t wall_x1, x2;                          // search values
   int32_t xmin, xmax;                           // x limits
   double   u;                                   // uniform random number to be converted
   double   f;                                   // probability function value
   double   accura;                              // absolute accuracy
   int      updown;                              // 1 = search down, 2 = search up, 3 = both

   // Make objects for calculating mean and probability.
   // It is more efficient to have two identical objects, one for down search
   // and one for up search, because they are obtimized for consecutive x values.
   CWalleniusNCHypergeometric wnch1(n, m, N, odds, accuracy);
   CWalleniusNCHypergeometric wnch2(n, m, N, odds, accuracy);

   accura = accuracy * 0.01;
   if (accura > 1E-7) accura = 1E-7;             // absolute accuracy

   wall_x1 = (int32_t)(wnch1.mean());            // start at floor and ceiling of mean
   x2 = wall_x1 + 1;
   xmin = m+n-N; if (xmin<0) xmin = 0;           // calculate limits
   xmax = n;     if (xmax>m) xmax = m;
   updown = 3;                                   // start searching both up and down

   while(1) {                                    // loop until accepted (normally executes only once)
      u = Random();                              // uniform random number to be converted
      while (updown) {                           // search loop
         if (updown & 1) {                       // search down
            if (wall_x1 < xmin) {
               updown &= ~1;}                    // stop searching down
            else {
               f = wnch1.probability(wall_x1);
               u -= f;                           // subtract probability until 0
               if (u <= 0.) return wall_x1;
               wall_x1--;
               if (f < accura) updown &= ~1;     // stop searching down
            }
         }
         if (updown & 2) {                       // search up
            if (x2 > xmax) {
               updown &= ~2;                     // stop searching up
            }
            else {
               f = wnch2.probability(x2);
               u -= f;                           // subtract probability until 0
               if (u <= 0.) return x2;
               x2++;
               if (f < accura) updown &= ~2;     // stop searching down
            }
         }
      }
   }
}


/***********************************************************************
Multivariate Wallenius noncentral hypergeometric distribution
***********************************************************************/



/******************************************************************************
Fisher's noncentral hypergeometric distribution
******************************************************************************/
int32_t StochasticLib3::FishersNCHyp (int32_t n, int32_t m, int32_t N, double odds) {
   /*
   This function generates a random variate with Fisher's noncentral
   hypergeometric distribution.

   This distribution resembles Wallenius noncentral hypergeometric distribution
   and the two distributions are sometimes confused. A more detailed 
   explanation of this distribution is given below under the multivariate
   Fisher's noncentral hypergeometric distribution (MultiFishersNCHyp). 
   For further documentation see nchyp.pdf, awailable from www.agner.org/random

   This function uses inversion by chop-down search from zero when parameters
   are small, and the ratio-of-uniforms rejection method when the former 
   method would be too slow or would give overflow.
   */   
   int32_t fak, addd;                  // used for undoing transformations
   int32_t x;                          // result

   // check if parameters are valid
   if (n > N || m > N || n < 0 || m < 0 || odds <= 0.) {
      if (odds == 0.) {
         if (n > N-m) FatalError("Not enough items with nonzero weight in function FishersNCHyp");
         return 0;
      }
      FatalError("Parameter out of range in function FishersNCHyp");}

   if (odds == 1.) {
      // use hypergeometric function if odds == 1
      return Hypergeometric(n, m, N);
   }

   // symmetry transformations
   fak = 1;  addd = 0;
   if (m > N/2) {
      // invert m
      m = N - m;
      fak = -1;  addd = n;
   }

   if (n > N/2) {
      // invert n
      n = N - n;
      addd += fak * m;  fak = - fak;
   }

   if (n > m) {
      // swap n and m
      x = n;  n = m;  m = x;
   }

   // cases with only one possible result end here
   if (n == 0 || odds == 0.) return addd;

   if (fak == -1) {
      // reciprocal odds if inverting
      odds = 1. / odds;
   }

   // choose method
   if (n < 30 && N < 1024 && odds > 1.E-5 && odds < 1.E5) {
      // use inversion by chop down method
      x = FishersNCHypInversion (n, m, N, odds);
   }
   else {
      // use ratio-of-uniforms method
      x = FishersNCHypRatioOfUnifoms (n, m, N, odds);
   }

   // undo symmetry transformations  
   return x * fak + addd;
}

/***********************************************************************
Subfunctions used by FishersNCHyp
***********************************************************************/

int32_t StochasticLib3::FishersNCHypInversion (int32_t n, int32_t m, int32_t N, double odds) {
   /* 
   Subfunction for FishersNCHyp distribution.
   Implements Fisher's noncentral hypergeometric distribution by inversion 
   method, using chop-down search starting at zero.

   Valid only for 0 <= n <= m <= N/2.
   Without overflow check the parameters must be limited to n < 30, N < 1024,
   and 1.E-5 < odds < 1.E5. This limitation is acceptable because this method 
   is slow for higher n.

   The execution time of this function grows with n.

   See the file nchyp.pdf for theoretical explanation.
   */ 
   int32_t x;                          // x value
   int32_t L;                          // derived parameter
   double f;                           // scaled function value 
   double sum;                         // scaled sum of function values
   double a1, a2, b1, b2, f1, f2;      // factors in recursive calculation
   double u;                           // uniform random variate

   L = N - m - n;

   if (n != fnc_n_last || m != fnc_m_last || N != fnc_N_last || odds != fnc_o_last) {
      // parameters have changed. set-up
      fnc_n_last = n; fnc_m_last = m; fnc_N_last = N; fnc_o_last = odds;

      // f(0) is set to an arbitrary value because it cancels out.
      // A low value is chosen to avoid overflow.
      fnc_f0 = 1.E-100;

      // calculate summation of e(x), using the formula:
      // f(x) = f(x-1) * (m-x+1)*(n-x+1)*odds / (x*(L+x))
      // All divisions are avoided by scaling the parameters
      sum = f = fnc_f0;  fnc_scale = 1.;
      a1 = m;  a2 = n;  b1 = 1;  b2 = L + 1;
      for (x = 1; x <= n; x++) {
         f1 = a1 * a2 * odds;
         f2 = b1 * b2;
         a1--;  a2--;  b1++;  b2++;
         f *= f1;
         sum *= f2;
         fnc_scale *= f2;
         sum += f;
         // overflow check. not needed if parameters are limited:
         // if (sum > 1E100) {sum *= 1E-100; f *= 1E-100; fnc_scale *= 1E-100;}
      }
      fnc_f0 *= fnc_scale;
      fnc_scale = sum;
      // now f(0) = fnc_f0 / fnc_scale.
      // We are still avoiding all divisions by saving the scale factor
   }

   // uniform random
   u = Random() * fnc_scale;

   // recursive calculation:
   // f(x) = f(x-1) * (m-x+1)*(n-x+1)*odds / (x*(L+x))
   f = fnc_f0;  x = 0;  a1 = m;  a2 = n;  b1 = 0;  b2 = L;
   do {
      u -= f;
      if (u <= 0) break;
      x++;  b1++;  b2++;
      f *= a1 * a2 * odds;
      u *= b1 * b2;
      // overflow check. not needed if parameters are limited:
      // if (u > 1.E100) {u *= 1E-100;  f *= 1E-100;}
      a1--;  a2--;
   }
   while (x < n);
   return x;
}

int32_t StochasticLib3::FishersNCHypRatioOfUnifoms (int32_t n, int32_t m, int32_t N, double odds) {
   /* 
   Subfunction for FishersNCHyp distribution. 
   Valid for 0 <= n <= m <= N/2, odds != 1

   Fisher's noncentral hypergeometric distribution by ratio-of-uniforms 
   rejection method.

   The execution time of this function is almost independent of the parameters.
   */ 
   int32_t L;                          // N-m-n
   int32_t mode;                       // mode
   double mean;                        // mean
   double variance;                    // variance
   double x;                           // real sample
   int32_t k;                          // integer sample
   double u;                           // uniform random
   double lf;                          // ln(f(x))
   double AA, BB, g1, g2;              // temporary

   L = N - m - n;

   if (n != fnc_n_last || m != fnc_m_last || N != fnc_N_last || odds != fnc_o_last) {
      // parameters have changed. set-up
      fnc_n_last = n;  fnc_m_last = m;  fnc_N_last = N;  fnc_o_last = odds;

      // find approximate mean
      AA = (m+n)*odds+L; BB = sqrt(AA*AA - 4*odds*(odds-1)*m*n);
      mean = (AA-BB)/(2*(odds-1));

      // find approximate variance
      AA = mean * (m-mean); BB = (n-mean)*(mean+L);
      variance = N*AA*BB/((N-1)*(m*BB+(n+L)*AA));

      // compute log(odds)
      fnc_logb = log(odds);

      // find center and width of hat function
      fnc_a = mean + 0.5;
      fnc_h = 1.028 + 1.717*sqrt(variance+0.5) + 0.032*fabs(fnc_logb);

      // find safety bound
      fnc_bound = (int32_t)(mean + 4.0 * fnc_h);
      if (fnc_bound > n) fnc_bound = n;

      // find mode
      mode = (int32_t)(mean);
      g1 =(double)(m-mode)*(n-mode)*odds;
      g2 =(double)(mode+1)*(L+mode+1);
      if (g1 > g2 && mode < n) mode++;

      // value at mode to scale with:
      fnc_lfm = mode * fnc_logb - fc_lnpk(mode, L, m, n);
   }

   while(1) {
      u = Random();
      if (u == 0) continue;                      // avoid divide by 0
      x = fnc_a + fnc_h * (Random()-0.5)/u;
      if (x < 0. || x > 2E9) continue;           // reject, avoid overflow
      k = (int32_t)(x);                          // truncate
      if (k > fnc_bound) continue;               // reject if outside safety bound
      lf = k*fnc_logb - fc_lnpk(k,L,m,n) - fnc_lfm;// compute function value
      if (u * (4.0 - u) - 3.0 <= lf) break;      // lower squeeze accept
      if (u * (u-lf) > 1.0) continue;            // upper squeeze reject
      if (2.0 * log(u) <= lf) break;             // final acceptance
   }
   return k;
}

}  // namespace biasedurn

/* ---------------------------------------------------------------- bernoulli (binom_gen subclass) */

struct Bernoulli : Disc {
    Bernoulli() { name = "bernoulli"; shapes = "p"; nshape = 1; b = 1; doc = "A Bernoulli discrete random variable."; }
    bool argcheck(const double *s) const override { return s[0] >= 0 && s[0] <= 1; }
    double logpmf(double x, const double *s) const override
    {
        /* binom._logpmf(x, 1, p) */
        const double n = 1, p = s[0], k = std::floor(x);
        const double combiln = sc::gammaln(n + 1) - (sc::gammaln(k + 1) + sc::gammaln(n - k + 1));
        return combiln + sc::xlogy(k, p) + sc::xlog1py(n - k, -p);
    }
    double pmf(double x, const double *s) const override { return sc::_binom_pmf(x, 1, s[0]); }
    double cdf(double x, const double *s) const override { return sc::_binom_cdf(std::floor(x), 1, s[0]); }
    double sf(double x, const double *s) const override { return sc::_binom_sf(std::floor(x), 1, s[0]); }
    double isf(double x, const double *s) const override { return sc::_binom_isf(x, 1, s[0]); }
    double ppf(double q, const double *s) const override { return sc::_binom_ppf(q, 1, s[0]); }
    /* _stats(p) takes no `moments`: binom._stats(1, p) runs with its default 'mv' (skew, kurtosis None) */
    void stats(const double *s, int, Stats4 &st) const override
    {
        const double n = 1, p = s[0];
        const double mu = n * p;
        st.set(0, mu);
        st.set(1, mu - n * (p * p));
    }
    double entropy(const double *s) const override { return sc::entr(s[0]) + sc::entr(1 - s[0]); }
    void rvs(bitgen_t *bg, int64_t n, const double *const *sa, double *out) const override
    {
        for (int64_t i = 0; i < n; i++) {
            binomial_t bin = {};
            out[i] = (double)random_binomial(bg, sa[0][i], 1, &bin);
        }
    }
};
Registrar r_bernoulli(new Bernoulli);

/* ---------------------------------------------------------------- betabinom */

struct Betabinom : Disc {
    Betabinom() { name = "betabinom"; shapes = "n, a, b"; nshape = 3; doc = "A beta-binomial discrete random variable."; }
    bool argcheck(const double *s) const override { return s[0] >= 0 && isintegral(s[0]) && s[1] > 0 && s[2] > 0; }
    void get_support(const double *s, double &lo, double &hi) const override { lo = 0; hi = s[0]; }
    double logpmf(double x, const double *s) const override
    {
        const double n = s[0], a_ = s[1], b_ = s[2], k = std::floor(x);
        const double combiln = -std::log(n + 1) - sc::betaln(n - k + 1, k + 1);
        return combiln + sc::betaln(k + a_, n - k + b_) - sc::betaln(a_, b_);
    }
    double pmf(double x, const double *s) const override { return std::exp(logpmf(x, s)); }
    void stats(const double *s, int moments, Stats4 &st) const override
    {
        const double n = s[0], a_ = s[1], b_ = s[2];
        const double e_p = a_ / (a_ + b_);
        const double e_q = 1 - e_p;
        const double mu = n * e_p;
        const double var = n * (a_ + b_ + n) * e_p * e_q / (a_ + b_ + 1);
        st.set(0, mu);
        st.set(1, var);
        if (moments & MOM_S) {
            double g1 = 1.0 / std::sqrt(var);
            g1 *= (a_ + b_ + 2 * n) * (b_ - a_);
            g1 /= (a_ + b_ + 2) * (a_ + b_);
            st.set(2, g1);
        }
        if (moments & MOM_K) {
            double g2 = a_ + b_;
            g2 *= (a_ + b_ - 1 + 6 * n);
            g2 += 3 * a_ * b_ * (n - 2);
            g2 += 6 * (n * n);
            g2 -= 3 * e_p * b_ * n * (6 - n);
            g2 -= 18 * e_p * e_q * (n * n);
            g2 *= sq(a_ + b_) * (1 + a_ + b_);
            g2 /= (n * a_ * b_ * (a_ + b_ + 2) * (a_ + b_ + 3) * (a_ + b_ + n));
            g2 -= 3;
            st.set(3, g2);
        }
    }
    void rvs(bitgen_t *bg, int64_t n, const double *const *sa, double *out) const override
    {
        std::vector<double> p((size_t)n);
        for (int64_t i = 0; i < n; i++) p[(size_t)i] = random_beta(bg, sa[1][i], sa[2][i]);
        for (int64_t i = 0; i < n; i++) {
            binomial_t bin = {};
            out[i] = (double)random_binomial(bg, p[(size_t)i], np_int64(sa[0][i]), &bin);
        }
    }
};
Registrar r_betabinom(new Betabinom);

/* ---------------------------------------------------------------- betanbinom */

struct Betanbinom : Disc {
    Betanbinom() { name = "betanbinom"; shapes = "n, a, b"; nshape = 3; doc = "A beta-negative-binomial discrete random variable."; }
    bool argcheck(const double *s) const override { return s[0] >= 0 && isintegral(s[0]) && s[1] > 0 && s[2] > 0; }
    double logpmf(double x, const double *s) const override
    {
        const double n = s[0], a_ = s[1], b_ = s[2], k = std::floor(x);
        const double combiln = -std::log(n + k) - sc::betaln(n, k + 1);
        return combiln + sc::betaln(a_ + n, b_ + k) - sc::betaln(a_, b_);
    }
    double pmf(double x, const double *s) const override { return std::exp(logpmf(x, s)); }
    void stats(const double *s, int moments, Stats4 &st) const override
    {
        const double n = s[0], a_ = s[1], b_ = s[2];
        st.set(0, a_ > 1 ? n * b_ / (a_ - 1.) : INF);
        st.set(1, a_ > 2 ? (n * b_ * (n + a_ - 1.) * (a_ + b_ - 1.) / ((a_ - 2.) * sq(a_ - 1.))) : INF);
        if (moments & MOM_S)
            st.set(2, a_ > 3 ? ((2 * n + a_ - 1.) * (2 * b_ + a_ - 1.) / (a_ - 3.) /
                                std::sqrt(n * b_ * (n + a_ - 1.) * (b_ + a_ - 1.) / (a_ - 2.)))
                             : INF);
        if (moments & MOM_K) {
            double g2 = INF;
            if (a_ > 4) {
                const double term = (a_ - 2.);
                const double term_2 = (sq(a_ - 1.) * (sq(a_) + a_ * (6 * b_ - 1.) + 6. * (b_ - 1.) * b_) +
                                       3. * sq(n) * ((a_ + 5.) * sq(b_) + (a_ + 5.) * (a_ - 1.) * b_ + 2. * sq(a_ - 1.)) +
                                       3 * (a_ - 1.) * n * ((a_ + 5.) * sq(b_) + (a_ + 5.) * (a_ - 1.) * b_ + 2. * sq(a_ - 1.)));
                const double denominator = ((a_ - 4.) * (a_ - 3.) * b_ * n * (a_ + b_ - 1.) * (a_ + n - 1.));
                g2 = term * term_2 / denominator - 3.;
            }
            st.set(3, g2);
        }
    }
    void rvs(bitgen_t *bg, int64_t n, const double *const *sa, double *out) const override
    {
        std::vector<double> p((size_t)n);
        for (int64_t i = 0; i < n; i++) p[(size_t)i] = random_beta(bg, sa[1][i], sa[2][i]);
        for (int64_t i = 0; i < n; i++) out[i] = (double)random_negative_binomial(bg, sa[0][i], p[(size_t)i]);
    }
};
Registrar r_betanbinom(new Betanbinom);

/* ---------------------------------------------------------------- boltzmann */

struct Boltzmann : Disc {
    Boltzmann() { name = "boltzmann"; shapes = "lambda_, N"; nshape = 2; doc = "A Boltzmann (Truncated Discrete Exponential) random variable."; }
    bool argcheck(const double *s) const override { return s[0] > 0 && s[1] > 0 && isintegral(s[1]); }
    void get_support(const double *s, double &lo, double &hi) const override { lo = a; hi = s[1] - 1; }
    double pmf(double k, const double *s) const override
    {
        const double lambda_ = s[0], N = s[1];
        const double fact = (1 - std::exp(-lambda_)) / (1 - std::exp(-lambda_ * N));
        return fact * std::exp(-lambda_ * k);
    }
    double cdf(double x, const double *s) const override
    {
        const double lambda_ = s[0], N = s[1], k = std::floor(x);
        return (1 - std::exp(-lambda_ * (k + 1))) / (1 - std::exp(-lambda_ * N));
    }
    double ppf(double q, const double *s) const override
    {
        const double lambda_ = s[0], N = s[1];
        const double qnew = q * (1 - std::exp(-lambda_ * N));
        const double vals = std::ceil(-1.0 / lambda_ * std::log(1 - qnew) - 1);
        const double vals1 = std::min(std::max(vals - 1, 0.0), INF);
        const double temp = cdf(vals1, s);
        return temp >= q ? vals1 : vals;
    }
    void stats(const double *s, int, Stats4 &st) const override
    {
        const double lambda_ = s[0], N = s[1];
        const double z = std::exp(-lambda_);
        const double zN = std::exp(-lambda_ * N);
        const double mu = z / (1.0 - z) - N * zN / (1 - zN);
        const double var = z / sq(1.0 - z) - N * N * zN / sq(1 - zN);
        const double trm = (1 - zN) / (1 - z);
        const double trm2 = (z * sq(trm) - N * N * zN);
        double g1 = z * (1 + z) * std::pow(trm, 3.0) - std::pow(N, 3.0) * zN * (1 + zN);
        g1 = g1 / std::pow(trm2, 1.5);
        double g2 = z * (1 + 4 * z + z * z) * std::pow(trm, 4.0) - std::pow(N, 4.0) * zN * (1 + 4 * zN + zN * zN);
        g2 = g2 / trm2 / trm2;
        st.set(mu, var, g1, g2);
    }
};
Registrar r_boltzmann(new Boltzmann);

/* ---------------------------------------------------------------- dlaplace */

struct Dlaplace : Disc {
    Dlaplace() { name = "dlaplace"; shapes = "a"; nshape = 1; a = -INF; doc = "A  Laplacian discrete random variable."; }
    double pmf(double k, const double *s) const override { return std::tanh(s[0] / 2.0) * std::exp(-s[0] * std::fabs(k)); }
    double cdf(double x, const double *s) const override
    {
        const double k = std::floor(x), a_ = s[0];
        if (k >= 0) return 1.0 - std::exp(-a_ * k) / (std::exp(a_) + 1);
        return std::exp(a_ * (k + 1)) / (std::exp(a_) + 1);
    }
    double ppf(double q, const double *s) const override
    {
        const double a_ = s[0];
        const double c = 1 + std::exp(a_);
        const double vals = std::ceil(q < 1.0 / (1 + std::exp(-a_)) ? std::log(q * c) / a_ - 1 : -std::log((1 - q) * c) / a_);
        const double vals1 = vals - 1;
        return cdf(vals1, s) >= q ? vals1 : vals;
    }
    void stats(const double *s, int, Stats4 &st) const override
    {
        const double ea = std::exp(s[0]);
        const double mu2 = 2. * ea / sq(ea - 1.);
        const double mu4 = 2. * ea * (sq(ea) + 10. * ea + 1.) / std::pow(ea - 1., 4.0);
        st.set(0., mu2, 0., mu4 / sq(mu2) - 3.);
    }
    double entropy(const double *s) const override { return s[0] / std::sinh(s[0]) - std::log(std::tanh(s[0] / 2.0)); }
    void rvs(bitgen_t *bg, int64_t n, const double *const *sa, double *out) const override
    {
        std::vector<int64_t> x((size_t)n);
        for (int64_t i = 0; i < n; i++) x[(size_t)i] = random_geometric(bg, -std::expm1(-sa[0][i]));
        for (int64_t i = 0; i < n; i++) out[i] = (double)(x[(size_t)i] - random_geometric(bg, -std::expm1(-sa[0][i])));
    }
};
Registrar r_dlaplace(new Dlaplace);

/* ---------------------------------------------------------------- geom */

struct Geom : Disc {
    Geom() { name = "geom"; shapes = "p"; nshape = 1; a = 1; doc = "A geometric discrete random variable."; }
    bool argcheck(const double *s) const override { return s[0] <= 1 && s[0] > 0; }
    double pmf(double k, const double *s) const override { return std::pow(1 - s[0], k - 1) * s[0]; }
    double logpmf(double k, const double *s) const override { return sc::xlog1py(k - 1, -s[0]) + std::log(s[0]); }
    double cdf(double x, const double *s) const override { return -std::expm1(std::log1p(-s[0]) * std::floor(x)); }
    double sf(double x, const double *s) const override { return std::exp(logsf(x, s)); }
    double logsf(double x, const double *s) const override { return std::floor(x) * std::log1p(-s[0]); }
    double ppf(double q, const double *s) const override
    {
        const double vals = std::ceil(std::log1p(-q) / std::log1p(-s[0]));
        const double temp = cdf(vals - 1, s);
        return (temp >= q && vals > 0) ? vals - 1 : vals;
    }
    void stats(const double *s, int, Stats4 &st) const override
    {
        const double p = s[0];
        const double mu = 1.0 / p;
        const double qr = 1.0 - p;
        const double var = qr / p / p;
        const double g1 = (2.0 - p) / std::sqrt(qr);
        const double g2 = (((0.0 * p + 1) * p + -6) * p + 6) / (1.0 - p);   /* np.polyval([1, -6, 6], p) */
        st.set(mu, var, g1, g2);
    }
    double entropy(const double *s) const override { return -std::log(s[0]) - std::log1p(-s[0]) * (1.0 - s[0]) / s[0]; }
    void rvs(bitgen_t *bg, int64_t n, const double *const *sa, double *out) const override
    {
        for (int64_t i = 0; i < n; i++) {
            const int64_t r = random_geometric(bg, sa[0][i]);
            out[i] = (double)(r < 0 ? INT64_MAX : r);
        }
    }
};
Registrar r_geom(new Geom);

/* ---------------------------------------------------------------- hypergeom */

struct Hypergeom : Disc {
    Hypergeom() { name = "hypergeom"; shapes = "M, n, N"; nshape = 3; doc = "A hypergeometric discrete random variable."; }
    bool argcheck(const double *s) const override
    {
        const double M = s[0], n = s[1], N = s[2];
        return M > 0 && n >= 0 && N >= 0 && n <= M && N <= M && isintegral(M) && isintegral(n) && isintegral(N);
    }
    void get_support(const double *s, double &lo, double &hi) const override
    {
        const double M = s[0], n = s[1], N = s[2];
        lo = std::max(N - (M - n), 0.0);
        hi = std::min(n, N);
    }
    double logpmf(double k, const double *s) const override
    {
        const double tot = s[0], good = s[1], N = s[2];
        const double bad = tot - good;
        return (sc::betaln(good + 1, 1) + sc::betaln(bad + 1, 1) + sc::betaln(tot - N + 1, N + 1) -
                sc::betaln(k + 1, good - k + 1) - sc::betaln(N - k + 1, bad - N + k + 1) - sc::betaln(tot + 1, 1));
    }
    double pmf(double k, const double *s) const override { return sc::_hypergeom_pmf(k, s[1], s[2], s[0]); }
    double cdf(double k, const double *s) const override { return sc::_hypergeom_cdf(k, s[1], s[2], s[0]); }
    double sf(double k, const double *s) const override { return sc::_hypergeom_sf(k, s[1], s[2], s[0]); }
    void stats(const double *s, int, Stats4 &st) const override
    {
        const double M = 1. * s[0], n = 1. * s[1], N = 1. * s[2];
        const double m = M - n;
        double g2 = M * (M + 1) - 6. * N * (M - N) - 6. * n * m;
        g2 *= (M - 1) * M * M;
        g2 += 6. * n * N * (M - N) * m * (5. * M - 6);
        g2 /= n * N * (M - N) * m * (M - 2.) * (M - 3.);
        st.set(sc::_hypergeom_mean(n, N, M), sc::_hypergeom_variance(n, N, M), sc::_hypergeom_skewness(n, N, M), g2);
    }
    /* the public pmf / logcdf / logsf (loc 0), which _entropy, _logsf and _logcdf call */
    double pub_pmf(double k, const double *s) const
    {
        double lo, hi;
        get_support(s, lo, hi);
        if (!(k >= lo && k <= hi && std::floor(k) == k)) return 0.0;
        const double p = pmf(k, s);
        return std::isnan(p) ? p : std::min(std::max(p, 0.0), 1.0);
    }
    double pub_logcdf(double k, const double *s) const
    {
        double lo, hi;
        get_support(s, lo, hi);
        if (k >= lo && k < hi) return logcdf(k, s);
        return k >= hi ? 0.0 : -INF;
    }
    double pub_logsf(double k, const double *s) const
    {
        double lo, hi;
        get_support(s, lo, hi);
        if (k >= lo && k < hi) return logsf(k, s);
        return k < lo ? 0.0 : -INF;
    }
    double entropy(const double *s) const override
    {
        const double M = s[0], n = s[1], N = s[2];
        const double start = N - (M - n), stop = std::min(n, N) + 1;   /* np.r_[start:stop] */
        const int64_t m = alloc_count(std::ceil(stop - start));   /* NumPy's MemoryError past the budget / RAM */
        std::vector<double> v((size_t)m);
        for (int64_t i = 0; i < m; i++) v[(size_t)i] = sc::entr(pub_pmf(start + (double)i, s));
        return tsr_psum(v.data(), m);
    }
    double logsf(double quant, const double *s) const override
    {
        const double tot = s[0], good = s[1], draw = s[2];
        if ((quant + 0.5) * (tot + 0.5) < (good - 0.5) * (draw - 0.5))
            return std::log1p(-std::exp(pub_logcdf(quant, s)));
        std::vector<double> v;
        for (double k2 = quant + 1; k2 < draw + 1; k2 += 1) v.push_back(logpmf(k2, s));   /* np.arange(quant + 1, draw + 1) */
        return logsumexp(v);
    }
    double logcdf(double quant, const double *s) const override
    {
        const double tot = s[0], good = s[1], draw = s[2];
        if ((quant + 0.5) * (tot + 0.5) > (good - 0.5) * (draw - 0.5))
            return std::log1p(-std::exp(pub_logsf(quant, s)));
        const int64_t m = alloc_count(std::ceil(quant + 1));   /* np.arange(0, quant + 1) */
        std::vector<double> v((size_t)m);
        for (int64_t i = 0; i < m; i++) v[(size_t)i] = logpmf((double)i, s);
        return logsumexp(v);
    }
    void rvs(bitgen_t *bg, int64_t n, const double *const *sa, double *out) const override
    {
        for (int64_t i = 0; i < n; i++)
            out[i] = (double)random_hypergeometric(bg, np_int64(sa[1][i]), np_int64(sa[0][i] - sa[1][i]), np_int64(sa[2][i]));
    }
};
Registrar r_hypergeom(new Hypergeom);

/* ---------------------------------------------------------------- logser */

struct Logser : Disc {
    Logser() { name = "logser"; shapes = "p"; nshape = 1; a = 1; doc = "A Logarithmic (Log-Series, Series) discrete random variable."; }
    bool argcheck(const double *s) const override { return s[0] > 0 && s[0] < 1; }
    double pmf(double k, const double *s) const override { return -std::pow(s[0], k) * 1.0 / k / sc::log1p(-s[0]); }
    double sf(double k, const double *s) const override
    {
        const double tiny = 1e-100, p = s[0];
        return -sc::betainc(k + 1, tiny, p) * sc::beta(k + 1, tiny) / std::log1p(-p);
    }
    void stats(const double *s, int, Stats4 &st) const override
    {
        const double p = s[0];
        const double r = sc::log1p(-p);
        const double mu = p / (p - 1.0) / r;
        const double mu2p = -p / r / sq(p - 1.0);
        const double var = mu2p - mu * mu;
        const double mu3p = -p / r * (1.0 + p) / std::pow(1.0 - p, 3.0);
        const double mu3 = mu3p - 3 * mu * mu2p + 2 * std::pow(mu, 3.0);
        const double g1 = mu3 / std::pow(var, 1.5);
        const double mu4p = -p / r * (1.0 / sq(p - 1) - 6 * p / std::pow(p - 1, 3.0) + 6 * p * p / std::pow(p - 1, 4.0));
        const double mu4 = mu4p - 4 * mu3p * mu + 6 * mu2p * mu * mu - 3 * std::pow(mu, 4.0);
        const double g2 = mu4 / sq(var) - 3.0;
        st.set(mu, var, g1, g2);
    }
    void rvs(bitgen_t *bg, int64_t n, const double *const *sa, double *out) const override
    {
        for (int64_t i = 0; i < n; i++) out[i] = (double)random_logseries(bg, sa[0][i]);
    }
};
Registrar r_logser(new Logser);

/* ---------------------------------------------------------------- nbinom */

struct Nbinom : Disc {
    Nbinom() { name = "nbinom"; shapes = "n, p"; nshape = 2; doc = "A negative binomial discrete random variable."; }
    bool argcheck(const double *s) const override { return s[0] > 0 && s[1] > 0 && s[1] <= 1; }
    double pmf(double x, const double *s) const override { return sc::_nbinom_pmf(x, s[0], s[1]); }
    double logpmf(double x, const double *s) const override
    {
        const double n = s[0], p = s[1];
        const double coeff = sc::gammaln(n + x) - sc::gammaln(x + 1) - sc::gammaln(n);
        return coeff + n * std::log(p) + sc::xlog1py(x, -p);
    }
    double cdf(double x, const double *s) const override { return sc::_nbinom_cdf(std::floor(x), s[0], s[1]); }
    double logcdf(double x, const double *s) const override
    {
        const double k = std::floor(x), n = s[0], p = s[1];
        const double c = cdf(k, s);
        if (c > 0.5) return std::log1p(-sc::betainc(k + 1, n, 1 - p));
        return std::log(c);
    }
    double sf(double x, const double *s) const override { return sc::_nbinom_sf(std::floor(x), s[0], s[1]); }
    double isf(double x, const double *s) const override { return sc::_nbinom_isf(x, s[0], s[1]); }
    double ppf(double q, const double *s) const override { return sc::_nbinom_ppf(q, s[0], s[1]); }
    void stats(const double *s, int, Stats4 &st) const override
    {
        const double n = s[0], p = s[1];
        st.set(sc::_nbinom_mean(n, p), sc::_nbinom_variance(n, p), sc::_nbinom_skewness(n, p), sc::_nbinom_kurtosis_excess(n, p));
    }
    void rvs(bitgen_t *bg, int64_t n, const double *const *sa, double *out) const override
    {
        for (int64_t i = 0; i < n; i++) out[i] = (double)random_negative_binomial(bg, sa[0][i], sa[1][i]);
    }
};
Registrar r_nbinom(new Nbinom);

/* ---------------------------------------------------------------- nhypergeom */

struct Nhypergeom : Disc {
    Nhypergeom() { name = "nhypergeom"; shapes = "M, n, r"; nshape = 3; doc = "A negative hypergeometric discrete random variable."; }
    bool argcheck(const double *s) const override
    {
        const double M = s[0], n = s[1], r = s[2];
        return n >= 0 && n <= M && r >= 0 && r <= M - n && isintegral(M) && isintegral(n) && isintegral(r);
    }
    void get_support(const double *s, double &lo, double &hi) const override { lo = 0; hi = s[1]; }
    double logpmf(double k, const double *s) const override
    {
        const double M = s[0], n = s[1], r = s[2];
        if (!(r != 0 || k != 0)) return 0.0;
        return (-sc::betaln(k + 1, r) + sc::betaln(k + r, 1) - sc::betaln(n - k + 1, M - r - n + 1) +
                sc::betaln(M - r - k + 1, 1) + sc::betaln(n + 1, M - n + 1) - sc::betaln(M + 1, 1));
    }
    double pmf(double k, const double *s) const override { return std::exp(logpmf(k, s)); }
    void stats(const double *s, int, Stats4 &st) const override
    {
        const double M = 1. * s[0], n = 1. * s[1], r = 1. * s[2];
        st.set(0, r * n / (M - n + 1));
        st.set(1, r * (M + 1) * n / ((M - n + 1) * (M - n + 2)) * (1 - r / (M - n + 1)));
    }
    /* _rvs1: the public cdf over the support, inverted with interp1d(cdf, ks, kind='next', fill_value='extrapolate') */
    void rvs1(bitgen_t *bg, const double *s, int64_t size, double *out) const
    {
        double lo, hi;
        get_support(s, lo, hi);
        std::vector<double> ks, cdfs;
        for (double k = lo; k < hi + 1; k += 1) {
            ks.push_back(k);
            cdfs.push_back(k >= hi ? 1.0 : std::min(std::max(cdf(k, s), 0.0), 1.0));
        }
        std::vector<size_t> idx(ks.size());
        for (size_t i = 0; i < idx.size(); i++) idx[i] = i;
        std::stable_sort(idx.begin(), idx.end(), [&](size_t x, size_t y) { return cdfs[x] < cdfs[y]; });
        std::vector<double> xs(ks.size()), ys(ks.size());
        for (size_t i = 0; i < idx.size(); i++) {
            xs[i] = std::nextafter(cdfs[idx[i]], INF);
            ys[i] = ks[idx[i]];
        }
        const int64_t len = (int64_t)xs.size();
        for (int64_t i = 0; i < size; i++) {
            const double u = random_standard_uniform(bg);
            int64_t j = std::upper_bound(xs.begin(), xs.end(), u) - xs.begin();   /* searchsorted side='right' */
            j = std::min(std::max<int64_t>(j, 0), len - 1);
            out[i] = (double)(int64_t)ys[(size_t)j];
        }
    }
    void rvs(bitgen_t *bg, int64_t n, const double *const *sa, double *out) const override
    {
        double s[3];
        if (same_shapes(n, nshape, sa)) {
            for (int k = 0; k < 3; k++) s[k] = sa[k][0];
            rvs1(bg, s, n, out);
            return;
        }
        for (int64_t i = 0; i < n; i++) {
            for (int k = 0; k < 3; k++) s[k] = sa[k][i];
            rvs1(bg, s, 1, out + i);
        }
    }
};
Registrar r_nhypergeom(new Nhypergeom);

/* ---------------------------------------------------------------- planck */

struct Planck : Disc {
    Planck() { name = "planck"; shapes = "lambda_"; nshape = 1; a = 0; doc = "A Planck discrete exponential random variable."; }
    bool argcheck(const double *s) const override { return s[0] > 0; }
    double pmf(double k, const double *s) const override { return -std::expm1(-s[0]) * std::exp(-s[0] * k); }
    double cdf(double x, const double *s) const override { return -std::expm1(-s[0] * (std::floor(x) + 1)); }
    double sf(double x, const double *s) const override { return std::exp(logsf(x, s)); }
    double logsf(double x, const double *s) const override { return -s[0] * (std::floor(x) + 1); }
    double ppf(double q, const double *s) const override
    {
        const double vals = std::ceil(-1.0 / s[0] * std::log1p(-q) - 1);
        const double vals1 = std::min(std::max(vals - 1, a), b);
        const double temp = cdf(vals1, s);
        return temp >= q ? vals1 : vals;
    }
    void stats(const double *s, int, Stats4 &st) const override
    {
        const double l = s[0];
        st.set(1 / std::expm1(l), std::exp(-l) / sq(std::expm1(-l)), 2 * std::cosh(l / 2.0), 4 + 2 * std::cosh(l));
    }
    double entropy(const double *s) const override
    {
        const double C = -std::expm1(-s[0]);
        return s[0] * std::exp(-s[0]) / C - std::log(C);
    }
    void rvs(bitgen_t *bg, int64_t n, const double *const *sa, double *out) const override
    {
        for (int64_t i = 0; i < n; i++) out[i] = (double)random_geometric(bg, -std::expm1(-sa[0][i])) - 1.0;
    }
};
Registrar r_planck(new Planck);

/* ---------------------------------------------------------------- randint */

struct Randint : Disc {
    Randint() { name = "randint"; shapes = "low, high"; nshape = 2; doc = "A uniform discrete random variable."; }
    bool argcheck(const double *s) const override { return s[1] > s[0] && isintegral(s[0]) && isintegral(s[1]); }
    void get_support(const double *s, double &lo, double &hi) const override { lo = s[0]; hi = s[1] - 1; }
    double pmf(double k, const double *s) const override
    {
        const double low = s[0], high = s[1];
        const double p = 1.0 / ((double)(int64_t)high - low);
        return (k >= low && k < high) ? p : 0.;
    }
    double cdf(double x, const double *s) const override { return (std::floor(x) - s[0] + 1.) / (s[1] - s[0]); }
    double ppf(double q, const double *s) const override
    {
        const double low = s[0], high = s[1];
        const double vals = std::ceil(q * (high - low) + low) - 1;
        const double vals1 = std::min(std::max(vals - 1, low), high);
        const double temp = cdf(vals1, s);
        return temp >= q ? vals1 : vals;
    }
    void stats(const double *s, int, Stats4 &st) const override
    {
        const double m2 = s[1], m1 = s[0];
        const double mu = (m2 + m1 - 1.0) / 2;
        const double d = m2 - m1;
        const double var = (d * d - 1) / 12.0;
        st.set(mu, var, 0.0, -6.0 / 5.0 * (d * d + 1.0) / (d * d - 1.0));
    }
    double entropy(const double *s) const override { return std::log(s[1] - s[0]); }
    /* rng_integers(random_state, low, high, size): Generator.integers(low, high, size) as int64; the
       np.vectorize path (array shapes) draws one scalar per element */
    void rvs(bitgen_t *bg, int64_t n, const double *const *sa, double *out) const override
    {
        std::vector<uint64_t> buf((size_t)std::max<int64_t>(n, 1));
        if (same_shapes(n, nshape, sa)) {
            const int64_t low = (int64_t)sa[0][0], high = (int64_t)sa[1][0] - 1;
            random_bounded_uint64_fill(bg, (uint64_t)low, (uint64_t)high - (uint64_t)low, n, false, buf.data());
            for (int64_t i = 0; i < n; i++) out[i] = (double)(int64_t)buf[(size_t)i];
            return;
        }
        for (int64_t i = 0; i < n; i++) {
            const int64_t low = (int64_t)sa[0][i], high = (int64_t)sa[1][i] - 1;
            random_bounded_uint64_fill(bg, (uint64_t)low, (uint64_t)high - (uint64_t)low, 1, false, buf.data());
            out[i] = (double)(int64_t)buf[0];
        }
    }
};
Registrar r_randint(new Randint);

/* ---------------------------------------------------------------- skellam */

struct Skellam : Disc {
    Skellam() { name = "skellam"; shapes = "mu1, mu2"; nshape = 2; a = -INF; doc = "A  Skellam discrete random variable."; }
    double pmf(double x, const double *s) const override
    {
        const double mu1 = s[0], mu2 = s[1];
        return x < 0 ? sc::_ncx2_pdf(2 * mu2, 2 * (1 - x), 2 * mu1) * 2 : sc::_ncx2_pdf(2 * mu1, 2 * (1 + x), 2 * mu2) * 2;
    }
    double cdf(double x0, const double *s) const override
    {
        const double x = std::floor(x0), mu1 = s[0], mu2 = s[1];
        return x < 0 ? sc::chndtr(2 * mu2, -2 * x, 2 * mu1) : sc::_ncx2_sf(2 * mu1, 2 * (x + 1), 2 * mu2);
    }
    void stats(const double *s, int, Stats4 &st) const override
    {
        const double mean = s[0] - s[1], var = s[0] + s[1];
        st.set(mean, var, mean / std::sqrt(std::pow(var, 3.0)), 1 / var);
    }
    void rvs(bitgen_t *bg, int64_t n, const double *const *sa, double *out) const override
    {
        std::vector<int64_t> x((size_t)n);
        for (int64_t i = 0; i < n; i++) x[(size_t)i] = random_poisson(bg, sa[0][i]);
        for (int64_t i = 0; i < n; i++) out[i] = (double)(x[(size_t)i] - random_poisson(bg, sa[1][i]));
    }
};
Registrar r_skellam(new Skellam);

/* ---------------------------------------------------------------- yulesimon */

struct Yulesimon : Disc {
    Yulesimon() { name = "yulesimon"; shapes = "alpha"; nshape = 1; a = 1; doc = "A Yule-Simon discrete random variable."; }
    bool argcheck(const double *s) const override { return s[0] > 0; }
    double pmf(double x, const double *s) const override { return s[0] * sc::beta(x, s[0] + 1); }
    double logpmf(double x, const double *s) const override { return std::log(s[0]) + sc::betaln(x, s[0] + 1); }
    double cdf(double x, const double *s) const override { return 1 - x * sc::beta(x, s[0] + 1); }
    double sf(double x, const double *s) const override { return x * sc::beta(x, s[0] + 1); }
    double logsf(double x, const double *s) const override { return std::log(x) + sc::betaln(x, s[0] + 1); }
    void stats(const double *s, int, Stats4 &st) const override
    {
        const double alpha = s[0];
        const double mu = alpha <= 1 ? INF : alpha / (alpha - 1);
        double mu2 = alpha > 2 ? sq(alpha) / ((alpha - 2.0) * sq(alpha - 1)) : INF;
        if (alpha <= 1) mu2 = NaN;
        double g1 = alpha > 3 ? std::sqrt(alpha - 2) * sq(alpha + 1) / (alpha * (alpha - 3)) : INF;
        if (alpha <= 2) g1 = NaN;
        double g2 = alpha > 4 ? alpha + 3 + ((11 * std::pow(alpha, 3.0) - 49 * alpha - 22) / (alpha * (alpha - 4) * (alpha - 3))) : INF;
        if (alpha <= 2) g2 = NaN;
        st.set(mu, mu2, g1, g2);
    }
    void rvs(bitgen_t *bg, int64_t n, const double *const *sa, double *out) const override
    {
        std::vector<double> e1((size_t)n);
        for (int64_t i = 0; i < n; i++) e1[(size_t)i] = random_standard_exponential(bg);
        for (int64_t i = 0; i < n; i++) {
            const double e2 = random_standard_exponential(bg);
            out[i] = std::ceil(-e1[(size_t)i] / std::log1p(-std::exp(-e2 / sa[0][i])));
        }
    }
};
Registrar r_yulesimon(new Yulesimon);

/* ---------------------------------------------------------------- zipf */

struct Zipf : Disc {
    Zipf() { name = "zipf"; shapes = "a"; nshape = 1; a = 1; doc = "A Zipf (Zeta) discrete random variable."; }
    bool argcheck(const double *s) const override { return s[0] > 1; }
    double pmf(double k, const double *s) const override { return 1.0 / sc::_zeta(s[0], 1) * std::pow(k, -s[0]); }
    double munp(int n, const double *s) const override
    {
        const double a_ = s[0];
        return a_ > n + 1 ? sc::_zeta(a_ - n, 1) / sc::_zeta(a_, 1) : INF;
    }
    void rvs(bitgen_t *bg, int64_t n, const double *const *sa, double *out) const override
    {
        for (int64_t i = 0; i < n; i++) out[i] = (double)random_zipf(bg, sa[0][i]);
    }
};
Registrar r_zipf(new Zipf);

/* ---------------------------------------------------------------- zipfian */

struct Zipfian : Disc {
    Zipfian() { name = "zipfian"; shapes = "a, n"; nshape = 2; a = 1; doc = "A Zipfian discrete random variable."; }
    bool argcheck(const double *s) const override
    {
        const double n = s[1];
        const double c = std::isnan(n) ? n : std::min(std::max(n, 1.0), 9007199254740992.0);
        return s[0] >= 0 && n == (double)(int64_t)c;
    }
    void get_support(const double *s, double &lo, double &hi) const override { lo = 1; hi = std::floor(s[1]); }
    double pmf(double k0, const double *s) const override
    {
        const double k = std::floor(k0), n = std::floor(s[1]);
        return sc::_normalized_gen_harmonic(k, k, n, s[0]);
    }
    double cdf(double k0, const double *s) const override
    {
        const double k = std::floor(k0), n = std::floor(s[1]);
        return sc::_normalized_gen_harmonic(1, k, n, s[0]);
    }
    double sf(double k0, const double *s) const override
    {
        const double k = std::floor(k0), n = std::floor(s[1]);
        return sc::_normalized_gen_harmonic(k + 1, n, n, s[0]);
    }
    void stats(const double *s, int, Stats4 &st) const override
    {
        const double a_ = s[0], n = std::floor(s[1]);
        const double Hna = sc::_gen_harmonic(n, a_);
        const double Hna1 = sc::_gen_harmonic(n, a_ - 1);
        const double Hna2 = sc::_gen_harmonic(n, a_ - 2);
        const double Hna3 = sc::_gen_harmonic(n, a_ - 3);
        const double Hna4 = sc::_gen_harmonic(n, a_ - 4);
        const double mu1 = Hna1 / Hna;
        const double mu2n = (Hna2 * Hna - std::pow(Hna1, 2.0));
        const double mu2d = std::pow(Hna, 2.0);
        const double mu2 = mu2n / mu2d;
        const double g1 = (Hna3 / Hna - 3 * Hna1 * Hna2 / std::pow(Hna, 2.0) + 2 * std::pow(Hna1, 3.0) / std::pow(Hna, 3.0)) /
                          std::pow(mu2, 3.0 / 2);
        double g2 = (std::pow(Hna, 3.0) * Hna4 - 4 * std::pow(Hna, 2.0) * Hna1 * Hna3 + 6 * Hna * std::pow(Hna1, 2.0) * Hna2 -
                     3 * std::pow(Hna1, 4.0)) / std::pow(mu2n, 2.0);
        g2 -= 3;
        st.set(mu1, mu2, g1, g2);
    }
};
Registrar r_zipfian(new Zipfian);

/* ---------------------------------------------------------------- nchypergeom_fisher, nchypergeom_wallenius */

/* _nchypergeom_gen: the pmf and moments of a BiasedUrn object built per element as dist(N, n, M, odds, 1e-12)
   (C++ arguments: draws, good, total); rvs from a fresh StochasticLib3 per set of scalar shape parameters */

/* _biasedurn.pyx passes M, n and N as C ints: past INT_MAX that conversion is SciPy's OverflowError */
inline int32_t urn_int(double v)
{
    if (v != v) throw std::invalid_argument("cannot convert float NaN to integer");
    if (!(v > -2147483649.0 && v < 2147483648.0)) throw std::overflow_error("value too large to convert to int");
    return (int32_t)v;
}

template <class Urn, bool Fisher> struct NCHypergeom : Disc {
    bool argcheck(const double *s) const override
    {
        auto isint = [](double x) { return !std::isnan(x) && std::fabs(x) < 9.2e18 && (double)(int64_t)x == x; };
        const double M = s[0], n = s[1], N = s[2], odds = s[3];
        return isint(M) && M >= 0 && isint(n) && n >= 0 && isint(N) && N >= 0 && odds > 0 && N <= M && n <= M;
    }
    void get_support(const double *s, double &lo, double &hi) const override
    {
        const double N = s[0], m1 = s[1], n = s[2];   /* follow Wikipedia notation */
        const double m2 = N - m1;
        lo = std::max(0.0, n - m2);
        hi = std::min(n, m1);
    }
    double pmf(double x, const double *s) const override
    {
        if (std::isnan(x) || std::isnan(s[0]) || std::isnan(s[1]) || std::isnan(s[2])) return NaN;
        try {
            Urn urn(urn_int(s[2]), urn_int(s[1]), urn_int(s[0]), s[3], 1e-12);
            return urn.probability((int32_t)x);
        } catch (const std::exception &e) {
            fail(e.what());
            return NaN;
        }
    }
    void stats(const double *s, int moments, Stats4 &st) const override
    {
        if (!(moments & (MOM_M | MOM_V))) return;
        double m = NaN, v = NaN;
        if (!(std::isnan(s[0]) || std::isnan(s[1]) || std::isnan(s[2]))) {
            try {
                Urn urn(urn_int(s[2]), urn_int(s[1]), urn_int(s[0]), s[3], 1e-12);
                urn.moments(&m, &v);
            } catch (const std::exception &e) {
                fail(e.what());
            }
        }
        st.set(0, m);
        st.set(1, v);
    }
    void rvs1(bitgen_t *bg, const double *s, int64_t size, double *out) const
    {
        biasedurn::StochasticLib3 urn(0);
        urn.next_double = &biasedurn::next_double;
        urn.next_normal = &biasedurn::next_normal;
        biasedurn::glob_rng = bg;
        const int32_t n = urn_int(s[2]), m = urn_int(s[1]), N = urn_int(s[0]);
        try {
            for (int64_t i = 0; i < size; i++) out[i] = Fisher ? urn.FishersNCHyp(n, m, N, s[3]) : urn.WalleniusNCHyp(n, m, N, s[3]);
        } catch (const std::exception &e) {
            fail(e.what());
        }
    }
    void rvs(bitgen_t *bg, int64_t n, const double *const *sa, double *out) const override
    {
        double s[4];
        if (same_shapes(n, nshape, sa)) {
            for (int k = 0; k < 4; k++) s[k] = sa[k][0];
            rvs1(bg, s, n, out);
            return;
        }
        for (int64_t i = 0; i < n; i++) {
            for (int k = 0; k < 4; k++) s[k] = sa[k][i];
            rvs1(bg, s, 1, out + i);
        }
    }
};

struct NchypergeomFisher : NCHypergeom<biasedurn::CFishersNCHypergeometric, true> {
    NchypergeomFisher() { name = "nchypergeom_fisher"; shapes = "M, n, N, odds"; nshape = 4; doc = "A Fisher's noncentral hypergeometric discrete random variable."; }
};
Registrar r_nchypergeom_fisher(new NchypergeomFisher);

struct NchypergeomWallenius : NCHypergeom<biasedurn::CWalleniusNCHypergeometric, false> {
    NchypergeomWallenius() { name = "nchypergeom_wallenius"; shapes = "M, n, N, odds"; nshape = 4; doc = "A Wallenius' noncentral hypergeometric discrete random variable."; }
};
Registrar r_nchypergeom_wallenius(new NchypergeomWallenius);

}  // namespace
}  // namespace tsd
