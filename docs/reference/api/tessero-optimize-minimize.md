# Minimize

`Tessero\Optimize\Minimize`

scipy.optimize.minimize for smooth problems of modest dimension, with the
objective written as a plain PHP callable over list<float>.

  BFGS         scipy's algorithm step for step (inverse-Hessian update,
               strong-Wolfe line search from scalar_search_wolfe2)
  L-BFGS-B     limited-memory BFGS with box bounds (projected two-loop
               recursion; converges to the same KKT point as the Fortran code,
               intermediate iterates differ)
  Nelder-Mead  a line-by-line port of scipy's simplex (same iterates)

Gradients default to scipy's 2-point finite differences.

## Methods

### approxGrad

```php
static approxGrad(callable $f, array $x, ?float $f0 = null, ?float $relStep = null): array
```

scipy.optimize.approx_fprime-style forward difference gradient.

### bracket

```php
static bracket(callable $func, float $xa = 0.0, float $xb = 1.0, float $growLimit = 110.0, int $maxiter = 1000): array
```

scipy.optimize.bracket: bracket the minimum of a scalar function downhill from the initial points,
returning [xa, xb, xc, fa, fb, fc, funcalls] with fb <= fa and fb <= fc.

### brent

```php
static brent(callable $fun, ?array $brack = null): float
```

scipy.optimize.brent: Brent's method on a downhill bracket, returning the minimiser (full_output=0).

### brute

```php
static brute(callable $func, array $ranges, int $Ns = 20): array|float
```

scipy.optimize.brute (finish=None): minimise by evaluating func on a regular grid over the given
ranges (Ns points per dimension, inclusive) and returning the grid point with the smallest value.

### checkGrad

```php
static checkGrad(callable $func, callable $grad, array $x0, float $epsilon = 1.4901161193847656E-8): float
```

scipy.optimize.check_grad: the 2-norm of the difference between the analytic gradient and a
forward-difference approximation of it at x0 (absolute step epsilon, as in scipy).

### dot

```php
static dot(array $a, array $b): float
```

### fmin

```php
static fmin(callable $func, array $x0, array $options = []): array
```

scipy.optimize.fmin: minimise a scalar function with the Nelder-Mead simplex and return the minimiser
(the default full_output=False behaviour of scipy.optimize.fmin). Options: xtol, ftol, maxiter, maxfun.

### fminBfgs

```php
static fminBfgs(callable $fun, array $x0, ?callable $fprime = null): array
```

scipy.optimize.fmin_bfgs: minimise with BFGS and return the minimiser xopt.

### fminCg

```php
static fminCg(callable $fun, array $x0, ?callable $fprime = null): array
```

scipy.optimize.fmin_cg: nonlinear conjugate-gradient (Polak-Ribiere+) minimiser.

### fminCobyla

```php
static fminCobyla(callable $func, array $x0, array $cons, float $rhobeg = 1.0, float $rhoend = 0.0001, int $maxfun = 1000): array
```

scipy.optimize.fmin_cobyla: constrained minimisation by linear approximation (Powell's COBYLA). cons is
a list of callables c(x) >= 0 (inequality constraints only, as in scipy). Returns the minimiser x.

### fminLbfgsb

```php
static fminLbfgsb(callable $func, array $x0, ?callable $fprime = null, ?array $bounds = null): array
```

scipy.optimize.fmin_l_bfgs_b: bound-constrained L-BFGS-B minimiser. scipy returns
(x, f, d); here we return the minimiser x only, matching the other fmin_* wrappers.

### fminNcg

```php
static fminNcg(callable $f, array $x0, callable $fprime, ?callable $fhessp = null): array
```

scipy.optimize.fmin_ncg: truncated-Newton (Newton-CG) minimiser; fprime required, hessp optional.

### fminPowell

```php
static fminPowell(callable $func, array $x0, array $options = []): array
```

scipy.optimize.fmin_powell: derivative-free direction-set minimiser.

### fminSlsqp

```php
static fminSlsqp(callable $func, array $x0, array $eqcons = [], array $ieqcons = [], ?array $bounds = null, ?callable $fprime = null): array
```

scipy.optimize.fmin_slsqp: sequential least-squares programming with equality / inequality constraints
and bounds. eqcons / ieqcons are lists of callables c(x) returning a scalar (eq: c==0, ineq: c>=0),
exactly as scipy's fmin_slsqp. Returns the minimiser x.

### fminTnc

```php
static fminTnc(callable $func, array $x0, ?callable $fprime = null, ?array $bounds = null): array
```

scipy.optimize.fmin_tnc: bound-constrained minimisation by a truncated-Newton algorithm. Returns the
minimiser x (scipy returns (x, nfeval, rc); here we return x only, matching the other fmin_* wrappers).

### fminbound

```php
static fminbound(callable $func, float $x1, float $x2, float $xtol = 1.0E-5, int $maxfun = 500): float
```

scipy.optimize.fminbound: bounded scalar minimisation by Brent's method with golden-section fallback - a
port of scipy's _minimize_scalar_bounded, returning the minimiser (the default full_output=0 behaviour).

### golden

```php
static golden(callable $fun, ?array $brack = null, float $tol = 1.4901161193847656E-8, int $maxiter = 5000): float
```

scipy.optimize.golden: golden-section search, returning the minimiser (a port of _minimize_scalar_golden).

### minimize

```php
static minimize(callable $fun, array $x0, ?callable $jac = null, string $method = 'BFGS', ?array $bounds = null, array $options = []): Tessero\Optimize\OptimizeResult
```

### minimizeScalar

```php
static minimizeScalar(callable $func, ?array $bracket = null, ?array $bounds = null, string $method = '', array $options = []): Tessero\Optimize\OptimizeResult
```

scipy.optimize.minimize_scalar: minimise a function of one variable. Methods 'brent' (default; Brent's
method on a downhill bracket, a port of scipy's Brent class and bracket()) and 'bounded' (fminbound). The
result's x is a one-element list (Tessero's OptimizeResult carries a vector x).

### rosen

```php
static rosen(array $x): float
```

The Rosenbrock test function (scipy.optimize.rosen): sum 100 (x_{i+1}-x_i^2)^2 + (1-x_i)^2.

### rosenDer

```php
static rosenDer(array $x): array
```

Gradient of the Rosenbrock function (scipy.optimize.rosen_der).

### rosenHess

```php
static rosenHess(array $x): array
```

Hessian matrix of the Rosenbrock function (scipy.optimize.rosen_hess).

### rosenHessProd

```php
static rosenHessProd(array $x, array $p): array
```

Hessian of the Rosenbrock function times a vector p (scipy.optimize.rosen_hess_prod).
