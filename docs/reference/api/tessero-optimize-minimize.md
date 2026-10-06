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

### dot

```php
static dot(array $a, array $b): float
```

### minimize

```php
static minimize(callable $fun, array $x0, ?callable $jac = null, string $method = 'BFGS', ?array $bounds = null, array $options = []): Tessero\Optimize\OptimizeResult
```
