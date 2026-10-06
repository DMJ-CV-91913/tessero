# LinprogResult

`Tessero\Optimize\LinprogResult` implements `JsonSerializable`

Result of LinearProgramming::linprog (field names follow scipy.optimize.OptimizeResult).

## Methods

### __construct

```php
__construct(?array $x, ?float $fun, bool $success, int $status, string $message, int $nit, array $ineqlinMarginals = [], array $eqlinMarginals = [], array $reducedCosts = [])
```

### jsonSerialize

```php
jsonSerialize(): array
```

### slack

```php
slack(array $A_ub, array $b_ub): array
```

Slack of each A_ub row at the solution: b_ub - A_ub x.
