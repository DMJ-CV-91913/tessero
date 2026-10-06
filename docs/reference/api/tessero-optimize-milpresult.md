# MilpResult

`Tessero\Optimize\MilpResult` implements `JsonSerializable`

Result of LinearProgramming::milp.

## Methods

### __construct

```php
__construct(?array $x, ?float $fun, bool $success, int $status, string $message, int $nodes, float $bestBound)
```

### gap

```php
gap(): float
```

Relative gap between the incumbent and the best bound (0 when proven optimal).

### jsonSerialize

```php
jsonSerialize(): array
```
