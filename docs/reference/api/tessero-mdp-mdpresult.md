# MdpResult

`Tessero\Mdp\MdpResult` implements `JsonSerializable`

Solution of a Markov decision process.

## Methods

### __construct

```php
__construct(Tessero\NDArray $values, Tessero\NDArray $policy, int $iterations, bool $converged, string $method, ?float $delta = null)
```

### jsonSerialize

```php
jsonSerialize(): array
```
