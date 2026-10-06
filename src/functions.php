<?php

declare(strict_types=1);

namespace Tessero;

/**
 * Short constructors, so scripts read like NumPy:  use function Tessero\{arr, zeros};
 */
function arr(mixed $data, DType|string|null $dtype = null): NDArray
{
    return NDArray::array($data, $dtype);
}

/** @param list<int>|int $shape */
function zeros(array|int $shape, DType|string $dtype = DType::Float64): NDArray
{
    return NDArray::zeros($shape, $dtype);
}

/** @param list<int>|int $shape */
function ones(array|int $shape, DType|string $dtype = DType::Float64): NDArray
{
    return NDArray::ones($shape, $dtype);
}

function arange(int|float $start, int|float|null $stop = null, int|float $step = 1, DType|string|null $dtype = null): NDArray
{
    return NDArray::arange($start, $stop, $step, $dtype);
}

function linspace(float $start, float $stop, int $num = 50, bool $endpoint = true): NDArray
{
    return NDArray::linspace($start, $stop, $num, $endpoint);
}
