<?php

declare(strict_types=1);

namespace Tessero\Laravel\Facades;

use Illuminate\Support\Facades\Facade;
use Tessero\Laravel\TesseroManager;

/**
 * @method static string backend()
 * @method static \Tessero\Ext\NDArray|\Tessero\NDArray array(mixed $data, ?string $dtype = null)
 * @method static \Tessero\Ext\NDArray|\Tessero\NDArray zeros(array|int $shape, string $dtype = 'float64')
 * @method static \Tessero\Ext\NDArray|\Tessero\NDArray memmap(string $filename, string $mode = 'r+', ?array $shape = null, string $dtype = 'float64', int $offset = 0)
 * @method static \Tessero\Ext\NDArray|\Tessero\NDArray load(string $path, ?string $mmapMode = null)
 * @method static \Tessero\Ext\NDArray|\Tessero\NDArray openMemmap(string $path, string $mode = 'r+', ?array $shape = null, string $dtype = 'float64', bool $fortranOrder = false)
 * @method static \Tessero\Ext\NDArray|\Tessero\NDArray fromBytes(string $bytes, string $dtype, array $shape)
 * @method static array linprog(mixed $c, mixed $A_ub = null, mixed $b_ub = null, mixed $A_eq = null, mixed $b_eq = null, ?array $bounds = null)
 * @method static array milp(mixed $c, bool|array $integrality = true, mixed $A_ub = null, mixed $b_ub = null, mixed $A_eq = null, mixed $b_eq = null, ?array $bounds = null)
 * @method static array solveMdp(mixed $P, mixed $R, float $gamma, string $method = 'policy_iteration')
 * @method static \Tessero\Ext\NDArray|\Tessero\NDArray normal(int $seed, array|int $shape, float $mean = 0.0, float $sd = 1.0)
 * @method static \Tessero\Ext\NDArray|\Tessero\NDArray random(int $seed, array|int $shape)
 * @method static array info()
 *
 * @see TesseroManager
 */
final class Tessero extends Facade
{
    protected static function getFacadeAccessor(): string
    {
        return TesseroManager::class;
    }
}
