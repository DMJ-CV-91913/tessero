#!/usr/bin/env python3
"""Generate Tessero\\Constants (scipy.constants parity) and its fixture.

Emits src/Constants.php with every public scalar constant of scipy.constants as a case-sensitive PHP class
constant (PHP method names are case-insensitive, so G vs g must be constants, not methods), plus the standalone
helpers convert_temperature / lambda2nu / nu2lambda. Also emits tests/fixtures/parity/constants.json with the
reference values and a few helper-call cases. The lookup helpers value/unit/precision/find and physical_constants
(which need the large CODATA table) are left for a later pass.

    python3 tools/parity/gen_constants.py
"""
import json
import os

import scipy.constants as sc

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

names = sorted(n for n in dir(sc) if not n.startswith('_') and isinstance(getattr(sc, n), (int, float))
               and not isinstance(getattr(sc, n), bool))
vals = {n: float(getattr(sc, n)) for n in names}
pc = {k: (float(v[0]), str(v[1]), float(v[2])) for k, v in sc.physical_constants.items()}
_SAMPLE = ['speed of light in vacuum', 'Planck constant', 'elementary charge', 'electron mass',
           'proton mass', 'Avogadro constant', 'Boltzmann constant', 'electron g factor']
from scipy.constants import _codata as _cd
_CURRENT = sorted(_cd._current_constants)


def php_str(v):
    return "'" + v.replace('\\', '\\\\').replace("'", "\\'") + "'"


def php_float(v):
    if v != v:
        return 'NAN'
    if v == float('inf'):
        return 'INF'
    if v == float('-inf'):
        return '-INF'
    return repr(v)


lines = ['<?php', '', 'declare(strict_types=1);', '', 'namespace Tessero;', '',
         '/**', ' * scipy.constants: CODATA physical constants, SI prefixes and unit conversions, plus the',
         ' * temperature and wavelength/frequency helpers. Values are generated from scipy.constants by',
         ' * tools/parity/gen_constants.php; do not edit. The value/unit/precision lookups come in a later pass.',
         ' */', 'final class Constants', '{']
for n in names:
    lines.append(f'    public const {n} = {php_float(vals[n])};')
lines.append('')
lines.append('    /** CODATA physical constants (scipy.constants.physical_constants): name => [value, unit, uncertainty]. */')
lines.append('    private const CODATA = [')
for _k in sorted(pc):
    _val, _unit, _unc = pc[_k]
    lines.append(f'        {php_str(_k)} => [{php_float(_val)}, {php_str(_unit)}, {php_float(_unc)}],')
lines.append('    ];')
lines.append('')
lines.append('    /** The current-year CODATA keys; scipy.constants.find searches only these (not deprecated aliases). */')
lines.append('    private const CURRENT = [')
for _ck in _CURRENT:
    lines.append(f'        {php_str(_ck)},')
lines.append('    ];')
helpers = r'''
    /** scipy.constants.convert_temperature: between Kelvin, Celsius, Fahrenheit and Rankine. */
    public static function convertTemperature(float $val, string $oldScale, string $newScale): float
    {
        $tempK = match (strtolower($oldScale)[0]) {
            'c' => $val + 273.15,
            'k' => $val,
            'f' => ($val - 32.0) * 5.0 / 9.0 + 273.15,
            'r' => $val * 5.0 / 9.0,
            default => throw new \Tessero\Exceptions\TesseroException("convert_temperature: unknown scale '{$oldScale}'"),
        };
        return match (strtolower($newScale)[0]) {
            'c' => $tempK - 273.15,
            'k' => $tempK,
            'f' => ($tempK - 273.15) * 9.0 / 5.0 + 32.0,
            'r' => $tempK * 9.0 / 5.0,
            default => throw new \Tessero\Exceptions\TesseroException("convert_temperature: unknown scale '{$newScale}'"),
        };
    }

    /** scipy.constants.lambda2nu: convert a wavelength to an optical frequency (c / lambda). */
    public static function lambda2nu(float $lambda): float
    {
        return self::speed_of_light / $lambda;
    }

    /** scipy.constants.nu2lambda: convert an optical frequency to a wavelength (c / nu). */
    public static function nu2lambda(float $nu): float
    {
        return self::speed_of_light / $nu;
    }

    /** scipy.constants.value: the value of a CODATA physical constant by name. */
    public static function value(string $key): float
    {
        return self::codata($key)[0];
    }

    /** scipy.constants.unit: the unit string of a CODATA physical constant. */
    public static function unit(string $key): string
    {
        return self::codata($key)[1];
    }

    /** scipy.constants.precision: the relative precision (uncertainty / value) of a CODATA constant. */
    public static function precision(string $key): float
    {
        $e = self::codata($key);

        return $e[2] / $e[0];
    }

    /** scipy.constants.find: the sorted CODATA keys containing $sub (all keys when $sub is null). */
    public static function find(?string $sub = null): array
    {
        $keys = self::CURRENT;
        if ($sub !== null) {
            $needle = strtolower($sub);
            $keys = array_values(array_filter($keys, static fn (string $k): bool => str_contains(strtolower($k), $needle)));
        }
        sort($keys, SORT_STRING);

        return $keys;
    }

    /** scipy.constants.physical_constants: the full CODATA table, name => [value, unit, uncertainty]. */
    public static function physicalConstants(): array
    {
        return self::CODATA;
    }

    /** @return array{0: float, 1: string, 2: float} */
    private static function codata(string $key): array
    {
        if (! isset(self::CODATA[$key])) {
            throw new \Tessero\Exceptions\TesseroException("physical_constants: unknown constant '{$key}'");
        }

        return self::CODATA[$key];
    }
}
'''
lines.append(helpers)
with open(os.path.join(ROOT, 'src', 'Constants.php'), 'w') as fh:
    fh.write('\n'.join(lines))

fixture = {
    'module': 'constants',
    'values': vals,
    'convert_temperature': [
        {'val': v, 'old': o, 'new': nw, 'expect': float(sc.convert_temperature(v, o, nw))}
        for v, o, nw in ((100.0, 'Celsius', 'Kelvin'), (32.0, 'Fahrenheit', 'Celsius'),
                         (300.0, 'Kelvin', 'Fahrenheit'), (0.0, 'Celsius', 'Rankine'),
                         (491.67, 'Rankine', 'Kelvin'))],
    'lambda2nu': [{'x': x, 'expect': float(sc.lambda2nu(x))} for x in (5e-7, 1e-6, 2.5e-7)],
    'nu2lambda': [{'x': x, 'expect': float(sc.nu2lambda(x))} for x in (6e14, 3e14, 1.2e15)],
    'value': [{'key': k, 'expect': float(sc.value(k))} for k in _SAMPLE],
    'unit': [{'key': k, 'expect': sc.unit(k)} for k in _SAMPLE],
    'precision': [{'key': k, 'expect': float(sc.precision(k))} for k in _SAMPLE],
    'find': [{'sub': ss, 'expect': sorted(sc.find(ss))} for ss in ('electron mass', 'proton', 'Planck', 'mag. mom.')],
    'physical_constants_count': len(sc.physical_constants),
}
with open(os.path.join(ROOT, 'tests', 'fixtures', 'parity', 'constants.json'), 'w') as fh:
    json.dump(fixture, fh, separators=(',', ':'))

print(f"gen_constants: {len(names)} constants + 3 helpers; fixture with {len(names)} values")
