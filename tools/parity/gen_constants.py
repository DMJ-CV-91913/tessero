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
}
with open(os.path.join(ROOT, 'tests', 'fixtures', 'parity', 'constants.json'), 'w') as fh:
    json.dump(fixture, fh, separators=(',', ':'))

print(f"gen_constants: {len(names)} constants + 3 helpers; fixture with {len(names)} values")
