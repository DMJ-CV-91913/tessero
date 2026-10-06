<?php

declare(strict_types=1);

namespace Tessero;

/**
 * scipy.constants: CODATA physical constants, SI prefixes and unit conversions, plus the
 * temperature and wavelength/frequency helpers. Values are generated from scipy.constants by
 * tools/parity/gen_constants.php; do not edit. The value/unit/precision lookups come in a later pass.
 */
final class Constants
{
    public const Avogadro = 6.02214076e+23;
    public const Boltzmann = 1.380649e-23;
    public const Btu = 1055.05585262;
    public const Btu_IT = 1055.05585262;
    public const Btu_th = 1054.3502644888888;
    public const G = 6.6743e-11;
    public const Julian_year = 31557600.0;
    public const N_A = 6.02214076e+23;
    public const Planck = 6.62607015e-34;
    public const R = 8.31446261815324;
    public const Rydberg = 10973731.568157;
    public const Stefan_Boltzmann = 5.6703744191844314e-08;
    public const Wien = 0.0028977719551851727;
    public const acre = 4046.8564223999992;
    public const alpha = 0.0072973525643;
    public const angstrom = 1e-10;
    public const arcmin = 0.0002908882086657216;
    public const arcminute = 0.0002908882086657216;
    public const arcsec = 4.84813681109536e-06;
    public const arcsecond = 4.84813681109536e-06;
    public const astronomical_unit = 149597870700.0;
    public const atm = 101325.0;
    public const atmosphere = 101325.0;
    public const atomic_mass = 1.66053906892e-27;
    public const atto = 1e-18;
    public const au = 149597870700.0;
    public const bar = 100000.0;
    public const barrel = 0.15898729492799998;
    public const bbl = 0.15898729492799998;
    public const blob = 175.12683524647636;
    public const c = 299792458.0;
    public const calorie = 4.184;
    public const calorie_IT = 4.1868;
    public const calorie_th = 4.184;
    public const carat = 0.0002;
    public const centi = 0.01;
    public const day = 86400.0;
    public const deci = 0.1;
    public const degree = 0.017453292519943295;
    public const degree_Fahrenheit = 0.5555555555555556;
    public const deka = 10.0;
    public const dyn = 1e-05;
    public const dyne = 1e-05;
    public const e = 1.602176634e-19;
    public const eV = 1.602176634e-19;
    public const electron_mass = 9.1093837139e-31;
    public const electron_volt = 1.602176634e-19;
    public const elementary_charge = 1.602176634e-19;
    public const epsilon_0 = 8.8541878188e-12;
    public const erg = 1e-07;
    public const exa = 1e+18;
    public const exbi = 1.152921504606847e+18;
    public const femto = 1e-15;
    public const fermi = 1e-15;
    public const fine_structure = 0.0072973525643;
    public const fluid_ounce = 2.9573529562499998e-05;
    public const fluid_ounce_US = 2.9573529562499998e-05;
    public const fluid_ounce_imp = 2.84130625e-05;
    public const foot = 0.30479999999999996;
    public const g = 9.80665;
    public const gallon = 0.0037854117839999997;
    public const gallon_US = 0.0037854117839999997;
    public const gallon_imp = 0.00454609;
    public const gas_constant = 8.31446261815324;
    public const gibi = 1073741824.0;
    public const giga = 1000000000.0;
    public const golden = 1.618033988749895;
    public const golden_ratio = 1.618033988749895;
    public const grain = 6.479891e-05;
    public const gram = 0.001;
    public const gravitational_constant = 6.6743e-11;
    public const h = 6.62607015e-34;
    public const hbar = 1.0545718176461565e-34;
    public const hectare = 10000.0;
    public const hecto = 100.0;
    public const horsepower = 745.6998715822701;
    public const hour = 3600.0;
    public const hp = 745.6998715822701;
    public const inch = 0.0254;
    public const k = 1.380649e-23;
    public const kgf = 9.80665;
    public const kibi = 1024.0;
    public const kilo = 1000.0;
    public const kilogram_force = 9.80665;
    public const kmh = 0.2777777777777778;
    public const knot = 0.5144444444444445;
    public const lb = 0.45359236999999997;
    public const lbf = 4.4482216152605;
    public const light_year = 9460730472580800.0;
    public const liter = 0.001;
    public const litre = 0.001;
    public const long_ton = 1016.0469088;
    public const m_e = 9.1093837139e-31;
    public const m_n = 1.67492750056e-27;
    public const m_p = 1.67262192595e-27;
    public const m_u = 1.66053906892e-27;
    public const mach = 340.5;
    public const mebi = 1048576.0;
    public const mega = 1000000.0;
    public const metric_ton = 1000.0;
    public const micro = 1e-06;
    public const micron = 1e-06;
    public const mil = 2.5399999999999997e-05;
    public const mile = 1609.3439999999998;
    public const milli = 0.001;
    public const minute = 60.0;
    public const mmHg = 133.32236842105263;
    public const mph = 0.44703999999999994;
    public const mu_0 = 1.25663706127e-06;
    public const nano = 1e-09;
    public const nautical_mile = 1852.0;
    public const neutron_mass = 1.67492750056e-27;
    public const ounce = 0.028349523124999998;
    public const oz = 0.028349523124999998;
    public const parsec = 3.085677581491367e+16;
    public const pebi = 1125899906842624.0;
    public const peta = 1000000000000000.0;
    public const pi = 3.141592653589793;
    public const pico = 1e-12;
    public const point = 0.00035277777777777776;
    public const pound = 0.45359236999999997;
    public const pound_force = 4.4482216152605;
    public const proton_mass = 1.67262192595e-27;
    public const psi = 6894.757293168361;
    public const pt = 0.00035277777777777776;
    public const quecto = 1e-30;
    public const quetta = 1e+30;
    public const ronna = 1e+27;
    public const ronto = 1e-27;
    public const short_ton = 907.1847399999999;
    public const sigma = 5.6703744191844314e-08;
    public const slinch = 175.12683524647636;
    public const slug = 14.593902937206364;
    public const speed_of_light = 299792458.0;
    public const speed_of_sound = 340.5;
    public const stone = 6.3502931799999995;
    public const survey_foot = 0.3048006096012192;
    public const survey_mile = 1609.3472186944373;
    public const tebi = 1099511627776.0;
    public const tera = 1000000000000.0;
    public const ton_TNT = 4184000000.0;
    public const torr = 133.32236842105263;
    public const troy_ounce = 0.031103476799999998;
    public const troy_pound = 0.37324172159999996;
    public const u = 1.66053906892e-27;
    public const week = 604800.0;
    public const yard = 0.9143999999999999;
    public const year = 31536000.0;
    public const yobi = 1.2089258196146292e+24;
    public const yocto = 1e-24;
    public const yotta = 1e+24;
    public const zebi = 1.1805916207174113e+21;
    public const zepto = 1e-21;
    public const zero_Celsius = 273.15;
    public const zetta = 1e+21;

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
