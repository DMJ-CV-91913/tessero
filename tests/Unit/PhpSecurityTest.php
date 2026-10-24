<?php

declare(strict_types=1);

namespace Tessero\Tests\Unit;

use PHPUnit\Framework\TestCase;
use Tessero\Native\Library;

/**
 * The PHP security floor: 8.2.35, 8.3.36, 8.4.27, 8.5.12 or newer on each branch (owner decision 2026-10-24).
 * The FFI loader, the extension's MINIT check and the three composer.json constraints must agree.
 */
final class PhpSecurityTest extends TestCase
{
    private const FLOOR = ['8.2' => 80235, '8.3' => 80336, '8.4' => 80427, '8.5' => 80512];

    private const CONSTRAINT = '~8.2.35 || ~8.3.36 || ~8.4.27 || ^8.5.12';

    public function testEachBranchStartsAtItsSecurityRelease(): void
    {
        foreach (self::FLOOR as $branch => $floor) {
            $this->assertFalse(Library::phpPatched($floor - 1), "{$branch}: the release before the fix");
            $this->assertTrue(Library::phpPatched($floor), "{$branch}: the security release");
            $this->assertTrue(Library::phpPatched($floor + 1), "{$branch}: a later patch release");
            $this->assertFalse(Library::phpPatched(intdiv($floor, 100) * 100), "{$branch}.0");
            $this->assertTrue(Library::phpPatched(intdiv($floor, 100) * 100 + 99), "{$branch}.99");
        }
        $this->assertFalse(Library::phpPatched(80199), 'PHP 8.1 is below the supported range');
        $this->assertFalse(Library::phpPatched(70433));
        $this->assertTrue(Library::phpPatched(80600), 'a newer branch carries the fixes');
        $this->assertTrue(Library::phpPatched(90000));
    }

    public function testTheRunningPhpIsPatched(): void
    {
        $this->assertTrue(Library::phpPatched(PHP_VERSION_ID), 'PHP ' . PHP_VERSION . ' is below the security floor');
    }

    public function testComposerConstraintsMatchTheFloor(): void
    {
        foreach (['composer.json', 'laravel/composer.json', 'ext/composer.json'] as $file) {
            $json = json_decode((string) file_get_contents(__DIR__ . '/../../' . $file), true, 512, JSON_THROW_ON_ERROR);
            $this->assertSame(self::CONSTRAINT, $json['require']['php'], $file);
        }
    }

    public function testTheExtensionChecksTheSameFloor(): void
    {
        $c = (string) file_get_contents(__DIR__ . '/../../ext/tessero.c');
        foreach (self::FLOOR as $branch => $floor) {
            $this->assertMatchesRegularExpression("/PHP_VERSION_ID < {$floor}\\b/", $c, "{$branch}: compile-time floor");
            $this->assertStringContainsString("v >= {$floor};", $c, "{$branch}: run-time floor");
        }
    }
}
