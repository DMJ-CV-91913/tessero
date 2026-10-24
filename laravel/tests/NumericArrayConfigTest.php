<?php

declare(strict_types=1);

namespace {
    // Laravel's config() helper, for this test only: other bridge tests never call it.
    if (! function_exists('config')) {
        function config(string $key, mixed $default = null): mixed
        {
            return \Tessero\Laravel\Tests\NumericArrayConfigTest::$config[$key] ?? $default;
        }
    }
}

namespace Tessero\Laravel\Tests {
    use PHPUnit\Framework\TestCase;
    use Tessero\Laravel\Rules\NumericArray;

    require_once __DIR__ . '/stubs.php';

    /**
     * config('tessero.max_elements') was never read: the rule always used its hardcoded 1,000,000.
     */
    final class NumericArrayConfigTest extends TestCase
    {
        /** @var array<string, mixed> */
        public static array $config = [];

        protected function tearDown(): void
        {
            self::$config = [];
        }

        public function testTheRuleDefaultsToTheConfiguredCap(): void
        {
            self::$config = ['tessero.max_elements' => 10];
            $this->assertSame(10, NumericArray::configuredMaxElements());
            $this->assertStringContainsString('the limit is 10', (string) NumericArray::check(range(1, 11)));
            $this->assertNull(NumericArray::check(range(1, 10)));
            $messages = [];
            (new NumericArray())->validate('v', range(1, 11), function (string $m) use (&$messages): void { $messages[] = $m; });
            $this->assertCount(1, $messages);
            $this->assertNull(NumericArray::check(range(1, 11), maxElements: 20), 'an explicit cap still wins');
        }

        public function testMissingOrInvalidConfigFallsBackToTheDefault(): void
        {
            $this->assertSame(NumericArray::DEFAULT_MAX_ELEMENTS, NumericArray::configuredMaxElements());
            self::$config = ['tessero.max_elements' => 0];
            $this->assertSame(NumericArray::DEFAULT_MAX_ELEMENTS, NumericArray::configuredMaxElements());
        }

        public function testNestingDepthIsCapped(): void
        {
            $deep = [1.0];
            for ($i = 0; $i < NumericArray::MAX_NDIM; $i++) {
                $deep = [$deep];
            }
            $this->assertStringContainsString('at most 32 dimensions', (string) NumericArray::check($deep));
        }
    }
}
