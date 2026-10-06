<?php

declare(strict_types=1);

namespace Tessero\Laravel;

use Illuminate\Contracts\Events\Dispatcher;
use Illuminate\Contracts\Routing\ResponseFactory;
use Illuminate\Http\JsonResponse;
use Illuminate\Support\ServiceProvider;
use Tessero\Ext\NDArray as ExtArray;
use Tessero\Laravel\Console\DoctorCommand;
use Tessero\NDArray as FfiArray;

/**
 * Registers Tessero in a Laravel application:
 *   - config/tessero.php (php artisan vendor:publish --tag=tessero-config)
 *   - TesseroManager singleton + Tessero facade
 *   - runtime settings (threads, native memory budget) applied at boot and,
 *     under Octane, again at the start of every request and queued job, so
 *     long-lived workers never inherit another request's settings
 *   - response()->ndarray($array): JSON written straight from native memory
 *   - php artisan tessero:doctor
 */
final class TesseroServiceProvider extends ServiceProvider
{
    public function register(): void
    {
        $this->mergeConfigFrom(__DIR__ . '/../config/tessero.php', 'tessero');
        $this->app->singleton(TesseroManager::class, fn ($app): TesseroManager => new TesseroManager((array) $app['config']->get('tessero', [])));
        $this->app->alias(TesseroManager::class, 'tessero');
    }

    public function boot(): void
    {
        if ($this->app->runningInConsole()) {
            $this->publishes([__DIR__ . '/../config/tessero.php' => $this->app->configPath('tessero.php')], 'tessero-config');
            $this->commands([DoctorCommand::class]);
        }

        $manager = $this->app->make(TesseroManager::class);
        try {
            $manager->applyRuntimeSettings();
        } catch (\Throwable $e) {
            // no backend yet (e.g. during composer install); tessero:doctor explains what is missing
        }

        $this->registerWorkerHooks($manager);
        $this->registerResponseMacro();
    }

    /** Re-apply settings per Octane request and per queued job (both reuse one PHP process). */
    private function registerWorkerHooks(TesseroManager $manager): void
    {
        if (! $this->app->bound('events')) {
            return;
        }
        /** @var Dispatcher $events */
        $events = $this->app->make('events');
        $reapply = static function () use ($manager): void {
            try {
                $manager->applyRuntimeSettings();
            } catch (\Throwable) {
            }
        };
        foreach ([
            'Laravel\\Octane\\Events\\RequestReceived',
            'Laravel\\Octane\\Events\\TaskReceived',
            'Illuminate\\Queue\\Events\\JobProcessing',
        ] as $event) {
            $events->listen($event, $reapply);
        }
    }

    private function registerResponseMacro(): void
    {
        if (! $this->app->bound(ResponseFactory::class)) {
            return;
        }
        $factory = $this->app->make(ResponseFactory::class);
        if (! method_exists($factory, 'macro')) {
            return;
        }
        $factory->macro('ndarray', function (ExtArray|FfiArray $array, int $status = 200, array $headers = []): JsonResponse {
            $response = new JsonResponse(null, $status, $headers);
            $response->setJson($array->toJson());

            return $response;
        });
    }
}
