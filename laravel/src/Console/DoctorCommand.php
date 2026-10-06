<?php

declare(strict_types=1);

namespace Tessero\Laravel\Console;

use Illuminate\Console\Command;
use Tessero\Laravel\TesseroManager;

/** php artisan tessero:doctor - which backend runs, SIMD, BLAS, memory, and what to fix. */
final class DoctorCommand extends Command
{
    protected $signature = 'tessero:doctor {--json : Print machine-readable output}';

    protected $description = 'Check the Tessero installation (backend, SIMD, BLAS, memory budget, preload)';

    public function handle(TesseroManager $tessero): int
    {
        try {
            $info = $tessero->info();
        } catch (\Throwable $e) {
            $this->error($e->getMessage());
            $this->line('Install the native extension (pie install tessero/tessero-ext) or enable FFI; see docs/getting-started/installation.md.');

            return self::FAILURE;
        }
        if ($this->option('json')) {
            $this->line(json_encode($info, JSON_PRETTY_PRINT | JSON_UNESCAPED_SLASHES));

            return self::SUCCESS;
        }
        $this->info('Backend: ' . $info['backend']);
        foreach (['ext' => 'Native extension', 'ffi' => 'FFI library'] as $key => $label) {
            if (! isset($info[$key])) {
                $this->line("  {$label}: not available");
                continue;
            }
            $this->line("  {$label}:");
            foreach ($info[$key] as $k => $v) {
                $this->line(sprintf('    %-16s %s', $k, is_bool($v) ? ($v ? 'yes' : 'no') : (string) ($v ?? '-')));
            }
        }
        if (PHP_SAPI !== 'cli' || ! isset($info['ext'])) {
            $this->line('Under PHP-FPM without ext-tessero, add vendor/tessero/tessero/resources/preload.php to opcache.preload.');
        }

        return self::SUCCESS;
    }
}
