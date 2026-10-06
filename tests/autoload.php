<?php

// Composer-free autoloader so the suite runs from a plain checkout.
spl_autoload_register(static function (string $class): void {
    foreach (['Tessero\\Tests\\' => __DIR__ . '/', 'Tessero\\' => __DIR__ . '/../src/'] as $prefix => $dir) {
        if (str_starts_with($class, $prefix)) {
            $file = $dir . str_replace('\\', '/', substr($class, strlen($prefix))) . '.php';
            if (is_file($file)) {
                require $file;
            }

            return;
        }
    }
});
require_once __DIR__ . '/../src/functions.php';
