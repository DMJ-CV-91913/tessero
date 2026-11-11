# TesseroServiceProvider

`Tessero\Laravel\TesseroServiceProvider` extends `Illuminate\Support\ServiceProvider`

Registers Tessero in a Laravel application:
  - config/tessero.php (php artisan vendor:publish --tag=tessero-config)
  - TesseroManager singleton + Tessero facade
  - runtime settings (threads, native memory budget) applied when libtessero
    first loads (a request that never uses Tessero does not load it) and,
    under Octane and queue workers, again at the start of every request and
    job, so long-lived workers never inherit another request's settings
  - response()->ndarray($array): JSON written straight from native memory
  - php artisan tessero:doctor

## Methods

### boot

```php
boot(): void
```

### register

```php
register(): void
```
