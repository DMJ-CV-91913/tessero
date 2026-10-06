# 0008. The project is named Tessero

- Status: Accepted
- Date: 2026-09-26

## Context

The design briefs used the working names "phpNum", "phpSci" and "NumPHP".
These collide with existing projects: `sciphp/numphp` (active, pure PHP),
NumPHP, and the archived PHPSci organisation. Funders, package registries and
users would confuse them.

## Decision

The platform is named **Tessero**: the Composer packages are `tessero/tessero`,
`tessero/tessero-ext` and `tessero/laravel`, the PHP namespace is `Tessero\`,
the extension is `tessero` with namespace `Tessero\Ext`, and the C prefix is
`tsr_`. The [design brief map](../prompt-mapping.md) translates the working
names.

## Consequences

- No collision with existing packages or extensions.
- Documents that use the working names need the mapping page.
