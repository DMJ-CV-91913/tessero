<?php

declare(strict_types=1);

namespace Tessero\Internal;

/*
 * With the optional native extension (ext-tessero) loaded, NDArray inherits from the
 * extension's internal class Tessero\Ext\Operand and gains + - * / ** % .
 * Without it, this is an empty base class and everything else is unchanged.
 *
 * A subclass, not class_alias(): PHP 8.2 refuses to alias an internal class (8.3 allows it),
 * so with the extension loaded NDArray failed to load at all on 8.2.
 */
if (extension_loaded('tessero') && class_exists(\Tessero\Ext\Operand::class, false)) {
    /** @internal */
    abstract class OperandBase extends \Tessero\Ext\Operand
    {
    }
} else {
    /** @internal */
    abstract class OperandBase
    {
    }
}
