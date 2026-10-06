<?php

declare(strict_types=1);

namespace Tessero\Internal;

/*
 * With the optional native extension (ext-tessero) loaded, NDArray inherits from the
 * extension's internal class Tessero\Ext\Operand and gains + - * / ** % .
 * Without it, this is an empty base class and everything else is unchanged.
 */
if (extension_loaded('tessero') && class_exists(\Tessero\Ext\Operand::class, false)) {
    class_alias(\Tessero\Ext\Operand::class, OperandBase::class);
} else {
    /** @internal */
    abstract class OperandBase
    {
    }
}
