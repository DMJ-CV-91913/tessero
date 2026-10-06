--TEST--
Operand: userland classes extending Tessero\Ext\Operand get arithmetic operators
--EXTENSIONS--
tessero
--FILE--
<?php
final class Money extends Tessero\Ext\Operand
{
    public function __construct(public readonly int $cents) {}
    public function add($o): self { return new self($this->cents + ($o instanceof self ? $o->cents : (int) $o)); }
    public function sub($o): self { return new self($this->cents - ($o instanceof self ? $o->cents : (int) $o)); }
    public function rsub($o): self { return new self((int) $o - $this->cents); }
    public function mul($o): self { return new self($this->cents * (int) $o); }
}
$a = new Money(250);
echo ($a + new Money(50))->cents, " ", ($a - 50)->cents, " ", (1000 - $a)->cents, " ", (3 * $a)->cents, "\n";
try { $a / 2; } catch (Error $e) { echo get_class($e), "\n"; }
?>
--EXPECT--
300 200 750 750
TypeError
