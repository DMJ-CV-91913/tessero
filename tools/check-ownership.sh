#!/usr/bin/env bash
# Static memory-ownership lint (ADR 0015): the layer that allocates memory must own its deallocation.
# The runtime gate (-DTSR_OWNERSHIP_CHECK in the sanitizer, fuzz and ASan CI builds) catches what runs; this catches
# the patterns on paths no test reaches. bash + perl only. Exit 1 on any finding.
#
#   K1  kernel: a pointer from the counted allocator (tsr_alloc, tsr_calloc, tsr_alloc_n, fn_arg_doubles) passed to
#       free(), or one from the C library (malloc, calloc, realloc, strdup, tsr_scratch_n) passed to tsr_free() or
#       fn_free_doubles(), within one function (by the variable's most recent assignment)
#   K2  kernel: C-library memory stored into a result or array descriptor (.arr.data / ->data = malloc(...)):
#       results cross the ABI and are released with tsr_free
#   K3  kernel: Zend memory or API (emalloc, efree, zval, zend_*, ...) under csrc/: OpenMP workers run kernel code
#       and Zend's allocator is per request and single threaded
#   E1  extension: the raw C library allocator (malloc, calloc, realloc, free, strdup); ext memory is Zend's
#   E2  extension: efree()/pefree() of kernel memory (arr.data, ->block, a tsr_result)
#   E3  extension: `#pragma omp`: threads must never run Zend code
#   H1  tessero.h: a function pointer (a callback the kernel could call from a worker) without the marker
#       "ADR 0015: calling thread" on the same line
#   P1  PHP binding: FFI::free() of kernel memory (->data, raw(), an address) or of anything not created with
#       FFI::new(..., false) in the same file
#   P2  PHP binding: tsr_free() outside the two owners (Native/Buffer.php, Native/Registry.php)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export LC_ALL=C

perl -e '
use strict; use warnings;
my $bad = 0;
sub report { my ($rule, $file, $line, $text) = @_; $text =~ s/^\s+//; print "$rule $file:$line: $text\n"; $bad++; }
# comments and string literals blanked (newlines kept, so line numbers stay right)
sub code_of {
    my ($s, $php) = @_;
    $s =~ s{/\*.*?\*/}{ (my $c = $&) =~ s/[^\n]/ /g; $c }gse;
    $s =~ s{"(?:\\.|[^"\\\n])*"}{ (my $c = $&) =~ s/./ /g; "\"" . substr($c, 1, -1) . "\"" }ge;
    $s =~ s{//[^\n]*}{ " " x length($&) }ge;
    $s =~ s{(?<![\$\w])#(?!\s*(include|define|if|ifdef|ifndef|endif|else|elif|undef|pragma|error))[^\n]*}{ " " x length($&) }ge if $php;
    return $s;
}
sub slurp { my $f = shift; local $/; open my $h, "<", $f or die "$f: $!"; my $s = <$h>; close $h; $s }

my @kernel = (glob("csrc/src/*.c"), glob("csrc/src/*.h"), glob("csrc/cxx/*.cpp"), glob("csrc/cxx/*.hpp"));
my $T = qr/\b(?:tsr_alloc|tsr_calloc|tsr_alloc_n|fn_arg_doubles)\s*\(/;
my $L = qr/\b(?:malloc|calloc|realloc|strdup|tsr_scratch_n)\s*\(/;
for my $f (@kernel) {
    my @lines = split /\n/, code_of(slurp($f), 0), -1;
    my %fam;                                       # variable -> T or L, by its latest assignment in this function
    for my $i (0 .. $#lines) {
        my $l = $lines[$i];
        %fam = () if $l =~ /^[}]/;                 # a function ends at a closing brace in column 0
        while ($l =~ /([A-Za-z_]\w*(?:(?:->|\.)\w+|\[[^\]=]*\])*)\s*=\s*(?:\([^()]*\)\s*)?([A-Za-z_]\w*)\s*\(/g) {
            my ($v, $fn) = ($1, $2);
            if ("$fn(" =~ $T) { $fam{$v} = "T" } elsif ("$fn(" =~ $L) { $fam{$v} = "L" } else { delete $fam{$v} }
        }
        while ($l =~ /(?<![\w>.])free\s*\(\s*(?:\([^()]*\)\s*)?([A-Za-z_]\w*(?:(?:->|\.)\w+|\[[^\]]*\])*)\s*\)/g) {
            report("K1", $f, $i + 1, "free() of $1, which came from the counted allocator: use tsr_free") if ($fam{$1} // "") eq "T";
        }
        while ($l =~ /\b(tsr_free|fn_free_doubles)\s*\(\s*(?:\([^()]*\)\s*)?([A-Za-z_]\w*(?:(?:->|\.)\w+|\[[^\]]*\])*)\s*,/g) {
            report("K1", $f, $i + 1, "$1() of $2, which came from the C library: use free") if ($fam{$2} // "") eq "L";
        }
        report("K2", $f, $i + 1, $l) if $l =~ /(?:\.|->)(?:arr\.)?data\s*=\s*(?:\([^()]*\)\s*)?$L/ && $f !~ m{/alloc\.c$};
        report("K3", $f, $i + 1, $l) if $l =~ /\b(?:emalloc|efree|ecalloc|erealloc|estrdup|pemalloc|pefree|zval|zend_\w+|ZEND_\w+|Z_TYPE\w*)\b/;
    }
}
for my $f (glob("ext/*.c"), glob("ext/*.h")) {
    my @lines = split /\n/, code_of(slurp($f), 0), -1;
    for my $i (0 .. $#lines) {
        my $l = $lines[$i];
        report("E1", $f, $i + 1, $l) if $l =~ /(?<![\w>.])(?:malloc|calloc|realloc|free|strdup)\s*\(/;
        report("E2", $f, $i + 1, $l) if $l =~ /\b(?:efree|pefree)\s*\([^;]*(?:arr\.data|->block\b|\bres\[|tsr_result)/;
        report("E3", $f, $i + 1, $l) if $l =~ /#\s*pragma\s+omp\b/;
    }
}
{
    my $raw = slurp("csrc/include/tessero.h");
    my @src = split /\n/, $raw, -1;
    my @lines = split /\n/, code_of($raw, 0), -1;
    for my $i (0 .. $#lines) {
        report("H1", "csrc/include/tessero.h", $i + 1, $src[$i]) if $lines[$i] =~ /\(\s*\*\s*\w*\s*\)\s*\(/ && $src[$i] !~ /ADR 0015: calling thread/;
    }
}
my %owners = map { $_ => 1 } ("src/Native/Buffer.php", "src/Native/Registry.php");
my @php;
my @todo = ("src");
while (my $d = shift @todo) {
    opendir(my $dh, $d) or die "$d: $!";
    for my $e (sort grep { !/^\./ } readdir $dh) { my $p = "$d/$e"; if (-d $p) { push @todo, $p } elsif ($p =~ /\.php$/) { push @php, $p } }
    closedir $dh;
}
for my $f (@php) {
    my $code = code_of(slurp($f), 1);
    my %unmanaged;
    while ($code =~ /(\$\w+)\s*=\s*\$\w+->new\([^;]*,\s*false\s*\)/g) { $unmanaged{$1} = 1 }
    while ($code =~ /(\$\w+)\s*=\s*FFI::new\([^;]*,\s*false\s*\)/g) { $unmanaged{$1} = 1 }
    my %held;                                       # arrays that collect unmanaged buffers: $keep[] = $buf
    while ($code =~ /(\$\w+)\[\]\s*=\s*(\$\w+)\s*;/g) { $held{$1} = 1 if $unmanaged{$2} }
    my @lines = split /\n/, $code, -1;
    for my $i (0 .. $#lines) {
        my $l = $lines[$i];
        while ($l =~ /FFI::free\(\s*([^)]*)\)/g) {
            my $arg = $1;
            my $ok = $arg =~ /^\$\w+$/ && ($unmanaged{$arg} || grep { $code =~ /foreach\s*\(\s*\Q$_\E\s+as\s+\Q$arg\E\s*\)/ } keys %held);
            report("P1", $f, $i + 1, $l) if !$ok || $arg =~ /->data|raw\(|address/;
        }
        report("P2", $f, $i + 1, $l) if $l =~ /->tsr_free\(/ && !$owners{$f};
    }
}
if ($bad) { print STDERR "check-ownership: $bad finding(s); see docs/architecture/adr/0015-memory-ownership.md\n"; exit 1 }
print "check-ownership: clean (kernel, extension, tessero.h, PHP binding)\n";
'
