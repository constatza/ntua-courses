# Depth-independent loader: copy to <deck>/.latexmkrc.
# Finds beamer-ntua via $NTUA_BEAMER_HOME or by walking up to a sibling `beamer-ntua/`.
use Cwd qw(getcwd);
my $home = $ENV{NTUA_BEAMER_HOME};
unless ($home) {
    my $dir = getcwd();
    while ($dir =~ m{/}) {
        if ( -f "$dir/beamer-ntua/.latexmkrc" ) { $home = "$dir/beamer-ntua"; last; }
        $dir =~ s{/[^/]*$}{};
    }
}
die "beamer-ntua not found; set NTUA_BEAMER_HOME\n" unless $home;
require "$home/.latexmkrc";
1;
