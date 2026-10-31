use strict;
use warnings;
no strict 'vars';
use Cwd qw(abs_path getcwd);
use File::Basename qw(dirname fileparse);
use File::Spec;
use Config;

my $path_sep = $Config{path_sep} || ':';
my $source_path;
my $source_dir;
my $source_name;

sub ntua_detect_source_path {
    for my $arg (@ARGV) {
        next if $arg =~ /^-/;

        my $abs_source = abs_path( File::Spec->rel2abs($arg, getcwd()) );
        next unless defined $abs_source;

        return $abs_source;
    }

    return undef;
}

$source_path = ntua_detect_source_path();

if ( defined $source_path ) {
    my $existing_bibinputs = $ENV{BIBINPUTS} // q{};
    $source_dir = dirname($source_path);
    ($source_name) = fileparse($source_path, qr/\.[^.]*/);

    my $source_assets_dir = File::Spec->catdir($source_dir, 'assets');

    $aux_dir = File::Spec->catdir($source_dir, '.build');
    $out_dir = $aux_dir;

    $ENV{BIBINPUTS}
      = $source_dir . $path_sep . $source_assets_dir . $path_sep . $existing_bibinputs;
}

END {
    return unless defined $source_dir and defined $source_name;

    unlink map {
      File::Spec->catfile($source_dir, $source_name . $_)
    } qw(
      .aux
      .bbl
      .bcf
      .blg
      .fdb_latexmk
      .fls
      .log
      .nav
      .out
      .pdf
      .run.xml
      .snm
      .synctex.gz
      .toc
    );
}

1;
