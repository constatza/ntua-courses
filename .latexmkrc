use strict;
use warnings;
no strict 'vars';
use Cwd qw(abs_path getcwd);
use File::Basename qw(dirname fileparse);
use File::Spec;
use Config;

my $repo_root = abs_path( dirname(__FILE__) );
my $framework_dir = File::Spec->catdir($repo_root, 'packages', 'ntuabeamer-class', 'framework');
my $framework_assets_dir = File::Spec->catdir($framework_dir, 'assets');

my $path_sep = $Config{path_sep} || ':';
my $existing_texinputs = $ENV{TEXINPUTS} // q{};
$ENV{TEXINPUTS}
  = $framework_dir . '//' . $path_sep . $framework_assets_dir . '//' . $path_sep . $existing_texinputs;

sub ntua_detect_source_path {
    for my $arg (@ARGV) {
        next if $arg =~ /^-/;

        my $abs_source = abs_path( File::Spec->rel2abs($arg, getcwd()) );
        next unless defined $abs_source;

        return $abs_source;
    }

    return undef;
}

my $source_path = ntua_detect_source_path();

if ( defined $source_path ) {
    my $existing_bibinputs = $ENV{BIBINPUTS} // q{};
    my $source_dir = dirname($source_path);
    my $source_assets_dir = File::Spec->catdir($source_dir, 'assets');

    $aux_dir = File::Spec->catdir($source_dir, '.build');
    $out_dir = $aux_dir;

    $ENV{BIBINPUTS}
      = $source_dir . $path_sep . $source_assets_dir . $path_sep . $existing_bibinputs;
}

1;
