#!/bin/sh
set -eu

repo_root=$(CDPATH='' cd -- "$(dirname "$0")/.." && pwd)
smoke_root=${NTUA_SMOKE_ROOT:-"$repo_root/.tmp-tests/workflow"}
bin_dir="$smoke_root/bin"
support_dir="$smoke_root/support"
demo_dir="$smoke_root/demo"
export_dir="$smoke_root/exported"
texmf_dir="$smoke_root/texmf"
external_dir="$smoke_root/external"
standalone_dir="$smoke_root/standalone"

rm -rf "$smoke_root"
mkdir -p "$smoke_root"

cd "$repo_root"
texlua scripts/ntua-beamer.lua setup --bin-dir "$bin_dir" --support-dir "$support_dir"

PATH="$bin_dir:$PATH"
export PATH

ntua-beamer doctor
ntua-beamer new "$demo_dir"
(cd "$demo_dir" && ntua-beamer build)

test -f "$demo_dir/.build/main.pdf"

ntua-beamer export "$demo_dir/main.tex" --output "$export_dir/"
test -f "$export_dir/main.pdf"

ntuabeamer-class package
test -f "$repo_root/dist/ntuabeamer-modular/ntuabeamer.cls"
test -f "$repo_root/dist/ntuabeamer-standalone/ntuabeamer.cls"

TEXMFHOME="$texmf_dir" ntuabeamer-class install
installed_cls=$(TEXMFHOME="$texmf_dir" kpsewhich ntuabeamer.cls)
test "$installed_cls" = "$texmf_dir/tex/latex/ntuabeamer/ntuabeamer.cls"

mkdir -p "$external_dir"
cat > "$external_dir/main.tex" <<'EOF'
\documentclass{ntuabeamer}
\title{Installed Class Check}
\author{NTUA}
\date{\today}

\begin{document}
\begin{frame}[plain]
  \makepresentationtitle
\end{frame}
\begin{frame}{Hello}
  Hello.
\end{frame}
\end{document}
EOF

(cd "$external_dir" && TEXMFHOME="$texmf_dir" latexmk -pdf -interaction=nonstopmode -halt-on-error -auxdir=.build -outdir=.build main.tex)
test -f "$external_dir/.build/main.pdf"

mkdir -p "$standalone_dir"
cp "$repo_root/dist/ntuabeamer-standalone/ntuabeamer.cls" "$standalone_dir/ntuabeamer.cls"
cat > "$standalone_dir/main.tex" <<'EOF'
\documentclass{ntuabeamer}
\title{Standalone Class Check}
\author{NTUA}
\date{\today}

\begin{document}
\begin{frame}[plain]
  \makepresentationtitle
\end{frame}
\begin{frame}{Hello}
  Hello.
\end{frame}
\end{document}
EOF

(cd "$standalone_dir" && latexmk -pdf -interaction=nonstopmode -halt-on-error -auxdir=.build -outdir=.build main.tex)
test -f "$standalone_dir/.build/main.pdf"

printf '%s\n' "Workflow smoke test passed."
