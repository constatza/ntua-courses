#!/bin/sh
set -eu

REPO_SLUG="${NTUA_BEAMER_REPO:-constatza/beamer-ntua}"
REPO_REF="${NTUA_BEAMER_REF:-main}"
BASE_URL="${NTUA_BEAMER_BASE_URL:-https://raw.githubusercontent.com/$REPO_SLUG/$REPO_REF}"

require_command() {
  command -v "$1" >/dev/null 2>&1 || {
    printf '%s\n' "Error: required command not found: $1" >&2
    exit 1
  }
}

download_file() {
  target="$1"
  url="$2"

  if command -v curl >/dev/null 2>&1; then
    curl -fsSL "$url" -o "$target"
    return
  fi

  if command -v wget >/dev/null 2>&1; then
    wget -qO "$target" "$url"
    return
  fi

  printf '%s\n' "Error: curl or wget is required to download the class files." >&2
  exit 1
}

refresh_tex_db() {
  if command -v mktexlsr >/dev/null 2>&1; then
    mktexlsr >/dev/null 2>&1 || true
    return
  fi

  if command -v initexmf >/dev/null 2>&1; then
    initexmf --update-fndb >/dev/null 2>&1 || true
  fi
}

resolve_texmfhome() {
  if [ -n "${TEXMFHOME:-}" ]; then
    printf '%s\n' "$TEXMFHOME"
    return
  fi

  require_command kpsewhich
  texmfhome="$(kpsewhich -var-value=TEXMFHOME)"
  [ -n "$texmfhome" ] || {
    printf '%s\n' "Error: could not resolve TEXMFHOME via kpsewhich." >&2
    exit 1
  }

  printf '%s\n' "$texmfhome"
}

require_command sh

tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/ntuabeamer-install.XXXXXX")"
cleanup() {
  rm -rf "$tmpdir"
}
trap cleanup EXIT HUP INT TERM

mkdir -p "$tmpdir/assets"

for file in ntuabeamer.cls ntuabeamer.sty macros.tex theme.tex; do
  download_file "$tmpdir/$file" "$BASE_URL/packages/ntuabeamer-class/framework/$file"
done

for asset in mgroup.png ntua.png school-civil-engineering.jpg; do
  download_file "$tmpdir/assets/$asset" "$BASE_URL/packages/ntuabeamer-class/framework/assets/$asset"
done

texmfhome="$(resolve_texmfhome)"
install_dir="$texmfhome/tex/latex/ntuabeamer"

rm -rf "$install_dir"
mkdir -p "$install_dir"

cp "$tmpdir/ntuabeamer.cls" "$install_dir/ntuabeamer.cls"
cp "$tmpdir/ntuabeamer.sty" "$install_dir/ntuabeamer.sty"
cp "$tmpdir/macros.tex" "$install_dir/macros.tex"
cp "$tmpdir/theme.tex" "$install_dir/theme.tex"
cp "$tmpdir/assets/"* "$install_dir/"

refresh_tex_db

printf '%s\n' "Installed class -> $install_dir"
