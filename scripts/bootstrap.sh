#!/bin/sh
# Bootstrap ntua-courses after clone.
#
#   git clone <repo> ntua-courses && cd ntua-courses && sh scripts/bootstrap.sh
#
# Initializes the course submodules at their pinned commits, then installs the ntuabeamer
# class (to TEXMFHOME) and build-course/release-course (to ~/.local/bin) at the beamer-ntua
# tag the courses pin, the same way CI does. beamer-ntua is an independent repo, not a submodule.
set -eu

echo "==> syncing submodule URLs"
git submodule sync

echo "==> initializing submodules"
git submodule update --init --recursive

ref=$(awk '/constatza\/beamer-ntua/ {f=1; next} f && /rev:/ {print $2; exit}' ml-course/.pre-commit-config.yaml)
[ -n "$ref" ] || { echo "error: no beamer-ntua rev in ml-course/.pre-commit-config.yaml" >&2; exit 1; }

echo "==> installing ntuabeamer $ref (class to TEXMFHOME, tools to ~/.local/bin)"
installer=$(mktemp)
trap 'rm -f "$installer"' EXIT
curl -fsSL -o "$installer" "https://raw.githubusercontent.com/constatza/beamer-ntua/$ref/scripts/install-class.sh"
NTUA_BEAMER_REF="$ref" sh "$installer"

echo "==> done. Make sure ~/.local/bin is on PATH, then build a course with:"
echo "    cd ml-course && build-course"
echo "    cd stochastic-course && build-course"
