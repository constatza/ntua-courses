#!/bin/sh
# Bootstrap ntua-courses after clone.
#
#   git clone <repo> ntua-courses && cd ntua-courses && sh scripts/bootstrap.sh
#
# Initializes submodules at their pinned commits and installs the
# ntuabeamer beamer class (to TEXMFHOME) and build-course/release-course
# (to ~/.local/bin) from the pinned beamer-ntua checkout, so every deck builds.
set -eu

echo "==> syncing submodule URLs"
git submodule sync

echo "==> initializing submodules"
git submodule update --init --recursive

echo "==> installing ntuabeamer beamer class (to TEXMFHOME)"
# Install from the pinned checkout, not from main, so the class and the tools match the pin.
NTUA_BEAMER_BASE_URL="file://$PWD/beamer-ntua" sh beamer-ntua/scripts/install-class.sh

echo "==> done. Make sure ~/.local/bin is on PATH, then build a course with:"
echo "    cd ml-course && build-course"
echo "    cd stochastic-course && build-course"
