#!/bin/sh
# Bootstrap ntua-courses after clone.
#
#   git clone <repo> ntua-courses && cd ntua-courses && sh scripts/bootstrap.sh
#
# Initializes submodules at their pinned commits and installs the
# ntuabeamer beamer class (once, to TEXMFHOME) so every deck builds.
set -eu

echo "==> syncing submodule URLs"
git submodule sync

echo "==> initializing submodules"
git submodule update --init --recursive

echo "==> installing ntuabeamer beamer class (to TEXMFHOME)"
sh beamer-ntua/scripts/install-class.sh

echo "==> done. Build a course with:"
echo "    cd machine-learning && sh tools/build-all"
echo "    cd stochastic-methods && sh tools/build-all"
