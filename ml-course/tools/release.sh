#!/bin/sh
# Create a GitHub release for machine-learning course PDFs.
#
# Usage: ./tools/release.sh <tag>
#   e.g. ./tools/release.sh v0.1.0-alpha.1
#
# Preconditions (checked):
#   - working tree clean, on branch main
#   - gh authenticated
#   - tools/build-all passes (every target "ok")
#   - tag does not already exist locally or on origin
#
# The script builds every deck, collects the final PDFs from each
# presentations/<slug>/.build/main.pdf, tags the commit, pushes the tag,
# and creates a GitHub release (prerelease for alpha/rc tags) with the
# PDFs as release assets. Built PDFs are never committed to git.
set -eu

TAG="${1:?usage: $0 <tag>  (e.g. v0.1.0-alpha.1)}"
REPO="machine-learning"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STAGE="$(mktemp -d -t ml-release-XXXXXX)"
NOTES="$STAGE/release-notes.md"
ASSETS=""

cleanup() {
  rm -rf "$STAGE"
}
trap cleanup EXIT

# --- preconditions ---------------------------------------------------------
cd "$ROOT"

if [ "$(git branch --show-current)" != "main" ]; then
  echo "error: must be on branch main (currently on $(git branch --show-current))" >&2
  exit 1
fi

if [ -n "$(git status --porcelain)" ]; then
  echo "error: working tree is not clean; commit or stash changes first" >&2
  git status --short >&2
  exit 1
fi

if ! gh auth status >/dev/null 2>&1; then
  echo "error: gh is not authenticated; run 'gh auth login'" >&2
  exit 1
fi

if git rev-parse "refs/tags/$TAG" >/dev/null 2>&1; then
  echo "error: tag $TAG already exists locally" >&2
  exit 1
fi

if git ls-remote --exit-code --tags origin "refs/tags/$TAG" >/dev/null 2>&1; then
  echo "error: tag $TAG already exists on origin" >&2
  exit 1
fi

# --- build -----------------------------------------------------------------
echo "==> running tools/build-all"
if ! sh tools/build-all; then
  echo "error: build-all failed; aborting release" >&2
  exit 1
fi

# --- collect PDFs ----------------------------------------------------------
for d in presentations/*/; do
  pdf="$d.build/main.pdf"
  if [ -f "$pdf" ]; then
    slug="$(basename "$d")"
    cp "$pdf" "$STAGE/$slug.pdf"
    ASSETS="$ASSETS $STAGE/$slug.pdf"
    DECK_LIST="$DECK_LIST- $slug\n"
  fi
done

if [ -z "$ASSETS" ]; then
  echo "error: no PDFs found in presentations/*/.build/main.pdf" >&2
  exit 1
fi

SHORT="$(git rev-parse --short HEAD)"

# --- release notes ---------------------------------------------------------
{
  echo "# $REPO $TAG"
  echo ""
  echo "Built from \`$SHORT\`."
  echo ""
  echo "## Decks"
  echo ""
  # shellcheck disable=SC2059
  printf -- "$DECK_LIST"
  echo ""
  echo "## Install"
  echo ""
  echo "The decks use the \`ntuabeamer\` class from \`beamer-ntua\`. Install it once:"
  echo ""
  echo "\`\`\`sh"
  echo "curl -fsSL https://raw.githubusercontent.com/constatza/beamer-ntua/main/scripts/install-class.sh | sh"
  echo "\`\`\`"
  echo ""
  echo "## Assets"
  echo ""
  echo "Download the PDF for any deck from the release assets below. Each PDF is the"
  echo "output of \`latexmk\` for that deck; source lives in \`presentations/<deck>/\`."
} > "$NOTES"

# --- tag and push ----------------------------------------------------------
echo "==> tagging $TAG"
git tag -a "$TAG" -m "release($REPO): $TAG"

echo "==> pushing tag to origin"
git push origin "$TAG"

# --- create GitHub release -------------------------------------------------
is_prerelease="false"
case "$TAG" in
  *-alpha*|*-beta*|*-rc*) is_prerelease="true" ;;
esac

echo "==> creating GitHub release $TAG (prerelease=$is_prerelease)"
# shellcheck disable=SC2086
gh release create "$TAG" \
  --repo "constatza/$REPO" \
  --title "$REPO $TAG" \
  --notes-file "$NOTES" \
  $([ "$is_prerelease" = "true" ] && echo --prerelease) \
  $ASSETS

rm -rf "$STAGE"

echo "==> done: https://github.com/constatza/$REPO/releases/tag/$TAG"
