#!/bin/sh
# Create a GitHub release for stochastic-methods course PDFs.
#
# Usage: ./tools/release.sh <tag>
#   e.g. ./tools/release.sh v0.1.0-alpha.1
#
# Preconditions (checked):
#   - working tree clean, on branch main
#   - gh authenticated
#   - tools/build-all passes (every target "ok")
#   - tag does not already exist locally or on origin
#   - every deck/note in presentations/ and notes/ has a built .build/main.pdf
#
# The script builds every deck and note, collects the final PDFs, tags the
# commit, pushes the tag, and creates a GitHub release (prerelease for
# alpha/rc tags) with the PDFs as release assets. Built PDFs are never
# committed to git. If the release creation fails after the tag is pushed,
# the tag is rolled back locally and remotely so no invalid release state
# remains.
set -eu

TAG="${1:?usage: $0 <tag>  (e.g. v0.1.0-alpha.1)}"
REPO="stochastic-methods"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STAGE="$(mktemp -d -t sm-release-XXXXXX)"
NOTES="$STAGE/release-notes.md"
DECKS="$STAGE/decks.txt"
ASSETS=""
TAG_PUSHED=0

cleanup() {
  rm -rf "$STAGE"
  # Rollback: if we pushed a tag but release creation failed, remove it.
  if [ "$TAG_PUSHED" = "1" ]; then
    echo "==> rolling back tag $TAG (release creation failed)"
    git tag -d "$TAG" >/dev/null 2>&1 || true
    git push origin --delete "$TAG" >/dev/null 2>&1 || true
  fi
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

# --- expected-deck check: every presentation/note must have a built PDF ---
MISSING=""
for d in presentations/*/ notes/*/; do
  slug="$(basename "$d")"
  pdf="$d.build/main.pdf"
  if [ ! -f "$pdf" ]; then
    MISSING="$MISSING $slug"
  fi
done

if [ -n "$MISSING" ]; then
  echo "error: missing built PDFs for:$MISSING" >&2
  exit 1
fi

# --- collect PDFs ----------------------------------------------------------
collect_pdfs() {
  dir="$1"
  for d in "$dir"/*/; do
    slug="$(basename "$d")"
    name="$(basename "$dir")"
    cp "$d.build/main.pdf" "$STAGE/${name}_${slug}.pdf"
    ASSETS="$ASSETS $STAGE/${name}_${slug}.pdf"
    echo "- ${name}/${slug}" >> "$DECKS"
  done
}

collect_pdfs presentations
collect_pdfs notes
collect_pdfs exercises

if [ -z "$ASSETS" ]; then
  echo "error: no PDFs found" >&2
  exit 1
fi

SHORT="$(git rev-parse --short HEAD)"

# --- release notes ---------------------------------------------------------
{
  echo "# $REPO $TAG"
  echo ""
  echo "Built from \`$SHORT\`."
  echo ""
  echo "## Decks and notes"
  echo ""
  cat "$DECKS"
  echo ""
  echo "## Install"
  echo ""
  echo "Decks use the \`ntuabeamer\` class from \`beamer-ntua\`. Install it once:"
  echo ""
  echo "\`\`\`sh"
  echo "curl -fsSL https://raw.githubusercontent.com/constatza/beamer-ntua/main/scripts/install-class.sh | sh"
  echo "\`\`\`"
  echo ""
  echo "## Assets"
  echo ""
  echo "Download the PDF for any deck or note from the release assets below. Each PDF"
  echo "is the output of \`latexmk\`; source lives in \`presentations/<slug>/\` or"
  echo "\`notes/<slug>/\`."
} > "$NOTES"

# --- tag and push ----------------------------------------------------------
if [ -z "${CI:-}" ]; then
  echo "==> tagging $TAG"
  git tag -a "$TAG" -m "release($REPO): $TAG"

  echo "==> pushing tag to origin"
  git push origin "$TAG"
  TAG_PUSHED=1
else
  echo "==> running in CI; tag $TAG already pushed, skipping tag step"
fi

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

TAG_PUSHED=0  # release succeeded, no rollback needed
rm -rf "$STAGE"

echo "==> done: https://github.com/constatza/$REPO/releases/tag/$TAG"
