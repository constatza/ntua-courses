#!/bin/sh
# Checks the failure guarantees of tools/ in a throwaway repo. Uses plain `article` documents,
# a dummy class and a stub `gh`, so it needs only latexmk and git.
set -eu

tools=$(CDPATH='' cd -- "$(dirname "$0")/../tools" && pwd)
work=$(mktemp -d "${TMPDIR:-/tmp}/tools-smoke.XXXXXX")
trap 'rm -rf "$work"' EXIT

fail() { echo "FAIL: $*" >&2; exit 1; }
tex() { printf '\\documentclass{article}\\begin{document}%s\\end{document}\n' "$1"; }

mkdir -p "$work/origin.git" "$work/repo"
git -C "$work/origin.git" init -q --bare
cd "$work/repo"
git init -q -b main
git remote add origin "$work/origin.git"
cp -r "$tools" tools
mkdir -p beamer-ntua/packages/ntuabeamer-class/framework c/presentations/a c/exercises/e
printf '\\ProvidesClass{ntuabeamer}\\LoadClass{article}\n' >beamer-ntua/packages/ntuabeamer-class/framework/ntuabeamer.cls
tex ok >c/presentations/a/main.tex
tex sheet >c/exercises/e/main.tex
tex solutions >c/exercises/e/answers.tex
printf '[topics.t]\nowner = "presentations/a"\nexercises = ["exercises/e"]\n' >c/TOPICS.toml
git add -A && git -c user.name=t -c user.email=t@t commit -qm init
git tag c/v1 && git push -q origin main c/v1

# 1. good build lists exactly the built PDFs
tools/build-course c >/dev/null || fail "good build failed"
want="exercise_e.pdf
exercise_e_answers.pdf
presentation_a.pdf"
[ "$(sort c/dist/MANIFEST)" = "$(echo "$want" | sort)" ] || fail "manifest: $(cat c/dist/MANIFEST)"

[ -s c/dist/TEXLIVE ] || fail "TeX Live version not recorded in dist/"

# 2. a broken source leaves no dist/ and no stale .build PDF
printf '\\documentclass{article}\\begin{document}\\undefinedcommand\\end{document}\n' >c/presentations/a/main.tex
tools/build-course c >/dev/null && fail "broken build reported success"
[ ! -e c/dist ] || fail "dist/ survived a failed build"
[ ! -e c/presentations/a/.build/main.pdf ] || fail "stale PDF survived a failed build"
tex ok >c/presentations/a/main.tex

# 3. TOPICS.toml naming a missing path fails the build
printf '[topics.u]\nowner = "exercises/ghost"\n' >>c/TOPICS.toml
tools/build-course c >/dev/null && fail "dangling TOPICS.toml path accepted"
git checkout -q c/TOPICS.toml

# 3b. a tracked target directory without main.tex fails the layout check
mkdir -p c/presentations/empty && : >c/presentations/empty/notes.txt && git add c/presentations/empty
tools/check-course-layout c >/dev/null && fail "target without main.tex accepted"
git rm -qf c/presentations/empty/notes.txt

# 4. tools/ci --tag validation
tools/ci --tag nope >/dev/null 2>&1 && fail "tag without a prefix accepted"
tools/ci --tag ghost/v1 >/dev/null 2>&1 && fail "tag with an unknown course prefix accepted"
tools/ci --tag c/v1 >/dev/null || fail "tools/ci --tag c/v1 failed"

# 5. release with a stub gh: answers are excluded, draft is published
mkdir "$work/bin"
cat >"$work/bin/gh" <<'STUB'
#!/bin/sh
echo "$*" >>"$GH_LOG"
case "$1 $2" in
  "release view") [ -f "$GH_STATE" ] && { cat "$GH_STATE"; exit 0; } || exit 1 ;;
  "release create") echo true >"$GH_STATE" ;;
  "release edit") [ -z "${GH_FAIL_EDIT:-}" ] || exit 1; echo false >"$GH_STATE" ;;
  "release delete") rm -f "$GH_STATE" ;;
esac
exit 0
STUB
chmod +x "$work/bin/gh"
export GH_LOG="$work/gh.log" GH_STATE="$work/gh.state" PATH="$work/bin:$PATH"
tools/release-course c c/v1 >/dev/null || fail "release failed"
grep -q 'release create c/v1 --draft' "$GH_LOG" || fail "release not created as draft"
grep -q 'presentation_a.pdf' "$GH_LOG" || fail "deck asset missing"
! grep -q '_answers.pdf' "$GH_LOG" || fail "answers published by default"
[ "$(cat "$GH_STATE")" = false ] || fail "release not published"

# 6. published tag is refused; a failing publish deletes the draft and a rerun converges
tools/release-course c c/v1 >/dev/null 2>&1 && fail "re-release of a published tag accepted"
rm -f "$GH_STATE"
GH_FAIL_EDIT=1 tools/release-course c c/v1 >/dev/null 2>&1 && fail "failed publish reported success"
[ ! -f "$GH_STATE" ] || fail "draft left behind after failure"
echo true >"$GH_STATE"   # a stale draft from an interrupted earlier run
tools/release-course c c/v1 >/dev/null || fail "retry after a stale draft failed"
[ "$(cat "$GH_STATE")" = false ] || fail "retry did not publish"
tools/release-course c c/v9 >/dev/null 2>&1 && fail "tag not on origin accepted"
tools/release-course c d/v1 >/dev/null 2>&1 && fail "tag with a foreign prefix accepted"

echo "tools smoke: ok"
