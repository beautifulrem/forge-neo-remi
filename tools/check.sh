#!/usr/bin/env bash
# Every quality gate of the Remi WebUI in one command (scorecard dimension 8).
#
#   tools/check.sh                    # everything
#   tools/check.sh --skip e2e         # leave steps out (comma-separated names or prefixes)
#   tools/check.sh --only ruff,pytest # run only these steps (prefixes: "pytest" = core + extensions)
#   tools/check.sh --list             # list the steps
#   tools/check.sh --strict           # a skipped step (a missing optional interpreter) fails the run
#   tools/check.sh --serial           # one step at a time, to rule out two steps getting in each other's way
#
# Steps: a Gradio-free Python - ruff (Remi code strict, test suites, the built-in extensions'
# scripts and libraries, the upstream-origin runtime, the whole tree for syntax errors and
# Gradio imports) - pytest for
# tests/ (with a coverage floor for remi/) and for every extensions-builtin/*/tests directory
# separately, then a coverage floor for each Remi file outside remi/ over all of them - OpenAPI,
# /sdapi/v1 (against Forge Neo's; skipped, with the reason, when a shallow clone cannot fetch the
# upstream commit), settings and tab schema contracts, upstream line endings - the legacy ui() recorder's self-test
# (skipped, with the reason, without its Gradio interpreter) - i18n coverage (server zh_CN catalog, SPA
# chrome catalog) - pnpm install (frozen lockfile), build (tsc + vite), the API docs viewers that
# build vendors, the JS bundle budget, oxlint, vitest (with coverage floors), Playwright end-to-end
# tests and axe scans against the mock backend.
# Steps keep running after a failure; the script ends with a summary table and exits 1
# when any step failed (or, with --strict, was skipped).
# Three lanes run at once, each one step after another: python (the Gradio check, ruff,
# pytest:tests), suites (the extension suites, the contracts, the legacy ui() recorder, i18n:server)
# and frontend (pnpm to e2e). pytest:coverage combines the coverage data of every pytest step, so
# it runs once both Python lanes have ended. Each step prints its line when it ends; the summary
# lists the steps in the order above. Stopping the run (Ctrl-C, SIGTERM) stops every lane and the
# steps they run, and the script ends by that signal; killed outright (SIGKILL), its lanes start
# no further step (the running ones go on, as a step did when one ran at a time).
#
# Python: $PYTHON, else venv/ or .venv/ in the repo, else python3 - the first one with
# requirements.txt and requirements-dev.txt (pytest, pytest-cov) installed, preferring one
# without Gradio (env:gradio-free fails when the chosen one can import Gradio: the WebUI
# must start without it). Ruff is taken from that environment, then PATH, then `uvx ruff`.
# COMMANDLINE_ARGS is unset for every step (modules.paths_internal would add it to the WebUI's
# command line). Logs: $REMI_CHECK_LOGS (default a temporary directory, printed at the end). The
# real-model smoke test is separate: tools/smoke_real.py.
# legacy-ui (the legacy ui() recorder's self-test) needs the recorder's own Gradio 4.40 interpreter,
# built once with the network by tools/legacy_ui_env.sh: $REMI_LEGACY_PYTHON, else the one that
# script builds by default (../remi-gradio-env/bin/python). Without it the step is reported as
# "skip" with the reason (a failure when named in --only, or with --strict). So does
# legacy-ui:fixtures, which records every committed legacy ui() fixture again and compares it.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 2
# modules.paths_internal adds COMMANDLINE_ARGS to the WebUI's command line: a --data-dir a shell
# exports would give the test suites the owner's settings file.
unset COMMANDLINE_ARGS

ONLY=""
SKIP=""
LIST=0
STRICT=0
SERIAL=0
while [ $# -gt 0 ]; do
  case "$1" in
    --strict) STRICT=1; shift ;;
    --serial) SERIAL=1; shift ;;
    --only) [ $# -ge 2 ] || { echo "--only needs a value (see --help)" >&2; exit 2; }; ONLY="$2"; shift 2 ;;
    --only=*) ONLY="${1#*=}"; shift ;;
    --skip) [ $# -ge 2 ] || { echo "--skip needs a value (see --help)" >&2; exit 2; }; SKIP="$2"; shift 2 ;;
    --skip=*) SKIP="${1#*=}"; shift ;;
    --list) LIST=1; shift ;;
    -h|--help) awk 'NR == 1 { next } /^#/ { sub(/^# ?/, ""); print; next } { exit }' "$0"; exit 0 ;;
    *) echo "unknown argument: $1 (see --help)" >&2; exit 2 ;;
  esac
done

# region Environment

GRADIO_PROBE='import importlib.util, sys; found = [name for name in ("gradio", "gradio_client", "gradio_rangeslider") if importlib.util.find_spec(name)]; print(" ".join(found)); sys.exit(1 if found else 0)'

pick_python() {
  # $PYTHON is used as given; otherwise the first candidate without Gradio, else the first one.
  local candidate fallback=""
  if [ -n "${PYTHON:-}" ]; then
    echo "$PYTHON"
    return 0
  fi
  for candidate in "$ROOT/venv/bin/python" "$ROOT/.venv/bin/python" "$(command -v python3 || true)"; do
    [ -n "$candidate" ] && [ -x "$candidate" ] || continue
    "$candidate" -c "import fastapi, pytest, pytest_cov, torch" >/dev/null 2>&1 || continue
    if "$candidate" -c "$GRADIO_PROBE" >/dev/null 2>&1; then
      echo "$candidate"
      return 0
    fi
    [ -n "$fallback" ] || fallback="$candidate"
  done
  [ -n "$fallback" ] && echo "$fallback"
}

# Python steps fail (with this message) when there is none; frontend-only runs do not need it.
PY="$(pick_python)"
py() {
  if [ -z "$PY" ]; then
    echo "No Python with the WebUI requirements and pytest found: set PYTHON=/path/to/python" >&2
    return 2
  fi
  "$PY" "$@"
}

if [ -n "$PY" ] && "$PY" -m ruff --version >/dev/null 2>&1; then
  RUFF=("$PY" -m ruff)
elif command -v ruff >/dev/null 2>&1; then
  RUFF=(ruff)
else
  RUFF=(uvx ruff)
fi

# The legacy ui() recorder's Gradio 4.40 interpreter (tools/legacy_ui_env.sh's default location).
LEGACY_DEFAULT="$(cd "$ROOT/.." && pwd)/remi-gradio-env/bin/python"
if [ -n "${REMI_LEGACY_PYTHON:-}" ]; then
  LEGACY_PY="$REMI_LEGACY_PYTHON"
elif [ -x "$LEGACY_DEFAULT" ]; then
  LEGACY_PY="$LEGACY_DEFAULT"
else
  LEGACY_PY=""
fi

LOGS="${REMI_CHECK_LOGS:-$(mktemp -d "${TMPDIR:-/tmp}/remi-check.XXXXXX")}"
mkdir -p "$LOGS"
# pytest-cov's data file goes with the logs, not into the repository.
export COVERAGE_FILE="$LOGS/.coverage"
# Port of the e2e test's own mock dev server: away from a developer's `pnpm dev` (5173/5199).
export REMI_E2E_PORT="${REMI_E2E_PORT:-5299}"
# A real-server try (playwright.config.ts) would replace the mock suite with the few @real tests.
unset REMI_E2E_BASE_URL REMI_E2E_GREP
export PYTHONDONTWRITEBYTECODE=1

# endregion

# region Step bookkeeping

DECLARED=() # every step name, run or not (to report --only/--skip items that name no step)
# The steps to run, in the order declared: name, lane, and the command (quoted for eval).
QUEUED_NAMES=()
QUEUED_LANES=()
QUEUED_CMDS=()
LANE=python # the lane of the steps declared next: python, suites, after or frontend

matches() { # matches <name> <comma list>: exact name or prefix
  local item
  local IFS=,
  for item in $2; do
    [ -n "$item" ] || continue
    case "$1" in "$item"|"$item":*|"$item"-*) return 0 ;; esac
  done
  return 1
}

wanted() {
  if [ -n "$ONLY" ] && ! matches "$1" "$ONLY"; then return 1; fi
  if [ -n "$SKIP" ] && matches "$1" "$SKIP"; then return 1; fi
  return 0
}

# A step's command exits with this status when it cannot run here (tools/sdapi_contract.py: the
# upstream revision is not in a shallow clone and could not be fetched): the step is reported as
# "skip" with the last line of its output as the reason (--strict counts it as failed).
SKIP_STATUS=77

log_of() { echo "$LOGS/${1//[:\/]/_}.log"; }

# step <name> <command...>: declares a step of the current $LANE; it runs with its output in
# $LOGS/<name>.log
step() {
  local name="$1"
  shift
  DECLARED+=("$name")
  if [ "$LIST" = 1 ]; then
    echo "$name"
    return 0
  fi
  wanted "$name" || return 0
  QUEUED_NAMES+=("$name")
  QUEUED_LANES+=("$LANE")
  QUEUED_CMDS+=("$(printf '%q ' "$@")")
}

skip_because() { # the command of a step that cannot run here: its reason, then the skip status
  echo "$1"
  return $SKIP_STATUS
}

# skip_step <name> <reason>: a step that cannot run here (a missing optional interpreter). It is
# still listed, and the summary reports it as "skip" with the reason; --strict counts it as failed.
skip_step() { step "$1" skip_because "$2"; }

# pytest that also fails when nothing was collected (pytest exits 5).
run_pytest() {
  py -m pytest "$@" -q -p no:cacheprovider -W ignore::DeprecationWarning
}

# endregion

# region Python

gradio_free() {
  local found
  if found="$(py -c "$GRADIO_PROBE")"; then
    echo "$PY cannot import Gradio"
    return 0
  fi
  [ -n "$found" ] || return 2
  echo "$PY can import: $found" >&2
  echo "The WebUI is Gradio-free; run the checks with an interpreter without Gradio (set PYTHON=...) so a Gradio import cannot hide." >&2
  return 1
}

step env:gradio-free gradio_free

# Which rules apply to which file is pyproject.toml's business (strict for Remi's files).
step ruff:remi "${RUFF[@]}" check remi tools webui/tools docs tests/remi_*.py tests/test_remi_*.py modules/extra_networks_pages.py modules/redact.py
step ruff:tests "${RUFF[@]}" check tests extensions-builtin/*/tests
# also the modules at an extension's root (install.py, preload.py, sd_forge_lora/networks.py, ...) and joycaption_lib/
step ruff:extensions "${RUFF[@]}" check extensions-builtin/*/scripts extensions-builtin/*/lib_* extensions-builtin/*/*.py extensions-builtin/sd-webui-joycaption-modified/joycaption_lib
# The upstream-origin runtime Remi edits (legacy rules); modules_forge/packages is vendored as is.
step ruff:core "${RUFF[@]}" check modules modules_forge backend scripts launch.py webui.py --extend-exclude modules_forge/packages
step ruff:syntax "${RUFF[@]}" check . --select E9,TID251 --output-format concise

# Coverage floors (tools/check_coverage.py): remi/ as a whole (the measured line coverage of
# tests/, minus a small margin), and each Remi file outside remi/ on its own (pyproject.toml's
# strict list: modules/redact.py, the ControlNet unit editor, ...). Those files are tested by
# different suites, so every pytest step measures them into a data file of its own
# ($LOGS/.coverage.pytest_<name>) and pytest:coverage combines the data of this log folder.
COVERAGE_FLOOR_PYTHON=92
COVERAGE_FLOOR_STRICT=90
coverage_data() { echo "$LOGS/.coverage.${1//[:\/]/_}"; }
# pytest_cov <step name> <pytest args...>: run_pytest measuring the Remi files outside remi/
pytest_cov() {
  local data sources
  data="$(coverage_data "$1")"
  shift
  sources="$(py tools/check_coverage.py sources)" || return 2
  rm -f "$data"
  # shellcheck disable=SC2086 # one --cov=<source> per word
  COVERAGE_FILE="$data" run_pytest "$@" $sources --cov-report=
}
pytest_tests() {
  local status=0
  # -m: the spa_build tests read webui/dist, which pnpm:build (another lane) empties while it runs
  pytest_cov pytest:tests tests -m "not spa_build" --cov=remi || status=$?
  py tools/check_coverage.py remi --data "$(coverage_data pytest:tests)" --floor "$COVERAGE_FLOOR_PYTHON" || status=$((status ? status : 1))
  return $status
}
step pytest:tests pytest_tests
LANE=suites
for dir in extensions-builtin/*/tests; do
  [ -d "$dir" ] || continue
  ext="${dir#extensions-builtin/}"
  step "pytest:${ext%/tests}" pytest_cov "pytest:${ext%/tests}" "$dir"
done
strict_coverage() {
  py tools/check_coverage.py strict --floor "$COVERAGE_FLOOR_STRICT" --data "$LOGS"/.coverage.pytest_*
}
LANE=after
step pytest:coverage strict_coverage
LANE=suites

step contract:openapi py tools/export_openapi.py --check
# /sdapi/v1 against Forge Neo's at the last absorbed commit; a shallow clone (CI) fetches that
# commit first, and only when it cannot is the step a "skip" with the reason (also in pytest:tests).
step contract:sdapi py tools/sdapi_contract.py
step contract:settings py tools/dump_settings.py --check
# The real tab schemas the SPA contract tests (webui/src/ui/contract.test.tsx) render.
step contract:schemas py webui/tools/dump_schemas.py --check
# Files shared with Forge Neo keep upstream's CRLF line endings (clean merges of Neo updates).
step contract:line-endings py tools/check_line_endings.py
# The legacy ui() recorder end to end: records a known script in the last Gradio-era revision and
# compares the record with its committed fixtures (docs/remi-webui/upstream-absorption.md §2.5).
# Its interpreter is optional (Gradio 4.40, built once with the network): without one the step is
# a "skip" with the reason, unless --only names it (then it fails with the same message).
LEGACY_MISSING="no Gradio 4.40 interpreter: build it with tools/legacy_ui_env.sh (to $LEGACY_DEFAULT) or set REMI_LEGACY_PYTHON"
legacy_ui_self_test() {
  if [ -z "$LEGACY_PY" ]; then
    echo "$LEGACY_MISSING" >&2
    return 2
  fi
  "$LEGACY_PY" tools/record_legacy_ui.py --self-test --report "$LOGS/legacy-ui.json"
}
if [ -n "$LEGACY_PY" ] || { [ -n "$ONLY" ] && matches legacy-ui "$ONLY"; }; then
  step legacy-ui legacy_ui_self_test
else
  skip_step legacy-ui "$LEGACY_MISSING"
fi
# Every committed legacy ui() fixture recorded again in that revision, each with its format and the
# WebUI context its ui() needs (tools/legacy_ui_recordings.py), and compared value by value.
legacy_ui_fixtures() {
  if [ -z "$LEGACY_PY" ]; then
    echo "$LEGACY_MISSING" >&2
    return 2
  fi
  "$LEGACY_PY" tools/record_legacy_ui.py --fixtures --check --report "$LOGS/legacy-ui-fixtures.json"
}
if [ -n "$LEGACY_PY" ] || { [ -n "$ONLY" ] && matches legacy-ui:fixtures "$ONLY"; }; then
  step legacy-ui:fixtures legacy_ui_fixtures
else
  skip_step legacy-ui:fixtures "$LEGACY_MISSING"
fi

step i18n:server run_pytest tests/test_remi_core_zh_cn.py

# endregion

# region Frontend

pnpm_in_webui() { (cd "$ROOT/webui" && pnpm "$@"); }

LANE=frontend
step pnpm:install pnpm_in_webui install --frozen-lockfile
step i18n:spa pnpm_in_webui exec vitest run src/locales
step pnpm:build pnpm_in_webui build
# The tests of that build (pytest marker spa_build): the API docs viewers it vendors
# (webui/vite.config.ts's apiDocs), as the server reads them (remi/server/api_docs.py). Right after
# the build a missing viewer fails instead of skipping (REMI_SPA_BUILT).
api_docs() { (export REMI_SPA_BUILT=1 && run_pytest tests -m spa_build); }
step api-docs api_docs
# The JS budget (scorecard dimension 5): a manifest build of the same sources, checked against
# webui/tools/perf/budget.json (initial JS, every route and lazy entry, recorded exceptions).
bundle_budget() {
  local out="$LOGS/bundle-dist"
  pnpm_in_webui exec vite build --manifest --outDir "$out" --emptyOutDir --logLevel warn &&
    (cd "$ROOT/webui" && node tools/perf/bundle.mjs "$out" --check --json "$LOGS/bundle.json")
}
step perf:bundle bundle_budget
step pnpm:lint pnpm_in_webui lint
step pnpm:vitest pnpm_in_webui exec vitest run --coverage
# Locally the line reporter keeps the log short; in CI (playwright.config.ts: any non-empty CI)
# the config's own reporters run, which also write the HTML report the workflow uploads.
E2E_REPORTER=()
[ -n "${CI:-}" ] || E2E_REPORTER=(--reporter=line)
step e2e pnpm_in_webui exec playwright test ${E2E_REPORTER[@]+"${E2E_REPORTER[@]}"}

# endregion

[ "$LIST" = 1 ] && exit 0

# region Run

STATE="$LOGS/.steps" # one file per run step: result, seconds, skip reason
rm -rf "$STATE"
mkdir -p "$STATE"

# run_step <index>: runs a queued step, records its result and prints its line
run_step() {
  local name="${QUEUED_NAMES[$1]}" log start seconds status result line reason=""
  log="$(log_of "$name")"
  start=$(date +%s)
  eval "${QUEUED_CMDS[$1]}" >"$log" 2>&1 </dev/null
  status=$?
  seconds=$(($(date +%s) - start))
  if [ $status -eq 0 ]; then
    result=pass
    line="pass (${seconds}s)"
  elif [ $status -eq $SKIP_STATUS ]; then
    result=skip
    reason="$(tail -n 1 "$log")"
    reason="${reason:-exit $SKIP_STATUS}"
    line="skip ($reason)"
  else
    result=FAIL
    line="FAIL (${seconds}s, exit $status)"
  fi
  printf '%s\t%s\t%s\n' "$result" "$seconds" "$reason" >"$STATE/$1"
  printf '==> %-44s %s\n' "$name" "$line"
}

MAIN_PID=$$
# Whether the script itself still runs: killed outright (SIGKILL, no trap runs), it leaves its lanes
# behind, and each would go on with its queue. A killed process its parent has not reaped yet is a
# zombie, which kill -0 still finds.
main_running() {
  kill -0 "$MAIN_PID" 2>/dev/null || return 1
  command -v ps >/dev/null || return 0
  case "$(ps -o stat= -p "$MAIN_PID" 2>/dev/null)" in *Z*) return 1 ;; esac
  return 0
}

run_lane() { # run_lane <lane>: its steps, one after another, while the script runs
  local i=0
  while [ $i -lt ${#QUEUED_NAMES[@]} ]; do
    if [ "${QUEUED_LANES[$i]}" = "$1" ]; then
      main_running || return 0
      run_step "$i"
    fi
    i=$((i + 1))
  done
}

descendants() { # descendants <pid>: every process under it
  command -v pgrep >/dev/null || return 0
  local child
  for child in $(pgrep -P "$1"); do
    echo "$child"
    descendants "$child"
  done
}

# Stopped, the script stops its lanes and every step they run (a step would go on in the background).
stop_lanes() {
  local lane pids=""
  for lane in $(jobs -pr); do
    pids="$pids $lane $(descendants "$lane")"
  done
  # shellcheck disable=SC2086 # one word per process
  [ -z "${pids// /}" ] || kill $pids 2>/dev/null
}
# Then it ends by that signal, as a stopped command does (a shell loop running it stops at Ctrl-C).
trap 'stop_lanes; trap - INT; kill -INT $$' INT
trap 'stop_lanes; trap - TERM; kill -TERM $$' TERM

# The lanes run in the background and the script waits for them, so a signal stops it at once.
if [ ${#QUEUED_NAMES[@]} -gt 0 ]; then
  if [ "$SERIAL" = 1 ]; then
    echo "Running ${#QUEUED_NAMES[@]} steps one at a time; logs in $LOGS"
    for lane in python suites after frontend; do
      run_lane "$lane" &
      wait $!
    done
  else
    echo "Running ${#QUEUED_NAMES[@]} steps in three lanes at once (python, suites, frontend); logs in $LOGS"
    run_lane python &
    python_lane=$!
    run_lane suites &
    suites_lane=$!
    run_lane frontend &
    frontend_lane=$!
    wait $python_lane $suites_lane
    run_lane after &
    wait $!
    wait $frontend_lane
  fi
fi

# endregion

# region Summary

NAMES=()
RESULTS=()
TIMES=()
SKIPPED=() # "<name>: <reason>" of each skipped step
i=0
while [ $i -lt ${#QUEUED_NAMES[@]} ]; do
  name="${QUEUED_NAMES[$i]}"
  result=FAIL # a step that left no result did not end
  seconds=0
  reason=""
  if [ -f "$STATE/$i" ]; then
    IFS=$'\t' read -r result seconds reason <"$STATE/$i"
  fi
  NAMES+=("$name")
  RESULTS+=("$result")
  TIMES+=("$seconds")
  if [ "$result" = skip ]; then SKIPPED+=("$name: $reason"); fi
  if [ "$result" = FAIL ]; then
    echo
    echo "$name failed; the end of $(log_of "$name"):"
    tail -n 40 "$(log_of "$name")" 2>/dev/null | sed 's/^/    | /'
  fi
  i=$((i + 1))
done

failed=0
skipped=0
echo
printf '%-44s %-6s %8s\n' "step" "result" "seconds"
printf '%-44s %-6s %8s\n' "--------------------------------------------" "------" "--------"
for i in "${!NAMES[@]}"; do
  printf '%-44s %-6s %8s\n' "${NAMES[$i]}" "${RESULTS[$i]}" "${TIMES[$i]}"
  case "${RESULTS[$i]}" in
    pass) ;;
    skip) skipped=$((skipped + 1)) ;;
    *) failed=$((failed + 1)) ;;
  esac
done
echo
echo "Python: ${PY:-none found}"
echo "Legacy: ${LEGACY_PY:-none found}"
echo "Logs:   $LOGS"
if [ ${#NAMES[@]} -eq 0 ]; then
  echo "No step matched --only/--skip." >&2
  exit 2
fi
unmatched=() # a mistyped item (--only ruff,pytset) would otherwise just leave its steps out
for item in ${ONLY//,/ } ${SKIP//,/ }; do
  found=0
  for name in "${DECLARED[@]}"; do
    if matches "$name" "$item"; then found=1; break; fi
  done
  [ $found = 1 ] || unmatched+=("$item")
done
if [ ${#unmatched[@]} -gt 0 ]; then
  echo "No step is named (or prefixed) ${unmatched[*]} (see --list)." >&2
  exit 2
fi
if [ $skipped -gt 0 ]; then
  echo "Skipped (not run, not passed):"
  printf '  %s\n' "${SKIPPED[@]}"
fi
passed=$((${#NAMES[@]} - failed - skipped))
if [ $failed -gt 0 ]; then
  echo "$failed of ${#NAMES[@]} steps failed ($passed passed, $skipped skipped)."
  exit 1
fi
if [ $skipped -gt 0 ]; then
  echo "$passed of ${#NAMES[@]} steps passed, $skipped skipped."
  [ "$STRICT" = 1 ] && { echo "--strict: a skipped step fails the run." >&2; exit 1; }
  exit 0
fi
echo "All ${#NAMES[@]} steps passed."

# endregion
