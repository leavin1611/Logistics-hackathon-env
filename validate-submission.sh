#!/usr/bin/env bash
# validate-submission.sh
set -uo pipefail

DOCKER_BUILD_TIMEOUT=600
RED='' GREEN='' YELLOW='' BOLD='' NC=''

run_with_timeout() {
  local secs="$1"; shift
  "$@"
}

portable_mktemp() {
  local prefix="${1:-validate}"
  mktemp "${TMPDIR:-/tmp}/${prefix}-XXXXXX" 2>/dev/null || mktemp
}

CLEANUP_FILES=()
cleanup() { rm -f "${CLEANUP_FILES[@]+"${CLEANUP_FILES[@]}"}"; }
trap cleanup EXIT

PING_URL="${1:-}"
REPO_DIR="${2:-.}"

if [ -z "$PING_URL" ]; then exit 1; fi

PING_URL="${PING_URL%/}"
export PING_URL
PASS=0

log()  { printf "[%s] %b\n" "$(date -u +%H:%M:%S)" "$*"; }
pass() { log "PASSED -- $1"; PASS=$((PASS + 1)); }
fail() { log "FAILED -- $1"; }
hint() { printf "  Hint: %b\n" "$1"; }
stop_at() {
  printf "\nValidation stopped at %s. Fix the above before continuing.\n" "$1"
  exit 1
}

log "Step 1/3: Pinging HF Space ($PING_URL/reset) ..."
CURL_OUTPUT=$(portable_mktemp "validate-curl")
CLEANUP_FILES+=("$CURL_OUTPUT")
HTTP_CODE=$(curl -s -L -o "$CURL_OUTPUT" -w "%{http_code}" -X POST -H "Content-Type: application/json" -d '{}' "$PING_URL/reset" --max-time 30 2>"$CURL_OUTPUT" || printf "000")

if [ "$HTTP_CODE" = "200" ]; then
  pass "HF Space is live and responds to /reset"
else
  fail "HF Space /reset returned HTTP $HTTP_CODE"
  stop_at "Step 1"
fi

log "Step 2/3: Running docker build ..."
if ! command -v docker &>/dev/null; then
  fail "docker command not found"
  stop_at "Step 2"
fi

BUILD_OK=false
BUILD_OUTPUT=$(docker build "$REPO_DIR" 2>&1) && BUILD_OK=true

if [ "$BUILD_OK" = true ]; then pass "Docker build succeeded"
else fail "Docker build failed"; stop_at "Step 2"; fi

log "Step 3/3: Running openenv validate ..."
VALIDATE_OK=false
VALIDATE_OUTPUT=$(cd "$REPO_DIR" && openenv validate 2>&1) && VALIDATE_OK=true

if [ "$VALIDATE_OK" = true ]; then pass "openenv validate passed"
else fail "openenv validate failed"; stop_at "Step 3"; fi

printf "\n  All 3/3 checks passed!\n  Your submission is ready to submit.\n"
exit 0
