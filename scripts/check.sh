#!/bin/bash
# Quality gate for the XBrain repository.
# Runs 10 code-quality, type, security, and test checks via `uv run`.
#
# Can be used locally or in CI/CD (GitHub Actions).
#
# Exit codes:
#   0 - No critical check failed (warnings allowed)
#   1 - One or more CRITICAL checks failed
#
# Check severities:
#   CRITICAL (exit 1 on failure): ruff check, ruff format, mypy, bandit,
#                                 detect-secrets, pytest, coverage
#   radon: D/E/F = critical (exit 1), C = warn-only (never blocks)
#   WARN ONLY (never blocks)     : vulture, interrogate, deptry

set -euo pipefail

# Detect CI environment (used to guard the GITHUB_STEP_SUMMARY output)
IS_CI="${GITHUB_ACTIONS:-false}"

# Color codes: enabled only on an interactive TTY, with NO_COLOR unset, and
# not in CI. Anything else (pipes, files, CI logs) gets blank codes.
if [ -t 1 ] && [ -z "${NO_COLOR:-}" ] && [ "$IS_CI" != "true" ]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    CYAN='\033[0;36m'
    BOLD='\033[1m'
    NC='\033[0m'
else
    RED='' GREEN='' YELLOW='' CYAN='' BOLD='' NC=''
fi

# Track results
RUFF_STATUS="pending"
FORMAT_STATUS="pending"
MYPY_STATUS="pending"
RADON_STATUS="pending"
RADON_MAX_GRADE=""
BANDIT_STATUS="pending"
VULTURE_STATUS="pending"
INTERROGATE_STATUS="pending"
INTERROGATE_PCT=""
SECRETS_STATUS="pending"
DEPTRY_STATUS="pending"
TESTS_STATUS="pending"
TESTS_COUNT=""
COVERAGE_PCT=""
COVERAGE_STATUS="pending"
FAILED_CHECKS=""

# Coverage threshold (matches [tool.coverage] fail_under in pyproject.toml)
COVERAGE_MIN=78

# Functions
print_error() { echo -e "${RED}❌ $1${NC}"; }
print_success() { echo -e "${GREEN}✅ $1${NC}"; }
print_warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
print_info() { echo "🔍 $1"; }

mark_failed() {
    if [ -z "$FAILED_CHECKS" ]; then
        FAILED_CHECKS="$1"
    else
        FAILED_CHECKS="$FAILED_CHECKS, $1"
    fi
}

# ----------------------------------------------------------------------------
# Summary rendering
#
# Both the terminal summary and the GitHub markdown table are driven from ONE
# ordered list of check records, built by build_check_records(). Each record
# is a single line with five "|"-separated fields:
#
#   name | status | term_detail | gh_detail | severity
#
#     name        - display label (terminal pads it to a fixed width)
#     status      - resolved status: pass | warn | fail
#     term_detail - trailing detail shown in the terminal row ("" = none)
#     gh_detail   - "Details" cell text in the GitHub table
#     severity    - record category (informational; renderers key on status)
#
# The delimiter is "|" (not a tab) on purpose: `read` treats tab as IFS
# whitespace and would coalesce an empty term_detail field, dropping a column.
# No detail string contains "|". Adding a check means adding ONE record here;
# both renderers pick it up.
# ----------------------------------------------------------------------------

# Resolved status for radon (pass shows "All A/B", warn/fail show the grade).
radon_term_detail() {
    case "$RADON_STATUS" in
        pass) echo "All A/B" ;;
        *)    echo "Has grade ${RADON_MAX_GRADE}" ;;
    esac
}
radon_gh_detail() {
    case "$RADON_STATUS" in
        pass) echo "All functions grade A/B" ;;
        warn) echo "Has grade ${RADON_MAX_GRADE}" ;;
        *)    echo "Has grade ${RADON_MAX_GRADE} (D/E/F)" ;;
    esac
}

build_check_records() {
    # name | status | term_detail | gh_detail | severity
    printf '%s|%s|%s|%s|%s\n' \
        "Ruff (linting)" "$RUFF_STATUS" "" \
        "$([ "$RUFF_STATUS" = pass ] && echo 'Code is clean' || echo 'Linting errors')" "critical"
    printf '%s|%s|%s|%s|%s\n' \
        "Ruff (format)" "$FORMAT_STATUS" "" \
        "$([ "$FORMAT_STATUS" = pass ] && echo 'Code is formatted' || echo 'Formatting errors')" "critical"
    printf '%s|%s|%s|%s|%s\n' \
        "Mypy (types)" "$MYPY_STATUS" "" \
        "$([ "$MYPY_STATUS" = pass ] && echo 'Type checks passed' || echo 'Type errors found')" "critical"
    printf '%s|%s|%s|%s|%s\n' \
        "Radon (complexity)" "$RADON_STATUS" "$(radon_term_detail)" \
        "$(radon_gh_detail)" "critical"
    printf '%s|%s|%s|%s|%s\n' \
        "Bandit (security)" "$BANDIT_STATUS" "" \
        "$([ "$BANDIT_STATUS" = pass ] && echo 'No security issues' || echo 'Security issues found')" "critical"
    printf '%s|%s|%s|%s|%s\n' \
        "Vulture (dead code)" "$VULTURE_STATUS" "" \
        "$([ "$VULTURE_STATUS" = pass ] && echo 'No dead code' || echo 'Possible dead code')" "warn"
    printf '%s|%s|%s|%s|%s\n' \
        "Interrogate (docs)" "$INTERROGATE_STATUS" "${INTERROGATE_PCT}%" \
        "$([ "$INTERROGATE_STATUS" = pass ] && echo "${INTERROGATE_PCT}% coverage" || echo "${INTERROGATE_PCT}% (target 60%)")" "warn"
    printf '%s|%s|%s|%s|%s\n' \
        "Detect-secrets" "$SECRETS_STATUS" "" \
        "$([ "$SECRETS_STATUS" = pass ] && echo 'No secrets found' || echo 'Potential secrets detected')" "critical"
    printf '%s|%s|%s|%s|%s\n' \
        "Tests" "$TESTS_STATUS" \
        "$([ "$TESTS_STATUS" = pass ] && echo "${TESTS_COUNT} tests" || echo '')" \
        "$([ "$TESTS_STATUS" = pass ] && echo "${TESTS_COUNT:-All} tests passed" || echo 'Test failures')" "critical"
    if [ -n "$COVERAGE_PCT" ]; then
        local gh_cov
        case "$COVERAGE_STATUS" in
            pass) gh_cov="Above ${COVERAGE_MIN}% minimum" ;;
            warn) gh_cov="Within 5pts of ${COVERAGE_MIN}% minimum" ;;
            *)    gh_cov="Below ${COVERAGE_MIN}% minimum" ;;
        esac
        printf '%s|%s|%s|%s|%s\n' \
            "Coverage" "$COVERAGE_STATUS" "${COVERAGE_PCT}%" "$gh_cov" "coverage"
    fi
    printf '%s|%s|%s|%s|%s\n' \
        "Deptry (deps)" "$DEPTRY_STATUS" "" \
        "$([ "$DEPTRY_STATUS" = pass ] && echo 'Dependencies OK' || echo 'Dependency notes')" "warn"
}

print_summary() {
    echo ""
    echo -e "${BOLD}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${BOLD}                    QUALITY SUMMARY                         ${NC}"
    echo -e "${BOLD}═══════════════════════════════════════════════════════════${NC}"
    echo ""

    local name status term_detail gh_detail severity
    while IFS='|' read -r name status term_detail gh_detail severity; do
        [ -z "$name" ] && continue
        local icon color verdict
        if [ "$status" = "pass" ]; then
            icon="${GREEN}✅${NC}" color="$GREEN" verdict="PASS"
        elif [ "$status" = "warn" ]; then
            icon="${YELLOW}⚠️${NC} " color="$YELLOW" verdict="WARN"
        else
            icon="${RED}❌${NC}" color="$RED" verdict="FAIL"
        fi
        if [ "$name" = "Coverage" ]; then
            # Coverage row prints the percentage in place of PASS/WARN/FAIL.
            printf "  %b %-20s ${color}%s${NC}\n" "$icon" "$name" "$term_detail"
        elif [ -n "$term_detail" ]; then
            printf "  %b %-20s ${color}%s${NC} - %s\n" "$icon" "$name" "$verdict" "$term_detail"
        else
            printf "  %b %-20s ${color}%s${NC}\n" "$icon" "$name" "$verdict"
        fi
    done < <(build_check_records)

    echo ""
    echo -e "${BOLD}───────────────────────────────────────────────────────────${NC}"

    if [ -n "$FAILED_CHECKS" ]; then
        echo -e "  ${RED}${BOLD}FAILED${NC} - Critical issues in: ${RED}$FAILED_CHECKS${NC}"
    else
        echo -e "  ${GREEN}${BOLD}ALL CRITICAL CHECKS PASSED${NC}"
    fi
    echo -e "${BOLD}───────────────────────────────────────────────────────────${NC}"
    echo ""
}

print_github_summary() {
    [ -z "${GITHUB_STEP_SUMMARY:-}" ] && return

    {
        echo "## 🔍 Quality Gate Results"
        echo ""
        echo "| Check | Status | Details |"
        echo "|-------|--------|---------|"

        local name status term_detail gh_detail severity
        while IFS='|' read -r name status term_detail gh_detail severity; do
            [ -z "$name" ] && continue
            local cell
            if [ "$name" = "Coverage" ]; then
                # Coverage status cell carries an icon + the percentage.
                if [ "$status" = "pass" ]; then
                    cell="✅ ${term_detail}"
                elif [ "$status" = "warn" ]; then
                    cell="⚠️ ${term_detail}"
                else
                    cell="❌ ${term_detail}"
                fi
            elif [ "$status" = "pass" ]; then
                cell="✅ PASS"
            elif [ "$status" = "warn" ]; then
                cell="⚠️ WARN"
            else
                cell="❌ FAIL"
            fi
            echo "| ${name} | ${cell} | ${gh_detail} |"
        done < <(build_check_records)

        echo ""
        if [ -n "$FAILED_CHECKS" ]; then
            echo "### ❌ Failed: ${FAILED_CHECKS}"
        else
            echo "### ✅ All critical checks passed"
        fi
    } >> "$GITHUB_STEP_SUMMARY"
}

# ============================================================================
# 1. RUFF CHECK - Code linting  (CRITICAL)
# ============================================================================
print_info "Running Ruff linter on src/ and tests/..."
if uv run ruff check src tests 2>&1; then
    print_success "Ruff: code is clean"
    RUFF_STATUS="pass"
else
    print_error "Ruff: linting errors found"
    RUFF_STATUS="fail"
    mark_failed "Ruff"
fi

# ============================================================================
# 2. RUFF FORMAT - Formatting check  (CRITICAL)
# ============================================================================
print_info "Running Ruff format check on src/ and tests/..."
if uv run ruff format --check src tests 2>&1; then
    print_success "Ruff format: code is formatted"
    FORMAT_STATUS="pass"
else
    print_error "Ruff format: formatting errors found"
    FORMAT_STATUS="fail"
    mark_failed "Format"
fi

# ============================================================================
# 3. MYPY - Type checking  (CRITICAL)
# ============================================================================
print_info "Running Mypy type checker on src/xbrain..."
if uv run mypy src/xbrain 2>&1; then
    print_success "Mypy: type checks passed"
    MYPY_STATUS="pass"
else
    print_error "Mypy: type errors found"
    MYPY_STATUS="fail"
    mark_failed "Mypy"
fi

# ============================================================================
# 4. RADON - Complexity analysis  (WARN on C, FAIL on D/E/F)
# ============================================================================
print_info "Analyzing code complexity with Radon (src/xbrain)..."
# radon's exit code is not consulted — grade detection is done by grep below —
# so `|| true` keeps a radon error from aborting the whole script under set -e.
RADON_OUTPUT=$(uv run radon cc src/xbrain -s 2>&1) || true

if echo "$RADON_OUTPUT" | grep -qE ' - [D-F] \([0-9]+\)$'; then
    # Grade D/E/F is a hard failure
    RADON_MAX_GRADE=$(echo "$RADON_OUTPUT" | grep -oE ' - [D-F] \([0-9]+\)$' | grep -oE '[D-F]' | head -1)
    print_error "Radon: complexity grade ${RADON_MAX_GRADE} detected (D/E/F is not allowed)"
    echo "$RADON_OUTPUT" | grep -E ' - [D-F] \([0-9]+\)$'
    RADON_STATUS="fail"
    mark_failed "Radon"
elif echo "$RADON_OUTPUT" | grep -qE ' - C \([0-9]+\)$'; then
    # Grade C is a warning only
    RADON_MAX_GRADE="C"
    print_warning "Radon: grade C function(s) detected (warning only)"
    echo "$RADON_OUTPUT" | grep -E ' - C \([0-9]+\)$'
    RADON_STATUS="warn"
else
    print_success "Radon: all functions grade A/B"
    RADON_STATUS="pass"
fi

echo ""
echo -e "${BOLD}   Average complexity (src/xbrain):${NC}"
uv run radon cc src/xbrain -a -s | tail -1
echo ""

# ============================================================================
# 5. BANDIT - Security linting  (CRITICAL)
# ============================================================================
print_info "Running Bandit security linter (src/xbrain)..."
if uv run bandit -r src/xbrain -ll -q 2>&1; then
    print_success "Bandit: no security issues found"
    BANDIT_STATUS="pass"
else
    print_error "Bandit: security issues detected"
    BANDIT_STATUS="fail"
    mark_failed "Bandit"
fi

# ============================================================================
# 6. VULTURE - Dead code detection  (WARN ONLY)
# ============================================================================
print_info "Running Vulture dead code detection (src/xbrain)..."
VULTURE_OUTPUT=$(uv run vulture src/xbrain --min-confidence 80 2>&1) || true
if [ -z "$VULTURE_OUTPUT" ]; then
    print_success "Vulture: no dead code detected"
    VULTURE_STATUS="pass"
else
    print_warning "Vulture: possible dead code found (warning only)"
    echo "$VULTURE_OUTPUT"
    VULTURE_STATUS="warn"
fi

# ============================================================================
# 7. INTERROGATE - Docstring coverage  (WARN ONLY)
# ============================================================================
print_info "Running Interrogate docstring coverage (src/xbrain)..."
set +e
INTERROGATE_OUTPUT=$(uv run interrogate src/xbrain -v 2>&1)
INTERROGATE_EXIT=$?
set -e

# Extract coverage percentage from the "actual: NN.N%" field in one pass.
INTERROGATE_PCT=$(echo "$INTERROGATE_OUTPUT" | sed -nE 's/.*actual: ([0-9.]+).*/\1/p' | head -1)
INTERROGATE_PCT=${INTERROGATE_PCT:-0}

echo "$INTERROGATE_OUTPUT"

if [ "$INTERROGATE_EXIT" -eq 0 ]; then
    print_success "Interrogate: docstring coverage ${INTERROGATE_PCT}% (meets soft target)"
    INTERROGATE_STATUS="pass"
else
    print_warning "Interrogate: docstring coverage ${INTERROGATE_PCT}% (below soft target, warning only)"
    INTERROGATE_STATUS="warn"
fi

# ============================================================================
# 8. DETECT-SECRETS - Secret detection  (CRITICAL)
# ============================================================================
# Scans, then diffs the result against .secrets.baseline so only NEW secrets
# fail the gate. The baseline holds audited false positives (status-variable
# names in this very script) — see the commit that introduced it for the
# audit notes.
#
# The inline diff is deliberate: `detect-secrets scan --baseline` rewrites the
# baseline file in place and gives no clean pass/fail signal, so we scan fresh
# and diff in Python instead. The comparison key is the (filename,
# hashed_secret) tuple — NOT the hash alone — so the same secret value
# relocated into a real source file is still flagged rather than silently
# whitelisted by a baseline entry for a different file.
#
# The capture below is wrapped in `set +e`/`set -e`: the RHS is a pipeline and
# the python stage exits 1 when a new secret is found. Under `set -euo
# pipefail` an unguarded non-zero pipeline at an assignment would abort the
# whole script here, so the gate could never fail cleanly.
print_info "Running detect-secrets scan (src, tests, scripts)..."
SECRETS_OUTPUT=$(uv run detect-secrets scan src/xbrain tests scripts 2>&1)

set +e
SECRETS_NEW=$(echo "$SECRETS_OUTPUT" | python3 -c "
import json, sys
scan = json.load(sys.stdin)
try:
    with open('.secrets.baseline') as f:
        baseline = json.load(f)
except (OSError, ValueError):
    baseline = {'results': {}}
# Key by (filename, hashed_secret): a secret is 'known' only if the SAME value
# was audited in the SAME file. The baseline filename comes from each entry's
# own 'filename' field, falling back to the results-dict key.
known = {
    (s.get('filename', f), s['hashed_secret'])
    for f, ss in baseline.get('results', {}).items()
    for s in ss
}
new = [
    (f, s)
    for f, ss in scan.get('results', {}).items()
    for s in ss
    if (s.get('filename', f), s['hashed_secret']) not in known
]
for f, s in new:
    print(f\"  {f}: line {s.get('line_number', '?')} - {s['type']}\")
sys.exit(1 if new else 0)
" 2>&1)
SECRETS_EXIT=$?
set -e

if [ "$SECRETS_EXIT" -eq 0 ]; then
    print_success "Detect-secrets: no new secrets (baseline up to date)"
    SECRETS_STATUS="pass"
else
    print_error "Detect-secrets: NEW potential secrets detected (not in baseline)"
    echo "$SECRETS_NEW"
    SECRETS_STATUS="fail"
    mark_failed "Secrets"
fi

# ============================================================================
# 9. PYTEST - Tests with coverage  (CRITICAL)
# ============================================================================
print_info "Running tests with coverage..."
set +e
TEST_OUTPUT=$(uv run pytest --cov=src/xbrain --cov-report=term -q 2>&1)
TEST_EXIT_CODE=$?
set -e
echo "$TEST_OUTPUT"

# Extract number of tests passed
TESTS_COUNT=$(echo "$TEST_OUTPUT" | grep -oE '[0-9]+ passed' | head -1 | grep -oE '[0-9]+')

# Extract coverage percentage from TOTAL line
COVERAGE_PCT=$(echo "$TEST_OUTPUT" | grep "^TOTAL" | tail -1 | awk '{print $NF}' | tr -d '%')

# Determine coverage status:
#   pass : >= COVERAGE_MIN + 5
#   warn : COVERAGE_MIN .. COVERAGE_MIN + 4  (within 5 points above the minimum)
#   fail : < COVERAGE_MIN
if [ -n "$COVERAGE_PCT" ]; then
    COVERAGE_INT="${COVERAGE_PCT%%.*}"
    if [ "$COVERAGE_INT" -lt "$COVERAGE_MIN" ]; then
        COVERAGE_STATUS="fail"
        mark_failed "Coverage"
    elif [ "$COVERAGE_INT" -lt $((COVERAGE_MIN + 5)) ]; then
        COVERAGE_STATUS="warn"
    else
        COVERAGE_STATUS="pass"
    fi
elif [ "$TEST_EXIT_CODE" -eq 0 ]; then
    # Tests passed but the TOTAL line could not be parsed — coverage is
    # unmeasurable. A gate that cannot measure coverage must fail loudly,
    # not silently vanish. Set a placeholder so the summary row still renders.
    print_error "Coverage: could not parse coverage from pytest output"
    COVERAGE_PCT="?"
    COVERAGE_STATUS="fail"
    mark_failed "Coverage"
fi

if [ "$TEST_EXIT_CODE" -eq 0 ]; then
    print_success "Tests: all passed"
    TESTS_STATUS="pass"
else
    print_error "Tests: failures detected"
    TESTS_STATUS="fail"
    mark_failed "Tests"
fi

# ============================================================================
# 10. DEPTRY - Dependency checking  (WARN ONLY)
# ============================================================================
print_info "Running Deptry dependency check..."
if uv run deptry . 2>&1; then
    print_success "Deptry: dependencies are correct"
    DEPTRY_STATUS="pass"
else
    print_warning "Deptry: dependency notes found (warning only)"
    DEPTRY_STATUS="warn"
fi

# ============================================================================
# SUMMARY
# ============================================================================
print_summary
print_github_summary

# Exit with appropriate code: only CRITICAL failures block.
if [ -n "$FAILED_CHECKS" ]; then
    exit 1
else
    exit 0
fi
