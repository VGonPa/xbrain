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
#                                 detect-secrets, pytest, coverage, radon D/E/F
#   WARN ONLY (never blocks)     : radon grade C, vulture, interrogate, deptry

set -euo pipefail

# Detect CI environment
IS_CI="${GITHUB_ACTIONS:-false}"

# Color codes (disabled in CI)
if [ "$IS_CI" = "true" ]; then
    RED='' GREEN='' YELLOW='' CYAN='' BOLD='' NC=''
else
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    CYAN='\033[0;36m'
    BOLD='\033[1m'
    NC='\033[0m'
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

print_summary() {
    echo ""
    echo -e "${BOLD}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${BOLD}                    QUALITY SUMMARY                         ${NC}"
    echo -e "${BOLD}═══════════════════════════════════════════════════════════${NC}"
    echo ""

    # Ruff (linting)
    if [ "$RUFF_STATUS" = "pass" ]; then
        echo -e "  ${GREEN}✅${NC} Ruff (linting)       ${GREEN}PASS${NC}"
    else
        echo -e "  ${RED}❌${NC} Ruff (linting)       ${RED}FAIL${NC}"
    fi

    # Ruff (format)
    if [ "$FORMAT_STATUS" = "pass" ]; then
        echo -e "  ${GREEN}✅${NC} Ruff (format)        ${GREEN}PASS${NC}"
    else
        echo -e "  ${RED}❌${NC} Ruff (format)        ${RED}FAIL${NC}"
    fi

    # Mypy
    if [ "$MYPY_STATUS" = "pass" ]; then
        echo -e "  ${GREEN}✅${NC} Mypy (types)         ${GREEN}PASS${NC}"
    else
        echo -e "  ${RED}❌${NC} Mypy (types)         ${RED}FAIL${NC}"
    fi

    # Radon
    if [ "$RADON_STATUS" = "pass" ]; then
        echo -e "  ${GREEN}✅${NC} Radon (complexity)   ${GREEN}PASS${NC} - All A/B"
    elif [ "$RADON_STATUS" = "warn" ]; then
        echo -e "  ${YELLOW}⚠️${NC}  Radon (complexity)   ${YELLOW}WARN${NC} - Has grade ${RADON_MAX_GRADE}"
    else
        echo -e "  ${RED}❌${NC} Radon (complexity)   ${RED}FAIL${NC} - Has grade ${RADON_MAX_GRADE}"
    fi

    # Bandit
    if [ "$BANDIT_STATUS" = "pass" ]; then
        echo -e "  ${GREEN}✅${NC} Bandit (security)    ${GREEN}PASS${NC}"
    else
        echo -e "  ${RED}❌${NC} Bandit (security)    ${RED}FAIL${NC}"
    fi

    # Vulture
    if [ "$VULTURE_STATUS" = "pass" ]; then
        echo -e "  ${GREEN}✅${NC} Vulture (dead code)  ${GREEN}PASS${NC}"
    else
        echo -e "  ${YELLOW}⚠️${NC}  Vulture (dead code)  ${YELLOW}WARN${NC}"
    fi

    # Interrogate
    if [ "$INTERROGATE_STATUS" = "pass" ]; then
        echo -e "  ${GREEN}✅${NC} Interrogate (docs)   ${GREEN}PASS${NC} - ${INTERROGATE_PCT}%"
    else
        echo -e "  ${YELLOW}⚠️${NC}  Interrogate (docs)   ${YELLOW}WARN${NC} - ${INTERROGATE_PCT}%"
    fi

    # Detect-secrets
    if [ "$SECRETS_STATUS" = "pass" ]; then
        echo -e "  ${GREEN}✅${NC} Detect-secrets       ${GREEN}PASS${NC}"
    else
        echo -e "  ${RED}❌${NC} Detect-secrets       ${RED}FAIL${NC}"
    fi

    # Tests
    if [ "$TESTS_STATUS" = "pass" ]; then
        echo -e "  ${GREEN}✅${NC} Tests                ${GREEN}PASS${NC} - ${TESTS_COUNT} tests"
    else
        echo -e "  ${RED}❌${NC} Tests                ${RED}FAIL${NC}"
    fi

    # Coverage
    if [ -n "$COVERAGE_PCT" ]; then
        if [ "$COVERAGE_STATUS" = "pass" ]; then
            echo -e "  ${GREEN}✅${NC} Coverage             ${GREEN}${COVERAGE_PCT}%${NC}"
        elif [ "$COVERAGE_STATUS" = "warn" ]; then
            echo -e "  ${YELLOW}⚠️${NC}  Coverage             ${YELLOW}${COVERAGE_PCT}%${NC}"
        else
            echo -e "  ${RED}❌${NC} Coverage             ${RED}${COVERAGE_PCT}%${NC}"
        fi
    fi

    # Deptry
    if [ "$DEPTRY_STATUS" = "pass" ]; then
        echo -e "  ${GREEN}✅${NC} Deptry (deps)        ${GREEN}PASS${NC}"
    else
        echo -e "  ${YELLOW}⚠️${NC}  Deptry (deps)        ${YELLOW}WARN${NC}"
    fi

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

        if [ "$RUFF_STATUS" = "pass" ]; then
            echo "| Ruff (linting) | ✅ PASS | Code is clean |"
        else
            echo "| Ruff (linting) | ❌ FAIL | Linting errors |"
        fi

        if [ "$FORMAT_STATUS" = "pass" ]; then
            echo "| Ruff (format) | ✅ PASS | Code is formatted |"
        else
            echo "| Ruff (format) | ❌ FAIL | Formatting errors |"
        fi

        if [ "$MYPY_STATUS" = "pass" ]; then
            echo "| Mypy (types) | ✅ PASS | Type checks passed |"
        else
            echo "| Mypy (types) | ❌ FAIL | Type errors found |"
        fi

        if [ "$RADON_STATUS" = "pass" ]; then
            echo "| Radon (complexity) | ✅ PASS | All functions grade A/B |"
        elif [ "$RADON_STATUS" = "warn" ]; then
            echo "| Radon (complexity) | ⚠️ WARN | Has grade ${RADON_MAX_GRADE} |"
        else
            echo "| Radon (complexity) | ❌ FAIL | Has grade ${RADON_MAX_GRADE} (D/E/F) |"
        fi

        if [ "$BANDIT_STATUS" = "pass" ]; then
            echo "| Bandit (security) | ✅ PASS | No security issues |"
        else
            echo "| Bandit (security) | ❌ FAIL | Security issues found |"
        fi

        if [ "$VULTURE_STATUS" = "pass" ]; then
            echo "| Vulture (dead code) | ✅ PASS | No dead code |"
        else
            echo "| Vulture (dead code) | ⚠️ WARN | Possible dead code |"
        fi

        if [ "$INTERROGATE_STATUS" = "pass" ]; then
            echo "| Interrogate (docs) | ✅ PASS | ${INTERROGATE_PCT}% coverage |"
        else
            echo "| Interrogate (docs) | ⚠️ WARN | ${INTERROGATE_PCT}% (target 60%) |"
        fi

        if [ "$SECRETS_STATUS" = "pass" ]; then
            echo "| Detect-secrets | ✅ PASS | No secrets found |"
        else
            echo "| Detect-secrets | ❌ FAIL | Potential secrets detected |"
        fi

        if [ "$TESTS_STATUS" = "pass" ]; then
            echo "| Tests | ✅ PASS | ${TESTS_COUNT:-All} tests passed |"
        else
            echo "| Tests | ❌ FAIL | Test failures |"
        fi

        if [ -n "$COVERAGE_PCT" ]; then
            if [ "$COVERAGE_STATUS" = "pass" ]; then
                echo "| Coverage | ✅ ${COVERAGE_PCT}% | Above ${COVERAGE_MIN}% minimum |"
            elif [ "$COVERAGE_STATUS" = "warn" ]; then
                echo "| Coverage | ⚠️ ${COVERAGE_PCT}% | Within 5pts of ${COVERAGE_MIN}% minimum |"
            else
                echo "| Coverage | ❌ ${COVERAGE_PCT}% | Below ${COVERAGE_MIN}% minimum |"
            fi
        fi

        if [ "$DEPTRY_STATUS" = "pass" ]; then
            echo "| Deptry (deps) | ✅ PASS | Dependencies OK |"
        else
            echo "| Deptry (deps) | ⚠️ WARN | Dependency notes |"
        fi

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
RADON_OUTPUT=$(uv run radon cc src/xbrain -s 2>&1)

if echo "$RADON_OUTPUT" | grep -qE ' - [D-F] \([0-9]+\)$'; then
    # Grade D/E/F is a hard failure
    RADON_MAX_GRADE=$(echo "$RADON_OUTPUT" | grep -oE ' - [D-F] ' | head -1 | tr -d ' -')
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

# Extract coverage percentage
INTERROGATE_PCT=$(echo "$INTERROGATE_OUTPUT" | grep -oE 'actual: [0-9.]+' | grep -oE '[0-9.]+' | head -1)
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
print_info "Running detect-secrets scan (src, tests, scripts)..."
SECRETS_OUTPUT=$(uv run detect-secrets scan src/xbrain tests scripts 2>&1)

if echo "$SECRETS_OUTPUT" | grep -q '"results": {}'; then
    print_success "Detect-secrets: no secrets found"
    SECRETS_STATUS="pass"
else
    print_error "Detect-secrets: potential secrets detected"
    echo "$SECRETS_OUTPUT" | python3 -c "import json,sys; d=json.load(sys.stdin); [print(f'  {f}: line {s.get(\"line_number\",\"?\")} - {s[\"type\"]}') for f,ss in d['results'].items() for s in ss]" 2>/dev/null || echo "$SECRETS_OUTPUT" | head -30
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
