#!/usr/bin/env bash
#
# Helm Chart Validation and Testing Script
# Tests the ResearcherAI Helm chart without deploying to a cluster
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

HELM_CHART_PATH="./k8s/helm/researcherai"
TEST_OUTPUT_DIR="./k8s/test-output"

print_header() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}========================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# Create output directory
mkdir -p "$TEST_OUTPUT_DIR"

# Test 1: Helm Lint
test_helm_lint() {
    print_header "TEST 1: Helm Lint"

    if helm lint "$HELM_CHART_PATH" > "$TEST_OUTPUT_DIR/lint.log" 2>&1; then
        print_success "Helm lint passed"
        return 0
    else
        print_error "Helm lint failed"
        cat "$TEST_OUTPUT_DIR/lint.log"
        return 1
    fi
}

# Test 2: Template Rendering
test_template_rendering() {
    print_header "TEST 2: Template Rendering"

    if helm template test-release "$HELM_CHART_PATH" \
        --set app.secrets.googleApiKey="test-key" \
        > "$TEST_OUTPUT_DIR/rendered-templates.yaml" 2>&1; then
        print_success "Templates rendered successfully"

        # Count resources
        RESOURCE_COUNT=$(grep -c "^kind:" "$TEST_OUTPUT_DIR/rendered-templates.yaml" || echo "0")
        print_success "Generated $RESOURCE_COUNT Kubernetes resources"
        return 0
    else
        print_error "Template rendering failed"
        return 1
    fi
}

# Test 3: Dry Run
test_dry_run() {
    print_header "TEST 3: Dry Run Install"

    if helm install test-release "$HELM_CHART_PATH" \
        --dry-run \
        --debug \
        --set app.secrets.googleApiKey="test-key" \
        > "$TEST_OUTPUT_DIR/dry-run.log" 2>&1; then
        print_success "Dry run passed"
        return 0
    else
        print_error "Dry run failed"
        tail -20 "$TEST_OUTPUT_DIR/dry-run.log"
        return 1
    fi
}

# Test 4: Validate Required Fields
test_required_fields() {
    print_header "TEST 4: Required Fields Validation"

    local errors=0

    # Check Chart.yaml
    if [ -f "$HELM_CHART_PATH/Chart.yaml" ]; then
        print_success "Chart.yaml exists"

        # Check required fields
        for field in "name" "version" "appVersion" "description"; do
            if grep -q "^$field:" "$HELM_CHART_PATH/Chart.yaml"; then
                print_success "Chart.yaml has $field"
            else
                print_error "Chart.yaml missing $field"
                ((errors++))
            fi
        done
    else
        print_error "Chart.yaml not found"
        ((errors++))
    fi

    # Check values.yaml
    if [ -f "$HELM_CHART_PATH/values.yaml" ]; then
        print_success "values.yaml exists"
    else
        print_error "values.yaml not found"
        ((errors++))
    fi

    # Check templates directory
    if [ -d "$HELM_CHART_PATH/templates" ]; then
        TEMPLATE_COUNT=$(find "$HELM_CHART_PATH/templates" -name "*.yaml" -o -name "*.tpl" | wc -l)
        print_success "Found $TEMPLATE_COUNT template files"
    else
        print_error "templates directory not found"
        ((errors++))
    fi

    return $errors
}

# Test 5: Validate Template Syntax
test_template_syntax() {
    print_header "TEST 5: Template Syntax Validation"

    local errors=0

    # Check for common template errors
    print_success "Checking for template syntax issues..."

    # Check for unmatched braces
    if grep -r "{{[^}]*$" "$HELM_CHART_PATH/templates" 2>/dev/null; then
        print_error "Found unmatched opening braces"
        ((errors++))
    else
        print_success "No unmatched braces found"
    fi

    # Check for proper indentation helpers
    if ! grep -r "nindent\|indent" "$HELM_CHART_PATH/templates" > /dev/null 2>&1; then
        print_warning "No indentation helpers found (this might be intentional)"
    else
        print_success "Indentation helpers present"
    fi

    return $errors
}

# Test 6: Validate Resource Definitions
test_resource_definitions() {
    print_header "TEST 6: Resource Definitions"

    local errors=0
    local rendered="$TEST_OUTPUT_DIR/rendered-templates.yaml"

    if [ ! -f "$rendered" ]; then
        print_error "Rendered templates not found, skipping resource validation"
        return 1
    fi

    # Check for expected resources
    local expected_resources=(
        "Namespace"
        "ConfigMap"
        "Secret"
        "Deployment"
        "StatefulSet"
        "Service"
        "Ingress"
        "HorizontalPodAutoscaler"
        "PodDisruptionBudget"
    )

    for resource in "${expected_resources[@]}"; do
        if grep -q "^kind: $resource$" "$rendered"; then
            print_success "Found $resource"
        else
            print_warning "$resource not found (might be conditional)"
        fi
    done

    # Check for Kafka resources
    if grep -q "kafka.strimzi.io" "$rendered"; then
        print_success "Found Kafka CRDs (Strimzi)"
    else
        print_warning "Kafka CRDs not found (might be conditional)"
    fi

    return $errors
}

# Test 7: Values Override
test_values_override() {
    print_header "TEST 7: Values Override Test"

    # Test with custom values
    cat > "$TEST_OUTPUT_DIR/test-values.yaml" <<EOF
app:
  replicaCount: 5
  secrets:
    googleApiKey: "override-test-key"

neo4j:
  enabled: false

qdrant:
  enabled: false

kafka:
  enabled: false
EOF

    if helm template test-release "$HELM_CHART_PATH" \
        -f "$TEST_OUTPUT_DIR/test-values.yaml" \
        > "$TEST_OUTPUT_DIR/override-test.yaml" 2>&1; then
        print_success "Values override test passed"

        # Verify override worked
        if grep -q "replicas: 5" "$TEST_OUTPUT_DIR/override-test.yaml"; then
            print_success "Replica count override verified"
        else
            print_error "Replica count override failed"
            return 1
        fi

        return 0
    else
        print_error "Values override test failed"
        return 1
    fi
}

# Test 8: Check for Hardcoded Values
test_hardcoded_values() {
    print_header "TEST 8: Hardcoded Values Check"

    local warnings=0

    # Check for potential hardcoded values
    if grep -r "password.*:.*\".*\"" "$HELM_CHART_PATH/templates" 2>/dev/null | grep -v "{{"; then
        print_warning "Found potential hardcoded passwords"
        ((warnings++))
    fi

    if grep -r "apiKey.*:.*\".*\"" "$HELM_CHART_PATH/templates" 2>/dev/null | grep -v "{{"; then
        print_warning "Found potential hardcoded API keys"
        ((warnings++))
    fi

    if [ $warnings -eq 0 ]; then
        print_success "No obvious hardcoded secrets found"
    fi

    return 0  # Warnings don't fail the test
}

# Test Summary
print_summary() {
    print_header "Test Summary"

    local total=$1
    local passed=$2
    local failed=$((total - passed))

    echo "Total Tests: $total"
    echo "Passed: ${GREEN}$passed${NC}"
    echo "Failed: ${RED}$failed${NC}"
    echo ""

    if [ $failed -eq 0 ]; then
        print_success "All tests passed!"
        echo ""
        echo "Chart is ready for deployment."
        echo "Test artifacts saved in: $TEST_OUTPUT_DIR"
        return 0
    else
        print_error "$failed test(s) failed"
        echo ""
        echo "Please review the errors above and in: $TEST_OUTPUT_DIR"
        return 1
    fi
}

# Main execution
main() {
    echo ""
    echo "╔═══════════════════════════════════════════════════════╗"
    echo "║                                                       ║"
    echo "║         Helm Chart Validation & Testing              ║"
    echo "║                                                       ║"
    echo "╚═══════════════════════════════════════════════════════╝"
    echo ""

    local total_tests=0
    local passed_tests=0

    # Run tests
    if test_helm_lint; then ((passed_tests++)); fi; ((total_tests++))
    if test_template_rendering; then ((passed_tests++)); fi; ((total_tests++))
    if test_dry_run; then ((passed_tests++)); fi; ((total_tests++))
    if test_required_fields; then ((passed_tests++)); fi; ((total_tests++))
    if test_template_syntax; then ((passed_tests++)); fi; ((total_tests++))
    if test_resource_definitions; then ((passed_tests++)); fi; ((total_tests++))
    if test_values_override; then ((passed_tests++)); fi; ((total_tests++))
    if test_hardcoded_values; then ((passed_tests++)); fi; ((total_tests++))

    # Print summary
    print_summary $total_tests $passed_tests
}

# Run main
main "$@"
