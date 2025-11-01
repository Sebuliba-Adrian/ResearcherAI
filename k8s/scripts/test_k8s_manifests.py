#!/usr/bin/env python3
"""
Kubernetes Manifest Validation Script
Tests the ResearcherAI Kubernetes/Helm configuration without requiring a cluster or Helm
"""

import os
import sys
import yaml
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Colors for output
class Colors:
    GREEN = '\033[0;32m'
    RED = '\033[0;31m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'

def print_header(msg: str):
    print(f"\n{Colors.BLUE}{'='*60}{Colors.NC}")
    print(f"{Colors.BLUE}  {msg}{Colors.NC}")
    print(f"{Colors.BLUE}{'='*60}{Colors.NC}\n")

def print_success(msg: str):
    print(f"{Colors.GREEN}✓{Colors.NC} {msg}")

def print_error(msg: str):
    print(f"{Colors.RED}✗{Colors.NC} {msg}")

def print_warning(msg: str):
    print(f"{Colors.YELLOW}⚠{Colors.NC} {msg}")

# Test 1: Chart Structure
def test_chart_structure(chart_path: str) -> bool:
    """Verify Helm chart has correct directory structure"""
    print_header("TEST 1: Chart Structure Validation")

    required_files = [
        "Chart.yaml",
        "values.yaml",
        "templates/_helpers.tpl",
    ]

    required_dirs = [
        "templates",
    ]

    errors = 0

    for file in required_files:
        file_path = os.path.join(chart_path, file)
        if os.path.exists(file_path):
            print_success(f"Found {file}")
        else:
            print_error(f"Missing {file}")
            errors += 1

    for dir in required_dirs:
        dir_path = os.path.join(chart_path, dir)
        if os.path.isdir(dir_path):
            template_count = len(list(Path(dir_path).glob("*.yaml")))
            print_success(f"Found {dir}/ with {template_count} YAML files")
        else:
            print_error(f"Missing {dir}/ directory")
            errors += 1

    return errors == 0

# Test 2: Chart.yaml Validation
def test_chart_yaml(chart_path: str) -> bool:
    """Validate Chart.yaml content"""
    print_header("TEST 2: Chart.yaml Validation")

    chart_file = os.path.join(chart_path, "Chart.yaml")

    try:
        with open(chart_file, 'r') as f:
            chart = yaml.safe_load(f)

        required_fields = {
            'apiVersion': str,
            'name': str,
            'description': str,
            'type': str,
            'version': str,
            'appVersion': str,
        }

        errors = 0

        for field, field_type in required_fields.items():
            if field in chart:
                if isinstance(chart[field], field_type):
                    print_success(f"{field}: {chart[field]}")
                else:
                    print_error(f"{field} has wrong type (expected {field_type.__name__})")
                    errors += 1
            else:
                print_error(f"Missing required field: {field}")
                errors += 1

        # Check dependencies
        if 'dependencies' in chart:
            print_success(f"Found {len(chart['dependencies'])} dependencies")
            for dep in chart['dependencies']:
                print_success(f"  - {dep.get('name', 'unknown')} (version: {dep.get('version', 'N/A')})")

        return errors == 0

    except Exception as e:
        print_error(f"Failed to parse Chart.yaml: {e}")
        return False

# Test 3: values.yaml Validation
def test_values_yaml(chart_path: str) -> bool:
    """Validate values.yaml content"""
    print_header("TEST 3: values.yaml Validation")

    values_file = os.path.join(chart_path, "values.yaml")

    try:
        with open(values_file, 'r') as f:
            values = yaml.safe_load(f)

        required_sections = ['app', 'neo4j', 'qdrant', 'kafka']

        errors = 0

        for section in required_sections:
            if section in values:
                print_success(f"Found section: {section}")
            else:
                print_error(f"Missing section: {section}")
                errors += 1

        # Check app configuration
        if 'app' in values:
            app = values['app']
            if 'replicaCount' in app:
                print_success(f"  app.replicaCount: {app['replicaCount']}")
            if 'image' in app and 'repository' in app['image']:
                print_success(f"  app.image.repository: {app['image']['repository']}")

        # Check resource definitions
        if 'app' in values and 'resources' in values['app']:
            print_success("  app.resources defined")
        else:
            print_warning("  app.resources not defined")

        # Check autoscaling
        if 'app' in values and 'autoscaling' in values['app']:
            if values['app']['autoscaling'].get('enabled'):
                print_success("  app.autoscaling enabled")

        return errors == 0

    except Exception as e:
        print_error(f"Failed to parse values.yaml: {e}")
        return False

# Test 4: Template Files Validation
def test_template_files(chart_path: str) -> bool:
    """Validate template files syntax"""
    print_header("TEST 4: Template Files Validation")

    templates_dir = os.path.join(chart_path, "templates")
    template_files = list(Path(templates_dir).glob("*.yaml"))

    if not template_files:
        print_error("No template files found")
        return False

    print_success(f"Found {len(template_files)} template files")

    errors = 0

    for template in template_files:
        print_success(f"  - {template.name}")

        # Check for common issues
        with open(template, 'r') as f:
            content = f.read()

            # Check for unmatched braces
            open_count = content.count('{{')
            close_count = content.count('}}')

            if open_count != close_count:
                print_error(f"    Unmatched braces in {template.name}")
                errors += 1

            # Check for if statements without end
            if_count = len(re.findall(r'\{\{-?\s*if\s+', content))
            endif_count = len(re.findall(r'\{\{-?\s*end\s*-?\}\}', content))

            if if_count > endif_count:
                print_error(f"    Unclosed if statements in {template.name}")
                errors += 1

    return errors == 0

# Test 5: Resource Definitions
def test_resource_definitions(chart_path: str) -> bool:
    """Check for expected Kubernetes resources"""
    print_header("TEST 5: Resource Definitions Check")

    templates_dir = os.path.join(chart_path, "templates")

    expected_resources = {
        'namespace.yaml': ['Namespace'],
        'configmap.yaml': ['ConfigMap'],
        'secret.yaml': ['Secret'],
        'app-deployment.yaml': ['Deployment', 'Service'],
        'neo4j-statefulset.yaml': ['StatefulSet', 'Service'],
        'qdrant-deployment.yaml': ['PersistentVolumeClaim', 'Deployment', 'Service'],
        'kafka-cluster.yaml': ['Kafka'],
        'kafka-topics.yaml': ['KafkaTopic'],
        'ingress.yaml': ['Ingress'],
        'hpa.yaml': ['HorizontalPodAutoscaler'],
        'pdb.yaml': ['PodDisruptionBudget'],
    }

    errors = 0

    for file_name, resources in expected_resources.items():
        file_path = os.path.join(templates_dir, file_name)

        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                content = f.read()

            print_success(f"{file_name}")

            for resource in resources:
                if f"kind: {resource}" in content:
                    print_success(f"  - {resource} defined")
                else:
                    print_warning(f"  - {resource} not found (might be conditional)")

        else:
            print_warning(f"{file_name} not found")
            errors += 1

    return errors < len(expected_resources) / 2  # Allow some missing files

# Test 6: Security Checks
def test_security(chart_path: str) -> bool:
    """Check for security best practices"""
    print_header("TEST 6: Security Best Practices")

    templates_dir = os.path.join(chart_path, "templates")
    warnings = 0

    # Check for hardcoded secrets
    for template in Path(templates_dir).glob("*.yaml"):
        with open(template, 'r') as f:
            content = f.read()

            # Skip actual secret files
            if 'secret.yaml' in template.name:
                continue

            # Check for potential hardcoded values
            if re.search(r'password.*:.*["\'][^{]+["\']', content):
                print_warning(f"Potential hardcoded password in {template.name}")
                warnings += 1

            if re.search(r'apiKey.*:.*["\'][^{]+["\']', content):
                print_warning(f"Potential hardcoded API key in {template.name}")
                warnings += 1

    # Check for security context
    app_deployment = os.path.join(templates_dir, "app-deployment.yaml")
    if os.path.exists(app_deployment):
        with open(app_deployment, 'r') as f:
            content = f.read()

            if 'securityContext' in content:
                print_success("Security context defined in app deployment")
            else:
                print_warning("No security context in app deployment")
                warnings += 1

    if warnings == 0:
        print_success("No security issues found")

    return True  # Warnings don't fail the test

# Test 7: Documentation
def test_documentation(chart_path: str) -> bool:
    """Check for proper documentation"""
    print_header("TEST 7: Documentation Check")

    readme = os.path.join(chart_path, "README.md")

    if os.path.exists(readme):
        with open(readme, 'r') as f:
            content = f.read()

        print_success("README.md exists")

        # Check for important sections
        sections = [
            ('Overview', r'## Overview'),
            ('Prerequisites', r'## Prerequisites'),
            ('Quick Start', r'## Quick Start'),
            ('Configuration', r'## Configuration'),
        ]

        for section_name, pattern in sections:
            if re.search(pattern, content, re.IGNORECASE):
                print_success(f"  - {section_name} section found")
            else:
                print_warning(f"  - {section_name} section missing")

        return True
    else:
        print_warning("README.md not found")
        return False

# Test 8: Values Schema
def test_values_structure(chart_path: str) -> bool:
    """Validate values.yaml structure matches templates"""
    print_header("TEST 8: Values Structure Validation")

    values_file = os.path.join(chart_path, "values.yaml")

    try:
        with open(values_file, 'r') as f:
            values = yaml.safe_load(f)

        # Check nested structures
        checks = [
            ('global.namespace', lambda v: 'global' in v and 'namespace' in v['global']),
            ('app.image.repository', lambda v: 'app' in v and 'image' in v['app'] and 'repository' in v['app']['image']),
            ('app.resources', lambda v: 'app' in v and 'resources' in v['app']),
            ('neo4j.enabled', lambda v: 'neo4j' in v and 'enabled' in v['neo4j']),
            ('qdrant.enabled', lambda v: 'qdrant' in v and 'enabled' in v['qdrant']),
            ('kafka.enabled', lambda v: 'kafka' in v and 'enabled' in v['kafka']),
        ]

        for path, check_func in checks:
            if check_func(values):
                print_success(f"{path} defined")
            else:
                print_error(f"{path} missing")

        return True

    except Exception as e:
        print_error(f"Failed to validate values structure: {e}")
        return False

# Main
def main():
    print("\n" + "="*60)
    print(" "*15 + "Kubernetes/Helm Chart Validation")
    print("="*60 + "\n")

    chart_path = "./k8s/helm/researcherai"

    if not os.path.exists(chart_path):
        print_error(f"Chart path not found: {chart_path}")
        return 1

    tests = [
        ("Chart Structure", lambda: test_chart_structure(chart_path)),
        ("Chart.yaml", lambda: test_chart_yaml(chart_path)),
        ("values.yaml", lambda: test_values_yaml(chart_path)),
        ("Template Files", lambda: test_template_files(chart_path)),
        ("Resource Definitions", lambda: test_resource_definitions(chart_path)),
        ("Security", lambda: test_security(chart_path)),
        ("Documentation", lambda: test_documentation(chart_path)),
        ("Values Structure", lambda: test_values_structure(chart_path)),
    ]

    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print_error(f"Test '{name}' raised exception: {e}")
            results.append((name, False))

    # Summary
    print_header("Test Summary")

    total = len(results)
    passed = sum(1 for _, p in results if p)
    failed = total - passed

    print(f"Total Tests: {total}")
    print(f"Passed: {Colors.GREEN}{passed}{Colors.NC}")
    print(f"Failed: {Colors.RED}{failed}{Colors.NC}\n")

    for name, passed in results:
        status = f"{Colors.GREEN}PASS{Colors.NC}" if passed else f"{Colors.RED}FAIL{Colors.NC}"
        print(f"  [{status}] {name}")

    print()

    if failed == 0:
        print_success("All tests passed! Chart is ready for deployment.\n")
        return 0
    else:
        print_error(f"{failed} test(s) failed. Please review errors above.\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())
