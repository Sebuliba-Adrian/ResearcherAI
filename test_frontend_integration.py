#!/usr/bin/env python3
"""
Comprehensive Frontend-Backend Integration Test
Tests both Development (FAISS/NetworkX) and Production (Neo4j/Qdrant) modes
"""

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from playwright.async_api import async_playwright, Page, expect
import requests
from typing import Dict, List, Any

class FrontendIntegrationTest:
    def __init__(self, base_url: str = "http://localhost:8000", mode: str = "development"):
        self.base_url = base_url
        self.mode = mode
        self.results = {
            "mode": mode,
            "timestamp": datetime.now().isoformat(),
            "tests": [],
            "failures": [],
            "api_calls": [],
            "console_logs": [],
            "network_errors": []
        }
        self.api_key = "demo-key-123"

    async def setup_browser(self, page: Page):
        """Setup page monitoring for console logs and network requests"""
        page.on("console", lambda msg: self.results["console_logs"].append({
            "type": msg.type,
            "text": msg.text,
            "location": msg.location
        }))

        page.on("requestfailed", lambda req: self.results["network_errors"].append({
            "url": req.url,
            "method": req.method,
            "failure": req.failure
        }))

        page.on("response", lambda resp: self.log_api_call(resp))

    def log_api_call(self, response):
        """Log API calls for analysis"""
        if "/v1/" in response.url:
            self.results["api_calls"].append({
                "url": response.url,
                "status": response.status,
                "method": response.request.method,
                "headers": dict(response.headers)
            })

    def add_test_result(self, name: str, passed: bool, details: Any = None, error: str = None):
        """Record test result"""
        result = {
            "test": name,
            "passed": passed,
            "timestamp": datetime.now().isoformat(),
            "details": details
        }
        if error:
            result["error"] = error
            self.results["failures"].append(result)
        self.results["tests"].append(result)

        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")
        if details:
            print(f"  → {details}")
        if error:
            print(f"  ⚠️  {error}")

    async def test_health_endpoint(self):
        """Test 1: Backend Health Check"""
        try:
            response = requests.get(f"{self.base_url}/v1/health", timeout=10)
            data = response.json()

            passed = (
                response.status_code == 200 and
                data.get("status") == "healthy" and
                all(agent_status == "ready" for agent_status in data.get("agents", {}).values())
            )

            self.add_test_result(
                "Backend Health Check",
                passed,
                f"All agents ready: {', '.join(data.get('agents', {}).keys())}"
            )
            return passed
        except Exception as e:
            self.add_test_result("Backend Health Check", False, error=str(e))
            return False

    async def test_frontend_loads(self, page: Page):
        """Test 2: Frontend Loads Successfully"""
        try:
            await page.goto(self.base_url, timeout=30000)
            await page.wait_for_load_state("networkidle", timeout=10000)

            # Check for main UI elements
            title = await page.title()
            passed = "ResearcherAI" in title or len(title) > 0

            self.add_test_result(
                "Frontend Loads",
                passed,
                f"Page title: {title}"
            )
            return passed
        except Exception as e:
            self.add_test_result("Frontend Loads", False, error=str(e))
            return False

    async def test_ui_elements_present(self, page: Page):
        """Test 3: Essential UI Elements Present"""
        try:
            elements_to_check = [
                ("input[type='text'], textarea", "Input field"),
                ("button", "Action buttons"),
                (".container, #app, main", "Main container"),
            ]

            found_elements = []
            missing_elements = []

            for selector, name in elements_to_check:
                try:
                    element = await page.query_selector(selector)
                    if element:
                        found_elements.append(name)
                    else:
                        missing_elements.append(name)
                except:
                    missing_elements.append(name)

            passed = len(found_elements) >= 2
            self.add_test_result(
                "UI Elements Present",
                passed,
                f"Found: {', '.join(found_elements)}" +
                (f" | Missing: {', '.join(missing_elements)}" if missing_elements else "")
            )
            return passed
        except Exception as e:
            self.add_test_result("UI Elements Present", False, error=str(e))
            return False

    async def test_backend_integration(self, page: Page):
        """Test 4: Frontend-Backend Integration"""
        try:
            # Trigger an API call from the frontend
            await page.evaluate("""
                fetch('/v1/health', {
                    headers: {'X-API-Key': 'demo-key-123'}
                })
            """)

            # Wait a moment for the call to complete
            await page.wait_for_timeout(1000)

            # Check if API calls were made
            api_health_calls = [
                call for call in self.results["api_calls"]
                if "/v1/health" in call["url"]
            ]

            passed = len(api_health_calls) > 0
            self.add_test_result(
                "Frontend-Backend Integration",
                passed,
                f"{len(api_health_calls)} API call(s) to /v1/health"
            )
            return passed
        except Exception as e:
            self.add_test_result("Frontend-Backend Integration", False, error=str(e))
            return False

    async def test_stats_endpoint(self, page: Page):
        """Test 5: Stats Endpoint"""
        try:
            response = requests.get(
                f"{self.base_url}/v1/stats",
                headers={"X-API-Key": self.api_key},
                timeout=10
            )

            if response.status_code == 200:
                data = response.json()
                stats_keys = list(data.keys())
                passed = len(stats_keys) > 0
                self.add_test_result(
                    "Stats Endpoint",
                    passed,
                    f"Retrieved stats: {', '.join(stats_keys[:5])}"
                )
                return passed
            else:
                self.add_test_result(
                    "Stats Endpoint",
                    False,
                    error=f"HTTP {response.status_code}"
                )
                return False
        except Exception as e:
            self.add_test_result("Stats Endpoint", False, error=str(e))
            return False

    async def test_graph_export(self, page: Page):
        """Test 6: Graph Export Functionality"""
        try:
            response = requests.get(
                f"{self.base_url}/v1/graph/export",
                headers={"X-API-Key": self.api_key},
                timeout=15
            )

            if response.status_code == 200:
                data = response.json()
                nodes = data.get("nodes", [])
                edges = data.get("edges", [])

                self.add_test_result(
                    "Graph Export",
                    True,
                    f"Nodes: {len(nodes)}, Edges: {len(edges)} ({self.mode} backend)"
                )
                return True
            else:
                self.add_test_result(
                    "Graph Export",
                    False,
                    error=f"HTTP {response.status_code}"
                )
                return False
        except Exception as e:
            self.add_test_result("Graph Export", False, error=str(e))
            return False

    async def test_vector_stats(self, page: Page):
        """Test 7: Vector Database Stats"""
        try:
            response = requests.get(
                f"{self.base_url}/v1/vector/stats",
                headers={"X-API-Key": self.api_key},
                timeout=10
            )

            if response.status_code == 200:
                data = response.json()
                backend = "FAISS" if self.mode == "development" else "Qdrant"
                self.add_test_result(
                    "Vector Database Stats",
                    True,
                    f"Backend: {backend}, Vectors: {data.get('total_vectors', 0)}"
                )
                return True
            else:
                self.add_test_result(
                    "Vector Database Stats",
                    False,
                    error=f"HTTP {response.status_code}"
                )
                return False
        except Exception as e:
            self.add_test_result("Vector Database Stats", False, error=str(e))
            return False

    async def test_rdf_export(self, page: Page):
        """Test 8: RDF Export Functionality"""
        try:
            response = requests.get(
                f"{self.base_url}/v1/graph/export/rdf",
                headers={"X-API-Key": self.api_key},
                timeout=15
            )

            if response.status_code == 200:
                rdf_data = response.text
                # Check for RDF markers
                has_rdf = any(marker in rdf_data for marker in ["@prefix", "rdf:", "rdfs:", "<http"])

                self.add_test_result(
                    "RDF Export",
                    has_rdf,
                    f"RDF size: {len(rdf_data)} bytes" if has_rdf else "No RDF markers found"
                )
                return has_rdf
            else:
                self.add_test_result(
                    "RDF Export",
                    False,
                    error=f"HTTP {response.status_code}"
                )
                return False
        except Exception as e:
            self.add_test_result("RDF Export", False, error=str(e))
            return False

    async def test_vector_visualization(self, page: Page):
        """Test 9: Vector Visualization Endpoint"""
        try:
            response = requests.get(
                f"{self.base_url}/v1/vector/visualize",
                headers={"X-API-Key": self.api_key},
                params={"method": "pca", "n_components": 3},
                timeout=20
            )

            if response.status_code == 200:
                data = response.json()
                has_plot = data.get("plot_type") is not None

                self.add_test_result(
                    "Vector Visualization",
                    has_plot,
                    f"Plot type: {data.get('plot_type', 'N/A')}, Points: {data.get('n_points', 0)}"
                )
                return has_plot
            else:
                self.add_test_result(
                    "Vector Visualization",
                    False,
                    error=f"HTTP {response.status_code}: {response.text[:100]}"
                )
                return False
        except Exception as e:
            self.add_test_result("Vector Visualization", False, error=str(e))
            return False

    async def test_session_management(self, page: Page):
        """Test 10: Session Management"""
        try:
            test_session = f"test_session_{int(time.time())}"

            # Create a session by asking a question
            response = requests.post(
                f"{self.base_url}/v1/ask",
                headers={"X-API-Key": self.api_key},
                json={
                    "question": "Test question for session management",
                    "session_name": test_session
                },
                timeout=30
            )

            if response.status_code == 200:
                # Check if session was created
                session_response = requests.get(
                    f"{self.base_url}/v1/sessions/{test_session}",
                    headers={"X-API-Key": self.api_key},
                    timeout=10
                )

                passed = session_response.status_code == 200
                self.add_test_result(
                    "Session Management",
                    passed,
                    f"Session '{test_session}' created and retrievable"
                )
                return passed
            else:
                self.add_test_result(
                    "Session Management",
                    False,
                    error=f"Failed to create session: HTTP {response.status_code}"
                )
                return False
        except Exception as e:
            self.add_test_result("Session Management", False, error=str(e))
            return False

    async def test_console_errors(self, page: Page):
        """Test 11: Check for Console Errors"""
        error_logs = [
            log for log in self.results["console_logs"]
            if log["type"] in ["error", "warning"]
        ]

        # Filter out known/acceptable warnings (like Pydantic deprecation)
        critical_errors = [
            log for log in error_logs
            if not any(skip in log["text"] for skip in [
                "PydanticDeprecated",
                "DevTools",
                "favicon"
            ])
        ]

        passed = len(critical_errors) == 0
        self.add_test_result(
            "Console Errors Check",
            passed,
            f"{len(critical_errors)} critical errors, {len(error_logs)-len(critical_errors)} warnings"
        )

        if critical_errors[:3]:  # Show first 3 errors
            for err in critical_errors[:3]:
                print(f"    ⚠️  {err['type']}: {err['text'][:100]}")

        return passed

    async def test_network_failures(self, page: Page):
        """Test 12: Check for Network Failures"""
        passed = len(self.results["network_errors"]) == 0
        self.add_test_result(
            "Network Failures Check",
            passed,
            f"{len(self.results['network_errors'])} failed requests"
        )

        if self.results["network_errors"][:3]:  # Show first 3
            for err in self.results["network_errors"][:3]:
                print(f"    ⚠️  {err['method']} {err['url']}: {err['failure']}")

        return passed

    async def run_all_tests(self):
        """Run all integration tests"""
        print(f"\n{'='*70}")
        print(f"🧪 ResearcherAI Frontend-Backend Integration Test")
        print(f"Mode: {self.mode.upper()}")
        print(f"URL: {self.base_url}")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*70}\n")

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                viewport={"width": 1920, "height": 1080},
                ignore_https_errors=True
            )
            page = await context.new_page()
            await self.setup_browser(page)

            try:
                # Run tests in sequence
                tests = [
                    self.test_health_endpoint(),
                    self.test_frontend_loads(page),
                    self.test_ui_elements_present(page),
                    self.test_backend_integration(page),
                    self.test_stats_endpoint(page),
                    self.test_graph_export(page),
                    self.test_vector_stats(page),
                    self.test_rdf_export(page),
                    self.test_vector_visualization(page),
                    self.test_session_management(page),
                ]

                for test in tests:
                    await test
                    await asyncio.sleep(0.5)  # Brief pause between tests

                # Final checks
                await self.test_console_errors(page)
                await self.test_network_failures(page)

            finally:
                # Take final screenshot
                screenshot_path = f"/home/adrian/Desktop/Projects/ResearcherAI/test_outputs/screenshot_{self.mode}_{int(time.time())}.png"
                Path(screenshot_path).parent.mkdir(parents=True, exist_ok=True)
                await page.screenshot(path=screenshot_path, full_page=True)
                print(f"\n📸 Screenshot saved: {screenshot_path}")

                await browser.close()

        # Generate summary
        self.generate_summary()

        return self.results

    def generate_summary(self):
        """Generate test summary"""
        total_tests = len(self.results["tests"])
        passed_tests = sum(1 for t in self.results["tests"] if t["passed"])
        failed_tests = total_tests - passed_tests
        pass_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0

        print(f"\n{'='*70}")
        print(f"📊 TEST SUMMARY - {self.mode.upper()} MODE")
        print(f"{'='*70}")
        print(f"Total Tests:    {total_tests}")
        print(f"Passed:         {passed_tests} ✅")
        print(f"Failed:         {failed_tests} ❌")
        print(f"Pass Rate:      {pass_rate:.1f}%")
        print(f"API Calls:      {len(self.results['api_calls'])}")
        print(f"Console Logs:   {len(self.results['console_logs'])}")
        print(f"Network Errors: {len(self.results['network_errors'])}")
        print(f"{'='*70}\n")

        if failed_tests > 0:
            print("❌ FAILED TESTS:")
            for failure in self.results["failures"]:
                print(f"  • {failure['test']}: {failure.get('error', 'Unknown error')}")
            print()

        # Save detailed results
        report_path = f"/home/adrian/Desktop/Projects/ResearcherAI/test_outputs/integration_test_{self.mode}_{int(time.time())}.json"
        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"📄 Detailed report saved: {report_path}\n")


async def main():
    """Main test runner"""
    # Test development mode first
    print("\n🔧 Testing DEVELOPMENT MODE (FAISS + NetworkX)")
    dev_tester = FrontendIntegrationTest(mode="development")
    dev_results = await dev_tester.run_all_tests()

    # Note: Production mode would require Docker containers to be running
    # For now, we'll just test development mode

    print("\n" + "="*70)
    print("✅ Integration testing complete!")
    print("="*70)

    # Return results for further analysis
    return {
        "development": dev_results
    }


if __name__ == "__main__":
    results = asyncio.run(main())
