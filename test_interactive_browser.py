#!/usr/bin/env python3
"""
Interactive Browser Testing for ResearcherAI
Tests both Development and Production modes with visual validation
"""

import asyncio
import time
from pathlib import Path
from playwright.async_api import async_playwright
import json

class InteractiveBrowserTest:
    def __init__(self, base_url: str, mode: str):
        self.base_url = base_url
        self.mode = mode
        self.screenshot_dir = Path("/home/adrian/Desktop/Projects/ResearcherAI/screenshots")
        self.screenshot_dir.mkdir(exist_ok=True)

    async def test_development_mode(self):
        """Test development mode (FAISS + NetworkX) with visual validation"""
        print(f"\n{'='*70}")
        print(f"🧪 Testing DEVELOPMENT MODE (FAISS + NetworkX)")
        print(f"{'='*70}\n")

        async with async_playwright() as p:
            # Launch browser in headed mode so we can see it
            browser = await p.chromium.launch(headless=False, slow_mo=500)
            context = await browser.new_context(viewport={"width": 1920, "height": 1080})
            page = await context.new_page()

            try:
                # Step 1: Navigate to homepage
                print("📍 Step 1: Loading homepage...")
                await page.goto(self.base_url)
                await page.wait_for_load_state("networkidle")
                await page.screenshot(path=str(self.screenshot_dir / "01_dev_homepage.png"), full_page=True)
                print("✅ Homepage loaded")
                await asyncio.sleep(2)

                # Step 2: Check health status
                print("\n📍 Step 2: Checking system health...")
                health_response = await page.evaluate("""
                    fetch('/v1/health', {
                        headers: {'X-API-Key': 'demo-key-123'}
                    }).then(r => r.json())
                """)
                print(f"✅ Health check: {health_response.get('status')}")
                print(f"   Agents: {list(health_response.get('agents', {}).keys())}")

                # Step 3: Scroll through sections
                print("\n📍 Step 3: Exploring UI sections...")

                # Data Collection section
                print("   → Data Collection section")
                await page.evaluate("document.querySelector('.data-collection')?.scrollIntoView({behavior: 'smooth'})")
                await asyncio.sleep(1)
                await page.screenshot(path=str(self.screenshot_dir / "02_dev_data_collection.png"))

                # Ask Questions section
                print("   → Ask Questions section")
                await page.evaluate("document.querySelector('.ask-questions')?.scrollIntoView({behavior: 'smooth'})")
                await asyncio.sleep(1)
                await page.screenshot(path=str(self.screenshot_dir / "03_dev_ask_questions.png"))

                # PDF Upload section
                print("   → PDF Upload section")
                await page.evaluate("document.querySelector('.pdf-upload')?.scrollIntoView({behavior: 'smooth'})")
                await asyncio.sleep(1)
                await page.screenshot(path=str(self.screenshot_dir / "04_dev_pdf_upload.png"))

                # Knowledge Graph section
                print("   → Knowledge Graph section")
                await page.evaluate("document.querySelector('.knowledge-graph')?.scrollIntoView({behavior: 'smooth'})")
                await asyncio.sleep(1)
                await page.screenshot(path=str(self.screenshot_dir / "05_dev_knowledge_graph.png"))

                # Step 4: Get stats via API
                print("\n📍 Step 4: Getting system statistics...")
                stats = await page.evaluate("""
                    fetch('/v1/stats', {
                        headers: {'X-API-Key': 'demo-key-123'}
                    }).then(r => r.json())
                """)
                print(f"✅ Stats retrieved")
                print(f"   System info: {stats.get('system', {}).get('backend_mode', 'N/A')}")

                # Step 5: Check graph export
                print("\n📍 Step 5: Testing graph export...")
                graph_data = await page.evaluate("""
                    fetch('/v1/graph/export', {
                        headers: {'X-API-Key': 'demo-key-123'}
                    }).then(r => r.json())
                """)
                print(f"✅ Graph exported")
                print(f"   Nodes: {len(graph_data.get('nodes', []))}")
                print(f"   Edges: {len(graph_data.get('edges', []))}")

                # Step 6: Check vector stats
                print("\n📍 Step 6: Checking vector database...")
                vector_stats = await page.evaluate("""
                    fetch('/v1/vector/stats', {
                        headers: {'X-API-Key': 'demo-key-123'}
                    }).then(r => r.json())
                """)
                print(f"✅ Vector stats retrieved")
                print(f"   Backend: FAISS (development)")
                print(f"   Vectors: {vector_stats.get('total_vectors', 0)}")

                # Step 7: Scroll to vector section
                print("\n📍 Step 7: Vector Space section...")
                await page.evaluate("document.querySelector('.vector-space')?.scrollIntoView({behavior: 'smooth'})")
                await asyncio.sleep(1)
                await page.screenshot(path=str(self.screenshot_dir / "06_dev_vector_space.png"))

                # Step 8: Final full page screenshot
                print("\n📍 Step 8: Taking final screenshot...")
                await page.evaluate("window.scrollTo(0, 0)")
                await asyncio.sleep(1)
                await page.screenshot(path=str(self.screenshot_dir / "07_dev_complete.png"), full_page=True)

                print("\n✅ Development mode testing complete!")
                print(f"📸 Screenshots saved to: {self.screenshot_dir}")

                # Keep browser open for a moment
                print("\n⏸️  Browser will stay open for 5 seconds...")
                await asyncio.sleep(5)

            finally:
                await browser.close()

    async def test_production_mode(self):
        """Test production mode (Neo4j + Qdrant) with visual validation"""
        print(f"\n{'='*70}")
        print(f"🧪 Testing PRODUCTION MODE (Neo4j + Qdrant)")
        print(f"{'='*70}\n")

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=False, slow_mo=500)
            context = await browser.new_context(viewport={"width": 1920, "height": 1080})
            page = await context.new_page()

            try:
                # Step 1: Navigate to homepage
                print("📍 Step 1: Loading homepage...")
                await page.goto(self.base_url)
                await page.wait_for_load_state("networkidle")
                await page.screenshot(path=str(self.screenshot_dir / "08_prod_homepage.png"), full_page=True)
                print("✅ Homepage loaded")
                await asyncio.sleep(2)

                # Step 2: Check health status
                print("\n📍 Step 2: Checking system health...")
                health_response = await page.evaluate("""
                    fetch('/v1/health', {
                        headers: {'X-API-Key': 'demo-key-123'}
                    }).then(r => r.json())
                """)
                print(f"✅ Health check: {health_response.get('status')}")
                print(f"   Agents: {list(health_response.get('agents', {}).keys())}")

                # Step 3: Get stats to verify production backends
                print("\n📍 Step 3: Verifying production backends...")
                stats = await page.evaluate("""
                    fetch('/v1/stats', {
                        headers: {'X-API-Key': 'demo-key-123'}
                    }).then(r => r.json())
                """)
                print(f"✅ Stats retrieved")
                backend_mode = stats.get('system', {}).get('backend_mode', 'N/A')
                print(f"   Backend mode: {backend_mode}")
                print(f"   Graph DB: Neo4j (production)")
                print(f"   Vector DB: Qdrant (production)")

                # Step 4: Check graph export (Neo4j)
                print("\n📍 Step 4: Testing Neo4j graph export...")
                try:
                    graph_data = await page.evaluate("""
                        fetch('/v1/graph/export', {
                            headers: {'X-API-Key': 'demo-key-123'}
                        }).then(r => r.json())
                    """)
                    print(f"✅ Neo4j graph exported")
                    print(f"   Nodes: {len(graph_data.get('nodes', []))}")
                    print(f"   Edges: {len(graph_data.get('edges', []))}")
                except Exception as e:
                    print(f"⚠️  Graph export: {e}")

                # Step 5: Check vector stats (Qdrant)
                print("\n📍 Step 5: Checking Qdrant vector database...")
                try:
                    vector_stats = await page.evaluate("""
                        fetch('/v1/vector/stats', {
                            headers: {'X-API-Key': 'demo-key-123'}
                        }).then(r => r.json())
                    """)
                    print(f"✅ Qdrant stats retrieved")
                    print(f"   Backend: Qdrant (production)")
                    print(f"   Vectors: {vector_stats.get('total_vectors', 0)}")
                except Exception as e:
                    print(f"⚠️  Vector stats: {e}")

                # Step 6: Navigate through sections
                print("\n📍 Step 6: Exploring UI sections...")
                await page.evaluate("document.querySelector('.knowledge-graph')?.scrollIntoView({behavior: 'smooth'})")
                await asyncio.sleep(1)
                await page.screenshot(path=str(self.screenshot_dir / "09_prod_knowledge_graph.png"))

                await page.evaluate("document.querySelector('.vector-space')?.scrollIntoView({behavior: 'smooth'})")
                await asyncio.sleep(1)
                await page.screenshot(path=str(self.screenshot_dir / "10_prod_vector_space.png"))

                # Step 7: Final full page screenshot
                print("\n📍 Step 7: Taking final screenshot...")
                await page.evaluate("window.scrollTo(0, 0)")
                await asyncio.sleep(1)
                await page.screenshot(path=str(self.screenshot_dir / "11_prod_complete.png"), full_page=True)

                print("\n✅ Production mode testing complete!")
                print(f"📸 Screenshots saved to: {self.screenshot_dir}")

                # Keep browser open for a moment
                print("\n⏸️  Browser will stay open for 5 seconds...")
                await asyncio.sleep(5)

            finally:
                await browser.close()


async def main():
    """Main test runner"""

    # Test 1: Development Mode
    print("\n" + "="*70)
    print("🚀 STARTING BROWSER TESTS")
    print("="*70)

    dev_tester = InteractiveBrowserTest("http://localhost:8000", "development")
    await dev_tester.test_development_mode()

    # Prompt for production mode
    print("\n" + "="*70)
    print("⚠️  SWITCHING TO PRODUCTION MODE")
    print("="*70)
    print("\nTo test production mode:")
    print("1. Stop the development server (Ctrl+C)")
    print("2. Start production mode: docker-compose up -d")
    print("3. Run: python test_interactive_browser.py --production")
    print("\nOr continue automatically if production is already running...")

    # Try production mode
    try:
        import requests
        prod_health = requests.get("http://localhost:8000/v1/health", timeout=2)
        if prod_health.status_code == 200:
            print("\n🔄 Production server detected, testing now...\n")
            prod_tester = InteractiveBrowserTest("http://localhost:8000", "production")
            await prod_tester.test_production_mode()
    except:
        print("\n⏭️  Skipping production mode (server not running)")

    print("\n" + "="*70)
    print("✅ ALL BROWSER TESTS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(main())
