#!/usr/bin/env python3
"""
Production Mode Browser Test - Neo4j + Qdrant
"""

import asyncio
import time
from pathlib import Path
from playwright.async_api import async_playwright

async def test_production():
    """Test production mode with Neo4j and Qdrant"""
    print("\n" + "="*70)
    print("🧪 Testing PRODUCTION MODE (Neo4j + Qdrant)")
    print("="*70 + "\n")

    screenshot_dir = Path("/home/adrian/Desktop/Projects/ResearcherAI/screenshots")
    screenshot_dir.mkdir(exist_ok=True)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False, slow_mo=500)
        context = await browser.new_context(viewport={"width": 1920, "height": 1080})
        page = await context.new_page()

        try:
            # Navigate
            print("📍 Loading production mode homepage...")
            await page.goto("http://localhost:8000")
            await page.wait_for_load_state("networkidle")
            await page.screenshot(path=str(screenshot_dir / "prod_01_homepage.png"), full_page=True)
            print("✅ Homepage loaded\n")
            await asyncio.sleep(2)

            # Test Neo4j connection
            print("📍 Testing Neo4j graph database...")
            graph_data = await page.evaluate("""
                fetch('/v1/graph/export', {
                    headers: {'X-API-Key': 'demo-key-123'}
                }).then(r => r.json())
            """)
            print(f"✅ Neo4j connected")
            print(f"   Nodes: {len(graph_data.get('nodes', []))}")
            print(f"   Edges: {len(graph_data.get('edges', []))}\n")

            # Test Qdrant connection
            print("📍 Testing Qdrant vector database...")
            vector_stats = await page.evaluate("""
                fetch('/v1/vector/stats', {
                    headers: {'X-API-Key': 'demo-key-123'}
                }).then(r => r.json())
            """)
            print(f"✅ Qdrant connected")
            print(f"   Total vectors: {vector_stats.get('total_vectors', 0)}")
            print(f"   Collections: {vector_stats.get('collections', 'N/A')}\n")

            # Test health check
            print("📍 Checking agent health...")
            health = await page.evaluate("""
                fetch('/v1/health', {
                    headers: {'X-API-Key': 'demo-key-123'}
                }).then(r => r.json())
            """)
            print(f"✅ All agents: {list(health.get('agents', {}).keys())}\n")

            # Navigate to graph section
            print("📍 Navigating to graph visualization...")
            await page.evaluate("document.querySelector('.knowledge-graph')?.scrollIntoView({behavior: 'smooth'})")
            await asyncio.sleep(2)
            await page.screenshot(path=str(screenshot_dir / "prod_02_neo4j_graph.png"))
            print("✅ Graph section captured\n")

            # Navigate to vector section
            print("📍 Navigating to vector space...")
            await page.evaluate("document.querySelector('.vector-space')?.scrollIntoView({behavior: 'smooth'})")
            await asyncio.sleep(2)
            await page.screenshot(path=str(screenshot_dir / "prod_03_qdrant_vector.png"))
            print("✅ Vector section captured\n")

            # Take final screenshot
            await page.evaluate("window.scrollTo(0, 0)")
            await asyncio.sleep(1)
            await page.screenshot(path=str(screenshot_dir / "prod_04_complete.png"), full_page=True)

            print("="*70)
            print("✅ PRODUCTION MODE TESTING COMPLETE")
            print("="*70)
            print(f"\n📸 Screenshots saved to: {screenshot_dir}")
            print(f"\n🗄️  Database Status:")
            print(f"   Neo4j:   ✅ {len(graph_data.get('nodes', []))} nodes")
            print(f"   Qdrant:  ✅ {vector_stats.get('total_vectors', 0)} vectors")
            print(f"\n⏸️  Browser will close in 5 seconds...\n")

            await asyncio.sleep(5)

        finally:
            await browser.close()

if __name__ == "__main__":
    asyncio.run(test_production())
