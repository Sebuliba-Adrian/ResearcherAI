#!/usr/bin/env python3
"""
End-to-End Auto-Embedding Verification Test
Proves that embeddings are automatically generated when collecting papers
"""

import asyncio
import time
from pathlib import Path
from playwright.async_api import async_playwright
import requests
import json

class EmbeddingVerificationTest:
    def __init__(self):
        self.base_url = "http://localhost:8000"
        self.api_key = "demo-key-123"
        self.screenshot_dir = Path("screenshots")
        self.screenshot_dir.mkdir(exist_ok=True)

    async def run_test(self):
        """Run complete end-to-end test with screenshots"""
        print("\n" + "="*70)
        print("🧪 AUTO-EMBEDDING VERIFICATION TEST")
        print("="*70 + "\n")

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=False, slow_mo=800)
            context = await browser.new_context(viewport={"width": 1920, "height": 1080})
            page = await context.new_page()

            try:
                # Step 1: Check initial vector stats (BEFORE collection)
                print("📍 Step 1: Checking vector database BEFORE collection...")
                await page.goto(self.base_url)
                await page.wait_for_load_state("networkidle")

                stats_before = requests.get(
                    f"{self.base_url}/v1/vector/stats",
                    headers={"X-API-Key": self.api_key}
                ).json()

                print(f"   Vectors BEFORE: {stats_before.get('total_vectors', 0)}")
                await page.screenshot(path=str(self.screenshot_dir / "test_01_before_collection.png"), full_page=True)
                await asyncio.sleep(2)

                # Step 2: Navigate to collection section
                print("\n📍 Step 2: Navigating to data collection...")
                await page.evaluate("document.querySelector('#query-input')?.scrollIntoView({behavior: 'smooth'})")
                await asyncio.sleep(1)
                await page.screenshot(path=str(self.screenshot_dir / "test_02_collection_form.png"))

                # Step 3: Fill collection form
                print("\n📍 Step 3: Filling collection form...")
                await page.fill("#query-input", "large language models")
                await page.select_option("#max-results-select", "3")  # Only 3 papers for quick test

                # Check only arXiv source
                await page.evaluate("""
                    document.querySelectorAll('input[name="source"]').forEach(cb => cb.checked = false);
                    document.querySelector('input[value="arxiv"]').checked = true;
                """)

                await asyncio.sleep(1)
                await page.screenshot(path=str(self.screenshot_dir / "test_03_form_filled.png"))
                print("   ✅ Form filled: 'large language models', 3 papers from arXiv")

                # Step 4: Start collection
                print("\n📍 Step 4: Starting paper collection...")
                print("   ⏳ This will take 30-60 seconds...")

                await page.click("#collect-button")
                await asyncio.sleep(2)
                await page.screenshot(path=str(self.screenshot_dir / "test_04_collection_started.png"))

                # Wait for collection to complete (max 2 minutes)
                print("   ⏳ Waiting for collection to complete...")
                collection_complete = False
                for i in range(24):  # 24 x 5 seconds = 2 minutes
                    await asyncio.sleep(5)

                    # Check if results appeared
                    results_text = await page.evaluate("document.body.textContent")
                    if "collected" in results_text.lower() or "papers" in results_text.lower():
                        print(f"   ⏱️  Collection appears complete after {(i+1)*5} seconds")
                        collection_complete = True
                        break

                    if i % 3 == 0:  # Progress update every 15 seconds
                        print(f"   ⏳ Still collecting... ({(i+1)*5}s elapsed)")

                await page.screenshot(path=str(self.screenshot_dir / "test_05_collection_complete.png"), full_page=True)

                # Step 5: Check vector stats AFTER collection
                print("\n📍 Step 5: Checking vector database AFTER collection...")
                await asyncio.sleep(5)  # Give time for embedding generation

                stats_after = requests.get(
                    f"{self.base_url}/v1/vector/stats",
                    headers={"X-API-Key": self.api_key}
                ).json()

                vectors_after = stats_after.get('total_vectors', 0)
                print(f"   Vectors AFTER:  {vectors_after}")
                print(f"   Vectors ADDED:  {vectors_after - stats_before.get('total_vectors', 0)}")

                # Step 6: Navigate to vector section
                print("\n📍 Step 6: Navigating to vector space section...")
                await page.evaluate("document.querySelector('.vector-space')?.scrollIntoView({behavior: 'smooth'})")
                await asyncio.sleep(2)
                await page.screenshot(path=str(self.screenshot_dir / "test_06_vector_section.png"))

                # Step 7: Test vector visualization (should work now!)
                print("\n📍 Step 7: Testing vector visualization...")
                if vectors_after > 0:
                    # Click visualize button if it exists
                    try:
                        await page.click("#visualize-button", timeout=5000)
                        await asyncio.sleep(5)  # Wait for visualization to render
                        await page.screenshot(path=str(self.screenshot_dir / "test_07_vector_visualization.png"))
                        print("   ✅ Vector visualization generated!")
                    except:
                        print("   ⚠️  Visualization button not found (may need manual trigger)")
                        # Try via API
                        viz_result = requests.get(
                            f"{self.base_url}/v1/vector/visualize?method=pca&dimensions=3",
                            headers={"X-API-Key": self.api_key}
                        )
                        if viz_result.status_code == 200:
                            print("   ✅ Vector visualization API works!")
                        else:
                            print(f"   ⚠️  Visualization status: {viz_result.status_code}")
                else:
                    print("   ⚠️  No vectors to visualize")

                # Step 8: Get detailed stats
                print("\n📍 Step 8: Getting detailed statistics...")
                graph_stats = requests.get(
                    f"{self.base_url}/v1/graph/export",
                    headers={"X-API-Key": self.api_key}
                ).json()

                print("\n" + "="*70)
                print("📊 FINAL RESULTS")
                print("="*70)
                print(f"Papers Collected:    ~3 (requested)")
                print(f"Graph Nodes:         {len(graph_stats.get('nodes', []))}")
                print(f"Graph Edges:         {len(graph_stats.get('edges', []))}")
                print(f"Vector Embeddings:   {vectors_after}")
                print(f"Embeddings Created:  {vectors_after - stats_before.get('total_vectors', 0)}")
                print("="*70)

                # Verdict
                print("\n" + "="*70)
                if vectors_after > stats_before.get('total_vectors', 0):
                    print("✅ AUTO-EMBEDDING VERIFIED!")
                    print("   Embeddings were automatically generated during collection!")
                    print(f"   System created {vectors_after - stats_before.get('total_vectors', 0)} embeddings")
                else:
                    print("❌ AUTO-EMBEDDING FAILED")
                    print("   No embeddings were created")
                print("="*70)

                # Save results
                results = {
                    "test_date": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "vectors_before": stats_before.get('total_vectors', 0),
                    "vectors_after": vectors_after,
                    "vectors_created": vectors_after - stats_before.get('total_vectors', 0),
                    "graph_nodes": len(graph_stats.get('nodes', [])),
                    "graph_edges": len(graph_stats.get('edges', [])),
                    "auto_embedding_working": vectors_after > stats_before.get('total_vectors', 0)
                }

                with open(self.screenshot_dir / "embedding_test_results.json", 'w') as f:
                    json.dump(results, f, indent=2)

                print(f"\n📸 Screenshots saved to: {self.screenshot_dir}/")
                print(f"📄 Results saved to: {self.screenshot_dir}/embedding_test_results.json")

                print("\n⏸️  Browser will stay open for 10 seconds...")
                await asyncio.sleep(10)

            finally:
                await browser.close()

if __name__ == "__main__":
    tester = EmbeddingVerificationTest()
    asyncio.run(tester.run_test())
