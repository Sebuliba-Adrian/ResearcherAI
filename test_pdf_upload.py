"""
Test PDF Upload Functionality
=============================

Tests the PDF upload endpoint with a real PDF file
"""

import os
import sys
import requests
import logging
from utils.pdf_parser import parse_pdf_file

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_pdf_parser():
    """Test PDF parsing with a downloaded arXiv paper"""
    print("=" * 100)
    print("TEST 1: PDF Parser - Download and Parse arXiv Paper")
    print("=" * 100)
    print()

    # Download a sample PDF from arXiv
    arxiv_url = "https://arxiv.org/pdf/1706.03762.pdf"  # "Attention Is All You Need"
    pdf_path = "./test_outputs/attention_is_all_you_need.pdf"

    os.makedirs("./test_outputs", exist_ok=True)

    try:
        # Download PDF
        print(f"📥 Downloading PDF from: {arxiv_url}")
        response = requests.get(arxiv_url, timeout=30)
        response.raise_for_status()

        with open(pdf_path, 'wb') as f:
            f.write(response.content)

        print(f"✅ Downloaded {len(response.content) / 1024:.1f} KB")
        print()

        # Parse PDF
        print("📄 Parsing PDF...")
        paper = parse_pdf_file(pdf_path)

        # Display results
        print("=" * 100)
        print("EXTRACTED METADATA")
        print("=" * 100)
        print(f"ID: {paper['id']}")
        print(f"Title: {paper['title']}")
        print(f"Authors: {', '.join(paper['authors'][:3])}{'...' if len(paper['authors']) > 3 else ''}")
        print(f"Published: {paper['published']}")
        print(f"Source: {paper['source']}")
        print(f"Topics: {', '.join(paper['topics'][:5])}")
        print(f"Num Pages: {paper['num_pages']}")
        print()
        print(f"Abstract ({len(paper['abstract'])} chars):")
        print(paper['abstract'][:300] + "..." if len(paper['abstract']) > 300 else paper['abstract'])
        print()
        print(f"Full Text: {len(paper['full_text'])} characters extracted")
        print()

        return paper

    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_api_upload(pdf_path: str):
    """Test PDF upload via API (requires running API server)"""
    print("=" * 100)
    print("TEST 2: API Upload Endpoint")
    print("=" * 100)
    print()

    API_URL = "http://localhost:8000/v1/upload/pdf"
    API_KEY = "demo-key-123"

    if not os.path.exists(pdf_path):
        print(f"❌ PDF file not found: {pdf_path}")
        print("   Run test_pdf_parser() first to download the PDF")
        return

    try:
        print(f"📤 Uploading PDF to API: {API_URL}")
        print(f"   File: {pdf_path}")
        print()

        with open(pdf_path, 'rb') as f:
            files = {'file': ('attention_paper.pdf', f, 'application/pdf')}
            headers = {'X-API-Key': API_KEY}

            response = requests.post(API_URL, files=files, headers=headers, timeout=60)

        if response.status_code == 200:
            result = response.json()
            print("✅ Upload successful!")
            print()
            print("Response:")
            print(f"  Success: {result['success']}")
            print(f"  Message: {result['message']}")
            print(f"  Session: {result['session_name']}")
            print()
            print("  Paper:")
            print(f"    ID: {result['paper']['id']}")
            print(f"    Title: {result['paper']['title']}")
            print(f"    Authors: {len(result['paper']['authors'])} authors")
            print(f"    Pages: {result['paper']['num_pages']}")
            print()
            print("  Graph Stats:")
            print(f"    {result['graph_stats']}")
            print()
            print("  Vector Stats:")
            print(f"    {result['vector_stats']}")
            print()

        else:
            print(f"❌ Upload failed with status {response.status_code}")
            print(f"   Response: {response.text}")
            print()
            print("   NOTE: Make sure the API server is running:")
            print("   $ python api_gateway.py")

    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API server")
        print()
        print("   Please start the API server first:")
        print("   $ source venv/bin/activate")
        print("   $ export GOOGLE_API_KEY='your-api-key'")
        print("   $ python api_gateway.py")
        print()

    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run all tests"""
    print("\n")
    print("╔" + "═" * 98 + "╗")
    print("║" + " " * 30 + "PDF UPLOAD FUNCTIONALITY TEST" + " " * 39 + "║")
    print("╚" + "═" * 98 + "╝")
    print()

    # Test 1: PDF Parser
    paper = test_pdf_parser()

    if not paper:
        print("\n❌ PDF parsing failed. Skipping API test.")
        return 1

    print("\n" + "=" * 100)
    print()

    # Ask if user wants to test API
    print("Would you like to test the API upload endpoint?")
    print("(This requires the API server to be running)")
    print()
    response = input("Test API upload? (y/n): ").strip().lower()

    if response == 'y':
        test_api_upload("./test_outputs/attention_is_all_you_need.pdf")
    else:
        print("\nSkipping API test.")
        print()
        print("To test the API manually:")
        print("1. Start the API server: python api_gateway.py")
        print("2. Run: python test_pdf_upload.py")
        print("3. Choose 'y' when prompted")

    print()
    print("=" * 100)
    print("✅ TESTS COMPLETE")
    print("=" * 100)

    return 0


if __name__ == "__main__":
    sys.exit(main())
