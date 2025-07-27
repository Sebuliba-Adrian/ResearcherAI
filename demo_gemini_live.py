#!/usr/bin/env python3
"""
Live Demo: Gemini-Powered RAG (Minimal Dependencies)
Shows REAL intelligent reasoning vs simple word matching
"""

import json
import re
import os

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("⚠️  Install: pip install google-generativeai")

# Configuration
GOOGLE_API_KEY = "AIzaSyCGUWaN4uzBBrnXFZ_qWBqKaeSVa13Lip4"
GEMINI_MODEL = "gemini-2.5-flash"

def read_document(path="sample_knowledge.txt"):
    """Read document"""
    with open(path, 'r') as f:
        return f.read()

def chunk_text(text, size=400):
    """Split into chunks"""
    text = re.sub(r'\s+', ' ', text)
    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks = []
    current = ""
    for sent in sentences:
        if len(current) + len(sent) < size:
            current += sent + " "
        else:
            if current.strip():
                chunks.append(current.strip())
            current = sent + " "

    if current.strip():
        chunks.append(current.strip())

    return chunks

def gemini_search_and_answer(query, chunks):
    """Use Gemini to intelligently search and answer"""

    if not GEMINI_AVAILABLE:
        return "❌ Gemini not available. Install google-generativeai"

    # Configure Gemini
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel(GEMINI_MODEL)

    # Step 1: Let Gemini find relevant chunks
    chunks_text = "\n\n---CHUNK SEPARATOR---\n\n".join([
        f"CHUNK {i+1}:\n{chunk}"
        for i, chunk in enumerate(chunks)
    ])

    retrieval_prompt = f"""You are a smart document search system.

Here are the available text chunks:

{chunks_text}

User Query: "{query}"

Task:
1. Identify which chunks (by number) are relevant to answering this query
2. If NO chunks are relevant, say "NO_RELEVANT_CHUNKS"
3. Return ONLY a JSON array of relevant chunk numbers, e.g., [1, 3, 5]

JSON:"""

    try:
        response = model.generate_content(retrieval_prompt)
        response_text = response.text.strip()

        print(f"\n🔍 Gemini Analysis:")
        print(f"Raw response: {response_text}")

        # Check if no relevant chunks
        if "NO_RELEVANT_CHUNKS" in response_text.upper():
            return "❌ No relevant information found in the document."

        # Extract chunk numbers
        start = response_text.find("[")
        end = response_text.rfind("]") + 1

        if start != -1 and end > start:
            chunk_numbers = json.loads(response_text[start:end])
            relevant_chunks = [chunks[i-1] for i in chunk_numbers if 0 < i <= len(chunks)]

            print(f"✅ Found {len(relevant_chunks)} relevant chunks")

            if not relevant_chunks:
                return "❌ No relevant information found."

            # Step 2: Use Gemini to answer based on relevant chunks
            context = "\n\n".join(relevant_chunks)

            answer_prompt = f"""You are a helpful AI assistant.

Context from document:
{context}

User Question: {query}

Instructions:
1. Answer the question based ONLY on the context provided
2. If the context doesn't contain the answer, say "The document doesn't contain information about [topic]"
3. Be concise and accurate
4. Cite what you used from the context

Answer:"""

            response = model.generate_content(answer_prompt)
            return response.text.strip()

        else:
            return "❌ Could not parse Gemini response"

    except Exception as e:
        return f"❌ Gemini error: {str(e)}"

def simple_word_match(query, chunks):
    """OLD METHOD: Simple word overlap (for comparison)"""
    query_words = set(query.lower().split())

    scored = []
    for chunk in chunks:
        chunk_words = set(chunk.lower().split())
        overlap = query_words.intersection(chunk_words)
        score = len(overlap) / len(query_words.union(chunk_words)) if query_words.union(chunk_words) else 0
        scored.append((chunk, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    best_chunk = scored[0][0] if scored else ""

    return f"Based on the knowledge base:\n{best_chunk[:200]}..."

def main():
    """Run interactive demo"""
    print("="*60)
    print("🚀 Gemini-Powered RAG Demo")
    print("="*60)
    print()

    if not GEMINI_AVAILABLE:
        print("❌ google-generativeai not installed")
        print("Run: pip install google-generativeai")
        return

    # Load document
    print("📘 Loading document...")
    text = read_document()
    chunks = chunk_text(text)
    print(f"✅ Loaded {len(chunks)} chunks\n")

    print("💡 Try asking questions to see the difference!")
    print("   Example: 'Who created Claude?'")
    print("   Example: 'What is CRISPR?'")
    print("   Type 'exit' to quit\n")

    while True:
        try:
            query = input("👤 You: ").strip()

            if not query:
                continue

            if query.lower() in ['exit', 'quit', 'q']:
                print("\n👋 Goodbye!")
                break

            # Show both methods for comparison
            print("\n" + "="*60)
            print("📊 COMPARISON: Simple vs Gemini")
            print("="*60)

            print("\n❌ OLD METHOD (Simple Word Overlap):")
            print("-"*60)
            simple_answer = simple_word_match(query, chunks)
            print(simple_answer)

            print("\n" + "="*60)
            print("\n✅ GEMINI METHOD (Intelligent Reasoning):")
            print("-"*60)
            gemini_answer = gemini_search_and_answer(query, chunks)
            print(gemini_answer)
            print()

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
