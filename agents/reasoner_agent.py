"""
ReasoningAgent - Complex Reasoning with Conversation Memory
==========================================================

Synthesizes answers from graph and vector search results
Maintains conversation history for contextual understanding
"""

import os
import logging
from typing import List, Dict, Optional
import google.generativeai as genai

logger = logging.getLogger(__name__)


class ReasoningAgent:
    """Complex reasoning with conversation memory"""

    def __init__(self, graph_agent, vector_agent, config: Optional[Dict] = None):
        """
        Initialize with graph and vector agents

        config = {
            "conversation_memory": 5,  # Number of turns to remember
            "max_context_length": 4000,  # Max chars in context
            "temperature": 0.7,
            "max_tokens": 2048
        }
        """
        self.graph_agent = graph_agent
        self.vector_agent = vector_agent
        self.config = config or {}

        self.conversation_memory = self.config.get("conversation_memory", 5)
        self.max_context_length = self.config.get("max_context_length", 4000)
        self.temperature = self.config.get("temperature", 0.7)
        self.max_tokens = self.config.get("max_tokens", 2048)

        self.conversation_history = []

        # Initialize Gemini for reasoning
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            self.config.get("model", "gemini-2.0-flash"),
            generation_config={
                "temperature": self.temperature,
                "max_output_tokens": self.max_tokens
            }
        )

        logger.info(f"ReasoningAgent initialized (memory={self.conversation_memory} turns)")

    def synthesize_answer(self, query: str) -> str:
        """Synthesize answer from graph and vector search"""
        logger.info(f"Synthesizing answer for: '{query}'")

        # Build conversation context
        conversation_context = self._build_conversation_context()

        # Retrieve from graph
        graph_results = self.graph_agent.query_graph(query, max_hops=2)
        graph_context = self._format_graph_results(graph_results)

        # Retrieve from vector database
        vector_results = self.vector_agent.search(query, top_k=5)
        vector_context = self._format_vector_results(vector_results)

        # Build comprehensive prompt
        prompt = self._build_prompt(query, conversation_context, graph_context, vector_context)

        # Generate answer
        try:
            response = self.model.generate_content(prompt)
            answer = response.text.strip()
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            answer = f"I encountered an error while generating the answer: {e}"

        # Save to conversation history
        self.conversation_history.append({
            "query": query,
            "answer": answer,
            "graph_results": len(graph_results),
            "vector_results": len(vector_results)
        })

        # Trim history if too long
        if len(self.conversation_history) > self.conversation_memory:
            self.conversation_history = self.conversation_history[-self.conversation_memory:]

        logger.info(f"Answer synthesized (graph={len(graph_results)}, vector={len(vector_results)})")
        return answer

    def _build_conversation_context(self) -> str:
        """Build conversation context from history"""
        if not self.conversation_history:
            return ""

        context = "PREVIOUS CONVERSATION:\n"
        for i, turn in enumerate(self.conversation_history[-3:], 1):
            context += f"\nTurn {i}:\n"
            context += f"  User: {turn['query']}\n"
            context += f"  Agent: {turn['answer'][:200]}...\n"

        return context

    def _format_graph_results(self, results: List[Dict]) -> str:
        """Format graph query results"""
        if not results:
            return "No graph relationships found."

        context = "KNOWLEDGE GRAPH RELATIONSHIPS:\n"
        for i, path in enumerate(results[:10], 1):
            nodes = path.get("nodes", [])
            rels = path.get("relationships", [])

            if len(nodes) >= 2:
                path_str = nodes[0]
                for j in range(len(rels)):
                    if j + 1 < len(nodes):
                        path_str += f" -{rels[j]}-> {nodes[j+1]}"
                context += f"{i}. {path_str}\n"

        return context

    def _format_vector_results(self, results: List[Dict]) -> str:
        """Format vector search results"""
        if not results:
            return "No relevant documents found."

        context = "RELEVANT RESEARCH:\n"
        for i, chunk in enumerate(results, 1):
            context += f"\n{i}. {chunk.get('title', 'Unknown')} (Score: {chunk.get('score', 0):.3f})\n"
            context += f"   Source: {chunk.get('source', 'Unknown')}\n"
            context += f"   {chunk.get('text', '')[:300]}...\n"

        return context

    def _build_prompt(self, query: str, conv_context: str, graph_context: str, vector_context: str) -> str:
        """Build comprehensive prompt for answer generation"""
        prompt = f"""You are a research assistant with access to a knowledge graph and research papers.

{conv_context}

CURRENT QUESTION: {query}

AVAILABLE INFORMATION:

{graph_context}

{vector_context}

INSTRUCTIONS:
1. If there is previous conversation context, use it to understand references (e.g., "them", "that", "they")
2. Synthesize information from both the knowledge graph and research papers
3. Provide a comprehensive, accurate answer
4. If information is limited, acknowledge what you know and don't know
5. Cite sources when possible (e.g., "According to [source]...")
6. Be conversational and helpful

ANSWER:"""

        # Truncate if too long
        if len(prompt) > self.max_context_length:
            # Prioritize current query and recent context
            prompt = f"""You are a research assistant with access to a knowledge graph and research papers.

CURRENT QUESTION: {query}

{vector_context[:self.max_context_length // 2]}

{graph_context[:self.max_context_length // 4]}

Provide a comprehensive answer based on the available information.

ANSWER:"""

        return prompt

    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        logger.info("Conversation history cleared")

    def get_history(self) -> List[Dict]:
        """Get conversation history"""
        return self.conversation_history

    def get_stats(self) -> Dict:
        """Get reasoning statistics"""
        return {
            "conversation_turns": len(self.conversation_history),
            "memory_limit": self.conversation_memory,
            "total_graph_results": sum(t.get("graph_results", 0) for t in self.conversation_history),
            "total_vector_results": sum(t.get("vector_results", 0) for t in self.conversation_history)
        }
