"""
SummarizationAgent - Intelligent Paper & Conversation Summarization
==================================================================

Multi-mode summarization with different styles and lengths
"""

import os
import logging
from typing import List, Dict, Optional, Literal
from enum import Enum
import google.generativeai as genai

logger = logging.getLogger(__name__)


class SummaryStyle(str, Enum):
    """Summary style options"""
    EXECUTIVE = "executive"           # Brief, key points only
    TECHNICAL = "technical"           # Detailed, preserves technical terms
    ABSTRACT = "abstract"             # Academic abstract style
    BULLET_POINTS = "bullet_points"   # Bulleted list format
    NARRATIVE = "narrative"           # Story-like, flowing text
    COMPARISON = "comparison"         # Compare multiple papers
    TIMELINE = "timeline"             # Chronological developments


class SummaryLength(str, Enum):
    """Summary length options"""
    BRIEF = "brief"         # 2-3 sentences
    SHORT = "short"         # 1 paragraph (100-150 words)
    MEDIUM = "medium"       # 2-3 paragraphs (200-300 words)
    DETAILED = "detailed"   # 3-5 paragraphs (400-600 words)
    COMPREHENSIVE = "comprehensive"  # Full analysis (800+ words)


class SummarizationAgent:
    """Intelligent summarization with multiple modes and styles"""

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize SummarizationAgent

        config = {
            "model": "gemini-2.0-flash",
            "temperature": 0.3,  # Lower for consistent summaries
            "default_style": "executive",
            "default_length": "medium"
        }
        """
        self.config = config or {}
        self.model_name = self.config.get("model", "gemini-2.0-flash")
        self.temperature = self.config.get("temperature", 0.3)
        self.default_style = self.config.get("default_style", SummaryStyle.EXECUTIVE)
        self.default_length = self.config.get("default_length", SummaryLength.MEDIUM)

        # Initialize Gemini
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(self.model_name)

        # Summary history
        self.summaries = []

        logger.info(f"SummarizationAgent initialized with {self.model_name}")

    # ========================================================================
    # SINGLE PAPER SUMMARIZATION
    # ========================================================================

    def summarize_paper(
        self,
        paper: Dict,
        style: SummaryStyle = None,
        length: SummaryLength = None,
        focus_areas: Optional[List[str]] = None
    ) -> Dict:
        """
        Summarize a single research paper

        Args:
            paper: Paper metadata (title, abstract, authors, year, etc.)
            style: Summary style (executive, technical, abstract, etc.)
            length: Summary length (brief, short, medium, detailed, comprehensive)
            focus_areas: Specific aspects to focus on (e.g., ["methodology", "results"])

        Returns:
            {
                "summary": str,
                "key_points": List[str],
                "methodology": str (optional),
                "findings": str (optional),
                "impact": str (optional),
                "style": str,
                "length": str,
                "word_count": int
            }
        """
        style = style or self.default_style
        length = length or self.default_length

        # Build prompt based on style and length
        prompt = self._build_paper_prompt(paper, style, length, focus_areas)

        try:
            response = self.model.generate_content(
                prompt,
                generation_config={"temperature": self.temperature}
            )

            summary_text = response.text.strip()

            # Extract structured information based on style
            result = self._parse_summary_response(summary_text, style)

            result.update({
                "paper_title": paper.get("title", "Unknown"),
                "style": style,
                "length": length,
                "word_count": len(summary_text.split())
            })

            logger.info(f"Paper summarized: {paper.get('title', 'Unknown')[:50]}... ({result['word_count']} words)")
            self.summaries.append(result)

            return result

        except Exception as e:
            logger.error(f"Paper summarization failed: {e}")
            return {
                "error": str(e),
                "summary": f"Failed to summarize: {str(e)}",
                "style": style,
                "length": length
            }

    # ========================================================================
    # MULTI-PAPER SUMMARIZATION
    # ========================================================================

    def summarize_collection(
        self,
        papers: List[Dict],
        style: SummaryStyle = SummaryStyle.EXECUTIVE,
        focus: str = "research trends"
    ) -> Dict:
        """
        Summarize a collection of papers

        Args:
            papers: List of papers
            style: Summary style
            focus: What to focus on (trends, methods, findings, gaps)

        Returns:
            {
                "overall_summary": str,
                "key_themes": List[str],
                "research_trends": List[str],
                "knowledge_gaps": List[str],
                "top_papers": List[Dict],
                "timeline": Dict (if applicable)
            }
        """
        if not papers:
            return {"error": "No papers to summarize", "overall_summary": ""}

        # Build collection summary prompt
        prompt = f"""Analyze this collection of {len(papers)} research papers and provide a comprehensive summary focusing on: {focus}

PAPERS:
"""
        for i, paper in enumerate(papers[:20], 1):  # Limit to 20 for context
            prompt += f"\n{i}. {paper.get('title', 'Unknown')} ({paper.get('year', 'N/A')})\n"
            if 'abstract' in paper:
                prompt += f"   {paper['abstract'][:200]}...\n"
            prompt += f"   Field: {paper.get('field', 'Unknown')}\n"

        prompt += f"""

Provide a {style} summary that includes:

1. OVERALL SUMMARY (3-5 sentences)
   - Main research themes
   - Common methodologies
   - Key findings

2. KEY THEMES (bullet points)
   - Identify 3-5 major themes

3. RESEARCH TRENDS
   - Emerging patterns
   - Evolution over time

4. KNOWLEDGE GAPS
   - What's missing
   - Future research directions

5. TOP PAPERS (3 most impactful)
   - Title, year, and why it's significant

Format as JSON:
{{
    "overall_summary": "...",
    "key_themes": ["theme1", "theme2", ...],
    "research_trends": ["trend1", "trend2", ...],
    "knowledge_gaps": ["gap1", "gap2", ...],
    "top_papers": [
        {{"title": "...", "year": 2020, "significance": "..."}},
        ...
    ]
}}
"""

        try:
            response = self.model.generate_content(
                prompt,
                generation_config={"temperature": self.temperature}
            )

            # Parse JSON response
            import json
            result_text = response.text.strip()

            # Extract JSON from markdown if present
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0].strip()

            result = json.loads(result_text)
            result["papers_analyzed"] = len(papers)
            result["style"] = style

            logger.info(f"Collection summarized: {len(papers)} papers")
            self.summaries.append(result)

            return result

        except Exception as e:
            logger.error(f"Collection summarization failed: {e}")
            # Fallback to simple summary
            return {
                "overall_summary": f"Collection of {len(papers)} papers analyzed. Error: {str(e)}",
                "papers_analyzed": len(papers),
                "error": str(e)
            }

    # ========================================================================
    # CONVERSATION SUMMARIZATION
    # ========================================================================

    def summarize_conversation(
        self,
        conversation_history: List[Dict],
        style: SummaryStyle = SummaryStyle.BULLET_POINTS
    ) -> Dict:
        """
        Summarize a conversation/research session

        Args:
            conversation_history: List of Q&A turns
            style: Summary style

        Returns:
            {
                "session_summary": str,
                "questions_asked": List[str],
                "key_insights": List[str],
                "topics_covered": List[str],
                "turn_count": int
            }
        """
        if not conversation_history:
            return {"error": "No conversation to summarize"}

        prompt = f"""Summarize this research conversation session ({len(conversation_history)} turns):

CONVERSATION:
"""
        for i, turn in enumerate(conversation_history, 1):
            prompt += f"\nTurn {i}:\n"
            prompt += f"Question: {turn.get('query', turn.get('question', 'Unknown'))}\n"
            prompt += f"Answer: {turn.get('answer', '')[:300]}...\n"

        prompt += f"""

Provide a {style} summary including:

1. SESSION SUMMARY (2-3 sentences overview)
2. QUESTIONS ASKED (list all questions)
3. KEY INSIGHTS (main discoveries/learnings)
4. TOPICS COVERED (research areas discussed)

Format as JSON:
{{
    "session_summary": "...",
    "questions_asked": ["q1", "q2", ...],
    "key_insights": ["insight1", "insight2", ...],
    "topics_covered": ["topic1", "topic2", ...]
}}
"""

        try:
            response = self.model.generate_content(
                prompt,
                generation_config={"temperature": self.temperature}
            )

            import json
            result_text = response.text.strip()

            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0].strip()

            result = json.loads(result_text)
            result["turn_count"] = len(conversation_history)
            result["style"] = style

            logger.info(f"Conversation summarized: {len(conversation_history)} turns")
            return result

        except Exception as e:
            logger.error(f"Conversation summarization failed: {e}")
            return {
                "session_summary": f"Session with {len(conversation_history)} Q&A turns",
                "turn_count": len(conversation_history),
                "error": str(e)
            }

    # ========================================================================
    # COMPARATIVE SUMMARIZATION
    # ========================================================================

    def compare_papers(
        self,
        papers: List[Dict],
        comparison_aspects: Optional[List[str]] = None
    ) -> Dict:
        """
        Compare multiple papers side-by-side

        Args:
            papers: List of papers to compare (typically 2-5)
            comparison_aspects: What to compare (methodology, results, datasets, etc.)

        Returns:
            {
                "comparison_summary": str,
                "similarities": List[str],
                "differences": List[str],
                "strengths_weaknesses": Dict,
                "recommendation": str
            }
        """
        if len(papers) < 2:
            return {"error": "Need at least 2 papers to compare"}

        aspects = comparison_aspects or ["methodology", "results", "datasets", "contributions"]

        prompt = f"""Compare these {len(papers)} research papers across: {', '.join(aspects)}

PAPERS:
"""
        for i, paper in enumerate(papers, 1):
            prompt += f"\n{i}. {paper.get('title', 'Unknown')} ({paper.get('year', 'N/A')})\n"
            prompt += f"   Authors: {', '.join(paper.get('authors', ['Unknown'])[:3])}\n"
            if 'abstract' in paper:
                prompt += f"   Abstract: {paper['abstract'][:300]}...\n"

        prompt += """

Provide a comparative analysis in JSON format:
{
    "comparison_summary": "2-3 sentence overview of how these papers relate",
    "similarities": ["similarity1", "similarity2", ...],
    "differences": ["difference1", "difference2", ...],
    "strengths_weaknesses": {
        "paper1": {"strengths": [...], "weaknesses": [...]},
        "paper2": {"strengths": [...], "weaknesses": [...]}
    },
    "recommendation": "Which paper to read first/use and why"
}
"""

        try:
            response = self.model.generate_content(
                prompt,
                generation_config={"temperature": self.temperature}
            )

            import json
            result_text = response.text.strip()

            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0].strip()

            result = json.loads(result_text)
            result["papers_compared"] = len(papers)

            logger.info(f"Papers compared: {len(papers)}")
            return result

        except Exception as e:
            logger.error(f"Paper comparison failed: {e}")
            return {
                "comparison_summary": f"Comparison of {len(papers)} papers failed",
                "error": str(e)
            }

    # ========================================================================
    # HELPER METHODS
    # ========================================================================

    def _build_paper_prompt(
        self,
        paper: Dict,
        style: SummaryStyle,
        length: SummaryLength,
        focus_areas: Optional[List[str]]
    ) -> str:
        """Build prompt for single paper summarization"""

        # Length guidelines
        length_guide = {
            SummaryLength.BRIEF: "2-3 sentences (50 words max)",
            SummaryLength.SHORT: "1 paragraph (100-150 words)",
            SummaryLength.MEDIUM: "2-3 paragraphs (200-300 words)",
            SummaryLength.DETAILED: "3-5 paragraphs (400-600 words)",
            SummaryLength.COMPREHENSIVE: "Full analysis (800+ words)"
        }

        # Style guidelines
        style_guide = {
            SummaryStyle.EXECUTIVE: "Brief, actionable insights for decision-makers",
            SummaryStyle.TECHNICAL: "Preserve technical details and terminology",
            SummaryStyle.ABSTRACT: "Academic abstract format with background, methods, results, conclusion",
            SummaryStyle.BULLET_POINTS: "Key points as clear bullet list",
            SummaryStyle.NARRATIVE: "Engaging story-like flow",
            SummaryStyle.COMPARISON: "Compare with related work",
            SummaryStyle.TIMELINE: "Chronological progression of ideas"
        }

        prompt = f"""Summarize this research paper in {style_guide.get(style, 'clear')} style.
Length: {length_guide.get(length, 'medium')}

PAPER:
Title: {paper.get('title', 'Unknown')}
Authors: {', '.join(paper.get('authors', ['Unknown'])[:5])}
Year: {paper.get('year', 'N/A')}
Source: {paper.get('source', 'Unknown')}
"""

        if 'abstract' in paper:
            prompt += f"\nAbstract: {paper['abstract']}\n"

        if 'content' in paper:
            prompt += f"\nFull Text: {paper['content'][:2000]}...\n"

        if focus_areas:
            prompt += f"\nFOCUS ON: {', '.join(focus_areas)}\n"

        # Add structure based on style
        if style == SummaryStyle.BULLET_POINTS:
            prompt += """
Provide summary as:
• Main Contribution: ...
• Methodology: ...
• Key Results: ...
• Impact: ...
• Limitations: ...
"""
        elif style == SummaryStyle.ABSTRACT:
            prompt += """
Provide in academic abstract format:
Background: ...
Methods: ...
Results: ...
Conclusions: ...
"""
        elif style == SummaryStyle.TECHNICAL:
            prompt += """
Include:
- Technical approach and algorithms
- Datasets and experimental setup
- Quantitative results with metrics
- Limitations and future work
"""

        prompt += "\nSUMMARY:"
        return prompt

    def _parse_summary_response(self, text: str, style: SummaryStyle) -> Dict:
        """Parse summary response into structured format"""

        # Basic structure
        result = {"summary": text}

        # Try to extract key points if bullet style
        if style == SummaryStyle.BULLET_POINTS or "•" in text or "- " in text:
            lines = text.split("\n")
            key_points = [line.strip("• -").strip() for line in lines if line.strip().startswith(("•", "-", "*"))]
            if key_points:
                result["key_points"] = key_points

        return result

    # ========================================================================
    # UTILITY METHODS
    # ========================================================================

    def get_summary_stats(self) -> Dict:
        """Get summarization statistics"""
        return {
            "total_summaries": len(self.summaries),
            "summary_types": {
                "paper": sum(1 for s in self.summaries if "paper_title" in s),
                "collection": sum(1 for s in self.summaries if "papers_analyzed" in s),
                "conversation": sum(1 for s in self.summaries if "turn_count" in s)
            },
            "avg_word_count": sum(s.get("word_count", 0) for s in self.summaries) / len(self.summaries) if self.summaries else 0
        }

    def clear_history(self):
        """Clear summary history"""
        self.summaries = []
        logger.info("Summary history cleared")
