"""
CriticAgent - Self-Evaluation & Quality Assurance
=================================================

Reviews outputs from other agents for quality, accuracy, and completeness
"""

import logging
import google.generativeai as genai
from typing import Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class CriticAgent:
    """Self-critic agent for quality assurance and output validation"""

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize CriticAgent with Gemini for evaluation

        config = {
            "model": "gemini-2.0-flash",
            "temperature": 0.1,  # Low temperature for consistent evaluation
            "quality_threshold": 0.7  # Minimum quality score to pass
        }
        """
        self.config = config or {}
        self.model_name = self.config.get("model", "gemini-2.0-flash")
        self.temperature = self.config.get("temperature", 0.1)
        self.quality_threshold = self.config.get("quality_threshold", 0.7)

        # Initialize Gemini
        self.model = genai.GenerativeModel(self.model_name)

        # Evaluation history
        self.evaluations = []

        logger.info(f"CriticAgent initialized with {self.model_name}")

    def evaluate_paper_collection(self, papers: List[Dict]) -> Dict:
        """
        Evaluate the quality of collected papers

        Returns:
            {
                "overall_score": float (0-1),
                "relevance_score": float (0-1),
                "diversity_score": float (0-1),
                "quality_score": float (0-1),
                "issues": List[str],
                "recommendations": List[str],
                "passed": bool
            }
        """
        if not papers:
            return {
                "overall_score": 0.0,
                "issues": ["No papers collected"],
                "passed": False
            }

        # Analyze paper metadata
        fields = [p.get('field', 'Unknown') for p in papers]
        years = [p.get('year', 0) for p in papers]
        sources = [p.get('source', 'Unknown') for p in papers]

        # Calculate diversity
        field_diversity = len(set(fields)) / len(papers) if papers else 0
        source_diversity = len(set(sources)) / len(papers) if papers else 0
        year_range = max(years) - min(years) if years and min(years) > 0 else 0

        diversity_score = (field_diversity + source_diversity) / 2

        # Prepare evaluation prompt
        paper_titles = [p.get('title', 'Untitled')[:100] for p in papers[:10]]  # First 10

        prompt = f"""
As a research paper quality evaluator, analyze this collection:

Papers collected: {len(papers)}
Sample titles:
{chr(10).join(f"- {t}" for t in paper_titles)}

Fields: {', '.join(set(fields))}
Year range: {min(years) if years else 'N/A'} - {max(years) if years else 'N/A'}
Sources: {', '.join(set(sources))}

Evaluate on a scale of 0.0-1.0:
1. RELEVANCE: How relevant are these papers to each other?
2. QUALITY: How reputable/impactful are these papers?
3. COMPLETENESS: Is the collection comprehensive?

Provide your evaluation in this exact JSON format:
{{
    "relevance_score": 0.0-1.0,
    "quality_score": 0.0-1.0,
    "completeness_score": 0.0-1.0,
    "issues": ["issue1", "issue2"],
    "recommendations": ["rec1", "rec2"]
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
            # Extract JSON from markdown code blocks if present
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0].strip()

            evaluation = json.loads(result_text)

            # Calculate overall score
            overall_score = (
                evaluation.get("relevance_score", 0.5) * 0.4 +
                evaluation.get("quality_score", 0.5) * 0.4 +
                evaluation.get("completeness_score", 0.5) * 0.2
            )

            result = {
                "overall_score": round(overall_score, 2),
                "relevance_score": round(evaluation.get("relevance_score", 0.5), 2),
                "quality_score": round(evaluation.get("quality_score", 0.5), 2),
                "completeness_score": round(evaluation.get("completeness_score", 0.5), 2),
                "diversity_score": round(diversity_score, 2),
                "issues": evaluation.get("issues", []),
                "recommendations": evaluation.get("recommendations", []),
                "passed": overall_score >= self.quality_threshold,
                "metadata": {
                    "papers_count": len(papers),
                    "fields": list(set(fields)),
                    "year_range": f"{min(years) if years else 'N/A'}-{max(years) if years else 'N/A'}",
                    "sources": list(set(sources))
                }
            }

            logger.info(f"Paper collection evaluated: score={overall_score:.2f}, passed={result['passed']}")
            self.evaluations.append(result)

            return result

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            # Return None for scores to indicate evaluation failure
            return {
                "overall_score": diversity_score,  # Use diversity as fallback
                "relevance_score": None,
                "quality_score": None,
                "completeness_score": None,
                "diversity_score": diversity_score,
                "issues": [f"LLM evaluation unavailable: {str(e)[:100]}"],
                "recommendations": ["Check API key configuration"],
                "passed": False
            }

    def evaluate_answer(self, question: str, answer: str, context: Dict) -> Dict:
        """
        Evaluate the quality of an answer from ReasoningAgent

        Args:
            question: User's question
            answer: Agent's answer
            context: Supporting context (papers, graph data, etc.)

        Returns:
            {
                "overall_score": float (0-1),
                "accuracy_score": float (0-1),
                "completeness_score": float (0-1),
                "clarity_score": float (0-1),
                "citation_score": float (0-1),
                "issues": List[str],
                "suggestions": List[str],
                "passed": bool
            }
        """
        # Get context info
        papers_used = context.get('papers_used', [])
        graph_data = context.get('graph_data', {})

        prompt = f"""
As a research answer quality evaluator, analyze this Q&A pair:

QUESTION: {question}

ANSWER: {answer}

CONTEXT:
- Papers referenced: {len(papers_used)}
- Graph nodes: {graph_data.get('nodes', 0)}
- Graph edges: {graph_data.get('edges', 0)}

Evaluate on a scale of 0.0-1.0:
1. ACCURACY: Is the answer factually correct?
2. COMPLETENESS: Does it fully answer the question?
3. CLARITY: Is it clear and well-structured?
4. CITATIONS: Are sources properly referenced?

Provide evaluation in this exact JSON format:
{{
    "accuracy_score": 0.0-1.0,
    "completeness_score": 0.0-1.0,
    "clarity_score": 0.0-1.0,
    "citation_score": 0.0-1.0,
    "issues": ["issue1", "issue2"],
    "suggestions": ["suggestion1", "suggestion2"]
}}
"""

        try:
            response = self.model.generate_content(
                prompt,
                generation_config={"temperature": self.temperature}
            )

            # Parse JSON
            import json
            result_text = response.text.strip()
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0].strip()

            evaluation = json.loads(result_text)

            # Calculate overall score
            overall_score = (
                evaluation.get("accuracy_score", 0.5) * 0.4 +
                evaluation.get("completeness_score", 0.5) * 0.3 +
                evaluation.get("clarity_score", 0.5) * 0.2 +
                evaluation.get("citation_score", 0.5) * 0.1
            )

            result = {
                "overall_score": round(overall_score, 2),
                "accuracy_score": round(evaluation.get("accuracy_score", 0.5), 2),
                "completeness_score": round(evaluation.get("completeness_score", 0.5), 2),
                "clarity_score": round(evaluation.get("clarity_score", 0.5), 2),
                "citation_score": round(evaluation.get("citation_score", 0.5), 2),
                "issues": evaluation.get("issues", []),
                "suggestions": evaluation.get("suggestions", []),
                "passed": overall_score >= self.quality_threshold,
                "timestamp": datetime.now().isoformat()
            }

            logger.info(f"Answer evaluated: score={overall_score:.2f}, passed={result['passed']}")
            self.evaluations.append(result)

            return result

        except Exception as e:
            logger.error(f"Answer evaluation failed: {e}")
            return {
                "overall_score": 0.5,
                "accuracy_score": None,
                "completeness_score": None,
                "clarity_score": None,
                "citation_score": None,
                "issues": [f"LLM evaluation unavailable: {str(e)[:100]}"],
                "suggestions": ["Check API key configuration"],
                "passed": False
            }

    def evaluate_graph_quality(self, graph_stats: Dict) -> Dict:
        """
        Evaluate knowledge graph construction quality

        Args:
            graph_stats: Statistics from KnowledgeGraphAgent

        Returns:
            Quality evaluation with score and issues
        """
        nodes = graph_stats.get('nodes', 0)
        edges = graph_stats.get('edges', 0)

        # Calculate graph metrics
        density = edges / (nodes * (nodes - 1)) if nodes > 1 else 0
        avg_degree = (2 * edges) / nodes if nodes > 0 else 0

        issues = []
        recommendations = []

        # Check for common issues
        if nodes < 10:
            issues.append("Graph too small - insufficient papers processed")
            recommendations.append("Collect more papers to build richer knowledge graph")

        if density < 0.1:
            issues.append("Graph too sparse - weak relationships between entities")
            recommendations.append("Improve entity extraction or relationship detection")

        if avg_degree < 2:
            issues.append("Low connectivity - entities not well connected")
            recommendations.append("Extract more relationships from paper content")

        # Calculate quality score
        node_score = min(nodes / 50, 1.0)  # Target: 50+ nodes
        density_score = min(density / 0.3, 1.0)  # Target: 30% density
        degree_score = min(avg_degree / 4, 1.0)  # Target: avg degree 4

        overall_score = (node_score * 0.4 + density_score * 0.3 + degree_score * 0.3)

        result = {
            "overall_score": round(overall_score, 2),
            "node_score": round(node_score, 2),
            "density_score": round(density_score, 2),
            "degree_score": round(degree_score, 2),
            "metrics": {
                "nodes": nodes,
                "edges": edges,
                "density": round(density, 3),
                "avg_degree": round(avg_degree, 2)
            },
            "issues": issues,
            "recommendations": recommendations,
            "passed": overall_score >= self.quality_threshold
        }

        logger.info(f"Graph evaluated: nodes={nodes}, edges={edges}, score={overall_score:.2f}")
        self.evaluations.append(result)

        return result

    def get_evaluation_history(self) -> List[Dict]:
        """Get all evaluation history"""
        return self.evaluations

    def get_overall_quality_report(self) -> Dict:
        """Generate overall quality report across all evaluations"""
        if not self.evaluations:
            return {"message": "No evaluations performed yet"}

        avg_score = sum(e.get('overall_score', 0) for e in self.evaluations) / len(self.evaluations)
        passed_count = sum(1 for e in self.evaluations if e.get('passed', False))

        all_issues = []
        all_recommendations = []
        for e in self.evaluations:
            all_issues.extend(e.get('issues', []))
            all_recommendations.extend(e.get('recommendations', []))

        return {
            "total_evaluations": len(self.evaluations),
            "average_score": round(avg_score, 2),
            "passed_count": passed_count,
            "pass_rate": round(passed_count / len(self.evaluations), 2),
            "common_issues": list(set(all_issues))[:5],
            "top_recommendations": list(set(all_recommendations))[:5],
            "quality_trend": "improving" if avg_score > 0.7 else "needs_attention"
        }
