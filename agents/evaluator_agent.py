"""
EvaluatorAgent - The Secret Weapon Against Chaos
==================================================

Prevents infinite loops, cascade failures, and success ambiguity.
Acts as a gatekeeper with explicit success criteria and quality gates.

Based on production patterns from the multi-agent systems guide.
"""

import logging
import time
from typing import Dict, List, Optional, Any
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class ExecutionStatus(str, Enum):
    """Execution status for agent tasks"""
    SUCCESS = "success"
    FAILURE = "failure"
    LOOP_DETECTED = "loop_detected"
    TIMEOUT = "timeout"
    CRITERIA_NOT_MET = "criteria_not_met"
    ESCALATION_REQUIRED = "escalation_required"


@dataclass
class SuccessCriteria:
    """Explicit success criteria for agent tasks"""
    min_output_length: int = 0
    max_output_length: int = 50000
    required_fields: List[str] = field(default_factory=list)
    forbidden_patterns: List[str] = field(default_factory=list)
    quality_threshold: float = 0.7
    max_execution_time: float = 120.0  # seconds
    expected_output_type: Optional[type] = None


@dataclass
class ExecutionRecord:
    """Record of an agent execution"""
    agent_name: str
    task_id: str
    input_hash: str
    output_hash: str
    timestamp: datetime
    execution_time: float
    status: ExecutionStatus


class EvaluatorAgent:
    """
    Evaluator/Gatekeeper agent that prevents chaos in multi-agent systems.

    Key responsibilities:
    1. Define explicit success criteria before execution
    2. Check every agent output against criteria
    3. Detect infinite loops
    4. Escalate when success criteria can't be met
    5. Prevent cascade failures
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize EvaluatorAgent

        config = {
            "loop_detection_window": 5,  # Check last N executions
            "max_retries": 3,
            "escalation_threshold": 0.5,
            "loop_similarity_threshold": 0.9
        }
        """
        self.config = config or {}

        # Configuration
        self.loop_detection_window = self.config.get("loop_detection_window", 5)
        self.max_retries = self.config.get("max_retries", 3)
        self.escalation_threshold = self.config.get("escalation_threshold", 0.5)
        self.loop_similarity_threshold = self.config.get("loop_similarity_threshold", 0.9)

        # Execution history for loop detection
        self.execution_history: List[ExecutionRecord] = []

        # Retry tracking
        self.retry_counts: Dict[str, int] = {}

        # Statistics
        self.stats = {
            "total_evaluations": 0,
            "successes": 0,
            "failures": 0,
            "loops_detected": 0,
            "escalations": 0,
            "timeouts": 0
        }

        logger.info("EvaluatorAgent initialized with loop detection and success criteria")

    # ========================================================================
    # CORE EVALUATION METHODS
    # ========================================================================

    def evaluate(
        self,
        agent_name: str,
        task_id: str,
        agent_input: Any,
        agent_output: Any,
        criteria: SuccessCriteria,
        execution_time: float
    ) -> Dict[str, Any]:
        """
        Evaluate an agent's execution against success criteria.

        This is the main entry point - call this after every agent execution.

        Returns:
            {
                "status": ExecutionStatus,
                "passed": bool,
                "issues": List[str],
                "should_retry": bool,
                "should_escalate": bool,
                "metadata": Dict
            }
        """
        self.stats["total_evaluations"] += 1
        start_time = time.time()

        issues = []

        # 1. Check for infinite loops (CRITICAL)
        loop_detected = self._check_for_loops(agent_name, task_id, agent_input, agent_output)
        if loop_detected:
            self.stats["loops_detected"] += 1
            return self._create_evaluation_result(
                status=ExecutionStatus.LOOP_DETECTED,
                passed=False,
                issues=["Infinite loop detected - similar inputs/outputs in recent history"],
                should_retry=False,
                should_escalate=True
            )

        # 2. Check execution time
        if execution_time > criteria.max_execution_time:
            self.stats["timeouts"] += 1
            issues.append(f"Execution time ({execution_time:.2f}s) exceeded limit ({criteria.max_execution_time}s)")

        # 3. Validate output against criteria
        validation_issues = self._validate_output(agent_output, criteria)
        issues.extend(validation_issues)

        # 4. Record execution
        self._record_execution(agent_name, task_id, agent_input, agent_output, execution_time)

        # 5. Determine if retry is appropriate
        retry_count = self.retry_counts.get(task_id, 0)
        should_retry = len(issues) > 0 and retry_count < self.max_retries
        should_escalate = len(issues) > 0 and retry_count >= self.max_retries

        # 6. Determine final status
        if len(issues) == 0:
            status = ExecutionStatus.SUCCESS
            self.stats["successes"] += 1
            passed = True
        elif execution_time > criteria.max_execution_time:
            status = ExecutionStatus.TIMEOUT
            self.stats["timeouts"] += 1
            passed = False
        else:
            status = ExecutionStatus.CRITERIA_NOT_MET
            self.stats["failures"] += 1
            passed = False

        if should_escalate:
            self.stats["escalations"] += 1

        evaluation_time = time.time() - start_time

        logger.info(
            f"Evaluation complete: agent={agent_name}, status={status}, "
            f"passed={passed}, issues={len(issues)}, eval_time={evaluation_time:.3f}s"
        )

        return self._create_evaluation_result(
            status=status,
            passed=passed,
            issues=issues,
            should_retry=should_retry,
            should_escalate=should_escalate,
            metadata={
                "retry_count": retry_count,
                "evaluation_time": evaluation_time,
                "execution_time": execution_time
            }
        )

    def _check_for_loops(
        self,
        agent_name: str,
        task_id: str,
        agent_input: Any,
        agent_output: Any
    ) -> bool:
        """
        Detect infinite loops by checking recent execution history.

        A loop is detected when:
        1. Same agent with similar inputs produces similar outputs
        2. Happens within the detection window
        """
        input_hash = self._hash_data(agent_input)
        output_hash = self._hash_data(agent_output)

        # Get recent executions for this agent
        recent_executions = [
            record for record in self.execution_history[-self.loop_detection_window:]
            if record.agent_name == agent_name
        ]

        # Check for similar patterns
        for record in recent_executions:
            input_similarity = self._calculate_similarity(input_hash, record.input_hash)
            output_similarity = self._calculate_similarity(output_hash, record.output_hash)

            # Loop detected if both input and output are very similar
            if (input_similarity > self.loop_similarity_threshold and
                output_similarity > self.loop_similarity_threshold):
                logger.warning(
                    f"Loop detected: agent={agent_name}, task={task_id}, "
                    f"input_similarity={input_similarity:.2f}, "
                    f"output_similarity={output_similarity:.2f}"
                )
                return True

        return False

    def _validate_output(self, output: Any, criteria: SuccessCriteria) -> List[str]:
        """
        Validate output against success criteria.

        Returns list of validation issues (empty if all checks pass).
        """
        issues = []

        # Check output type
        if criteria.expected_output_type and not isinstance(output, criteria.expected_output_type):
            issues.append(
                f"Output type mismatch: expected {criteria.expected_output_type}, "
                f"got {type(output)}"
            )
            return issues  # Can't do further validation if type is wrong

        # For dict outputs, check required fields
        if isinstance(output, dict):
            for field in criteria.required_fields:
                if field not in output:
                    issues.append(f"Missing required field: {field}")

        # Check output length for strings
        if isinstance(output, str):
            if len(output) < criteria.min_output_length:
                issues.append(
                    f"Output too short: {len(output)} chars (min: {criteria.min_output_length})"
                )
            if len(output) > criteria.max_output_length:
                issues.append(
                    f"Output too long: {len(output)} chars (max: {criteria.max_output_length})"
                )

            # Check for forbidden patterns
            for pattern in criteria.forbidden_patterns:
                if pattern.lower() in output.lower():
                    issues.append(f"Forbidden pattern found: {pattern}")

        return issues

    def _record_execution(
        self,
        agent_name: str,
        task_id: str,
        agent_input: Any,
        agent_output: Any,
        execution_time: float
    ):
        """Record execution for loop detection and audit trail"""
        record = ExecutionRecord(
            agent_name=agent_name,
            task_id=task_id,
            input_hash=self._hash_data(agent_input),
            output_hash=self._hash_data(agent_output),
            timestamp=datetime.now(),
            execution_time=execution_time,
            status=ExecutionStatus.SUCCESS  # Will be updated if needed
        )

        self.execution_history.append(record)

        # Keep history manageable (last 1000 records)
        if len(self.execution_history) > 1000:
            self.execution_history = self.execution_history[-1000:]

    # ========================================================================
    # RETRY AND ESCALATION LOGIC
    # ========================================================================

    def should_retry(self, task_id: str) -> bool:
        """Check if a task should be retried"""
        retry_count = self.retry_counts.get(task_id, 0)
        return retry_count < self.max_retries

    def record_retry(self, task_id: str):
        """Record a retry attempt"""
        self.retry_counts[task_id] = self.retry_counts.get(task_id, 0) + 1
        logger.info(f"Retry recorded for task {task_id}: attempt {self.retry_counts[task_id]}")

    def should_escalate(self, task_id: str, quality_score: float) -> bool:
        """
        Determine if a task should be escalated to human review.

        Escalation happens when:
        1. Max retries exceeded
        2. Quality score below threshold
        3. Loop detected
        """
        retry_count = self.retry_counts.get(task_id, 0)

        if retry_count >= self.max_retries:
            logger.warning(f"Escalation triggered: max retries exceeded for task {task_id}")
            return True

        if quality_score < self.escalation_threshold:
            logger.warning(
                f"Escalation triggered: quality score {quality_score:.2f} "
                f"below threshold {self.escalation_threshold}"
            )
            return True

        return False

    def reset_retry_count(self, task_id: str):
        """Reset retry count for a task (call after success)"""
        if task_id in self.retry_counts:
            del self.retry_counts[task_id]

    # ========================================================================
    # HELPER METHODS
    # ========================================================================

    def _hash_data(self, data: Any) -> str:
        """Create hash of data for similarity comparison"""
        import hashlib

        if data is None:
            return "none"

        # Convert to string representation
        if isinstance(data, dict):
            # Sort keys for consistent hashing
            data_str = str(sorted(data.items()))
        elif isinstance(data, list):
            data_str = str(data)
        else:
            data_str = str(data)

        # Create hash
        return hashlib.md5(data_str.encode()).hexdigest()

    def _calculate_similarity(self, hash1: str, hash2: str) -> float:
        """
        Calculate similarity between two hashes.

        Simple implementation: exact match = 1.0, different = 0.0
        Could be enhanced with fuzzy matching.
        """
        return 1.0 if hash1 == hash2 else 0.0

    def _create_evaluation_result(
        self,
        status: ExecutionStatus,
        passed: bool,
        issues: List[str],
        should_retry: bool,
        should_escalate: bool,
        metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Create standardized evaluation result"""
        return {
            "status": status,
            "passed": passed,
            "issues": issues,
            "should_retry": should_retry,
            "should_escalate": should_escalate,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat()
        }

    # ========================================================================
    # STATISTICS AND MONITORING
    # ========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get evaluation statistics"""
        total = self.stats["total_evaluations"]

        return {
            **self.stats,
            "success_rate": self.stats["successes"] / total if total > 0 else 0,
            "failure_rate": self.stats["failures"] / total if total > 0 else 0,
            "loop_rate": self.stats["loops_detected"] / total if total > 0 else 0,
            "escalation_rate": self.stats["escalations"] / total if total > 0 else 0,
            "active_retries": len(self.retry_counts)
        }

    def get_execution_history(self, agent_name: Optional[str] = None, limit: int = 100) -> List[Dict]:
        """Get execution history for monitoring"""
        history = self.execution_history

        if agent_name:
            history = [r for r in history if r.agent_name == agent_name]

        return [
            {
                "agent_name": r.agent_name,
                "task_id": r.task_id,
                "timestamp": r.timestamp.isoformat(),
                "execution_time": r.execution_time,
                "status": r.status
            }
            for r in history[-limit:]
        ]

    def clear_history(self):
        """Clear execution history (for testing or cleanup)"""
        self.execution_history = []
        self.retry_counts = {}
        logger.info("Execution history cleared")


# ========================================================================
# PRE-DEFINED SUCCESS CRITERIA FOR COMMON TASKS
# ========================================================================

class StandardCriteria:
    """Pre-defined success criteria for common tasks"""

    @staticmethod
    def data_collection() -> SuccessCriteria:
        """Criteria for data collection tasks"""
        return SuccessCriteria(
            min_output_length=100,
            required_fields=["papers_collected", "sources"],
            max_execution_time=180.0,
            expected_output_type=dict
        )

    @staticmethod
    def graph_processing() -> SuccessCriteria:
        """Criteria for graph processing tasks"""
        return SuccessCriteria(
            required_fields=["nodes", "edges"],
            max_execution_time=120.0,
            expected_output_type=dict
        )

    @staticmethod
    def vector_processing() -> SuccessCriteria:
        """Criteria for vector processing tasks"""
        return SuccessCriteria(
            required_fields=["embeddings_added"],
            max_execution_time=60.0,
            expected_output_type=dict
        )

    @staticmethod
    def reasoning() -> SuccessCriteria:
        """Criteria for reasoning tasks"""
        return SuccessCriteria(
            min_output_length=100,
            max_output_length=10000,
            required_fields=["answer"],
            forbidden_patterns=["I don't know", "cannot answer", "error"],
            quality_threshold=0.7,
            max_execution_time=30.0,
            expected_output_type=dict
        )

    @staticmethod
    def llamaindex_indexing() -> SuccessCriteria:
        """Criteria for LlamaIndex indexing tasks"""
        return SuccessCriteria(
            required_fields=["documents_indexed"],
            max_execution_time=90.0,
            expected_output_type=dict
        )
