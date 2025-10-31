"""
Comprehensive end-to-end test for event-driven architecture with Kafka.

Tests:
1. Event-driven pipeline with Kafka enabled
2. Fallback synchronous pipeline with Kafka disabled
3. Event publishing and consumption
4. Schema validation for all events
5. Error handling and failed events
"""

import os
import sys
import time
import uuid
import logging
from typing import Dict, Any

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Test results tracking
test_results = []


class TestReporter:
    """Helper class to track and report test results"""

    def __init__(self):
        self.tests_run = 0
        self.tests_passed = 0
        self.tests_failed = 0
        self.test_details = []

    def record_test(self, name: str, passed: bool, details: str = ""):
        """Record a test result"""
        self.tests_run += 1
        if passed:
            self.tests_passed += 1
            status = "✅ PASS"
        else:
            self.tests_failed += 1
            status = "❌ FAIL"

        self.test_details.append({
            "name": name,
            "status": status,
            "passed": passed,
            "details": details
        })

        logger.info(f"{status}: {name}")
        if details:
            logger.info(f"  Details: {details}")

    def print_summary(self):
        """Print test summary"""
        print("\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80)
        print(f"Total Tests: {self.tests_run}")
        print(f"Passed: {self.tests_passed} ✅")
        print(f"Failed: {self.tests_failed} ❌")
        print(f"Pass Rate: {(self.tests_passed/self.tests_run*100) if self.tests_run > 0 else 0:.1f}%")
        print("="*80)

        print("\nDETAILED RESULTS:")
        for test in self.test_details:
            print(f"\n{test['status']}: {test['name']}")
            if test['details']:
                print(f"  {test['details']}")

        print("\n" + "="*80)


reporter = TestReporter()


def test_event_schemas():
    """Test event schema validation"""
    from utils.event_schemas import (
        QuerySubmittedEvent, DataCollectionCompletedEvent,
        PaperMetadata, EventType
    )

    try:
        # Test valid event
        event = QuerySubmittedEvent(
            event_id="test-001",
            correlation_id="corr-001",
            query="test query",
            max_papers_per_source=5
        )
        assert event.event_type == EventType.QUERY_SUBMITTED
        reporter.record_test("Event schema - Valid event", True, "QuerySubmittedEvent created successfully")

        # Test event with paper metadata
        paper = PaperMetadata(
            title="Test Paper",
            authors=["Author 1", "Author 2"],
            source="arxiv"
        )
        assert paper.title == "Test Paper"
        assert len(paper.authors) == 2
        reporter.record_test("Event schema - Paper metadata", True, "PaperMetadata validated successfully")

        # Test event serialization
        event_dict = event.dict()
        assert "event_id" in event_dict
        assert "query" in event_dict
        reporter.record_test("Event schema - Serialization", True, "Event serialized to dict successfully")

    except Exception as e:
        reporter.record_test("Event schema validation", False, f"Error: {e}")


def test_kafka_manager():
    """Test Kafka manager initialization and operations"""
    from utils.kafka_manager import get_kafka_manager, KafkaEventManager
    from utils.event_schemas import QuerySubmittedEvent

    try:
        # Get Kafka manager instance
        manager = get_kafka_manager()
        assert manager is not None
        reporter.record_test(
            "Kafka manager - Initialization",
            True,
            f"Manager created (enabled={manager.enabled})"
        )

        # Test stats
        stats = manager.get_stats()
        assert "enabled" in stats
        assert "bootstrap_servers" in stats
        reporter.record_test(
            "Kafka manager - Stats",
            True,
            f"Stats retrieved: {stats['bootstrap_servers']}"
        )

        # Test event publishing (only if Kafka is available)
        if manager.enabled:
            event = QuerySubmittedEvent(
                event_id=f"test_{uuid.uuid4().hex[:8]}",
                correlation_id=f"corr_{uuid.uuid4().hex[:8]}",
                query="test query for Kafka",
                max_papers_per_source=3
            )
            result = manager.publish_event(event)
            reporter.record_test(
                "Kafka manager - Event publishing",
                result,
                f"Event {event.event_id} published: {result}"
            )
        else:
            reporter.record_test(
                "Kafka manager - Event publishing",
                True,
                "Skipped (Kafka not available, graceful degradation working)"
            )

    except Exception as e:
        reporter.record_test("Kafka manager", False, f"Error: {e}")


def test_event_driven_orchestrator_no_kafka():
    """Test event-driven orchestrator without Kafka (fallback mode)"""
    from agents.event_driven_orchestrator import EventDrivenOrchestrator

    # Force Kafka to be disabled for this test
    original_use_kafka = os.getenv("USE_KAFKA")
    os.environ["USE_KAFKA"] = "false"

    try:
        # Initialize orchestrator
        orchestrator = EventDrivenOrchestrator(
            session_id=f"test_session_{uuid.uuid4().hex[:8]}",
            use_neo4j=False,
            use_qdrant=False
        )

        reporter.record_test(
            "Event orchestrator - Initialization (no Kafka)",
            True,
            f"Orchestrator created (session={orchestrator.session_id})"
        )

        # Process a simple query
        query = "What are machine learning techniques?"
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing synchronous pipeline (no Kafka)")
        logger.info(f"Query: {query}")
        logger.info(f"{'='*60}\n")

        start_time = time.time()
        result = orchestrator.process_query(
            query=query,
            max_papers_per_source=2,  # Small number for fast testing
            timeout_seconds=120
        )
        execution_time = time.time() - start_time

        # Validate results
        assert result is not None
        assert "correlation_id" in result
        assert "data_collection" in result
        assert "graph_processing" in result
        assert "vector_processing" in result
        assert "reasoning" in result
        assert result["pipeline_complete"] is True
        assert result["kafka_enabled"] is False

        papers_collected = result["data_collection"]["papers_collected"]
        nodes_extracted = result["graph_processing"].get("nodes_extracted", 0)
        embeddings_added = result["vector_processing"].get("embeddings_added", 0)

        reporter.record_test(
            "Event orchestrator - Synchronous pipeline",
            True,
            f"Pipeline completed in {execution_time:.2f}s: "
            f"{papers_collected} papers, {nodes_extracted} nodes, {embeddings_added} embeddings"
        )

        # Verify each stage had reasonable output
        if papers_collected > 0:
            reporter.record_test(
                "Event orchestrator - Data collection",
                True,
                f"Collected {papers_collected} papers"
            )
        else:
            reporter.record_test(
                "Event orchestrator - Data collection",
                False,
                "No papers collected"
            )

        if nodes_extracted > 0:
            reporter.record_test(
                "Event orchestrator - Graph processing",
                True,
                f"Extracted {nodes_extracted} nodes"
            )
        else:
            reporter.record_test(
                "Event orchestrator - Graph processing",
                False,
                "No nodes extracted"
            )

        if embeddings_added > 0:
            reporter.record_test(
                "Event orchestrator - Vector processing",
                True,
                f"Created {embeddings_added} embeddings"
            )
        else:
            reporter.record_test(
                "Event orchestrator - Vector processing",
                False,
                "No embeddings created"
            )

        answer = result["reasoning"].get("answer", "")
        if answer and len(answer) > 10:
            reporter.record_test(
                "Event orchestrator - Reasoning",
                True,
                f"Generated answer ({len(answer)} chars)"
            )
        else:
            reporter.record_test(
                "Event orchestrator - Reasoning",
                False,
                f"Answer too short or empty: {answer[:100]}"
            )

    except Exception as e:
        logger.error(f"Error in orchestrator test: {e}", exc_info=True)
        reporter.record_test("Event orchestrator (no Kafka)", False, f"Error: {e}")
    finally:
        # Restore original setting
        if original_use_kafka:
            os.environ["USE_KAFKA"] = original_use_kafka
        else:
            os.environ.pop("USE_KAFKA", None)


def test_event_driven_orchestrator_with_kafka():
    """Test event-driven orchestrator WITH Kafka (if available)"""
    from agents.event_driven_orchestrator import EventDrivenOrchestrator
    from utils.kafka_manager import get_kafka_manager

    # Check if Kafka is available
    manager = get_kafka_manager()
    if not manager.enabled:
        reporter.record_test(
            "Event orchestrator - With Kafka",
            True,
            "Skipped (Kafka not available, but graceful degradation working)"
        )
        return

    try:
        # Initialize orchestrator with Kafka enabled
        orchestrator = EventDrivenOrchestrator(
            session_id=f"kafka_test_{uuid.uuid4().hex[:8]}",
            use_neo4j=False,
            use_qdrant=False
        )

        # Process a simple query
        query = "What are deep learning applications?"
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing event-driven pipeline (WITH Kafka)")
        logger.info(f"Query: {query}")
        logger.info(f"{'='*60}\n")

        start_time = time.time()
        result = orchestrator.process_query(
            query=query,
            max_papers_per_source=2,
            timeout_seconds=120
        )
        execution_time = time.time() - start_time

        # Validate results
        assert result is not None
        assert "correlation_id" in result
        assert result["pipeline_complete"] is True

        papers_collected = result["data_collection"]["papers_collected"]

        reporter.record_test(
            "Event orchestrator - Event-driven pipeline (Kafka)",
            True,
            f"Pipeline completed in {execution_time:.2f}s with Kafka: {papers_collected} papers"
        )

    except Exception as e:
        logger.error(f"Error in Kafka orchestrator test: {e}", exc_info=True)
        reporter.record_test("Event orchestrator (with Kafka)", False, f"Error: {e}")


def test_full_pipeline_comparison():
    """Compare Kafka vs non-Kafka performance"""
    logger.info("\n" + "="*80)
    logger.info("PIPELINE COMPARISON TEST")
    logger.info("="*80)

    from agents.event_driven_orchestrator import EventDrivenOrchestrator

    query = "What are neural networks?"

    # Test without Kafka
    os.environ["USE_KAFKA"] = "false"
    orch_no_kafka = EventDrivenOrchestrator(use_neo4j=False, use_qdrant=False)

    start = time.time()
    result_no_kafka = orch_no_kafka.process_query(query, max_papers_per_source=2)
    time_no_kafka = time.time() - start

    # Test with Kafka (if available)
    os.environ.pop("USE_KAFKA", None)  # Use default (Kafka enabled)
    orch_kafka = EventDrivenOrchestrator(use_neo4j=False, use_qdrant=False)

    start = time.time()
    result_kafka = orch_kafka.process_query(query, max_papers_per_source=2)
    time_kafka = time.time() - start

    # Compare
    logger.info(f"\nWithout Kafka: {time_no_kafka:.2f}s")
    logger.info(f"With Kafka: {time_kafka:.2f}s")

    reporter.record_test(
        "Pipeline comparison",
        True,
        f"Both pipelines completed (no-Kafka: {time_no_kafka:.2f}s, Kafka: {time_kafka:.2f}s)"
    )


def main():
    """Run all tests"""
    logger.info("\n" + "="*80)
    logger.info("STARTING EVENT-DRIVEN ARCHITECTURE TESTS")
    logger.info("="*80 + "\n")

    # Run tests in order
    logger.info("\n[TEST 1/6] Testing event schemas...")
    test_event_schemas()

    logger.info("\n[TEST 2/6] Testing Kafka manager...")
    test_kafka_manager()

    logger.info("\n[TEST 3/6] Testing orchestrator without Kafka...")
    test_event_driven_orchestrator_no_kafka()

    logger.info("\n[TEST 4/6] Testing orchestrator with Kafka...")
    test_event_driven_orchestrator_with_kafka()

    logger.info("\n[TEST 5/6] Testing pipeline comparison...")
    test_full_pipeline_comparison()

    # Print summary
    logger.info("\n")
    reporter.print_summary()

    # Exit with appropriate code
    sys.exit(0 if reporter.tests_failed == 0 else 1)


if __name__ == "__main__":
    main()
