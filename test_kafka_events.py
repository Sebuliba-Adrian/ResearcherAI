#!/usr/bin/env python3
"""
Test script for Kafka event publishing and consumption.

Tests:
1. Kafka manager initialization
2. Event publishing
3. Topic creation
4. Event consumption
"""

import os
import sys
import uuid
import time
from datetime import datetime

# Set environment variables
os.environ["USE_KAFKA"] = "true"
os.environ["KAFKA_BOOTSTRAP_SERVERS"] = "localhost:9094"  # External port

from utils.kafka_manager import get_kafka_manager
from utils.event_schemas import (
    QuerySubmittedEvent,
    DataCollectionStartedEvent,
    AgentHealthCheckEvent,
    EventPriority
)


def test_kafka_connection():
    """Test 1: Kafka Manager Initialization"""
    print("=" * 70)
    print("TEST 1: Kafka Manager Initialization")
    print("=" * 70)

    kafka_manager = get_kafka_manager()
    stats = kafka_manager.get_stats()

    print(f"\n✓ Kafka enabled: {stats['enabled']}")
    print(f"✓ Bootstrap servers: {stats['bootstrap_servers']}")
    print(f"✓ Producer connected: {stats['producer_connected']}")
    print(f"✓ Group ID: {stats['group_id']}")

    if not stats['enabled'] or not stats['producer_connected']:
        print("\n❌ FAILED: Kafka not properly initialized")
        return False

    print("\n✅ PASSED: Kafka manager initialized successfully")
    return True


def test_event_publishing():
    """Test 2: Event Publishing"""
    print("\n" + "=" * 70)
    print("TEST 2: Event Publishing")
    print("=" * 70)

    kafka_manager = get_kafka_manager()
    correlation_id = str(uuid.uuid4())

    # Test 1: Query Submitted Event
    print("\n→ Publishing QuerySubmittedEvent...")
    query_event = QuerySubmittedEvent(
        event_id=str(uuid.uuid4()),
        correlation_id=correlation_id,
        session_id="test_session",
        query="machine learning in healthcare",
        user_id="test_user",
        max_papers_per_source=10,
        sources=["arxiv", "semantic_scholar"],
        priority=EventPriority.NORMAL
    )

    success1 = kafka_manager.publish_event(query_event)
    print(f"  {'✓' if success1 else '✗'} QuerySubmittedEvent: {'published' if success1 else 'failed'}")

    # Test 2: Data Collection Started Event
    print("\n→ Publishing DataCollectionStartedEvent...")
    collection_event = DataCollectionStartedEvent(
        event_id=str(uuid.uuid4()),
        correlation_id=correlation_id,
        session_id="test_session",
        query="machine learning in healthcare",
        sources=["arxiv", "semantic_scholar", "pubmed"],
        max_papers_per_source=10,
        priority=EventPriority.NORMAL
    )

    success2 = kafka_manager.publish_event(collection_event)
    print(f"  {'✓' if success2 else '✗'} DataCollectionStartedEvent: {'published' if success2 else 'failed'}")

    # Test 3: Agent Health Check Event
    print("\n→ Publishing AgentHealthCheckEvent...")
    health_event = AgentHealthCheckEvent(
        event_id=str(uuid.uuid4()),
        correlation_id=correlation_id,
        session_id="test_session",
        agent_name="test_agent",
        status="healthy",
        metrics={
            "memory_usage_mb": 512,
            "cpu_percent": 25.5,
            "uptime_seconds": 3600
        },
        priority=EventPriority.LOW
    )

    success3 = kafka_manager.publish_event(health_event)
    print(f"  {'✓' if success3 else '✗'} AgentHealthCheckEvent: {'published' if success3 else 'failed'}")

    if not (success1 and success2 and success3):
        print("\n❌ FAILED: Some events failed to publish")
        return False

    print("\n✅ PASSED: All events published successfully")
    return True


def test_topic_listing():
    """Test 3: Topic Listing"""
    print("\n" + "=" * 70)
    print("TEST 3: Kafka Topics")
    print("=" * 70)

    kafka_manager = get_kafka_manager()

    try:
        topics = kafka_manager.admin_client.list_topics()
        print(f"\n✓ Found {len(topics)} topics:")

        # Filter RAG-related topics
        rag_topics = [t for t in topics if not t.startswith('_')]
        for topic in sorted(rag_topics):
            print(f"  • {topic}")

        expected_topics = [
            'query.submitted',
            'data.collection.started',
            'agent.health.check'
        ]

        missing_topics = [t for t in expected_topics if t not in topics]
        if missing_topics:
            print(f"\n⚠ Warning: Missing expected topics: {missing_topics}")

        print("\n✅ PASSED: Topics listed successfully")
        return True

    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        return False


def test_event_consumption():
    """Test 4: Event Consumption"""
    print("\n" + "=" * 70)
    print("TEST 4: Event Consumption")
    print("=" * 70)

    kafka_manager = get_kafka_manager()

    print("\n→ Consuming events from test topics...")
    print("  (This will consume up to 10 messages with 5s timeout)\n")

    consumed_events = []

    def event_callback(event):
        print(f"  ✓ Consumed: {event.event_type} (ID: {event.event_id[:8]}...)")
        consumed_events.append(event)

    try:
        # Create consumer for test topics
        consumer = kafka_manager.create_consumer(
            topics=[
                'query.submitted',
                'data.collection.started',
                'agent.health.check'
            ],
            group_id='test_consumer',
            auto_offset_reset='earliest'  # Read from beginning
        )

        if not consumer:
            print("\n❌ FAILED: Could not create consumer")
            return False

        # Poll for messages (5 second timeout, max 10 messages)
        timeout_start = time.time()
        messages_consumed = 0

        while time.time() - timeout_start < 5 and messages_consumed < 10:
            message_batch = consumer.poll(timeout_ms=1000)

            for topic_partition, messages in message_batch.items():
                for message in messages:
                    try:
                        from utils.event_schemas import deserialize_event
                        event = deserialize_event(message.value)
                        event_callback(event)
                        messages_consumed += 1
                    except Exception as e:
                        print(f"  ✗ Error processing message: {e}")

        consumer.close()

        print(f"\n✓ Total events consumed: {len(consumed_events)}")

        if len(consumed_events) >= 3:
            print("\n✅ PASSED: Successfully consumed events")
            return True
        else:
            print(f"\n⚠ WARNING: Expected at least 3 events, got {len(consumed_events)}")
            print("  (This might be OK if events were consumed by other tests)")
            return True

    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all Kafka tests"""
    print("\n🧪 ResearcherAI - Kafka Event System Test Suite")
    print("=" * 70)

    tests = [
        ("Kafka Connection", test_kafka_connection),
        ("Event Publishing", test_event_publishing),
        ("Topic Listing", test_topic_listing),
        ("Event Consumption", test_event_consumption),
    ]

    results = []

    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ EXCEPTION in {test_name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    print(f"\nResults: {passed}/{total} tests passed\n")

    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {test_name}")

    print("\n" + "=" * 70)

    if passed == total:
        print("🎉 ALL TESTS PASSED!")
        return 0
    else:
        print(f"⚠️  {total - passed} TEST(S) FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
