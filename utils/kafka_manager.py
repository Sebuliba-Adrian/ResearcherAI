"""
Kafka Event Manager for event-driven agent communication.

Provides high-level API for publishing and consuming events across agents.
"""

import os
import json
import uuid
import logging
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime
from kafka import KafkaProducer, KafkaConsumer
from kafka.admin import KafkaAdminClient, NewTopic
from kafka.errors import KafkaError, NoBrokersAvailable
import time

from utils.event_schemas import BaseEvent, EventType, deserialize_event

logger = logging.getLogger(__name__)


class KafkaEventManager:
    """
    Manages Kafka event production and consumption for agent communication.

    Features:
    - Automatic topic creation
    - Event serialization/deserialization
    - Retries and error handling
    - Graceful degradation (falls back to direct calls if Kafka unavailable)
    """

    def __init__(
        self,
        bootstrap_servers: Optional[str] = None,
        group_id: Optional[str] = None,
        auto_create_topics: bool = True,
        max_retries: int = 3,
        retry_backoff_ms: int = 1000
    ):
        """
        Initialize Kafka event manager.

        Args:
            bootstrap_servers: Kafka bootstrap servers (defaults to env var)
            group_id: Consumer group ID
            auto_create_topics: Whether to auto-create topics
            max_retries: Maximum retry attempts for operations
            retry_backoff_ms: Backoff between retries in milliseconds
        """
        self.bootstrap_servers = bootstrap_servers or os.getenv(
            "KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"
        )
        self.group_id = group_id or "rag-agents"
        self.auto_create_topics = auto_create_topics
        self.max_retries = max_retries
        self.retry_backoff_ms = retry_backoff_ms

        # Check if Kafka is enabled
        self.enabled = os.getenv("USE_KAFKA", "true").lower() == "true"

        # Initialize producer and admin client
        self.producer: Optional[KafkaProducer] = None
        self.admin_client: Optional[KafkaAdminClient] = None
        self.consumers: Dict[str, KafkaConsumer] = {}

        if self.enabled:
            self._initialize_kafka()
        else:
            logger.info("Kafka is disabled, event-driven features will not be available")

    def _initialize_kafka(self):
        """Initialize Kafka producer and admin client"""
        try:
            # Initialize producer
            self.producer = KafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8') if k else None,
                acks='all',  # Wait for all replicas
                retries=self.max_retries,
                max_in_flight_requests_per_connection=1,  # Preserve order
                compression_type='gzip'
            )

            # Initialize admin client
            self.admin_client = KafkaAdminClient(
                bootstrap_servers=self.bootstrap_servers,
                client_id='rag-admin'
            )

            logger.info(f"Kafka connected to {self.bootstrap_servers}")

            # Create default topics
            if self.auto_create_topics:
                self._create_default_topics()

        except NoBrokersAvailable:
            logger.warning(
                f"Kafka brokers not available at {self.bootstrap_servers}. "
                "Running in degraded mode without event streaming."
            )
            self.enabled = False
            self.producer = None
            self.admin_client = None
        except Exception as e:
            logger.error(f"Failed to initialize Kafka: {e}")
            self.enabled = False
            self.producer = None
            self.admin_client = None

    def _create_default_topics(self):
        """Create default topics for agent communication"""
        topics = [
            # Query topics
            NewTopic(name="query.submitted", num_partitions=3, replication_factor=1),
            NewTopic(name="query.validated", num_partitions=3, replication_factor=1),

            # Data collection topics
            NewTopic(name="data.collection.started", num_partitions=3, replication_factor=1),
            NewTopic(name="data.collection.completed", num_partitions=3, replication_factor=1),
            NewTopic(name="data.collection.failed", num_partitions=3, replication_factor=1),

            # Graph processing topics
            NewTopic(name="graph.processing.started", num_partitions=3, replication_factor=1),
            NewTopic(name="graph.processing.completed", num_partitions=3, replication_factor=1),
            NewTopic(name="graph.processing.failed", num_partitions=3, replication_factor=1),

            # Vector processing topics
            NewTopic(name="vector.processing.started", num_partitions=3, replication_factor=1),
            NewTopic(name="vector.processing.completed", num_partitions=3, replication_factor=1),
            NewTopic(name="vector.processing.failed", num_partitions=3, replication_factor=1),

            # Reasoning topics
            NewTopic(name="reasoning.started", num_partitions=3, replication_factor=1),
            NewTopic(name="reasoning.completed", num_partitions=3, replication_factor=1),
            NewTopic(name="reasoning.failed", num_partitions=3, replication_factor=1),

            # System topics
            NewTopic(name="agent.health.check", num_partitions=1, replication_factor=1),
            NewTopic(name="agent.error", num_partitions=3, replication_factor=1),
        ]

        try:
            existing_topics = self.admin_client.list_topics()
            topics_to_create = [t for t in topics if t.name not in existing_topics]

            if topics_to_create:
                self.admin_client.create_topics(new_topics=topics_to_create, validate_only=False)
                logger.info(f"Created {len(topics_to_create)} Kafka topics")
        except Exception as e:
            logger.warning(f"Failed to create topics: {e}")

    def publish_event(
        self,
        event: BaseEvent,
        key: Optional[str] = None
    ) -> bool:
        """
        Publish an event to Kafka.

        Args:
            event: Event to publish
            key: Optional partition key (defaults to correlation_id)

        Returns:
            True if published successfully, False otherwise
        """
        if not self.enabled or not self.producer:
            logger.debug(f"Kafka disabled, skipping event: {event.event_type}")
            return False

        try:
            # Use correlation_id as key for ordering
            partition_key = key or event.correlation_id

            # Convert event to dict
            event_data = event.dict()

            # Determine topic from event type
            topic = event.event_type.value

            # Send to Kafka
            future = self.producer.send(
                topic=topic,
                key=partition_key,
                value=event_data
            )

            # Wait for send to complete (with timeout)
            record_metadata = future.get(timeout=10)

            logger.debug(
                f"Published event {event.event_id} to {topic} "
                f"(partition {record_metadata.partition}, offset {record_metadata.offset})"
            )

            return True

        except KafkaError as e:
            logger.error(f"Kafka error publishing event {event.event_id}: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to publish event {event.event_id}: {e}")
            return False

    def create_consumer(
        self,
        topics: List[str],
        group_id: Optional[str] = None,
        auto_offset_reset: str = 'latest'
    ) -> Optional[KafkaConsumer]:
        """
        Create a Kafka consumer for specified topics.

        Args:
            topics: List of topic names to subscribe to
            group_id: Consumer group ID (defaults to instance group_id)
            auto_offset_reset: Where to start reading ('earliest' or 'latest')

        Returns:
            KafkaConsumer instance or None if Kafka disabled
        """
        if not self.enabled:
            logger.debug("Kafka disabled, cannot create consumer")
            return None

        try:
            consumer = KafkaConsumer(
                *topics,
                bootstrap_servers=self.bootstrap_servers,
                group_id=group_id or self.group_id,
                auto_offset_reset=auto_offset_reset,
                enable_auto_commit=True,
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                key_deserializer=lambda k: k.decode('utf-8') if k else None,
                max_poll_records=10,
                session_timeout_ms=30000,
                heartbeat_interval_ms=10000
            )

            consumer_id = f"{group_id or self.group_id}_{uuid.uuid4().hex[:8]}"
            self.consumers[consumer_id] = consumer

            logger.info(f"Created consumer {consumer_id} for topics: {topics}")

            return consumer

        except Exception as e:
            logger.error(f"Failed to create consumer: {e}")
            return None

    def consume_events(
        self,
        topics: List[str],
        callback: Callable[[BaseEvent], None],
        group_id: Optional[str] = None,
        max_messages: Optional[int] = None,
        timeout_ms: int = 1000
    ):
        """
        Consume events from topics and process them with callback.

        Args:
            topics: List of topic names to consume from
            callback: Function to process each event
            group_id: Consumer group ID
            max_messages: Maximum messages to consume (None = infinite)
            timeout_ms: Poll timeout in milliseconds
        """
        if not self.enabled:
            logger.debug("Kafka disabled, cannot consume events")
            return

        consumer = self.create_consumer(topics, group_id)
        if not consumer:
            return

        try:
            messages_consumed = 0
            logger.info(f"Starting to consume from topics: {topics}")

            while True:
                # Check if we've consumed enough messages
                if max_messages and messages_consumed >= max_messages:
                    break

                # Poll for messages
                message_batch = consumer.poll(timeout_ms=timeout_ms)

                for topic_partition, messages in message_batch.items():
                    for message in messages:
                        try:
                            # Deserialize event
                            event = deserialize_event(message.value)

                            # Process event with callback
                            callback(event)

                            messages_consumed += 1

                        except Exception as e:
                            logger.error(
                                f"Error processing message from {topic_partition.topic}: {e}"
                            )

        except KeyboardInterrupt:
            logger.info("Consumer interrupted by user")
        finally:
            consumer.close()
            logger.info(f"Consumer closed. Consumed {messages_consumed} messages.")

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about Kafka connection"""
        return {
            "enabled": self.enabled,
            "bootstrap_servers": self.bootstrap_servers,
            "producer_connected": self.producer is not None,
            "active_consumers": len(self.consumers),
            "group_id": self.group_id
        }

    def close(self):
        """Close all Kafka connections"""
        if self.producer:
            self.producer.flush()
            self.producer.close()
            logger.info("Kafka producer closed")

        for consumer_id, consumer in self.consumers.items():
            consumer.close()
            logger.info(f"Kafka consumer {consumer_id} closed")

        self.consumers.clear()


# ============================================================================
# Global instance (singleton pattern)
# ============================================================================

_kafka_manager: Optional[KafkaEventManager] = None


def get_kafka_manager() -> KafkaEventManager:
    """
    Get or create global Kafka event manager instance.

    Returns:
        KafkaEventManager instance
    """
    global _kafka_manager
    if _kafka_manager is None:
        _kafka_manager = KafkaEventManager()
    return _kafka_manager


def close_kafka_manager():
    """Close global Kafka event manager"""
    global _kafka_manager
    if _kafka_manager:
        _kafka_manager.close()
        _kafka_manager = None
