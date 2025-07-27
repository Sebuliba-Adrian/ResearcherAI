"""
SchedulerAgent - Automated Background Data Collection
====================================================

Schedules and runs autonomous data collection tasks
"""

import logging
import time
from datetime import datetime
from typing import Dict, Optional, Callable
from threading import Thread, Event
import schedule

logger = logging.getLogger(__name__)


class SchedulerAgent:
    """Automated background data collection scheduler"""

    def __init__(self, orchestrator, config: Optional[Dict] = None):
        """
        Initialize scheduler

        config = {
            "schedule": "0 */6 * * *",  # Cron-style schedule
            "sources": ["arxiv", "semantic_scholar", ...],
            "default_query": "machine learning AI",
            "max_per_source": 10,
            "enabled": False
        }
        """
        self.orchestrator = orchestrator
        self.config = config or {}

        self.schedule_pattern = self.config.get("schedule", "0 */6 * * *")  # Every 6 hours
        self.sources = self.config.get("sources", ["arxiv", "semantic_scholar"])
        self.default_query = self.config.get("default_query", "machine learning AI")
        self.max_per_source = self.config.get("max_per_source", 10)
        self.enabled = self.config.get("enabled", False)

        self.running = False
        self.stop_event = Event()
        self.scheduler_thread = None

        self.stats = {
            "total_runs": 0,
            "successful_runs": 0,
            "failed_runs": 0,
            "last_run": None,
            "next_run": None,
            "total_papers_collected": 0
        }

        logger.info(f"SchedulerAgent initialized (enabled={self.enabled})")

    def start(self):
        """Start the scheduler in background thread"""
        if self.running:
            logger.warning("Scheduler already running")
            return

        if not self.enabled:
            logger.info("Scheduler is disabled in config")
            return

        self.running = True
        self.stop_event.clear()

        # Setup schedule (simplified - runs every N hours)
        hours = self._parse_schedule()
        schedule.every(hours).hours.do(self._run_collection)

        # Start scheduler thread
        self.scheduler_thread = Thread(target=self._run_scheduler, daemon=True)
        self.scheduler_thread.start()

        logger.info(f"Scheduler started (runs every {hours} hours)")
        self.stats["next_run"] = datetime.now().isoformat()

    def stop(self):
        """Stop the scheduler"""
        if not self.running:
            logger.warning("Scheduler not running")
            return

        self.running = False
        self.stop_event.set()

        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)

        schedule.clear()
        logger.info("Scheduler stopped")

    def _parse_schedule(self) -> int:
        """Parse cron-style schedule to hours (simplified)"""
        # Simplified parser - extracts hours from "0 */N * * *" format
        parts = self.schedule_pattern.split()
        if len(parts) >= 2:
            hour_part = parts[1]
            if "*/" in hour_part:
                return int(hour_part.split("/")[1])

        # Default to 6 hours
        return 6

    def _run_scheduler(self):
        """Scheduler loop"""
        logger.info("Scheduler loop started")

        while self.running and not self.stop_event.is_set():
            schedule.run_pending()
            time.sleep(60)  # Check every minute

        logger.info("Scheduler loop stopped")

    def _run_collection(self):
        """Execute data collection"""
        logger.info("=== AUTOMATED DATA COLLECTION STARTED ===")
        self.stats["total_runs"] += 1
        self.stats["last_run"] = datetime.now().isoformat()

        try:
            # Run collection through orchestrator
            result = self.orchestrator.collect_data(
                query=self.default_query,
                max_per_source=self.max_per_source
            )

            papers_collected = result.get("papers_collected", 0)
            self.stats["successful_runs"] += 1
            self.stats["total_papers_collected"] += papers_collected

            logger.info(f"Automated collection complete: {papers_collected} papers collected")

        except Exception as e:
            self.stats["failed_runs"] += 1
            logger.error(f"Automated collection failed: {e}")

        # Calculate next run
        hours = self._parse_schedule()
        from datetime import timedelta
        next_run_time = datetime.now() + timedelta(hours=hours)
        self.stats["next_run"] = next_run_time.isoformat()

        logger.info("=== AUTOMATED DATA COLLECTION FINISHED ===")

    def run_now(self, query: Optional[str] = None):
        """Manually trigger collection immediately"""
        logger.info("Manual data collection triggered")

        query = query or self.default_query

        try:
            result = self.orchestrator.collect_data(
                query=query,
                max_per_source=self.max_per_source
            )

            papers_collected = result.get("papers_collected", 0)
            self.stats["total_papers_collected"] += papers_collected

            logger.info(f"Manual collection complete: {papers_collected} papers collected")
            return result

        except Exception as e:
            logger.error(f"Manual collection failed: {e}")
            return {"error": str(e), "papers_collected": 0}

    def get_stats(self) -> Dict:
        """Get scheduler statistics"""
        return {
            **self.stats,
            "running": self.running,
            "enabled": self.enabled,
            "schedule": self.schedule_pattern,
            "default_query": self.default_query
        }

    def update_config(self, **kwargs):
        """Update scheduler configuration"""
        for key, value in kwargs.items():
            if key in self.config:
                self.config[key] = value
                setattr(self, key, value)
                logger.info(f"Updated {key} = {value}")

        # Restart if running
        if self.running:
            logger.info("Restarting scheduler with new config...")
            self.stop()
            time.sleep(1)
            self.start()
