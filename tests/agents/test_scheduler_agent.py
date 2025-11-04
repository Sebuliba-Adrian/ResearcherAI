"""
Tests for SchedulerAgent
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from agents.scheduler_agent import SchedulerAgent


class TestSchedulerAgent:
    """Test SchedulerAgent class"""

    def test_initialization(self):
        """Test scheduler initializes with default config"""
        orchestrator = Mock()
        agent = SchedulerAgent(orchestrator)
        
        assert agent.orchestrator == orchestrator
        assert agent.running is False
        assert agent.stats["total_runs"] == 0

    def test_initialization_with_custom_config(self):
        """Test scheduler with custom configuration"""
        orchestrator = Mock()
        config = {
            "schedule": "0 */12 * * *",
            "sources": ["arxiv", "pubmed"],
            "default_query": "deep learning",
            "max_per_source": 20,
            "enabled": True
        }
        
        agent = SchedulerAgent(orchestrator, config)
        
        assert agent.schedule_pattern == "0 */12 * * *"
        assert agent.sources == ["arxiv", "pubmed"]
        assert agent.default_query == "deep learning"
        assert agent.max_per_source == 20
        assert agent.enabled is True

    def test_start_when_disabled(self):
        """Test start does nothing when disabled"""
        orchestrator = Mock()
        config = {"enabled": False}
        agent = SchedulerAgent(orchestrator, config)
        
        agent.start()
        
        assert agent.running is False

    @patch('agents.scheduler_agent.Thread')
    @patch('agents.scheduler_agent.schedule')
    def test_start_when_enabled(self, mock_schedule, mock_thread):
        """Test start begins scheduler thread when enabled"""
        orchestrator = Mock()
        config = {"enabled": True}
        agent = SchedulerAgent(orchestrator, config)
        
        agent.start()
        
        assert agent.running is True
        mock_thread.assert_called_once()

    def test_start_when_already_running(self):
        """Test start does nothing when already running"""
        orchestrator = Mock()
        agent = SchedulerAgent(orchestrator)
        agent.running = True
        
        agent.start()
        
        # Should not change state
        assert agent.running is True

    @patch('agents.scheduler_agent.schedule')
    def test_stop(self, mock_schedule):
        """Test stopping the scheduler"""
        orchestrator = Mock()
        agent = SchedulerAgent(orchestrator)
        agent.running = True
        agent.stop_event = Mock()
        
        agent.stop()
        
        assert agent.running is False
        agent.stop_event.set.assert_called_once()
        mock_schedule.clear.assert_called_once()

    def test_stop_when_not_running(self):
        """Test stop does nothing when not running"""
        orchestrator = Mock()
        agent = SchedulerAgent(orchestrator)
        
        agent.stop()
        
        assert agent.running is False

    def test_parse_schedule_6_hours(self):
        """Test parsing schedule pattern for 6 hours"""
        orchestrator = Mock()
        config = {"schedule": "0 */6 * * *"}
        agent = SchedulerAgent(orchestrator, config)
        
        hours = agent._parse_schedule()
        
        assert hours == 6

    def test_parse_schedule_12_hours(self):
        """Test parsing schedule pattern for 12 hours"""
        orchestrator = Mock()
        config = {"schedule": "0 */12 * * *"}
        agent = SchedulerAgent(orchestrator, config)
        
        hours = agent._parse_schedule()
        
        assert hours == 12

    def test_parse_schedule_invalid_defaults_to_6(self):
        """Test invalid schedule defaults to 6 hours"""
        orchestrator = Mock()
        config = {"schedule": "invalid"}
        agent = SchedulerAgent(orchestrator, config)
        
        hours = agent._parse_schedule()
        
        assert hours == 6

    def test_get_stats(self):
        """Test getting scheduler statistics"""
        orchestrator = Mock()
        agent = SchedulerAgent(orchestrator)
        
        stats = agent.stats
        
        assert "total_runs" in stats
        assert "successful_runs" in stats
        assert "failed_runs" in stats
        assert stats["total_runs"] == 0

    def test_run_collection(self):
        """Test running scheduled collection"""
        orchestrator = Mock()
        orchestrator.collect_data = Mock(return_value={"papers": [{"title": "Test"}]})
        
        agent = SchedulerAgent(orchestrator)
        
        # Manually call the collection method
        agent._run_collection()
        
        orchestrator.collect_data.assert_called_once()
        assert agent.stats["total_runs"] == 1
        assert agent.stats["successful_runs"] == 1

    def test_run_collection_failure(self):
        """Test handling collection failure"""
        orchestrator = Mock()
        orchestrator.collect_data = Mock(side_effect=Exception("API Error"))
        
        agent = SchedulerAgent(orchestrator)
        
        # Manually call the collection method
        agent._run_collection()
        
        assert agent.stats["total_runs"] == 1
        assert agent.stats["failed_runs"] == 1

    def test_run_scheduler_loop(self):
        """Test scheduler loop execution"""
        orchestrator = Mock()
        agent = SchedulerAgent(orchestrator)
        agent.running = False  # Will exit immediately
        
        # This should return immediately since running=False
        agent._run_scheduler()
        
        # Should not crash

    def test_parse_schedule_hourly(self):
        """Test parsing hourly schedule"""
        orchestrator = Mock()
        config = {"schedule": "0 */1 * * *"}
        agent = SchedulerAgent(orchestrator, config)
        
        hours = agent._parse_schedule()
        
        assert hours == 1
