#!/usr/bin/env python3
"""
ResearcherAI - Production Multi-Agent RAG System
===============================================

Containerized multi-agent research system with:
- 7 data sources (arXiv, Semantic Scholar, Zenodo, PubMed, Web, HuggingFace, Kaggle)
- Dual database backends (Neo4j/NetworkX for graphs, Qdrant/FAISS for vectors)
- Conversation memory and multi-session support
- Automated data collection scheduler
- Interactive CLI interface
"""

import os
import sys
import yaml
import logging
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("./volumes/logs/rag_system.log")
    ]
)

logger = logging.getLogger(__name__)


def load_config():
    """Load configuration from YAML file"""
    config_path = Path("./config/settings.yaml")

    if not config_path.exists():
        logger.warning(f"Config file not found at {config_path}, using defaults")
        return {}

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Substitute environment variables
    config = _substitute_env_vars(config)

    return config


def _substitute_env_vars(config):
    """Recursively substitute ${VAR} with environment variables"""
    if isinstance(config, dict):
        return {k: _substitute_env_vars(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [_substitute_env_vars(item) for item in config]
    elif isinstance(config, str) and config.startswith("${") and config.endswith("}"):
        var_name = config[2:-1]
        return os.getenv(var_name, config)
    else:
        return config


def print_banner():
    """Print startup banner"""
    print("\n" + "=" * 70)
    print("🤖 ResearcherAI - Multi-Agent RAG System v2.0")
    print("=" * 70)
    print()


def print_help():
    """Print available commands"""
    print("""
Available Commands:
===================

Research:
  <question>          Ask a research question
  collect [query]     Collect papers from all sources

Session Management:
  sessions            List all research sessions
  switch <name>       Switch to different session
  new <name>          Create new session

Information:
  stats               Show system statistics
  memory              Show conversation history
  graph               Visualize knowledge graph

Scheduler:
  schedule start      Start automated collection
  schedule stop       Stop automated collection
  schedule now        Run collection immediately
  schedule status     Show scheduler status

System:
  save                Save current session
  help                Show this help message
  quit / exit         Exit (auto-saves)
""")


def main():
    """Main entry point"""
    print_banner()

    # Create required directories
    Path("./volumes/sessions").mkdir(parents=True, exist_ok=True)
    Path("./volumes/logs").mkdir(parents=True, exist_ok=True)
    Path("./volumes/cache").mkdir(parents=True, exist_ok=True)

    # Load configuration
    logger.info("Loading configuration...")
    config = load_config()

    # Determine session name
    session_name = sys.argv[1] if len(sys.argv) > 1 else "default"

    # Initialize orchestrator
    logger.info(f"Initializing system with session '{session_name}'...")

    try:
        from agents import OrchestratorAgent, SchedulerAgent

        orchestrator = OrchestratorAgent(session_name, config)

        # Initialize scheduler (optional)
        scheduler = None
        if config.get("agents", {}).get("scheduler", {}).get("enabled", False):
            scheduler = SchedulerAgent(orchestrator, config.get("agents", {}).get("scheduler", {}))
            scheduler.start()

        logger.info("✅ System initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize system: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Print initial stats
    stats = orchestrator.get_stats()
    print(f"\n📊 Current Session: {stats['session']}")
    print(f"   Papers collected: {stats['metadata']['papers_collected']}")
    print(f"   Conversations: {stats['metadata']['conversations']}")
    print(f"   Graph: {stats['graph']['nodes']} nodes, {stats['graph']['edges']} edges ({stats['graph']['backend']})")
    print(f"   Vector DB: {stats['vector']['chunks']} chunks ({stats['vector']['backend']})")

    if scheduler:
        scheduler_stats = scheduler.get_stats()
        print(f"   Scheduler: {'Running' if scheduler_stats['running'] else 'Stopped'}")

    print("\nType 'help' for available commands, or ask a research question.")

    # Interactive loop
    try:
        while True:
            print()
            user_input = input("You: ").strip()

            if not user_input:
                continue

            # Parse command
            parts = user_input.lower().split()
            command = parts[0] if parts else ""

            # Handle commands
            if command in ["exit", "quit", "q"]:
                orchestrator.close()
                if scheduler:
                    scheduler.stop()
                print("\n👋 Goodbye!")
                break

            elif command == "help":
                print_help()

            elif command == "sessions":
                sessions = orchestrator.list_sessions()
                if sessions:
                    print(f"\n📂 Available Sessions ({len(sessions)}):\n")
                    for s in sessions:
                        marker = "→" if s["name"] == orchestrator.session_name else " "
                        print(f"{marker} {s['name']}")
                        print(f"   Papers: {s['papers_collected']}, Conversations: {s['conversations']}")
                        print(f"   Last modified: {s['last_modified']}")
                        print()
                else:
                    print("\nNo sessions found.")

            elif command == "switch" and len(parts) > 1:
                new_session = " ".join(parts[1:])
                orchestrator.switch_session(new_session)
                print(f"\n✅ Switched to session: {new_session}")

            elif command == "new" and len(parts) > 1:
                new_session = " ".join(parts[1:])
                orchestrator.switch_session(new_session)
                print(f"\n✅ Created new session: {new_session}")

            elif command == "collect":
                query = " ".join(parts[1:]) if len(parts) > 1 else "machine learning AI"
                print(f"\n📡 Collecting papers for: '{query}'...")
                result = orchestrator.collect_data(query, max_per_source=10)
                print(f"\n✅ Collected {result['papers_collected']} papers")
                print(f"   Graph: +{result['graph_stats']['nodes_added']} nodes, +{result['graph_stats']['edges_added']} edges")
                print(f"   Vector DB: +{result['vector_stats']['chunks_added']} chunks")

            elif command == "stats":
                stats = orchestrator.get_stats()
                print(f"\n📊 System Statistics:")
                print(f"\nSession: {stats['session']}")
                print(f"  Created: {stats['metadata']['created']}")
                print(f"  Papers collected: {stats['metadata']['papers_collected']}")
                print(f"  Conversations: {stats['metadata']['conversations']}")
                print(f"\nKnowledge Graph ({stats['graph']['backend']}):")
                print(f"  Nodes: {stats['graph']['nodes']}")
                print(f"  Edges: {stats['graph']['edges']}")
                print(f"\nVector Database ({stats['vector']['backend']}):")
                print(f"  Chunks: {stats['vector']['chunks']}")
                print(f"  Dimension: {stats['vector']['dimension']}")
                print(f"\nReasoning Agent:")
                print(f"  Conversation turns: {stats['reasoning']['conversation_turns']}")
                print(f"  Memory limit: {stats['reasoning']['memory_limit']} turns")

                if scheduler:
                    scheduler_stats = scheduler.get_stats()
                    print(f"\nScheduler:")
                    print(f"  Status: {'Running' if scheduler_stats['running'] else 'Stopped'}")
                    print(f"  Total runs: {scheduler_stats['total_runs']}")
                    print(f"  Papers collected: {scheduler_stats['total_papers_collected']}")
                    if scheduler_stats['last_run']:
                        print(f"  Last run: {scheduler_stats['last_run']}")

            elif command == "memory":
                history = orchestrator.reasoning_agent.get_history()
                if history:
                    print(f"\n💾 Conversation History ({len(history)} turns):\n")
                    for i, turn in enumerate(history, 1):
                        print(f"{i}. Q: {turn['query']}")
                        print(f"   A: {turn['answer'][:150]}...")
                        print()
                else:
                    print("\nNo conversation history yet.")

            elif command == "graph":
                filename = f"graph_{orchestrator.session_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
                orchestrator.visualize_graph(filename)
                print(f"\n✅ Graph visualization saved to {filename}")

            elif command == "save":
                orchestrator.save_session()
                print(f"\n✅ Session '{orchestrator.session_name}' saved")

            elif command == "schedule":
                if not scheduler:
                    print("\n❌ Scheduler not enabled in config")
                    continue

                if len(parts) < 2:
                    print("\nUsage: schedule [start|stop|now|status]")
                    continue

                subcmd = parts[1]

                if subcmd == "start":
                    scheduler.start()
                    print("\n✅ Scheduler started")

                elif subcmd == "stop":
                    scheduler.stop()
                    print("\n✅ Scheduler stopped")

                elif subcmd == "now":
                    query = " ".join(parts[2:]) if len(parts) > 2 else None
                    print(f"\n📡 Running collection now...")
                    result = scheduler.run_now(query)
                    print(f"✅ Collected {result.get('papers_collected', 0)} papers")

                elif subcmd == "status":
                    scheduler_stats = scheduler.get_stats()
                    print(f"\n📅 Scheduler Status:")
                    print(f"  Running: {scheduler_stats['running']}")
                    print(f"  Schedule: {scheduler_stats['schedule']}")
                    print(f"  Total runs: {scheduler_stats['total_runs']}")
                    print(f"  Successful: {scheduler_stats['successful_runs']}")
                    print(f"  Failed: {scheduler_stats['failed_runs']}")
                    if scheduler_stats['last_run']:
                        print(f"  Last run: {scheduler_stats['last_run']}")
                    if scheduler_stats['next_run']:
                        print(f"  Next run: {scheduler_stats['next_run']}")

            else:
                # Treat as research question
                print("\n🧠 Thinking...")
                answer = orchestrator.ask(user_input)
                print(f"\n🤖 Answer:\n{answer}")

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        orchestrator.close()
        if scheduler:
            scheduler.stop()
        print("👋 Goodbye!")

    except Exception as e:
        logger.error(f"Error in main loop: {e}")
        import traceback
        traceback.print_exc()
        orchestrator.close()
        if scheduler:
            scheduler.stop()
        sys.exit(1)


if __name__ == "__main__":
    main()
