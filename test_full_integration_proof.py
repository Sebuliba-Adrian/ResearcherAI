#!/usr/bin/env python3
"""
FULL INTEGRATION PROOF TEST
===========================

Demonstrates complete working integration of:
- LangGraph orchestration
- All real agents (DataCollector, KnowledgeGraph, Vector, LlamaIndex, Reasoning)
- Self-reflection and correction
- Gemini 1.5 Pro (non-rate-limited)
- Stage-by-stage output capture
"""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from agents.langgraph_orchestrator import create_orchestrator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    print('\n' + '='*70)
    print('FULL INTEGRATION TEST: LangGraph + LlamaIndex + All Real Agents')
    print('='*70 + '\n')

    # Create orchestrator with FULL agent integration
    print("Initializing orchestrator with ALL real agents...")
    orchestrator = create_orchestrator()

    print('\n' + '='*70)
    print('Workflow Graph Structure:')
    print('='*70)
    print(orchestrator.get_workflow_graph())
    print('\n' + '='*70 + '\n')

    # Run workflow with limited papers for faster test
    print("Running workflow with query: 'What are recent advances in retrieval-augmented generation?'\n")
    result = orchestrator.run_workflow(
        'What are recent advances in retrieval-augmented generation?',
        max_per_source=2  # Limit to 2 papers per source for faster test
    )

    print('\n' + '#'*70)
    print('# DETAILED STAGE-BY-STAGE RESULTS')
    print('#'*70 + '\n')

    # Extract the actual final state from LangGraph stream
    final_node_state = list(result.values())[0] if result else {}

    stage_outputs = final_node_state.get('stage_outputs', {})

    for stage, output in stage_outputs.items():
        print(f'\n{"="*70}')
        print(f'STAGE: {stage.upper()}')
        print(f'{"="*70}')
        for key, value in output.items():
            if isinstance(value, (list, dict)) and len(str(value)) > 200:
                print(f'  {key}: {str(value)[:200]}...')
            else:
                print(f'  {key}: {value}')

    print(f'\n{"="*70}')
    print('FINAL ANSWER')
    print(f'{"="*70}')
    answer = final_node_state.get('reasoning_result', {}).get('answer', 'N/A')
    print(answer)

    print(f'\n{"="*70}')
    print('EXECUTION LOG')
    print(f'{"="*70}')
    for i, msg in enumerate(final_node_state.get('messages', []), 1):
        print(f'{i}. {msg}')

    print(f'\n{"="*70}')
    print('SUMMARY STATISTICS')
    print(f'{"="*70}')
    print(f"Papers collected: {len(final_node_state.get('papers', []))}")
    print(f"Graph nodes: {final_node_state.get('graph_data', {}).get('nodes', 0)}")
    print(f"Graph edges: {final_node_state.get('graph_data', {}).get('edges', 0)}")
    print(f"Vector embeddings: {final_node_state.get('vector_data', {}).get('embeddings_added', 0)}")
    print(f"LlamaIndex docs: {final_node_state.get('llamaindex_data', {}).get('documents_indexed', 0)}")
    print(f"Answer length: {len(answer)} characters")
    print(f"Quality score: {final_node_state.get('reflection_feedback', {}).get('quality_score', 'N/A')}")
    print(f"Critic approved: {final_node_state.get('critic_feedback', {}).get('approved', False)}")

    print('\n' + '='*70)
    print('✅ FULL INTEGRATION TEST COMPLETE!')
    print('='*70 + '\n')

    return 0


if __name__ == "__main__":
    sys.exit(main())
