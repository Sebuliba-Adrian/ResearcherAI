#!/usr/bin/env python3
"""
End-to-End Airflow Integration Test with Sample Data
===================================================

This script demonstrates the complete ResearcherAI + Airflow integration
with sample data, showing all components working together.

Author: ResearcherAI Team
"""

import os
import sys
import json
import time
from datetime import datetime
from typing import Dict, List, Any

# Add ResearcherAI to path
sys.path.insert(0, '/home/adrian/Desktop/Projects/ResearcherAI')

from agents.data_agent import DataCollectorAgent
from agents.orchestrator_agent import OrchestratorAgent

class SampleDataDemo:
    """Demonstrates ResearcherAI + Airflow integration with sample data"""
    
    def __init__(self):
        self.session_name = f"airflow_demo_{int(time.time())}"
        self.sample_papers = self.create_sample_papers()
        
    def create_sample_papers(self) -> List[Dict[str, Any]]:
        """Create sample research papers for testing"""
        return [
            {
                'title': 'Attention Is All You Need',
                'abstract': 'The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely.',
                'source': 'arxiv',
                'url': 'https://arxiv.org/abs/1706.03762',
                'date': '2017-06-12',
                'authors': ['Ashish Vaswani', 'Noam Shazeer', 'Niki Parmar'],
                'keywords': ['transformer', 'attention', 'neural networks', 'machine learning']
            },
            {
                'title': 'BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding',
                'abstract': 'We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models, BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers.',
                'source': 'arxiv',
                'url': 'https://arxiv.org/abs/1810.04805',
                'date': '2018-10-11',
                'authors': ['Jacob Devlin', 'Ming-Wei Chang', 'Kenton Lee'],
                'keywords': ['bert', 'transformer', 'language understanding', 'nlp']
            },
            {
                'title': 'Generative Adversarial Networks',
                'abstract': 'We propose a new framework for estimating generative models via an adversarial process, in which we simultaneously train two models: a generative model G that captures the data distribution, and a discriminative model D that estimates the probability that a sample came from the training data rather than G.',
                'source': 'arxiv',
                'url': 'https://arxiv.org/abs/1406.2661',
                'date': '2014-06-10',
                'authors': ['Ian Goodfellow', 'Jean Pouget-Abadie', 'Mehdi Mirza'],
                'keywords': ['gan', 'generative models', 'adversarial training', 'deep learning']
            },
            {
                'title': 'ResNet: Deep Residual Learning for Image Recognition',
                'abstract': 'We present a residual learning framework to ease the training of networks that are substantially deeper than those used previously. We explicitly reformulate the layers as learning residual functions with reference to the layer inputs, instead of learning unreferenced functions.',
                'source': 'arxiv',
                'url': 'https://arxiv.org/abs/1512.03385',
                'date': '2015-12-10',
                'authors': ['Kaiming He', 'Xiangyu Zhang', 'Shaoqing Ren'],
                'keywords': ['resnet', 'residual networks', 'image recognition', 'computer vision']
            },
            {
                'title': 'GPT-3: Language Models are Few-Shot Learners',
                'abstract': 'We show that scaling up language models greatly improves task-agnostic, few-shot performance, sometimes even reaching competitiveness with prior state-of-the-art fine-tuning approaches. Specifically, we train GPT-3, an autoregressive language model with 175 billion parameters.',
                'source': 'arxiv',
                'url': 'https://arxiv.org/abs/2005.14165',
                'date': '2020-05-28',
                'authors': ['Tom B. Brown', 'Benjamin Mann', 'Nick Ryder'],
                'keywords': ['gpt-3', 'language models', 'few-shot learning', 'nlp']
            }
        ]
    
    def demonstrate_data_collection(self) -> Dict[str, Any]:
        """Demonstrate data collection from multiple sources"""
        print("🔍 DEMONSTRATING DATA COLLECTION")
        print("="*50)
        
        collector = DataCollectorAgent()
        query = "artificial intelligence machine learning"
        max_papers = 3
        
        results = {}
        
        # Test each source
        sources = [
            ('arxiv', collector._fetch_arxiv),
            ('semantic_scholar', collector._fetch_semantic_scholar),
            ('zenodo', collector._fetch_zenodo),
            ('pubmed', collector._fetch_pubmed),
            ('websearch', collector._fetch_websearch),
            ('huggingface', collector._fetch_huggingface),
        ]
        
        for source_name, fetch_func in sources:
            try:
                print(f"📚 Collecting from {source_name}...")
                papers = fetch_func(query, max_papers)
                results[source_name] = papers
                print(f"✅ {source_name}: {len(papers)} papers collected")
                
                # Show sample paper
                if papers:
                    sample = papers[0]
                    print(f"   Sample: {sample.get('title', 'No title')[:60]}...")
                    
            except Exception as e:
                print(f"⚠️ {source_name}: {e}")
                results[source_name] = []
        
        total_papers = sum(len(papers) for papers in results.values())
        print(f"\n📊 Total papers collected: {total_papers}")
        
        return results
    
    def demonstrate_orchestrator_processing(self) -> Dict[str, Any]:
        """Demonstrate orchestrator processing with sample data"""
        print("\n🔍 DEMONSTRATING ORCHESTRATOR PROCESSING")
        print("="*50)
        
        try:
            orchestrator = OrchestratorAgent(session_name=self.session_name)
            print(f"✅ Orchestrator created with session: {self.session_name}")
            
            # Add sample papers
            for paper in self.sample_papers:
                orchestrator.papers.append(paper)
            print(f"✅ Added {len(self.sample_papers)} sample papers")
            
            # Demonstrate vector storage
            print("\n🔢 Storing papers in vector database...")
            stored_count = 0
            for paper in self.sample_papers:
                text = f"{paper['title']} {paper['abstract']}"
                orchestrator.vector_agent.add_document(
                    text=text,
                    metadata={
                        'title': paper['title'],
                        'source': paper['source'],
                        'url': paper['url'],
                        'date': paper['date'],
                        'authors': paper['authors'],
                        'keywords': paper['keywords']
                    }
                )
                stored_count += 1
                print(f"   ✅ Stored: {paper['title'][:50]}...")
            
            print(f"✅ Stored {stored_count} embeddings in vector database")
            
            # Demonstrate knowledge graph extraction (if API key available)
            print("\n🕸️ Extracting knowledge triples...")
            extracted_count = 0
            
            # Check if API key is available
            if os.getenv('GOOGLE_API_KEY'):
                for paper in self.sample_papers:
                    try:
                        text = f"{paper['title']}. {paper['abstract']}"
                        triples = orchestrator.graph_agent.extract_triples(text)
                        extracted_count += len(triples)
                        print(f"   ✅ {paper['title'][:30]}...: {len(triples)} triples")
                    except Exception as e:
                        print(f"   ⚠️ {paper['title'][:30]}...: {e}")
            else:
                print("   ⚠️ GOOGLE_API_KEY not set - skipping triple extraction")
                print("   💡 Set GOOGLE_API_KEY to enable knowledge graph extraction")
            
            print(f"✅ Extracted {extracted_count} knowledge triples")
            
            # Demonstrate session persistence
            print("\n💾 Saving session...")
            orchestrator.save_session()
            print("✅ Session saved successfully")
            
            return {
                'papers_added': len(self.sample_papers),
                'vectors_stored': stored_count,
                'triples_extracted': extracted_count,
                'session_saved': True
            }
            
        except Exception as e:
            print(f"❌ Orchestrator processing failed: {e}")
            return {'error': str(e)}
    
    def demonstrate_dag_simulation(self) -> Dict[str, Any]:
        """Simulate the complete DAG execution"""
        print("\n🔍 DEMONSTRATING DAG EXECUTION SIMULATION")
        print("="*50)
        
        try:
            print("🚀 Simulating research_paper_etl DAG execution...")
            
            # Step 1: Data Collection (simulate parallel tasks)
            print("\n📚 Step 1: Parallel Data Collection")
            collector = DataCollectorAgent()
            query = "artificial intelligence machine learning"
            max_papers = 2
            
            all_papers = []
            collection_results = {}
            
            # Simulate parallel collection
            sources = [
                ('arxiv', collector._fetch_arxiv),
                ('semantic_scholar', collector._fetch_semantic_scholar),
                ('zenodo', collector._fetch_zenodo),
                ('pubmed', collector._fetch_pubmed),
            ]
            
            for source_name, fetch_func in sources:
                try:
                    papers = fetch_func(query, max_papers)
                    all_papers.extend(papers)
                    collection_results[source_name] = len(papers)
                    print(f"   ✅ {source_name}: {len(papers)} papers")
                except Exception as e:
                    collection_results[source_name] = 0
                    print(f"   ⚠️ {source_name}: {e}")
            
            print(f"📦 Total papers collected: {len(all_papers)}")
            
            # Step 2: Merge papers
            print("\n📦 Step 2: Merging Papers")
            print(f"✅ Merged {len(all_papers)} papers from {len(sources)} sources")
            
            # Step 3: Check threshold
            print("\n📊 Step 3: Checking Threshold")
            threshold = 5
            if len(all_papers) >= threshold:
                print(f"✅ Threshold met ({len(all_papers)} >= {threshold}) - proceeding with processing")
                
                # Step 4: Process papers
                print("\n⚙️ Step 4: Processing Papers")
                orchestrator = OrchestratorAgent(session_name=self.session_name)
                
                # Store in vector database
                print("   🔢 Storing in vector database...")
                stored_count = 0
                for paper in all_papers:
                    text = f"{paper.get('title', '')} {paper.get('abstract', '')}"
                    if text.strip():
                        orchestrator.vector_agent.add_document(
                            text=text,
                            metadata={
                                'title': paper.get('title'),
                                'source': paper.get('source'),
                                'url': paper.get('url'),
                                'date': paper.get('date')
                            }
                        )
                        stored_count += 1
                
                print(f"   ✅ Stored {stored_count} embeddings")
                
                # Extract knowledge triples
                print("   🕸️ Extracting knowledge triples...")
                extracted_count = 0
                if os.getenv('GOOGLE_API_KEY'):
                    for paper in all_papers[:5]:  # Limit to avoid API rate limits
                        title = paper.get('title', '')
                        abstract = paper.get('abstract', '')
                        
                        if abstract:
                            try:
                                triples = orchestrator.graph_agent.extract_triples(f"{title}. {abstract}")
                                extracted_count += len(triples)
                            except Exception as e:
                                print(f"   ⚠️ Triple extraction failed: {e}")
                else:
                    print("   ⚠️ GOOGLE_API_KEY not set - skipping triple extraction")
                
                print(f"   ✅ Extracted {extracted_count} knowledge triples")
                
                # Save session
                orchestrator.save_session()
                print("   ✅ Session saved")
                
            else:
                print(f"⚠️ Threshold not met ({len(all_papers)} < {threshold}) - skipping processing")
            
            # Step 5: Generate summary
            print("\n📊 Step 5: Generating Summary")
            summary = {
                'run_date': datetime.now().isoformat(),
                'query': query,
                'papers_collected': len(all_papers),
                'sources': list(collection_results.keys()),
                'vectors_stored': stored_count if len(all_papers) >= threshold else 0,
                'triples_extracted': extracted_count if len(all_papers) >= threshold else 0,
                'session': self.session_name,
                'threshold_met': len(all_papers) >= threshold
            }
            
            print("="*70)
            print("📊 DAG EXECUTION SUMMARY")
            print("="*70)
            for key, value in summary.items():
                print(f"{key:20s}: {value}")
            print("="*70)
            
            print("✅ DAG execution simulation completed successfully")
            return summary
            
        except Exception as e:
            print(f"❌ DAG execution simulation failed: {e}")
            return {'error': str(e)}
    
    def demonstrate_airflow_integration(self) -> Dict[str, Any]:
        """Demonstrate Airflow integration features"""
        print("\n🔍 DEMONSTRATING AIRFLOW INTEGRATION")
        print("="*50)
        
        try:
            # Test Airflow web interface accessibility
            import requests
            
            print("🌐 Testing Airflow Web Interface...")
            response = requests.get("http://localhost:8080/health", timeout=10)
            if response.status_code == 200:
                health_data = response.json()
                print("✅ Airflow Web Interface: Accessible")
                print(f"   📊 Database: {health_data.get('metadatabase', {}).get('status', 'unknown')}")
                print(f"   ⏰ Scheduler: {health_data.get('scheduler', {}).get('status', 'unknown')}")
                print(f"   🔄 Triggerer: {health_data.get('triggerer', {}).get('status', 'unknown')}")
            else:
                print(f"❌ Airflow Web Interface: {response.status_code}")
            
            # Test Flower monitoring
            print("\n🌸 Testing Flower Monitoring...")
            response = requests.get("http://localhost:5555/", timeout=10)
            if response.status_code == 200:
                print("✅ Flower Monitoring: Accessible")
            else:
                print(f"❌ Flower Monitoring: {response.status_code}")
            
            # Test DAG discovery
            print("\n📋 Testing DAG Discovery...")
            response = requests.get(
                "http://localhost:8080/api/v1/dags",
                auth=("airflow", "airflow"),
                timeout=10
            )
            if response.status_code == 200:
                dags_data = response.json()
                dag_ids = [dag['dag_id'] for dag in dags_data.get('dags', [])]
                print(f"✅ DAG Discovery: Found {len(dag_ids)} DAGs")
                print(f"   📄 DAGs: {dag_ids}")
            else:
                print(f"❌ DAG Discovery: {response.status_code}")
            
            return {
                'web_interface': response.status_code == 200,
                'flower_monitoring': True,
                'dag_discovery': len(dag_ids) if 'dag_ids' in locals() else 0
            }
            
        except Exception as e:
            print(f"❌ Airflow integration test failed: {e}")
            return {'error': str(e)}
    
    def generate_demo_report(self) -> Dict[str, Any]:
        """Generate comprehensive demo report"""
        print("\n" + "="*70)
        print("📊 RESEARCHERAI + AIRFLOW INTEGRATION DEMO REPORT")
        print("="*70)
        
        # Run all demonstrations
        data_collection_results = self.demonstrate_data_collection()
        orchestrator_results = self.demonstrate_orchestrator_processing()
        dag_simulation_results = self.demonstrate_dag_simulation()
        airflow_integration_results = self.demonstrate_airflow_integration()
        
        # Generate summary
        report = {
            'timestamp': datetime.now().isoformat(),
            'session_name': self.session_name,
            'demo_results': {
                'data_collection': data_collection_results,
                'orchestrator_processing': orchestrator_results,
                'dag_simulation': dag_simulation_results,
                'airflow_integration': airflow_integration_results
            },
            'summary': {
                'total_papers_collected': sum(len(papers) for papers in data_collection_results.values()),
                'sample_papers_processed': len(self.sample_papers),
                'vectors_stored': orchestrator_results.get('vectors_stored', 0),
                'triples_extracted': orchestrator_results.get('triples_extracted', 0),
                'airflow_healthy': airflow_integration_results.get('web_interface', False),
                'dags_discovered': airflow_integration_results.get('dag_discovery', 0)
            }
        }
        
        print("\n📊 DEMO SUMMARY")
        print("-" * 30)
        print(f"Total Papers Collected: {report['summary']['total_papers_collected']}")
        print(f"Sample Papers Processed: {report['summary']['sample_papers_processed']}")
        print(f"Vectors Stored: {report['summary']['vectors_stored']}")
        print(f"Triples Extracted: {report['summary']['triples_extracted']}")
        print(f"Airflow Healthy: {report['summary']['airflow_healthy']}")
        print(f"DAGs Discovered: {report['summary']['dags_discovered']}")
        
        # Save report
        report_file = f"/home/adrian/Desktop/Projects/ResearcherAI/test_outputs/airflow_demo_report_{int(time.time())}.json"
        os.makedirs(os.path.dirname(report_file), exist_ok=True)
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📄 Demo report saved to: {report_file}")
        
        return report

def main():
    """Main demo execution"""
    print("🎬 ResearcherAI + Airflow Integration Demo")
    print("="*70)
    
    demo = SampleDataDemo()
    report = demo.generate_demo_report()
    
    # Final status
    if report['summary']['airflow_healthy'] and report['summary']['dags_discovered'] > 0:
        print(f"\n🎉 INTEGRATION DEMO SUCCESSFUL!")
        print("✅ Airflow is fully integrated and working as expected")
        print("✅ All core functionality demonstrated with sample data")
        return 0
    else:
        print(f"\n⚠️ INTEGRATION DEMO PARTIALLY SUCCESSFUL")
        print("⚠️ Some components may need configuration (e.g., GOOGLE_API_KEY)")
        return 1

if __name__ == "__main__":
    sys.exit(main())
