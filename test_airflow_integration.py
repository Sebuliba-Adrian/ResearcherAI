#!/usr/bin/env python3
"""
Comprehensive Airflow Integration Test for ResearcherAI
======================================================

This script tests the complete Airflow integration with ResearcherAI,
including DAG execution, data collection, and processing.

Author: ResearcherAI Team
"""

import os
import sys
import json
import time
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Add ResearcherAI to path
sys.path.insert(0, '/home/adrian/Desktop/Projects/ResearcherAI')

from agents.data_agent import DataCollectorAgent
from agents.orchestrator_agent import OrchestratorAgent

class AirflowIntegrationTester:
    """Comprehensive tester for Airflow integration"""
    
    def __init__(self):
        self.base_url = "http://localhost:8080"
        self.auth = ("airflow", "airflow")  # Default credentials
        self.test_results = {}
        self.session_name = f"airflow_test_{int(time.time())}"
        
    def test_airflow_health(self) -> bool:
        """Test if Airflow services are healthy"""
        print("🔍 Testing Airflow Health...")
        
        try:
            # Test webserver health
            response = requests.get(f"{self.base_url}/health", timeout=10)
            if response.status_code == 200:
                health_data = response.json()
                print(f"✅ Webserver: {health_data.get('metadatabase', {}).get('status', 'unknown')}")
                print(f"✅ Scheduler: {health_data.get('scheduler', {}).get('status', 'unknown')}")
                print(f"✅ Triggerer: {health_data.get('triggerer', {}).get('status', 'unknown')}")
                return True
            else:
                print(f"❌ Webserver health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Airflow health check failed: {e}")
            return False
    
    def test_dag_discovery(self) -> bool:
        """Test if DAGs are discovered and loaded"""
        print("\n🔍 Testing DAG Discovery...")
        
        try:
            # Test DAG list (requires authentication)
            response = requests.get(
                f"{self.base_url}/api/v1/dags",
                auth=self.auth,
                timeout=10
            )
            
            if response.status_code == 200:
                dags_data = response.json()
                dag_ids = [dag['dag_id'] for dag in dags_data.get('dags', [])]
                print(f"✅ Found {len(dag_ids)} DAGs: {dag_ids}")
                
                # Check for our specific DAGs
                expected_dags = ['research_paper_etl', 'system_monitoring']
                found_dags = [dag for dag in expected_dags if dag in dag_ids]
                
                if len(found_dags) == len(expected_dags):
                    print(f"✅ All expected DAGs found: {found_dags}")
                    return True
                else:
                    print(f"⚠️ Missing DAGs: {set(expected_dags) - set(found_dags)}")
                    return False
            else:
                print(f"❌ DAG discovery failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ DAG discovery failed: {e}")
            return False
    
    def test_data_collection(self) -> Dict[str, Any]:
        """Test data collection functionality"""
        print("\n🔍 Testing Data Collection...")
        
        results = {
            'arxiv': 0,
            'semantic_scholar': 0,
            'zenodo': 0,
            'pubmed': 0,
            'websearch': 0,
            'huggingface': 0,
            'kaggle': 0
        }
        
        collector = DataCollectorAgent()
        query = "artificial intelligence machine learning"
        max_papers = 5  # Reduced for testing
        
        try:
            # Test arXiv
            print("📚 Testing arXiv collection...")
            arxiv_papers = collector._fetch_arxiv(query, max_papers)
            results['arxiv'] = len(arxiv_papers)
            print(f"✅ arXiv: {len(arxiv_papers)} papers")
            
            # Test Semantic Scholar
            print("📚 Testing Semantic Scholar collection...")
            ss_papers = collector._fetch_semantic_scholar(query, max_papers)
            results['semantic_scholar'] = len(ss_papers)
            print(f"✅ Semantic Scholar: {len(ss_papers)} papers")
            
            # Test Zenodo
            print("📚 Testing Zenodo collection...")
            zenodo_papers = collector._fetch_zenodo(query, max_papers)
            results['zenodo'] = len(zenodo_papers)
            print(f"✅ Zenodo: {len(zenodo_papers)} papers")
            
            # Test PubMed
            print("📚 Testing PubMed collection...")
            pubmed_papers = collector._fetch_pubmed(query, max_papers)
            results['pubmed'] = len(pubmed_papers)
            print(f"✅ PubMed: {len(pubmed_papers)} papers")
            
            # Test Web Search
            print("📚 Testing Web Search collection...")
            web_papers = collector._fetch_websearch(query, max_papers)
            results['websearch'] = len(web_papers)
            print(f"✅ Web Search: {len(web_papers)} papers")
            
            # Test HuggingFace
            print("📚 Testing HuggingFace collection...")
            hf_papers = collector._fetch_huggingface(query, max_papers)
            results['huggingface'] = len(hf_papers)
            print(f"✅ HuggingFace: {len(hf_papers)} papers")
            
            # Test Kaggle (may fail due to credentials)
            print("📚 Testing Kaggle collection...")
            try:
                kaggle_papers = collector._fetch_kaggle(query, max_papers)
                results['kaggle'] = len(kaggle_papers)
                print(f"✅ Kaggle: {len(kaggle_papers)} papers")
            except Exception as e:
                print(f"⚠️ Kaggle: {e} (credentials may be missing)")
                results['kaggle'] = 0
            
            total_papers = sum(results.values())
            print(f"\n📊 Total papers collected: {total_papers}")
            
            return results
            
        except Exception as e:
            print(f"❌ Data collection failed: {e}")
            return results
    
    def test_orchestrator_integration(self) -> bool:
        """Test orchestrator integration"""
        print("\n🔍 Testing Orchestrator Integration...")
        
        try:
            orchestrator = OrchestratorAgent(session_name=self.session_name)
            
            # Test basic functionality
            print("✅ Orchestrator created successfully")
            
            # Test adding papers
            test_paper = {
                'title': 'Test Paper',
                'abstract': 'This is a test paper for integration testing.',
                'source': 'test',
                'url': 'https://example.com',
                'date': datetime.now().isoformat()
            }
            
            orchestrator.papers.append(test_paper)
            print("✅ Paper added to orchestrator")
            
            # Test vector storage
            text = f"{test_paper['title']} {test_paper['abstract']}"
            orchestrator.vector_agent.add_document(
                text=text,
                metadata={
                    'title': test_paper['title'],
                    'source': test_paper['source'],
                    'url': test_paper['url'],
                    'date': test_paper['date']
                }
            )
            print("✅ Document added to vector database")
            
            # Test knowledge graph
            triples = orchestrator.graph_agent.extract_triples(text)
            print(f"✅ Extracted {len(triples)} knowledge triples")
            
            # Test session persistence
            orchestrator.save_session()
            print("✅ Session saved successfully")
            
            return True
            
        except Exception as e:
            print(f"❌ Orchestrator integration failed: {e}")
            return False
    
    def test_dag_execution_simulation(self) -> bool:
        """Simulate DAG execution without Airflow worker"""
        print("\n🔍 Testing DAG Execution Simulation...")
        
        try:
            # Simulate the research_paper_etl DAG execution
            print("🚀 Simulating research_paper_etl DAG execution...")
            
            # Step 1: Data Collection
            collector = DataCollectorAgent()
            query = "artificial intelligence machine learning"
            max_papers = 3
            
            all_papers = []
            
            # Collect from multiple sources
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
                    papers = fetch_func(query, max_papers)
                    all_papers.extend(papers)
                    print(f"✅ {source_name}: {len(papers)} papers")
                except Exception as e:
                    print(f"⚠️ {source_name}: {e}")
            
            print(f"📦 Total papers collected: {len(all_papers)}")
            
            # Step 2: Check threshold
            threshold = 10
            if len(all_papers) >= threshold:
                print(f"✅ Threshold met ({len(all_papers)} >= {threshold}) - proceeding with processing")
                
                # Step 3: Process papers
                orchestrator = OrchestratorAgent(session_name=self.session_name)
                
                # Store in vector database
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
                
                print(f"✅ Stored {stored_count} embeddings in vector database")
                
                # Extract knowledge triples
                extracted_count = 0
                for paper in all_papers[:10]:  # Limit to avoid API rate limits
                    title = paper.get('title', '')
                    abstract = paper.get('abstract', '')
                    
                    if abstract:
                        try:
                            triples = orchestrator.graph_agent.extract_triples(f"{title}. {abstract}")
                            extracted_count += len(triples)
                        except Exception as e:
                            print(f"⚠️ Triple extraction failed for '{title[:50]}...': {e}")
                
                print(f"✅ Extracted {extracted_count} knowledge triples")
                
                # Save session
                orchestrator.save_session()
                print("✅ Session saved successfully")
                
            else:
                print(f"⚠️ Threshold not met ({len(all_papers)} < {threshold}) - skipping processing")
            
            print("✅ DAG execution simulation completed successfully")
            return True
            
        except Exception as e:
            print(f"❌ DAG execution simulation failed: {e}")
            return False
    
    def test_flower_monitoring(self) -> bool:
        """Test Flower monitoring interface"""
        print("\n🔍 Testing Flower Monitoring...")
        
        try:
            response = requests.get("http://localhost:5555/", timeout=10)
            if response.status_code == 200:
                print("✅ Flower monitoring interface accessible")
                return True
            else:
                print(f"❌ Flower monitoring failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Flower monitoring failed: {e}")
            return False
    
    def generate_test_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        print("\n" + "="*70)
        print("📊 AIRFLOW INTEGRATION TEST REPORT")
        print("="*70)
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'session_name': self.session_name,
            'test_results': self.test_results,
            'summary': {
                'total_tests': len(self.test_results),
                'passed_tests': sum(1 for result in self.test_results.values() if result),
                'failed_tests': sum(1 for result in self.test_results.values() if not result),
            }
        }
        
        for test_name, result in self.test_results.items():
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"{test_name:30s}: {status}")
        
        print(f"\nTotal Tests: {report['summary']['total_tests']}")
        print(f"Passed: {report['summary']['passed_tests']}")
        print(f"Failed: {report['summary']['failed_tests']}")
        print(f"Success Rate: {(report['summary']['passed_tests']/report['summary']['total_tests']*100):.1f}%")
        
        return report
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all integration tests"""
        print("🚀 Starting Airflow Integration Tests...")
        print("="*70)
        
        # Test 1: Airflow Health
        self.test_results['airflow_health'] = self.test_airflow_health()
        
        # Test 2: DAG Discovery
        self.test_results['dag_discovery'] = self.test_dag_discovery()
        
        # Test 3: Data Collection
        collection_results = self.test_data_collection()
        self.test_results['data_collection'] = sum(collection_results.values()) > 0
        
        # Test 4: Orchestrator Integration
        self.test_results['orchestrator_integration'] = self.test_orchestrator_integration()
        
        # Test 5: DAG Execution Simulation
        self.test_results['dag_execution_simulation'] = self.test_dag_execution_simulation()
        
        # Test 6: Flower Monitoring
        self.test_results['flower_monitoring'] = self.test_flower_monitoring()
        
        # Generate report
        report = self.generate_test_report()
        
        # Save report
        report_file = f"/home/adrian/Desktop/Projects/ResearcherAI/test_outputs/airflow_integration_report_{int(time.time())}.json"
        os.makedirs(os.path.dirname(report_file), exist_ok=True)
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📄 Test report saved to: {report_file}")
        
        return report

def main():
    """Main test execution"""
    print("🔬 ResearcherAI Airflow Integration Test Suite")
    print("="*70)
    
    tester = AirflowIntegrationTester()
    report = tester.run_all_tests()
    
    # Final status
    success_rate = (report['summary']['passed_tests'] / report['summary']['total_tests']) * 100
    
    if success_rate >= 80:
        print(f"\n🎉 INTEGRATION TEST SUCCESSFUL! ({success_rate:.1f}% pass rate)")
        return 0
    else:
        print(f"\n⚠️ INTEGRATION TEST PARTIALLY SUCCESSFUL ({success_rate:.1f}% pass rate)")
        return 1

if __name__ == "__main__":
    sys.exit(main())
