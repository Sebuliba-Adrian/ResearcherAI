// Common Types for ResearcherAI Frontend

export interface Paper {
  id: string;
  title: string;
  authors: string[];
  abstract: string;
  published: string;
  url: string;
  source: DataSource;
  citations?: number;
}

export type DataSource = 'arxiv' | 'semantic_scholar' | 'pubmed' | 'google_scholar' | 'zenodo' | 'web_search' | 'huggingface' | 'kaggle';

export interface Session {
  id: string;
  name: string;
  created_at: string;
  updated_at: string;
  paper_count: number;
}

export interface QueryResponse {
  id: string;
  question: string;
  answer: string;
  sources: Paper[];
  confidence: number;
  timestamp: string;
}

export interface GraphNode {
  id: string;
  label: string;
  type: 'paper' | 'concept' | 'author';
  weight?: number;
}

export interface GraphEdge {
  source: string;
  target: string;
  weight: number;
  type: 'citation' | 'similarity' | 'co-authorship';
}

export interface GraphData {
  nodes: GraphNode[];
  edges: GraphEdge[];
}

export interface VectorSearchResult {
  id: string;
  content: string;
  paper: Paper;
  similarity: number;
  metadata: Record<string, any>;
}

export interface ToastMessage {
  id: string;
  type: 'success' | 'error' | 'info' | 'warning';
  message: string;
  duration?: number;
}

export interface CollectFormData {
  query: string;
  sources: DataSource[];
  max_results: number;
  date_from?: string;
  date_to?: string;
}

export interface UploadedFile {
  id: string;
  name: string;
  size: number;
  type: string;
  progress: number;
  status: 'pending' | 'uploading' | 'completed' | 'error';
  error?: string;
}

export interface CollectionHistoryItem {
  id: string;
  query: string;
  sources: DataSource[];
  timestamp: string;
  resultsCount: number;
  maxResults: number;
}
