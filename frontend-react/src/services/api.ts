import axios from 'axios';
import type { AxiosInstance } from 'axios';
import type {
  Paper,
  DataSource,
  QueryResponse,
  GraphData,
  VectorSearchResult,
  Session,
  CollectFormData
} from '../types';

class APIService {
  private api: AxiosInstance;
  private baseURL: string;

  constructor() {
    this.baseURL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
    this.api = axios.create({
      baseURL: this.baseURL,
      headers: {
        'Content-Type': 'application/json',
      },
      timeout: 30000,
    });

    // Request interceptor for adding auth tokens if needed
    this.api.interceptors.request.use(
      (config) => {
        const token = localStorage.getItem('auth_token');
        if (token) {
          config.headers.Authorization = `Bearer ${token}`;
        }
        return config;
      },
      (error) => Promise.reject(error)
    );

    // Response interceptor for error handling
    this.api.interceptors.response.use(
      (response) => response,
      (error) => {
        if (error.response?.status === 401) {
          localStorage.removeItem('auth_token');
          window.location.href = '/login';
        }
        return Promise.reject(error);
      }
    );
  }

  // Data Collection endpoints
  async collectData(data: CollectFormData): Promise<{ papers: Paper[]; count: number }> {
    const response = await this.api.post('/api/collect', data);
    return response.data;
  }

  async getDataSources(): Promise<DataSource[]> {
    const response = await this.api.get('/api/sources');
    return response.data;
  }

  // Query/Ask endpoints
  async askQuestion(
    question: string,
    sessionId?: string,
    model?: string,
    temperature?: number
  ): Promise<QueryResponse> {
    const response = await this.api.post('/api/query', {
      question,
      session_id: sessionId,
      model,
      temperature,
    });
    return response.data;
  }

  async getConversationHistory(sessionId: string): Promise<QueryResponse[]> {
    const response = await this.api.get(`/api/sessions/${sessionId}/history`);
    return response.data;
  }

  // Graph endpoints
  async getGraphData(filters?: {
    node_types?: string[];
    limit?: number;
  }): Promise<GraphData> {
    const response = await this.api.get('/api/graph', { params: filters });
    return response.data;
  }

  async addGraphNode(node: {
    label: string;
    type: string;
    properties?: Record<string, any>;
  }): Promise<{ id: string }> {
    const response = await this.api.post('/api/graph/nodes', node);
    return response.data;
  }

  async deleteGraphNode(nodeId: string): Promise<void> {
    await this.api.delete(`/api/graph/nodes/${nodeId}`);
  }

  async expandGraphNode(nodeId: string, depth?: number): Promise<GraphData> {
    const response = await this.api.post(`/api/graph/nodes/${nodeId}/expand`, { depth });
    return response.data;
  }

  // Vector Search endpoints
  async vectorSearch(query: string, topK: number = 10, threshold: number = 0.7): Promise<{
    results: VectorSearchResult[];
    total: number;
    search_time: number;
  }> {
    const response = await this.api.post('/api/vector/search', {
      query,
      top_k: topK,
      threshold,
    });
    return response.data;
  }

  async getVectorStats(): Promise<{
    total_vectors: number;
    dimensions: number;
    index_size: number;
  }> {
    const response = await this.api.get('/api/vector/stats');
    return response.data;
  }

  // Upload endpoints
  async uploadFile(file: File, onProgress?: (progress: number) => void): Promise<{
    id: string;
    status: string;
    message: string;
  }> {
    const formData = new FormData();
    formData.append('file', file);

    const response = await this.api.post('/api/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      onUploadProgress: (progressEvent) => {
        if (progressEvent.total && onProgress) {
          const percentCompleted = Math.round(
            (progressEvent.loaded * 100) / progressEvent.total
          );
          onProgress(percentCompleted);
        }
      },
    });

    return response.data;
  }

  async processUploadedFiles(fileIds: string[]): Promise<{
    success: boolean;
    processed: number;
    failed: number;
  }> {
    const response = await this.api.post('/api/upload/process', { file_ids: fileIds });
    return response.data;
  }

  // Session endpoints
  async getSessions(): Promise<Session[]> {
    const response = await this.api.get('/api/sessions');
    return response.data;
  }

  async createSession(name: string): Promise<Session> {
    const response = await this.api.post('/api/sessions', { name });
    return response.data;
  }

  async loadSession(sessionId: string): Promise<Session> {
    const response = await this.api.get(`/api/sessions/${sessionId}`);
    return response.data;
  }

  async deleteSession(sessionId: string): Promise<void> {
    await this.api.delete(`/api/sessions/${sessionId}`);
  }

  async exportSession(sessionId: string): Promise<Blob> {
    const response = await this.api.get(`/api/sessions/${sessionId}/export`, {
      responseType: 'blob',
    });
    return response.data;
  }

  // Health check
  async healthCheck(): Promise<{ status: string; version: string }> {
    const response = await this.api.get('/api/health');
    return response.data;
  }
}

// Export singleton instance
export const apiService = new APIService();
export default apiService;
