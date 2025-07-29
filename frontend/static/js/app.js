/**
 * ResearcherAI Frontend Application
 * Modern JavaScript with ES6+ features
 */

// ============================================================================
// Configuration
// ============================================================================

const CONFIG = {
    API_BASE_URL: 'http://localhost:8000/v1',
    API_KEY_STORAGE_KEY: 'researcherai_api_key',
    DEFAULT_API_KEY: 'demo-key-123',
    ANIMATION_DELAY: 500,
    REASONING_STEP_DELAY: 1000
};

// ============================================================================
// State Management
// ============================================================================

const state = {
    apiKey: localStorage.getItem(CONFIG.API_KEY_STORAGE_KEY) || CONFIG.DEFAULT_API_KEY,
    currentSession: 'default',
    stats: {
        papers: 0,
        queries: 0
    }
};

// ============================================================================
// API Client
// ============================================================================

class APIClient {
    constructor(baseURL, apiKey) {
        this.baseURL = baseURL;
        this.apiKey = apiKey;
    }

    async request(endpoint, options = {}) {
        const url = `${this.baseURL}${endpoint}`;
        const headers = {
            'Content-Type': 'application/json',
            'X-API-Key': this.apiKey,
            ...options.headers
        };

        try {
            const response = await fetch(url, {
                ...options,
                headers
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.error || error.detail || 'Request failed');
            }

            return await response.json();
        } catch (error) {
            console.error('API Error:', error);
            throw error;
        }
    }

    async uploadFile(endpoint, file, additionalHeaders = {}) {
        const url = `${this.baseURL}${endpoint}`;
        const formData = new FormData();
        formData.append('file', file);

        const headers = {
            'X-API-Key': this.apiKey,
            ...additionalHeaders
        };

        try {
            const response = await fetch(url, {
                method: 'POST',
                headers,
                body: formData
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.error || error.detail || 'Upload failed');
            }

            return await response.json();
        } catch (error) {
            console.error('Upload Error:', error);
            throw error;
        }
    }

    // Health check
    async healthCheck() {
        return this.request('/health');
    }

    // Data collection
    async collectPapers(query, maxPerSource, sessionName) {
        return this.request('/collect', {
            method: 'POST',
            body: JSON.stringify({
                query,
                max_per_source: maxPerSource,
                session_name: sessionName
            })
        });
    }

    // Question answering
    async askQuestion(question, sessionName, useCritic) {
        return this.request('/ask', {
            method: 'POST',
            body: JSON.stringify({
                question,
                session_name: sessionName,
                use_critic: useCritic
            })
        });
    }

    // PDF upload
    async uploadPDF(file, sessionName) {
        const headers = sessionName ? { 'X-Session-Name': sessionName } : {};
        return this.uploadFile('/upload/pdf', file, headers);
    }

    // Session management
    async getSession(sessionName) {
        return this.request(`/sessions/${sessionName}`);
    }

    async deleteSession(sessionName) {
        return this.request(`/sessions/${sessionName}`, {
            method: 'DELETE'
        });
    }

    // Stats
    async getStats() {
        return this.request('/stats');
    }
}

// Initialize API client
let apiClient = new APIClient(CONFIG.API_BASE_URL, state.apiKey);

// ============================================================================
// UI Utilities
// ============================================================================

function showToast(title, message, type = 'info') {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;

    const icons = {
        success: `<svg class="icon toast-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
            <path d="M9 12l2 2 4-4m6 2a9 9 0 1 1-18 0 9 9 0 0 1 18 0z"/>
        </svg>`,
        error: `<svg class="icon toast-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
            <path d="M12 8v4m0 4h.01M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0z"/>
        </svg>`,
        info: `<svg class="icon toast-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
            <path d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0z"/>
        </svg>`
    };

    toast.innerHTML = `
        ${icons[type]}
        <div class="toast-content">
            <div class="toast-title">${title}</div>
            <div class="toast-message">${message}</div>
        </div>
        <button class="btn-icon" onclick="this.parentElement.remove()">
            <svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                <path d="M18 6L6 18M6 6l12 12"/>
            </svg>
        </button>
    `;

    container.appendChild(toast);

    // Auto-remove after 5 seconds
    setTimeout(() => {
        toast.style.animation = 'slideInRight 0.3s reverse';
        setTimeout(() => toast.remove(), 300);
    }, 5000);
}

function updateStats() {
    document.getElementById('stat-papers').textContent = state.stats.papers.toLocaleString();
    document.getElementById('stat-queries').textContent = state.stats.queries.toLocaleString();
}

function setLoading(buttonId, isLoading) {
    const button = document.getElementById(buttonId);
    if (!button) return;

    if (isLoading) {
        button.disabled = true;
        button.dataset.originalHtml = button.innerHTML;
        button.innerHTML = `
            <div class="spinner"></div>
            <span>Processing...</span>
        `;
    } else {
        button.disabled = false;
        button.innerHTML = button.dataset.originalHtml;
    }
}

function scrollToSection(sectionId) {
    const section = document.getElementById(sectionId);
    if (section) {
        section.scrollIntoView({ behavior: 'smooth' });
    }
}

function openModal(modalId) {
    const modal = document.getElementById(modalId);
    if (modal) {
        modal.classList.remove('hidden');
    }
}

function closeModal(modalId) {
    const modal = document.getElementById(modalId);
    if (modal) {
        modal.classList.add('hidden');
    }
}

// ============================================================================
// Reasoning Chain Animation
// ============================================================================

class ReasoningChainAnimator {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.steps = ['vector', 'graph', 'reasoning', 'critic'];
        this.currentStep = 0;
    }

    show() {
        this.container.classList.remove('hidden');
        this.reset();
    }

    hide() {
        this.container.classList.add('hidden');
        this.reset();
    }

    reset() {
        this.currentStep = 0;
        const stepElements = this.container.querySelectorAll('.chain-step');
        stepElements.forEach(el => {
            el.classList.remove('active', 'completed');
        });
    }

    async animateStep(stepName, duration = CONFIG.REASONING_STEP_DELAY) {
        const stepElement = this.container.querySelector(`[data-step="${stepName}"]`);
        if (!stepElement) return;

        // Activate step
        stepElement.classList.add('active');

        // Wait for duration
        await new Promise(resolve => setTimeout(resolve, duration));

        // Complete step
        stepElement.classList.remove('active');
        stepElement.classList.add('completed');
    }

    async animateAll(skipCritic = false) {
        this.show();

        await this.animateStep('vector', 1500);
        await this.animateStep('graph', 1500);
        await this.animateStep('reasoning', 2000);

        if (!skipCritic) {
            await this.animateStep('critic', 1000);
        }
    }

    updateStepDescription(stepName, description) {
        const stepElement = this.container.querySelector(`[data-step="${stepName}"]`);
        if (!stepElement) return;

        const descElement = stepElement.querySelector('.step-description');
        if (descElement) {
            descElement.textContent = description;
        }
    }
}

// ============================================================================
// Progress Bar Utilities
// ============================================================================

function updateProgress(progressId, percentage, step = null) {
    const container = document.getElementById(progressId);
    if (!container) return;

    // Update percentage
    const percentageEl = container.querySelector('.progress-percentage');
    const fillEl = container.querySelector('.progress-fill');

    if (percentageEl) {
        percentageEl.textContent = `${Math.round(percentage)}%`;
    }

    if (fillEl) {
        fillEl.style.width = `${percentage}%`;
    }

    // Update step if provided
    if (step !== null) {
        const steps = container.querySelectorAll('.progress-step');
        steps.forEach((stepEl, index) => {
            if (index === step) {
                stepEl.classList.add('active');
            } else if (index < step) {
                stepEl.classList.add('active');
            } else {
                stepEl.classList.remove('active');
            }
        });
    }
}

// ============================================================================
// Data Collection Handler
// ============================================================================

async function handleCollectForm(event) {
    event.preventDefault();

    const query = document.getElementById('collect-query').value;
    const maxPerSource = parseInt(document.getElementById('collect-max').value);
    const sessionName = document.getElementById('collect-session').value || null;

    // Show progress
    const progressContainer = document.getElementById('collect-progress');
    progressContainer.classList.remove('hidden');
    updateProgress('collect-progress', 0, 0);

    setLoading('collect-btn', true);

    try {
        // Simulate progress animation
        updateProgress('collect-progress', 20, 0);

        const result = await apiClient.collectPapers(query, maxPerSource, sessionName);

        updateProgress('collect-progress', 60, 1);
        await new Promise(resolve => setTimeout(resolve, 500));

        updateProgress('collect-progress', 90, 2);
        await new Promise(resolve => setTimeout(resolve, 500));

        updateProgress('collect-progress', 100, 3);

        // Update stats
        state.stats.papers += result.papers_collected;
        updateStats();

        // Show results
        displayCollectionResults(result);

        // Hide progress after a moment
        setTimeout(() => {
            progressContainer.classList.add('hidden');
        }, 1000);

        showToast(
            'Collection Complete',
            `Collected ${result.papers_collected} papers successfully`,
            'success'
        );

    } catch (error) {
        console.error('Collection error:', error);
        showToast('Collection Failed', error.message, 'error');
        progressContainer.classList.add('hidden');
    } finally {
        setLoading('collect-btn', false);
    }
}

function displayCollectionResults(result) {
    const resultsContainer = document.getElementById('collect-results');
    const resultsContent = document.getElementById('collect-results-content');

    const graphStats = result.graph_stats || {};
    const vectorStats = result.vector_stats || {};
    const criticEval = result.critic_evaluation || null;

    let html = `
        <div class="result-section">
            <h4>Collection Summary</h4>
            <div class="result-grid">
                <div class="result-item">
                    <div class="result-label">Papers Collected</div>
                    <div class="result-value">${result.papers_collected}</div>
                </div>
                <div class="result-item">
                    <div class="result-label">Session</div>
                    <div class="result-value">${result.session_name}</div>
                </div>
            </div>
        </div>

        <div class="result-section">
            <h4>Knowledge Graph</h4>
            <div class="result-grid">
                <div class="result-item">
                    <div class="result-label">Nodes Added</div>
                    <div class="result-value">${graphStats.nodes_added || 0}</div>
                </div>
                <div class="result-item">
                    <div class="result-label">Relationships</div>
                    <div class="result-value">${graphStats.relationships_added || 0}</div>
                </div>
            </div>
        </div>

        <div class="result-section">
            <h4>Vector Database</h4>
            <div class="result-grid">
                <div class="result-item">
                    <div class="result-label">Embeddings Created</div>
                    <div class="result-value">${vectorStats.documents_added || 0}</div>
                </div>
            </div>
        </div>
    `;

    if (criticEval) {
        html += `
            <div class="result-section">
                <h4>Quality Evaluation</h4>
                <div class="result-grid">
                    <div class="result-item">
                        <div class="result-label">Overall Score</div>
                        <div class="result-value">${criticEval.overall_score?.toFixed(2) || 'N/A'}/10</div>
                    </div>
                    <div class="result-item">
                        <div class="result-label">Diversity</div>
                        <div class="result-value">${criticEval.diversity_score?.toFixed(2) || 'N/A'}/10</div>
                    </div>
                    <div class="result-item">
                        <div class="result-label">Relevance</div>
                        <div class="result-value">${criticEval.relevance_score?.toFixed(2) || 'N/A'}/10</div>
                    </div>
                    <div class="result-item">
                        <div class="result-label">Quality</div>
                        <div class="result-value">${criticEval.quality_score?.toFixed(2) || 'N/A'}/10</div>
                    </div>
                </div>
            </div>
        `;
    }

    resultsContent.innerHTML = html;
    resultsContainer.style.display = 'block';

    // Smooth scroll to results
    setTimeout(() => {
        resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }, 300);
}

// ============================================================================
// Question Answering Handler
// ============================================================================

async function handleAskForm(event) {
    event.preventDefault();

    const question = document.getElementById('ask-question').value;
    const sessionName = document.getElementById('ask-session').value || 'default';
    const useCritic = document.getElementById('use-critic').checked;

    setLoading('ask-btn', true);

    // Initialize reasoning chain animator
    const reasoningAnimator = new ReasoningChainAnimator('reasoning-chain');

    try {
        // Start reasoning chain animation
        const animationPromise = reasoningAnimator.animateAll(!useCritic);

        // Make API call
        const result = await apiClient.askQuestion(question, sessionName, useCritic);

        // Wait for animation to complete
        await animationPromise;

        // Update stats
        state.stats.queries += 1;
        updateStats();

        // Display answer
        displayAnswer(result);

        // Hide reasoning chain
        setTimeout(() => {
            reasoningAnimator.hide();
        }, 1000);

        showToast('Answer Generated', 'Your question has been answered', 'success');

    } catch (error) {
        console.error('Ask error:', error);
        showToast('Question Failed', error.message, 'error');
        reasoningAnimator.hide();
    } finally {
        setLoading('ask-btn', false);
    }
}

function displayAnswer(result) {
    const resultsContainer = document.getElementById('ask-results');
    const answerContent = document.getElementById('ask-answer-content');

    let html = `
        <div class="answer-text">
            <p>${result.answer}</p>
        </div>
    `;

    // Add sources if available
    if (result.sources && result.sources.length > 0) {
        html += `
            <div class="sources-list">
                <h4>Sources (${result.sources.length} papers)</h4>
                ${result.sources.map(source => `
                    <div class="source-item">
                        <div class="source-title">${source.title || 'Untitled'}</div>
                        <div class="source-meta">
                            ${source.authors ? source.authors.slice(0, 3).join(', ') : 'Unknown authors'}
                            ${source.year ? ` • ${source.year}` : ''}
                        </div>
                    </div>
                `).join('')}
            </div>
        `;
    }

    // Add graph insights if available
    if (result.graph_insights && Object.keys(result.graph_insights).length > 0) {
        html += `
            <div class="result-section mt-2">
                <h4>Knowledge Graph Insights</h4>
                <pre style="color: var(--text-secondary); font-size: 0.9rem; overflow-x: auto;">${JSON.stringify(result.graph_insights, null, 2)}</pre>
            </div>
        `;
    }

    // Add critic evaluation if available
    if (result.critic_evaluation) {
        const eval = result.critic_evaluation;
        html += `
            <div class="result-section mt-2">
                <h4>Quality Assessment</h4>
                <div class="result-grid">
                    <div class="result-item">
                        <div class="result-label">Overall Score</div>
                        <div class="result-value">${eval.overall_score?.toFixed(2) || 'N/A'}/10</div>
                    </div>
                    <div class="result-item">
                        <div class="result-label">Accuracy</div>
                        <div class="result-value">${eval.accuracy_score?.toFixed(2) || 'N/A'}/10</div>
                    </div>
                    <div class="result-item">
                        <div class="result-label">Completeness</div>
                        <div class="result-value">${eval.completeness_score?.toFixed(2) || 'N/A'}/10</div>
                    </div>
                    <div class="result-item">
                        <div class="result-label">Clarity</div>
                        <div class="result-value">${eval.clarity_score?.toFixed(2) || 'N/A'}/10</div>
                    </div>
                </div>
                ${eval.feedback ? `<p class="mt-1" style="color: var(--text-secondary);">${eval.feedback}</p>` : ''}
            </div>
        `;
    }

    answerContent.innerHTML = html;
    resultsContainer.style.display = 'block';

    // Smooth scroll to answer
    setTimeout(() => {
        resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }, 300);
}

function copyAnswer() {
    const answerText = document.querySelector('.answer-text p');
    if (answerText) {
        navigator.clipboard.writeText(answerText.textContent);
        showToast('Copied', 'Answer copied to clipboard', 'success');
    }
}

// ============================================================================
// PDF Upload Handler
// ============================================================================

async function handlePDFUpload(file) {
    const sessionName = document.getElementById('collect-session')?.value || null;

    const progressContainer = document.getElementById('upload-progress');
    const resultsContainer = document.getElementById('upload-results');

    progressContainer.classList.remove('hidden');
    updateProgress('upload-progress', 0);

    try {
        // Simulate upload progress
        updateProgress('upload-progress', 30);

        const result = await apiClient.uploadPDF(file, sessionName);

        updateProgress('upload-progress', 100);

        // Display results
        resultsContainer.innerHTML = `
            <h4 style="color: var(--primary-color); margin-bottom: 1rem;">Upload Successful</h4>
            <div class="result-grid">
                <div class="result-item">
                    <div class="result-label">Paper Title</div>
                    <div class="result-value" style="font-size: 1rem;">${result.paper.title}</div>
                </div>
                <div class="result-item">
                    <div class="result-label">Authors</div>
                    <div class="result-value" style="font-size: 1rem;">${result.paper.authors.slice(0, 3).join(', ')}</div>
                </div>
                <div class="result-item">
                    <div class="result-label">Pages</div>
                    <div class="result-value">${result.paper.num_pages}</div>
                </div>
                <div class="result-item">
                    <div class="result-label">Session</div>
                    <div class="result-value">${result.session_name}</div>
                </div>
            </div>
        `;
        resultsContainer.classList.remove('hidden');

        // Update stats
        state.stats.papers += 1;
        updateStats();

        setTimeout(() => {
            progressContainer.classList.add('hidden');
        }, 1000);

        showToast('Upload Complete', `Successfully uploaded "${file.name}"`, 'success');

    } catch (error) {
        console.error('Upload error:', error);
        showToast('Upload Failed', error.message, 'error');
        progressContainer.classList.add('hidden');
        resultsContainer.classList.add('hidden');
    }
}

function setupPDFUpload() {
    const uploadZone = document.getElementById('upload-zone');
    const fileInput = document.getElementById('pdf-file');

    // Drag and drop
    uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadZone.classList.add('drag-over');
    });

    uploadZone.addEventListener('dragleave', () => {
        uploadZone.classList.remove('drag-over');
    });

    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.classList.remove('drag-over');

        const files = e.dataTransfer.files;
        if (files.length > 0 && files[0].type === 'application/pdf') {
            handlePDFUpload(files[0]);
        } else {
            showToast('Invalid File', 'Please upload a PDF file', 'error');
        }
    });

    // File input
    fileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file && file.type === 'application/pdf') {
            handlePDFUpload(file);
        } else {
            showToast('Invalid File', 'Please upload a PDF file', 'error');
        }
        // Reset input
        fileInput.value = '';
    });

    // Click to upload
    uploadZone.addEventListener('click', () => {
        fileInput.click();
    });
}

// ============================================================================
// Session Management
// ============================================================================

async function loadSessions() {
    // In a real implementation, you would fetch sessions from the backend
    // For now, show a placeholder
    showToast('Coming Soon', 'Session listing will be implemented', 'info');
}

// ============================================================================
// Theme Toggle
// ============================================================================

function setupThemeToggle() {
    const themeToggle = document.getElementById('theme-toggle');
    const savedTheme = localStorage.getItem('theme') || 'dark';

    if (savedTheme === 'light') {
        document.body.classList.add('light-theme');
    }

    themeToggle.addEventListener('click', () => {
        document.body.classList.toggle('light-theme');
        const theme = document.body.classList.contains('light-theme') ? 'light' : 'dark';
        localStorage.setItem('theme', theme);
    });
}

// ============================================================================
// API Key Management
// ============================================================================

function setupAPIKey() {
    const apiKeyBtn = document.getElementById('api-key-btn');
    const apiKeyInput = document.getElementById('api-key-input');

    apiKeyBtn.addEventListener('click', () => {
        openModal('api-key-modal');
        apiKeyInput.value = state.apiKey;
    });
}

function saveApiKey() {
    const apiKeyInput = document.getElementById('api-key-input');
    const newKey = apiKeyInput.value.trim();

    if (newKey) {
        state.apiKey = newKey;
        localStorage.setItem(CONFIG.API_KEY_STORAGE_KEY, newKey);
        apiClient = new APIClient(CONFIG.API_BASE_URL, newKey);

        showToast('API Key Saved', 'Your API key has been updated', 'success');
        closeModal('api-key-modal');
    } else {
        showToast('Invalid Key', 'Please enter a valid API key', 'error');
    }
}

// ============================================================================
// Navigation
// ============================================================================

function setupNavigation() {
    const navLinks = document.querySelectorAll('.nav-link');

    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const targetId = link.getAttribute('href').substring(1);

            // Update active state
            navLinks.forEach(l => l.classList.remove('active'));
            link.classList.add('active');

            // Scroll to section
            scrollToSection(targetId);
        });
    });

    // Update active nav on scroll
    const observer = new IntersectionObserver(
        (entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const id = entry.target.id;
                    navLinks.forEach(link => {
                        if (link.getAttribute('href') === `#${id}`) {
                            navLinks.forEach(l => l.classList.remove('active'));
                            link.classList.add('active');
                        }
                    });
                }
            });
        },
        { threshold: 0.5 }
    );

    document.querySelectorAll('section[id]').forEach(section => {
        observer.observe(section);
    });
}

// ============================================================================
// Health Check
// ============================================================================

async function performHealthCheck() {
    try {
        const health = await apiClient.healthCheck();
        console.log('Health check:', health);

        if (health.status === 'healthy') {
            // Optionally load initial stats
            try {
                const stats = await apiClient.getStats();
                if (stats.system && stats.system.papers_collected !== undefined) {
                    state.stats.papers = stats.system.papers_collected;
                }
                if (stats.system && stats.system.questions_answered !== undefined) {
                    state.stats.queries = stats.system.questions_answered;
                }
                updateStats();
            } catch (error) {
                console.warn('Could not load stats:', error);
            }
        }
    } catch (error) {
        console.warn('Backend not available:', error);
        showToast(
            'Backend Offline',
            'Could not connect to API. Make sure the server is running.',
            'error'
        );
    }
}

// ============================================================================
// Graph Visualization
// ============================================================================

let graphNetwork = null;
let graphData = { nodes: [], edges: [] };
let currentLayout = 'barnesHut';

async function loadGraphVisualization() {
    const graphLoading = document.getElementById('graph-loading');
    const graphEmpty = document.getElementById('graph-empty');
    const graphNetworkDiv = document.getElementById('graph-network');

    try {
        graphLoading.classList.remove('hidden');
        graphEmpty.classList.add('hidden');

        const response = await apiClient.request('/graph/export', { method: 'GET' });

        if (!response.nodes || response.nodes.length === 0) {
            graphLoading.classList.add('hidden');
            graphEmpty.classList.remove('hidden');
            showToast('No Graph Data', 'Collect some papers first to build the knowledge graph.', 'info');
            return;
        }

        graphData = {
            nodes: response.nodes,
            edges: response.edges
        };

        // Update stats
        document.getElementById('graph-nodes-count').textContent = response.stats.nodes || 0;
        document.getElementById('graph-edges-count').textContent = response.stats.edges || 0;
        document.getElementById('graph-session').textContent = response.session_name || 'default';

        // Initialize visualization
        renderGraph();

        graphLoading.classList.add('hidden');
        graphNetworkDiv.classList.add('active');

        showToast('Graph Loaded', `${response.nodes.length} nodes and ${response.edges.length} edges loaded.`, 'success');

    } catch (error) {
        console.error('Graph loading failed:', error);
        graphLoading.classList.add('hidden');
        graphEmpty.classList.remove('hidden');
        showToast('Graph Load Failed', error.message, 'error');
    }
}

function renderGraph() {
    const container = document.getElementById('graph-network');

    // Transform data for vis.js
    const visNodes = graphData.nodes.map(node => {
        // Determine color based on type
        let color = getNodeColor(node.type);

        return {
            id: node.id,
            label: node.label,
            title: createNodeTooltip(node),
            shape: node.type === 'paper' ? 'box' : 'dot',
            size: node.type === 'paper' ? 30 : 20,
            color: {
                background: color,
                border: color,
                highlight: {
                    background: color,
                    border: '#ffffff'
                }
            },
            font: {
                color: '#ffffff',
                size: 14,
                face: 'Inter'
            }
        };
    });

    const visEdges = graphData.edges.map(edge => ({
        from: edge.source,
        to: edge.target,
        label: edge.label || '',
        title: edge.label || '',
        arrows: {
            to: { enabled: true, scaleFactor: 0.5 }
        },
        color: {
            color: 'rgba(255, 255, 255, 0.2)',
            highlight: 'rgba(0, 245, 255, 0.6)'
        },
        font: {
            color: '#ffffff',
            size: 10,
            strokeWidth: 0
        },
        smooth: {
            type: 'continuous',
            roundness: 0.5
        }
    }));

    const data = {
        nodes: new vis.DataSet(visNodes),
        edges: new vis.DataSet(visEdges)
    };

    const options = getGraphOptions(currentLayout);

    // Destroy existing network
    if (graphNetwork !== null) {
        graphNetwork.destroy();
    }

    // Create new network
    graphNetwork = new vis.Network(container, data, options);

    // Add event listeners
    graphNetwork.on('click', function(params) {
        if (params.nodes.length > 0) {
            const nodeId = params.nodes[0];
            const node = graphData.nodes.find(n => n.id === nodeId);
            if (node) {
                showNodeDetails(node);
            }
        }
    });

    graphNetwork.on('stabilizationIterationsDone', function() {
        graphNetwork.setOptions({ physics: false });
    });
}

function getNodeColor(type) {
    switch(type?.toLowerCase()) {
        case 'paper':
            return '#00f5ff'; // Primary color
        case 'author':
            return '#ff006e'; // Secondary color
        case 'entity':
            return '#7b2ff7'; // Accent color
        default:
            return '#00f5ff';
    }
}

function createNodeTooltip(node) {
    let html = `<strong>${node.label}</strong><br>`;
    html += `Type: ${node.type}<br>`;

    if (node.properties) {
        if (node.properties.title) {
            html += `Title: ${node.properties.title}<br>`;
        }
        if (node.properties.source) {
            html += `Source: ${node.properties.source}<br>`;
        }
    }

    return html;
}

function showNodeDetails(node) {
    const details = JSON.stringify(node.properties || {}, null, 2);
    showToast(
        node.label,
        `Type: ${node.type}\n${details.slice(0, 200)}...`,
        'info',
        5000
    );
}

function getGraphOptions(layout) {
    const baseOptions = {
        nodes: {
            borderWidth: 2,
            borderWidthSelected: 4,
            shadow: {
                enabled: true,
                color: 'rgba(0, 0, 0, 0.5)',
                size: 10,
                x: 0,
                y: 0
            }
        },
        edges: {
            width: 1,
            selectionWidth: 3,
            shadow: false
        },
        interaction: {
            hover: true,
            tooltipDelay: 100,
            navigationButtons: true,
            keyboard: true,
            zoomView: true,
            dragView: true
        },
        physics: {
            enabled: true,
            stabilization: {
                enabled: true,
                iterations: 100,
                updateInterval: 10
            }
        }
    };

    // Layout-specific options
    switch(layout) {
        case 'barnesHut':
            baseOptions.physics.barnesHut = {
                gravitationalConstant: -8000,
                centralGravity: 0.3,
                springLength: 200,
                springConstant: 0.04,
                damping: 0.09,
                avoidOverlap: 0.1
            };
            break;

        case 'forceAtlas2Based':
            baseOptions.physics.forceAtlas2Based = {
                gravitationalConstant: -50,
                centralGravity: 0.01,
                springLength: 150,
                springConstant: 0.08,
                damping: 0.4,
                avoidOverlap: 0
            };
            baseOptions.physics.solver = 'forceAtlas2Based';
            break;

        case 'repulsion':
            baseOptions.physics.repulsion = {
                centralGravity: 0.2,
                springLength: 200,
                springConstant: 0.05,
                nodeDistance: 200,
                damping: 0.09
            };
            baseOptions.physics.solver = 'repulsion';
            break;

        case 'hierarchical':
            baseOptions.layout = {
                hierarchical: {
                    enabled: true,
                    direction: 'UD',
                    sortMethod: 'directed',
                    levelSeparation: 150,
                    nodeSpacing: 200
                }
            };
            baseOptions.physics.enabled = false;
            break;
    }

    return baseOptions;
}

function changeGraphLayout() {
    const layoutSelect = document.getElementById('graph-layout');
    currentLayout = layoutSelect.value;

    if (graphNetwork && graphData.nodes.length > 0) {
        renderGraph();
        showToast('Layout Changed', `Switched to ${currentLayout} layout`, 'success');
    }
}

function resetGraphZoom() {
    if (graphNetwork) {
        graphNetwork.fit({
            animation: {
                duration: 1000,
                easingFunction: 'easeInOutQuad'
            }
        });
        showToast('Zoom Reset', 'Graph view reset to fit all nodes', 'success');
    }
}

// ============================================================================
// RDF Export/Import
// ============================================================================

let selectedRDFFile = null;

async function exportRDF() {
    const format = document.getElementById('rdf-export-format').value;

    try {
        showToast('Exporting RDF', `Generating ${format.toUpperCase()} format...`, 'info');

        // Call API endpoint
        const url = `${CONFIG.API_BASE_URL}/graph/export/rdf?format=${format}`;
        const response = await fetch(url, {
            method: 'GET',
            headers: {
                'X-API-Key': state.apiKey
            }
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'RDF export failed');
        }

        // Get the blob
        const blob = await response.blob();

        // Get filename from headers or create default
        const contentDisposition = response.headers.get('Content-Disposition');
        let filename = 'knowledge_graph.ttl';

        if (contentDisposition) {
            const filenameMatch = contentDisposition.match(/filename="?(.+?)"?$/);
            if (filenameMatch) {
                filename = filenameMatch[1];
            }
        }

        // Create download link
        const downloadUrl = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = downloadUrl;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(downloadUrl);

        showToast('RDF Exported', `Downloaded as ${filename}`, 'success');

    } catch (error) {
        console.error('RDF export failed:', error);
        showToast('Export Failed', error.message, 'error');
    }
}

function setupRDFUpload() {
    const fileInput = document.getElementById('rdf-file');
    const uploadBtn = document.getElementById('rdf-upload-btn');
    const uploadZone = document.getElementById('rdf-upload-zone');

    fileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            selectedRDFFile = file;
            uploadBtn.disabled = false;
            uploadZone.querySelector('p').textContent = `Selected: ${file.name}`;
            uploadZone.style.borderColor = 'var(--primary-color)';
        }
    });
}

async function uploadRDF() {
    if (!selectedRDFFile) {
        showToast('No File Selected', 'Please select an RDF file first', 'error');
        return;
    }

    const merge = document.getElementById('rdf-merge').checked;

    try {
        showToast('Importing RDF', `Uploading ${selectedRDFFile.name}...`, 'info');

        // Create form data
        const formData = new FormData();
        formData.append('file', selectedRDFFile);

        // Call API endpoint
        const url = `${CONFIG.API_BASE_URL}/graph/import/rdf?merge=${merge}`;
        const response = await fetch(url, {
            method: 'POST',
            headers: {
                'X-API-Key': state.apiKey
            },
            body: formData
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'RDF import failed');
        }

        const result = await response.json();

        showToast(
            'RDF Imported',
            `Successfully imported ${result.import_stats.nodes_imported} nodes and ${result.import_stats.edges_imported} edges`,
            'success',
            5000
        );

        // Reset file input
        document.getElementById('rdf-file').value = '';
        selectedRDFFile = null;
        document.getElementById('rdf-upload-btn').disabled = true;
        document.getElementById('rdf-upload-zone').querySelector('p').textContent = 'Click to select RDF file';
        document.getElementById('rdf-upload-zone').style.borderColor = 'var(--glass-border)';

        // Reload graph visualization
        if (graphNetwork) {
            loadGraphVisualization();
        }

        // Update stats
        updateStats();

    } catch (error) {
        console.error('RDF import failed:', error);
        showToast('Import Failed', error.message, 'error');
    }
}

// ============================================================================
// Initialization
// ============================================================================

function init() {
    console.log('Initializing ResearcherAI Frontend...');

    // Setup event listeners
    document.getElementById('collect-form').addEventListener('submit', handleCollectForm);
    document.getElementById('ask-form').addEventListener('submit', handleAskForm);

    // Setup features
    setupThemeToggle();
    setupAPIKey();
    setupPDFUpload();
    setupRDFUpload();
    setupNavigation();

    // Make functions globally available
    window.scrollToSection = scrollToSection;
    window.openModal = openModal;
    window.closeModal = closeModal;
    window.saveApiKey = saveApiKey;
    window.loadSessions = loadSessions;
    window.copyAnswer = copyAnswer;
    window.loadGraphVisualization = loadGraphVisualization;
    window.changeGraphLayout = changeGraphLayout;
    window.resetGraphZoom = resetGraphZoom;
    window.exportRDF = exportRDF;
    window.uploadRDF = uploadRDF;

    // Initial health check
    performHealthCheck();

    // Update stats display
    updateStats();

    console.log('ResearcherAI Frontend Ready!');
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}
