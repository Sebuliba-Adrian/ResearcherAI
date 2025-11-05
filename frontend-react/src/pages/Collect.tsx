import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import GlassCard from '../components/Common/GlassCard';
import type { DataSource, Paper, CollectionHistoryItem } from '../types';

interface SourceOption {
  id: DataSource;
  name: string;
  description: string;
  icon: React.ReactNode;
}

const Collect: React.FC = () => {
  const [selectedSource, setSelectedSource] = useState<DataSource>('arxiv');
  const [query, setQuery] = useState('');
  const [maxResults, setMaxResults] = useState(10);
  const [dateFrom, setDateFrom] = useState('');
  const [dateTo, setDateTo] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [results, setResults] = useState<Paper[]>([]);
  const [collectionHistory, setCollectionHistory] = useState<CollectionHistoryItem[]>([]);

  const sources: SourceOption[] = [
    {
      id: 'arxiv',
      name: 'arXiv',
      description: 'Open-access archive for scholarly articles in physics, mathematics, computer science, and more',
      icon: (
        <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
        </svg>
      ),
    },
    {
      id: 'semantic_scholar',
      name: 'Semantic Scholar',
      description: 'AI-powered research tool for scientific literature with citation analysis',
      icon: (
        <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
        </svg>
      ),
    },
    {
      id: 'zenodo',
      name: 'Zenodo',
      description: 'Open-access repository for research outputs including datasets and publications',
      icon: (
        <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 8h14M5 8a2 2 0 110-4h14a2 2 0 110 4M5 8v10a2 2 0 002 2h10a2 2 0 002-2V8m-9 4h4" />
        </svg>
      ),
    },
    {
      id: 'pubmed',
      name: 'PubMed',
      description: 'Database of biomedical and life sciences literature from MEDLINE and journals',
      icon: (
        <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
        </svg>
      ),
    },
    {
      id: 'web_search',
      name: 'Web Search',
      description: 'General web search for research articles and academic content',
      icon: (
        <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
        </svg>
      ),
    },
    {
      id: 'huggingface',
      name: 'HuggingFace',
      description: 'Machine learning models, datasets, and research papers from the AI community',
      icon: (
        <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z" />
        </svg>
      ),
    },
    {
      id: 'kaggle',
      name: 'Kaggle',
      description: 'Data science community with datasets, notebooks, and competitions',
      icon: (
        <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4m0 5c0 2.21-3.582 4-8 4s-8-1.79-8-4" />
        </svg>
      ),
    },
  ];

  const handleCollect = async () => {
    if (!query.trim()) return;

    setIsLoading(true);

    // Simulate API call
    setTimeout(() => {
      const mockResults: Paper[] = Array.from({ length: Math.min(maxResults, 5) }, (_, i) => ({
        id: `paper-${Date.now()}-${i}`,
        title: `Research Paper ${i + 1}: ${query}`,
        authors: [`Author ${i + 1}`, `Co-Author ${i + 1}`],
        abstract: `This is a comprehensive study on ${query}. The research explores various aspects and presents novel findings that contribute to the field. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.`,
        published: new Date(Date.now() - Math.random() * 365 * 24 * 60 * 60 * 1000).toISOString().split('T')[0],
        url: `https://example.com/paper-${i}`,
        source: selectedSource,
        citations: Math.floor(Math.random() * 500),
      }));

      setResults(mockResults);

      // Add to history
      const historyItem: CollectionHistoryItem = {
        id: `history-${Date.now()}`,
        query,
        sources: [selectedSource],
        timestamp: new Date().toISOString(),
        resultsCount: mockResults.length,
        maxResults,
      };

      setCollectionHistory((prev) => [historyItem, ...prev.slice(0, 9)]);
      setIsLoading(false);
    }, 2000);
  };

  const rerunQuery = (historyItem: CollectionHistoryItem) => {
    setQuery(historyItem.query);
    setSelectedSource(historyItem.sources[0]);
    setMaxResults(historyItem.maxResults);
  };

  const formatTimestamp = (timestamp: string) => {
    const date = new Date(timestamp);
    const now = new Date();
    const diff = now.getTime() - date.getTime();
    const minutes = Math.floor(diff / 60000);
    const hours = Math.floor(minutes / 60);
    const days = Math.floor(hours / 24);

    if (minutes < 1) return 'Just now';
    if (minutes < 60) return `${minutes}m ago`;
    if (hours < 24) return `${hours}h ago`;
    return `${days}d ago`;
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 p-6">
      <div className="max-w-7xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8"
        >
          <h1 className="text-4xl font-bold text-white mb-2">Data Collection</h1>
          <p className="text-white/60">Search and collect research papers from multiple sources</p>
        </motion.div>

        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          {/* Main Content */}
          <div className="lg:col-span-3 space-y-6">
            {/* Source Selector */}
            <GlassCard className="p-6">
              <h2 className="text-xl font-semibold text-white mb-4">Select Data Source</h2>
              <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-3">
                {sources.map((source) => (
                  <motion.button
                    key={source.id}
                    onClick={() => setSelectedSource(source.id)}
                    className={`
                      relative p-4 rounded-xl border-2 transition-all text-left
                      ${selectedSource === source.id
                        ? 'border-blue-500 bg-blue-500/20 shadow-lg shadow-blue-500/20'
                        : 'border-white/10 bg-white/5 hover:bg-white/10'
                      }
                    `}
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                  >
                    <div className="flex items-start gap-3">
                      <div className={`${selectedSource === source.id ? 'text-blue-400' : 'text-white/60'}`}>
                        {source.icon}
                      </div>
                      <div className="flex-1 min-w-0">
                        <h3 className="font-semibold text-white text-sm mb-1">
                          {source.name}
                        </h3>
                        <p className="text-xs text-white/50 line-clamp-2">
                          {source.description}
                        </p>
                      </div>
                    </div>
                    {selectedSource === source.id && (
                      <motion.div
                        className="absolute top-2 right-2"
                        initial={{ scale: 0 }}
                        animate={{ scale: 1 }}
                      >
                        <div className="w-5 h-5 rounded-full bg-blue-500 flex items-center justify-center">
                          <svg className="w-3 h-3 text-white" fill="currentColor" viewBox="0 0 20 20">
                            <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                          </svg>
                        </div>
                      </motion.div>
                    )}
                  </motion.button>
                ))}
              </div>
            </GlassCard>

            {/* Query Input Section */}
            <GlassCard className="p-6">
              <h2 className="text-xl font-semibold text-white mb-4">Search Query</h2>
              <div className="space-y-4">
                {/* Query Textarea */}
                <div>
                  <label className="block text-sm font-medium text-white/80 mb-2">
                    Research Query
                  </label>
                  <textarea
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    placeholder="Enter your research query or keywords..."
                    className="w-full h-32 px-4 py-3 rounded-xl bg-white/5 border border-white/10 text-white placeholder-white/40 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent backdrop-blur-xl resize-none"
                  />
                </div>

                {/* Parameters */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-white/80 mb-2">
                      Max Results
                    </label>
                    <input
                      type="number"
                      value={maxResults}
                      onChange={(e) => setMaxResults(parseInt(e.target.value) || 10)}
                      min="1"
                      max="100"
                      className="w-full px-4 py-2 rounded-xl bg-white/5 border border-white/10 text-white placeholder-white/40 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent backdrop-blur-xl"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-white/80 mb-2">
                      Date From
                    </label>
                    <input
                      type="date"
                      value={dateFrom}
                      onChange={(e) => setDateFrom(e.target.value)}
                      className="w-full px-4 py-2 rounded-xl bg-white/5 border border-white/10 text-white placeholder-white/40 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent backdrop-blur-xl"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-white/80 mb-2">
                      Date To
                    </label>
                    <input
                      type="date"
                      value={dateTo}
                      onChange={(e) => setDateTo(e.target.value)}
                      className="w-full px-4 py-2 rounded-xl bg-white/5 border border-white/10 text-white placeholder-white/40 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent backdrop-blur-xl"
                    />
                  </div>
                </div>

                {/* Collect Button */}
                <motion.button
                  onClick={handleCollect}
                  disabled={!query.trim() || isLoading}
                  className={`
                    w-full py-3 px-6 rounded-xl font-semibold text-white
                    ${!query.trim() || isLoading
                      ? 'bg-white/10 cursor-not-allowed'
                      : 'bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700 shadow-lg shadow-blue-500/30'
                    }
                    transition-all duration-200
                  `}
                  whileHover={query.trim() && !isLoading ? { scale: 1.02 } : undefined}
                  whileTap={query.trim() && !isLoading ? { scale: 0.98 } : undefined}
                >
                  {isLoading ? (
                    <span className="flex items-center justify-center gap-2">
                      <svg className="animate-spin h-5 w-5" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                      </svg>
                      Collecting...
                    </span>
                  ) : (
                    'Start Collection'
                  )}
                </motion.button>
              </div>
            </GlassCard>

            {/* Results Display */}
            <AnimatePresence mode="wait">
              {isLoading && (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="space-y-4"
                >
                  <h2 className="text-xl font-semibold text-white mb-4">Loading Results...</h2>
                  {Array.from({ length: 3 }).map((_, i) => (
                    <GlassCard key={i} className="p-6">
                      <div className="animate-pulse space-y-3">
                        <div className="h-6 bg-white/10 rounded w-3/4" />
                        <div className="h-4 bg-white/10 rounded w-1/2" />
                        <div className="space-y-2">
                          <div className="h-3 bg-white/10 rounded" />
                          <div className="h-3 bg-white/10 rounded" />
                          <div className="h-3 bg-white/10 rounded w-5/6" />
                        </div>
                      </div>
                    </GlassCard>
                  ))}
                </motion.div>
              )}

              {!isLoading && results.length > 0 && (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="space-y-4"
                >
                  <div className="flex items-center justify-between">
                    <h2 className="text-xl font-semibold text-white">
                      Results ({results.length})
                    </h2>
                    <span className="text-sm text-white/60">
                      Source: {sources.find((s) => s.id === selectedSource)?.name}
                    </span>
                  </div>

                  <div className="grid grid-cols-1 gap-4">
                    {results.map((paper, index) => (
                      <motion.div
                        key={paper.id}
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: index * 0.1 }}
                      >
                        <GlassCard hover className="p-6">
                          <div className="space-y-3">
                            {/* Title */}
                            <h3 className="text-lg font-semibold text-white hover:text-blue-400 transition-colors cursor-pointer">
                              {paper.title}
                            </h3>

                            {/* Authors & Date */}
                            <div className="flex flex-wrap items-center gap-3 text-sm text-white/60">
                              <span className="flex items-center gap-1">
                                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4.354a4 4 0 110 5.292M15 21H3v-1a6 6 0 0112 0v1zm0 0h6v-1a6 6 0 00-9-5.197M13 7a4 4 0 11-8 0 4 4 0 018 0z" />
                                </svg>
                                {paper.authors.join(', ')}
                              </span>
                              <span className="text-white/30">•</span>
                              <span className="flex items-center gap-1">
                                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
                                </svg>
                                {paper.published}
                              </span>
                              {paper.citations !== undefined && (
                                <>
                                  <span className="text-white/30">•</span>
                                  <span className="flex items-center gap-1">
                                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
                                    </svg>
                                    {paper.citations} citations
                                  </span>
                                </>
                              )}
                            </div>

                            {/* Abstract */}
                            <p className="text-white/70 text-sm line-clamp-3">
                              {paper.abstract}
                            </p>

                            {/* Source Badge & Actions */}
                            <div className="flex items-center justify-between pt-2">
                              <span className="px-3 py-1 rounded-full text-xs font-medium bg-blue-500/20 text-blue-400 border border-blue-500/30">
                                {sources.find((s) => s.id === paper.source)?.name}
                              </span>
                              <motion.a
                                href={paper.url}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="text-sm text-blue-400 hover:text-blue-300 flex items-center gap-1"
                                whileHover={{ x: 5 }}
                              >
                                View Paper
                                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 5l7 7m0 0l-7 7m7-7H3" />
                                </svg>
                              </motion.a>
                            </div>
                          </div>
                        </GlassCard>
                      </motion.div>
                    ))}
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          {/* Collection History Sidebar */}
          <div className="lg:col-span-1">
            <GlassCard className="p-6 sticky top-6">
              <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                Recent Queries
              </h2>

              {collectionHistory.length === 0 ? (
                <div className="text-center py-8 text-white/40 text-sm">
                  <svg className="w-12 h-12 mx-auto mb-2 opacity-50" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                  </svg>
                  No recent queries
                </div>
              ) : (
                <div className="space-y-3 max-h-[600px] overflow-y-auto">
                  {collectionHistory.map((item, index) => (
                    <motion.div
                      key={item.id}
                      initial={{ opacity: 0, x: 20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: index * 0.05 }}
                      className="p-3 rounded-lg bg-white/5 border border-white/10 hover:bg-white/10 transition-colors group"
                    >
                      <div className="space-y-2">
                        <p className="text-sm text-white line-clamp-2 font-medium">
                          {item.query}
                        </p>
                        <div className="flex items-center gap-2 text-xs text-white/50">
                          <span>{formatTimestamp(item.timestamp)}</span>
                          <span className="text-white/30">•</span>
                          <span>{item.resultsCount} results</span>
                        </div>
                        <div className="flex items-center gap-2">
                          {item.sources.map((source) => (
                            <span
                              key={source}
                              className="px-2 py-0.5 rounded text-xs bg-blue-500/20 text-blue-400 border border-blue-500/30"
                            >
                              {sources.find((s) => s.id === source)?.name}
                            </span>
                          ))}
                        </div>
                        <motion.button
                          onClick={() => rerunQuery(item)}
                          className="w-full mt-2 py-1.5 px-3 rounded-lg bg-blue-500/20 text-blue-400 text-xs font-medium hover:bg-blue-500/30 transition-colors opacity-0 group-hover:opacity-100"
                          whileHover={{ scale: 1.02 }}
                          whileTap={{ scale: 0.98 }}
                        >
                          Re-run Query
                        </motion.button>
                      </div>
                    </motion.div>
                  ))}
                </div>
              )}
            </GlassCard>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Collect;
