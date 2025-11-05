import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import GlassCard from '../Common/GlassCard';
import LoadingSpinner from '../Common/LoadingSpinner';
import type { VectorSearchResult } from '../../types';

interface VectorSearchProps {
  onSearch: (query: string, limit: number) => Promise<VectorSearchResult[]>;
}

const VectorSearch: React.FC<VectorSearchProps> = ({ onSearch }) => {
  const [query, setQuery] = useState('');
  const [limit, setLimit] = useState(10);
  const [results, setResults] = useState<VectorSearchResult[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [hasSearched, setHasSearched] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;

    setIsLoading(true);
    setHasSearched(true);
    try {
      const searchResults = await onSearch(query.trim(), limit);
      setResults(searchResults);
    } catch (error) {
      console.error('Search error:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const getSimilarityColor = (similarity: number) => {
    if (similarity >= 0.8) return 'text-green-400 bg-green-500/20';
    if (similarity >= 0.6) return 'text-yellow-400 bg-yellow-500/20';
    return 'text-orange-400 bg-orange-500/20';
  };

  return (
    <div className="space-y-6">
      {/* Search Form */}
      <GlassCard className="p-6">
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <h2 className="text-2xl font-bold text-white mb-2">Vector Search</h2>
            <p className="text-white/60 text-sm">
              Find semantically similar content in your research collection
            </p>
          </div>

          {/* Search Input */}
          <div className="space-y-2">
            <label htmlFor="vectorQuery" className="block text-sm font-medium text-white/90">
              Search Query
            </label>
            <div className="relative">
              <input
                id="vectorQuery"
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Enter concepts, keywords, or questions..."
                className="
                  w-full px-4 py-3 pl-12 rounded-xl
                  bg-white/10 border border-white/20
                  text-white placeholder-white/40
                  focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent
                  transition-all
                "
                required
                disabled={isLoading}
              />
              <svg
                className="absolute left-4 top-1/2 transform -translate-y-1/2 w-5 h-5 text-white/40"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
              </svg>
            </div>
          </div>

          {/* Result Limit */}
          <div className="space-y-2">
            <label htmlFor="resultLimit" className="block text-sm font-medium text-white/90">
              Results to return: {limit}
            </label>
            <input
              id="resultLimit"
              type="range"
              min="5"
              max="50"
              step="5"
              value={limit}
              onChange={(e) => setLimit(Number(e.target.value))}
              className="w-full h-2 rounded-lg appearance-none cursor-pointer bg-white/10"
              disabled={isLoading}
            />
          </div>

          {/* Submit Button */}
          <motion.button
            type="submit"
            disabled={isLoading || !query.trim()}
            className="
              w-full py-3 rounded-xl font-semibold
              bg-gradient-to-r from-purple-500 to-pink-600
              text-white
              disabled:opacity-50 disabled:cursor-not-allowed
              hover:shadow-lg hover:shadow-purple-500/50
              transition-all
            "
            whileHover={!isLoading ? { scale: 1.02 } : {}}
            whileTap={!isLoading ? { scale: 0.98 } : {}}
          >
            {isLoading ? (
              <LoadingSpinner size="sm" />
            ) : (
              <span className="flex items-center justify-center gap-2">
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                </svg>
                Search
              </span>
            )}
          </motion.button>

          {/* Info */}
          <div className="flex items-start gap-2 p-3 rounded-xl bg-purple-500/10 border border-purple-500/20">
            <svg className="w-5 h-5 text-purple-400 flex-shrink-0 mt-0.5" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
            </svg>
            <p className="text-sm text-purple-200">
              Vector search uses semantic similarity to find relevant content, even if exact keywords don't match.
            </p>
          </div>
        </form>
      </GlassCard>

      {/* Results */}
      {isLoading && (
        <GlassCard className="p-12">
          <LoadingSpinner size="lg" message="Searching vector database..." />
        </GlassCard>
      )}

      {!isLoading && hasSearched && (
        <AnimatePresence>
          {results.length === 0 ? (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
            >
              <GlassCard className="p-12 text-center">
                <svg className="w-16 h-16 mx-auto text-white/40 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.172 16.172a4 4 0 015.656 0M9 10h.01M15 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                <p className="text-white/60">No results found</p>
                <p className="text-white/40 text-sm mt-2">Try a different search query</p>
              </GlassCard>
            </motion.div>
          ) : (
            <div className="space-y-4">
              <div className="flex items-center justify-between px-2">
                <h3 className="text-lg font-semibold text-white">
                  Found {results.length} results
                </h3>
              </div>

              {results.map((result, index) => (
                <motion.div
                  key={result.id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.05 }}
                >
                  <GlassCard className="p-6 hover:bg-white/15 transition-colors">
                    <div className="flex items-start gap-4">
                      {/* Similarity Score */}
                      <div className="flex-shrink-0">
                        <div className={`w-16 h-16 rounded-xl ${getSimilarityColor(result.similarity)} flex flex-col items-center justify-center`}>
                          <span className="text-2xl font-bold">{Math.round(result.similarity * 100)}</span>
                          <span className="text-xs opacity-80">%</span>
                        </div>
                      </div>

                      {/* Content */}
                      <div className="flex-1 min-w-0">
                        {/* Paper Info */}
                        <div className="flex items-start justify-between gap-4 mb-3">
                          <div className="flex-1">
                            <h4 className="text-white font-semibold mb-1 line-clamp-2">
                              {result.paper.title}
                            </h4>
                            <p className="text-white/60 text-sm">
                              {result.paper.authors.slice(0, 3).join(', ')}
                              {result.paper.authors.length > 3 && ` +${result.paper.authors.length - 3}`}
                            </p>
                          </div>
                          <a
                            href={result.paper.url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="flex-shrink-0 p-2 rounded-lg bg-white/10 hover:bg-white/20 transition-colors"
                            aria-label="View paper"
                          >
                            <svg className="w-5 h-5 text-white/70" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                            </svg>
                          </a>
                        </div>

                        {/* Matched Content */}
                        <div className="p-3 rounded-xl bg-white/5 border border-white/10 mb-3">
                          <p className="text-white/80 text-sm line-clamp-3">
                            {result.content}
                          </p>
                        </div>

                        {/* Metadata */}
                        <div className="flex flex-wrap items-center gap-2">
                          <span className="text-xs px-2 py-1 rounded-full bg-blue-500/20 text-blue-300">
                            {result.paper.source}
                          </span>
                          <span className="text-xs text-white/40">
                            {result.paper.published}
                          </span>
                          {result.paper.citations && (
                            <span className="text-xs text-white/40 flex items-center gap-1">
                              <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
                              </svg>
                              {result.paper.citations} citations
                            </span>
                          )}
                        </div>
                      </div>
                    </div>
                  </GlassCard>
                </motion.div>
              ))}
            </div>
          )}
        </AnimatePresence>
      )}
    </div>
  );
};

export default VectorSearch;
