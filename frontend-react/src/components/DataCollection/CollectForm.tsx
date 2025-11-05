import React, { useState } from 'react';
import { motion } from 'framer-motion';
import GlassCard from '../Common/GlassCard';
import SourceSelector from './SourceSelector';
import LoadingSpinner from '../Common/LoadingSpinner';
import type { CollectFormData, DataSource } from '../../types';

interface CollectFormProps {
  onSubmit: (data: CollectFormData) => Promise<void>;
}

const CollectForm: React.FC<CollectFormProps> = ({ onSubmit }) => {
  const [query, setQuery] = useState('');
  const [sources, setSources] = useState<DataSource[]>(['arxiv']);
  const [maxResults, setMaxResults] = useState(10);
  const [dateFrom, setDateFrom] = useState('');
  const [dateTo, setDateTo] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!query.trim() || sources.length === 0) {
      return;
    }

    setIsLoading(true);
    try {
      await onSubmit({
        query: query.trim(),
        sources,
        max_results: maxResults,
        date_from: dateFrom || undefined,
        date_to: dateTo || undefined,
      });
    } finally {
      setIsLoading(false);
    }
  };

  const exampleQueries = [
    'Machine learning applications in healthcare',
    'Climate change impact on agriculture',
    'Quantum computing algorithms',
    'Neural networks for natural language processing',
  ];

  return (
    <GlassCard className="p-6 md:p-8">
      <form onSubmit={handleSubmit} className="space-y-6">
        {/* Header */}
        <div>
          <h2 className="text-2xl font-bold text-white mb-2">
            Collect Research Papers
          </h2>
          <p className="text-white/60 text-sm">
            Search for academic papers from multiple sources
          </p>
        </div>

        {/* Search Query */}
        <div className="space-y-2">
          <label htmlFor="query" className="block text-sm font-medium text-white/90">
            Search Query *
          </label>
          <div className="relative">
            <input
              id="query"
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Enter your research topic..."
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

          {/* Example Queries */}
          <div className="flex flex-wrap gap-2">
            {exampleQueries.slice(0, 2).map((example) => (
              <button
                key={example}
                type="button"
                onClick={() => setQuery(example)}
                className="text-xs px-3 py-1 rounded-full bg-white/5 hover:bg-white/10 text-white/60 hover:text-white transition-colors"
                disabled={isLoading}
              >
                {example}
              </button>
            ))}
          </div>
        </div>

        {/* Source Selector */}
        <SourceSelector selectedSources={sources} onChange={setSources} />

        {/* Max Results */}
        <div className="space-y-2">
          <label htmlFor="maxResults" className="block text-sm font-medium text-white/90">
            Maximum Results: {maxResults}
          </label>
          <input
            id="maxResults"
            type="range"
            min="5"
            max="100"
            step="5"
            value={maxResults}
            onChange={(e) => setMaxResults(Number(e.target.value))}
            className="w-full h-2 rounded-lg appearance-none cursor-pointer bg-white/10"
            disabled={isLoading}
          />
          <div className="flex justify-between text-xs text-white/40">
            <span>5</span>
            <span>50</span>
            <span>100</span>
          </div>
        </div>

        {/* Advanced Options Toggle */}
        <button
          type="button"
          onClick={() => setShowAdvanced(!showAdvanced)}
          className="flex items-center gap-2 text-sm text-blue-400 hover:text-blue-300 transition-colors"
          disabled={isLoading}
        >
          <svg
            className={`w-4 h-4 transition-transform ${showAdvanced ? 'rotate-180' : ''}`}
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </svg>
          Advanced Options
        </button>

        {/* Advanced Options */}
        <motion.div
          initial={false}
          animate={{ height: showAdvanced ? 'auto' : 0, opacity: showAdvanced ? 1 : 0 }}
          className="overflow-hidden"
        >
          <div className="space-y-4 pt-2">
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div className="space-y-2">
                <label htmlFor="dateFrom" className="block text-sm font-medium text-white/90">
                  Date From
                </label>
                <input
                  id="dateFrom"
                  type="date"
                  value={dateFrom}
                  onChange={(e) => setDateFrom(e.target.value)}
                  className="
                    w-full px-4 py-3 rounded-xl
                    bg-white/10 border border-white/20
                    text-white
                    focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent
                    transition-all
                  "
                  disabled={isLoading}
                />
              </div>
              <div className="space-y-2">
                <label htmlFor="dateTo" className="block text-sm font-medium text-white/90">
                  Date To
                </label>
                <input
                  id="dateTo"
                  type="date"
                  value={dateTo}
                  onChange={(e) => setDateTo(e.target.value)}
                  className="
                    w-full px-4 py-3 rounded-xl
                    bg-white/10 border border-white/20
                    text-white
                    focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent
                    transition-all
                  "
                  disabled={isLoading}
                />
              </div>
            </div>
          </div>
        </motion.div>

        {/* Submit Button */}
        <motion.button
          type="submit"
          disabled={isLoading || !query.trim() || sources.length === 0}
          className="
            w-full py-4 rounded-xl font-semibold
            bg-gradient-to-r from-blue-500 to-purple-600
            text-white
            disabled:opacity-50 disabled:cursor-not-allowed
            hover:shadow-lg hover:shadow-blue-500/50
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
              Collect Papers
            </span>
          )}
        </motion.button>

        {/* Info Message */}
        <div className="flex items-start gap-2 p-4 rounded-xl bg-blue-500/10 border border-blue-500/20">
          <svg className="w-5 h-5 text-blue-400 flex-shrink-0 mt-0.5" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
          </svg>
          <p className="text-sm text-blue-200">
            Papers will be processed and stored in your current session. You can query them using AI once collection is complete.
          </p>
        </div>
      </form>
    </GlassCard>
  );
};

export default CollectForm;
