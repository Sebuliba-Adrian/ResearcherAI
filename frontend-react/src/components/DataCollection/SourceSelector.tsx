import React from 'react';
import { motion } from 'framer-motion';
import type { DataSource } from '../../types';

interface SourceSelectorProps {
  selectedSources: DataSource[];
  onChange: (sources: DataSource[]) => void;
}

const SourceSelector: React.FC<SourceSelectorProps> = ({
  selectedSources,
  onChange,
}) => {
  const sources: { id: DataSource; name: string; description: string; icon: React.ReactNode }[] = [
    {
      id: 'arxiv',
      name: 'arXiv',
      description: 'Open-access scientific papers',
      icon: (
        <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 24 24">
          <path d="M12 2L2 7v10c0 5.55 3.84 10.74 9 12 5.16-1.26 9-6.45 9-12V7l-10-5zm0 2.18l8 3.6v8.72c0 4.29-3.17 8.28-8 9.45-4.83-1.17-8-5.16-8-9.45V7.78l8-3.6z" />
        </svg>
      ),
    },
    {
      id: 'semantic_scholar',
      name: 'Semantic Scholar',
      description: 'AI-powered research engine',
      icon: (
        <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 24 24">
          <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z" />
        </svg>
      ),
    },
    {
      id: 'pubmed',
      name: 'PubMed',
      description: 'Biomedical literature database',
      icon: (
        <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 24 24">
          <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zm-5 14H7v-2h7v2zm3-4H7v-2h10v2zm0-4H7V7h10v2z" />
        </svg>
      ),
    },
    {
      id: 'google_scholar',
      name: 'Google Scholar',
      description: 'Academic search engine',
      icon: (
        <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 24 24">
          <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-1 15h-2v-6h2v6zm4 0h-2v-6h2v6zm-2-8c-1.1 0-2-.9-2-2s.9-2 2-2 2 .9 2 2-.9 2-2 2z" />
        </svg>
      ),
    },
  ];

  const toggleSource = (sourceId: DataSource) => {
    if (selectedSources.includes(sourceId)) {
      onChange(selectedSources.filter((s) => s !== sourceId));
    } else {
      onChange([...selectedSources, sourceId]);
    }
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <label className="block text-sm font-medium text-white/90">
          Select Data Sources
        </label>
        <span className="text-xs text-white/60">
          {selectedSources.length} selected
        </span>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        {sources.map((source) => {
          const isSelected = selectedSources.includes(source.id);

          return (
            <motion.button
              key={source.id}
              type="button"
              onClick={() => toggleSource(source.id)}
              className={`
                relative p-4 rounded-xl border-2 transition-all text-left
                ${isSelected
                  ? 'border-blue-500 bg-blue-500/20 shadow-lg shadow-blue-500/20'
                  : 'border-white/10 bg-white/5 hover:bg-white/10'
                }
              `}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
            >
              {/* Selection Indicator */}
              <motion.div
                className={`
                  absolute top-3 right-3 w-5 h-5 rounded-full border-2 flex items-center justify-center
                  ${isSelected
                    ? 'border-blue-500 bg-blue-500'
                    : 'border-white/30'
                  }
                `}
                initial={false}
                animate={{ scale: isSelected ? 1 : 0.8 }}
              >
                {isSelected && (
                  <motion.svg
                    className="w-3 h-3 text-white"
                    fill="currentColor"
                    viewBox="0 0 20 20"
                    initial={{ scale: 0 }}
                    animate={{ scale: 1 }}
                  >
                    <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                  </motion.svg>
                )}
              </motion.div>

              {/* Content */}
              <div className="flex items-start gap-3 pr-8">
                <div className={`${isSelected ? 'text-blue-400' : 'text-white/60'} mt-1`}>
                  {source.icon}
                </div>
                <div className="flex-1">
                  <h3 className="font-semibold text-white mb-1">
                    {source.name}
                  </h3>
                  <p className="text-xs text-white/60">
                    {source.description}
                  </p>
                </div>
              </div>
            </motion.button>
          );
        })}
      </div>

      {/* Select All / Deselect All */}
      <div className="flex gap-2">
        <button
          type="button"
          onClick={() => onChange(sources.map((s) => s.id))}
          className="text-xs text-blue-400 hover:text-blue-300 transition-colors"
        >
          Select All
        </button>
        <span className="text-white/30">|</span>
        <button
          type="button"
          onClick={() => onChange([])}
          className="text-xs text-blue-400 hover:text-blue-300 transition-colors"
        >
          Deselect All
        </button>
      </div>

      {selectedSources.length === 0 && (
        <motion.p
          className="text-sm text-yellow-400/80 flex items-center gap-2"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
        >
          <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
          </svg>
          Please select at least one data source
        </motion.p>
      )}
    </div>
  );
};

export default SourceSelector;
