import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import GlassCard from '../Common/GlassCard';
import type { QueryResponse } from '../../types';

interface ResponseDisplayProps {
  responses: QueryResponse[];
}

const ResponseDisplay: React.FC<ResponseDisplayProps> = ({ responses }) => {
  const [expandedSources, setExpandedSources] = useState<Set<string>>(new Set());

  const toggleSources = (responseId: string) => {
    const newExpanded = new Set(expandedSources);
    if (newExpanded.has(responseId)) {
      newExpanded.delete(responseId);
    } else {
      newExpanded.add(responseId);
    }
    setExpandedSources(newExpanded);
  };

  const formatMarkdown = (text: string) => {
    // Simple markdown parsing for bold, italic, and code blocks
    return text
      .split('\n')
      .map((line, i) => {
        // Headers
        if (line.startsWith('### ')) {
          return <h3 key={i} className="text-lg font-semibold text-white mt-4 mb-2">{line.slice(4)}</h3>;
        }
        if (line.startsWith('## ')) {
          return <h2 key={i} className="text-xl font-bold text-white mt-4 mb-2">{line.slice(3)}</h2>;
        }
        if (line.startsWith('# ')) {
          return <h1 key={i} className="text-2xl font-bold text-white mt-4 mb-2">{line.slice(2)}</h1>;
        }

        // Code blocks
        if (line.startsWith('```')) {
          return null; // Handle code blocks separately
        }

        // Lists
        if (line.startsWith('- ') || line.startsWith('* ')) {
          return (
            <li key={i} className="ml-4 text-white/80">
              {line.slice(2)}
            </li>
          );
        }

        // Regular text with inline formatting
        const formatted = line
          .replace(/\*\*(.*?)\*\*/g, '<strong class="text-white font-semibold">$1</strong>')
          .replace(/\*(.*?)\*/g, '<em class="text-white/90 italic">$1</em>')
          .replace(/`(.*?)`/g, '<code class="px-1.5 py-0.5 bg-white/10 rounded text-blue-300 text-sm font-mono">$1</code>');

        return line ? (
          <p key={i} className="text-white/80 mb-2" dangerouslySetInnerHTML={{ __html: formatted }} />
        ) : (
          <br key={i} />
        );
      });
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'text-green-400';
    if (confidence >= 0.6) return 'text-yellow-400';
    return 'text-orange-400';
  };

  const getConfidenceLabel = (confidence: number) => {
    if (confidence >= 0.8) return 'High';
    if (confidence >= 0.6) return 'Medium';
    return 'Low';
  };

  if (responses.length === 0) {
    return (
      <GlassCard className="p-8 text-center">
        <div className="text-white/60 space-y-4">
          <svg className="w-16 h-16 mx-auto opacity-50" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
          </svg>
          <p className="text-lg">No responses yet</p>
          <p className="text-sm">Ask a question to get AI-powered answers</p>
        </div>
      </GlassCard>
    );
  }

  return (
    <div className="space-y-4">
      <AnimatePresence>
        {responses.map((response, index) => (
          <motion.div
            key={response.id}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ delay: index * 0.1 }}
          >
            <GlassCard className="p-6">
              {/* Question */}
              <div className="mb-4 pb-4 border-b border-white/10">
                <div className="flex items-start gap-3">
                  <div className="w-8 h-8 rounded-lg bg-blue-500/20 flex items-center justify-center flex-shrink-0">
                    <svg className="w-5 h-5 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8.228 9c.549-1.165 2.03-2 3.772-2 2.21 0 4 1.343 4 3 0 1.4-1.278 2.575-3.006 2.907-.542.104-.994.54-.994 1.093m0 3h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                  </div>
                  <div className="flex-1">
                    <p className="text-white/90 font-medium">{response.question}</p>
                    <p className="text-white/40 text-xs mt-1">
                      {new Date(response.timestamp).toLocaleString()}
                    </p>
                  </div>
                </div>
              </div>

              {/* Answer */}
              <div className="mb-4">
                <div className="flex items-start gap-3">
                  <div className="w-8 h-8 rounded-lg bg-purple-500/20 flex items-center justify-center flex-shrink-0">
                    <svg className="w-5 h-5 text-purple-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                    </svg>
                  </div>
                  <div className="flex-1 prose prose-invert max-w-none">
                    {formatMarkdown(response.answer)}
                  </div>
                </div>
              </div>

              {/* Confidence & Sources */}
              <div className="flex items-center justify-between pt-4 border-t border-white/10">
                <div className="flex items-center gap-2">
                  <span className="text-sm text-white/60">Confidence:</span>
                  <span className={`text-sm font-semibold ${getConfidenceColor(response.confidence)}`}>
                    {getConfidenceLabel(response.confidence)} ({Math.round(response.confidence * 100)}%)
                  </span>
                </div>

                <button
                  onClick={() => toggleSources(response.id)}
                  className="flex items-center gap-2 text-sm text-blue-400 hover:text-blue-300 transition-colors"
                >
                  <span>{response.sources.length} Sources</span>
                  <svg
                    className={`w-4 h-4 transition-transform ${expandedSources.has(response.id) ? 'rotate-180' : ''}`}
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                  </svg>
                </button>
              </div>

              {/* Expanded Sources */}
              <AnimatePresence>
                {expandedSources.has(response.id) && (
                  <motion.div
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: 'auto', opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    transition={{ duration: 0.3 }}
                    className="overflow-hidden"
                  >
                    <div className="mt-4 pt-4 border-t border-white/10 space-y-3">
                      {response.sources.map((source) => (
                        <div
                          key={source.id}
                          className="p-4 rounded-xl bg-white/5 hover:bg-white/10 transition-colors"
                        >
                          <div className="flex items-start justify-between gap-4">
                            <div className="flex-1">
                              <h4 className="text-white font-medium mb-1">{source.title}</h4>
                              <p className="text-white/60 text-sm mb-2">
                                {source.authors.slice(0, 3).join(', ')}
                                {source.authors.length > 3 && ` +${source.authors.length - 3} more`}
                              </p>
                              <p className="text-white/50 text-xs line-clamp-2">{source.abstract}</p>
                              <div className="flex items-center gap-2 mt-2">
                                <span className="text-xs px-2 py-1 rounded-full bg-blue-500/20 text-blue-300">
                                  {source.source}
                                </span>
                                <span className="text-xs text-white/40">{source.published}</span>
                              </div>
                            </div>
                            <a
                              href={source.url}
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
                        </div>
                      ))}
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </GlassCard>
          </motion.div>
        ))}
      </AnimatePresence>
    </div>
  );
};

export default ResponseDisplay;
