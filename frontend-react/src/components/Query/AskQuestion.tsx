import React, { useState } from 'react';
import { motion } from 'framer-motion';
import GlassCard from '../Common/GlassCard';
import LoadingSpinner from '../Common/LoadingSpinner';

interface AskQuestionProps {
  onSubmit: (question: string) => Promise<void>;
  isLoading?: boolean;
}

const AskQuestion: React.FC<AskQuestionProps> = ({ onSubmit, isLoading = false }) => {
  const [question, setQuestion] = useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!question.trim() || isLoading) return;

    await onSubmit(question.trim());
    setQuestion('');
  };

  const exampleQuestions = [
    'What are the main findings in machine learning research?',
    'Summarize recent advances in quantum computing',
    'What are the challenges in climate modeling?',
    'Compare different approaches to neural network optimization',
    'What are the ethical implications discussed in AI papers?',
  ];

  const handleExampleClick = (example: string) => {
    setQuestion(example);
  };

  return (
    <GlassCard className="p-6 md:p-8">
      <div className="space-y-6">
        {/* Header */}
        <div>
          <h2 className="text-2xl font-bold text-white mb-2">
            Ask a Question
          </h2>
          <p className="text-white/60 text-sm">
            Query your research collection using natural language
          </p>
        </div>

        {/* Question Form */}
        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="relative">
            <textarea
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="What would you like to know about your research papers?"
              rows={4}
              className="
                w-full px-4 py-3 pr-12 rounded-xl
                bg-white/10 border border-white/20
                text-white placeholder-white/40
                focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent
                transition-all resize-none
              "
              disabled={isLoading}
            />

            {/* Character Count */}
            <div className="absolute bottom-3 right-3 text-xs text-white/40">
              {question.length} / 1000
            </div>
          </div>

          {/* Submit Button */}
          <motion.button
            type="submit"
            disabled={isLoading || !question.trim()}
            className="
              w-full py-3 rounded-xl font-semibold
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
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8.228 9c.549-1.165 2.03-2 3.772-2 2.21 0 4 1.343 4 3 0 1.4-1.278 2.575-3.006 2.907-.542.104-.994.54-.994 1.093m0 3h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                Ask Question
              </span>
            )}
          </motion.button>
        </form>

        {/* Example Questions */}
        <div className="space-y-3">
          <p className="text-sm font-medium text-white/70">Example Questions:</p>
          <div className="space-y-2">
            {exampleQuestions.map((example, index) => (
              <motion.button
                key={index}
                type="button"
                onClick={() => handleExampleClick(example)}
                className="
                  w-full text-left px-4 py-3 rounded-xl
                  bg-white/5 hover:bg-white/10
                  border border-white/10 hover:border-blue-500/50
                  text-white/70 hover:text-white
                  transition-all text-sm
                "
                whileHover={{ x: 4 }}
                disabled={isLoading}
              >
                <span className="flex items-center gap-2">
                  <svg className="w-4 h-4 text-blue-400 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                  </svg>
                  {example}
                </span>
              </motion.button>
            ))}
          </div>
        </div>

        {/* Info Box */}
        <div className="flex items-start gap-3 p-4 rounded-xl bg-purple-500/10 border border-purple-500/20">
          <svg className="w-5 h-5 text-purple-400 flex-shrink-0 mt-0.5" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
          </svg>
          <div className="text-sm text-purple-200 space-y-1">
            <p className="font-medium">AI-Powered Answers</p>
            <p className="text-purple-200/80">
              Questions are answered using RAG (Retrieval-Augmented Generation) with your collected papers.
              The AI will cite specific sources in its response.
            </p>
          </div>
        </div>

        {/* Features */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
          {[
            {
              icon: (
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                </svg>
              ),
              title: 'Fast',
              description: 'Quick AI responses',
            },
            {
              icon: (
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              ),
              title: 'Accurate',
              description: 'Source-cited answers',
            },
            {
              icon: (
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
                </svg>
              ),
              title: 'Contextual',
              description: 'Based on your papers',
            },
          ].map((feature, index) => (
            <div
              key={index}
              className="flex items-center gap-3 p-3 rounded-xl bg-white/5 border border-white/10"
            >
              <div className="text-blue-400">{feature.icon}</div>
              <div>
                <p className="text-sm font-medium text-white">{feature.title}</p>
                <p className="text-xs text-white/60">{feature.description}</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </GlassCard>
  );
};

export default AskQuestion;
