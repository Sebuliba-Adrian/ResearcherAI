import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import GlassCard from '../components/Common/GlassCard';

// Types
interface VectorSearchResult {
  id: string;
  title: string;
  content: string;
  similarity: number;
  metadata: {
    source: string;
    date: string;
    author: string;
    category: string;
  };
  embedding?: number[];
}

interface EmbeddingPoint {
  x: number;
  y: number;
  id: string;
  isQuery?: boolean;
  isTopResult?: boolean;
  similarity?: number;
}

interface SearchStats {
  totalVectors: number;
  averageSimilarity: number;
  searchTime: number;
  resultsCount: number;
}

const Vector: React.FC = () => {
  // State management
  const [query, setQuery] = useState('');
  const [topK, setTopK] = useState(10);
  const [similarityThreshold, setSimilarityThreshold] = useState(0.5);
  const [isSearching, setIsSearching] = useState(false);
  const [results, setResults] = useState<VectorSearchResult[]>([]);
  const [stats, setStats] = useState<SearchStats>({
    totalVectors: 15234,
    averageSimilarity: 0,
    searchTime: 0,
    resultsCount: 0,
  });
  const [showVisualization, setShowVisualization] = useState(false);
  const [embeddingPoints, setEmbeddingPoints] = useState<EmbeddingPoint[]>([]);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Generate mock embedding visualization data
  const generateEmbeddingPoints = (resultsCount: number): EmbeddingPoint[] => {
    const points: EmbeddingPoint[] = [];

    // Query point (center)
    points.push({
      x: 250,
      y: 250,
      id: 'query',
      isQuery: true,
    });

    // Top results (close to query)
    for (let i = 0; i < Math.min(resultsCount, topK); i++) {
      const angle = (i / resultsCount) * Math.PI * 2;
      const distance = 50 + Math.random() * 100;
      points.push({
        x: 250 + Math.cos(angle) * distance,
        y: 250 + Math.sin(angle) * distance,
        id: `result-${i}`,
        isTopResult: true,
        similarity: 1 - (distance / 150),
      });
    }

    // Other vectors (scattered)
    for (let i = 0; i < 50; i++) {
      points.push({
        x: Math.random() * 500,
        y: Math.random() * 500,
        id: `vector-${i}`,
      });
    }

    return points;
  };

  // Draw embedding visualization on canvas
  useEffect(() => {
    if (!showVisualization || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw grid
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 500; i += 50) {
      ctx.beginPath();
      ctx.moveTo(i, 0);
      ctx.lineTo(i, 500);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(0, i);
      ctx.lineTo(500, i);
      ctx.stroke();
    }

    // Draw points
    embeddingPoints.forEach((point) => {
      ctx.beginPath();

      if (point.isQuery) {
        // Query point - large purple glow
        ctx.arc(point.x, point.y, 12, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(168, 85, 247, 0.8)';
        ctx.fill();
        ctx.strokeStyle = 'rgba(168, 85, 247, 1)';
        ctx.lineWidth = 3;
        ctx.stroke();

        // Glow effect
        const gradient = ctx.createRadialGradient(point.x, point.y, 0, point.x, point.y, 30);
        gradient.addColorStop(0, 'rgba(168, 85, 247, 0.3)');
        gradient.addColorStop(1, 'rgba(168, 85, 247, 0)');
        ctx.fillStyle = gradient;
        ctx.arc(point.x, point.y, 30, 0, Math.PI * 2);
        ctx.fill();
      } else if (point.isTopResult) {
        // Top result points - connected to query
        ctx.strokeStyle = 'rgba(236, 72, 153, 0.3)';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(250, 250);
        ctx.lineTo(point.x, point.y);
        ctx.stroke();

        ctx.beginPath();
        ctx.arc(point.x, point.y, 8, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(236, 72, 153, 0.7)';
        ctx.fill();
        ctx.strokeStyle = 'rgba(236, 72, 153, 1)';
        ctx.lineWidth = 2;
        ctx.stroke();
      } else {
        // Other points - small gray
        ctx.arc(point.x, point.y, 3, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(255, 255, 255, 0.2)';
        ctx.fill();
      }
    });
  }, [embeddingPoints, showVisualization]);

  // Handle search
  const handleSearch = async () => {
    if (!query.trim()) return;

    setIsSearching(true);
    const startTime = Date.now();

    // Simulate API call
    await new Promise((resolve) => setTimeout(resolve, 1500));

    // Generate mock results
    const mockResults: VectorSearchResult[] = Array.from({ length: topK }, (_, i) => ({
      id: `result-${i}`,
      title: `Research Paper ${i + 1}: ${query.slice(0, 30)}...`,
      content: `This is a highly relevant excerpt from the document that matches your query about "${query}". The semantic similarity algorithm has identified this content as closely related to your search intent. Advanced natural language processing techniques were used to compute vector embeddings and measure cosine similarity in high-dimensional space.`,
      similarity: Math.max(similarityThreshold, 0.95 - i * 0.05 - Math.random() * 0.1),
      metadata: {
        source: ['ArXiv', 'PubMed', 'IEEE', 'Nature', 'Science'][Math.floor(Math.random() * 5)],
        date: new Date(2023 + Math.random(), Math.floor(Math.random() * 12), Math.floor(Math.random() * 28)).toLocaleDateString(),
        author: ['Dr. Smith', 'Prof. Johnson', 'Dr. Chen', 'Prof. Williams', 'Dr. Kumar'][Math.floor(Math.random() * 5)],
        category: ['Machine Learning', 'Neural Networks', 'NLP', 'Computer Vision', 'AI Ethics'][Math.floor(Math.random() * 5)],
      },
    })).filter(r => r.similarity >= similarityThreshold);

    const searchTime = Date.now() - startTime;
    const avgSimilarity = mockResults.reduce((acc, r) => acc + r.similarity, 0) / mockResults.length;

    setResults(mockResults);
    setStats({
      totalVectors: 15234 + Math.floor(Math.random() * 1000),
      averageSimilarity: avgSimilarity || 0,
      searchTime,
      resultsCount: mockResults.length,
    });

    // Generate visualization data
    setEmbeddingPoints(generateEmbeddingPoints(mockResults.length));
    setShowVisualization(true);
    setIsSearching(false);
  };

  // Get similarity color
  const getSimilarityColor = (similarity: number): string => {
    if (similarity >= 0.9) return 'from-green-500 to-emerald-600';
    if (similarity >= 0.8) return 'from-blue-500 to-cyan-600';
    if (similarity >= 0.7) return 'from-yellow-500 to-orange-600';
    return 'from-orange-500 to-red-600';
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 p-8">
      <div className="max-w-7xl mx-auto space-y-8">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center"
        >
          <h1 className="text-5xl font-bold text-white mb-4">
            Vector Search
            <span className="bg-gradient-to-r from-purple-400 to-pink-600 bg-clip-text text-transparent">
              {' '}Intelligence
            </span>
          </h1>
          <p className="text-white/60 text-lg">
            Semantic search powered by advanced embedding models
          </p>
        </motion.div>

        {/* Search Interface */}
        <GlassCard className="p-8">
          <div className="space-y-6">
            {/* Query Textarea */}
            <div>
              <label className="block text-white/90 font-semibold mb-3 text-sm uppercase tracking-wider">
                Semantic Query
              </label>
              <div className="relative">
                <textarea
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder="Enter your semantic search query... Describe concepts, ideas, or questions in natural language."
                  rows={4}
                  className="
                    w-full px-6 py-4 rounded-2xl
                    bg-white/5 backdrop-blur-xl
                    border-2 border-white/10
                    text-white placeholder-white/30
                    focus:outline-none focus:border-purple-500/50 focus:ring-4 focus:ring-purple-500/20
                    transition-all duration-300
                    resize-none
                  "
                  disabled={isSearching}
                />
                <motion.div
                  className="absolute bottom-4 right-4"
                  animate={{ opacity: [0.5, 1, 0.5] }}
                  transition={{ duration: 2, repeat: Infinity }}
                >
                  <svg className="w-6 h-6 text-purple-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                  </svg>
                </motion.div>
              </div>
            </div>

            {/* Controls Row */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Top-K Slider */}
              <div>
                <label className="flex items-center justify-between text-white/90 font-semibold mb-3 text-sm">
                  <span>Top-K Results</span>
                  <span className="text-purple-400 text-lg font-bold">{topK}</span>
                </label>
                <input
                  type="range"
                  min="5"
                  max="50"
                  step="5"
                  value={topK}
                  onChange={(e) => setTopK(Number(e.target.value))}
                  disabled={isSearching}
                  className="
                    w-full h-3 rounded-full appearance-none cursor-pointer
                    bg-white/10 backdrop-blur-xl
                    [&::-webkit-slider-thumb]:appearance-none
                    [&::-webkit-slider-thumb]:w-6
                    [&::-webkit-slider-thumb]:h-6
                    [&::-webkit-slider-thumb]:rounded-full
                    [&::-webkit-slider-thumb]:bg-gradient-to-r
                    [&::-webkit-slider-thumb]:from-purple-500
                    [&::-webkit-slider-thumb]:to-pink-600
                    [&::-webkit-slider-thumb]:cursor-pointer
                    [&::-webkit-slider-thumb]:shadow-lg
                    [&::-webkit-slider-thumb]:shadow-purple-500/50
                    [&::-webkit-slider-thumb]:transition-transform
                    [&::-webkit-slider-thumb]:hover:scale-110
                  "
                />
                <div className="flex justify-between text-white/40 text-xs mt-2">
                  <span>5</span>
                  <span>50</span>
                </div>
              </div>

              {/* Similarity Threshold Slider */}
              <div>
                <label className="flex items-center justify-between text-white/90 font-semibold mb-3 text-sm">
                  <span>Similarity Threshold</span>
                  <span className="text-pink-400 text-lg font-bold">{similarityThreshold.toFixed(2)}</span>
                </label>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.05"
                  value={similarityThreshold}
                  onChange={(e) => setSimilarityThreshold(Number(e.target.value))}
                  disabled={isSearching}
                  className="
                    w-full h-3 rounded-full appearance-none cursor-pointer
                    bg-white/10 backdrop-blur-xl
                    [&::-webkit-slider-thumb]:appearance-none
                    [&::-webkit-slider-thumb]:w-6
                    [&::-webkit-slider-thumb]:h-6
                    [&::-webkit-slider-thumb]:rounded-full
                    [&::-webkit-slider-thumb]:bg-gradient-to-r
                    [&::-webkit-slider-thumb]:from-pink-500
                    [&::-webkit-slider-thumb]:to-rose-600
                    [&::-webkit-slider-thumb]:cursor-pointer
                    [&::-webkit-slider-thumb]:shadow-lg
                    [&::-webkit-slider-thumb]:shadow-pink-500/50
                    [&::-webkit-slider-thumb]:transition-transform
                    [&::-webkit-slider-thumb]:hover:scale-110
                  "
                />
                <div className="flex justify-between text-white/40 text-xs mt-2">
                  <span>0.00</span>
                  <span>1.00</span>
                </div>
              </div>
            </div>

            {/* Search Button */}
            <motion.button
              onClick={handleSearch}
              disabled={isSearching || !query.trim()}
              className="
                w-full py-4 rounded-2xl
                bg-gradient-to-r from-purple-500 via-pink-500 to-rose-500
                text-white font-bold text-lg
                disabled:opacity-50 disabled:cursor-not-allowed
                relative overflow-hidden
                shadow-xl shadow-purple-500/30
              "
              whileHover={!isSearching ? { scale: 1.02 } : {}}
              whileTap={!isSearching ? { scale: 0.98 } : {}}
            >
              {isSearching ? (
                <div className="flex items-center justify-center gap-3">
                  <motion.div
                    className="w-6 h-6 border-3 border-white/30 border-t-white rounded-full"
                    animate={{ rotate: 360 }}
                    transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
                  />
                  <span>Searching Vector Space...</span>
                </div>
              ) : (
                <span className="flex items-center justify-center gap-3">
                  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                  </svg>
                  Search
                </span>
              )}
              {/* Animated gradient overlay */}
              {isSearching && (
                <motion.div
                  className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent"
                  animate={{ x: ['-100%', '100%'] }}
                  transition={{ duration: 1.5, repeat: Infinity, ease: 'linear' }}
                />
              )}
            </motion.button>
          </div>
        </GlassCard>

        {/* Stats Panel */}
        {results.length > 0 && (
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="grid grid-cols-2 md:grid-cols-4 gap-4"
          >
            {[
              { label: 'Total Vectors', value: stats.totalVectors.toLocaleString(), icon: '🗄️', color: 'blue' },
              { label: 'Avg Similarity', value: `${(stats.averageSimilarity * 100).toFixed(1)}%`, icon: '🎯', color: 'green' },
              { label: 'Search Time', value: `${stats.searchTime}ms`, icon: '⚡', color: 'yellow' },
              { label: 'Results Found', value: stats.resultsCount.toString(), icon: '📊', color: 'purple' },
            ].map((stat, index) => (
              <GlassCard key={index} className="p-6 text-center">
                <motion.div
                  initial={{ scale: 0 }}
                  animate={{ scale: 1 }}
                  transition={{ delay: index * 0.1, type: 'spring' }}
                >
                  <div className="text-3xl mb-2">{stat.icon}</div>
                  <div className={`text-2xl font-bold text-${stat.color}-400 mb-1`}>
                    {stat.value}
                  </div>
                  <div className="text-white/60 text-sm">{stat.label}</div>
                </motion.div>
              </GlassCard>
            ))}
          </motion.div>
        )}

        {/* Visualization Section */}
        {showVisualization && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
          >
            <GlassCard className="p-8">
              <div className="flex items-center justify-between mb-6">
                <div>
                  <h2 className="text-2xl font-bold text-white mb-2">
                    Embedding Visualization
                  </h2>
                  <p className="text-white/60 text-sm">
                    2D projection of vector embeddings using t-SNE
                  </p>
                </div>
                <div className="flex gap-4 text-sm">
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded-full bg-purple-500" />
                    <span className="text-white/70">Query</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded-full bg-pink-500" />
                    <span className="text-white/70">Top Results</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded-full bg-white/20" />
                    <span className="text-white/70">Other Vectors</span>
                  </div>
                </div>
              </div>

              <div className="relative bg-black/30 rounded-2xl p-4 backdrop-blur-xl border border-white/10">
                <canvas
                  ref={canvasRef}
                  width={500}
                  height={500}
                  className="w-full h-auto max-h-[500px] rounded-xl"
                />

                {/* Glass overlay controls */}
                <div className="absolute top-6 right-6 space-y-2">
                  <GlassCard className="p-3">
                    <button className="flex items-center gap-2 text-white/70 hover:text-white transition-colors text-sm">
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                      </svg>
                      Zoom In
                    </button>
                  </GlassCard>
                  <GlassCard className="p-3">
                    <button className="flex items-center gap-2 text-white/70 hover:text-white transition-colors text-sm">
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                      </svg>
                      Reset View
                    </button>
                  </GlassCard>
                </div>
              </div>
            </GlassCard>
          </motion.div>
        )}

        {/* Results Display */}
        <AnimatePresence mode="wait">
          {results.length > 0 && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="space-y-6"
            >
              <div className="flex items-center justify-between">
                <h2 className="text-2xl font-bold text-white">
                  Search Results ({results.length})
                </h2>
                <div className="flex gap-2">
                  <button className="px-4 py-2 rounded-xl bg-white/10 hover:bg-white/20 text-white/70 hover:text-white transition-all text-sm">
                    Sort by Relevance
                  </button>
                  <button className="px-4 py-2 rounded-xl bg-white/10 hover:bg-white/20 text-white/70 hover:text-white transition-all text-sm">
                    Export Results
                  </button>
                </div>
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {results.map((result, index) => (
                  <motion.div
                    key={result.id}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: index * 0.05 }}
                  >
                    <GlassCard
                      hover
                      className="p-6 h-full hover:shadow-2xl hover:shadow-purple-500/20 transition-all duration-300"
                    >
                      <div className="space-y-4">
                        {/* Header with Similarity Score */}
                        <div className="flex items-start justify-between gap-4">
                          <h3 className="text-white font-bold text-lg flex-1 line-clamp-2">
                            {result.title}
                          </h3>
                          <div className="flex-shrink-0">
                            <div className={`
                              px-4 py-2 rounded-xl
                              bg-gradient-to-r ${getSimilarityColor(result.similarity)}
                              text-white font-bold text-sm
                              shadow-lg
                            `}>
                              {(result.similarity * 100).toFixed(1)}%
                            </div>
                          </div>
                        </div>

                        {/* Animated Progress Bar */}
                        <div className="relative h-2 bg-white/10 rounded-full overflow-hidden">
                          <motion.div
                            className={`h-full bg-gradient-to-r ${getSimilarityColor(result.similarity)}`}
                            initial={{ width: 0 }}
                            animate={{ width: `${result.similarity * 100}%` }}
                            transition={{ duration: 1, delay: index * 0.05, ease: 'easeOut' }}
                          />
                          <motion.div
                            className="absolute inset-0 bg-gradient-to-r from-transparent via-white/30 to-transparent"
                            animate={{ x: ['-100%', '100%'] }}
                            transition={{ duration: 1.5, repeat: Infinity, ease: 'linear' }}
                          />
                        </div>

                        {/* Content Snippet */}
                        <div className="p-4 bg-black/20 rounded-xl border border-white/10">
                          <p className="text-white/80 text-sm leading-relaxed line-clamp-4">
                            {result.content}
                          </p>
                        </div>

                        {/* Metadata */}
                        <div className="grid grid-cols-2 gap-3">
                          <div className="flex items-center gap-2 text-white/60 text-sm">
                            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
                            </svg>
                            {result.metadata.source}
                          </div>
                          <div className="flex items-center gap-2 text-white/60 text-sm">
                            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
                            </svg>
                            {result.metadata.date}
                          </div>
                          <div className="flex items-center gap-2 text-white/60 text-sm">
                            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                            </svg>
                            {result.metadata.author}
                          </div>
                          <div className="flex items-center gap-2 text-white/60 text-sm">
                            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 7h.01M7 3h5c.512 0 1.024.195 1.414.586l7 7a2 2 0 010 2.828l-7 7a2 2 0 01-2.828 0l-7-7A1.994 1.994 0 013 12V7a4 4 0 014-4z" />
                            </svg>
                            {result.metadata.category}
                          </div>
                        </div>

                        {/* Action Buttons */}
                        <div className="flex gap-2 pt-2">
                          <motion.button
                            className="flex-1 py-2 rounded-xl bg-purple-500/20 hover:bg-purple-500/30 text-purple-300 text-sm font-medium transition-colors"
                            whileHover={{ scale: 1.02 }}
                            whileTap={{ scale: 0.98 }}
                          >
                            View Details
                          </motion.button>
                          <motion.button
                            className="px-4 py-2 rounded-xl bg-white/10 hover:bg-white/20 text-white/70 hover:text-white transition-colors"
                            whileHover={{ scale: 1.05 }}
                            whileTap={{ scale: 0.95 }}
                          >
                            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 5a2 2 0 012-2h10a2 2 0 012 2v16l-7-3.5L5 21V5z" />
                            </svg>
                          </motion.button>
                        </div>
                      </div>
                    </GlassCard>
                  </motion.div>
                ))}
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Empty State */}
        {!isSearching && results.length === 0 && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="text-center py-20"
          >
            <GlassCard className="p-12">
              <motion.div
                animate={{ y: [0, -10, 0] }}
                transition={{ duration: 3, repeat: Infinity, ease: 'easeInOut' }}
              >
                <svg className="w-24 h-24 mx-auto text-white/20 mb-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                </svg>
              </motion.div>
              <h3 className="text-2xl font-bold text-white mb-3">
                Start Your Semantic Search
              </h3>
              <p className="text-white/60 max-w-md mx-auto">
                Enter a query above to search through our vector database using advanced semantic similarity matching.
              </p>
            </GlassCard>
          </motion.div>
        )}
      </div>
    </div>
  );
};

export default Vector;
