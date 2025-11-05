import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import GlassCard from '../components/Common/GlassCard';
import type { Session } from '../types';

interface ExtendedSession extends Session {
  query_count: number;
  duration_minutes: number;
  last_activity?: string;
}

interface SessionsPageProps {
  sessions?: ExtendedSession[];
  currentSessionId?: string;
  onLoadSession?: (sessionId: string) => void;
  onDeleteSession?: (sessionId: string) => void;
  onExportSession?: (sessionId: string) => void;
  onCreateSession?: (name: string) => void;
}

const Sessions: React.FC<SessionsPageProps> = ({
  sessions = [],
  currentSessionId,
  onLoadSession,
  onDeleteSession,
  onExportSession,
  onCreateSession,
}) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [sortBy, setSortBy] = useState<'date' | 'name' | 'queries'>('date');
  const [isCreating, setIsCreating] = useState(false);
  const [newSessionName, setNewSessionName] = useState('');
  const [deleteConfirmId, setDeleteConfirmId] = useState<string | null>(null);
  const [hoveredSessionId, setHoveredSessionId] = useState<string | null>(null);

  // Demo sessions if none provided
  const [demoSessions, setDemoSessions] = useState<ExtendedSession[]>([
    {
      id: '1',
      name: 'Machine Learning Research 2024',
      created_at: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString(),
      updated_at: new Date(Date.now() - 1 * 60 * 60 * 1000).toISOString(),
      paper_count: 47,
      query_count: 23,
      duration_minutes: 145,
      last_activity: '1 hour ago',
    },
    {
      id: '2',
      name: 'Neural Networks Study',
      created_at: new Date(Date.now() - 14 * 24 * 60 * 60 * 1000).toISOString(),
      updated_at: new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString(),
      paper_count: 32,
      query_count: 15,
      duration_minutes: 98,
      last_activity: '1 day ago',
    },
    {
      id: '3',
      name: 'Computer Vision Papers',
      created_at: new Date(Date.now() - 21 * 24 * 60 * 60 * 1000).toISOString(),
      updated_at: new Date(Date.now() - 3 * 24 * 60 * 60 * 1000).toISOString(),
      paper_count: 58,
      query_count: 31,
      duration_minutes: 203,
      last_activity: '3 days ago',
    },
  ]);

  const displaySessions = sessions.length > 0 ? sessions : demoSessions;

  const handleCreateSession = () => {
    if (!newSessionName.trim()) return;

    if (onCreateSession) {
      onCreateSession(newSessionName.trim());
    } else {
      // Demo: Add to demo sessions
      const newSession: ExtendedSession = {
        id: `${Date.now()}`,
        name: newSessionName.trim(),
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
        paper_count: 0,
        query_count: 0,
        duration_minutes: 0,
        last_activity: 'Just now',
      };
      setDemoSessions([newSession, ...demoSessions]);
    }

    setNewSessionName('');
    setIsCreating(false);
  };

  const handleDeleteSession = (sessionId: string) => {
    if (onDeleteSession) {
      onDeleteSession(sessionId);
    } else {
      setDemoSessions(demoSessions.filter(s => s.id !== sessionId));
    }
    setDeleteConfirmId(null);
  };

  const handleLoadSession = (sessionId: string) => {
    if (onLoadSession) {
      onLoadSession(sessionId);
    }
  };

  const handleExportSession = (sessionId: string, sessionName: string) => {
    if (onExportSession) {
      onExportSession(sessionId);
    } else {
      // Demo: Create a download
      const dataStr = JSON.stringify({ sessionId, sessionName, exportDate: new Date() }, null, 2);
      const dataUri = 'data:application/json;charset=utf-8,' + encodeURIComponent(dataStr);
      const exportFileDefaultName = `session-${sessionName.replace(/\s+/g, '-')}.json`;

      const linkElement = document.createElement('a');
      linkElement.setAttribute('href', dataUri);
      linkElement.setAttribute('download', exportFileDefaultName);
      linkElement.click();
    }
  };

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
    });
  };

  const formatDuration = (minutes: number): string => {
    if (minutes < 60) return `${minutes}m`;
    const hours = Math.floor(minutes / 60);
    const mins = minutes % 60;
    return mins > 0 ? `${hours}h ${mins}m` : `${hours}h`;
  };

  const filteredSessions = displaySessions
    .filter(session =>
      session.name.toLowerCase().includes(searchQuery.toLowerCase())
    )
    .sort((a, b) => {
      switch (sortBy) {
        case 'date':
          return new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime();
        case 'name':
          return a.name.localeCompare(b.name);
        case 'queries':
          return b.query_count - a.query_count;
        default:
          return 0;
      }
    });

  const totalStats = {
    totalSessions: displaySessions.length,
    totalPapers: displaySessions.reduce((sum, s) => sum + s.paper_count, 0),
    totalQueries: displaySessions.reduce((sum, s) => sum + s.query_count, 0),
    totalTime: displaySessions.reduce((sum, s) => sum + s.duration_minutes, 0),
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-900 via-blue-900 to-indigo-900 p-6">
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-8"
        >
          <h1 className="text-4xl font-bold text-white mb-3">Research Sessions</h1>
          <p className="text-white/60 text-lg">
            Manage and track your research sessions
          </p>
        </motion.div>

        {/* Statistics Cards */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          {[
            { label: 'Total Sessions', value: totalStats.totalSessions, icon: 'M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10', color: 'from-blue-500 to-cyan-500' },
            { label: 'Total Papers', value: totalStats.totalPapers, icon: 'M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253', color: 'from-purple-500 to-pink-500' },
            { label: 'Total Queries', value: totalStats.totalQueries, icon: 'M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z', color: 'from-green-500 to-emerald-500' },
            { label: 'Research Time', value: formatDuration(totalStats.totalTime), icon: 'M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z', color: 'from-orange-500 to-red-500' },
          ].map((stat, index) => (
            <motion.div
              key={stat.label}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
            >
              <GlassCard className="p-6">
                <div className="flex items-center gap-4">
                  <div className={`p-3 rounded-xl bg-gradient-to-br ${stat.color}`}>
                    <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d={stat.icon} />
                    </svg>
                  </div>
                  <div>
                    <p className="text-2xl font-bold text-white">{stat.value}</p>
                    <p className="text-sm text-white/60">{stat.label}</p>
                  </div>
                </div>
              </GlassCard>
            </motion.div>
          ))}
        </div>

        {/* Controls */}
        <GlassCard className="p-6">
          <div className="flex flex-col md:flex-row gap-4 items-center">
            {/* Search */}
            <div className="flex-1 w-full">
              <div className="relative">
                <svg className="absolute left-4 top-1/2 transform -translate-y-1/2 w-5 h-5 text-white/40" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                </svg>
                <input
                  type="text"
                  placeholder="Search sessions..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="w-full pl-12 pr-4 py-3 rounded-xl bg-white/10 border border-white/20 text-white placeholder-white/40 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                />
              </div>
            </div>

            {/* Sort */}
            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value as 'date' | 'name' | 'queries')}
              className="px-4 py-3 rounded-xl bg-white/10 border border-white/20 text-white focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent cursor-pointer"
            >
              <option value="date" className="bg-gray-800">Sort by Date</option>
              <option value="name" className="bg-gray-800">Sort by Name</option>
              <option value="queries" className="bg-gray-800">Sort by Queries</option>
            </select>

            {/* Create Button */}
            <motion.button
              onClick={() => setIsCreating(true)}
              className="px-6 py-3 bg-gradient-to-r from-blue-500 to-purple-600 text-white rounded-xl font-semibold hover:shadow-lg transition-shadow whitespace-nowrap"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              <span className="flex items-center gap-2">
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                </svg>
                New Session
              </span>
            </motion.button>
          </div>
        </GlassCard>

        {/* Create Session Modal */}
        <AnimatePresence>
          {isCreating && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4"
              onClick={() => setIsCreating(false)}
            >
              <motion.div
                initial={{ scale: 0.9, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                exit={{ scale: 0.9, opacity: 0 }}
                onClick={(e) => e.stopPropagation()}
                className="w-full max-w-md"
              >
                <GlassCard className="p-6">
                  <h3 className="text-xl font-bold text-white mb-4">Create New Session</h3>
                  <div className="space-y-4">
                    <div>
                      <label htmlFor="sessionName" className="block text-sm font-medium text-white/90 mb-2">
                        Session Name
                      </label>
                      <input
                        id="sessionName"
                        type="text"
                        value={newSessionName}
                        onChange={(e) => setNewSessionName(e.target.value)}
                        onKeyPress={(e) => e.key === 'Enter' && handleCreateSession()}
                        placeholder="e.g., Deep Learning Research 2024"
                        className="w-full px-4 py-3 rounded-xl bg-white/10 border border-white/20 text-white placeholder-white/40 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                        autoFocus
                      />
                    </div>
                    <div className="flex gap-3">
                      <button
                        onClick={handleCreateSession}
                        disabled={!newSessionName.trim()}
                        className="flex-1 py-3 bg-gradient-to-r from-blue-500 to-purple-600 text-white rounded-xl font-semibold disabled:opacity-50 disabled:cursor-not-allowed"
                      >
                        Create
                      </button>
                      <button
                        onClick={() => {
                          setIsCreating(false);
                          setNewSessionName('');
                        }}
                        className="px-6 py-3 bg-white/10 hover:bg-white/20 text-white rounded-xl font-semibold transition-colors"
                      >
                        Cancel
                      </button>
                    </div>
                  </div>
                </GlassCard>
              </motion.div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Sessions Grid */}
        {filteredSessions.length === 0 ? (
          <GlassCard className="p-12 text-center">
            <div className="text-white/60 space-y-4">
              <svg className="w-16 h-16 mx-auto opacity-50" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
              </svg>
              <p className="text-lg">No sessions found</p>
              <p className="text-sm">
                {searchQuery ? 'Try a different search term' : 'Create a new session to get started'}
              </p>
            </div>
          </GlassCard>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            <AnimatePresence>
              {filteredSessions.map((session, index) => {
                const isActive = currentSessionId === session.id || (currentSessionId === undefined && session.id === '1');
                const isHovered = hoveredSessionId === session.id;

                return (
                  <motion.div
                    key={session.id}
                    initial={{ opacity: 0, scale: 0.9 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0, scale: 0.9 }}
                    transition={{ delay: index * 0.05 }}
                    onHoverStart={() => setHoveredSessionId(session.id)}
                    onHoverEnd={() => setHoveredSessionId(null)}
                  >
                    <GlassCard
                      className={`
                        p-6 cursor-pointer transition-all relative overflow-hidden
                        ${isActive
                          ? 'ring-2 ring-blue-500 bg-blue-500/10'
                          : 'hover:bg-white/10'
                        }
                      `}
                      onClick={() => handleLoadSession(session.id)}
                    >
                      {/* Animated glow for active session */}
                      {isActive && (
                        <motion.div
                          className="absolute inset-0 bg-gradient-to-r from-blue-500/20 to-purple-500/20"
                          animate={{
                            opacity: [0.3, 0.6, 0.3],
                          }}
                          transition={{
                            duration: 2,
                            repeat: Infinity,
                            ease: "easeInOut"
                          }}
                        />
                      )}

                      <div className="relative z-10">
                        {/* Session Header */}
                        <div className="flex items-start justify-between mb-4">
                          <h3 className="text-lg font-semibold text-white truncate flex-1 pr-2">
                            {session.name}
                          </h3>
                          {isActive && (
                            <motion.span
                              initial={{ scale: 0 }}
                              animate={{ scale: 1 }}
                              className="px-3 py-1 bg-blue-500 text-white text-xs rounded-full font-semibold flex items-center gap-1"
                            >
                              <span className="w-2 h-2 bg-white rounded-full animate-pulse" />
                              Active
                            </motion.span>
                          )}
                        </div>

                        {/* Session Info */}
                        <div className="space-y-3 mb-4">
                          <div className="grid grid-cols-2 gap-3">
                            <div className="p-3 rounded-lg bg-white/5 border border-white/10">
                              <div className="flex items-center gap-2 text-white/60 text-xs mb-1">
                                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
                                </svg>
                                Papers
                              </div>
                              <p className="text-xl font-bold text-white">{session.paper_count}</p>
                            </div>
                            <div className="p-3 rounded-lg bg-white/5 border border-white/10">
                              <div className="flex items-center gap-2 text-white/60 text-xs mb-1">
                                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
                                </svg>
                                Queries
                              </div>
                              <p className="text-xl font-bold text-white">{session.query_count}</p>
                            </div>
                          </div>

                          <div className="flex items-center justify-between text-white/60 text-xs">
                            <div className="flex items-center gap-1">
                              <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                              </svg>
                              {formatDuration(session.duration_minutes)}
                            </div>
                            <div className="flex items-center gap-1">
                              <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
                              </svg>
                              {formatDate(session.created_at)}
                            </div>
                          </div>

                          {session.last_activity && (
                            <div className="text-white/40 text-xs">
                              Last active: {session.last_activity}
                            </div>
                          )}
                        </div>

                        {/* Actions */}
                        <AnimatePresence>
                          {(isHovered || isActive) && (
                            <motion.div
                              initial={{ opacity: 0, height: 0 }}
                              animate={{ opacity: 1, height: 'auto' }}
                              exit={{ opacity: 0, height: 0 }}
                              className="pt-4 border-t border-white/10"
                            >
                              {deleteConfirmId === session.id ? (
                                <div className="flex gap-2">
                                  <button
                                    onClick={(e) => {
                                      e.stopPropagation();
                                      handleDeleteSession(session.id);
                                    }}
                                    className="flex-1 py-2 px-3 bg-red-500 hover:bg-red-600 text-white rounded-lg text-sm transition-colors font-semibold"
                                  >
                                    Confirm Delete
                                  </button>
                                  <button
                                    onClick={(e) => {
                                      e.stopPropagation();
                                      setDeleteConfirmId(null);
                                    }}
                                    className="flex-1 py-2 px-3 bg-white/10 hover:bg-white/20 text-white rounded-lg text-sm transition-colors"
                                  >
                                    Cancel
                                  </button>
                                </div>
                              ) : (
                                <div className="flex gap-2">
                                  <button
                                    onClick={(e) => {
                                      e.stopPropagation();
                                      handleLoadSession(session.id);
                                    }}
                                    className="flex-1 py-2 px-3 bg-gradient-to-r from-blue-500 to-purple-600 hover:shadow-lg text-white rounded-lg text-sm transition-all flex items-center justify-center gap-1 font-semibold"
                                  >
                                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                                    </svg>
                                    Load
                                  </button>
                                  <button
                                    onClick={(e) => {
                                      e.stopPropagation();
                                      handleExportSession(session.id, session.name);
                                    }}
                                    className="py-2 px-3 bg-white/5 hover:bg-white/10 text-white/70 hover:text-white rounded-lg text-sm transition-colors flex items-center justify-center"
                                    title="Export"
                                  >
                                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
                                    </svg>
                                  </button>
                                  <button
                                    onClick={(e) => {
                                      e.stopPropagation();
                                      setDeleteConfirmId(session.id);
                                    }}
                                    className="py-2 px-3 bg-red-500/20 hover:bg-red-500/30 text-red-300 hover:text-red-200 rounded-lg text-sm transition-colors flex items-center justify-center"
                                    title="Delete"
                                  >
                                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                                    </svg>
                                  </button>
                                </div>
                              )}
                            </motion.div>
                          )}
                        </AnimatePresence>
                      </div>
                    </GlassCard>
                  </motion.div>
                );
              })}
            </AnimatePresence>
          </div>
        )}
      </div>
    </div>
  );
};

export default Sessions;
