import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import GlassCard from '../Common/GlassCard';
import type { Session } from '../../types';

interface SessionManagerProps {
  sessions: Session[];
  currentSessionId?: string;
  onSelectSession: (sessionId: string) => void;
  onCreateSession: (name: string) => void;
  onDeleteSession: (sessionId: string) => void;
  onRenameSession: (sessionId: string, newName: string) => void;
}

const SessionManager: React.FC<SessionManagerProps> = ({
  sessions,
  currentSessionId,
  onSelectSession,
  onCreateSession,
  onDeleteSession,
  onRenameSession,
}) => {
  const [isCreating, setIsCreating] = useState(false);
  const [newSessionName, setNewSessionName] = useState('');
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editName, setEditName] = useState('');
  const [deleteConfirmId, setDeleteConfirmId] = useState<string | null>(null);

  const handleCreateSession = () => {
    if (!newSessionName.trim()) return;
    onCreateSession(newSessionName.trim());
    setNewSessionName('');
    setIsCreating(false);
  };

  const handleRename = (sessionId: string) => {
    if (!editName.trim()) return;
    onRenameSession(sessionId, editName.trim());
    setEditingId(null);
    setEditName('');
  };

  const startEdit = (session: Session) => {
    setEditingId(session.id);
    setEditName(session.name);
  };

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <GlassCard className="p-6">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-bold text-white mb-2">Sessions</h2>
            <p className="text-white/60 text-sm">
              Manage your research sessions
            </p>
          </div>
          <motion.button
            onClick={() => setIsCreating(true)}
            className="px-4 py-2 bg-gradient-to-r from-blue-500 to-purple-600 text-white rounded-xl font-semibold hover:shadow-lg transition-shadow"
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
                      placeholder="e.g., Machine Learning Research 2024"
                      className="
                        w-full px-4 py-3 rounded-xl
                        bg-white/10 border border-white/20
                        text-white placeholder-white/40
                        focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent
                      "
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
      {sessions.length === 0 ? (
        <GlassCard className="p-12 text-center">
          <div className="text-white/60 space-y-4">
            <svg className="w-16 h-16 mx-auto opacity-50" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
            </svg>
            <p className="text-lg">No sessions yet</p>
            <p className="text-sm">Create a new session to start collecting research papers</p>
          </div>
        </GlassCard>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          <AnimatePresence>
            {sessions.map((session, index) => (
              <motion.div
                key={session.id}
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.9 }}
                transition={{ delay: index * 0.05 }}
              >
                <GlassCard
                  className={`
                    p-6 cursor-pointer transition-all
                    ${currentSessionId === session.id
                      ? 'ring-2 ring-blue-500 bg-blue-500/10'
                      : 'hover:bg-white/10'
                    }
                  `}
                  onClick={() => {
                    if (editingId !== session.id && deleteConfirmId !== session.id) {
                      onSelectSession(session.id);
                    }
                  }}
                >
                  {/* Session Header */}
                  <div className="flex items-start justify-between mb-4">
                    <div className="flex-1 min-w-0">
                      {editingId === session.id ? (
                        <input
                          type="text"
                          value={editName}
                          onChange={(e) => setEditName(e.target.value)}
                          onKeyPress={(e) => e.key === 'Enter' && handleRename(session.id)}
                          onBlur={() => handleRename(session.id)}
                          className="w-full px-2 py-1 bg-white/10 border border-white/20 rounded text-white text-lg font-semibold"
                          autoFocus
                          onClick={(e) => e.stopPropagation()}
                        />
                      ) : (
                        <h3 className="text-lg font-semibold text-white truncate">
                          {session.name}
                        </h3>
                      )}
                    </div>

                    {currentSessionId === session.id && (
                      <span className="ml-2 px-2 py-1 bg-blue-500 text-white text-xs rounded-full">
                        Active
                      </span>
                    )}
                  </div>

                  {/* Session Stats */}
                  <div className="space-y-2 mb-4">
                    <div className="flex items-center gap-2 text-white/60 text-sm">
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
                      </svg>
                      {session.paper_count} papers
                    </div>
                    <div className="flex items-center gap-2 text-white/40 text-xs">
                      <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                      </svg>
                      Created {formatDate(session.created_at)}
                    </div>
                    <div className="flex items-center gap-2 text-white/40 text-xs">
                      <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                      </svg>
                      Updated {formatDate(session.updated_at)}
                    </div>
                  </div>

                  {/* Actions */}
                  <div className="flex gap-2 pt-4 border-t border-white/10">
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        startEdit(session);
                      }}
                      className="flex-1 py-2 px-3 bg-white/5 hover:bg-white/10 text-white/70 hover:text-white rounded-lg text-sm transition-colors flex items-center justify-center gap-2"
                    >
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
                      </svg>
                      Rename
                    </button>

                    {deleteConfirmId === session.id ? (
                      <>
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            onDeleteSession(session.id);
                            setDeleteConfirmId(null);
                          }}
                          className="flex-1 py-2 px-3 bg-red-500 hover:bg-red-600 text-white rounded-lg text-sm transition-colors"
                        >
                          Confirm
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
                      </>
                    ) : (
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          setDeleteConfirmId(session.id);
                        }}
                        className="flex-1 py-2 px-3 bg-red-500/20 hover:bg-red-500/30 text-red-300 hover:text-red-200 rounded-lg text-sm transition-colors flex items-center justify-center gap-2"
                      >
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                        </svg>
                        Delete
                      </button>
                    )}
                  </div>
                </GlassCard>
              </motion.div>
            ))}
          </AnimatePresence>
        </div>
      )}
    </div>
  );
};

export default SessionManager;
