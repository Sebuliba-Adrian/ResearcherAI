import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Send,
  Sparkles,
  Brain,
  User,
  FileText,
  Network,
  Search,
  Trash2,
  Settings2,
  ChevronDown,
  Loader2,
  MessageSquare,
  XCircle,
} from 'lucide-react';

// Types
interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  sources?: SourceDocument[];
  graphNodes?: GraphReference[];
  vectorResults?: VectorResult[];
}

interface SourceDocument {
  id: string;
  title: string;
  relevance: number;
  type: 'paper' | 'document' | 'note';
}

interface GraphReference {
  id: string;
  label: string;
  type: 'paper' | 'concept' | 'author';
  connections: number;
}

interface VectorResult {
  id: string;
  content: string;
  similarity: number;
  source: string;
}

interface ModelConfig {
  model: string;
  temperature: number;
  maxTokens: number;
  topP: number;
}

const GEMINI_MODELS = [
  { id: 'gemini-1.5-pro', name: 'Gemini 1.5 Pro', description: 'Most capable model' },
  { id: 'gemini-1.5-flash', name: 'Gemini 1.5 Flash', description: 'Fast and efficient' },
  { id: 'gemini-1.0-pro', name: 'Gemini 1.0 Pro', description: 'Stable and reliable' },
];

const Ask: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [showSidebar, setShowSidebar] = useState(true);
  const [selectedMessage, setSelectedMessage] = useState<Message | null>(null);

  const [config, setConfig] = useState<ModelConfig>({
    model: 'gemini-1.5-pro',
    temperature: 0.7,
    maxTokens: 2048,
    topP: 0.95,
  });

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`;
    }
  }, [input]);

  const handleSend = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: input,
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      // Simulate API call
      await new Promise((resolve) => setTimeout(resolve, 2000));

      const aiMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: `This is a simulated response to: "${input}". In a real implementation, this would connect to your backend API to query the Gemini model with the configured parameters.`,
        timestamp: new Date(),
        sources: [
          { id: '1', title: 'Research Paper on AI', relevance: 0.95, type: 'paper' },
          { id: '2', title: 'Machine Learning Overview', relevance: 0.87, type: 'document' },
          { id: '3', title: 'Neural Networks Study', relevance: 0.82, type: 'paper' },
        ],
        graphNodes: [
          { id: 'n1', label: 'Deep Learning', type: 'concept', connections: 12 },
          { id: 'n2', label: 'Transformer Models', type: 'concept', connections: 8 },
        ],
        vectorResults: [
          {
            id: 'v1',
            content: 'Relevant context from vector database...',
            similarity: 0.92,
            source: 'paper_123.pdf',
          },
          {
            id: 'v2',
            content: 'Another relevant passage...',
            similarity: 0.88,
            source: 'document_456.txt',
          },
        ],
      };

      setMessages((prev) => [...prev, aiMessage]);
      setSelectedMessage(aiMessage);
    } catch (error) {
      console.error('Error sending message:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const clearConversation = () => {
    if (window.confirm('Are you sure you want to clear the conversation?')) {
      setMessages([]);
      setSelectedMessage(null);
    }
  };

  const lastFiveTurns = Math.min(Math.ceil(messages.length / 2), 5);

  return (
    <div className="min-h-screen bg-slate-950 relative overflow-hidden">
      {/* Animated background */}
      <div className="animated-background">
        <div className="orb orb-1" />
        <div className="orb orb-2" />
        <div className="orb orb-3" />
        <div className="orb orb-4" />
      </div>
      <div className="grid-pattern" />

      <div className="relative z-10 h-screen flex flex-col">
        {/* Header */}
        <motion.header
          initial={{ y: -100, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          className="backdrop-blur-xl bg-black/20 border-b border-white/10 px-6 py-4"
        >
          <div className="max-w-screen-2xl mx-auto flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="p-3 rounded-2xl bg-gradient-to-br from-cyber-500 to-neon-purple shadow-neon-lg">
                <Brain className="w-6 h-6" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-neon">Ask AI</h1>
                <p className="text-sm text-gray-400">Query your research knowledge base</p>
              </div>
            </div>

            <div className="flex items-center gap-4">
              {/* Conversation memory indicator */}
              <motion.div
                whileHover={{ scale: 1.05 }}
                className="backdrop-blur-xl bg-white/10 border border-white/20 rounded-xl px-4 py-2 flex items-center gap-2"
              >
                <MessageSquare className="w-4 h-4 text-neon-purple" />
                <span className="text-sm text-gray-300">
                  Last <span className="font-bold text-neon-purple">{lastFiveTurns}</span> turns stored
                </span>
              </motion.div>

              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={clearConversation}
                className="backdrop-blur-xl bg-white/10 border border-white/20 rounded-xl px-4 py-2 flex items-center gap-2 hover:bg-red-500/20 hover:border-red-500/50 transition-all"
              >
                <Trash2 className="w-4 h-4" />
                <span className="text-sm">Clear</span>
              </motion.button>

              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={() => setShowSettings(!showSettings)}
                className="backdrop-blur-xl bg-white/10 border border-white/20 rounded-xl px-4 py-2 flex items-center gap-2 hover:bg-white/20 transition-all"
              >
                <Settings2 className={`w-4 h-4 ${showSettings ? 'animate-spin-slow' : ''}`} />
                <span className="text-sm">Settings</span>
              </motion.button>
            </div>
          </div>
        </motion.header>

        {/* Main content area */}
        <div className="flex-1 flex overflow-hidden">
          {/* Chat area */}
          <div className="flex-1 flex flex-col">
            {/* Messages */}
            <div className="flex-1 overflow-y-auto px-6 py-8 space-y-6">
              <div className="max-w-4xl mx-auto">
                <AnimatePresence mode="popLayout">
                  {messages.length === 0 && (
                    <motion.div
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: -20 }}
                      className="text-center py-20"
                    >
                      <div className="inline-block p-6 rounded-3xl bg-gradient-to-br from-cyber-500/20 to-neon-purple/20 border border-white/10 mb-6">
                        <Sparkles className="w-16 h-16 text-neon-purple" />
                      </div>
                      <h2 className="text-3xl font-bold mb-4 text-neon">
                        Ask me anything
                      </h2>
                      <p className="text-gray-400 max-w-md mx-auto">
                        Query your research knowledge base using AI. I'll search through papers,
                        documents, and graph connections to provide comprehensive answers.
                      </p>
                    </motion.div>
                  )}

                  {messages.map((message, index) => (
                    <MessageBubble
                      key={message.id}
                      message={message}
                      onSelect={() => setSelectedMessage(message)}
                      isSelected={selectedMessage?.id === message.id}
                      index={index}
                    />
                  ))}
                </AnimatePresence>

                {isLoading && (
                  <motion.div
                    initial={{ opacity: 0, scale: 0.9 }}
                    animate={{ opacity: 1, scale: 1 }}
                    className="flex items-start gap-4"
                  >
                    <div className="p-3 rounded-2xl bg-gradient-to-br from-neon-purple to-neon-pink shadow-neon">
                      <Brain className="w-6 h-6" />
                    </div>
                    <div className="flex-1">
                      <div className="backdrop-blur-xl bg-gradient-to-r from-neon-purple/20 to-neon-pink/20 border border-neon-purple/30 rounded-2xl p-6 shadow-neon">
                        <div className="flex items-center gap-3">
                          <Loader2 className="w-5 h-5 animate-spin text-neon-purple" />
                          <span className="text-gray-300">Thinking...</span>
                        </div>
                      </div>
                    </div>
                  </motion.div>
                )}

                <div ref={messagesEndRef} />
              </div>
            </div>

            {/* Settings panel */}
            <AnimatePresence>
              {showSettings && (
                <motion.div
                  initial={{ height: 0, opacity: 0 }}
                  animate={{ height: 'auto', opacity: 1 }}
                  exit={{ height: 0, opacity: 0 }}
                  className="border-t border-white/10 overflow-hidden"
                >
                  <div className="px-6 py-4 backdrop-blur-xl bg-black/20">
                    <div className="max-w-4xl mx-auto grid grid-cols-1 md:grid-cols-2 gap-6">
                      {/* Model selector */}
                      <div>
                        <label className="block text-sm font-medium text-gray-300 mb-2">
                          Model
                        </label>
                        <div className="relative">
                          <select
                            value={config.model}
                            onChange={(e) => setConfig({ ...config, model: e.target.value })}
                            className="w-full backdrop-blur-xl bg-white/10 border border-white/20 rounded-xl px-4 py-3 text-white outline-none focus:ring-2 focus:ring-cyber-500/50 appearance-none cursor-pointer"
                          >
                            {GEMINI_MODELS.map((model) => (
                              <option key={model.id} value={model.id} className="bg-slate-900">
                                {model.name} - {model.description}
                              </option>
                            ))}
                          </select>
                          <ChevronDown className="absolute right-4 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400 pointer-events-none" />
                        </div>
                      </div>

                      {/* Temperature slider */}
                      <div>
                        <label className="block text-sm font-medium text-gray-300 mb-2">
                          Temperature: <span className="text-cyber-400">{config.temperature.toFixed(2)}</span>
                        </label>
                        <input
                          type="range"
                          min="0"
                          max="1"
                          step="0.01"
                          value={config.temperature}
                          onChange={(e) =>
                            setConfig({ ...config, temperature: parseFloat(e.target.value) })
                          }
                          className="w-full h-2 bg-white/10 rounded-lg appearance-none cursor-pointer slider-thumb"
                        />
                        <div className="flex justify-between text-xs text-gray-500 mt-1">
                          <span>Focused</span>
                          <span>Creative</span>
                        </div>
                      </div>

                      {/* Max tokens */}
                      <div>
                        <label className="block text-sm font-medium text-gray-300 mb-2">
                          Max Tokens: <span className="text-cyber-400">{config.maxTokens}</span>
                        </label>
                        <input
                          type="range"
                          min="256"
                          max="8192"
                          step="256"
                          value={config.maxTokens}
                          onChange={(e) =>
                            setConfig({ ...config, maxTokens: parseInt(e.target.value) })
                          }
                          className="w-full h-2 bg-white/10 rounded-lg appearance-none cursor-pointer slider-thumb"
                        />
                      </div>

                      {/* Top P */}
                      <div>
                        <label className="block text-sm font-medium text-gray-300 mb-2">
                          Top P: <span className="text-cyber-400">{config.topP.toFixed(2)}</span>
                        </label>
                        <input
                          type="range"
                          min="0"
                          max="1"
                          step="0.01"
                          value={config.topP}
                          onChange={(e) =>
                            setConfig({ ...config, topP: parseFloat(e.target.value) })
                          }
                          className="w-full h-2 bg-white/10 rounded-lg appearance-none cursor-pointer slider-thumb"
                        />
                      </div>
                    </div>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>

            {/* Input area */}
            <div className="border-t border-white/10 backdrop-blur-xl bg-black/20 px-6 py-6">
              <div className="max-w-4xl mx-auto">
                <div className="backdrop-blur-xl bg-white/10 border border-white/20 rounded-2xl shadow-glass-lg overflow-hidden">
                  <textarea
                    ref={textareaRef}
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyPress={handleKeyPress}
                    placeholder="Ask a question about your research..."
                    className="w-full bg-transparent px-6 py-4 text-white placeholder-gray-400 outline-none resize-none max-h-48"
                    rows={1}
                    disabled={isLoading}
                  />
                  <div className="flex items-center justify-between px-6 py-3 border-t border-white/10">
                    <div className="flex items-center gap-2 text-sm text-gray-400">
                      <Sparkles className="w-4 h-4" />
                      <span>Powered by {GEMINI_MODELS.find((m) => m.id === config.model)?.name}</span>
                    </div>
                    <motion.button
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                      onClick={handleSend}
                      disabled={!input.trim() || isLoading}
                      className={`
                        px-6 py-2 rounded-xl font-semibold flex items-center gap-2
                        ${
                          input.trim() && !isLoading
                            ? 'bg-gradient-to-r from-cyber-500 to-neon-purple text-white shadow-neon-lg hover:shadow-neon'
                            : 'bg-white/10 text-gray-500 cursor-not-allowed'
                        }
                        transition-all duration-300
                      `}
                    >
                      {isLoading ? (
                        <>
                          <Loader2 className="w-4 h-4 animate-spin" />
                          <span>Sending...</span>
                        </>
                      ) : (
                        <>
                          <Send className="w-4 h-4" />
                          <span>Send</span>
                        </>
                      )}
                    </motion.button>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Context sidebar */}
          <AnimatePresence>
            {showSidebar && selectedMessage && (
              <motion.aside
                initial={{ x: 400, opacity: 0 }}
                animate={{ x: 0, opacity: 1 }}
                exit={{ x: 400, opacity: 0 }}
                className="w-96 border-l border-white/10 backdrop-blur-xl bg-black/20 overflow-y-auto"
              >
                <div className="p-6 space-y-6">
                  {/* Header */}
                  <div className="flex items-center justify-between">
                    <h3 className="text-lg font-bold text-neon">Context</h3>
                    <motion.button
                      whileHover={{ scale: 1.1 }}
                      whileTap={{ scale: 0.9 }}
                      onClick={() => setShowSidebar(false)}
                      className="p-2 rounded-lg hover:bg-white/10 transition-all"
                    >
                      <XCircle className="w-5 h-5" />
                    </motion.button>
                  </div>

                  {/* Source documents */}
                  {selectedMessage.sources && selectedMessage.sources.length > 0 && (
                    <div>
                      <div className="flex items-center gap-2 mb-3">
                        <FileText className="w-4 h-4 text-cyber-400" />
                        <h4 className="font-semibold text-gray-300">Source Documents</h4>
                      </div>
                      <div className="space-y-2">
                        {selectedMessage.sources.map((source, idx) => (
                          <motion.div
                            key={source.id}
                            initial={{ opacity: 0, x: 20 }}
                            animate={{ opacity: 1, x: 0 }}
                            transition={{ delay: idx * 0.1 }}
                            className="backdrop-blur-xl bg-white/5 border border-white/10 rounded-xl p-3 hover:bg-white/10 transition-all cursor-pointer"
                          >
                            <div className="flex items-start justify-between mb-2">
                              <span className="text-sm font-medium text-white line-clamp-2">
                                {source.title}
                              </span>
                              <span className="text-xs px-2 py-1 rounded-full bg-cyber-500/20 text-cyber-400 shrink-0 ml-2">
                                {(source.relevance * 100).toFixed(0)}%
                              </span>
                            </div>
                            <div className="w-full bg-white/10 rounded-full h-1.5">
                              <div
                                className="h-1.5 rounded-full bg-gradient-to-r from-cyber-500 to-neon-purple"
                                style={{ width: `${source.relevance * 100}%` }}
                              />
                            </div>
                          </motion.div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Graph nodes */}
                  {selectedMessage.graphNodes && selectedMessage.graphNodes.length > 0 && (
                    <div>
                      <div className="flex items-center gap-2 mb-3">
                        <Network className="w-4 h-4 text-neon-purple" />
                        <h4 className="font-semibold text-gray-300">Graph References</h4>
                      </div>
                      <div className="space-y-2">
                        {selectedMessage.graphNodes.map((node, idx) => (
                          <motion.div
                            key={node.id}
                            initial={{ opacity: 0, x: 20 }}
                            animate={{ opacity: 1, x: 0 }}
                            transition={{ delay: idx * 0.1 }}
                            className="backdrop-blur-xl bg-white/5 border border-white/10 rounded-xl p-3 hover:bg-white/10 transition-all cursor-pointer"
                          >
                            <div className="flex items-center justify-between">
                              <span className="text-sm font-medium text-white">{node.label}</span>
                              <span className="text-xs px-2 py-1 rounded-full bg-neon-purple/20 text-neon-purple">
                                {node.connections} links
                              </span>
                            </div>
                            <span className="text-xs text-gray-500 capitalize">{node.type}</span>
                          </motion.div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Vector search results */}
                  {selectedMessage.vectorResults && selectedMessage.vectorResults.length > 0 && (
                    <div>
                      <div className="flex items-center gap-2 mb-3">
                        <Search className="w-4 h-4 text-neon-pink" />
                        <h4 className="font-semibold text-gray-300">Vector Search Results</h4>
                      </div>
                      <div className="space-y-2">
                        {selectedMessage.vectorResults.map((result, idx) => (
                          <motion.div
                            key={result.id}
                            initial={{ opacity: 0, x: 20 }}
                            animate={{ opacity: 1, x: 0 }}
                            transition={{ delay: idx * 0.1 }}
                            className="backdrop-blur-xl bg-white/5 border border-white/10 rounded-xl p-3 hover:bg-white/10 transition-all"
                          >
                            <div className="flex items-center justify-between mb-2">
                              <span className="text-xs text-gray-400">{result.source}</span>
                              <span className="text-xs px-2 py-1 rounded-full bg-neon-pink/20 text-neon-pink">
                                {(result.similarity * 100).toFixed(0)}%
                              </span>
                            </div>
                            <p className="text-sm text-gray-300 line-clamp-3">{result.content}</p>
                          </motion.div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </motion.aside>
            )}
          </AnimatePresence>
        </div>
      </div>

      {/* Custom slider styles */}
      <style>{`
        .slider-thumb::-webkit-slider-thumb {
          appearance: none;
          width: 20px;
          height: 20px;
          border-radius: 50%;
          background: linear-gradient(135deg, #0070f3 0%, #a855f7 100%);
          cursor: pointer;
          box-shadow: 0 0 10px rgba(168, 85, 247, 0.5);
        }

        .slider-thumb::-moz-range-thumb {
          width: 20px;
          height: 20px;
          border-radius: 50%;
          background: linear-gradient(135deg, #0070f3 0%, #a855f7 100%);
          cursor: pointer;
          border: none;
          box-shadow: 0 0 10px rgba(168, 85, 247, 0.5);
        }

        .slider-thumb::-webkit-slider-runnable-track {
          background: linear-gradient(to right, rgba(0, 112, 243, 0.3), rgba(168, 85, 247, 0.3));
          border-radius: 4px;
        }

        .slider-thumb::-moz-range-track {
          background: linear-gradient(to right, rgba(0, 112, 243, 0.3), rgba(168, 85, 247, 0.3));
          border-radius: 4px;
        }
      `}</style>
    </div>
  );
};

// Message bubble component
interface MessageBubbleProps {
  message: Message;
  onSelect: () => void;
  isSelected: boolean;
  index: number;
}

const MessageBubble: React.FC<MessageBubbleProps> = ({
  message,
  onSelect,
  isSelected,
  index,
}) => {
  const isUser = message.role === 'user';

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.05 }}
      className={`flex items-start gap-4 ${isUser ? 'flex-row-reverse' : ''}`}
      onClick={onSelect}
    >
      {/* Avatar */}
      <motion.div
        whileHover={{ scale: 1.1, rotate: 5 }}
        className={`
          p-3 rounded-2xl shrink-0
          ${
            isUser
              ? 'bg-gradient-to-br from-cyber-500 to-cyber-600 shadow-[0_0_20px_rgba(0,112,243,0.5)]'
              : 'bg-gradient-to-br from-neon-purple to-neon-pink shadow-[0_0_20px_rgba(168,85,247,0.5)]'
          }
        `}
      >
        {isUser ? <User className="w-6 h-6" /> : <Brain className="w-6 h-6" />}
      </motion.div>

      {/* Message content */}
      <div className={`flex-1 max-w-2xl ${isUser ? 'flex flex-col items-end' : ''}`}>
        <motion.div
          whileHover={{ scale: 1.01 }}
          className={`
            backdrop-blur-xl border rounded-2xl p-6 cursor-pointer
            transition-all duration-300
            ${
              isUser
                ? 'bg-gradient-to-r from-cyber-500/20 to-cyber-600/20 border-cyber-500/30 shadow-[0_0_20px_rgba(0,112,243,0.3)]'
                : 'bg-gradient-to-r from-neon-purple/20 to-neon-pink/20 border-neon-purple/30 shadow-[0_0_20px_rgba(168,85,247,0.3)]'
            }
            ${
              isSelected
                ? isUser
                  ? 'ring-2 ring-cyber-500 shadow-[0_0_30px_rgba(0,112,243,0.5)]'
                  : 'ring-2 ring-neon-purple shadow-[0_0_30px_rgba(168,85,247,0.5)]'
                : ''
            }
          `}
        >
          <p className="text-white whitespace-pre-wrap leading-relaxed">{message.content}</p>

          {/* Metadata badges */}
          {!isUser && (message.sources || message.graphNodes || message.vectorResults) && (
            <div className="flex items-center gap-2 mt-4 pt-4 border-t border-white/10">
              {message.sources && message.sources.length > 0 && (
                <span className="text-xs px-2 py-1 rounded-full bg-cyber-500/20 text-cyber-400 flex items-center gap-1">
                  <FileText className="w-3 h-3" />
                  {message.sources.length} sources
                </span>
              )}
              {message.graphNodes && message.graphNodes.length > 0 && (
                <span className="text-xs px-2 py-1 rounded-full bg-neon-purple/20 text-neon-purple flex items-center gap-1">
                  <Network className="w-3 h-3" />
                  {message.graphNodes.length} nodes
                </span>
              )}
              {message.vectorResults && message.vectorResults.length > 0 && (
                <span className="text-xs px-2 py-1 rounded-full bg-neon-pink/20 text-neon-pink flex items-center gap-1">
                  <Search className="w-3 h-3" />
                  {message.vectorResults.length} results
                </span>
              )}
            </div>
          )}
        </motion.div>

        {/* Timestamp */}
        <span className="text-xs text-gray-500 mt-2">
          {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
        </span>
      </div>
    </motion.div>
  );
};

export default Ask;
