import React, { useState, useCallback, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import GlassCard from '../components/Common/GlassCard';
import type { UploadedFile } from '../types';

interface UploadPageProps {
  onProcess?: (files: UploadedFile[]) => Promise<void>;
}

const SUPPORTED_FORMATS = {
  'application/pdf': { ext: 'PDF', icon: 'pdf', color: 'text-red-400' },
  'text/plain': { ext: 'TXT', icon: 'txt', color: 'text-blue-400' },
  'application/json': { ext: 'JSON', icon: 'json', color: 'text-green-400' },
  'text/csv': { ext: 'CSV', icon: 'csv', color: 'text-yellow-400' },
};

const Upload: React.FC<UploadPageProps> = ({ onProcess }) => {
  const [isDragging, setIsDragging] = useState(false);
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([]);
  const [selectedFile, setSelectedFile] = useState<UploadedFile | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const isValidFile = (file: File): boolean => {
    return Object.keys(SUPPORTED_FORMATS).includes(file.type) ||
           file.name.endsWith('.csv');
  };

  const processFiles = async (files: FileList | File[]) => {
    const fileArray = Array.from(files).filter(isValidFile);

    if (fileArray.length === 0) {
      alert('No valid files selected. Please upload PDF, TXT, JSON, or CSV files.');
      return;
    }

    const newFiles: UploadedFile[] = fileArray.map((file) => ({
      id: `${Date.now()}-${Math.random().toString(36).substring(7)}`,
      name: file.name,
      size: file.size,
      type: file.type || 'text/csv',
      progress: 0,
      status: 'pending',
    }));

    setUploadedFiles((prev) => [...prev, ...newFiles]);

    // Simulate upload progress for each file
    for (const uploadFile of newFiles) {
      setUploadedFiles((prev) =>
        prev.map((f) =>
          f.id === uploadFile.id ? { ...f, status: 'uploading' } : f
        )
      );

      // Simulate progress
      for (let progress = 0; progress <= 100; progress += 10) {
        await new Promise((resolve) => setTimeout(resolve, 150));
        setUploadedFiles((prev) =>
          prev.map((f) =>
            f.id === uploadFile.id ? { ...f, progress } : f
          )
        );
      }

      setUploadedFiles((prev) =>
        prev.map((f) =>
          f.id === uploadFile.id ? { ...f, status: 'completed', progress: 100 } : f
        )
      );
    }
  };

  const handleDrop = useCallback(
    async (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);

      const files = e.dataTransfer.files;
      if (files.length > 0) {
        await processFiles(files);
      }
    },
    []
  );

  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      await processFiles(files);
    }
    // Reset input
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const removeFile = (fileId: string) => {
    setUploadedFiles((prev) => prev.filter((f) => f.id !== fileId));
    if (selectedFile?.id === fileId) {
      setSelectedFile(null);
    }
  };

  const clearAllFiles = () => {
    setUploadedFiles([]);
    setSelectedFile(null);
  };

  const handleProcessFiles = async () => {
    const completedFiles = uploadedFiles.filter(f => f.status === 'completed');
    if (completedFiles.length === 0) return;

    setIsProcessing(true);
    try {
      if (onProcess) {
        await onProcess(completedFiles);
      }
      // Simulate processing
      await new Promise(resolve => setTimeout(resolve, 2000));
      alert('Files successfully ingested into knowledge base!');
    } catch (error) {
      alert('Error processing files. Please try again.');
    } finally {
      setIsProcessing(false);
    }
  };

  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round((bytes / Math.pow(k, i)) * 100) / 100 + ' ' + sizes[i];
  };

  const getFileIcon = (type: string) => {
    const fileType = SUPPORTED_FORMATS[type as keyof typeof SUPPORTED_FORMATS];
    const color = fileType?.color || 'text-gray-400';

    if (type.includes('pdf')) {
      return (
        <svg className={`w-8 h-8 ${color}`} fill="currentColor" viewBox="0 0 20 20">
          <path fillRule="evenodd" d="M4 4a2 2 0 012-2h4.586A2 2 0 0112 2.586L15.414 6A2 2 0 0116 7.414V16a2 2 0 01-2 2H6a2 2 0 01-2-2V4zm2 6a1 1 0 011-1h6a1 1 0 110 2H7a1 1 0 01-1-1zm1 3a1 1 0 100 2h6a1 1 0 100-2H7z" clipRule="evenodd" />
        </svg>
      );
    }
    if (type.includes('json')) {
      return (
        <svg className={`w-8 h-8 ${color}`} fill="currentColor" viewBox="0 0 20 20">
          <path d="M3 4a1 1 0 011-1h12a1 1 0 011 1v2a1 1 0 01-1 1H4a1 1 0 01-1-1V4zM3 10a1 1 0 011-1h6a1 1 0 011 1v6a1 1 0 01-1 1H4a1 1 0 01-1-1v-6zM14 9a1 1 0 00-1 1v6a1 1 0 001 1h2a1 1 0 001-1v-6a1 1 0 00-1-1h-2z" />
        </svg>
      );
    }
    return (
      <svg className={`w-8 h-8 ${color}`} fill="currentColor" viewBox="0 0 20 20">
        <path fillRule="evenodd" d="M4 4a2 2 0 012-2h4.586A2 2 0 0112 2.586L15.414 6A2 2 0 0116 7.414V16a2 2 0 01-2 2H6a2 2 0 01-2-2V4zm2 6a1 1 0 011-1h6a1 1 0 110 2H7a1 1 0 01-1-1zm1 3a1 1 0 100 2h6a1 1 0 100-2H7z" clipRule="evenodd" />
      </svg>
    );
  };

  const getStatusColor = (status: UploadedFile['status']) => {
    switch (status) {
      case 'completed':
        return 'text-green-400';
      case 'error':
        return 'text-red-400';
      case 'uploading':
        return 'text-blue-400';
      default:
        return 'text-white/60';
    }
  };

  const getFileExtension = (filename: string) => {
    return filename.split('.').pop()?.toUpperCase() || 'FILE';
  };

  const renderPreview = (file: UploadedFile) => {
    return (
      <div className="h-full flex flex-col items-center justify-center p-8 text-center">
        <div className="mb-6">{getFileIcon(file.type)}</div>
        <h3 className="text-xl font-semibold text-white mb-2">{file.name}</h3>
        <div className="space-y-2 text-white/60">
          <p>Size: {formatFileSize(file.size)}</p>
          <p>Type: {getFileExtension(file.name)}</p>
          <p className={getStatusColor(file.status)}>
            Status: {file.status.charAt(0).toUpperCase() + file.status.slice(1)}
          </p>
        </div>
        {file.status === 'completed' && (
          <motion.div
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            className="mt-6"
          >
            <div className="w-16 h-16 rounded-full bg-green-500/20 flex items-center justify-center">
              <svg className="w-8 h-8 text-green-400" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
              </svg>
            </div>
          </motion.div>
        )}
      </div>
    );
  };

  const completedCount = uploadedFiles.filter(f => f.status === 'completed').length;

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-900 via-blue-900 to-indigo-900 p-6">
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-8"
        >
          <h1 className="text-4xl font-bold text-white mb-3">Upload Documents</h1>
          <p className="text-white/60 text-lg">
            Upload your research documents to build your knowledge base
          </p>
        </motion.div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Left Column - Upload Area & File List */}
          <div className="space-y-6">
            {/* Drag & Drop Zone */}
            <GlassCard className="p-6">
              <motion.div
                className={`
                  relative border-2 border-dashed rounded-2xl p-12 text-center transition-all
                  ${isDragging
                    ? 'border-blue-500 bg-blue-500/10 scale-[1.02]'
                    : 'border-white/20 hover:border-white/40 hover:bg-white/5'
                  }
                `}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                animate={{ scale: isDragging ? 1.02 : 1 }}
              >
                <input
                  ref={fileInputRef}
                  type="file"
                  multiple
                  accept=".pdf,.txt,.json,.csv,application/pdf,text/plain,application/json,text/csv"
                  onChange={handleFileSelect}
                  className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                  id="fileInput"
                />

                <motion.div
                  initial={{ scale: 1 }}
                  animate={{ scale: isDragging ? 1.1 : 1 }}
                  className="space-y-4"
                >
                  <div className="flex justify-center">
                    <motion.div
                      className="w-20 h-20 rounded-2xl bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center"
                      animate={{
                        rotate: isDragging ? 360 : 0,
                      }}
                      transition={{ duration: 0.6 }}
                    >
                      <svg className="w-10 h-10 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                      </svg>
                    </motion.div>
                  </div>

                  <div>
                    <p className="text-white text-lg font-semibold mb-2">
                      {isDragging ? 'Drop files here' : 'Drag and drop files here'}
                    </p>
                    <p className="text-white/60 text-sm mb-4">or</p>
                    <label
                      htmlFor="fileInput"
                      className="inline-block px-6 py-3 bg-gradient-to-r from-blue-500 to-purple-600 text-white rounded-xl font-semibold cursor-pointer hover:shadow-lg transition-shadow"
                    >
                      Browse Files
                    </label>
                  </div>

                  <div className="space-y-2">
                    <p className="text-white/60 text-sm">Supported formats:</p>
                    <div className="flex flex-wrap justify-center gap-2">
                      {Object.values(SUPPORTED_FORMATS).map((format) => (
                        <span
                          key={format.ext}
                          className={`px-3 py-1 rounded-full text-xs font-semibold ${format.color} bg-white/5 border border-white/10`}
                        >
                          {format.ext}
                        </span>
                      ))}
                    </div>
                    <p className="text-white/40 text-xs mt-2">
                      Max 50MB per file
                    </p>
                  </div>
                </motion.div>
              </motion.div>
            </GlassCard>

            {/* File List */}
            {uploadedFiles.length > 0 && (
              <GlassCard className="p-6">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-lg font-semibold text-white">
                    Uploaded Files ({uploadedFiles.length})
                  </h3>
                  <button
                    onClick={clearAllFiles}
                    className="text-sm text-white/60 hover:text-white transition-colors"
                  >
                    Clear All
                  </button>
                </div>

                <div className="space-y-2 max-h-[400px] overflow-y-auto custom-scrollbar">
                  <AnimatePresence>
                    {uploadedFiles.map((file) => (
                      <motion.div
                        key={file.id}
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        exit={{ opacity: 0, x: 20 }}
                        className={`
                          p-4 rounded-xl bg-white/5 border border-white/10
                          cursor-pointer transition-all
                          ${selectedFile?.id === file.id ? 'ring-2 ring-blue-500 bg-blue-500/10' : 'hover:bg-white/10'}
                        `}
                        onClick={() => setSelectedFile(file)}
                      >
                        <div className="flex items-center gap-3">
                          <div className={`${getStatusColor(file.status)}`}>
                            {getFileIcon(file.type)}
                          </div>

                          <div className="flex-1 min-w-0">
                            <p className="text-white font-medium truncate">{file.name}</p>
                            <div className="flex items-center gap-3 mt-1">
                              <span className="text-white/60 text-xs">
                                {formatFileSize(file.size)}
                              </span>
                              <span className={`text-xs font-medium ${getStatusColor(file.status)}`}>
                                {file.status === 'completed' && 'Completed'}
                                {file.status === 'uploading' && `${file.progress}%`}
                                {file.status === 'pending' && 'Pending'}
                                {file.status === 'error' && (file.error || 'Error')}
                              </span>
                            </div>

                            {file.status === 'uploading' && (
                              <div className="mt-2 h-1.5 bg-white/10 rounded-full overflow-hidden">
                                <motion.div
                                  className="h-full bg-gradient-to-r from-blue-500 to-purple-600"
                                  initial={{ width: 0 }}
                                  animate={{ width: `${file.progress}%` }}
                                  transition={{ duration: 0.3 }}
                                />
                              </div>
                            )}
                          </div>

                          <div className="flex items-center gap-2">
                            {file.status === 'completed' && (
                              <svg className="w-6 h-6 text-green-400" fill="currentColor" viewBox="0 0 20 20">
                                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                              </svg>
                            )}
                            {file.status === 'error' && (
                              <svg className="w-6 h-6 text-red-400" fill="currentColor" viewBox="0 0 20 20">
                                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                              </svg>
                            )}

                            {(file.status === 'completed' || file.status === 'error') && (
                              <button
                                onClick={(e) => {
                                  e.stopPropagation();
                                  removeFile(file.id);
                                }}
                                className="p-1 rounded-lg hover:bg-white/10 transition-colors"
                              >
                                <svg className="w-5 h-5 text-white/60" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                                </svg>
                              </button>
                            )}
                          </div>
                        </div>
                      </motion.div>
                    ))}
                  </AnimatePresence>
                </div>

                {/* Process Button */}
                {completedCount > 0 && (
                  <motion.button
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    onClick={handleProcessFiles}
                    disabled={isProcessing}
                    className="w-full mt-4 py-4 bg-gradient-to-r from-green-500 to-emerald-600 text-white rounded-xl font-semibold hover:shadow-lg transition-shadow disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                  >
                    {isProcessing ? (
                      <>
                        <svg className="animate-spin h-5 w-5" fill="none" viewBox="0 0 24 24">
                          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                        </svg>
                        Processing...
                      </>
                    ) : (
                      <>
                        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                        </svg>
                        Process {completedCount} {completedCount === 1 ? 'File' : 'Files'}
                      </>
                    )}
                  </motion.button>
                )}
              </GlassCard>
            )}
          </div>

          {/* Right Column - Preview Pane */}
          <div className="lg:sticky lg:top-6 h-fit">
            <GlassCard className="p-6">
              <h3 className="text-lg font-semibold text-white mb-4">Preview</h3>
              <div className="min-h-[400px] rounded-xl bg-white/5 border border-white/10">
                {selectedFile ? (
                  <AnimatePresence mode="wait">
                    <motion.div
                      key={selectedFile.id}
                      initial={{ opacity: 0, scale: 0.95 }}
                      animate={{ opacity: 1, scale: 1 }}
                      exit={{ opacity: 0, scale: 0.95 }}
                      className="h-full"
                    >
                      {renderPreview(selectedFile)}
                    </motion.div>
                  </AnimatePresence>
                ) : (
                  <div className="h-full flex flex-col items-center justify-center p-8 text-center">
                    <svg className="w-20 h-20 text-white/20 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                    </svg>
                    <p className="text-white/60">
                      {uploadedFiles.length === 0
                        ? 'Upload files to see preview'
                        : 'Select a file to preview'}
                    </p>
                  </div>
                )}
              </div>

              {/* Upload Statistics */}
              {uploadedFiles.length > 0 && (
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="mt-6 grid grid-cols-3 gap-4"
                >
                  <div className="p-4 rounded-xl bg-white/5 border border-white/10 text-center">
                    <p className="text-2xl font-bold text-white">{uploadedFiles.length}</p>
                    <p className="text-xs text-white/60 mt-1">Total</p>
                  </div>
                  <div className="p-4 rounded-xl bg-white/5 border border-white/10 text-center">
                    <p className="text-2xl font-bold text-green-400">{completedCount}</p>
                    <p className="text-xs text-white/60 mt-1">Completed</p>
                  </div>
                  <div className="p-4 rounded-xl bg-white/5 border border-white/10 text-center">
                    <p className="text-2xl font-bold text-blue-400">
                      {uploadedFiles.filter(f => f.status === 'uploading').length}
                    </p>
                    <p className="text-xs text-white/60 mt-1">Uploading</p>
                  </div>
                </motion.div>
              )}
            </GlassCard>
          </div>
        </div>
      </div>

      <style>{`
        .custom-scrollbar::-webkit-scrollbar {
          width: 8px;
        }
        .custom-scrollbar::-webkit-scrollbar-track {
          background: rgba(255, 255, 255, 0.05);
          border-radius: 10px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb {
          background: rgba(255, 255, 255, 0.2);
          border-radius: 10px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover {
          background: rgba(255, 255, 255, 0.3);
        }
      `}</style>
    </div>
  );
};

export default Upload;
