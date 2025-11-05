import React, { useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import GlassCard from '../Common/GlassCard';
import type { UploadedFile } from '../../types';

interface FileUploadProps {
  onUpload: (files: File[]) => Promise<void>;
}

const FileUpload: React.FC<FileUploadProps> = ({ onUpload }) => {
  const [isDragging, setIsDragging] = useState(false);
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([]);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const processFiles = async (files: FileList | File[]) => {
    const fileArray = Array.from(files);
    const newFiles: UploadedFile[] = fileArray.map((file) => ({
      id: Math.random().toString(36).substring(7),
      name: file.name,
      size: file.size,
      type: file.type,
      progress: 0,
      status: 'pending',
    }));

    setUploadedFiles((prev) => [...prev, ...newFiles]);

    // Simulate upload progress
    for (const uploadFile of newFiles) {
      setUploadedFiles((prev) =>
        prev.map((f) =>
          f.id === uploadFile.id ? { ...f, status: 'uploading' } : f
        )
      );

      // Simulate progress
      for (let progress = 0; progress <= 100; progress += 10) {
        await new Promise((resolve) => setTimeout(resolve, 200));
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

    // Call the actual upload handler
    try {
      await onUpload(fileArray);
    } catch (error) {
      setUploadedFiles((prev) =>
        prev.map((f) =>
          newFiles.some((nf) => nf.id === f.id)
            ? { ...f, status: 'error', error: 'Upload failed' }
            : f
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
    [onUpload]
  );

  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      await processFiles(files);
    }
  };

  const removeFile = (fileId: string) => {
    setUploadedFiles((prev) => prev.filter((f) => f.id !== fileId));
  };

  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
  };

  const getFileIcon = (type: string) => {
    if (type.includes('pdf')) {
      return (
        <svg className="w-8 h-8" fill="currentColor" viewBox="0 0 20 20">
          <path fillRule="evenodd" d="M4 4a2 2 0 012-2h4.586A2 2 0 0112 2.586L15.414 6A2 2 0 0116 7.414V16a2 2 0 01-2 2H6a2 2 0 01-2-2V4z" clipRule="evenodd" />
        </svg>
      );
    }
    return (
      <svg className="w-8 h-8" fill="currentColor" viewBox="0 0 20 20">
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

  return (
    <div className="space-y-6">
      {/* Upload Area */}
      <GlassCard className="p-6">
        <div className="space-y-4">
          <div>
            <h2 className="text-2xl font-bold text-white mb-2">Upload Papers</h2>
            <p className="text-white/60 text-sm">
              Upload PDF files or other documents to add to your research collection
            </p>
          </div>

          {/* Drop Zone */}
          <motion.div
            className={`
              relative border-2 border-dashed rounded-2xl p-12 text-center transition-all
              ${isDragging
                ? 'border-blue-500 bg-blue-500/10 scale-105'
                : 'border-white/20 hover:border-white/40 hover:bg-white/5'
              }
            `}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            animate={{ scale: isDragging ? 1.02 : 1 }}
          >
            <input
              type="file"
              multiple
              accept=".pdf,.doc,.docx,.txt"
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
                <div className="w-20 h-20 rounded-2xl bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center">
                  <svg className="w-10 h-10 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                  </svg>
                </div>
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

              <p className="text-white/40 text-xs">
                Supported formats: PDF, DOC, DOCX, TXT (Max 50MB per file)
              </p>
            </motion.div>
          </motion.div>
        </div>
      </GlassCard>

      {/* Uploaded Files List */}
      {uploadedFiles.length > 0 && (
        <GlassCard className="p-6">
          <h3 className="text-lg font-semibold text-white mb-4">
            Uploaded Files ({uploadedFiles.length})
          </h3>

          <div className="space-y-3">
            <AnimatePresence>
              {uploadedFiles.map((file) => (
                <motion.div
                  key={file.id}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: 20 }}
                  className="p-4 rounded-xl bg-white/5 border border-white/10"
                >
                  <div className="flex items-center gap-4">
                    {/* File Icon */}
                    <div className={`${getStatusColor(file.status)}`}>
                      {getFileIcon(file.type)}
                    </div>

                    {/* File Info */}
                    <div className="flex-1 min-w-0">
                      <p className="text-white font-medium truncate">{file.name}</p>
                      <div className="flex items-center gap-3 mt-1">
                        <span className="text-white/60 text-xs">
                          {formatFileSize(file.size)}
                        </span>
                        <span className={`text-xs font-medium ${getStatusColor(file.status)}`}>
                          {file.status === 'completed' && 'Completed'}
                          {file.status === 'uploading' && `Uploading ${file.progress}%`}
                          {file.status === 'pending' && 'Pending'}
                          {file.status === 'error' && (file.error || 'Error')}
                        </span>
                      </div>

                      {/* Progress Bar */}
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

                    {/* Status Icon */}
                    <div className="flex-shrink-0">
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
                    </div>

                    {/* Remove Button */}
                    {(file.status === 'completed' || file.status === 'error') && (
                      <button
                        onClick={() => removeFile(file.id)}
                        className="flex-shrink-0 p-2 rounded-lg hover:bg-white/10 transition-colors"
                        aria-label="Remove file"
                      >
                        <svg className="w-5 h-5 text-white/60" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                        </svg>
                      </button>
                    )}
                  </div>
                </motion.div>
              ))}
            </AnimatePresence>
          </div>
        </GlassCard>
      )}
    </div>
  );
};

export default FileUpload;
