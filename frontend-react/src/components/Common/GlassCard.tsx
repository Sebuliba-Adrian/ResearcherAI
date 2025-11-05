import React from 'react';
import { motion } from 'framer-motion';

interface GlassCardProps {
  children: React.ReactNode;
  className?: string;
  hover?: boolean;
  onClick?: () => void;
}

const GlassCard: React.FC<GlassCardProps> = ({
  children,
  className = '',
  hover = false,
  onClick,
}) => {
  return (
    <motion.div
      className={`
        backdrop-blur-xl bg-white/10 dark:bg-black/10
        border border-white/20 dark:border-white/10
        rounded-2xl shadow-xl
        ${hover ? 'cursor-pointer' : ''}
        ${className}
      `}
      onClick={onClick}
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      whileHover={hover ? {
        scale: 1.02,
        boxShadow: '0 20px 40px rgba(0, 0, 0, 0.2)'
      } : undefined}
    >
      {children}
    </motion.div>
  );
};

export default GlassCard;
