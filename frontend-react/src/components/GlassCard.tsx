import { motion } from 'framer-motion';
import type { ReactNode } from 'react';

interface GlassCardProps {
  children: ReactNode;
  className?: string;
  onClick?: () => void;
  hover?: boolean;
}

export const GlassCard = ({ children, className = '', onClick, hover = true }: GlassCardProps) => {
  return (
    <motion.div
      className={`glass-card p-6 ${className}`}
      onClick={onClick}
      whileHover={hover ? { scale: 1.02, boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.5)' } : {}}
      whileTap={hover ? { scale: 0.98 } : {}}
      transition={{ type: 'spring', stiffness: 300, damping: 20 }}
    >
      {children}
    </motion.div>
  );
};

export default GlassCard;
