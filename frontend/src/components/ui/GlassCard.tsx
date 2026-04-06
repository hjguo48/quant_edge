import React from 'react';

interface GlassCardProps {
  children: React.ReactNode;
  className?: string;
}

const GlassCard: React.FC<GlassCardProps> = ({ children, className = '' }) => {
  return (
    <div className={`glass-panel rounded-xl border border-outline-variant/10 p-6 ${className}`}>
      {children}
    </div>
  );
};

export default GlassCard;
