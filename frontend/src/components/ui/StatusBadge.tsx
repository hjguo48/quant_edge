import React from 'react';

export type Status = 'PASS' | 'FAIL' | 'PENDING' | 'ACTIVE' | 'ALERT';

interface StatusBadgeProps {
  status: Status;
  className?: string;
}

const StatusBadge: React.FC<StatusBadgeProps> = ({ status, className = '' }) => {
  const getColors = () => {
    switch (status) {
      case 'PASS':
      case 'ACTIVE':
        return 'bg-primary/10 text-primary border-primary/20';
      case 'FAIL':
      case 'ALERT':
        return 'bg-secondary/10 text-secondary border-secondary/20';
      case 'PENDING':
        return 'bg-surface-container-highest text-on-surface-variant border-outline-variant/20';
      default:
        return 'bg-surface-container-high text-on-surface-variant border-outline-variant/20';
    }
  };

  return (
    <span className={`px-2 py-0.5 rounded text-[10px] font-black border uppercase tracking-widest ${getColors()} ${className}`}>
      {status}
    </span>
  );
};

export default StatusBadge;
