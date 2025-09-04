import React from 'react';
import { StatBadgeProps } from '../types';

const StatBadge: React.FC<StatBadgeProps> = ({ label, value, variant = 'default' }) => {
  const getVariantClasses = () => {
    switch (variant) {
      case 'success':
        return 'bg-f1-good bg-opacity-10 text-f1-good border-f1-good border-opacity-25';
      case 'error':
        return 'bg-f1-bad bg-opacity-10 text-f1-bad border-f1-bad border-opacity-25';
      case 'info':
        return 'bg-f1-accent bg-opacity-10 text-f1-accent border-f1-accent border-opacity-25';
      default:
        return 'bg-f1-panel text-f1-text border-f1-border';
    }
  };

  return (
    <div className={`p-4 rounded-xl border ${getVariantClasses()}`}>
      <div className="text-2xl font-bold mb-1">{value}</div>
      <div className="text-sm font-medium opacity-80">{label}</div>
    </div>
  );
};

export default StatBadge;
