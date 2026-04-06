import React from 'react';
import GlassCard from '../components/ui/GlassCard';

const Backtest: React.FC = () => {
  return (
    <div className="space-y-6">
      <header className="mb-8">
        <h1 className="text-4xl font-extrabold tracking-tight text-on-surface">Strategy Backtest</h1>
        <p className="text-on-surface-variant">Configure parameters and evaluate historical performance.</p>
      </header>
      
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <GlassCard className="lg:col-span-4 h-96 flex items-center justify-center">
          <p className="text-on-surface-variant font-bold uppercase tracking-widest">Config Form - Coming Soon</p>
        </GlassCard>
        <GlassCard className="lg:col-span-8 h-96 flex items-center justify-center">
          <p className="text-on-surface-variant font-bold uppercase tracking-widest">Equity Curve - Coming Soon</p>
        </GlassCard>
      </div>
    </div>
  );
};

export default Backtest;
