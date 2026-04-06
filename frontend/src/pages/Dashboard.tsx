import React from 'react';
import GlassCard from '../components/ui/GlassCard';

const Dashboard: React.FC = () => {
  return (
    <div className="space-y-6">
      <header className="mb-8">
        <h1 className="text-4xl font-extrabold tracking-tight text-on-surface">Institutional Terminal</h1>
        <p className="text-on-surface-variant">Real-time market overview and portfolio summary.</p>
      </header>
      
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        <GlassCard className="lg:col-span-2 h-64 flex items-center justify-center">
          <p className="text-on-surface-variant font-bold uppercase tracking-widest">Market Overview - Coming Soon</p>
        </GlassCard>
        <GlassCard className="h-64 flex items-center justify-center">
          <p className="text-on-surface-variant font-bold uppercase tracking-widest">Watchlist - Coming Soon</p>
        </GlassCard>
      </div>
    </div>
  );
};

export default Dashboard;
