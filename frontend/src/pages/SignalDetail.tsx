import React from 'react';
import { useParams } from 'react-router-dom';
import GlassCard from '../components/ui/GlassCard';

const SignalDetail: React.FC = () => {
  const { ticker } = useParams<{ ticker: string }>();

  return (
    <div className="space-y-6">
      <header className="mb-8">
        <h1 className="text-4xl font-extrabold tracking-tight text-on-surface">Signal Detail - {ticker}</h1>
        <p className="text-on-surface-variant">Deep dive analysis for {ticker}.</p>
      </header>
      
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <GlassCard className="lg:col-span-8 h-96 flex items-center justify-center">
          <p className="text-on-surface-variant font-bold uppercase tracking-widest">K-Line Chart - Coming Soon</p>
        </GlassCard>
        <GlassCard className="lg:col-span-4 h-96 flex items-center justify-center">
          <p className="text-on-surface-variant font-bold uppercase tracking-widest">Model Metrics - Coming Soon</p>
        </GlassCard>
      </div>
    </div>
  );
};

export default SignalDetail;
