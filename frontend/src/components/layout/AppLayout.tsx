import React from 'react';
import { Outlet, useLocation, Link } from 'react-router-dom';
import SideNav from './SideNav';
import TopNav from './TopNav';

const pageTitles: Record<string, string> = {
  '/': 'Institutional Terminal',
  '/signals': 'Alpha Stream',
  '/portfolio': 'Institutional Portfolio',
  '/backtest': 'Strategy Backtest',
  '/g4gate': 'G4 Gate Verification',
};

const AppLayout: React.FC = () => {
  const location = useLocation();
  const currentPath = location.pathname;
  let title = pageTitles[currentPath] || 'Institutional Terminal';
  
  if (currentPath.startsWith('/signals/')) {
    title = `Signal Detail - ${currentPath.split('/').pop()}`;
  }

  return (
    <div className="min-h-screen bg-background">
      <TopNav title={title} />
      <SideNav />
      
      <main className="md:ml-64 mt-14 p-6 lg:p-8 min-h-[calc(100vh-3.5rem)] pb-24 md:pb-8">
        <Outlet />
      </main>
      
      {/* Mobile Bottom Nav */}
      <div className="md:hidden fixed bottom-0 left-0 right-0 bg-surface-container flex justify-around items-center h-16 z-50 border-t border-outline-variant/10 px-2">
        <Link to="/" className={`flex flex-col items-center gap-1 ${currentPath === '/' ? 'text-primary' : 'text-on-surface-variant'}`}>
          <span className="material-symbols-outlined">dashboard</span>
          <span className="text-[10px] font-bold">Main</span>
        </Link>
        <Link to="/signals" className={`flex flex-col items-center gap-1 ${currentPath.startsWith('/signals') ? 'text-primary' : 'text-on-surface-variant'}`}>
          <span className="material-symbols-outlined">show_chart</span>
          <span className="text-[10px] font-bold">Signals</span>
        </Link>
        <Link to="/portfolio" className={`flex flex-col items-center gap-1 ${currentPath === '/portfolio' ? 'text-primary' : 'text-on-surface-variant'}`}>
          <span className="material-symbols-outlined">account_balance_wallet</span>
          <span className="text-[10px] font-bold">Portfolio</span>
        </Link>
        <Link to="/backtest" className={`flex flex-col items-center gap-1 ${currentPath === '/backtest' ? 'text-primary' : 'text-on-surface-variant'}`}>
          <span className="material-symbols-outlined">biotech</span>
          <span className="text-[10px] font-bold">Backtest</span>
        </Link>
      </div>
    </div>
  );
};

export default AppLayout;
