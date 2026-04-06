import React from 'react';
import { NavLink } from 'react-router-dom';

const navItems = [
  { name: 'Dashboard', path: '/', icon: 'dashboard' },
  { name: 'Signals', path: '/signals', icon: 'show_chart' },
  { name: 'Portfolio', path: '/portfolio', icon: 'account_balance_wallet' },
  { name: 'Backtest', path: '/backtest', icon: 'biotech' },
  { name: 'G4 Gate', path: '/g4gate', icon: 'security' },
];

const SideNav: React.FC = () => {
  return (
    <aside className="fixed left-0 top-14 h-[calc(100vh-3.5rem)] w-64 bg-background border-r border-on-surface-variant/15 flex flex-col py-4 space-y-1 hidden md:flex z-40">
      <div className="px-6 mb-6">
        <h2 className="text-lg font-black text-on-surface">QuantEdge</h2>
        <p className="text-[10px] uppercase tracking-widest text-on-surface-variant/60">Institutional Terminal</p>
      </div>
      <nav className="flex-1 px-3 space-y-1">
        {navItems.map((item) => (
          <NavLink
            key={item.path}
            to={item.path}
            className={({ isActive }) =>
              `flex items-center space-x-3 px-4 py-2.5 rounded-lg transition-all duration-200 ${
                isActive
                  ? 'text-primary-container bg-surface-container border-r-2 border-primary-container font-bold'
                  : 'text-on-surface-variant hover:bg-surface-container hover:text-primary-container'
              }`
            }
          >
            <span className="material-symbols-outlined">{item.icon}</span>
            <span className="font-medium text-sm">{item.name}</span>
          </NavLink>
        ))}
      </nav>
      <div className="mt-auto px-6 py-4 border-t border-outline-variant/10">
        <div className="flex items-center space-x-3">
          <div className="w-8 h-8 rounded-full bg-surface-container-highest flex items-center justify-center overflow-hidden border border-outline-variant/20">
             <div className="w-full h-full bg-primary/20 flex items-center justify-center text-primary text-xs font-bold">AM</div>
          </div>
          <div>
            <p className="text-xs font-bold text-on-surface">Alex Mercer</p>
            <p className="text-[10px] text-on-surface-variant">Chief Strategist</p>
          </div>
        </div>
      </div>
    </aside>
  );
};

export default SideNav;
