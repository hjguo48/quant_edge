import React from 'react';

interface TopNavProps {
  title: string;
}

const TopNav: React.FC<TopNavProps> = ({ title }) => {
  return (
    <nav className="fixed top-0 w-full z-50 bg-surface-container flex justify-between items-center h-14 px-6 border-b border-on-surface-variant/10">
      <div className="flex items-center gap-8">
        <span className="text-xl font-bold tracking-tighter text-on-surface md:hidden">QuantEdge</span>
        <span className="hidden md:block text-sm font-bold text-on-surface">{title}</span>
      </div>
      <div className="flex items-center space-x-4">
        <div className="relative group">
          <span className="material-symbols-outlined text-on-surface-variant cursor-pointer hover:text-on-surface transition-colors">notifications</span>
          <div className="absolute top-0.5 right-0.5 w-2 h-2 bg-primary rounded-full border-2 border-surface-container"></div>
        </div>
        <span className="material-symbols-outlined text-on-surface-variant cursor-pointer hover:text-on-surface transition-colors">person</span>
      </div>
    </nav>
  );
};

export default TopNav;
