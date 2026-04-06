import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import AppLayout from './components/layout/AppLayout';
import Dashboard from './pages/Dashboard';
import Signals from './pages/Signals';
import SignalDetail from './pages/SignalDetail';
import Portfolio from './pages/Portfolio';
import Backtest from './pages/Backtest';
import G4Gate from './pages/G4Gate';

const App: React.FC = () => {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<AppLayout />}>
          <Route index element={<Dashboard />} />
          <Route path="signals" element={<Signals />} />
          <Route path="signals/:ticker" element={<SignalDetail />} />
          <Route path="portfolio" element={<Portfolio />} />
          <Route path="backtest" element={<Backtest />} />
          <Route path="g4gate" element={<G4Gate />} />
          {/* Default fallback */}
          <Route path="*" element={<Dashboard />} />
        </Route>
      </Routes>
    </Router>
  );
};

export default App;
