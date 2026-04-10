import { useState } from "react";
import Sidebar from "../components/Sidebar";
import TopBar from "../components/TopBar";
import Dashboard from "./Dashboard";
import Signals from "./Signals";
import SignalDetail from "./SignalDetail";
import Portfolio from "./Portfolio";
import Backtest from "./Backtest";
import GreyscaleMonitor from "./GreyscaleMonitor";

const PAGE_TITLES: Record<string, { title: string; subtitle: string }> = {
  dashboard: { title: "Dashboard", subtitle: "Model Output · Not Investment Advice" },
  signals: { title: "Signal Feed", subtitle: "Quantitative Model Signals · SEC Compliant" },
  "signal-detail": { title: "Signal Detail", subtitle: "Factor Decomposition · Model Output Only" },
  portfolio: { title: "Portfolio", subtitle: "Model Portfolio Analytics · Hypothetical" },
  backtest: { title: "Backtest Engine", subtitle: "Hypothetical Simulation · Not Predictive of Future Results" },
  greyscale: { title: "Greyscale Monitor", subtitle: "Trust Premium/Discount Tracker · Model Output" },
};

const Index = () => {
  const [activePage, setActivePage] = useState("dashboard");
  const [selectedTicker, setSelectedTicker] = useState("NVDA");

  const handleNavigate = (page: string) => {
    setActivePage(page);
    console.log("Navigating to:", page);
  };

  const handleSelectSignal = (ticker: string) => {
    setSelectedTicker(ticker);
    setActivePage("signal-detail");
    console.log("Selected signal for ticker:", ticker);
  };

  const meta = PAGE_TITLES[activePage] ?? PAGE_TITLES["dashboard"];

  const renderPage = () => {
    switch (activePage) {
      case "dashboard":
        return <Dashboard />;
      case "signals":
        return <Signals onSelectSignal={handleSelectSignal} />;
      case "signal-detail":
        return <SignalDetail ticker={selectedTicker} onBack={() => setActivePage("signals")} />;
      case "portfolio":
        return <Portfolio />;
      case "backtest":
        return <Backtest />;
      case "greyscale":
        return <GreyscaleMonitor />;
      default:
        return <Dashboard />;
    }
  };

  return (
    <div className="flex h-screen w-full overflow-hidden bg-background" style={{ minWidth: "1440px" }}>
      <Sidebar activePage={activePage} onNavigate={handleNavigate} />
      <div className="flex flex-col flex-1 min-w-0 overflow-hidden">
        <TopBar title={meta.title} subtitle={meta.subtitle} />
        <div className="flex-1 overflow-hidden flex flex-col">
          {renderPage()}
        </div>
      </div>
    </div>
  );
};

export default Index;
