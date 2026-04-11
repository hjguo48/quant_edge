import { useLocation, useNavigate, useParams } from "react-router-dom";
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

function getPageFromPath(pathname: string): string {
  if (pathname === "/" || pathname === "") return "dashboard";
  if (pathname === "/signals") return "signals";
  if (pathname.startsWith("/signals/")) return "signal-detail";
  if (pathname === "/portfolio") return "portfolio";
  if (pathname === "/backtest") return "backtest";
  if (pathname === "/greyscale") return "greyscale";
  return "dashboard";
}

const Index = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const { ticker } = useParams<{ ticker?: string }>();
  const activePage = getPageFromPath(location.pathname);
  const selectedTicker = (ticker || "NVDA").toUpperCase();

  const handleNavigate = (page: string) => {
    switch (page) {
      case "dashboard":
        navigate("/");
        return;
      case "signals":
        navigate("/signals");
        return;
      case "portfolio":
        navigate("/portfolio");
        return;
      case "backtest":
        navigate("/backtest");
        return;
      case "greyscale":
        navigate("/greyscale");
        return;
      default:
        navigate("/");
    }
  };

  const handleSelectSignal = (ticker: string) => {
    navigate(`/signals/${ticker.toUpperCase()}`);
  };

  const meta = PAGE_TITLES[activePage] ?? PAGE_TITLES["dashboard"];

  const renderPage = () => {
    switch (activePage) {
      case "dashboard":
        return <Dashboard onSelectSignal={handleSelectSignal} />;
      case "signals":
        return <Signals onSelectSignal={handleSelectSignal} />;
      case "signal-detail":
        return <SignalDetail ticker={selectedTicker} onBack={() => navigate("/signals")} />;
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
    <div className="flex h-screen w-full overflow-hidden bg-background">
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
