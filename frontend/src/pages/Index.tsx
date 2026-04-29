import { lazy, Suspense } from "react";
import { useLocation, useNavigate, useParams } from "react-router-dom";
import { useTranslation } from "react-i18next";
import Sidebar from "../components/Sidebar";
import TopBar from "../components/TopBar";

const Dashboard = lazy(() => import("./Dashboard"));
const Signals = lazy(() => import("./Signals"));
const SignalDetail = lazy(() => import("./SignalDetail"));
const Portfolio = lazy(() => import("./Portfolio"));
const Backtest = lazy(() => import("./Backtest"));
const GreyscaleMonitor = lazy(() => import("./GreyscaleMonitor"));

const PageFallback = () => {
  const { t } = useTranslation();
  return (
    <div className="flex-1 flex items-center justify-center text-xs font-medium uppercase tracking-widest text-muted-foreground">
      {t("common.loading")}
    </div>
  );
};

const PAGE_TITLE_KEYS: Record<string, string> = {
  dashboard: "dashboard",
  signals: "signals",
  "signal-detail": "signalDetail",
  portfolio: "portfolio",
  backtest: "backtest",
  greyscale: "greyscale",
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
  const { t } = useTranslation();
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

  const titleKey = PAGE_TITLE_KEYS[activePage] ?? "dashboard";
  const meta = {
    title: t(`pageTitles.${titleKey}.title`),
    subtitle: t(`pageTitles.${titleKey}.subtitle`),
  };

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
          <Suspense fallback={<PageFallback />}>{renderPage()}</Suspense>
        </div>
      </div>
    </div>
  );
};

export default Index;
