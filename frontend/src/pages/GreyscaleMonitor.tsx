import { useState, useEffect } from "react";
import { Eye, AlertOctagon, RefreshCw, TrendingUp, TrendingDown, Clock } from "lucide-react";
import MiniSparkline from "../components/MiniSparkline";

interface GrayAsset {
  symbol: string;
  name: string;
  price: number;
  change: number;
  premium: number;
  nav: number;
  volume: string;
  aum: string;
  status: "normal" | "warning" | "critical";
  sparkData: number[];
}

const ASSETS: GrayAsset[] = [
  { symbol: "GBTC", name: "Grayscale Bitcoin Trust", price: 58.42, change: 3.21, premium: -8.4, nav: 63.77, volume: "$284M", aum: "$21.8B", status: "normal", sparkData: [48, 52, 50, 55, 53, 58, 56, 60, 58, 63] },
  { symbol: "ETHE", name: "Grayscale Ethereum Trust", price: 28.14, change: -1.82, premium: -12.7, nav: 32.24, volume: "$42M", aum: "$7.2B", status: "warning", sparkData: [32, 28, 30, 25, 27, 22, 24, 19, 21, 16] },
  { symbol: "GDLC", name: "Grayscale Digital Large Cap", price: 42.87, change: 1.54, premium: -6.1, nav: 45.65, volume: "$8.2M", aum: "$580M", status: "normal", sparkData: [38, 40, 39, 42, 41, 44, 43, 45, 44, 47] },
  { symbol: "GLTC", name: "Grayscale Litecoin Trust", price: 12.34, change: -4.21, premium: -18.3, nav: 15.10, volume: "$2.1M", aum: "$210M", status: "critical", sparkData: [18, 16, 17, 14, 15, 12, 13, 10, 11, 8] },
  { symbol: "GETH", name: "Grayscale Ethereum Classic Trust", price: 8.92, change: 0.87, premium: -9.2, nav: 9.82, volume: "$1.4M", aum: "$98M", status: "normal", sparkData: [7, 8, 8, 9, 8, 9, 9, 10, 9, 10] },
  { symbol: "BCHG", name: "Grayscale Bitcoin Cash Trust", price: 6.71, change: -2.34, premium: -15.6, nav: 7.95, volume: "$0.8M", aum: "$42M", status: "warning", sparkData: [9, 8, 8, 7, 7, 6, 7, 6, 6, 5] },
];

function useLivePrice(initial: number) {
  const [price, setPrice] = useState(initial);
  useEffect(() => {
    const interval = setInterval(() => {
      setPrice((p) => parseFloat((p + (Math.random() - 0.5) * 0.1).toFixed(2)));
    }, 2000 + Math.random() * 1000);
    return () => clearInterval(interval);
  }, []);
  return price;
}

const LivePriceCell = ({ price: initPrice, isPos }: { price: number; isPos: boolean }) => {
  const price = useLivePrice(initPrice);
  return (
    <span className={`text-sm font-bold font-mono transition-colors duration-300 ${isPos ? "text-bull" : "text-bear"}`}>
      ${price.toFixed(2)}
    </span>
  );
};

const statusConfig = {
  normal: { label: "Normal", className: "tag-bull" },
  warning: { label: "Warning", className: "tag-neutral" },
  critical: { label: "Critical", className: "tag-bear" },
};

const GreyscaleMonitor = () => {
  const [lastUpdate, setLastUpdate] = useState(new Date());
  const [refreshing, setRefreshing] = useState(false);
  const [sortBy, setSortBy] = useState<"premium" | "change" | "aum">("premium");
  const [alertsOnly, setAlertsOnly] = useState(false);

  const handleRefresh = () => {
    setRefreshing(true);
    setTimeout(() => {
      setRefreshing(false);
      setLastUpdate(new Date());
    }, 1200);
  };

  const sorted = [...ASSETS]
    .filter((a) => !alertsOnly || a.status !== "normal")
    .sort((a, b) => {
      if (sortBy === "premium") return a.premium - b.premium;
      if (sortBy === "change") return b.change - a.change;
      return 0;
    });

  const avgPremium = (ASSETS.reduce((sum, a) => sum + a.premium, 0) / ASSETS.length).toFixed(1);
  const warningCount = ASSETS.filter((a) => a.status !== "normal").length;

  return (
    <div className="flex-1 overflow-y-auto p-6 space-y-5">
      {/* Header */}
      <div className="flex items-center justify-between fade-in-up">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-primary/10 border border-primary/20 flex items-center justify-center">
            <Eye size={18} className="text-primary" />
          </div>
          <div>
            <h2 className="text-xl font-bold text-foreground">Greyscale Monitor</h2>
            <p className="text-xs text-muted-foreground mt-0.5">Real-time trust premium/discount monitoring · Model Output</p>
          </div>
        </div>
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
            <Clock size={12} />
            <span>Updated: {lastUpdate.toLocaleTimeString()}</span>
          </div>
          <button
            onClick={handleRefresh}
            className="flex items-center gap-1.5 px-3 py-2 rounded-lg text-sm text-muted-foreground hover:text-foreground hover:bg-accent border border-border transition-all duration-200"
          >
            <RefreshCw size={14} className={refreshing ? "animate-spin" : ""} />
            Refresh
          </button>
        </div>
      </div>

      {/* Alert Banner */}
      {warningCount > 0 && (
        <div className="flex items-center gap-3 px-4 py-3 rounded-xl bg-destructive/10 border border-destructive/20 fade-in-up stagger-1">
          <AlertOctagon size={15} className="text-destructive flex-shrink-0" />
          <p className="text-sm text-destructive font-medium">
            {warningCount} trust(s) showing elevated discount / volatility — monitor closely
          </p>
          <button
            onClick={() => setAlertsOnly((v) => !v)}
            className="ml-auto text-xs font-semibold text-destructive hover:opacity-70 transition-opacity"
          >
            {alertsOnly ? "Show All" : "Show Alerts Only"}
          </button>
        </div>
      )}

      {/* Summary Metrics */}
      <div className="flex gap-4 fade-in-up stagger-1">
        {[
          { label: "Avg Premium/Discount", value: `${avgPremium}%`, icon: "activity", positive: false },
          { label: "Total Trusts Monitored", value: `${ASSETS.length}`, icon: "eye", positive: true },
          { label: "Alerts Active", value: `${warningCount}`, icon: "alert", positive: false },
          { label: "Largest Discount", value: "-18.3%", icon: "trend-down", positive: false },
        ].map(({ label, value, positive }, i) => (
          <div
            key={label}
            className={`flex-1 bg-card rounded-xl border p-4 card-hover fade-in-up ${positive ? "border-border" : "border-bear/20"}`}
            style={{ animationDelay: `${i * 50}ms` }}
          >
            <div className="text-xs text-muted-foreground uppercase tracking-wider mb-2">{label}</div>
            <div className={`text-2xl font-bold font-mono ${positive ? "text-foreground" : "text-bear"}`}>{value}</div>
          </div>
        ))}
      </div>

      {/* Filter Controls */}
      <div className="flex items-center justify-between fade-in-up stagger-2">
        <div className="flex items-center gap-2">
          <TrendingUp size={13} className="text-muted-foreground" />
          <span className="text-xs text-muted-foreground">Sort by:</span>
          {(["premium", "change", "aum"] as const).map((s) => (
            <button
              key={s}
              onClick={() => setSortBy(s)}
              className={`text-xs px-2.5 py-1.5 rounded-lg font-medium capitalize transition-all duration-200 ${
                sortBy === s ? "bg-primary text-primary-foreground" : "text-muted-foreground hover:text-foreground hover:bg-accent"
              }`}
            >
              {s === "aum" ? "AUM" : s.charAt(0).toUpperCase() + s.slice(1)}
            </button>
          ))}
        </div>
        <p className="text-xs text-muted-foreground">
          Showing {sorted.length} of {ASSETS.length} trusts
        </p>
      </div>

      {/* Table */}
      <div className="bg-card rounded-xl border border-border overflow-hidden fade-in-up stagger-3">
        {/* Header */}
        <div className="flex items-center px-5 py-3 border-b border-border bg-muted/20 text-xs font-semibold text-muted-foreground uppercase tracking-wider">
          <div className="w-32">Trust</div>
          <div className="flex-1">Name</div>
          <div className="w-24 text-right">Price</div>
          <div className="w-24 text-right">24H Change</div>
          <div className="w-24 text-right">NAV</div>
          <div className="w-28 text-right">Premium/Disc</div>
          <div className="w-24 text-right">Volume</div>
          <div className="w-24 text-right">AUM</div>
          <div className="w-24 text-center">Status</div>
          <div className="w-20 text-center">Trend</div>
        </div>

        {sorted.map((asset, i) => {
          const isPos = asset.change >= 0;
          const isPremiumPos = asset.premium >= 0;
          const status = statusConfig[asset.status];

          return (
            <div
              key={asset.symbol}
              className="flex items-center px-5 py-4 border-b border-border last:border-0 hover:bg-accent/40 transition-all duration-200 cursor-pointer fade-in-up"
              style={{ animationDelay: `${i * 50}ms` }}
            >
              <div className="w-32">
                <div className="text-sm font-bold text-foreground">{asset.symbol}</div>
              </div>
              <div className="flex-1 pr-4">
                <div className="text-sm text-foreground truncate">{asset.name}</div>
              </div>
              <div className="w-24 text-right">
                <LivePriceCell price={asset.price} isPos={isPos} />
              </div>
              <div className="w-24 text-right">
                <div className={`flex items-center justify-end gap-1 text-sm font-semibold ${isPos ? "text-bull" : "text-bear"}`}>
                  {isPos ? <TrendingUp size={12} /> : <TrendingDown size={12} />}
                  {isPos ? "+" : ""}{asset.change.toFixed(2)}%
                </div>
              </div>
              <div className="w-24 text-right">
                <span className="text-sm text-muted-foreground font-mono">${asset.nav.toFixed(2)}</span>
              </div>
              <div className="w-28 text-right">
                <div className={`text-sm font-bold font-mono ${isPremiumPos ? "text-bull" : "text-bear"}`}>
                  {isPremiumPos ? "+" : ""}{asset.premium.toFixed(1)}%
                </div>
                {/* Premium bar */}
                <div className="h-1 w-full bg-muted rounded-full mt-1 overflow-hidden">
                  <div
                    className="h-full rounded-full"
                    style={{
                      width: `${Math.min(100, (Math.abs(asset.premium) / 25) * 100)}%`,
                      backgroundColor: isPremiumPos ? "#00C805" : "#FF5252",
                      marginLeft: isPremiumPos ? "50%" : `${50 - Math.min(50, (Math.abs(asset.premium) / 25) * 50)}%`,
                    }}
                  />
                </div>
              </div>
              <div className="w-24 text-right">
                <span className="text-xs text-muted-foreground">{asset.volume}</span>
              </div>
              <div className="w-24 text-right">
                <span className="text-xs font-semibold text-foreground">{asset.aum}</span>
              </div>
              <div className="w-24 flex justify-center">
                <span className={`text-xs font-semibold px-2 py-0.5 rounded-md ${status.className}`}>
                  {status.label}
                </span>
              </div>
              <div className="w-20 flex justify-center">
                <MiniSparkline data={asset.sparkData} positive={isPos} width={60} height={26} animated={false} />
              </div>
            </div>
          );
        })}
      </div>

      <p className="text-xs text-muted-foreground text-center pb-2">
        Greyscale trust data · Model output only · Not investment advice · SEC compliant disclosure
      </p>
    </div>
  );
};

export default GreyscaleMonitor;
