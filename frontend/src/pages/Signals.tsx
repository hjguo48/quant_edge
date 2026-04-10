import { useState } from "react";
import { Filter, Search, SortDesc, Download, RefreshCw } from "lucide-react";
import SignalRow from "../components/SignalRow";

const SIGNALS_DATA = [
  { ticker: "NVDA", name: "NVIDIA Corp.", direction: "long" as const, confidence: 91, alpha: 3.24, time: "5m ago", sector: "Technology", sparkData: [18, 22, 20, 28, 25, 32, 30, 38, 35, 42] },
  { ticker: "AAPL", name: "Apple Inc.", direction: "long" as const, confidence: 83, alpha: 1.87, time: "8m ago", sector: "Technology", sparkData: [10, 12, 11, 15, 14, 18, 16, 20, 19, 24] },
  { ticker: "TSLA", name: "Tesla Inc.", direction: "short" as const, confidence: 74, alpha: -1.82, time: "12m ago", sector: "Consumer", sparkData: [32, 28, 30, 25, 27, 22, 24, 19, 21, 16] },
  { ticker: "MSFT", name: "Microsoft Corp.", direction: "long" as const, confidence: 83, alpha: 2.14, time: "18m ago", sector: "Technology", sparkData: [15, 17, 16, 20, 19, 22, 21, 25, 23, 27] },
  { ticker: "META", name: "Meta Platforms", direction: "long" as const, confidence: 67, alpha: 1.41, time: "31m ago", sector: "Technology", sparkData: [8, 11, 10, 13, 12, 16, 14, 18, 16, 20] },
  { ticker: "AMZN", name: "Amazon.com", direction: "short" as const, confidence: 58, alpha: -0.93, time: "45m ago", sector: "Consumer", sparkData: [22, 20, 21, 18, 19, 16, 17, 14, 15, 12] },
  { ticker: "JPM", name: "JPMorgan Chase", direction: "long" as const, confidence: 76, alpha: 1.56, time: "1h ago", sector: "Finance", sparkData: [12, 14, 13, 16, 15, 18, 17, 20, 19, 22] },
  { ticker: "XOM", name: "Exxon Mobil", direction: "short" as const, confidence: 62, alpha: -1.12, time: "1h ago", sector: "Energy", sparkData: [28, 25, 26, 22, 24, 20, 22, 18, 20, 16] },
  { ticker: "UNH", name: "UnitedHealth Group", direction: "long" as const, confidence: 88, alpha: 2.67, time: "2h ago", sector: "Healthcare", sparkData: [14, 16, 15, 19, 17, 22, 20, 24, 22, 27] },
  { ticker: "V", name: "Visa Inc.", direction: "long" as const, confidence: 71, alpha: 1.23, time: "2h ago", sector: "Finance", sparkData: [10, 12, 11, 14, 13, 16, 15, 18, 17, 20] },
  { ticker: "HD", name: "Home Depot", direction: "short" as const, confidence: 55, alpha: -0.74, time: "3h ago", sector: "Consumer", sparkData: [20, 18, 19, 16, 17, 14, 15, 12, 13, 10] },
  { ticker: "GOOGL", name: "Alphabet Inc.", direction: "long" as const, confidence: 79, alpha: 1.98, time: "3h ago", sector: "Technology", sparkData: [15, 17, 16, 20, 18, 23, 21, 25, 23, 28] },
];

const SECTORS = ["All Sectors", "Technology", "Finance", "Healthcare", "Energy", "Consumer"];
const DIRECTIONS = ["All", "Long Signals", "Short Signals", "Neutral"];

const Signals = ({ onSelectSignal = (_ticker: string) => {} }: { onSelectSignal?: (ticker: string) => void }) => {
  const [search, setSearch] = useState("");
  const [sector, setSector] = useState("All Sectors");
  const [direction, setDirection] = useState("All");
  const [minConf, setMinConf] = useState(0);
  const [sort, setSort] = useState<"confidence" | "alpha" | "time">("confidence");
  const [refreshing, setRefreshing] = useState(false);

  const filtered = SIGNALS_DATA
    .filter((s) => {
      const matchSearch = s.ticker.toLowerCase().includes(search.toLowerCase()) || s.name.toLowerCase().includes(search.toLowerCase());
      const matchSector = sector === "All Sectors" || s.sector === sector;
      const matchDir =
        direction === "All" ||
        (direction === "Long Signals" && s.direction === "long") ||
        (direction === "Short Signals" && s.direction === "short");
      const matchConf = s.confidence >= minConf;
      return matchSearch && matchSector && matchDir && matchConf;
    })
    .sort((a, b) => {
      if (sort === "confidence") return b.confidence - a.confidence;
      if (sort === "alpha") return Math.abs(b.alpha) - Math.abs(a.alpha);
      return 0;
    });

  const longCount = SIGNALS_DATA.filter((s) => s.direction === "long").length;
  const shortCount = SIGNALS_DATA.filter((s) => s.direction === "short").length;

  return (
    <div className="flex-1 overflow-y-auto p-6 space-y-5">
      {/* Header */}
      <div className="flex items-center justify-between fade-in-up">
        <div>
          <h2 className="text-xl font-bold text-foreground">Signal Feed</h2>
          <p className="text-sm text-muted-foreground mt-0.5">
            Model-generated signals · <span className="text-bull">{longCount} long</span> · <span className="text-bear">{shortCount} short</span> · Not investment advice
          </p>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => { setRefreshing(true); setTimeout(() => setRefreshing(false), 1000); }}
            className="flex items-center gap-1.5 px-3 py-2 rounded-lg text-sm text-muted-foreground hover:text-foreground hover:bg-accent border border-border transition-all duration-200"
          >
            <RefreshCw size={14} className={refreshing ? "animate-spin" : ""} />
            Refresh
          </button>
          <button className="flex items-center gap-1.5 px-3 py-2 rounded-lg text-sm bg-primary text-primary-foreground btn-primary">
            <Download size={14} />
            Export CSV
          </button>
        </div>
      </div>

      {/* Filters */}
      <div className="bg-card rounded-xl border border-border p-4 fade-in-up stagger-1">
        <div className="flex items-center gap-3 flex-wrap">
          <Filter size={14} className="text-muted-foreground" />

          {/* Search */}
          <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-muted border border-transparent focus-within:border-primary/40 transition-all">
            <Search size={13} className="text-muted-foreground" />
            <input
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search ticker…"
              className="bg-transparent text-sm text-foreground placeholder:text-muted-foreground outline-none w-32"
            />
          </div>

          {/* Direction */}
          <div className="flex items-center gap-1">
            {DIRECTIONS.map((d) => (
              <button
                key={d}
                onClick={() => setDirection(d)}
                className={`text-xs px-2.5 py-1.5 rounded-lg font-medium transition-all duration-200 ${
                  direction === d ? "bg-primary text-primary-foreground" : "text-muted-foreground hover:text-foreground hover:bg-accent"
                }`}
              >
                {d}
              </button>
            ))}
          </div>

          <div className="w-px h-5 bg-border" />

          {/* Sector */}
          <select
            value={sector}
            onChange={(e) => setSector(e.target.value)}
            className="bg-muted text-sm text-foreground px-3 py-2 rounded-lg border border-transparent outline-none cursor-pointer hover:bg-accent transition-colors"
          >
            {SECTORS.map((s) => (
              <option key={s} value={s} className="bg-popover">{s}</option>
            ))}
          </select>

          {/* Min Confidence */}
          <div className="flex items-center gap-2">
            <span className="text-xs text-muted-foreground whitespace-nowrap">Min conf:</span>
            <input
              type="range"
              min={0}
              max={90}
              step={5}
              value={minConf}
              onChange={(e) => setMinConf(Number(e.target.value))}
              className="w-24 accent-primary"
            />
            <span className="text-xs font-mono text-foreground w-6">{minConf}%</span>
          </div>

          <div className="ml-auto flex items-center gap-1.5">
            <SortDesc size={13} className="text-muted-foreground" />
            <span className="text-xs text-muted-foreground">Sort:</span>
            {(["confidence", "alpha", "time"] as const).map((s) => (
              <button
                key={s}
                onClick={() => setSort(s)}
                className={`text-xs px-2 py-1 rounded-md transition-colors ${sort === s ? "text-primary font-semibold" : "text-muted-foreground hover:text-foreground"}`}
              >
                {s.charAt(0).toUpperCase() + s.slice(1)}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Table */}
      <div className="bg-card rounded-xl border border-border overflow-hidden fade-in-up stagger-2">
        {/* Header Row */}
        <div className="flex items-center gap-4 px-5 py-3 border-b border-border bg-muted/30">
          <div className="w-24 text-xs font-semibold text-muted-foreground uppercase tracking-wider">Ticker</div>
          <div className="w-28 text-xs font-semibold text-muted-foreground uppercase tracking-wider">Signal</div>
          <div className="flex-1 text-xs font-semibold text-muted-foreground uppercase tracking-wider">Confidence</div>
          <div className="w-20 text-xs font-semibold text-muted-foreground uppercase tracking-wider text-right">Est. Alpha</div>
          <div className="w-20 text-xs font-semibold text-muted-foreground uppercase tracking-wider text-center">Trend</div>
          <div className="w-28 text-xs font-semibold text-muted-foreground uppercase tracking-wider text-right">Time</div>
          <div className="w-4" />
        </div>

        {filtered.length > 0 ? (
          filtered.map((s, i) => (
            <div key={s.ticker} className="fade-in-up" style={{ animationDelay: `${i * 40}ms` }}>
              <SignalRow
                {...s}
                onClick={() => onSelectSignal(s.ticker)}
              />
            </div>
          ))
        ) : (
          <div className="flex flex-col items-center justify-center py-16 text-muted-foreground">
            <Search size={32} className="mb-3 opacity-30" />
            <p className="text-sm">No signals match your filters</p>
          </div>
        )}
      </div>

      <p className="text-xs text-muted-foreground text-center pb-2">
        Showing {filtered.length} of {SIGNALS_DATA.length} model outputs · SEC compliant disclosure · For informational purposes only
      </p>
    </div>
  );
};

export default Signals;
