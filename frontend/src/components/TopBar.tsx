import { useState, useEffect } from "react";
import { Search, Bell, RefreshCw, Calendar } from "lucide-react";

interface TopBarProps {
  title?: string;
  subtitle?: string;
}

const TopBar = ({
  title = "Dashboard",
  subtitle = "Model Output · Not Investment Advice",
}: TopBarProps) => {
  const [time, setTime] = useState(new Date());
  const [searching, setSearching] = useState(false);
  const [query, setQuery] = useState("");
  const [refreshing, setRefreshing] = useState(false);

  useEffect(() => {
    const t = setInterval(() => setTime(new Date()), 1000);
    return () => clearInterval(t);
  }, []);

  const handleRefresh = () => {
    setRefreshing(true);
    setTimeout(() => setRefreshing(false), 1200);
  };

  const formatTime = (d: Date) =>
    d.toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false });

  const formatDate = (d: Date) =>
    d.toLocaleDateString("en-US", { weekday: "short", month: "short", day: "numeric", year: "numeric" });

  return (
    <div data-cmp="TopBar" className="flex items-center justify-between px-6 py-3.5 border-b border-border bg-surface/80 backdrop-blur-sm flex-shrink-0">
      {/* Left: Title */}
      <div>
        <h1 className="text-lg font-bold text-foreground leading-tight">{title}</h1>
        <p className="text-xs text-muted-foreground mt-0.5">{subtitle}</p>
      </div>

      {/* Center: Search */}
      <div className="flex-1 max-w-xs mx-8">
        <div className={`flex items-center gap-2 px-3 py-2 rounded-lg bg-muted border transition-all duration-200 ${searching ? "border-primary/50" : "border-transparent"}`}>
          <Search size={14} className="text-muted-foreground flex-shrink-0" />
          <input
            type="text"
            placeholder="Search tickers, signals…"
            value={query}
            onFocus={() => setSearching(true)}
            onBlur={() => setSearching(false)}
            onChange={(e) => setQuery(e.target.value)}
            className="bg-transparent text-sm text-foreground placeholder:text-muted-foreground outline-none w-full"
          />
        </div>
      </div>

      {/* Right: Actions */}
      <div className="flex items-center gap-3">
        <div className="flex items-center gap-1.5 text-muted-foreground">
          <Calendar size={13} />
          <span className="text-xs">{formatDate(time)}</span>
          <span className="text-xs font-mono text-primary ml-1">{formatTime(time)}</span>
        </div>

        <div className="w-px h-4 bg-border" />

        <button
          onClick={handleRefresh}
          className="p-1.5 rounded-lg text-muted-foreground hover:text-foreground hover:bg-accent transition-all duration-200"
        >
          <RefreshCw size={15} className={refreshing ? "animate-spin" : ""} />
        </button>

        <button className="relative p-1.5 rounded-lg text-muted-foreground hover:text-foreground hover:bg-accent transition-all duration-200">
          <Bell size={15} />
          <span className="absolute top-1 right-1 w-1.5 h-1.5 rounded-full bg-primary pulse-dot" />
        </button>

        <div className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg tag-bull text-xs font-semibold">
          <span className="w-1.5 h-1.5 rounded-full bg-primary pulse-dot" />
          LIVE
        </div>
      </div>
    </div>
  );
};

export default TopBar;
