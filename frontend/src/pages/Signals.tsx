import { useState, useMemo, useEffect } from "react";
import { Filter, Search, SortDesc, Download, RefreshCw, AlertCircle, ChevronLeft, ChevronRight } from "lucide-react";
import { useQuery } from "@tanstack/react-query";
import SignalRow from "../components/SignalRow";
import { fetchApi } from "../hooks/useApi";

interface Prediction {
  ticker: string;
  score: number;
  rank: number;
  percentile: number;
}

interface LatestPredictionsResponse {
  signal_date: string;
  week_number: number;
  universe_size: number;
  predictions: Prediction[];
}

const DIRECTIONS = ["All", "Long Signals", "Short Signals"];
const PAGE_SIZE = 10;

const Signals = ({ onSelectSignal = (_ticker: string) => {} }: { onSelectSignal?: (ticker: string) => void }) => {
  const [search, setSearch] = useState("");
  const [direction, setDirection] = useState("All");
  const [minConf, setMinConf] = useState(0);
  const [sort, setSort] = useState<"confidence" | "alpha" | "rank">("confidence");
  const [page, setPage] = useState(1);

  const { data, isLoading, error, refetch, isFetching } = useQuery<LatestPredictionsResponse>({
    queryKey: ["latestPredictions"],
    queryFn: () => fetchApi<LatestPredictionsResponse>("/api/predictions/latest"),
  });

  const predictions = data?.predictions || [];

  const filtered = useMemo(() => {
    return predictions
      .map(p => ({
        ticker: p.ticker,
        name: p.ticker, 
        direction: p.score > 0 ? "long" as const : "short" as const,
        confidence: Math.round(p.percentile),
        alpha: p.score,
        rank: p.rank,
        time: data?.signal_date || "Current",
        sector: "N/A",
        sparkData: [] as number[],
      }))
      .filter((s) => {
        const matchSearch = s.ticker.toLowerCase().includes(search.toLowerCase());
        const matchDir =
          direction === "All" ||
          (direction === "Long Signals" && s.direction === "long") ||
          (direction === "Short Signals" && s.direction === "short");
        const matchConf = s.confidence >= minConf;
        return matchSearch && matchDir && matchConf;
      })
      .sort((a, b) => {
        if (sort === "confidence") return b.confidence - a.confidence;
        if (sort === "alpha") return Math.abs(b.alpha) - Math.abs(a.alpha);
        if (sort === "rank") return a.rank - b.rank;
        return 0;
      });
  }, [predictions, search, direction, minConf, sort, data?.signal_date]);

  // Reset to page 1 when filters change
  useEffect(() => {
    setPage(1);
  }, [search, direction, minConf, sort]);

  const totalPages = Math.ceil(filtered.length / PAGE_SIZE);
  const paginated = filtered.slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE);

  const longCount = predictions.filter((p) => p.score > 0).length;
  const shortCount = predictions.filter((p) => p.score <= 0).length;

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
            onClick={() => refetch()}
            disabled={isFetching}
            className="flex items-center gap-1.5 px-3 py-2 rounded-lg text-sm text-muted-foreground hover:text-foreground hover:bg-accent border border-border transition-all duration-200 disabled:opacity-50"
          >
            <RefreshCw size={14} className={isFetching ? "animate-spin" : ""} />
            {isFetching ? "Refreshing..." : "Refresh"}
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
            {(["confidence", "alpha", "rank"] as const).map((s) => (
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
          <div className="w-20 text-xs font-semibold text-muted-foreground uppercase tracking-wider text-right">Score</div>
          <div className="w-20 text-xs font-semibold text-muted-foreground uppercase tracking-wider text-center">Trend</div>
          <div className="w-28 text-xs font-semibold text-muted-foreground uppercase tracking-wider text-right">Date</div>
          <div className="w-4" />
        </div>

        {isLoading ? (
          <div className="p-8 space-y-4">
            {Array.from({ length: 5 }).map((_, i) => (
              <div key={i} className="flex items-center gap-4 animate-pulse">
                <div className="h-10 bg-muted rounded w-24" />
                <div className="h-10 bg-muted rounded w-28" />
                <div className="h-4 bg-muted rounded flex-1" />
                <div className="h-10 bg-muted rounded w-20" />
                <div className="h-10 bg-muted rounded w-20" />
                <div className="h-10 bg-muted rounded w-28" />
              </div>
            ))}
          </div>
        ) : error ? (
          <div className="flex flex-col items-center justify-center py-16 text-muted-foreground">
            <AlertCircle size={32} className="mb-3 text-bear opacity-80" />
            <p className="text-sm">Failed to load signals: {(error as Error).message}</p>
            <button onClick={() => refetch()} className="mt-4 text-xs text-primary hover:underline">Try again</button>
          </div>
        ) : paginated.length > 0 ? (
          paginated.map((s, i) => (
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

      {/* Pagination Controls */}
      {totalPages > 1 && (
        <div className="flex items-center justify-between px-2 py-4 fade-in-up">
          <p className="text-xs text-muted-foreground">
            Showing <span className="font-semibold text-foreground">{(page - 1) * PAGE_SIZE + 1}</span> to <span className="font-semibold text-foreground">{Math.min(page * PAGE_SIZE, filtered.length)}</span> of <span className="font-semibold text-foreground">{filtered.length}</span> signals
          </p>
          <div className="flex items-center gap-1.5">
            <button
              onClick={() => setPage(p => Math.max(1, p - 1))}
              disabled={page === 1}
              className="p-1.5 rounded-lg border border-border hover:bg-accent disabled:opacity-30 disabled:hover:bg-transparent transition-colors"
            >
              <ChevronLeft size={16} />
            </button>
            <div className="flex items-center gap-1">
              {Array.from({ length: Math.min(5, totalPages) }).map((_, i) => {
                let pageNum = page;
                if (totalPages <= 5) {
                  pageNum = i + 1;
                } else if (page <= 3) {
                  pageNum = i + 1;
                } else if (page >= totalPages - 2) {
                  pageNum = totalPages - 4 + i;
                } else {
                  pageNum = page - 2 + i;
                }
                return (
                  <button
                    key={pageNum}
                    onClick={() => setPage(pageNum)}
                    className={`w-8 h-8 rounded-lg text-xs font-medium transition-all ${
                      page === pageNum ? "bg-primary text-primary-foreground shadow-sm" : "hover:bg-accent text-muted-foreground"
                    }`}
                  >
                    {pageNum}
                  </button>
                );
              })}
            </div>
            <button
              onClick={() => setPage(p => Math.min(totalPages, p + 1))}
              disabled={page === totalPages}
              className="p-1.5 rounded-lg border border-border hover:bg-accent disabled:opacity-30 disabled:hover:bg-transparent transition-colors"
            >
              <ChevronRight size={16} />
            </button>
          </div>
        </div>
      )}

      <p className="text-xs text-muted-foreground text-center pb-2">
        SEC compliant disclosure · For informational purposes only
      </p>
    </div>
  );
};

export default Signals;
