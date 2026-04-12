import { useState, useMemo, useEffect, useRef, useCallback, type CSSProperties } from "react";
import { createPortal } from "react-dom";
import { Filter, Search, SortDesc, Download, RefreshCw, AlertCircle, ChevronLeft, ChevronRight, ChevronDown, Star } from "lucide-react";
import { useQuery } from "@tanstack/react-query";
import SignalRow from "../components/SignalRow";
import { fetchApi } from "../hooks/useApi";
import { getSectorColor, PRIMARY_SECTORS } from "../constants/sectorColors";

interface Prediction {
  ticker: string;
  score: number;
  rank: number;
  percentile: number;
  sector?: string | null;
  company_name?: string | null;
}

interface LatestPredictionsResponse {
  signal_date: string;
  week_number: number;
  universe_size: number;
  predictions: Prediction[];
}

const DIRECTIONS = ["All", "Long Signals", "Short Signals"];
const PAGE_SIZE = 10;
const SORT_OPTIONS = [
  { key: "none", label: "Default" },
  { key: "conf_desc", label: "Conf \u2193" },
  { key: "conf_asc", label: "Conf \u2191" },
  { key: "score", label: "Score" },
  { key: "strength", label: "Strength" },
] as const;
type SortMode = (typeof SORT_OPTIONS)[number]["key"];

const formatDateShort = (dateStr?: string) => {
  if (!dateStr || dateStr === "Current") return "Current";
  const date = new Date(dateStr);
  if (isNaN(date.getTime())) return dateStr;
  return date.toLocaleDateString("en-US", { month: "short", day: "numeric" });
};

const interpolateColor = (progress: number) => {
  const colors = [
    { p: 0, r: 255, g: 82, b: 82, a: 0.85 },
    { p: 25, r: 255, g: 82, b: 82, a: 0.4 },
    { p: 50, r: 96, g: 123, b: 150, a: 0.3 },
    { p: 75, r: 0, g: 200, b: 5, a: 0.4 },
    { p: 100, r: 0, g: 200, b: 5, a: 0.85 },
  ];

  let i = 0;
  while (i < colors.length - 2 && progress > colors[i + 1].p) {
    i++;
  }

  const c1 = colors[i];
  const c2 = colors[i + 1];
  const range = c2.p - c1.p;
  const t = (progress - c1.p) / (range || 1);

  const r = Math.round(c1.r + (c2.r - c1.r) * t);
  const g = Math.round(c1.g + (c2.g - c1.g) * t);
  const b = Math.round(c1.b + (c2.b - c1.b) * t);
  const a = (c1.a + (c2.a - c1.a) * t).toFixed(2);

  return `rgba(${r}, ${g}, ${b}, ${a})`;
};

function hashTickerSeed(value: string): number {
  let hash = 0;
  for (let index = 0; index < value.length; index += 1) {
    hash = (hash * 31 + value.charCodeAt(index)) >>> 0;
  }
  return hash;
}

function generateDirectionalSparkData(score: number, seedKey: string): number[] {
  const base = 50;
  const direction = score > 0 ? 1 : -1;
  const magnitude = Math.max(8, Math.min(Math.abs(score) * 5, 30));
  const seed = hashTickerSeed(seedKey);

  return Array.from({ length: 10 }, (_, index) => {
    const normalizedNoise = Math.sin(seed + (index + 1) * 12.9898) * 43758.5453;
    const noise = ((normalizedNoise - Math.floor(normalizedNoise)) - 0.5) * 4;
    return base + direction * (index / 9) * magnitude + noise;
  });
}

const Signals = ({ onSelectSignal = (_ticker: string) => {} }: { onSelectSignal?: (ticker: string) => void }) => {
  const [search, setSearch] = useState("");
  const [direction, setDirection] = useState("All");
  const [sectorFilter, setSectorFilter] = useState("All Sectors");
  const [sectorOpen, setSectorOpen] = useState(false);
  const [showWatchlist, setShowWatchlist] = useState(false);
  const sectorBtnRef = useRef<HTMLButtonElement>(null);
  const [dropdownPos, setDropdownPos] = useState({ top: 0, left: 0 });
  const [minConf, setMinConf] = useState(0);
  const [sort, setSort] = useState<SortMode>("none");
  const [page, setPage] = useState(1);

  // Watchlist state
  const [watchlist, setWatchlist] = useState<string[]>(() => {
    const saved = localStorage.getItem("watchlist");
    return saved ? JSON.parse(saved) : [];
  });

  useEffect(() => {
    localStorage.setItem("watchlist", JSON.stringify(watchlist));
  }, [watchlist]);

  const toggleWatchlist = (ticker: string) => {
    setWatchlist(prev => 
      prev.includes(ticker) 
        ? prev.filter(t => t !== ticker) 
        : [...prev, ticker]
    );
  };

  const toggleSectorDropdown = useCallback(() => {
    if (!sectorOpen && sectorBtnRef.current) {
      const rect = sectorBtnRef.current.getBoundingClientRect();
      setDropdownPos({ top: rect.bottom + 4, left: rect.left });
    }
    setSectorOpen((prev) => !prev);
  }, [sectorOpen]);

  const apiPath = useMemo(() => {
    if (showWatchlist) {
      if (watchlist.length === 0) return null;
      return `/api/predictions/batch?tickers=${watchlist.join(",")}`;
    }
    return "/api/predictions/latest";
  }, [showWatchlist, watchlist]);

  const { data, isLoading, error, refetch, isFetching } = useQuery<LatestPredictionsResponse>({
    queryKey: ["predictions", apiPath],
    queryFn: () => fetchApi<LatestPredictionsResponse>(apiPath!),
    retry: false,
    enabled: !!apiPath || !showWatchlist,
  });

  const predictions = data?.predictions || [];
  const sliderStyle = useMemo(
    () =>
      ({
        "--slider-progress": `${(minConf / 90) * 100}%`,
        "--slider-fill-color": interpolateColor((minConf / 90) * 100),
      }) as CSSProperties,
    [minConf],
  );

  const signalRows = useMemo(() => {
    return predictions.map((prediction) => ({
      ticker: prediction.ticker,
      name: prediction.company_name || prediction.ticker,
      direction: prediction.score > 0 ? ("long" as const) : ("short" as const),
      confidence: Math.round(prediction.percentile),
      score: prediction.score,
      rank: prediction.rank,
      sector: prediction.sector || "—",
      shuffleKey: hashTickerSeed(prediction.ticker + ":" + prediction.rank),
      sparkData: generateDirectionalSparkData(
        prediction.score,
        `${prediction.ticker}:${prediction.rank}:${prediction.score.toFixed(6)}`,
      ),
    }));
  }, [predictions]);

  const filtered = useMemo(() => {
    return signalRows
      .filter((s) => {
        const matchSearch = s.ticker.toLowerCase().includes(search.toLowerCase());
        const matchDir =
          direction === "All" ||
          (direction === "Long Signals" && s.direction === "long") ||
          (direction === "Short Signals" && s.direction === "short");
        const matchConf = s.confidence >= minConf;
        const matchSector = sectorFilter === "All Sectors" || s.sector === sectorFilter;
        return matchSearch && matchDir && matchConf && matchSector;
      })
      .sort((a, b) => {
        if (sort === "conf_desc") return b.confidence - a.confidence;
        if (sort === "conf_asc") return a.confidence - b.confidence;
        if (sort === "score") return b.score - a.score;
        if (sort === "strength") return Math.abs(b.score) - Math.abs(a.score);
        if (sort === "none") return a.shuffleKey - b.shuffleKey;
        return 0;
      });
  }, [signalRows, search, direction, minConf, sort, sectorFilter]);

  // Reset to page 1 when filters change
  useEffect(() => {
    setPage(1);
  }, [search, direction, minConf, sort, sectorFilter, showWatchlist]);

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
            Week {data?.week_number || "—"} · {formatDateShort(data?.signal_date)} · <span className="text-bull">{longCount} long</span> · <span className="text-bear">{shortCount} short</span> · Not investment advice
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

      {/* Filters Panel */}
      <div className="bg-card rounded-xl border border-border p-4 space-y-4 fade-in-up stagger-1">
        {/* Row 1: Search, Watchlist, and Sort */}
        <div className="flex items-center justify-between gap-4">
          <div className="flex items-center gap-3 flex-1 min-w-0">
            {/* Search */}
            <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-muted border border-transparent focus-within:border-primary/40 transition-all flex-1 max-w-sm">
              <Search size={13} className="text-muted-foreground" />
              <input
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                placeholder="Search ticker or name…"
                className="bg-transparent text-sm text-foreground placeholder:text-muted-foreground outline-none w-full"
              />
            </div>

            {/* Watchlist Toggle */}
            <button
              onClick={() => setShowWatchlist(!showWatchlist)}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm transition-all flex-shrink-0 ${
                showWatchlist ? "bg-primary text-primary-foreground font-bold" : "bg-muted text-muted-foreground hover:bg-accent hover:text-foreground"
              }`}
            >
              <Star size={14} className={showWatchlist ? "fill-current" : ""} />
              Watchlist
              <span className={`px-1.5 py-0.5 rounded-full text-[10px] ${showWatchlist ? "bg-white/20" : "bg-muted-foreground/20"}`}>
                {watchlist.length}
              </span>
            </button>
          </div>

          {/* Sort Controls */}
          <div className="flex items-center gap-2 flex-shrink-0">
            <SortDesc
              size={14}
              className="text-muted-foreground cursor-pointer hover:text-foreground transition-colors mr-1"
              onClick={() => {
                const keys = SORT_OPTIONS.map((o) => o.key);
                const idx = keys.indexOf(sort);
                setSort(keys[(idx + 1) % keys.length] as SortMode);
              }}
            />
            <div className="flex gap-1 bg-muted/50 p-1 rounded-lg border border-border/50">
              {SORT_OPTIONS.map((option) => (
                <button
                  key={option.key}
                  onClick={() => setSort(option.key)}
                  className={`px-2.5 py-1 rounded-md text-[11px] font-bold transition-all ${
                    sort === option.key ? "bg-card text-primary shadow-sm" : "text-muted-foreground hover:text-foreground"
                  }`}
                >
                  {option.label}
                </button>
              ))}
            </div>
          </div>
        </div>

        <div className="h-px bg-border/50" />

        {/* Row 2: Direction, Sector, and Confidence */}
        <div className="flex items-center gap-6">
          <div className="flex items-center gap-3">
            <span className="text-[10px] font-black uppercase tracking-widest text-muted-foreground/50">Direction</span>
            <div className="flex gap-1 bg-muted/50 p-1 rounded-lg border border-border/50">
              {DIRECTIONS.map((d) => (
                <button
                  key={d}
                  onClick={() => setDirection(d)}
                  className={`px-3 py-1 rounded-md text-xs font-bold transition-all ${
                    direction === d ? "bg-card text-primary shadow-sm" : "text-muted-foreground hover:text-foreground"
                  }`}
                >
                  {d}
                </button>
              ))}
            </div>
          </div>

          <div className="flex items-center gap-3">
            <span className="text-[10px] font-black uppercase tracking-widest text-muted-foreground/50">Sector</span>
            <div className="flex-shrink-0">
              <button
                ref={sectorBtnRef}
                onClick={toggleSectorDropdown}
                className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-muted/50 border border-transparent hover:border-white/10 hover:bg-accent transition-all text-xs min-w-[140px] group"
              >
                <span className={`truncate font-medium ${sectorFilter === "All Sectors" ? "text-muted-foreground/70" : "text-muted-foreground"}`}>
                  {sectorFilter}
                </span>
                <ChevronDown size={14} className={`ml-auto text-muted-foreground transition-transform duration-300 ${sectorOpen ? "rotate-180 text-primary" : "group-hover:text-foreground"}`} />
              </button>

              {sectorOpen && createPortal(
                <>
                  <div className="fixed inset-0 z-[9998]" onClick={() => setSectorOpen(false)} />
                  <div
                    className="fixed z-[9999] bg-[#1A2540] backdrop-blur-2xl border border-white/10 rounded-xl shadow-[0_20px_50px_rgba(0,0,0,0.5)] py-2 min-w-[220px] max-h-[400px] overflow-y-auto no-scrollbar animate-in fade-in zoom-in-95 slide-in-from-top-2 duration-200"
                    style={{ top: dropdownPos.top, left: dropdownPos.left }}
                  >
                    <button
                      onClick={() => { setSectorFilter("All Sectors"); setSectorOpen(false); }}
                      className={`w-full flex items-center justify-between px-4 py-2.5 text-sm transition-all duration-200 ${
                        sectorFilter === "All Sectors" 
                          ? "text-primary font-bold bg-primary/5" 
                          : "text-muted-foreground hover:bg-white/5 hover:text-foreground"
                      }`}
                    >
                      <span>All Sectors</span>
                      {sectorFilter === "All Sectors" && <div className="w-1.5 h-1.5 rounded-full bg-primary shadow-[0_0_8px_#00C805]" />}
                    </button>

                    <div className="h-px bg-white/5 my-1 mx-2" />

                    {PRIMARY_SECTORS.map((s) => (
                      <button
                        key={s}
                        onClick={() => { setSectorFilter(s); setSectorOpen(false); }}
                        className={`w-full flex items-center justify-between px-4 py-2.5 text-sm transition-all duration-200 ${
                          sectorFilter === s 
                            ? "text-primary font-bold bg-primary/5" 
                            : "text-muted-foreground hover:bg-white/5 hover:text-foreground"
                        }`}
                      >
                        <span>{s}</span>
                        {sectorFilter === s && <div className="w-1.5 h-1.5 rounded-full bg-primary shadow-[0_0_8px_#00C805]" />}
                      </button>
                    ))}
                  </div>
                </>,
                document.body
              )}
            </div>
          </div>

          <div className="flex items-center gap-3 ml-auto">
            <span className="text-[10px] font-black uppercase tracking-widest text-muted-foreground/50">Confidence</span>
            <div className="flex items-center gap-3">
              <input
                type="range"
                min={0}
                max={90}
                step={5}
                value={minConf}
                onChange={(e) => setMinConf(Number(e.target.value))}
                className="signal-confidence-slider"
                style={sliderStyle}
                aria-label="Minimum confidence"
              />
              <span className="w-10 text-right text-xs font-mono font-black text-primary">{minConf}%</span>
            </div>
          </div>
        </div>
      </div>

      {/* Table */}
      <div className="bg-card rounded-xl border border-border overflow-hidden fade-in-up stagger-2">
        {/* Header Row */}
        <div className="flex items-center gap-4 px-5 py-3 border-b border-border bg-muted/30">
          <div className="w-8" />
          <div className="w-24 text-xs font-semibold text-muted-foreground uppercase tracking-wider">Ticker</div>
          <div className="w-28 text-xs font-semibold text-muted-foreground uppercase tracking-wider">Signal</div>
          <div className="flex-1 text-xs font-semibold text-muted-foreground uppercase tracking-wider">Confidence</div>
          <div className="w-20 text-xs font-semibold text-muted-foreground uppercase tracking-wider text-right">Score</div>
          <div className="w-24 text-xs font-semibold text-muted-foreground uppercase tracking-wider text-center">Trend</div>
          <div className="w-32 text-xs font-semibold text-muted-foreground uppercase tracking-wider text-right">Sector</div>
          <div className="w-4" />
        </div>

        {isLoading ? (
          <div className="p-8 space-y-4">
            {Array.from({ length: 5 }).map((_, i) => (
              <div key={i} className="flex items-center gap-4 animate-pulse">
                <div className="w-8 h-4 bg-muted rounded" />
                <div className="h-10 bg-muted rounded w-24" />
                <div className="h-10 bg-muted rounded w-28" />
                <div className="h-4 bg-muted rounded flex-1" />
                <div className="h-10 bg-muted rounded w-20" />
                <div className="h-10 bg-muted rounded w-24" />
                <div className="h-10 bg-muted rounded w-32" />
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
            <div key={s.ticker} className="flex items-center group/row border-b border-border hover:bg-accent/30 transition-colors">
              <button
                onClick={(e) => { e.stopPropagation(); toggleWatchlist(s.ticker); }}
                className={`ml-5 transition-colors ${watchlist.includes(s.ticker) ? "text-primary scale-110" : "text-muted-foreground hover:text-foreground opacity-20 group-hover/row:opacity-100"}`}
              >
                <Star size={14} className={watchlist.includes(s.ticker) ? "fill-current" : ""} />
              </button>
              <div className="flex-1 min-w-0">
                <SignalRow
                  ticker={s.ticker}
                  name={s.name}
                  direction={s.direction}
                  confidence={s.confidence}
                  score={s.score}
                  sparkData={s.sparkData}
                  sector={s.sector}
                  onClick={() => onSelectSignal(s.ticker)}
                />
              </div>
            </div>
          ))
        ) : (
          <div className="flex flex-col items-center justify-center py-16 text-muted-foreground">
            {showWatchlist ? (
              <>
                <Star size={32} className="mb-3 opacity-20" />
                <p className="text-sm font-bold text-foreground">Watchlist is empty</p>
                <p className="text-xs mt-1">Star symbols to track them here</p>
              </>
            ) : (
              <>
                <Search size={32} className="mb-3 opacity-30" />
                <p className="text-sm">No signals match your filters</p>
              </>
            )}
          </div>
        )}
      </div>

      {/* Pagination Controls */}
      {totalPages > 1 && (
        <div className="flex items-center justify-between px-2 py-4 fade-in-up">
          <p className="text-xs text-muted-foreground font-medium">
            Showing <span className="font-black text-foreground">{(page - 1) * PAGE_SIZE + 1}</span> to <span className="font-black text-foreground">{Math.min(page * PAGE_SIZE, filtered.length)}</span> of <span className="font-black text-foreground">{filtered.length}</span> signals
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
                    className={`w-8 h-8 rounded-lg text-xs font-black transition-all ${
                      page === pageNum ? "bg-primary text-primary-foreground shadow-lg scale-105" : "hover:bg-accent text-muted-foreground"
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
