import { useState, useMemo, useEffect, useRef, useCallback, type CSSProperties } from "react";
import { createPortal } from "react-dom";
import { Filter, Search, SortDesc, SortAsc, Download, RefreshCw, AlertCircle, ChevronLeft, ChevronRight, ChevronsLeft, ChevronsRight, ChevronDown, Star } from "lucide-react";
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

const DIRECTIONS = ["All", "Strong", "Watch", "Buffer"];
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

  type Tier = "strong" | "long" | "watch" | "buffer";
  const computeTier = (score: number, percentile: number): Tier => {
    if (score <= 0) return "buffer";
    if (percentile >= 75) return "strong";
    if (percentile < 25) return "watch";
    return "long";
  };

  const signalRows = useMemo(() => {
    return predictions.map((prediction) => {
      const confidence = Math.round(prediction.percentile);
      const tier = computeTier(prediction.score, prediction.percentile);
      return {
        ticker: prediction.ticker,
        name: prediction.company_name || prediction.ticker,
        direction: prediction.score > 0 ? ("long" as const) : ("neutral" as const),
        tier,
        confidence,
        score: prediction.score,
        rank: prediction.rank,
        sector: prediction.sector || "—",
        shuffleKey: hashTickerSeed(prediction.ticker + ":" + prediction.rank),
        sparkData: generateDirectionalSparkData(
          prediction.score,
          `${prediction.ticker}:${prediction.rank}:${prediction.score.toFixed(6)}`,
        ),
      };
    });
  }, [predictions]);

  const filtered = useMemo(() => {
    return signalRows
      .filter((s) => {
        const matchSearch = s.ticker.toLowerCase().includes(search.toLowerCase()) || s.name.toLowerCase().includes(search.toLowerCase());
        const matchDir =
          direction === "All" ||
          (direction === "Strong" && s.tier === "strong") ||
          (direction === "Watch" && s.tier === "watch") ||
          (direction === "Buffer" && s.tier === "buffer");
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

  const tierCounts = useMemo(() => {
    const counts = { strong: 0, long: 0, watch: 0, buffer: 0 };
    for (const p of predictions) {
      const t = computeTier(p.score, p.percentile);
      counts[t]++;
    }
    return counts;
  }, [predictions]);

  return (
    <div className="flex-1 overflow-y-auto p-6 space-y-5">
      {/* Header */}
      <div className="flex items-center justify-between fade-in-up">
        <div>
          <h2 className="text-xl font-bold text-foreground font-black uppercase tracking-widest">Signal Feed</h2>
          <p className="text-[10px] text-muted-foreground mt-1 uppercase tracking-[0.2em] font-medium">
            Week {data?.week_number || "—"} · {formatDateShort(data?.signal_date)} · <span className="text-bull-strong font-bold">{tierCounts.strong} strong</span> · <span className="text-bull font-bold">{tierCounts.long} long</span> · <span className="text-bull-watch font-bold">{tierCounts.watch} watch</span> · <span className="text-muted-foreground font-bold">{tierCounts.buffer} buffer</span> · Not investment advice
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
          <button className="flex items-center gap-1.5 px-3 py-2 rounded-lg text-sm bg-primary text-primary-foreground btn-primary font-medium">
            <Download size={14} />
            Export CSV
          </button>
        </div>
      </div>

      {/* Unified Filter Bar - Single Row */}
      <div className="bg-card rounded-2xl border border-border p-2.5 fade-in-up stagger-1 shadow-xl">
        <div className="flex items-center gap-2 flex-nowrap overflow-x-auto no-scrollbar">
          {/* Watchlist Toggle */}
          <button
            onClick={() => setShowWatchlist(!showWatchlist)}
            className={`flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-medium transition-all duration-300 border flex-shrink-0 ${
              showWatchlist ? "bg-primary/10 border-primary/30 text-primary shadow-inner" : "bg-muted/50 border-white/5 text-muted-foreground hover:bg-accent hover:text-foreground"
            }`}
          >
            <Star size={14} className={showWatchlist ? "fill-current" : ""} />
            Watchlist
            <span className={`ml-1 px-1.5 py-0.5 rounded-md text-[9px] ${showWatchlist ? "bg-primary text-primary-foreground font-bold" : "bg-muted-foreground/20 text-muted-foreground font-medium"}`}>
              {watchlist.length}
            </span>
          </button>

          <div className="w-px h-6 bg-white/5 mx-1 flex-shrink-0" />

          {/* Search */}
          <div className="flex items-center gap-2 px-4 py-2 rounded-xl bg-muted/50 border border-transparent focus-within:border-primary/30 transition-all flex-1 min-w-[180px] max-w-xs group">
            <Search size={14} className="text-muted-foreground group-focus-within:text-primary transition-colors" />
            <input
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search ticker or company..."
              className="bg-transparent text-xs text-foreground placeholder:text-muted-foreground/50 outline-none w-full font-medium"
            />
          </div>

          <div className="w-px h-6 bg-white/5 mx-1 flex-shrink-0" />

          {/* Sector Dropdown */}
          <div className="flex-shrink-0">
            <button
              ref={sectorBtnRef}
              onClick={toggleSectorDropdown}
              className={`flex items-center gap-2 px-4 py-2 rounded-xl transition-all text-xs font-medium border min-w-[140px] group ${
                sectorFilter === "All Sectors" ? "bg-muted/50 border-transparent text-muted-foreground hover:bg-accent" : "bg-primary/5 border-primary/20 text-foreground"
              }`}
            >
              <span className="truncate">{sectorFilter}</span>
              <ChevronDown size={14} className={`ml-auto text-muted-foreground transition-transform duration-300 ${sectorOpen ? "rotate-180 text-primary" : "group-hover:text-foreground"}`} />
            </button>

            {sectorOpen && createPortal(
              <>
                <div className="fixed inset-0 z-[9998]" onClick={() => setSectorOpen(false)} />
                <div
                  className="fixed z-[9999] bg-[#1A2540] backdrop-blur-2xl border border-white/10 rounded-2xl shadow-[0_25px_60px_rgba(0,0,0,0.6)] py-2 min-w-[220px] max-h-[450px] overflow-y-auto no-scrollbar animate-in fade-in zoom-in-95 slide-in-from-top-3 duration-200"
                  style={{ top: dropdownPos.top, left: dropdownPos.left }}
                >
                  <button
                    onClick={() => { setSectorFilter("All Sectors"); setSectorOpen(false); }}
                    className={`w-full flex items-center px-5 py-3 text-xs font-medium transition-all duration-200 ${
                      sectorFilter === "All Sectors" 
                        ? "text-primary bg-primary/5 font-bold" 
                        : "text-muted-foreground hover:bg-white/5 hover:text-foreground"
                    }`}
                  >
                    All Sectors
                  </button>

                  <div className="h-px bg-white/5 my-1 mx-3" />

                  {PRIMARY_SECTORS.map((s) => (
                    <button
                      key={s}
                      onClick={() => { setSectorFilter(s); setSectorOpen(false); }}
                      className={`w-full flex items-center px-5 py-3 text-xs font-medium transition-all duration-200 ${
                        sectorFilter === s 
                          ? "text-primary bg-primary/5 font-bold" 
                          : "text-muted-foreground hover:bg-white/5 hover:text-foreground"
                      }`}
                    >
                      <span>{s}</span>
                    </button>
                  ))}
                </div>
              </>,
              document.body
            )}
          </div>

          <div className="w-px h-6 bg-white/5 mx-1 flex-shrink-0" />

          {/* Direction Toggle */}
          <div className="flex gap-1 bg-muted p-1 rounded-xl flex-shrink-0">
            {DIRECTIONS.map((d) => (
              <button
                key={d}
                onClick={() => setDirection(d)}
                className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all duration-200 ${
                  direction === d ? "bg-card text-foreground shadow-custom" : "text-muted-foreground hover:text-foreground"
                }`}
              >
                {d}
              </button>
            ))}
          </div>

          <div className="w-px h-6 bg-white/5 mx-1 flex-shrink-0" />

          {/* Confidence Slider Group */}
          <div className="flex items-center gap-3 px-3 py-1.5 rounded-lg bg-muted/50 border border-transparent hover:border-white/5 transition-all flex-1 max-w-sm">
            <span className="text-xs font-medium text-muted-foreground/70 whitespace-nowrap">Min Conf</span>
            <div className="flex items-center gap-3 flex-1">
              <input
                type="range"
                min={0}
                max={90}
                step={5}
                value={minConf}
                onChange={(e) => setMinConf(Number(e.target.value))}
                className="signal-confidence-slider !w-full"
                style={sliderStyle}
                aria-label="Minimum confidence"
              />
              <span 
                className="w-10 text-right text-xs font-mono font-bold transition-colors duration-300"
                style={{ color: "var(--slider-fill-color)" }}
              >
                {minConf}%
              </span>
            </div>
          </div>

          {/* Sort Controls */}
          <div className="ml-auto flex items-center gap-2 flex-shrink-0 pr-1">
            <button
              onClick={() => setSort(sort === "conf_desc" ? "conf_asc" : "conf_desc")}
              className={`p-2 rounded-xl transition-all ${sort.startsWith("conf") ? "bg-primary/10 text-primary border border-primary/20" : "text-muted-foreground hover:text-foreground bg-muted/50 border border-transparent"}`}
              title="Toggle Confidence Sort"
            >
              {sort === "conf_asc" ? <SortAsc size={14} /> : <SortDesc size={14} />}
            </button>
            
            <div className="flex gap-1 bg-muted p-1 rounded-xl">
              {SORT_OPTIONS.filter(o => !o.key.startsWith("conf")).map((option) => (
                <button
                  key={option.key}
                  onClick={() => setSort(option.key)}
                  className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all duration-200 ${
                    sort === option.key ? "bg-card text-foreground shadow-custom" : "text-muted-foreground hover:text-foreground"
                  }`}
                >
                  {option.label}
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Table */}
      <div className="bg-card rounded-2xl border border-border overflow-hidden fade-in-up stagger-2 shadow-2xl">
        {/* Header Row */}
        <div className="flex items-center gap-4 px-6 py-4 border-b border-border bg-muted/20">
          <div className="w-8" />
          <div className="w-24 text-[10px] font-bold text-muted-foreground uppercase tracking-[0.2em]">Security</div>
          <div className="w-28 text-[10px] font-bold text-muted-foreground uppercase tracking-[0.2em]">Signal</div>
          <div className="flex-1 text-[10px] font-bold text-muted-foreground uppercase tracking-[0.2em]">Model Confidence</div>
          <div className="w-20 text-[10px] font-bold text-muted-foreground uppercase tracking-[0.2em] text-right">Raw Score</div>
          <div className="w-24 text-[10px] font-bold text-muted-foreground uppercase tracking-[0.2em] text-center">Trend</div>
          <div className="w-32 text-[10px] font-bold text-muted-foreground uppercase tracking-[0.2em] text-right">Sector</div>
          <div className="w-4" />
        </div>

        {isLoading ? (
          <div className="p-8 space-y-4">
            {Array.from({ length: 5 }).map((_, i) => (
              <div key={i} className="flex items-center gap-4 animate-pulse px-6 py-4">
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
          <div className="flex flex-col items-center justify-center py-24 text-muted-foreground font-medium text-center">
            <AlertCircle size={48} className="mb-4 text-bear opacity-50" />
            <p className="text-sm">Failed to load signals</p>
            <p className="text-xs mt-1 font-medium opacity-60">{(error as Error).message}</p>
            <button onClick={() => refetch()} className="mt-6 px-4 py-2 rounded-xl bg-primary text-primary-foreground font-medium text-[10px] uppercase tracking-widest shadow-lg">Retry Connection</button>
          </div>
        ) : paginated.length > 0 ? (
          paginated.map((s, i) => (
            <div key={s.ticker} className="flex items-center group/row border-b border-white/[0.03] last:border-0 hover:bg-accent/50 transition-colors">
              <button
                onClick={(e) => { e.stopPropagation(); toggleWatchlist(s.ticker); }}
                className={`ml-6 transition-all duration-300 ${watchlist.includes(s.ticker) ? "text-primary scale-125 drop-shadow-[0_0_8px_rgba(0,200,5,0.4)]" : "text-muted-foreground/30 hover:text-foreground opacity-100 group-hover/row:opacity-100"}`}
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
          <div className="flex flex-col items-center justify-center py-24 text-muted-foreground bg-muted/5 font-medium text-center">
            {showWatchlist ? (
              <>
                <Star size={48} className="mb-4 opacity-10" />
                <p className="text-sm font-medium uppercase tracking-widest text-foreground">Watchlist Empty</p>
                <p className="text-[10px] mt-2 opacity-50">Star securities to track them here</p>
              </>
            ) : (
              <>
                <Search size={48} className="mb-4 opacity-10" />
                <p className="text-sm font-medium uppercase tracking-widest text-foreground">No matches found</p>
                <p className="text-[10px] mt-2 opacity-50">Adjust filters to broaden search</p>
              </>
            )}
          </div>
        )}
      </div>

      {/* Pagination Controls */}
      {totalPages > 1 && (
        <div className="flex items-center justify-between px-4 py-6 fade-in-up">
          <p className="text-[10px] text-muted-foreground font-medium uppercase tracking-widest">
            Showing <span className="text-foreground">{(page - 1) * PAGE_SIZE + 1}</span> to <span className="text-foreground">{Math.min(page * PAGE_SIZE, filtered.length)}</span> of <span className="text-foreground">{filtered.length}</span>
          </p>
          <div className="flex items-center gap-2">
            <button
              onClick={() => setPage(1)}
              disabled={page === 1}
              className="p-2 rounded-xl bg-muted/50 border border-white/5 hover:bg-accent disabled:opacity-20 disabled:hover:bg-transparent transition-all shadow-inner"
              title="First Page"
            >
              <ChevronsLeft size={16} />
            </button>
            <button
              onClick={() => setPage(p => Math.max(1, p - 1))}
              disabled={page === 1}
              className="p-2 rounded-xl bg-muted/50 border border-white/5 hover:bg-accent disabled:opacity-20 disabled:hover:bg-transparent transition-all shadow-inner"
              title="Previous Page"
            >
              <ChevronLeft size={16} />
            </button>
            <div className="flex items-center gap-1.5">
              {Array.from({ length: Math.min(5, totalPages) }).map((_, i) => {
                let pageNum = page;
                if (totalPages <= 5) pageNum = i + 1;
                else if (page <= 3) pageNum = i + 1;
                else if (page >= totalPages - 2) pageNum = totalPages - 4 + i;
                else pageNum = page - 2 + i;
                return (
                  <button
                    key={pageNum}
                    onClick={() => setPage(pageNum)}
                    className={`w-9 h-9 rounded-xl text-[10px] font-medium uppercase transition-all border ${
                      page === pageNum ? "bg-primary text-primary-foreground border-primary shadow-xl" : "bg-muted/30 text-muted-foreground border-white/5 hover:text-foreground"
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
              className="p-2 rounded-xl bg-muted/50 border border-white/5 hover:bg-accent disabled:opacity-20 disabled:hover:bg-transparent transition-all shadow-inner"
              title="Next Page"
            >
              <ChevronRight size={16} />
            </button>
            <button
              onClick={() => setPage(totalPages)}
              disabled={page === totalPages}
              className="p-2 rounded-xl bg-muted/50 border border-white/5 hover:bg-accent disabled:opacity-20 disabled:hover:bg-transparent transition-all shadow-inner"
              title="Last Page"
            >
              <ChevronsRight size={16} />
            </button>
          </div>
        </div>
      )}

      <p className="text-[10px] font-medium uppercase tracking-[0.3em] text-muted-foreground/30 text-center pb-4">
        SEC Compliant Model Logic · Informational Dataset Alpha-22
      </p>
    </div>
  );
};

export default Signals;
