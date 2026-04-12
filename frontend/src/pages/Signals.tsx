import { useState, useMemo, useEffect, useRef, useCallback, type CSSProperties } from "react";
import { createPortal } from "react-dom";
import { Filter, Search, SortDesc, Download, RefreshCw, AlertCircle, ChevronLeft, ChevronRight, ChevronDown } from "lucide-react";
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
  const sectorBtnRef = useRef<HTMLButtonElement>(null);
  const [dropdownPos, setDropdownPos] = useState({ top: 0, left: 0 });
  const [minConf, setMinConf] = useState(0);
  const [sort, setSort] = useState<SortMode>("none");
  const [page, setPage] = useState(1);

  const toggleSectorDropdown = useCallback(() => {
    if (!sectorOpen && sectorBtnRef.current) {
      const rect = sectorBtnRef.current.getBoundingClientRect();
      setDropdownPos({ top: rect.bottom + 4, left: rect.left });
    }
    setSectorOpen((prev) => !prev);
  }, [sectorOpen]);

  const { data, isLoading, error, refetch, isFetching } = useQuery<LatestPredictionsResponse>({
    queryKey: ["latestPredictions"],
    queryFn: () => fetchApi<LatestPredictionsResponse>("/api/predictions/latest"),
    retry: false,
  });

  const predictions = data?.predictions || [];
  const sliderStyle = useMemo(
    () =>
      ({
        "--slider-progress": `${(minConf / 90) * 100}%`,
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
  }, [search, direction, minConf, sort, sectorFilter]);

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

          {/* Sector Dropdown */}
          <div>
            <button
              ref={sectorBtnRef}
              onClick={toggleSectorDropdown}
              className="flex items-center gap-2 px-3 py-2 rounded-lg bg-muted border border-transparent hover:bg-accent transition-all text-sm w-[160px]"
            >
              {sectorFilter === "All Sectors" ? (
                <span className="text-foreground">All Sectors</span>
              ) : (
                <span
                  className="text-xs font-semibold px-2 py-0.5 rounded-md border truncate max-w-[120px]"
                  style={{
                    backgroundColor: getSectorColor(sectorFilter).bg,
                    color: getSectorColor(sectorFilter).text,
                    borderColor: getSectorColor(sectorFilter).border,
                  }}
                >
                  {sectorFilter}
                </span>
              )}
              <ChevronDown size={14} className={`ml-auto text-muted-foreground transition-transform duration-200 ${sectorOpen ? "rotate-180" : ""}`} />
            </button>

            {sectorOpen && createPortal(
              <>
                <div className="fixed inset-0 z-[9998]" onClick={() => setSectorOpen(false)} />
                <div
                  className="fixed z-[9999] bg-card border border-border rounded-xl shadow-lg py-1.5 min-w-[200px] max-h-[320px] overflow-y-auto"
                  style={{ top: dropdownPos.top, left: dropdownPos.left }}
                >
                  <button
                    onClick={() => { setSectorFilter("All Sectors"); setSectorOpen(false); }}
                    className={`w-full text-left px-3 py-2 text-sm transition-colors ${sectorFilter === "All Sectors" ? "bg-accent text-foreground font-medium" : "text-muted-foreground hover:bg-accent/50 hover:text-foreground"}`}
                  >
                    All Sectors
                  </button>
                  {PRIMARY_SECTORS.map((s) => (
                    <button
                      key={s}
                      onClick={() => { setSectorFilter(s); setSectorOpen(false); }}
                      className={`w-full flex items-center gap-2 px-3 py-2 text-sm transition-colors ${sectorFilter === s ? "bg-accent/70" : "hover:bg-accent/30"}`}
                    >
                      <span
                        className="text-[10px] font-semibold px-2 py-0.5 rounded-md border"
                        style={{
                          backgroundColor: getSectorColor(s).bg,
                          color: getSectorColor(s).text,
                          borderColor: getSectorColor(s).border,
                        }}
                      >
                        {s}
                      </span>
                    </button>
                  ))}
                </div>
              </>,
              document.body
            )}
          </div>

          <div className="w-px h-5 bg-border" />

          {/* Min Confidence */}
          <div className="flex items-center gap-3">
            <span className="text-xs text-muted-foreground whitespace-nowrap">Min conf:</span>
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
            <span className="w-10 text-right text-xs font-mono font-semibold text-foreground">{minConf}%</span>
          </div>

          <div className="ml-auto flex items-center gap-1.5">
            <SortDesc
              size={13}
              className="text-muted-foreground cursor-pointer hover:text-foreground transition-colors"
              onClick={() => {
                const keys = SORT_OPTIONS.map((o) => o.key);
                const idx = keys.indexOf(sort);
                setSort(keys[(idx + 1) % keys.length] as SortMode);
              }}
            />
            <span className="text-xs text-muted-foreground">Sort:</span>
            {SORT_OPTIONS.map((option) => (
              <button
                key={option.key}
                onClick={() => setSort(option.key)}
                className={`text-xs px-2 py-1 rounded-md transition-colors ${
                  sort === option.key ? "text-primary font-semibold" : "text-muted-foreground hover:text-foreground"
                }`}
              >
                {option.label}
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
          <div className="w-24 text-xs font-semibold text-muted-foreground uppercase tracking-wider text-center">Trend</div>
          <div className="w-32 text-xs font-semibold text-muted-foreground uppercase tracking-wider text-right">Sector</div>
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
            <div key={s.ticker} className="fade-in-up" style={{ animationDelay: `${i * 40}ms` }}>
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
