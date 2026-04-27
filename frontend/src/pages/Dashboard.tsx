import { useState, useMemo, useRef, useCallback, useEffect, useLayoutEffect, type CSSProperties } from "react";
import { createPortal } from "react-dom";
import { Info, Layers, BarChart2 } from "lucide-react";
import StatCard from "../components/StatCard";
import KLineChart from "../components/KLineChart";
import HeatmapChart from "../components/HeatmapChart";
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, BarChart, Bar, Cell } from "recharts";
import { useQuery } from "@tanstack/react-query";
import { fetchApi } from "../hooks/useApi";

const factorData = [
  { factor: "Momentum", ic: 0.142, positive: true },
  { factor: "Value", ic: -0.038, positive: false },
  { factor: "Quality", ic: 0.211, positive: true },
  { factor: "Low Vol", ic: 0.087, positive: true },
  { factor: "Growth", ic: 0.156, positive: true },
  { factor: "Carry", ic: -0.021, positive: false },
  { factor: "Reversal", ic: -0.063, positive: false },
];

interface MarketSector {
  sector: string;
  avg_change_pct: number;
  total_volume: number;
  ticker_count: number;
}

interface MarketOverviewResponse {
  as_of: string;
  latest_trade_date: string;
  spy: {
    ticker: string;
    trade_date: string;
    price: number;
    previous_close: number;
    change: number;
    change_pct: number;
    volume: number;
  };
  breadth: {
    advancing: number;
    declining: number;
    unchanged: number;
    total: number;
    advance_decline_ratio: number;
    advance_pct: number;
  };
  sectors: MarketSector[];
  vix: {
    series_id: string;
    observation_date: string;
    value: number;
    knowledge_time: string;
  };
}

interface MarketIndicesResponse {
  ticker: string;
  as_of: string;
  days: number;
  start_date: string;
  end_date: string;
  prices: Array<{
    trade_date: string;
    open: number;
    high: number;
    low: number;
    close: number;
    adj_close: number;
    volume: number;
  }>;
}

interface MarketSectorsResponse {
  as_of: string;
  days: number;
  start_date: string;
  end_date: string;
  sectors: MarketSector[];
}

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

interface FlyingTickerState {
  ticker: string;
  startX: number;
  startY: number;
  endX: number;
  endY: number;
}

const CustomTooltip = ({ active, payload }: { active?: boolean; payload?: any[] }) => {
  if (!active || !payload?.length) return null;
  const point = payload[0]?.payload;
  const rawDate = point?.date ? new Date(point.date) : null;
  const formattedDate =
    rawDate && !Number.isNaN(rawDate.getTime())
      ? rawDate.toLocaleDateString("en-US", {
          year: "numeric",
          month: "short",
          day: "numeric",
        })
      : point?.date;
  return (
    <div className="bg-popover border border-border rounded-lg px-3 py-2 shadow-custom">
      <p className="text-xs text-muted-foreground font-medium">{formattedDate}</p>
      <p className="text-sm font-bold text-bull font-mono">
        ${payload[0].value.toLocaleString("en-US", { maximumFractionDigits: 2 })}
      </p>
    </div>
  );
};

interface DashboardProps {
  onSelectSignal?: (ticker: string) => void;
}

const DASHBOARD_RANGES = [
  { key: "7D", label: "7D", days: 5 },
  { key: "30D", label: "30D", days: 21 },
  { key: "90D", label: "90D", days: 63 },
  { key: "1Y", label: "1Y", days: 252 },
  { key: "5Y", label: "5Y", days: 1260 },
  { key: "All", label: "All", days: 2520 },
] as const;

type DashboardRangeKey = (typeof DASHBOARD_RANGES)[number]["key"];

const Dashboard = ({ onSelectSignal = () => {} }: DashboardProps) => {
  const [selectedRangeKey, setSelectedRangeKey] = useState<DashboardRangeKey>("30D");
  const [chartTicker, setChartTicker] = useState("SPY");
  const [activeTicker, setActiveTicker] = useState<string | null>(null);
  const [flyingTicker, setFlyingTicker] = useState<FlyingTickerState | null>(null);
  const [isJelly, setIsJelly] = useState(false);

  const chartContainerRef = useRef<HTMLDivElement>(null);
  const rangeSelectorRef = useRef<HTMLDivElement>(null);
  const rangeButtonRefs = useRef<Array<HTMLButtonElement | null>>([]);
  const [rangeSliderStyle, setRangeSliderStyle] = useState<CSSProperties>({
    width: 0,
    transform: "translateX(0px)",
    opacity: 0,
  });

  const selectedRange =
    DASHBOARD_RANGES.find((range) => range.key === selectedRangeKey) ?? DASHBOARD_RANGES[1];

  const {
    data: overview,
    isLoading: isOverviewLoading,
    isError: isOverviewError,
  } = useQuery<MarketOverviewResponse>({
    queryKey: ["marketOverview"],
    queryFn: () => fetchApi<MarketOverviewResponse>("/api/market/overview"),
    retry: false,
  });

  const {
    data: indices,
    isLoading: isIndicesLoading,
    isError: isIndicesError,
  } = useQuery<MarketIndicesResponse>({
    queryKey: ["marketIndices", selectedRange.key, selectedRange.days],
    queryFn: () => fetchApi<MarketIndicesResponse>(`/api/market/indices?days=${selectedRange.days}`),
    retry: false,
    placeholderData: (previousData) => previousData,
  });

  const {
    data: sectorsData,
    isLoading: isSectorsLoading,
    isError: isSectorsError,
  } = useQuery<MarketSectorsResponse>({
    queryKey: ["marketSectors", 1],
    queryFn: () => fetchApi<MarketSectorsResponse>(`/api/market/sectors?days=1`),
    retry: false,
  });

  const {
    data: predictionsData,
    isLoading: isPredictionsLoading,
  } = useQuery<LatestPredictionsResponse>({
    queryKey: ["latestPredictions"],
    queryFn: () => fetchApi<LatestPredictionsResponse>("/api/predictions/latest"),
    retry: false,
  });

  const spyPrice = overview?.spy?.price || 0;
  const spyChangePct = overview?.spy?.change_pct || 0;
  const vixValue = overview?.vix?.value || 0;
  const breadth = overview?.breadth || { advancing: 0, declining: 0, advance_pct: 0 };
  const sectors = sectorsData?.sectors || overview?.sectors || [];

  const chartData =
    indices?.prices?.map((price) => ({
      date: price.trade_date,
      pnl: price.adj_close || price.close,
    })) || [];

  const topSignals = useMemo(() => {
    if (!predictionsData?.predictions) return [];
    return [...predictionsData.predictions]
      .sort((a, b) => Math.abs(b.score) - Math.abs(a.score))
      .slice(0, 5)
      .map(p => ({
        ticker: p.ticker,
        direction: p.score > 0 ? "long" : "neutral",
        confidence: Math.round(p.percentile),
        score: p.score
      }));
  }, [predictionsData]);

  // Default chart to top signal's ticker once loaded
  useEffect(() => {
    if (topSignals.length > 0 && chartTicker === "SPY" && !activeTicker) {
      setChartTicker(topSignals[0].ticker);
      setActiveTicker(topSignals[0].ticker);
    }
  }, [topSignals, chartTicker, activeTicker]);

  useLayoutEffect(() => {
    const updateRangeSlider = () => {
      const selectedIndex = DASHBOARD_RANGES.findIndex((range) => range.key === selectedRangeKey);
      const button = rangeButtonRefs.current[selectedIndex];

      if (!button) {
        setRangeSliderStyle((current) => ({ ...current, opacity: 0 }));
        return;
      }

      setRangeSliderStyle({
        width: button.offsetWidth,
        transform: `translateX(${Math.max(0, button.offsetLeft - 4)}px)`,
        opacity: 1,
      });
    };

    updateRangeSlider();

    const resizeObserver = new ResizeObserver(updateRangeSlider);
    const container = rangeSelectorRef.current;
    if (container) {
      resizeObserver.observe(container);
    }
    rangeButtonRefs.current.forEach((button) => {
      if (button) {
        resizeObserver.observe(button);
      }
    });
    window.addEventListener("resize", updateRangeSlider);

    return () => {
      resizeObserver.disconnect();
      window.removeEventListener("resize", updateRangeSlider);
    };
  }, [selectedRangeKey]);

  const handleTickerToChart = useCallback((ticker: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (ticker === chartTicker) return;

    const btnRect = (e.currentTarget as HTMLElement).getBoundingClientRect();
    const targetRect = chartContainerRef.current?.getBoundingClientRect();

    if (targetRect) {
      setFlyingTicker({
        ticker,
        startX: btnRect.left + btnRect.width / 2 - 24,
        startY: btnRect.top + btnRect.height / 2 - 16,
        endX: targetRect.left + 28,
        endY: targetRect.top + 28,
      });

      setTimeout(() => {
        setChartTicker(ticker);
        setActiveTicker(ticker);
        setFlyingTicker(null);
        setIsJelly(true);
        setTimeout(() => setIsJelly(false), 700);
      }, 800);
    }
  }, [chartTicker]);

  const sectorHeatmapCols = sectors.map((sector) => sector.sector);
  const sectorHeatmapValues = sectors.length > 0 ? [sectors.map((sector) => sector.avg_change_pct)] : undefined;
  const hasError = isOverviewError || isIndicesError || isSectorsError;
  const breadthTrend =
    breadth.advancing === breadth.declining ? "neutral" : breadth.advancing > breadth.declining ? "up" : "down";

  return (
    <div className="flex-1 overflow-y-auto p-6 space-y-6">
      {/* SEC Banner */}
      <div className="flex items-center gap-2 px-4 py-2.5 rounded-lg bg-accent border border-border">
        <Info size={14} className="text-muted-foreground flex-shrink-0" />
        <p className="text-xs text-muted-foreground font-medium">
          <span className="font-semibold text-foreground">Model Output Only</span>
          {` — All signals are generated by quantitative models and do not constitute investment advice. Past model performance does not guarantee future results. For institutional use only.`}
        </p>
      </div>

      {hasError && (
        <div className="flex items-center gap-2 px-4 py-2.5 rounded-lg bg-accent border border-border">
          <Info size={14} className="text-bear flex-shrink-0" />
          <p className="text-xs text-muted-foreground font-medium">
            Some market widgets could not be loaded. Available live data is still shown where possible.
          </p>
        </div>
      )}

      {/* Stat Cards */}
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-5">
        <StatCard
          label="SPY Price"
          value={isOverviewLoading ? "—" : `$${spyPrice.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`}
          change={spyChangePct}
          changeLabel="1D Change"
          trend={isOverviewLoading ? "neutral" : spyChangePct >= 0 ? "up" : "down"}
          delay={50}
        />
        <StatCard
          label="Market Breadth"
          value={isOverviewLoading ? "—" : `${breadth.advancing}/${breadth.declining}`}
          change={breadth.advance_pct || 0}
          changeLabel="Advancing share"
          trend={isOverviewLoading ? "neutral" : breadthTrend}
          delay={100}
        />
        <StatCard
          label="VIX Index"
          value={isOverviewLoading ? "—" : vixValue.toFixed(2)}
          change={0}
          changeLabel="Volatility"
          trend="neutral"
          delay={150}
        />
        <StatCard
          label="Top Sector"
          value={isSectorsLoading ? "—" : sectors[0]?.sector || "N/A"}
          change={sectors[0]?.avg_change_pct || 0}
          changeLabel="Avg Return"
          trend={
            isSectorsLoading
              ? "neutral"
              : (sectors[0]?.avg_change_pct || 0) === 0
                ? "neutral"
                : (sectors[0]?.avg_change_pct || 0) > 0
                  ? "up"
                  : "down"
          }
          delay={200}
        />
        <StatCard
          label="Sector Count"
          value={isSectorsLoading ? "—" : sectors.length.toString()}
          change={0}
          changeLabel="Active Sectors"
          trend="neutral"
          delay={250}
        />
      </div>

      {/* Main Charts Row */}
      <div className="flex flex-col gap-5 xl:flex-row">
        {/* Portfolio PnL (Using SPY indices history) */}
        <div className="flex-1 bg-card rounded-xl border border-border p-5 fade-in-up stagger-2">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h3 className="text-sm font-semibold text-foreground">Cumulative P&L (SPY Ref)</h3>
              <p className="text-xs text-muted-foreground mt-0.5 font-medium">
                Model output · {selectedRange.label} window
                {indices?.end_date ? ` · through ${indices.end_date}` : ""}
              </p>
            </div>
            <div ref={rangeSelectorRef} className="relative flex items-center gap-1 bg-accent/50 p-1 rounded-lg">
              <div
                aria-hidden="true"
                className="absolute bottom-1 left-1 top-1 rounded-md bg-card shadow-sm transition-all duration-300 ease-out"
                style={rangeSliderStyle}
              />
              {DASHBOARD_RANGES.map((range, index) => (
                <button
                  key={range.key}
                  ref={(node) => {
                    rangeButtonRefs.current[index] = node;
                  }}
                  onClick={() => setSelectedRangeKey(range.key)}
                  className={`relative z-10 px-3 py-1 text-[10px] font-bold rounded-md transition-colors duration-300 ${
                    selectedRangeKey === range.key
                      ? "text-foreground"
                      : "text-muted-foreground hover:text-foreground"
                  }`}
                >
                  {range.label}
                </button>
              ))}
            </div>
          </div>
          {isIndicesLoading ? (
            <div className="h-[180px] rounded-lg bg-muted animate-pulse" />
          ) : chartData.length > 0 ? (
            <ResponsiveContainer width="100%" height={180}>
              <AreaChart data={chartData}>
                <defs>
                  <linearGradient id="pnlGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#00C805" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="#00C805" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <XAxis
                  dataKey="date"
                  tick={{ fill: "#607B96", fontSize: 10 }}
                  axisLine={false}
                  tickLine={false}
                  interval="preserveStartEnd"
                  minTickGap={40}
                  tickFormatter={(value: string) => {
                    const parsed = new Date(value);
                    if (Number.isNaN(parsed.getTime())) return value;
                    return parsed.toLocaleDateString("en-US", chartData.length > 252
                      ? { month: "short", year: "numeric" }
                      : { month: "short", day: "numeric" });
                  }}
                />
                <YAxis hide domain={["auto", "auto"]} />
                <Tooltip content={<CustomTooltip />} />
                <Area type="monotone" dataKey="pnl" stroke="#00C805" strokeWidth={2} fill="url(#pnlGrad)" />
              </AreaChart>
            </ResponsiveContainer>
          ) : (
            <div className="flex h-[180px] items-center justify-center rounded-lg border border-dashed border-border bg-surface text-sm text-muted-foreground font-medium">
              No index history available.
            </div>
          )}
        </div>

        {/* Factor IC */}
        <div className="w-72 bg-card rounded-xl border border-border p-5 fade-in-up stagger-3 flex-shrink-0">
          <h3 className="text-sm font-semibold text-foreground mb-1">Factor IC Scores</h3>
          <p className="text-xs text-muted-foreground mb-4 font-medium">Information coefficient, 20-day</p>
          <ResponsiveContainer width="100%" height={180}>
            <BarChart data={factorData} layout="vertical" margin={{ left: 0, right: 8, top: 0, bottom: 0 }}>
              <XAxis type="number" domain={[-0.3, 0.3]} tick={{ fill: "#607B96", fontSize: 10 }} axisLine={false} tickLine={false} />
              <YAxis type="category" dataKey="factor" tick={{ fill: "#607B96", fontSize: 10 }} axisLine={false} tickLine={false} width={55} />
              <Tooltip
                cursor={{ fill: "rgba(255,255,255,0.03)" }}
                content={({ active, payload }: any) => {
                  if (!active || !payload?.length) return null;
                  const v = payload[0].value;
                  return (
                    <div className="bg-popover border border-border rounded-lg px-2.5 py-1.5 shadow-custom">
                      <p className={`text-xs font-bold ${v >= 0 ? "text-bull" : "text-bear"}`}>IC: {v.toFixed(3)}</p>
                    </div>
                  );
                }}
              />
              <Bar dataKey="ic" radius={[0, 3, 3, 0]}>
                {factorData.map((entry, i) => (
                  <Cell key={i} fill={entry.ic >= 0 ? "#00C805" : "#FF5252"} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* KLine + Recent Signals - Aligned Heights */}
      <div className="flex flex-col gap-5 xl:flex-row items-stretch">
        <div 
          ref={chartContainerRef}
          className={`flex-1 min-w-0 bg-card rounded-xl border flex flex-col transition-all duration-500 overflow-hidden relative ${isJelly ? "animate-jelly border-primary/50" : "border-border"}`}
        >
          <KLineChart key={chartTicker} ticker={chartTicker} height={320} defaultRange="1M" />
        </div>

        <div className="w-72 bg-card rounded-xl border border-border flex-shrink-0 flex flex-col fade-in-up stagger-4">
          <div className="flex items-center justify-between px-5 py-4 border-b border-border">
            <h3 className="text-sm font-semibold text-foreground">Top Signals</h3>
            <Layers size={14} className="text-muted-foreground" />
          </div>
          <div className="flex-1 overflow-y-auto no-scrollbar divide-y divide-border">
            {isPredictionsLoading ? (
              Array.from({ length: 5 }).map((_, i) => (
                <div key={i} className="px-5 py-4 animate-pulse space-y-2">
                  <div className="h-4 bg-muted rounded w-1/2" />
                  <div className="h-3 bg-muted rounded w-3/4" />
                </div>
              ))
            ) : topSignals.length > 0 ? (
              topSignals.map((s) => {
                const isActive = activeTicker === s.ticker;
                return (
                  <div 
                    key={s.ticker} 
                    className={`flex w-full items-center transition-all duration-300 ${isActive ? "opacity-40" : "hover:bg-accent/40"}`}
                  >
                    <button
                      onClick={() => onSelectSignal(s.ticker)}
                      className="flex-1 flex items-center justify-between px-5 py-3 cursor-pointer text-left"
                    >
                      <div>
                        <div className="text-sm font-bold text-foreground">{s.ticker}</div>
                        <span className={`text-[10px] font-semibold px-1.5 py-0.5 rounded-sm ${s.direction === "long" ? "tag-bull" : "tag-neutral"}`}>
                          {s.direction === "long" ? "LONG" : "BUFFER"}
                        </span>
                      </div>
                      <div className="text-right">
                        <div className="text-[10px] text-muted-foreground font-medium">{s.confidence}% conf</div>
                        <div className={`text-sm font-bold font-mono ${s.direction === "long" ? "text-bull" : "text-muted-foreground"}`}>
                          {s.score > 0 ? "+" : ""}{s.score.toFixed(4)}
                        </div>
                      </div>
                    </button>
                    <button 
                      onClick={(e) => handleTickerToChart(s.ticker, e)}
                      className={`mr-4 p-2 rounded-lg transition-all ${isActive ? "bg-primary/20 text-primary" : "bg-white/5 text-muted-foreground hover:bg-white/10 hover:text-foreground"}`}
                      title={`View ${s.ticker} chart`}
                      aria-label={`View ${s.ticker} chart`}
                    >
                      <BarChart2 size={14} />
                    </button>
                  </div>
                );
              })
            ) : (
              <div className="p-10 text-center text-xs text-muted-foreground font-medium">
                No active signals available
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Heatmap - Redesigned to be full width and at the very bottom */}
      <div className="fade-in-up stagger-5 pt-2">
        <HeatmapChart
          title="Sector Performance Heatmap"
          subtitle={sectorsData?.end_date ? `Latest close · ${sectorsData.end_date}` : "Latest close"}
          rows={["1D %"]}
          cols={sectorHeatmapCols}
          values={sectorHeatmapValues}
          valueFormatter={(value) => `${value >= 0 ? "+" : ""}${value.toFixed(1)}%`}
          tooltipFormatter={(value) => `${value >= 0 ? "+" : ""}${value.toFixed(2)}%`}
        />
      </div>

      {/* Flying Ticker Animation Portal */}
      {flyingTicker && createPortal(
        <div 
          className="fixed z-[9999] pointer-events-none rounded-xl bg-primary px-4 py-1.5 text-sm font-black text-primary-foreground shadow-[0_0_20px_rgba(0,200,5,0.5)] animate-fly-to-chart"
          style={{
            left: flyingTicker.startX,
            top: flyingTicker.startY,
            "--start-x": `${flyingTicker.startX}px`,
            "--start-y": `${flyingTicker.startY}px`,
            "--end-x": `${flyingTicker.endX}px`,
            "--end-y": `${flyingTicker.endY}px`,
          } as CSSProperties}
        >
          {flyingTicker.ticker}
        </div>,
        document.body
      )}
    </div>
  );
};

export default Dashboard;
