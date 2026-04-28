import { useState, useMemo, useRef, useCallback, useEffect, useLayoutEffect, type CSSProperties } from "react";
import { createPortal } from "react-dom";
import { Info, Layers, BarChart2 } from "lucide-react";
import { useTranslation } from "react-i18next";
import StatCard from "../components/StatCard";
import KLineChart from "../components/KLineChart";
import HeatmapChart from "../components/HeatmapChart";
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";
import { useQuery } from "@tanstack/react-query";
import { fetchApi } from "../hooks/useApi";
import type { GreyscalePerformanceResponse } from "../types/greyscale";

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

const CustomTooltip = ({ active, payload }: { active?: boolean; payload?: Array<{ value?: number; payload?: { date?: string | number; price?: number; volume?: number } }> }) => {
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
        ${(payload[0].value ?? 0).toLocaleString("en-US", { maximumFractionDigits: 2 })}
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
  const { t } = useTranslation();
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

  const {
    data: greyscalePerformance,
  } = useQuery<GreyscalePerformanceResponse>({
    queryKey: ["greyscalePerformance"],
    queryFn: () => fetchApi<GreyscalePerformanceResponse>("/api/greyscale/performance"),
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
    // Long-only champion: rank by score descending so STRONG/LONG names lead and
    // any buffer-held (negative-score) name falls to the bottom rather than
    // ranking high on raw |score| (Codex review Finding 4).
    return [...predictionsData.predictions]
      .sort((a, b) => b.score - a.score)
      .slice(0, 5)
      .map(p => {
        let tier: "strong" | "long" | "watch" | "buffer";
        if (p.score <= 0) tier = "buffer";
        else if (p.percentile >= 75) tier = "strong";
        else if (p.percentile < 25) tier = "watch";
        else tier = "long";
        return {
          ticker: p.ticker,
          direction: p.score > 0 ? "long" : "neutral",
          tier,
          confidence: Math.round(p.percentile),
          score: p.score
        };
      });
  }, [predictionsData]);

  // Default chart to top signal's ticker once loaded
  useEffect(() => {
    if (topSignals.length > 0 && chartTicker === "SPY" && !activeTicker) {
      // eslint-disable-next-line react-hooks/set-state-in-effect -- one-time bootstrap from async data
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
          <span className="font-semibold text-foreground">{t("dashboard.secBanner.modelOutputOnly")}</span>
          {t("dashboard.secBanner.disclaimer")}
        </p>
      </div>

      {hasError && (
        <div className="flex items-center gap-2 px-4 py-2.5 rounded-lg bg-accent border border-border">
          <Info size={14} className="text-bear flex-shrink-0" />
          <p className="text-xs text-muted-foreground font-medium">
            {t("dashboard.errorBanner")}
          </p>
        </div>
      )}

      {/* Stat Cards */}
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-5">
        <StatCard
          label={t("dashboard.stats.spyPrice")}
          value={isOverviewLoading ? "—" : `$${spyPrice.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`}
          change={spyChangePct}
          changeLabel={t("dashboard.stats.spyChange")}
          trend={isOverviewLoading ? "neutral" : spyChangePct >= 0 ? "up" : "down"}
          delay={50}
        />
        <StatCard
          label={t("dashboard.stats.marketBreadth")}
          value={isOverviewLoading ? "—" : `${breadth.advancing}/${breadth.declining}`}
          change={breadth.advance_pct || 0}
          changeLabel={t("dashboard.stats.advancingShare")}
          trend={isOverviewLoading ? "neutral" : breadthTrend}
          delay={100}
        />
        <StatCard
          label={t("dashboard.stats.vix")}
          value={isOverviewLoading ? "—" : vixValue.toFixed(2)}
          change={0}
          changeLabel={t("dashboard.stats.volatility")}
          trend="neutral"
          delay={150}
        />
        <StatCard
          label={t("dashboard.stats.topSector")}
          value={isSectorsLoading ? "—" : sectors[0]?.sector || "N/A"}
          change={sectors[0]?.avg_change_pct || 0}
          changeLabel={t("dashboard.stats.avgReturn")}
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
          label={t("dashboard.stats.sectorCount")}
          value={isSectorsLoading ? "—" : sectors.length.toString()}
          change={0}
          changeLabel={t("dashboard.stats.activeSectors")}
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
              <h3 className="text-sm font-semibold text-foreground">{t("dashboard.pnlChart.title")}</h3>
              <p className="text-xs text-muted-foreground mt-0.5 font-medium">
                {t("dashboard.pnlChart.subtitle", { range: selectedRange.label })}
                {indices?.end_date ? t("dashboard.pnlChart.throughDate", { date: indices.end_date }) : ""}
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
              {t("dashboard.pnlChart.noData")}
            </div>
          )}
        </div>

        {/* Greyscale Paper Performance (W13.2) */}
        <div className="w-72 bg-card rounded-xl border border-border p-5 fade-in-up stagger-3 flex-shrink-0">
          {(() => {
            // Pick the shortest horizon that has any non-pending data; default to 1d.
            const horizonPref: ("1d" | "5d" | "20d" | "60d")[] = ["1d", "5d", "20d", "60d"];
            const activeHorizon =
              horizonPref.find((h) => (greyscalePerformance?.cumulative?.[h]?.weeks_realized ?? 0) > 0)
              ?? horizonPref.find((h) => {
                const lw = greyscalePerformance?.per_week?.[greyscalePerformance.per_week.length - 1];
                return lw?.horizons?.[h]?.status && lw.horizons[h].status !== "pending";
              })
              ?? "1d";
            const horizonLabel = activeHorizon.replace("d", "-day");
            const cumBlock = greyscalePerformance?.cumulative?.[activeHorizon];
            const latestWeek = greyscalePerformance?.per_week?.[greyscalePerformance.per_week.length - 1];
            const latestBlock = latestWeek?.horizons?.[activeHorizon];
            const cumReturn = cumBlock?.return ?? null;
            const dd = cumBlock?.max_drawdown ?? null;
            const winrate = cumBlock?.winrate_vs_spy ?? null;
            const weeksRealized = cumBlock?.weeks_realized ?? 0;
            const weeklyCurve = cumBlock?.weekly_curve ?? [];
            const fmtPct = (v: number | null) => v == null ? "—" : `${(v * 100).toFixed(2)}%`;
            const colorOf = (v: number | null) => v == null ? "text-muted-foreground" : v >= 0 ? "text-bull" : "text-bear";

            const isPending = (latestBlock?.status === "pending") || cumReturn == null;

            return (
              <>
                <h3 className="text-sm font-semibold text-foreground mb-1">{t("dashboard.paperPerf.title")}</h3>
                <p className="text-xs text-muted-foreground mb-4 font-medium">
                  {t("dashboard.paperPerf.subtitle", { horizon: horizonLabel })}
                </p>
                {isPending ? (
                  <div className="flex flex-col items-center justify-center h-[180px] border border-dashed border-border rounded-lg bg-surface text-xs text-muted-foreground space-y-1">
                    <p className="font-bold uppercase tracking-widest">{t("dashboard.paperPerf.awaiting")}</p>
                    <p className="text-[10px] opacity-70 text-center px-2">
                      {t("dashboard.paperPerf.awaitingDetail")}
                    </p>
                  </div>
                ) : (
                  <>
                    <div className="grid grid-cols-2 gap-3 mb-3">
                      <div>
                        <p className="text-[10px] text-muted-foreground uppercase tracking-wider mb-0.5">{t("dashboard.paperPerf.thisWeek")}</p>
                        <p className={`text-base font-black font-mono ${colorOf(latestBlock?.portfolio_return ?? null)}`}>
                          {fmtPct(latestBlock?.portfolio_return ?? null)}
                        </p>
                      </div>
                      <div>
                        <p className="text-[10px] text-muted-foreground uppercase tracking-wider mb-0.5">{t("dashboard.paperPerf.vsSpy")}</p>
                        <p className={`text-base font-black font-mono ${colorOf(latestBlock?.excess ?? null)}`}>
                          {fmtPct(latestBlock?.excess ?? null)}
                        </p>
                      </div>
                      <div>
                        <p className="text-[10px] text-muted-foreground uppercase tracking-wider mb-0.5">{t("dashboard.paperPerf.cumulative")}</p>
                        <p className={`text-base font-black font-mono ${colorOf(cumReturn)}`}>
                          {fmtPct(cumReturn)}
                        </p>
                      </div>
                      <div>
                        <p className="text-[10px] text-muted-foreground uppercase tracking-wider mb-0.5">{t("dashboard.paperPerf.maxDd")}</p>
                        <p className={`text-base font-black font-mono ${colorOf(dd)}`}>
                          {fmtPct(dd)}
                        </p>
                      </div>
                    </div>
                    {weeklyCurve.length > 0 && (
                      <ResponsiveContainer width="100%" height={70}>
                        <AreaChart data={weeklyCurve} margin={{ left: 0, right: 0, top: 0, bottom: 0 }}>
                          <defs>
                            <linearGradient id="cumGrad" x1="0" y1="0" x2="0" y2="1">
                              <stop offset="5%" stopColor={cumReturn != null && cumReturn >= 0 ? "#00C805" : "#FF5252"} stopOpacity={0.3} />
                              <stop offset="95%" stopColor={cumReturn != null && cumReturn >= 0 ? "#00C805" : "#FF5252"} stopOpacity={0} />
                            </linearGradient>
                          </defs>
                          <XAxis dataKey="signal_date" hide />
                          <YAxis hide domain={["auto", "auto"]} />
                          <Area
                            type="monotone"
                            dataKey="cumulative_return"
                            stroke={cumReturn != null && cumReturn >= 0 ? "#00C805" : "#FF5252"}
                            strokeWidth={1.5}
                            fill="url(#cumGrad)"
                          />
                        </AreaChart>
                      </ResponsiveContainer>
                    )}
                    <div className="flex justify-between items-center mt-2 text-[10px] text-muted-foreground">
                      <span>{t("dashboard.paperPerf.weeksRealized", { count: weeksRealized })}</span>
                      {winrate != null && (
                        <span className={colorOf(winrate - 0.5)}>
                          {t("dashboard.paperPerf.winSuffix", { value: (winrate * 100).toFixed(0) })}
                        </span>
                      )}
                    </div>
                  </>
                )}
              </>
            );
          })()}
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
            <h3 className="text-sm font-semibold text-foreground">{t("dashboard.topSignals.title")}</h3>
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
                        <span className={`text-[10px] font-semibold px-1.5 py-0.5 rounded-sm ${s.tier === "strong" ? "tag-bull-strong" : s.tier === "watch" ? "tag-bull-watch" : s.tier === "buffer" ? "tag-neutral" : "tag-bull"}`}>
                          {t(`tiers.${s.tier ?? "long"}`)}
                        </span>
                      </div>
                      <div className="text-right">
                        <div className="text-[10px] text-muted-foreground font-medium">{t("dashboard.topSignals.confSuffix", { value: s.confidence })}</div>
                        <div className={`text-sm font-bold font-mono ${s.tier === "strong" ? "text-bull-strong" : s.tier === "watch" ? "text-bull-watch" : s.tier === "buffer" ? "text-muted-foreground" : "text-bull"}`}>
                          {s.score > 0 ? "+" : ""}{s.score.toFixed(4)}
                        </div>
                      </div>
                    </button>
                    <button
                      onClick={(e) => handleTickerToChart(s.ticker, e)}
                      className={`mr-4 p-2 rounded-lg transition-all ${isActive ? "bg-primary/20 text-primary" : "bg-white/5 text-muted-foreground hover:bg-white/10 hover:text-foreground"}`}
                      title={t("dashboard.topSignals.viewChart", { ticker: s.ticker })}
                      aria-label={t("dashboard.topSignals.viewChart", { ticker: s.ticker })}
                    >
                      <BarChart2 size={14} />
                    </button>
                  </div>
                );
              })
            ) : (
              <div className="p-10 text-center text-xs text-muted-foreground font-medium">
                {t("dashboard.topSignals.noSignals")}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Heatmap - Redesigned to be full width and at the very bottom */}
      <div className="fade-in-up stagger-5 pt-2">
        <HeatmapChart
          title={t("dashboard.heatmap.title")}
          subtitle={sectorsData?.end_date ? t("dashboard.heatmap.subtitleWithDate", { date: sectorsData.end_date }) : t("dashboard.heatmap.subtitle")}
          rows={[t("dashboard.heatmap.rowLabel")]}
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
