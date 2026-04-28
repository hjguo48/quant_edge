import { useState, useMemo, useEffect } from "react";
import { TrendingUp, PieChart, DollarSign, RefreshCw, Calculator, ShoppingCart, ShieldCheck, ArrowRight, AlertCircle, ChevronLeft, ChevronRight, ChevronsLeft, ChevronsRight } from "lucide-react";
import { useQuery } from "@tanstack/react-query";
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";
import { useTranslation } from "react-i18next";
import StatCard from "../components/StatCard";
import { fetchApi } from "../hooks/useApi";
import { getSectorColor } from "../constants/sectorColors";
import {
  GREYSCALE_HORIZONS,
  type GreyscaleHorizonKey,
  type GreyscalePerformanceResponse,
} from "../types/greyscale";

interface PortfolioHolding {
  ticker: string;
  weight: number;
  score: number;
  sector?: string | null;
  company_name?: string | null;
}

interface PortfolioCurrentResponse {
  signal_date: string;
  week_number: number;
  holding_count: number;
  gross_exposure: number;
  cash_weight: number;
  portfolio_beta: number | null;
  cvar_95: number | null;
  turnover: number;
  risk_pass: boolean;
  holdings: PortfolioHolding[];
}

interface PortfolioSummaryResponse {
  signal_date: string;
  week_number: number;
  holding_count: number;
  gross_exposure: number;
  cash_weight: number;
  turnover: number;
  portfolio_beta: number | null;
  cvar_95: number | null;
  risk_pass: boolean;
}

interface BudgetAllocation {
  ticker: string;
  weight: number;
  dollar_amount: number;
}

interface BudgetResponse {
  total_budget: number;
  allocations: BudgetAllocation[];
}

interface RebalanceOrder {
  ticker: string;
  action: "buy" | "sell" | "hold";
  weight_prev: number;
  weight_new: number;
  weight_delta: number;
}

interface RebalanceResponse {
  signal_date: string;
  orders: RebalanceOrder[];
}

const HOLDINGS_PAGE_SIZE = 5;
const SECTOR_PANEL_TOP_N = 6;

const Portfolio = () => {
  const { t } = useTranslation();
  const [tab, setTab] = useState("holdings");
  const [budgetStr, setBudgetStr] = useState("100000");
  const [, setBudgetStrPrev] = useState("100000");
  const totalBudget = parseInt(budgetStr) || 0;
  const [debouncedTotalBudget, setDebouncedTotalBudget] = useState(100000);
  const [holdingsPage, setHoldingsPage] = useState(1);

  // Reset Optimal Allocation page when switching tabs
  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect -- reset pagination on tab switch
    setHoldingsPage(1);
  }, [tab]);

  // Debounce budget calculation to avoid excessive API calls and layout flickering
  useEffect(() => {
    if (totalBudget < 1000) {
      // eslint-disable-next-line react-hooks/set-state-in-effect -- floor budget below threshold
      setDebouncedTotalBudget(0);
      return;
    }

    const timer = setTimeout(() => {
      setDebouncedTotalBudget(totalBudget);
    }, 800);
    return () => clearTimeout(timer);
  }, [totalBudget]);

  const currentQuery = useQuery<PortfolioCurrentResponse>({
    queryKey: ["portfolioCurrent"],
    queryFn: () => fetchApi<PortfolioCurrentResponse>("/api/portfolio/current"),
    retry: 1,
    refetchOnWindowFocus: false,
  });

  const summaryQuery = useQuery<PortfolioSummaryResponse>({
    queryKey: ["portfolioSummary"],
    queryFn: () => fetchApi<PortfolioSummaryResponse>("/api/portfolio/summary"),
    retry: 1,
    refetchOnWindowFocus: false,
  });

  const budgetQuery = useQuery<BudgetResponse>({
    queryKey: ["portfolioBudget", debouncedTotalBudget],
    queryFn: () => fetchApi<BudgetResponse>(`/api/portfolio/budget?total_budget=${debouncedTotalBudget}`),
    retry: 1,
    refetchOnWindowFocus: false,
    enabled: debouncedTotalBudget >= 1000,
    staleTime: 10000, // Cache for 10 seconds
  });

  const rebalanceQuery = useQuery<RebalanceResponse>({
    queryKey: ["portfolioRebalance"],
    queryFn: () => fetchApi<RebalanceResponse>("/api/portfolio/rebalance"),
    retry: 1,
    refetchOnWindowFocus: false,
  });

  const performanceQuery = useQuery<GreyscalePerformanceResponse>({
    queryKey: ["greyscalePerformance"],
    queryFn: () => fetchApi<GreyscalePerformanceResponse>("/api/greyscale/performance"),
    retry: 1,
    refetchOnWindowFocus: false,
    staleTime: 60_000,
  });

  const [activeHorizon, setActiveHorizon] = useState<GreyscaleHorizonKey>("1d");

  // Auto-pick the shortest horizon with realized data once loaded
  const performance = performanceQuery.data;
  const autoPickedHorizon = useMemo<GreyscaleHorizonKey | null>(() => {
    if (!performance) return null;
    return (
      GREYSCALE_HORIZONS.find((h) => (performance.cumulative?.[h]?.weeks_realized ?? 0) > 0) ?? null
    );
  }, [performance]);

  const [horizonAutoPicked, setHorizonAutoPicked] = useState(false);
  if (autoPickedHorizon && !horizonAutoPicked) {
    setHorizonAutoPicked(true);
    setActiveHorizon(autoPickedHorizon);
  }

  const isLoading = currentQuery.isLoading || summaryQuery.isLoading;
  const isError = currentQuery.isError || summaryQuery.isError;

  const current = currentQuery.data;
  const summary = summaryQuery.data;

  const sectorAggregation = useMemo(() => {
    const holdings = current?.holdings ?? [];
    if (!holdings.length) return [] as { sector: string; weight: number; tickerCount: number; tickers: string[] }[];
    const grouped = new Map<string, { weight: number; tickers: string[] }>();
    for (const h of holdings) {
      const key = h.sector?.trim() || "Unknown";
      const entry = grouped.get(key) ?? { weight: 0, tickers: [] };
      entry.weight += h.weight;
      entry.tickers.push(h.ticker);
      grouped.set(key, entry);
    }
    return Array.from(grouped.entries())
      .map(([sector, v]) => ({
        sector,
        weight: v.weight,
        tickerCount: v.tickers.length,
        tickers: v.tickers,
      }))
      .sort((a, b) => b.weight - a.weight);
  }, [current]);

  const stats = useMemo(() => {
    if (!summary) return [];
    return [
      { 
        label: "Holdings", 
        value: summary.holding_count.toString(), 
        change: summary.turnover * 100, 
        changeLabel: "Est. turnover", 
        trend: "neutral" as const 
      },
      { 
        label: "Gross Exposure", 
        value: `${(summary.gross_exposure * 100).toFixed(1)}%`, 
        change: (1 - summary.cash_weight) * 100, 
        changeLabel: "Net invested", 
        trend: "up" as const 
      },
      {
        label: "Portfolio Beta",
        value: summary.portfolio_beta != null ? summary.portfolio_beta.toFixed(2) : "—",
        change: 0,
        changeLabel: summary.portfolio_beta != null ? "vs. Benchmark" : "Not computed (W13 shadow mode)",
        trend: "neutral" as const
      },
      {
        label: "CVaR (95%)",
        value: summary.cvar_95 != null ? `${(summary.cvar_95 * 100).toFixed(2)}%` : "—",
        change: summary.risk_pass ? 0 : 1,
        changeLabel: summary.cvar_95 != null
          ? (summary.risk_pass ? "Risk check passed" : "Risk limit breach")
          : "Layer 3 shadow mode — see week_*.json",
        trend: summary.cvar_95 != null && !summary.risk_pass ? "down" as const : "neutral" as const,
      },
    ];
  }, [summary]);

  // Handle Animated Budget Input
  const handleBudgetChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    let val = e.target.value.replace(/\D/g, '');
    if (val === '') val = '0';
    if (val.length > 9) val = val.slice(0, 9);
    
    // Remove leading zeros only if length > 1
    if (val.length > 1 && val.startsWith('0')) {
      val = val.replace(/^0+/, '');
    }
    
    setBudgetStrPrev(budgetStr);
    setBudgetStr(val);
  };

  const renderAnimatedAmount = () => {
    const chars = budgetStr.split('');
    const isInvalid = totalBudget < 1000 && budgetStr !== '0';
    
    const formatted: (string | number)[] = [];
    let count = 0;
    for (let i = chars.length - 1; i >= 0; i--) {
      formatted.unshift(chars[i]);
      count++;
      if (count % 3 === 0 && i !== 0) {
        formatted.unshift(',');
      }
    }

    return (
      <div className="flex flex-col items-center justify-center min-h-[120px]">
        <div className="flex items-center justify-center font-mono text-5xl font-bold transition-colors duration-300">
          <span className={`mr-1 opacity-50 text-4xl ${isInvalid ? 'text-bear' : 'text-primary'}`}>$</span>
          <div className="flex items-baseline overflow-hidden py-2 h-[80px]">
            {formatted.map((char, idx) => {
              const isDigit = /\d/.test(char.toString());
              return (
                <span 
                  key={`${idx}-${char}`} 
                  className={`inline-block ${isDigit ? 'digit-animate-in' : ''} ${isInvalid ? 'text-bear' : 'text-foreground'}`}
                >
                  {char}
                </span>
              );
            })}
            <div className={`w-1 h-10 ml-1 animate-pulse ${isInvalid ? 'bg-bear' : 'bg-primary'}`} />
          </div>
        </div>
        {isInvalid && (
          <div className="text-bear text-[10px] font-black uppercase tracking-[0.2em] mt-2 animate-in fade-in slide-in-from-top-1">
            Minimum $1,000 Required
          </div>
        )}
      </div>
    );
  };

  if (isError) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center p-12 text-muted-foreground">
        <ShieldCheck size={48} className="mb-4 text-bear opacity-20" />
        <h2 className="text-xl font-bold text-foreground mb-2">Portfolio Data Unavailable</h2>
        <p className="max-w-md text-center text-sm mb-6">
          We encountered an error while fetching the current portfolio state. This may be due to a server connection issue or missing signal data for the current period.
        </p>
        <button 
          onClick={() => { currentQuery.refetch(); summaryQuery.refetch(); }}
          className="px-4 py-2 rounded-lg bg-primary text-primary-foreground font-medium text-sm hover:opacity-90 transition-opacity"
        >
          Retry Connection
        </button>
      </div>
    );
  }

  return (
    <div className="flex-1 overflow-y-auto p-6 space-y-5 bg-[#0D1421]">
      {/* Header */}
      <div className="flex items-center justify-between fade-in-up">
        <div>
          <h2 className="text-xl font-bold text-foreground">Active Portfolio</h2>
          <p className="text-sm text-muted-foreground mt-0.5">
            Model optimization output · Ref: {summary?.signal_date || "Current"} · Week {summary?.week_number}
          </p>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => { currentQuery.refetch(); summaryQuery.refetch(); }}
            className="flex items-center gap-1.5 px-3 py-2 rounded-lg text-sm text-muted-foreground hover:text-foreground hover:bg-accent border border-border transition-all"
          >
            <RefreshCw size={14} className={currentQuery.isFetching ? "animate-spin" : ""} />
            Sync Active
          </button>
        </div>
      </div>

      {/* Stats */}
      <div className="flex gap-4 fade-in-up stagger-1">
        {isLoading ? (
          Array.from({ length: 4 }).map((_, i) => (
            <div key={i} className="flex-1 h-32 bg-card rounded-xl border border-border animate-pulse" />
          ))
        ) : (
          stats.map((s, i) => (
            <div key={s.label} className="flex-1">
              <StatCard {...s} delay={i * 60} />
            </div>
          ))
        )}
      </div>

      {/* Main Area */}
      <div className="flex gap-5">
        <div className="flex-1 bg-card rounded-xl border border-border p-5 fade-in-up stagger-2 min-h-[300px]">
          {performanceQuery.isLoading ? (
            <div className="flex flex-col items-center justify-center h-full text-center p-8">
              <TrendingUp size={32} className="text-muted-foreground mb-4 opacity-20 animate-pulse" />
              <p className="text-xs text-muted-foreground">Loading paper P&amp;L...</p>
            </div>
          ) : performanceQuery.isError || !performance ? (
            <div className="flex flex-col items-center justify-center h-full text-center p-8">
              <TrendingUp size={32} className="text-muted-foreground mb-4 opacity-20" />
              <h3 className="text-sm font-semibold text-foreground mb-1">Performance Tracking</h3>
              <p className="text-xs text-muted-foreground max-w-xs">
                Awaiting first realized weekly close. Greyscale paper P&amp;L will populate after the first auto run.
              </p>
              <span className="mt-4 px-2.5 py-1 rounded-full bg-muted text-[10px] font-bold uppercase tracking-wider text-muted-foreground border border-border">
                Pending First Close
              </span>
            </div>
          ) : (() => {
            const cumBlock = performance.cumulative?.[activeHorizon];
            const latestWeek = performance.per_week?.[performance.per_week.length - 1];
            const latestBlock = latestWeek?.horizons?.[activeHorizon];
            const cumReturn = cumBlock?.return ?? null;
            const cumExcess = cumBlock?.excess ?? null;
            const dd = cumBlock?.max_drawdown ?? null;
            const winrate = cumBlock?.winrate_vs_spy ?? null;
            const weeksRealized = cumBlock?.weeks_realized ?? 0;
            const weeklyCurve = cumBlock?.weekly_curve ?? [];
            const recentWeeks = (performance.per_week ?? []).slice(-5).reverse();

            const fmtPct = (v: number | null, digits = 2) =>
              v == null ? "—" : `${(v * 100).toFixed(digits)}%`;
            const colorOf = (v: number | null) =>
              v == null ? "text-muted-foreground" : v >= 0 ? "text-bull" : "text-bear";

            const isPending = weeksRealized === 0 || cumReturn == null;

            return (
              <>
                <div className="flex items-start justify-between mb-4">
                  <div>
                    <h3 className="text-sm font-bold text-foreground">Performance Tracking</h3>
                    <p className="text-[10px] text-muted-foreground mt-0.5 font-medium">
                      Paper P&amp;L · vs {performance.benchmark} · dry-run only
                    </p>
                  </div>
                  <div className="flex gap-1 bg-muted/50 rounded-lg p-1 border border-border/50">
                    {GREYSCALE_HORIZONS.map((h) => (
                      <button
                        key={h}
                        onClick={() => setActiveHorizon(h)}
                        className={`px-2.5 py-1 rounded-md text-[10px] font-bold uppercase tracking-wider transition-all ${
                          activeHorizon === h
                            ? "bg-card text-foreground shadow-md border border-border"
                            : "text-muted-foreground hover:text-foreground"
                        }`}
                      >
                        {h}
                      </button>
                    ))}
                  </div>
                </div>

                {isPending ? (
                  <div className="flex flex-col items-center justify-center min-h-[180px] border border-dashed border-border rounded-lg bg-surface text-xs text-muted-foreground space-y-1">
                    <p className="font-bold uppercase tracking-widest">Awaiting first close</p>
                    <p className="text-[10px] opacity-70 text-center px-2">
                      {activeHorizon} horizon has no realized weeks yet. Realized P&amp;L materializes after price data lands for the next trading day after signal_date.
                    </p>
                  </div>
                ) : (
                  <>
                    <div className="grid grid-cols-4 gap-3 mb-4">
                      <div className="bg-muted/20 rounded-lg p-3 border border-border/50">
                        <p className="text-[10px] text-muted-foreground uppercase tracking-wider mb-1">Cumulative</p>
                        <p className={`text-lg font-black font-mono ${colorOf(cumReturn)}`}>
                          {fmtPct(cumReturn)}
                        </p>
                      </div>
                      <div className="bg-muted/20 rounded-lg p-3 border border-border/50">
                        <p className="text-[10px] text-muted-foreground uppercase tracking-wider mb-1">Excess vs SPY</p>
                        <p className={`text-lg font-black font-mono ${colorOf(cumExcess)}`}>
                          {fmtPct(cumExcess)}
                        </p>
                      </div>
                      <div className="bg-muted/20 rounded-lg p-3 border border-border/50">
                        <p className="text-[10px] text-muted-foreground uppercase tracking-wider mb-1">Max Drawdown</p>
                        <p className={`text-lg font-black font-mono ${colorOf(dd)}`}>
                          {fmtPct(dd)}
                        </p>
                      </div>
                      <div className="bg-muted/20 rounded-lg p-3 border border-border/50">
                        <p className="text-[10px] text-muted-foreground uppercase tracking-wider mb-1">Win Rate</p>
                        <p className={`text-lg font-black font-mono ${winrate == null ? "text-muted-foreground" : colorOf(winrate - 0.5)}`}>
                          {winrate == null ? "—" : `${(winrate * 100).toFixed(0)}%`}
                        </p>
                      </div>
                    </div>

                    {weeklyCurve.length > 0 && (
                      <div className="mb-3">
                        <ResponsiveContainer width="100%" height={120}>
                          <AreaChart data={weeklyCurve} margin={{ left: 0, right: 0, top: 4, bottom: 0 }}>
                            <defs>
                              <linearGradient id="portfolioCumGrad" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor={cumReturn != null && cumReturn >= 0 ? "#00C805" : "#FF5252"} stopOpacity={0.35} />
                                <stop offset="95%" stopColor={cumReturn != null && cumReturn >= 0 ? "#00C805" : "#FF5252"} stopOpacity={0} />
                              </linearGradient>
                            </defs>
                            <XAxis dataKey="signal_date" hide />
                            <YAxis hide domain={["auto", "auto"]} />
                            <Tooltip
                              contentStyle={{
                                background: "var(--popover)",
                                border: "1px solid var(--border)",
                                borderRadius: 8,
                                fontSize: 11,
                              }}
                              formatter={(value: number) => [`${(value * 100).toFixed(2)}%`, "Cumulative"]}
                              labelFormatter={(label: string) => `Week of ${label}`}
                            />
                            <Area
                              type="monotone"
                              dataKey="cumulative_return"
                              stroke={cumReturn != null && cumReturn >= 0 ? "#00C805" : "#FF5252"}
                              strokeWidth={2}
                              fill="url(#portfolioCumGrad)"
                            />
                          </AreaChart>
                        </ResponsiveContainer>
                      </div>
                    )}

                    {recentWeeks.length > 0 && (
                      <div>
                        <div className="flex items-center px-3 py-2 border-b border-border bg-muted/10 text-[10px] font-black text-muted-foreground uppercase tracking-[0.15em]">
                          <div className="w-24">Signal Date</div>
                          <div className="flex-1 text-right">Portfolio</div>
                          <div className="flex-1 text-right">SPY</div>
                          <div className="flex-1 text-right">Excess</div>
                          <div className="w-20 text-right">Status</div>
                        </div>
                        {recentWeeks.map((w) => {
                          const block = w.horizons?.[activeHorizon];
                          const status = block?.status ?? "pending";
                          return (
                            <div key={w.week_number} className="flex items-center px-3 py-2 border-b border-border/50 last:border-0 text-xs">
                              <div className="w-24 text-foreground font-medium">{w.signal_date ?? "—"}</div>
                              <div className={`flex-1 text-right font-mono ${colorOf(block?.portfolio_return ?? null)}`}>
                                {fmtPct(block?.portfolio_return ?? null)}
                              </div>
                              <div className={`flex-1 text-right font-mono ${colorOf(block?.spy_return ?? null)}`}>
                                {fmtPct(block?.spy_return ?? null)}
                              </div>
                              <div className={`flex-1 text-right font-mono ${colorOf(block?.excess ?? null)}`}>
                                {fmtPct(block?.excess ?? null)}
                              </div>
                              <div className="w-20 text-right">
                                <span className={`text-[9px] font-bold uppercase tracking-wider px-1.5 py-0.5 rounded ${
                                  status === "realized" ? "bg-bull/10 text-bull" :
                                  status === "partial" ? "bg-amber-500/10 text-amber-500" :
                                  "bg-muted text-muted-foreground"
                                }`}>
                                  {status}
                                </span>
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    )}

                    <div className="flex justify-between items-center mt-3 text-[10px] text-muted-foreground">
                      <span>{weeksRealized} weeks realized · last close {latestBlock?.horizon_end_date ?? "—"}</span>
                      {latestBlock && (
                        <span>
                          coverage {latestBlock.tickers_used}/{latestBlock.tickers_used + latestBlock.tickers_missing}
                        </span>
                      )}
                    </div>
                  </>
                )}
              </>
            );
          })()}
        </div>

        <div className="w-72 bg-card rounded-xl border border-border p-5 flex-shrink-0 fade-in-up stagger-3">
          {currentQuery.isLoading ? (
            <div className="flex flex-col items-center justify-center h-full text-center py-12">
              <PieChart size={28} className="text-muted-foreground mb-3 opacity-20 animate-pulse" />
              <p className="text-xs text-muted-foreground">{t("common.loading")}</p>
            </div>
          ) : sectorAggregation.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full text-center py-12">
              <PieChart size={28} className="text-muted-foreground mb-3 opacity-20" />
              <h3 className="text-sm font-semibold text-foreground mb-1">{t("portfolio.sectorWeights.title")}</h3>
              <p className="text-xs text-muted-foreground px-2">
                {t("portfolio.sectorWeights.noHoldings")}
              </p>
            </div>
          ) : (() => {
            const totalWeight = sectorAggregation.reduce((sum, s) => sum + s.weight, 0);
            const visible = sectorAggregation.slice(0, SECTOR_PANEL_TOP_N);
            const overflow = sectorAggregation.slice(SECTOR_PANEL_TOP_N);
            const otherWeight = overflow.reduce((sum, s) => sum + s.weight, 0);
            const otherTickerCount = overflow.reduce((sum, s) => sum + s.tickerCount, 0);
            const maxWeight = Math.max(...visible.map((s) => s.weight), otherWeight);
            return (
              <>
                <div className="mb-4">
                  <h3 className="text-sm font-bold text-foreground">{t("portfolio.sectorWeights.title")}</h3>
                  <p className="text-[10px] text-muted-foreground mt-0.5 font-medium">
                    {t("portfolio.sectorWeights.subtitle")}
                  </p>
                </div>
                <div className="space-y-2.5">
                  {visible.map((s) => {
                    const color = getSectorColor(s.sector);
                    const pct = (s.weight / totalWeight) * 100;
                    const barWidth = maxWeight > 0 ? (s.weight / maxWeight) * 100 : 0;
                    return (
                      <div key={s.sector}>
                        <div className="flex items-center justify-between text-[10px] mb-1">
                          <div className="flex items-center gap-1.5 min-w-0">
                            <div
                              className="w-1.5 h-1.5 rounded-full flex-shrink-0"
                              style={{ background: color.text }}
                            />
                            <span className="text-foreground font-semibold truncate">{s.sector}</span>
                            <span className="text-muted-foreground">·</span>
                            <span className="text-muted-foreground">{t("portfolio.sectorWeights.tickers", { count: s.tickerCount })}</span>
                          </div>
                          <span className="font-mono font-bold text-foreground flex-shrink-0">
                            {pct.toFixed(1)}%
                          </span>
                        </div>
                        <div className="h-1.5 bg-muted/40 rounded-full overflow-hidden">
                          <div
                            className="h-full rounded-full transition-all"
                            style={{
                              width: `${barWidth}%`,
                              background: color.text,
                              opacity: 0.7,
                            }}
                          />
                        </div>
                      </div>
                    );
                  })}
                  {overflow.length > 0 && (
                    <div>
                      <div className="flex items-center justify-between text-[10px] mb-1">
                        <div className="flex items-center gap-1.5 min-w-0">
                          <div className="w-1.5 h-1.5 rounded-full flex-shrink-0 bg-muted-foreground/50" />
                          <span className="text-muted-foreground font-semibold truncate">+{overflow.length} {t("common.of").replace(/\W/g, "")}</span>
                          <span className="text-muted-foreground">·</span>
                          <span className="text-muted-foreground">{t("portfolio.sectorWeights.tickers", { count: otherTickerCount })}</span>
                        </div>
                        <span className="font-mono font-bold text-muted-foreground flex-shrink-0">
                          {((otherWeight / totalWeight) * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div className="h-1.5 bg-muted/40 rounded-full overflow-hidden">
                        <div
                          className="h-full rounded-full bg-muted-foreground/40"
                          style={{ width: `${maxWeight > 0 ? (otherWeight / maxWeight) * 100 : 0}%` }}
                        />
                      </div>
                    </div>
                  )}
                </div>
              </>
            );
          })()}
        </div>
      </div>

      {/* Holdings Table */}
      <div className="bg-card rounded-2xl border border-border overflow-hidden fade-in-up stagger-4 shadow-2xl">
        <div className="flex items-center justify-between px-6 py-5 border-b border-border bg-muted/10">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-primary/10">
              <DollarSign size={18} className="text-primary" />
            </div>
            <div>
              <h3 className="text-base font-bold text-foreground">
                {tab === "holdings" ? "Optimal Allocation" : tab === "trades" ? "Rebalance Instructions" : "Capital Configurator"}
              </h3>
              <p className="text-xs text-muted-foreground mt-0.5">
                {tab === "holdings" ? "Calculated by mean-variance optimizer" : tab === "trades" ? "Execution orders for target weights" : "Adjust your trading capital"}
              </p>
            </div>
          </div>
          <div className="flex gap-1 bg-muted/50 rounded-xl p-1 border border-border/50">
            {[
              { id: "holdings", label: "Holdings", icon: ShieldCheck },
              { id: "trades", label: "Trades", icon: ShoppingCart },
              { id: "budget", label: "Budget", icon: Calculator },
            ].map((t) => (
              <button 
                key={t.id} 
                onClick={() => setTab(t.id)} 
                className={`flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-bold transition-all duration-300 ${tab === t.id ? "bg-card text-foreground shadow-xl border border-border scale-[1.02]" : "text-muted-foreground hover:text-foreground"}`}
              >
                <t.icon size={14} />
                {t.label}
              </button>
            ))}
          </div>
        </div>

        {tab === "holdings" && (() => {
          const allHoldings = current?.holdings ?? [];
          const totalHoldingsPages = Math.max(1, Math.ceil(allHoldings.length / HOLDINGS_PAGE_SIZE));
          const safePage = Math.min(holdingsPage, totalHoldingsPages);
          const paginatedHoldings = allHoldings.slice(
            (safePage - 1) * HOLDINGS_PAGE_SIZE,
            safePage * HOLDINGS_PAGE_SIZE,
          );
          return (
            <div className="animate-in fade-in duration-500">
              <div className="flex items-center px-6 py-3 border-b border-border bg-muted/20 text-[10px] font-black text-muted-foreground uppercase tracking-[0.2em]">
                <div className="w-32">Security</div>
                <div className="flex-1">Target Weight</div>
                <div className="w-32 text-right">Alpha Score</div>
                <div className="w-32 text-center">Direction</div>
                <div className="w-10" />
              </div>
              {isLoading ? (
                Array.from({ length: 6 }).map((_, i) => (
                  <div key={i} className="px-6 py-5 border-b border-border flex items-center gap-4 animate-pulse">
                    <div className="w-32 h-5 bg-muted rounded" />
                    <div className="flex-1 h-3 bg-muted rounded" />
                    <div className="w-32 h-5 bg-muted rounded" />
                    <div className="w-32 h-5 bg-muted rounded" />
                  </div>
                ))
              ) : paginatedHoldings.map((h, i) => {
                const isLong = h.score > 0;
                return (
                  <div
                    key={h.ticker}
                    className="flex items-center px-6 py-4 border-b border-border last:border-0 hover:bg-primary/[0.02] transition-colors group"
                    style={{ animationDelay: `${i * 30}ms` }}
                  >
                    <div className="w-32">
                      <div className="text-sm font-black text-foreground group-hover:text-primary transition-colors">{h.ticker}</div>
                      <div className="text-[10px] text-muted-foreground uppercase font-mono tracking-tighter">Equity Component</div>
                    </div>
                    <div className="flex-1">
                      <div className="flex items-center gap-4">
                        <div className="w-48 h-2 bg-muted rounded-full overflow-hidden border border-white/5">
                          <div
                            className={`h-full rounded-full transition-all duration-1000 ${isLong ? 'bar-glow-bull' : 'bar-glow-bear'}`}
                            style={{
                              width: `${Math.min(100, h.weight * 100 * 5)}%`,
                              backgroundColor: isLong ? "#00C805" : "#FF5252",
                            }}
                          />
                        </div>
                        <span className={`text-xs font-black font-mono ${isLong ? "text-bull" : "text-bear"}`}>
                          {(h.weight * 100).toFixed(2)}%
                        </span>
                      </div>
                    </div>
                    <div className="w-32 text-right">
                      <span className="text-xs font-mono font-bold text-foreground/80 bg-muted/50 px-2 py-1 rounded">{h.score.toFixed(4)}</span>
                    </div>
                    <div className="w-32 flex justify-center">
                      <span className={`text-[10px] font-black px-2.5 py-1 rounded-full border shadow-sm ${isLong ? "bg-bull/10 border-bull/30 text-bull" : "bg-bear/10 border-bear/30 text-bear"}`}>
                        {isLong ? "BULLISH" : "BEARISH"}
                      </span>
                    </div>
                    <div className="w-10 flex justify-end">
                      <ArrowRight size={14} className="text-muted-foreground opacity-0 group-hover:opacity-100 transition-all -translate-x-2 group-hover:translate-x-0" />
                    </div>
                  </div>
                );
              })}

              {/* Pagination Controls — mirrors Signal Feed */}
              {totalHoldingsPages > 1 && !isLoading && (
                <div className="flex items-center justify-between px-4 py-6">
                  <p className="text-[10px] text-muted-foreground font-medium uppercase tracking-widest">
                    Showing <span className="text-foreground">{(safePage - 1) * HOLDINGS_PAGE_SIZE + 1}</span> to <span className="text-foreground">{Math.min(safePage * HOLDINGS_PAGE_SIZE, allHoldings.length)}</span> of <span className="text-foreground">{allHoldings.length}</span>
                  </p>
                  <div className="flex items-center gap-2">
                    <button
                      onClick={() => setHoldingsPage(1)}
                      disabled={safePage === 1}
                      className="p-2 rounded-xl bg-muted/50 border border-white/5 hover:bg-accent disabled:opacity-20 disabled:hover:bg-transparent transition-all shadow-inner"
                      title="First Page"
                    >
                      <ChevronsLeft size={16} />
                    </button>
                    <button
                      onClick={() => setHoldingsPage(p => Math.max(1, p - 1))}
                      disabled={safePage === 1}
                      className="p-2 rounded-xl bg-muted/50 border border-white/5 hover:bg-accent disabled:opacity-20 disabled:hover:bg-transparent transition-all shadow-inner"
                      title="Previous Page"
                    >
                      <ChevronLeft size={16} />
                    </button>
                    <div className="flex items-center gap-1.5">
                      {Array.from({ length: Math.min(5, totalHoldingsPages) }).map((_, i) => {
                        let pageNum = safePage;
                        if (totalHoldingsPages <= 5) pageNum = i + 1;
                        else if (safePage <= 3) pageNum = i + 1;
                        else if (safePage >= totalHoldingsPages - 2) pageNum = totalHoldingsPages - 4 + i;
                        else pageNum = safePage - 2 + i;
                        return (
                          <button
                            key={pageNum}
                            onClick={() => setHoldingsPage(pageNum)}
                            className={`w-9 h-9 rounded-xl text-[10px] font-medium uppercase transition-all border ${
                              safePage === pageNum ? "bg-primary text-primary-foreground border-primary shadow-xl" : "bg-muted/30 text-muted-foreground border-white/5 hover:text-foreground"
                            }`}
                          >
                            {pageNum}
                          </button>
                        );
                      })}
                    </div>
                    <button
                      onClick={() => setHoldingsPage(p => Math.min(totalHoldingsPages, p + 1))}
                      disabled={safePage === totalHoldingsPages}
                      className="p-2 rounded-xl bg-muted/50 border border-white/5 hover:bg-accent disabled:opacity-20 disabled:hover:bg-transparent transition-all shadow-inner"
                      title="Next Page"
                    >
                      <ChevronRight size={16} />
                    </button>
                    <button
                      onClick={() => setHoldingsPage(totalHoldingsPages)}
                      disabled={safePage === totalHoldingsPages}
                      className="p-2 rounded-xl bg-muted/50 border border-white/5 hover:bg-accent disabled:opacity-20 disabled:hover:bg-transparent transition-all shadow-inner"
                      title="Last Page"
                    >
                      <ChevronsRight size={16} />
                    </button>
                  </div>
                </div>
              )}
            </div>
          );
        })()}

        {tab === "trades" && (
          <div className="animate-in slide-in-from-bottom-2 duration-500">
            <div className="flex items-center px-6 py-3 border-b border-border bg-muted/20 text-[10px] font-black text-muted-foreground uppercase tracking-[0.2em]">
              <div className="w-32">Ticker</div>
              <div className="w-32 text-center">Action</div>
              <div className="w-32 text-right">Prev %</div>
              <div className="w-32 text-right">Target %</div>
              <div className="flex-1 text-right">Allocation Shift</div>
            </div>
            {rebalanceQuery.isLoading ? (
              <div className="p-12 text-center text-xs text-muted-foreground animate-pulse">Calculating rebalance orders...</div>
            ) : rebalanceQuery.data?.orders.map((order) => (
              <div key={order.ticker} className="flex items-center px-6 py-4 border-b border-border last:border-0 hover:bg-accent/40 transition-colors">
                <div className="w-32 text-sm font-black text-foreground">{order.ticker}</div>
                <div className="w-32 flex justify-center">
                  <span className={`text-[10px] font-black px-3 py-1 rounded-md uppercase tracking-wider ${
                    order.action === "buy" ? "bg-bull text-white shadow-lg shadow-bull/20" : order.action === "sell" ? "bg-bear text-white shadow-lg shadow-bear/20" : "bg-muted text-muted-foreground"
                  }`}>
                    {order.action}
                  </span>
                </div>
                <div className="w-32 text-right text-xs font-mono text-muted-foreground">{(order.weight_prev * 100).toFixed(2)}%</div>
                <div className="w-32 text-right text-xs font-mono text-foreground font-bold">{(order.weight_new * 100).toFixed(2)}%</div>
                <div className="flex-1 text-right">
                  <span className={`inline-block px-2 py-1 rounded text-xs font-mono font-black ${order.weight_delta > 0 ? "text-bull bg-bull/5" : order.weight_delta < 0 ? "text-bear bg-bear/5" : "text-muted-foreground bg-muted/5"}`}>
                    {order.weight_delta > 0 ? "+" : ""}{(order.weight_delta * 100).toFixed(2)}%
                  </span>
                </div>
              </div>
            ))}
            {!rebalanceQuery.isLoading && (!rebalanceQuery.data || rebalanceQuery.data.orders.length === 0) && (
              <div className="py-20 text-center flex flex-col items-center">
                <ShieldCheck size={40} className="text-primary opacity-20 mb-4" />
                <p className="text-sm font-bold text-foreground">Portfolio In Sync</p>
                <p className="text-xs text-muted-foreground mt-1">No rebalance required for the current period.</p>
              </div>
            )}
          </div>
        )}

        {tab === "budget" && (
          <div className="p-10 animate-in zoom-in-95 duration-500">
            <div className="max-w-2xl mx-auto space-y-10">
              <div className="text-center space-y-2">
                <h4 className={`text-xs font-black uppercase tracking-[0.3em] transition-colors ${totalBudget < 1000 && budgetStr !== '0' ? 'text-bear' : 'text-primary'}`}>Capital Allocation</h4>
                <p className="text-muted-foreground text-sm">Enter your total trading budget to calculate dollar amounts</p>
              </div>

              <div className="relative group">
                <div className={`absolute inset-0 blur-3xl rounded-full transition-all duration-500 ${totalBudget < 1000 && budgetStr !== '0' ? 'bg-bear/10' : 'bg-primary/5 group-focus-within:bg-primary/10'}`} />
                <div className={`relative p-12 rounded-3xl border bg-white/[0.02] shadow-inner flex flex-col items-center justify-center transition-colors duration-500 ${totalBudget < 1000 && budgetStr !== '0' ? 'border-bear/30' : 'border-white/5'}`}>
                  <input 
                    type="text" 
                    inputMode="numeric"
                    value={budgetStr} 
                    onChange={handleBudgetChange}
                    className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10"
                    autoFocus
                  />
                  {renderAnimatedAmount()}
                  <p className="mt-6 text-[10px] font-bold text-muted-foreground uppercase tracking-widest animate-pulse">Click to adjust amount</p>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-6">
                <div className="p-6 rounded-2xl bg-muted/30 border border-border flex items-center justify-between">
                  <div>
                    <p className="text-[10px] font-black text-muted-foreground uppercase tracking-widest">Active Holdings</p>
                    <p className="text-2xl font-black text-foreground mt-1">{budgetQuery.data?.allocations.length || 0}</p>
                  </div>
                  <div className={`p-3 rounded-xl ${totalBudget < 1000 && budgetStr !== '0' ? 'bg-bear/10' : 'bg-primary/10'}`}>
                    <ShieldCheck size={20} className={totalBudget < 1000 && budgetStr !== '0' ? 'text-bear' : 'text-primary'} />
                  </div>
                </div>
                <div className="p-6 rounded-2xl bg-muted/30 border border-border flex items-center justify-between">
                  <div>
                    <p className="text-[10px] font-black text-muted-foreground uppercase tracking-widest">Avg. Position</p>
                    <p className="text-2xl font-black text-foreground mt-1">
                      ${totalBudget > 0 ? ((totalBudget / (budgetQuery.data?.allocations.length || 1))).toLocaleString(undefined, { maximumFractionDigits: 0 }) : '0'}
                    </p>
                  </div>
                  <div className="p-3 rounded-xl bg-bull/10">
                    <TrendingUp size={20} className="text-bull" />
                  </div>
                </div>
              </div>

              <div className="space-y-3 pt-4 min-h-[450px]">
                <div className="flex items-center px-4 py-2 text-[10px] font-black text-muted-foreground uppercase tracking-widest">
                  <div className="w-24">Ticker</div>
                  <div className="w-32 text-right">Optimal Weight</div>
                  <div className="flex-1 text-right">Cash Allocation</div>
                </div>
                
                {totalBudget < 1000 ? (
                  <div className="py-16 text-center flex flex-col items-center justify-center border border-dashed border-border rounded-2xl bg-muted/10">
                    <Calculator size={32} className="text-muted-foreground opacity-20 mb-3" />
                    <p className="text-xs text-muted-foreground italic">
                      Please enter a budget of at least $1,000 to see detailed allocations.
                    </p>
                  </div>
                ) : (debouncedTotalBudget !== totalBudget || budgetQuery.isFetching) ? (
                  <div className="py-16 text-center flex flex-col items-center justify-center animate-in fade-in">
                    <RefreshCw size={32} className="text-primary animate-spin opacity-20 mb-3" />
                    <p className="text-xs text-muted-foreground animate-pulse italic">Calculating allocation matrix...</p>
                  </div>
                ) : budgetQuery.isError ? (
                  <div className="py-16 text-center flex flex-col items-center justify-center border border-bear/20 rounded-2xl bg-bear/5 animate-in fade-in">
                    <AlertCircle size={32} className="text-bear opacity-50 mb-3" />
                    <p className="text-xs text-bear font-medium">Calculation Failed</p>
                    <p className="text-[10px] text-muted-foreground mt-1 mb-4">{(budgetQuery.error as Error).message}</p>
                    <button 
                      onClick={() => budgetQuery.refetch()}
                      className="px-3 py-1.5 rounded-lg bg-muted hover:bg-accent text-[10px] font-bold text-foreground transition-colors flex items-center gap-1.5"
                    >
                      <RefreshCw size={12} />
                      Retry Calculation
                    </button>
                  </div>
                ) : budgetQuery.data?.allocations && budgetQuery.data.allocations.length > 0 ? (
                  <div className="space-y-3 animate-in fade-in duration-500">
                    {budgetQuery.data.allocations.map((alloc) => (
                      <div key={alloc.ticker} className="flex items-center px-5 py-4 rounded-xl border border-white/[0.03] bg-muted/20 hover:border-primary/30 hover:bg-primary/[0.02] transition-all group">
                        <div className="w-24 text-sm font-black text-foreground group-hover:text-primary transition-colors">{alloc.ticker}</div>
                        <div className="w-32 text-right text-xs font-mono text-muted-foreground font-bold">{(alloc.weight * 100).toFixed(2)}%</div>
                        <div className="flex-1 text-right text-base font-mono font-black text-bull group-hover:scale-105 transition-transform origin-right">
                          ${alloc.dollar_amount.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="py-16 text-center flex flex-col items-center justify-center border border-dashed border-border rounded-2xl bg-muted/10">
                    <ShieldCheck size={32} className="text-muted-foreground opacity-20 mb-3" />
                    <p className="text-xs text-muted-foreground">No allocation data returned for this budget.</p>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
      </div>
      
      <div className="flex items-center justify-center gap-2 py-8">
        <ShieldCheck size={12} className="text-muted-foreground opacity-50" />
        <p className="text-[10px] text-muted-foreground uppercase tracking-[0.2em] font-medium">
          SEC Compliant Model Output · Not Investment Advice · Dynamic Risk Weighting
        </p>
      </div>
    </div>
  );
};

export default Portfolio;
