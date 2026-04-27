import { useState, useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { ArrowLeft, Brain, BarChart3, Info, TrendingUp, Layers, ShieldCheck, Target, AlertCircle, TrendingDown } from "lucide-react";
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Cell,
} from "recharts";
import KLineChart from "../components/KLineChart";
import StatCard from "../components/StatCard";
import ShapWaterfall from "../components/ShapWaterfall";
import SignalHistory from "../components/SignalHistory";
import { fetchApi } from "../hooks/useApi";
import { getSectorColor } from "../constants/sectorColors";

interface SignalDetailProps {
  ticker?: string;
  onBack?: () => void;
}

interface PredictionDetail {
  ticker: string;
  fusion_score: number;
  rank: number;
  total: number;
  percentile: number;
  model_scores: Record<string, number>;
  weight: number;
  signal_date: string;
  confidence: "high" | "medium" | "low";
  model_spread: number;
  model_agreement: number;
}

interface ExpectedReturnResponse {
  ticker: string;
  signal_date: string;
  percentile: number;
  quintile: number;
  data_source: string;
  ci_level: number;
  n_observations: number;
  annualized_excess: { estimate: number; ci_lower: number; ci_upper: number };
  sharpe: { estimate: number; ci_lower: number; ci_upper: number };
}

interface ShapFeature {
  feature: string;
  shap_value: number;
}

interface ShapResponse {
  ticker: string;
  signal_date: string;
  features: ShapFeature[];
}

interface HistoryPoint {
  week: number;
  signal_date: string;
  score: number;
  rank: number;
  total: number;
}

interface HistoryResponse {
  ticker: string;
  history: HistoryPoint[];
}

interface StockQuote {
  trade_date: string;
  open?: number | null;
  high?: number | null;
  low?: number | null;
  close?: number | null;
  adj_close?: number | null;
  volume?: number | null;
  previous_close?: number | null;
  change?: number | null;
  change_pct?: number | null;
}

interface StockDetailResponse {
  ticker: string;
  company_name: string;
  sector?: string | null;
  industry?: string | null;
  ipo_date?: string | null;
  market_cap?: number | null;
  latest_price?: StockQuote | null;
}

interface StockFundamentalsResponse {
  ticker: string;
  fiscal_period?: string | null;
  event_time?: string | null;
  knowledge_time?: string | null;
  metric_count: number;
  metrics: Record<string, number | null>;
}

interface StockTechnicalsResponse {
  ticker: string;
  trade_date?: string | null;
  close?: number | null;
  rsi_14?: number | null;
  macd?: number | null;
  macd_signal?: number | null;
  macd_histogram?: number | null;
  sma_20?: number | null;
  sma_50?: number | null;
  sma_200?: number | null;
  bb_upper?: number | null;
  bb_middle?: number | null;
  bb_lower?: number | null;
  bb_width?: number | null;
  bb_position?: number | null;
}

type TabKey = "overview" | "factors" | "backtest" | "risk";

const tabs: TabKey[] = ["overview", "factors", "backtest", "risk"];

const metricTokenMap: Record<string, string> = {
  adx: "ADX",
  bb: "BB",
  cagr: "CAGR",
  cci: "CCI",
  cvar: "CVaR",
  ebitda: "EBITDA",
  eps: "EPS",
  ev: "EV",
  fcf: "FCF",
  gdp: "GDP",
  ic: "IC",
  ipo: "IPO",
  macd: "MACD",
  pb: "P/B",
  pe: "P/E",
  ps: "P/S",
  roa: "ROA",
  roe: "ROE",
  rsi: "RSI",
  sec: "SEC",
  sma: "SMA",
  ttm: "TTM",
  var: "VaR",
  vix: "VIX",
  yoy: "YoY",
};

function formatCurrency(value?: number | null, maximumFractionDigits = 2): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return "—";
  return value.toLocaleString("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits,
  });
}

function formatCompactCurrency(value?: number | null): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return "—";
  const absolute = Math.abs(value);
  if (absolute >= 1_000_000_000_000) return `$${(value / 1_000_000_000_000).toFixed(2)}T`;
  if (absolute >= 1_000_000_000) return `$${(value / 1_000_000_000).toFixed(2)}B`;
  if (absolute >= 1_000_000) return `$${(value / 1_000_000).toFixed(2)}M`;
  if (absolute >= 1_000) return `$${(value / 1_000).toFixed(2)}K`;
  return formatCurrency(value, 0);
}

function formatNumber(value?: number | null, digits = 2): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return "—";
  return value.toLocaleString("en-US", {
    maximumFractionDigits: digits,
    minimumFractionDigits: digits,
  });
}

function formatCompactNumber(value?: number | null): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return "—";
  return new Intl.NumberFormat("en-US", {
    notation: "compact",
    maximumFractionDigits: 2,
  }).format(value);
}

function formatPercent(value?: number | null, digits = 2): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return "—";
  const prefix = value > 0 ? "+" : "";
  return `${prefix}${value.toFixed(digits)}%`;
}

function formatDate(value?: string | null): string {
  if (!value) return "—";
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return value;
  return parsed.toLocaleDateString("en-US", {
    year: "numeric",
    month: "short",
    day: "numeric",
  });
}

function formatMetricLabel(metric: string): string {
  return metric
    .split("_")
    .map((token) => metricTokenMap[token.toLowerCase()] ?? `${token.charAt(0).toUpperCase()}${token.slice(1)}`)
    .join(" ");
}

function shortenMetricLabel(metric: string): string {
  const label = formatMetricLabel(metric);
  return label.length > 20 ? `${label.slice(0, 19)}…` : label;
}

function getTrend(value?: number | null): "up" | "down" | "neutral" {
  if (typeof value !== "number" || !Number.isFinite(value) || value === 0) return "neutral";
  return value > 0 ? "up" : "down";
}

function getRsiColor(value?: number | null): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return "text-muted-foreground";
  if (value > 70) return "text-bear";
  if (value < 30) return "text-bull";
  return "text-amber-400";
}

function getErrorMessage(error: unknown): string {
  if (error instanceof Error) return error.message;
  return "Failed to load data.";
}

function LoadingCard({ className = "", lines = 4 }: { className?: string; lines?: number }) {
  return (
    <div className={`bg-card rounded-xl border border-border p-5 animate-pulse ${className}`}>
      <div className="space-y-3">
        {Array.from({ length: lines }, (_, index) => (
          <div
            key={index}
            className="h-4 rounded-md bg-muted"
            style={{ width: `${88 - index * 10}%` }}
          />
        ))}
      </div>
    </div>
  );
}

const ExpectedReturnCard = ({ data }: { data: ExpectedReturnResponse }) => {
  const renderRange = (label: string, estimate: number, lower: number, upper: number, isPercent: boolean = true) => {
    const min = Math.min(lower, estimate, upper);
    const max = Math.max(lower, estimate, upper);
    const range = max - min;
    const padding = range * 0.2;
    const displayMin = min - padding;
    const displayMax = max + padding;
    const displayRange = displayMax - displayMin;

    const getPos = (val: number) => ((val - displayMin) / displayRange) * 100;

    const format = (v: number) => isPercent ? `${(v * 100).toFixed(1)}%` : v.toFixed(3);

    return (
      <div className="space-y-2">
        <div className="flex justify-between items-center text-xs">
          <span className="text-muted-foreground font-medium">{label}</span>
          <span className="text-foreground font-mono font-black">{format(estimate)}</span>
        </div>
        <div className="relative h-6 flex items-center">
          <div className="absolute w-full h-1 bg-muted/50 rounded-full" />
          {/* CI Bar */}
          <div 
            className="absolute h-1 bg-primary/40 rounded-full"
            style={{ left: `${getPos(lower)}%`, width: `${getPos(upper) - getPos(lower)}%` }}
          />
          {/* Estimate Dot */}
          <div 
            className="absolute w-2.5 h-2.5 bg-primary rounded-full border-2 border-card shadow-[0_0_10px_#00C805]"
            style={{ left: `${getPos(estimate)}%`, transform: 'translateX(-50%)' }}
          />
          {/* Labels */}
          <div className="absolute top-4 left-0 w-full flex justify-between px-0.5">
            <span className="text-[9px] text-muted-foreground font-mono">{format(lower)}</span>
            <span className="text-[9px] text-muted-foreground font-mono">{format(upper)}</span>
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="bg-card rounded-2xl border border-border p-5 space-y-6 shadow-xl relative overflow-hidden group hover:border-primary/20 transition-all duration-300">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className="p-2 rounded-xl bg-primary/10 group-hover:bg-primary/20 transition-colors">
            <Target size={18} className="text-primary" />
          </div>
          <h3 className="text-sm font-black text-foreground uppercase tracking-widest">Expected Performance</h3>
        </div>
        <div className="bg-muted px-2.5 py-1 rounded-lg text-[10px] font-black uppercase tracking-[0.15em] text-primary border border-primary/10 shadow-inner flex items-center gap-1 whitespace-nowrap">
          Quintile {data.quintile}
        </div>
      </div>

      <div className="space-y-10 pt-2 pb-4 px-1">
        {renderRange("Annualized Excess Return", data.annualized_excess.estimate, data.annualized_excess.ci_lower, data.annualized_excess.ci_upper)}
        {renderRange("Sharpe Ratio", data.sharpe.estimate, data.sharpe.ci_lower, data.sharpe.ci_upper, false)}
      </div>

      <div className="pt-5 space-y-4 border-t border-white/5">
        <div className="flex items-start gap-3">
          <ShieldCheck size={14} className="text-primary mt-0.5 opacity-40" />
          <p className="text-[9px] text-muted-foreground leading-relaxed font-medium uppercase tracking-wider opacity-70">
            Based on historical model backtest performance. Past results do not guarantee future returns. Not investment advice.
          </p>
        </div>
        <div className="flex items-center justify-between px-1">
          <span className="text-[8px] text-muted-foreground/40 font-black uppercase tracking-widest">Statistical Source</span>
          <span className="text-[9px] text-muted-foreground/60 font-mono">10K block bootstrap · {data.n_observations} obs</span>
        </div>
      </div>
    </div>
  );
};

const SignalDetail = ({
  ticker = "AAPL",
  onBack = () => {},
}: SignalDetailProps) => {
  const [activeTab, setActiveTab] = useState<TabKey>("overview");
  const normalizedTicker = (ticker || "AAPL").toUpperCase();

  const detailQuery = useQuery<StockDetailResponse>({
    queryKey: ["stockDetail", normalizedTicker],
    queryFn: () => fetchApi<StockDetailResponse>(`/api/stocks/${normalizedTicker}`),
    enabled: Boolean(normalizedTicker),
    retry: false,
  });

  const predictionQuery = useQuery<PredictionDetail>({
    queryKey: ["prediction", normalizedTicker],
    queryFn: () => fetchApi<PredictionDetail>(`/api/predictions/${normalizedTicker}`),
    enabled: Boolean(normalizedTicker),
    retry: false,
  });

  const expectedReturnQuery = useQuery<ExpectedReturnResponse>({
    queryKey: ["expectedReturn", normalizedTicker],
    queryFn: () => fetchApi<ExpectedReturnResponse>(`/api/predictions/${normalizedTicker}/expected-return`),
    enabled: Boolean(normalizedTicker),
    retry: false,
  });

  const shapQuery = useQuery<ShapResponse>({
    queryKey: ["shap", normalizedTicker],
    queryFn: () => fetchApi<ShapResponse>(`/api/predictions/${normalizedTicker}/shap`),
    enabled: Boolean(normalizedTicker),
    retry: false,
  });

  const historyQuery = useQuery<HistoryResponse>({
    queryKey: ["history", normalizedTicker],
    queryFn: () => fetchApi<HistoryResponse>(`/api/predictions/${normalizedTicker}/history`),
    enabled: Boolean(normalizedTicker),
    retry: false,
  });

  const fundamentalsQuery = useQuery<StockFundamentalsResponse>({
    queryKey: ["stockFundamentals", normalizedTicker],
    queryFn: () => fetchApi<StockFundamentalsResponse>(`/api/stocks/${normalizedTicker}/fundamentals`),
    enabled: Boolean(normalizedTicker),
    retry: false,
  });

  const technicalsQuery = useQuery<StockTechnicalsResponse>({
    queryKey: ["stockTechnicals", normalizedTicker],
    queryFn: () => fetchApi<StockTechnicalsResponse>(`/api/stocks/${normalizedTicker}/technicals`),
    enabled: Boolean(normalizedTicker),
    retry: false,
  });

  const detail = detailQuery.data;
  const latestPrice = detail?.latest_price;
  const fundamentals = fundamentalsQuery.data;
  const technicals = technicalsQuery.data;
  const prediction = predictionQuery.data;
  const shap = shapQuery.data;
  const expectedReturn = expectedReturnQuery.data;
  const history = historyQuery.data?.history || [];

  const isPrediction404 = predictionQuery.error instanceof Error && predictionQuery.error.message.includes("404");
  const isShap404 = shapQuery.error instanceof Error && shapQuery.error.message.includes("404");

  const metricEntries = Object.entries(fundamentals?.metrics ?? {}).sort(([left], [right]) =>
    left.localeCompare(right),
  );

  const topMetricData = metricEntries
    .filter(([, value]) => typeof value === "number" && Number.isFinite(value))
    .map(([metric, value]) => ({
      metric,
      label: shortenMetricLabel(metric),
      fullLabel: formatMetricLabel(metric),
      value: Number(value),
      positive: Number(value) >= 0,
    }))
    .sort((left, right) => Math.abs(right.value) - Math.abs(left.value))
    .slice(0, 10)
    .reverse();

  const hasSectionError = fundamentalsQuery.isError || technicalsQuery.isError;

  if (detailQuery.isError) {
    return (
      <div className="flex-1 overflow-y-auto p-6 space-y-5 bg-[#0D1421]">
        <div className="flex items-start gap-4 fade-in-up">
          <button
            onClick={onBack}
            className="flex items-center gap-2 text-xs font-black uppercase tracking-widest text-muted-foreground hover:text-primary transition-all group mt-0.5"
          >
            <ArrowLeft size={14} className="transition-transform group-hover:-translate-x-1" />
            Return to Signals
          </button>
        </div>

        <div className="bg-card rounded-2xl border border-border p-8 fade-in-up stagger-1 shadow-2xl">
          <div className="flex items-center gap-3 mb-4">
            <AlertCircle size={24} className="text-bear" />
            <h2 className="text-xl font-black text-foreground uppercase tracking-widest">Load Error</h2>
          </div>
          <p className="text-sm text-muted-foreground font-medium">{getErrorMessage(detailQuery.error)}</p>
        </div>
      </div>
    );
  }

  const stats = [
    {
      label: "Market Cap",
      value: formatCompactCurrency(detail?.market_cap),
      change: latestPrice?.change_pct ?? 0,
      changeLabel: "Session move",
      trend: getTrend(latestPrice?.change_pct),
    },
    {
      label: "RSI (14)",
      value: formatNumber(technicals?.rsi_14, 1),
      change: typeof technicals?.rsi_14 === "number" ? technicals.rsi_14 - 50 : 0,
      changeLabel: "vs. neutral 50",
      trend: getTrend(typeof technicals?.rsi_14 === "number" ? technicals.rsi_14 - 50 : 0),
    },
    {
      label: "SMA 200",
      value: formatCurrency(technicals?.sma_200),
      change:
        typeof latestPrice?.close === "number" &&
        typeof technicals?.sma_200 === "number" &&
        technicals.sma_200 !== 0
          ? ((latestPrice.close / technicals.sma_200) - 1) * 100
          : 0,
      changeLabel: "vs. last close",
      trend:
        typeof latestPrice?.close === "number" &&
        typeof technicals?.sma_200 === "number" &&
        technicals.sma_200 !== 0
          ? getTrend(((latestPrice.close / technicals.sma_200) - 1) * 100)
          : "neutral",
    },
    {
      label: "BB Position",
      value: formatNumber(technicals?.bb_position, 2),
      change:
        typeof technicals?.bb_position === "number"
          ? (technicals.bb_position - 0.5) * 100
          : 0,
      changeLabel: "inside 20D band",
      trend:
        typeof technicals?.bb_position === "number"
          ? getTrend(technicals.bb_position - 0.5)
          : "neutral",
    },
  ] as const;

  const renderOverview = () => (
    <div className="space-y-6">
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-4 fade-in-up stagger-2">
        {detailQuery.isLoading || technicalsQuery.isLoading ? (
          Array.from({ length: 4 }, (_, index) => (
            <LoadingCard key={index} className="min-h-[148px]" lines={4} />
          ))
        ) : (
          stats.map((stat, index) => (
            <StatCard
              key={stat.label}
              label={stat.label}
              value={stat.value}
              change={stat.change}
              changeLabel={stat.changeLabel}
              trend={stat.trend}
              animateValue={false}
              delay={index * 60}
            />
          ))
        )}
      </div>

      <div className="flex flex-col gap-6 xl:flex-row items-stretch">
        <div className="flex-1 min-w-0 space-y-6">
          <div className="fade-in-up stagger-3">
            <KLineChart key={normalizedTicker} ticker={normalizedTicker} height={320} defaultRange="3M" />
          </div>
          {history.length > 0 && (
            <div className="bg-card rounded-2xl border border-border p-6 fade-in-up stagger-3 shadow-xl">
              <div className="flex items-center gap-3 mb-6">
                <div className="p-2 rounded-xl bg-primary/10">
                  <TrendingUp size={18} className="text-primary" />
                </div>
                <div>
                  <h3 className="text-sm font-black text-foreground uppercase tracking-widest">Model Signal History</h3>
                  <p className="text-[10px] text-muted-foreground uppercase tracking-tighter mt-0.5">Historical fusion scores across previous signal cycles</p>
                </div>
              </div>
              <SignalHistory history={history} height={220} />
            </div>
          )}
        </div>

        <div className="w-full xl:w-[340px] space-y-6 flex-shrink-0 flex flex-col fade-in-up stagger-3">
          {expectedReturn && <ExpectedReturnCard data={expectedReturn} />}
          
          <div className="bg-card rounded-2xl border border-border p-6 shadow-xl space-y-6 flex-1">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-xl bg-primary/10">
                <Brain size={18} className="text-primary" />
              </div>
              <div>
                <h3 className="text-sm font-black text-foreground uppercase tracking-widest">Technical Snapshot</h3>
                <p className="text-[10px] text-muted-foreground uppercase tracking-tighter mt-0.5">Momentum and trend indicators</p>
              </div>
            </div>

            {technicalsQuery.isLoading ? (
              <div className="space-y-4 animate-pulse">
                {Array.from({ length: 8 }, (_, index) => (
                  <div key={index} className="h-4 rounded-md bg-muted" style={{ width: `${92 - index * 4}%` }} />
                ))}
              </div>
            ) : technicalsQuery.isError ? (
              <p className="text-xs text-bear">{getErrorMessage(technicalsQuery.error)}</p>
            ) : (
              <div className="space-y-6">
                <div className="space-y-3">
                  <div className="text-[10px] font-black uppercase tracking-[0.2em] text-muted-foreground/50 border-b border-white/5 pb-1.5">Momentum</div>
                  <div className="flex items-center justify-between">
                    <span className="text-xs text-muted-foreground font-medium">RSI (14)</span>
                    <span className={`text-xs font-black font-mono px-2 py-0.5 rounded ${getRsiColor(technicals?.rsi_14)} bg-white/5`}>
                      {formatNumber(technicals?.rsi_14, 2)}
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-xs text-muted-foreground font-medium">MACD</span>
                    <span className="text-xs font-black font-mono text-foreground">{formatNumber(technicals?.macd, 4)}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-xs text-muted-foreground font-medium">Histogram</span>
                    <span className={`text-xs font-black font-mono ${getTrend(technicals?.macd_histogram) === "down" ? "text-bear" : "text-bull"}`}>
                      {formatNumber(technicals?.macd_histogram, 4)}
                    </span>
                  </div>
                </div>

                <div className="space-y-3">
                  <div className="text-[10px] font-black uppercase tracking-[0.2em] text-muted-foreground/50 border-b border-white/5 pb-1.5">Volatility</div>
                  <div className="flex items-center justify-between">
                    <span className="text-xs text-muted-foreground font-medium">BB Position</span>
                    <span className="text-xs font-black font-mono text-foreground bg-white/5 px-2 py-0.5 rounded">{formatNumber(technicals?.bb_position, 4)}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-xs text-muted-foreground font-medium">SMA 20</span>
                    <span className="text-xs font-black font-mono text-foreground">{formatCurrency(technicals?.sma_20)}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-xs text-muted-foreground font-medium">SMA 200</span>
                    <span className="text-xs font-black font-mono text-foreground">{formatCurrency(technicals?.sma_200)}</span>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      <div className="flex flex-col gap-6 xl:flex-row items-stretch xl:h-[420px]">
        <div className="flex-1 bg-card rounded-2xl border border-border p-6 shadow-xl fade-in-up stagger-4 flex flex-col min-h-[400px] xl:min-h-0">
          <div className="flex items-center gap-3 mb-6">
            <div className="p-2 rounded-xl bg-primary/10">
              <BarChart3 size={18} className="text-primary" />
            </div>
            <div>
              <h3 className="text-sm font-black text-foreground uppercase tracking-widest">Fundamental Composition</h3>
              <p className="text-[10px] text-muted-foreground uppercase tracking-tighter mt-0.5">Top financial metrics ranked by absolute magnitude</p>
            </div>
          </div>

          {fundamentalsQuery.isLoading ? (
            <div className="animate-pulse space-y-4 flex-1">
              {Array.from({ length: 6 }, (_, index) => (
                <div key={index} className="h-6 rounded-md bg-muted" style={{ width: `${94 - index * 6}%` }} />
              ))}
            </div>
          ) : fundamentalsQuery.isError ? (
            <p className="text-xs text-bear">{getErrorMessage(fundamentalsQuery.error)}</p>
          ) : topMetricData.length > 0 ? (
            <div className="flex-1 min-h-0">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={topMetricData}
                  layout="vertical"
                  margin={{ left: 12, right: 32, top: 0, bottom: 0 }}
                >
                  <XAxis
                    type="number"
                    tick={{ fill: "#607B96", fontSize: 10, fontWeight: 700 }}
                    axisLine={false}
                    tickLine={false}
                    tickFormatter={(value: number) => formatCompactNumber(value)}
                  />
                  <YAxis
                    type="category"
                    dataKey="label"
                    tick={{ fill: "#F5F7FA", fontSize: 10, fontWeight: 800 }}
                    axisLine={false}
                    tickLine={false}
                    width={120}
                  />
                  <Tooltip
                    cursor={{ fill: "rgba(255,255,255,0.03)" }}
                    content={({ active, payload }: { active?: boolean; payload?: Array<{ payload: { fullLabel: string; value: number; positive: boolean } }> }) => {
                      if (!active || !payload?.length) return null;
                      const point = payload[0].payload;
                      return (
                        <div className="bg-[#131C2E] border border-white/10 rounded-xl px-3 py-2 shadow-2xl backdrop-blur-xl">
                          <p className="text-[10px] font-black uppercase tracking-widest text-muted-foreground mb-1">{point.fullLabel}</p>
                          <p className={`text-sm font-black font-mono ${point.positive ? "text-bull" : "text-bear"}`}>
                            {point.value >= 0 ? "+" : ""}
                            {formatCompactNumber(point.value)}
                          </p>
                        </div>
                      );
                    }}
                  />
                  <Bar dataKey="value" radius={[0, 6, 6, 0]} barSize={16}>
                    {topMetricData.map((entry) => (
                      <Cell
                        key={entry.metric}
                        fill={entry.positive ? "#00C805" : "#FF5252"}
                        fillOpacity={0.85}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <div className="flex-1 flex items-center justify-center border border-dashed border-border rounded-2xl bg-muted/5">
              <p className="text-xs text-muted-foreground">No fundamental metrics are available.</p>
            </div>
          )}
        </div>

        <div className="w-full xl:w-[340px] bg-card rounded-2xl border border-border shadow-xl fade-in-up stagger-5 flex-shrink-0 flex flex-col min-h-[400px] xl:min-h-0">
          <div className="p-6 border-b border-white/5">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-xl bg-primary/10">
                <Layers size={18} className="text-primary" />
              </div>
              <div>
                <h3 className="text-sm font-black text-foreground uppercase tracking-widest">Financial Matrix</h3>
                <p className="text-[10px] text-muted-foreground uppercase tracking-tighter mt-0.5">Complete metric inventory</p>
              </div>
            </div>
          </div>

          <div className="flex-1 overflow-y-auto no-scrollbar p-6 pt-2">
            {fundamentalsQuery.isLoading ? (
              <div className="space-y-4 animate-pulse">
                {Array.from({ length: 12 }, (_, index) => (
                  <div key={index} className="h-4 rounded-md bg-muted" style={{ width: `${90 - index * 3}%` }} />
                ))}
              </div>
            ) : fundamentalsQuery.isError ? (
              <p className="text-xs text-bear">{getErrorMessage(fundamentalsQuery.error)}</p>
            ) : metricEntries.length > 0 ? (
              <div className="space-y-3.5">
                {metricEntries.map(([metric, value]) => (
                  <div key={metric} className="flex items-center justify-between gap-4 py-1 border-b border-white/[0.03] last:border-0 group transition-all duration-300">
                    <span className="text-[11px] font-bold text-muted-foreground group-hover:text-foreground group-hover:translate-x-1 transition-all">{formatMetricLabel(metric)}</span>
                    <span className={`text-[11px] font-black font-mono px-2 py-0.5 rounded bg-white/5 group-hover:bg-white/10 transition-colors ${typeof value === "number" && value < 0 ? "text-bear" : "text-foreground"}`}>
                      {typeof value === "number" ? formatCompactNumber(value) : "—"}
                    </span>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-xs text-muted-foreground">No fundamental metrics are available.</p>
            )}
          </div>
        </div>
      </div>
    </div>
  );

  return (
    <div className="flex-1 overflow-y-auto p-6 space-y-6 bg-[#0D1421]">
      <div className="flex items-start gap-4 fade-in-up">
        <button
          onClick={onBack}
          className="flex items-center gap-2 text-xs font-black uppercase tracking-widest text-muted-foreground hover:text-primary transition-all group mt-0.5"
        >
          <ArrowLeft size={14} className="transition-transform group-hover:-translate-x-1" />
          Return to Signals
        </button>
      </div>

      <div className="flex flex-col gap-4">
        {/* Row 1: Primary Entity Info */}
        <div className="flex items-center gap-6">
          <div className="flex items-center gap-4 min-w-0 flex-1">
            <div className="w-14 h-14 rounded-2xl bg-muted flex items-center justify-center flex-shrink-0">
              <span className="text-2xl font-black text-foreground">{normalizedTicker[0]}</span>
            </div>
            
            <div className="flex items-center gap-3 min-w-0">
              <h2 className="text-2xl font-black text-foreground tracking-tight flex-shrink-0">{detail?.ticker ?? normalizedTicker}</h2>
              <span className="text-sm font-bold text-muted-foreground truncate max-w-[240px]">
                · {detail?.company_name}
              </span>
              {detail?.sector && (
                <span
                  className="text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 rounded border flex-shrink-0"
                  style={{
                    backgroundColor: getSectorColor(detail.sector).bg,
                    color: getSectorColor(detail.sector).text,
                    borderColor: getSectorColor(detail.sector).border,
                  }}
                >
                  {detail.sector}
                </span>
              )}
              {detail?.industry && (
                <span className="text-[10px] font-bold text-muted-foreground/60 uppercase tracking-wider bg-white/5 px-2 py-0.5 rounded border border-white/5 flex-shrink-0">
                  {detail.industry}
                </span>
              )}
            </div>
          </div>

          <div className="flex items-center gap-6 flex-shrink-0">
            <div className="text-right">
              <div className="text-3xl font-black text-foreground font-mono tracking-tighter">
                {formatCurrency(latestPrice?.close)}
              </div>
              <div
                className={`text-sm font-bold font-mono mt-0.5 flex items-center justify-end gap-1.5 ${
                  getTrend(latestPrice?.change_pct) === "down" ? "text-bear" : "text-bull"
                }`}
              >
                {getTrend(latestPrice?.change_pct) === "up" ? <TrendingUp size={14} /> : <TrendingDown size={14} />}
                {formatCurrency(latestPrice?.change)} ({formatPercent(latestPrice?.change_pct)})
              </div>
            </div>
          </div>
        </div>

        <div className="h-px bg-white/5" />

        {/* Row 2: Signal Metadata */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            {predictionQuery.isLoading && !prediction ? (
              <div className="h-6 w-64 bg-muted animate-pulse rounded-lg" />
            ) : prediction ? (
              (() => {
                const tier: "strong" | "long" | "watch" | "buffer" =
                  prediction.fusion_score <= 0 ? "buffer"
                  : prediction.percentile >= 75 ? "strong"
                  : prediction.percentile < 25 ? "watch"
                  : "long";
                const TIER_META = {
                  strong: { label: "STRONG LONG", className: "tag-bull-strong" },
                  long:   { label: "LONG",        className: "tag-bull" },
                  watch:  { label: "WATCH LONG",  className: "tag-bull-watch" },
                  buffer: { label: "BUFFER",      className: "tag-neutral" },
                } as const;
                const meta = TIER_META[tier];
                const hasMultiModel = Object.keys(prediction.model_scores || {}).length > 1;
                return (
                  <>
                    <div className={`flex items-center gap-1.5 px-2 py-0.5 rounded-lg text-[10px] font-bold uppercase tracking-wider border ${meta.className}`}>
                      <Target size={12} />
                      {meta.label}
                    </div>

                    <span className="text-[11px] font-bold text-foreground font-mono">
                      Score: {prediction.fusion_score.toFixed(4)}
                    </span>

                    <span className="text-[11px] font-bold text-muted-foreground/80">
                      Rank #{prediction.rank} · Top {prediction.percentile.toFixed(1)}%
                    </span>

                    {hasMultiModel && prediction.confidence != null && (
                      <div className={`text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 rounded-lg border ${
                        prediction.confidence === "high" ? "bg-bull/10 border-bull/20 text-bull" :
                        prediction.confidence === "medium" ? "bg-amber-500/10 border-amber-500/20 text-amber-500" :
                        "bg-bear/10 border-bear/20 text-bear"
                      }`}>
                        {prediction.confidence} confidence
                      </div>
                    )}

                    {hasMultiModel && prediction.model_agreement != null && (
                      <span className="text-[10px] font-bold uppercase tracking-wider text-muted-foreground/60">
                        {Math.round(prediction.model_agreement * 100)}% Consensus
                      </span>
                    )}
                  </>
                );
              })()
            ) : isPrediction404 ? (
              <div className="flex items-center gap-2 px-2 py-0.5 rounded-lg border bg-muted/30 border-white/5 text-muted-foreground/50">
                <div className="w-1.5 h-1.5 rounded-full bg-muted-foreground/30" />
                <span className="text-[10px] font-bold uppercase tracking-wider">No Active Signal</span>
              </div>
            ) : null}
          </div>

          <div className="flex items-center gap-1.5 opacity-40">
            <ShieldCheck size={12} className="text-muted-foreground" />
            <span className="text-[10px] font-medium uppercase tracking-[0.1em] text-muted-foreground">
              Model Verified Outcome · Institutional use only
            </span>
          </div>
        </div>
      </div>

      <div className="flex gap-1 bg-muted rounded-xl p-1 w-fit mt-2">
        {tabs.map((tab) => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={`px-6 py-2 rounded-lg text-sm font-medium capitalize transition-all duration-200 relative ${
              activeTab === tab
                ? "bg-card text-foreground shadow-custom"
                : "text-muted-foreground hover:text-foreground"
            }`}
          >
            {tab}
          </button>
        ))}
      </div>

      {activeTab === "overview" ? (
        renderOverview()
      ) : activeTab === "factors" ? (
        <div className="space-y-6 animate-in fade-in duration-500">
          {isPrediction404 ? (
            <div className="bg-card rounded-2xl border border-border p-20 flex flex-col items-center justify-center text-center fade-in-up stagger-2 shadow-2xl">
              <div className="p-4 rounded-3xl bg-muted/50 mb-6">
                <Brain size={64} className="text-muted-foreground opacity-20" />
              </div>
              <h3 className="text-xl font-black text-foreground mb-3 uppercase tracking-widest">No Factor Attribution</h3>
              <p className="text-sm text-muted-foreground max-w-md leading-relaxed font-medium">
                Factor-level decomposition is currently only available for tickers within the high-conviction signal universe.
              </p>
            </div>
          ) : (
            <>
              <div className="bg-card rounded-2xl border border-border p-6 fade-in-up stagger-2 shadow-xl">
                <div className="flex items-center justify-between mb-8 border-b border-white/5 pb-6">
                  <div className="flex items-center gap-3">
                    <div className="p-2 rounded-xl bg-primary/10">
                      <Layers size={18} className="text-primary" />
                    </div>
                    <div>
                      <h3 className="text-sm font-black text-foreground uppercase tracking-widest">Feature Contribution (SHAP)</h3>
                      <p className="text-[10px] text-muted-foreground uppercase tracking-tighter mt-0.5">Top 15 attributes impacting the current model cycle</p>
                    </div>
                  </div>
                  {prediction?.signal_date && (
                    <div className="text-[10px] font-mono font-black text-muted-foreground/50 px-3 py-1.5 rounded-full bg-muted/50 border border-white/5 shadow-inner">
                      REFERENCE: {prediction.signal_date}
                    </div>
                  )}
                </div>
                
                {shapQuery.isLoading ? (
                  <div className="h-[450px] flex items-center justify-center animate-pulse bg-muted/20 rounded-2xl">
                    <p className="text-xs font-black uppercase tracking-[0.2em] text-muted-foreground animate-pulse">Running attribution engine...</p>
                  </div>
                ) : (isShap404 || (shap && shap.features.length === 0)) ? (
                  <div className="h-[350px] flex flex-col items-center justify-center border border-dashed border-border rounded-2xl bg-muted/5">
                    <Info size={32} className="text-muted-foreground mb-4 opacity-20" />
                    <p className="text-sm font-bold text-muted-foreground uppercase tracking-widest">SHAP attribution pending</p>
                    <p className="text-[10px] text-muted-foreground/50 mt-2 uppercase tracking-tighter">Detailed factor weights are updated periodically</p>
                  </div>
                ) : shapQuery.isError ? (
                  <div className="h-[350px] flex flex-col items-center justify-center border border-dashed border-border rounded-2xl bg-bear/5">
                    <AlertCircle size={32} className="text-bear mb-4 opacity-50" />
                    <p className="text-sm font-black text-bear uppercase tracking-widest">Attribution Feed Error</p>
                  </div>
                ) : (
                  <ShapWaterfall features={shap!.features} height={480} />
                )}
              </div>

              {(() => {
                const isMultiModel = prediction && Object.keys(prediction.model_scores || {}).length > 1;
                return (
                  <div className={`grid grid-cols-1 ${isMultiModel ? "md:grid-cols-3" : "md:grid-cols-1"} gap-6 fade-in-up stagger-3`}>
                    {isMultiModel && (
                      <div className="md:col-span-2 bg-card rounded-2xl border border-border overflow-hidden shadow-xl flex flex-col">
                        <div className="p-6 border-b border-white/5 bg-muted/10">
                          <div className="flex items-center gap-3">
                            <div className="p-2 rounded-xl bg-primary/10">
                              <ShieldCheck size={18} className="text-primary" />
                            </div>
                            <div>
                              <h3 className="text-sm font-black text-foreground uppercase tracking-widest">Architecture Consensus</h3>
                              <p className="text-[10px] text-muted-foreground uppercase tracking-tighter mt-0.5">Agreement across specialized model learners</p>
                            </div>
                          </div>
                        </div>
                        <div className="flex-1 overflow-x-auto no-scrollbar">
                          <table className="w-full text-left">
                            <thead>
                              <tr className="border-b border-white/5 bg-muted/5">
                                <th className="px-6 py-4 text-[10px] font-black text-muted-foreground uppercase tracking-[0.2em]">Learner</th>
                                <th className="px-6 py-4 text-[10px] font-black text-muted-foreground uppercase tracking-[0.2em] text-right">Raw Score</th>
                                <th className="px-6 py-4 text-[10px] font-black text-muted-foreground uppercase tracking-[0.2em] text-right">Influence</th>
                                <th className="px-6 py-4 text-[10px] font-black text-muted-foreground uppercase tracking-[0.2em] text-right">Contribution</th>
                              </tr>
                            </thead>
                            <tbody className="divide-y divide-white/[0.03]">
                              {prediction && Object.entries(prediction.model_scores).map(([model, score]) => (
                                <tr key={model} className="group hover:bg-white/[0.01] transition-colors">
                                  <td className="px-6 py-4 text-xs font-black text-foreground/80 uppercase tracking-widest">{model}</td>
                                  <td className={`px-6 py-4 text-xs font-black font-mono text-right ${score > 0 ? "text-bull" : "text-bear"}`}>
                                    {score.toFixed(4)}
                                  </td>
                                  <td className="px-6 py-4 text-[10px] font-bold text-muted-foreground text-right uppercase tracking-tighter">
                                    {(100 / Object.keys(prediction.model_scores).length).toFixed(1)}%
                                  </td>
                                  <td className={`px-6 py-4 text-xs font-black font-mono text-right ${score > 0 ? "text-bull" : "text-bear"}`}>
                                    {(score / Object.keys(prediction.model_scores).length).toFixed(4)}
                                  </td>
                                </tr>
                              ))}
                            </tbody>
                            <tfoot>
                              <tr className="border-t border-white/10">
                                <td className="px-6 py-5 text-xs font-black text-primary uppercase tracking-[0.2em]">Integrated Fusion</td>
                                <td className="px-6 py-5 text-right" />
                                <td className="px-6 py-5 text-right" />
                                <td className={`px-6 py-5 text-sm font-black font-mono text-right ${prediction && prediction.fusion_score > 0 ? "text-bull" : "text-bear"}`}>
                                  {prediction?.fusion_score.toFixed(4)}
                                </td>
                              </tr>
                            </tfoot>
                          </table>
                        </div>
                      </div>
                    )}

                    <div className="bg-card rounded-2xl border border-border p-6 shadow-xl space-y-6">
                      <h3 className="text-sm font-black text-foreground uppercase tracking-widest">Predictive Logic</h3>
                      <p className="text-xs text-muted-foreground leading-relaxed font-medium">
                        The current signal for <span className="text-foreground font-black tracking-tight">{normalizedTicker}</span> is primarily driven by
                        the <span className="text-primary font-black uppercase tracking-tighter">{(shap?.features[0]?.feature || "core model").replace(/_/g, " ")}</span> factor.
                        {isMultiModel
                          ? <> Architecture consensus is <span className="text-foreground font-black">{prediction && prediction.fusion_score > 0 ? "positively biased" : "defensively biased"}</span>.</>
                          : <> Model output is <span className="text-foreground font-black">{prediction && prediction.fusion_score > 0 ? "positively biased" : "buffer-held (signal weakening)"}</span>.</>}
                      </p>
                      <div className="p-4 rounded-2xl bg-muted/30 border border-white/5 shadow-inner space-y-4">
                        <div>
                          <p className="text-[9px] text-muted-foreground uppercase tracking-[0.2em] font-black mb-3">Model Conviction</p>
                          <div className="h-3 bg-muted rounded-full overflow-hidden border border-white/5 p-0.5">
                            <div
                              className={`h-full rounded-full transition-all duration-1000 ${prediction && prediction.fusion_score > 0 ? "bg-bull shadow-[0_0_10px_#00C805]" : "bg-bear shadow-[0_0_10px_#FF5252]"}`}
                              style={{ width: `${prediction ? Math.min(Math.abs(prediction.fusion_score) * 20, 100) : 0}%` }}
                            />
                          </div>
                        </div>
                        {isMultiModel && prediction?.model_spread != null && (
                          <div className="flex justify-between items-center">
                            <span className="text-[9px] text-muted-foreground font-black uppercase tracking-widest">Stability</span>
                            <span className="text-[10px] font-black font-mono text-foreground">{(1 - (prediction?.model_spread || 0)).toFixed(2)}</span>
                          </div>
                        )}
                        <div className="flex justify-between items-center">
                          <span className="text-[9px] text-muted-foreground font-black uppercase tracking-widest">Percentile</span>
                          <span className="text-[10px] font-black font-mono text-foreground">{prediction?.percentile?.toFixed(1)}%</span>
                        </div>
                      </div>
                    </div>
                  </div>
                );
              })()}
            </>
          )}
        </div>
      ) : (
        <div className="bg-card rounded-2xl border border-border p-20 flex flex-col items-center justify-center text-center fade-in-up stagger-2 shadow-2xl">
          <div className="p-4 rounded-3xl bg-muted/50 mb-6">
            <Info size={64} className="text-muted-foreground opacity-20" />
          </div>
          <h3 className="text-xl font-black text-foreground mb-3 uppercase tracking-widest">{activeTab} Integration</h3>
          <p className="text-sm text-muted-foreground max-w-md leading-relaxed font-medium">
            The {activeTab} engine is undergoing final stress testing and will be operational in the Phase 4 production release.
          </p>
          <div className="mt-8 px-6 py-2 rounded-xl bg-primary/10 text-[10px] font-black uppercase tracking-[0.2em] text-primary border border-primary/20 shadow-lg">
            Status: Phase 4 Milestone
          </div>
        </div>
      )}
    </div>
  );
};

export default SignalDetail;
