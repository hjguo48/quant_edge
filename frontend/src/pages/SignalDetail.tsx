import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { ArrowLeft, Brain, BarChart3, Info, TrendingUp, Layers } from "lucide-react";
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
      <div className="flex-1 overflow-y-auto p-6 space-y-5">
        <div className="flex items-start gap-4 fade-in-up">
          <button
            onClick={onBack}
            className="flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground transition-colors group mt-0.5"
          >
            <ArrowLeft size={16} className="transition-transform group-hover:-translate-x-0.5" />
            Back to Signals
          </button>
        </div>

        <div className="bg-card rounded-xl border border-border p-6 fade-in-up stagger-1">
          <div className="flex items-center gap-2 mb-3">
            <Info size={16} className="text-bear" />
            <h2 className="text-lg font-semibold text-foreground">Unable to load {normalizedTicker}</h2>
          </div>
          <p className="text-sm text-muted-foreground">{getErrorMessage(detailQuery.error)}</p>
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
    <>
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

      <div className="flex flex-col gap-5 xl:flex-row">
        <div className="flex-1 min-w-0">
          <div className="fade-in-up stagger-3">
            <KLineChart key={normalizedTicker} ticker={normalizedTicker} height={260} defaultRange="3M" />
          </div>
          {history.length > 0 && (
            <div className="mt-5 bg-card rounded-xl border border-border p-5 fade-in-up stagger-3">
              <div className="flex items-center gap-2 mb-4">
                <TrendingUp size={14} className="text-primary" />
                <h3 className="text-sm font-semibold text-foreground">Model Signal History</h3>
              </div>
              <SignalHistory history={history} height={200} />
            </div>
          )}
        </div>

        <div className="w-full xl:w-80 bg-card rounded-xl border border-border p-5 flex-shrink-0 fade-in-up stagger-3">
          <div className="flex items-center gap-2 mb-1">
            <Brain size={14} className="text-primary" />
            <h3 className="text-sm font-semibold text-foreground">Technical Indicators</h3>
          </div>
          <p className="text-xs text-muted-foreground mb-4">
            Latest snapshot{technicals?.trade_date ? ` · ${formatDate(technicals.trade_date)}` : ""}
          </p>

          {technicalsQuery.isLoading ? (
            <div className="space-y-3 animate-pulse">
              {Array.from({ length: 10 }, (_, index) => (
                <div key={index} className="h-4 rounded-md bg-muted" style={{ width: `${92 - index * 4}%` }} />
              ))}
            </div>
          ) : technicalsQuery.isError ? (
            <p className="text-xs text-bear">{getErrorMessage(technicalsQuery.error)}</p>
          ) : (
            <div className="space-y-5">
              <div className="space-y-3">
                <div className="text-[10px] font-semibold uppercase tracking-[0.18em] text-muted-foreground">Momentum</div>
                <div className="flex items-center justify-between">
                  <span className="text-xs text-muted-foreground">RSI 14</span>
                  <span className={`text-xs font-bold font-mono ${getRsiColor(technicals?.rsi_14)}`}>
                    {formatNumber(technicals?.rsi_14, 2)}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-xs text-muted-foreground">MACD</span>
                  <span className="text-xs font-bold font-mono text-foreground">{formatNumber(technicals?.macd, 4)}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-xs text-muted-foreground">Signal</span>
                  <span className="text-xs font-bold font-mono text-foreground">{formatNumber(technicals?.macd_signal, 4)}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-xs text-muted-foreground">Histogram</span>
                  <span className={`text-xs font-bold font-mono ${getTrend(technicals?.macd_histogram) === "down" ? "text-bear" : "text-bull"}`}>
                    {formatNumber(technicals?.macd_histogram, 4)}
                  </span>
                </div>
              </div>

              <div className="space-y-3">
                <div className="text-[10px] font-semibold uppercase tracking-[0.18em] text-muted-foreground">Bollinger Bands</div>
                <div className="flex items-center justify-between">
                  <span className="text-xs text-muted-foreground">Upper</span>
                  <span className="text-xs font-bold font-mono text-foreground">{formatCurrency(technicals?.bb_upper)}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-xs text-muted-foreground">Middle</span>
                  <span className="text-xs font-bold font-mono text-foreground">{formatCurrency(technicals?.bb_middle)}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-xs text-muted-foreground">Lower</span>
                  <span className="text-xs font-bold font-mono text-foreground">{formatCurrency(technicals?.bb_lower)}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-xs text-muted-foreground">Width</span>
                  <span className="text-xs font-bold font-mono text-foreground">{formatNumber(technicals?.bb_width, 4)}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-xs text-muted-foreground">Position</span>
                  <span className="text-xs font-bold font-mono text-foreground">{formatNumber(technicals?.bb_position, 4)}</span>
                </div>
              </div>

              <div className="space-y-3">
                <div className="text-[10px] font-semibold uppercase tracking-[0.18em] text-muted-foreground">Moving Averages</div>
                <div className="flex items-center justify-between">
                  <span className="text-xs text-muted-foreground">SMA 20</span>
                  <span className="text-xs font-bold font-mono text-foreground">{formatCurrency(technicals?.sma_20)}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-xs text-muted-foreground">SMA 50</span>
                  <span className="text-xs font-bold font-mono text-foreground">{formatCurrency(technicals?.sma_50)}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-xs text-muted-foreground">SMA 200</span>
                  <span className="text-xs font-bold font-mono text-foreground">{formatCurrency(technicals?.sma_200)}</span>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      <div className="flex flex-col gap-5 xl:flex-row">
        <div className="flex-1 bg-card rounded-xl border border-border p-5 fade-in-up stagger-4">
          <div className="flex items-center gap-2 mb-1">
            <BarChart3 size={14} className="text-primary" />
            <h3 className="text-sm font-semibold text-foreground">Fundamental Snapshot</h3>
          </div>
          <p className="text-xs text-muted-foreground mb-4">
            {fundamentals?.fiscal_period ? `${fundamentals.fiscal_period}` : "Latest filing"}
            {fundamentals?.event_time ? ` · ${formatDate(fundamentals.event_time)}` : ""}
          </p>

          {fundamentalsQuery.isLoading ? (
            <div className="animate-pulse space-y-3">
              {Array.from({ length: 8 }, (_, index) => (
                <div key={index} className="h-5 rounded-md bg-muted" style={{ width: `${94 - index * 6}%` }} />
              ))}
            </div>
          ) : fundamentalsQuery.isError ? (
            <p className="text-xs text-bear">{getErrorMessage(fundamentalsQuery.error)}</p>
          ) : topMetricData.length > 0 ? (
            <ResponsiveContainer width="100%" height={260}>
              <BarChart
                data={topMetricData}
                layout="vertical"
                margin={{ left: 8, right: 16, top: 0, bottom: 0 }}
              >
                <XAxis
                  type="number"
                  tick={{ fill: "#607B96", fontSize: 10 }}
                  axisLine={false}
                  tickLine={false}
                  tickFormatter={(value: number) => formatCompactNumber(value)}
                />
                <YAxis
                  type="category"
                  dataKey="label"
                  tick={{ fill: "#607B96", fontSize: 10 }}
                  axisLine={false}
                  tickLine={false}
                  width={110}
                />
                <Tooltip
                  cursor={{ fill: "rgba(255,255,255,0.03)" }}
                  content={({ active, payload }: { active?: boolean; payload?: Array<{ payload: { fullLabel: string; value: number; positive: boolean } }> }) => {
                    if (!active || !payload?.length) return null;
                    const point = payload[0].payload;
                    return (
                      <div className="bg-popover border border-border rounded-lg px-2.5 py-1.5 shadow-custom">
                        <p className="text-xs text-muted-foreground">{point.fullLabel}</p>
                        <p className={`text-xs font-bold ${point.positive ? "text-bull" : "text-bear"}`}>
                          {point.value >= 0 ? "+" : ""}
                          {formatCompactNumber(point.value)}
                        </p>
                      </div>
                    );
                  }}
                />
                <Bar dataKey="value" radius={[0, 4, 4, 0]}>
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
          ) : (
            <p className="text-xs text-muted-foreground">No fundamental metrics are available.</p>
          )}
        </div>

        <div className="w-full xl:w-80 bg-card rounded-xl border border-border p-5 flex-shrink-0 fade-in-up stagger-5">
          <div className="flex items-center gap-2 mb-1">
            <BarChart3 size={14} className="text-primary" />
            <h3 className="text-sm font-semibold text-foreground">Fundamental Metrics</h3>
          </div>
          <p className="text-xs text-muted-foreground mb-4">
            {fundamentals?.knowledge_time ? `Visible ${formatDate(fundamentals.knowledge_time)}` : "Latest reported metrics"}
          </p>

          {fundamentalsQuery.isLoading ? (
            <div className="space-y-3 animate-pulse">
              {Array.from({ length: 10 }, (_, index) => (
                <div key={index} className="h-4 rounded-md bg-muted" style={{ width: `${90 - index * 4}%` }} />
              ))}
            </div>
          ) : fundamentalsQuery.isError ? (
            <p className="text-xs text-bear">{getErrorMessage(fundamentalsQuery.error)}</p>
          ) : metricEntries.length > 0 ? (
            <div className="space-y-3 max-h-[320px] overflow-y-auto pr-1">
              {metricEntries.map(([metric, value]) => (
                <div key={metric} className="flex items-center justify-between gap-3">
                  <span className="text-xs text-muted-foreground">{formatMetricLabel(metric)}</span>
                  <span className={`text-xs font-bold font-mono ${typeof value === "number" && value < 0 ? "text-bear" : "text-foreground"}`}>
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
    </>
  );

  return (
    <div className="flex-1 overflow-y-auto p-6 space-y-5">
      <div className="flex items-start gap-4 fade-in-up">
        <button
          onClick={onBack}
          className="flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground transition-colors group mt-0.5"
        >
          <ArrowLeft size={16} className="transition-transform group-hover:-translate-x-0.5" />
          Back to Signals
        </button>
      </div>

      <div className="flex flex-col gap-5 xl:flex-row xl:items-center fade-in-up stagger-1">
        <div className="flex items-start gap-5 min-w-0">
          <div className="w-14 h-14 rounded-2xl bg-muted flex items-center justify-center flex-shrink-0">
            <span className="text-xl font-black text-foreground">{normalizedTicker[0]}</span>
          </div>

          {detailQuery.isLoading && !detail ? (
            <div className="space-y-3 animate-pulse min-w-[260px]">
              <div className="h-7 w-44 rounded-md bg-muted" />
              <div className="h-5 w-64 rounded-md bg-muted" />
              <div className="h-4 w-56 rounded-md bg-muted" />
            </div>
          ) : (
            <div className="min-w-0">
              <div className="flex flex-wrap items-center gap-3">
                <h2 className="text-2xl font-bold text-foreground">{detail?.ticker ?? normalizedTicker}</h2>
                {prediction && (
                  <div className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full border ${prediction.fusion_score > 0 ? "bg-bull/10 border-bull/20 text-bull" : "bg-bear/10 border-bear/20 text-bear"}`}>
                    <span className="text-xs font-bold uppercase tracking-tight">
                      Score: {prediction.fusion_score.toFixed(4)}
                    </span>
                    <span className="w-1 h-1 rounded-full bg-current opacity-40" />
                    <span className="text-xs font-bold">
                      Rank #{prediction.rank}
                    </span>
                    <span className="w-1 h-1 rounded-full bg-current opacity-40" />
                    <span className="text-xs font-bold">
                      Top {prediction.percentile.toFixed(1)}%
                    </span>
                  </div>
                )}
                {predictionQuery.isLoading && (
                  <div className="h-6 w-32 bg-muted animate-pulse rounded-full" />
                )}
                {isPrediction404 && (
                  <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-full border bg-muted/30 border-border text-muted-foreground">
                    <span className="text-[10px] font-bold uppercase">No Active Signal</span>
                  </div>
                )}
                {detail?.sector && (
                  <span
                    className="text-xs font-semibold px-2 py-0.5 rounded-md border"
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
                  <span className="text-xs text-muted-foreground px-2 py-1 rounded-lg bg-muted/50 border border-border/50">
                    {detail.industry}
                  </span>
                )}
              </div>
              <p className="text-sm text-muted-foreground mt-1">
                {detail?.company_name || "Loading company details"}
              </p>
              <p className="text-xs text-muted-foreground mt-2">
                {detail?.ipo_date ? `IPO ${formatDate(detail.ipo_date)} · ` : ""}
                {latestPrice?.trade_date ? `Updated ${formatDate(latestPrice.trade_date)}` : "Waiting for latest quote"}
              </p>
            </div>
          )}
        </div>

        <div className="xl:ml-auto flex flex-col gap-3 xl:items-end">
          {detailQuery.isLoading && !detail ? (
            <div className="space-y-2 animate-pulse xl:text-right">
              <div className="h-9 w-40 rounded-md bg-muted" />
              <div className="h-5 w-28 rounded-md bg-muted" />
            </div>
          ) : (
            <div className="xl:text-right">
              <div className="text-3xl font-bold text-foreground font-mono number-animate">
                {formatCurrency(latestPrice?.close)}
              </div>
              <div
                className={`text-sm font-semibold font-mono ${
                  getTrend(latestPrice?.change_pct) === "down" ? "text-bear" : "text-bull"
                }`}
              >
                {formatCurrency(latestPrice?.change)} · {formatPercent(latestPrice?.change_pct)}
              </div>
            </div>
          )}

          <div className="flex items-center gap-2 px-3 py-2 rounded-xl bg-muted border border-primary/20">
            <Info size={14} className="text-primary flex-shrink-0" />
            <span className="text-xs text-muted-foreground">
              Model output only. Not investment advice.
            </span>
          </div>
        </div>
      </div>

      {hasSectionError && (
        <div className="flex items-center gap-2 px-4 py-2.5 rounded-lg bg-accent border border-border fade-in-up stagger-1">
          <Info size={14} className="text-bear flex-shrink-0" />
          <p className="text-xs text-muted-foreground">
            Some sections could not be loaded. Available data is still shown where possible.
          </p>
        </div>
      )}

      <div className="flex gap-1 bg-muted rounded-xl p-1 w-fit fade-in-up stagger-1">
        {tabs.map((tab) => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={`px-4 py-2 rounded-lg text-sm font-medium capitalize transition-all duration-200 ${
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
        <div className="space-y-5">
          {isPrediction404 ? (
            <div className="bg-card rounded-xl border border-border p-12 flex flex-col items-center justify-center text-center fade-in-up stagger-2">
              <Brain size={48} className="text-muted-foreground mb-4 opacity-20" />
              <h3 className="text-lg font-bold text-foreground mb-2">No Factor Analysis Available</h3>
              <p className="text-sm text-muted-foreground max-w-md">
                This ticker is not currently part of the active signal universe. Factor-level attribution is only calculated for tickers with active model predictions.
              </p>
            </div>
          ) : (
            <>
              <div className="bg-card rounded-xl border border-border p-5 fade-in-up stagger-2">
                <div className="flex items-center justify-between mb-6">
                  <div>
                    <h3 className="text-sm font-semibold text-foreground">Feature Contribution (SHAP)</h3>
                    <p className="text-xs text-muted-foreground mt-0.5">Top 15 features impacting this week's signal</p>
                  </div>
                  {prediction?.signal_date && (
                    <div className="text-[10px] font-mono text-muted-foreground px-2 py-1 rounded bg-muted">
                      REF: {prediction.signal_date}
                    </div>
                  )}
                </div>
                
                {shapQuery.isLoading ? (
                  <div className="h-[400px] flex items-center justify-center animate-pulse bg-muted/20 rounded-lg">
                    <p className="text-sm text-muted-foreground">Analyzing feature importance...</p>
                  </div>
                ) : (isShap404 || (shap && shap.features.length === 0)) ? (
                  <div className="h-[300px] flex flex-col items-center justify-center border border-dashed border-border rounded-xl">
                    <Info size={24} className="text-muted-foreground mb-2 opacity-20" />
                    <p className="text-xs text-muted-foreground">
                      SHAP data not yet available for this ticker.
                    </p>
                    <p className="text-[10px] text-muted-foreground mt-1">Detailed attribution is updated periodically.</p>
                  </div>
                ) : shapQuery.isError ? (
                  <div className="h-[300px] flex flex-col items-center justify-center border border-dashed border-border rounded-xl">
                    <AlertCircle size={24} className="text-bear mb-2 opacity-50" />
                    <p className="text-xs text-muted-foreground">Failed to load factor analysis.</p>
                  </div>
                ) : (
                  <ShapWaterfall features={shap!.features} height={450} />
                )}
              </div>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 fade-in-up stagger-3">
                <div className="md:col-span-2 bg-card rounded-xl border border-border p-5">
                  <div className="flex items-center gap-2 mb-4">
                    <Layers size={14} className="text-primary" />
                    <h3 className="text-sm font-semibold text-foreground">Model Consensus</h3>
                  </div>
                  <div className="overflow-x-auto">
                    <table className="w-full text-left">
                      <thead>
                        <tr className="border-b border-border">
                          <th className="pb-3 text-xs font-semibold text-muted-foreground uppercase tracking-wider">Model Architecture</th>
                          <th className="pb-3 text-xs font-semibold text-muted-foreground uppercase tracking-wider text-right">Raw Score</th>
                          <th className="pb-3 text-xs font-semibold text-muted-foreground uppercase tracking-wider text-right">Weight</th>
                          <th className="pb-3 text-xs font-semibold text-muted-foreground uppercase tracking-wider text-right">Contribution</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-border/50">
                        {prediction && Object.entries(prediction.model_scores).map(([model, score]) => (
                          <tr key={model}>
                            <td className="py-3 text-xs font-medium text-foreground capitalize">{model}</td>
                            <td className={`py-3 text-xs font-mono font-bold text-right ${score > 0 ? "text-bull" : "text-bear"}`}>
                              {score.toFixed(4)}
                            </td>
                            <td className="py-3 text-xs text-muted-foreground text-right">
                              {(100 / Object.keys(prediction.model_scores).length).toFixed(1)}%
                            </td>
                            <td className={`py-3 text-xs font-mono font-bold text-right ${score > 0 ? "text-bull" : "text-bear"}`}>
                              {(score / Object.keys(prediction.model_scores).length).toFixed(4)}
                            </td>
                          </tr>
                        ))}
                        <tr className="bg-muted/30">
                          <td className="py-3 px-2 text-xs font-bold text-foreground">Fusion Score (Ensemble)</td>
                          <td className="py-3 text-right" />
                          <td className="py-3 text-right" />
                          <td className={`py-3 pr-2 text-xs font-mono font-black text-right border-t border-primary/20 ${prediction && prediction.fusion_score > 0 ? "text-bull" : "text-bear"}`}>
                            {prediction?.fusion_score.toFixed(4)}
                          </td>
                        </tr>
                      </tbody>
                    </table>
                  </div>
                </div>
                
                <div className="bg-card rounded-xl border border-border p-5">
                  <h3 className="text-sm font-semibold text-foreground mb-3">Analysis Note</h3>
                  <p className="text-xs text-muted-foreground leading-relaxed">
                    The current signal for <span className="text-foreground font-semibold">{normalizedTicker}</span> is primarily driven by 
                    the <span className="text-foreground font-semibold">{(shap?.features[0]?.feature || "underlying").replace(/_/g, " ")}</span> factor.
                    Consensus across model architectures is <span className="text-foreground font-semibold">{prediction && prediction.fusion_score > 0 ? "strongly positive" : "negative"}</span>.
                  </p>
                  <div className="mt-4 p-3 rounded-lg bg-surface border border-border">
                    <p className="text-[10px] text-muted-foreground uppercase tracking-widest font-bold mb-2">Signal Strength</p>
                    <div className="h-2 bg-muted rounded-full overflow-hidden">
                      <div 
                        className={`h-full rounded-full ${prediction && prediction.fusion_score > 0 ? "bg-bull" : "bg-bear"}`} 
                        style={{ width: `${prediction ? Math.min(Math.abs(prediction.fusion_score) * 20, 100) : 0}%` }}
                      />
                    </div>
                  </div>
                </div>
              </div>
            </>
          )}
        </div>
      ) : (
        <div className="bg-card rounded-xl border border-border p-12 flex flex-col items-center justify-center text-center fade-in-up stagger-2">
          <Info size={48} className="text-muted-foreground mb-4 opacity-20" />
          <h3 className="text-lg font-bold text-foreground mb-2 capitalize">{activeTab} Modules</h3>
          <p className="text-sm text-muted-foreground max-w-md">
            The {activeTab} engine is currently in the validation phase and will be integrated in the next platform update.
          </p>
          <div className="mt-6 px-4 py-2 rounded-lg bg-muted text-xs font-medium text-muted-foreground border border-border">
            Coming Soon · Phase 4 Milestone
          </div>
        </div>
      )}
    </div>
  );
};

export default SignalDetail;
