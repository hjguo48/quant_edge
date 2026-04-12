import { useState } from "react";
import { Info, Layers } from "lucide-react";
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

const recentSignals = [
  { ticker: "NVDA", direction: "long", confidence: 91, alpha: 3.2, time: "5m ago" },
  { ticker: "TSLA", direction: "short", confidence: 74, alpha: -1.8, time: "12m ago" },
  { ticker: "MSFT", direction: "long", confidence: 83, alpha: 2.1, time: "18m ago" },
  { ticker: "META", direction: "long", confidence: 67, alpha: 1.4, time: "31m ago" },
  { ticker: "AMZN", direction: "short", confidence: 58, alpha: -0.9, time: "45m ago" },
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

const CustomTooltip = ({ active, payload }: { active?: boolean; payload?: any[] }) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-popover border border-border rounded-lg px-3 py-2 shadow-custom">
      <p className="text-xs text-muted-foreground">Price</p>
      <p className="text-sm font-bold text-bull font-mono">
        ${payload[0].value.toLocaleString("en-US", { maximumFractionDigits: 2 })}
      </p>
    </div>
  );
};

interface DashboardProps {
  onSelectSignal?: (ticker: string) => void;
}

const DASHBOARD_RANGES = ["7", "30", "90", "365"] as const;
type DashboardRange = (typeof DASHBOARD_RANGES)[number];

const RANGE_LABELS: Record<DashboardRange, string> = {
  "7": "7D",
  "30": "30D",
  "90": "90D",
  "365": "1Y",
};

const Dashboard = ({ onSelectSignal = () => {} }: DashboardProps) => {
  const [timeRange, setTimeRange] = useState<DashboardRange>("30");

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
    queryKey: ["marketIndices", timeRange],
    queryFn: () => fetchApi<MarketIndicesResponse>(`/api/market/indices?days=${timeRange}`),
    retry: false,
    placeholderData: (previousData) => previousData,
  });

  const {
    data: sectorsData,
    isLoading: isSectorsLoading,
    isError: isSectorsError,
  } = useQuery<MarketSectorsResponse>({
    queryKey: ["marketSectors", 1],
    queryFn: () => fetchApi<MarketSectorsResponse>("/api/market/sectors?days=1"),
    retry: false,
  });

  const spyPrice = overview?.spy?.price || 0;
  const spyChangePct = overview?.spy?.change_pct || 0;
  const vixValue = overview?.vix?.value || 0;
  const breadth = overview?.breadth || { advancing: 0, declining: 0, advance_pct: 0 };
  const sectors = sectorsData?.sectors || overview?.sectors || [];

  const chartData =
    indices?.prices?.map((price) => ({
      day: new Date(price.trade_date).toLocaleDateString("en-US", {
        month: "short",
        day: "numeric",
      }),
      pnl: price.adj_close || price.close,
    })) || [];

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
        <p className="text-xs text-muted-foreground">
          <span className="font-semibold text-foreground">Model Output Only</span>
          {` — All signals are generated by quantitative models and do not constitute investment advice. Past model performance does not guarantee future results. For institutional use only.`}
        </p>
      </div>

      {hasError && (
        <div className="flex items-center gap-2 px-4 py-2.5 rounded-lg bg-accent border border-border">
          <Info size={14} className="text-bear flex-shrink-0" />
          <p className="text-xs text-muted-foreground">
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
              <p className="text-xs text-muted-foreground mt-0.5">
                Model output · {RANGE_LABELS[timeRange]} window
                {indices?.end_date ? ` · through ${indices.end_date}` : ""}
              </p>
            </div>
            <div className="flex items-center gap-1 bg-accent/50 p-1 rounded-lg">
              {DASHBOARD_RANGES.map((r) => (
                <button
                  key={r}
                  onClick={() => setTimeRange(r)}
                  className={`px-3 py-1 text-[10px] font-bold rounded-md transition-all ${
                    timeRange === r ? "bg-card text-foreground shadow-sm" : "text-muted-foreground hover:text-foreground"
                  }`}
                >
                  {RANGE_LABELS[r]}
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
                <XAxis dataKey="day" tick={{ fill: "#607B96", fontSize: 10 }} axisLine={false} tickLine={false} interval="preserveStartEnd" />
                <YAxis hide domain={["auto", "auto"]} />
                <Tooltip content={<CustomTooltip />} />
                <Area type="monotone" dataKey="pnl" stroke="#00C805" strokeWidth={2} fill="url(#pnlGrad)" />
              </AreaChart>
            </ResponsiveContainer>
          ) : (
            <div className="flex h-[180px] items-center justify-center rounded-lg border border-dashed border-border bg-surface text-sm text-muted-foreground">
              No index history available.
            </div>
          )}
        </div>

        {/* Factor IC */}
        <div className="w-72 bg-card rounded-xl border border-border p-5 fade-in-up stagger-3 flex-shrink-0">
          <h3 className="text-sm font-semibold text-foreground mb-1">Factor IC Scores</h3>
          <p className="text-xs text-muted-foreground mb-4">Information coefficient, 20-day</p>
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

      {/* KLine + Recent Signals */}
      <div className="flex flex-col gap-5 xl:flex-row">
        <div className="flex-1 min-w-0">
          <KLineChart ticker="NVDA" height={240} defaultRange="1M" />
        </div>
        <div className="w-72 bg-card rounded-xl border border-border flex-shrink-0 fade-in-up stagger-4">
          <div className="flex items-center justify-between px-5 py-4 border-b border-border">
            <h3 className="text-sm font-semibold text-foreground">Top Signals</h3>
            <Layers size={14} className="text-muted-foreground" />
          </div>
          <div className="divide-y divide-border">
            {recentSignals.map((s) => (
              <button
                key={s.ticker}
                onClick={() => onSelectSignal(s.ticker)}
                className="flex w-full items-center justify-between px-5 py-3 hover:bg-accent/40 transition-colors cursor-pointer text-left"
              >
                <div>
                  <div className="text-sm font-bold text-foreground">{s.ticker}</div>
                  <span className={`text-xs font-semibold px-1.5 py-0.5 rounded-sm ${s.direction === "long" ? "tag-bull" : "tag-bear"}`}>
                    {s.direction === "long" ? "LONG" : "SHORT"}
                  </span>
                </div>
                <div className="text-right">
                  <div className="text-xs text-muted-foreground">{s.confidence}% conf</div>
                  <div className={`text-sm font-bold font-mono ${s.alpha > 0 ? "text-bull" : "text-bear"}`}>
                    {s.alpha > 0 ? "+" : ""}{s.alpha.toFixed(2)}%
                  </div>
                </div>
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Heatmap */}
      <div className="fade-in-up stagger-5">
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
    </div>
  );
};

export default Dashboard;
