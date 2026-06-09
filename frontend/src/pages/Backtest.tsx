import { useTranslation } from "react-i18next";
import { useQuery } from "@tanstack/react-query";
import { AlertTriangle, FlaskConical, Target, TrendingUp } from "lucide-react";
import {
  ComposedChart,
  Area,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
} from "recharts";
import { fetchApi } from "../hooks/useApi";

interface ChampionProfile {
  strategy: string;
  horizon_days: number;
  net_ann_excess: number;
  gross_ann_excess: number;
  ir: number;
  sharpe: number;
  max_drawdown: number;
  avg_turnover_weekly: number;
  cost_drag_ann: number;
  n_periods: number;
  backtest_as_of: string | null;
  cost_model: Record<string, number> | string | null;
}

interface ConePoint {
  date: string;
  day_index: number;
  expected: number;
  upper_1s: number;
  lower_1s: number;
  upper_2s: number;
  lower_2s: number;
}

interface LiveExcessPoint {
  date: string;
  excess_cum_return: number;
  is_rebalance: boolean;
}

interface MetricComparison {
  backtest: number | null;
  live: number | null;
  unit: string | null;
  note: string | null;
}

interface BacktestVsLiveResponse {
  champion: ChampionProfile | null;
  expectation: { weekly_excess_mean: number; weekly_excess_std: number; source: string } | null;
  cone: ConePoint[];
  live: LiveExcessPoint[];
  comparison: Record<string, MetricComparison>;
}

const fmtPct = (v: number | null | undefined, digits = 2) =>
  v == null ? "—" : `${(v * 100).toFixed(digits)}%`;

const Backtest = () => {
  const { t } = useTranslation();

  const query = useQuery<BacktestVsLiveResponse>({
    queryKey: ["backtestVsLive"],
    queryFn: () => fetchApi<BacktestVsLiveResponse>("/api/backtest/vs-live"),
    staleTime: 5 * 60_000,
  });

  const data = query.data;
  const champion = data?.champion ?? null;

  // Merge cone + live onto one date grid for the chart
  const chartData = (data?.cone ?? []).map((c, i) => ({
    date: c.date,
    band2: [c.lower_2s, c.upper_2s] as [number, number],
    band1: [c.lower_1s, c.upper_1s] as [number, number],
    expected: c.expected,
    live: data?.live?.[i]?.excess_cum_return ?? null,
  }));

  const lastLive = data?.live?.length ? data.live[data.live.length - 1] : null;
  const lastCone = data?.cone?.length ? data.cone[data.cone.length - 1] : null;
  const liveInBand =
    lastLive != null && lastCone != null
      ? lastLive.excess_cum_return >= lastCone.lower_2s && lastLive.excess_cum_return <= lastCone.upper_2s
      : null;
  const liveColor = lastLive != null && lastLive.excess_cum_return >= 0 ? "#00C805" : "#FF5252";

  const comparison = data?.comparison ?? {};

  const statusChip = (kind: string, c: MetricComparison) => {
    if (c.backtest == null || c.live == null) {
      return { label: t("backtest.vsLive.statusPending"), cls: "bg-muted text-muted-foreground" };
    }
    if (kind === "turnover") {
      const ratio = c.backtest > 0 ? c.live / c.backtest : Infinity;
      if (ratio <= 2) return { label: t("backtest.vsLive.statusInBand"), cls: "bg-bull/10 text-bull" };
      if (ratio <= 5) return { label: t("backtest.vsLive.statusDeviating"), cls: "bg-amber-500/10 text-amber-500" };
      return { label: t("backtest.vsLive.statusSevere", { ratio: ratio.toFixed(0) }), cls: "bg-bear/10 text-bear" };
    }
    if (kind === "max_drawdown") {
      return Math.abs(c.live) <= Math.abs(c.backtest)
        ? { label: t("backtest.vsLive.statusInBand"), cls: "bg-bull/10 text-bull" }
        : { label: t("backtest.vsLive.statusDeviating"), cls: "bg-amber-500/10 text-amber-500" };
    }
    // weekly_excess: positive and >= half of backtest expectation = on track
    if (c.live >= c.backtest * 0.5) return { label: t("backtest.vsLive.statusInBand"), cls: "bg-bull/10 text-bull" };
    if (c.live >= 0) return { label: t("backtest.vsLive.statusDeviating"), cls: "bg-amber-500/10 text-amber-500" };
    return { label: t("backtest.vsLive.statusSevere", { ratio: "" }), cls: "bg-bear/10 text-bear" };
  };

  let costModelLabel = "";
  if (champion?.cost_model != null) {
    costModelLabel =
      typeof champion.cost_model === "string"
        ? champion.cost_model
        : `AC η=${champion.cost_model.eta} γ=${champion.cost_model.gamma}`;
  }

  const championStats = champion
    ? [
        { label: t("backtest.vsLive.netAnnExcess"), value: fmtPct(champion.net_ann_excess, 1), accent: champion.net_ann_excess >= 0 },
        { label: t("backtest.vsLive.grossAnnExcess"), value: fmtPct(champion.gross_ann_excess, 1), accent: null },
        { label: "IR", value: champion.ir.toFixed(2), accent: null },
        { label: "Sharpe", value: champion.sharpe.toFixed(2), accent: null },
        { label: t("backtest.vsLive.maxDrawdown"), value: fmtPct(-Math.abs(champion.max_drawdown), 1), accent: false },
        { label: t("backtest.vsLive.avgTurnover"), value: fmtPct(champion.avg_turnover_weekly, 2), accent: null },
        { label: t("backtest.vsLive.costDrag"), value: fmtPct(champion.cost_drag_ann, 2), accent: null },
        { label: t("backtest.vsLive.periods"), value: String(champion.n_periods), accent: null },
      ]
    : [];

  return (
    <div className="flex-1 overflow-y-auto p-6 space-y-5">
      {/* Header */}
      <div className="flex items-center justify-between fade-in-up">
        <div>
          <h2 className="text-xl font-bold text-foreground">{t("backtest.title")}</h2>
          <p className="text-xs text-muted-foreground mt-1">{t("backtest.vsLive.subtitle")}</p>
        </div>
        <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-muted border border-border text-xs text-muted-foreground">
          <AlertTriangle size={13} className="text-yellow-500" />
          {t("backtest.disclaimer")}
        </div>
      </div>

      {query.isLoading ? (
        <div className="flex flex-col items-center justify-center min-h-[400px] text-muted-foreground">
          <FlaskConical size={32} className="opacity-20 animate-pulse mb-3" />
          <p className="text-xs">{t("common.loading")}</p>
        </div>
      ) : query.isError || !data ? (
        <div className="flex flex-col items-center justify-center min-h-[400px] text-muted-foreground">
          <AlertTriangle size={32} className="opacity-30 mb-3" />
          <p className="text-xs">{t("backtest.vsLive.loadError")}</p>
        </div>
      ) : (
        <>
          {/* Section 1: Champion archive */}
          <div className="bg-card rounded-xl border border-border p-5 fade-in-up stagger-1">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-2">
                <FlaskConical size={15} className="text-primary" />
                <h3 className="text-sm font-bold text-foreground">{t("backtest.vsLive.championTitle")}</h3>
              </div>
              {champion && (
                <div className="text-[10px] text-muted-foreground font-mono">
                  {champion.strategy} · {champion.horizon_days}D · {costModelLabel}
                  {champion.backtest_as_of ? ` · as of ${champion.backtest_as_of}` : ""}
                </div>
              )}
            </div>
            {champion ? (
              <div className="grid grid-cols-8 gap-3">
                {championStats.map((s) => (
                  <div key={s.label} className="bg-muted/20 rounded-lg p-3 border border-border/50">
                    <p className="text-[10px] text-muted-foreground uppercase tracking-wider mb-1 truncate">{s.label}</p>
                    <p className={`text-base font-black font-mono ${
                      s.accent == null ? "text-foreground" : s.accent ? "text-bull" : "text-bear"
                    }`}>
                      {s.value}
                    </p>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-xs text-muted-foreground py-6 text-center">{t("backtest.vsLive.noChampion")}</p>
            )}
          </div>

          {/* Section 2: Expectation cone vs live */}
          <div className="bg-card rounded-xl border border-border p-5 fade-in-up stagger-2">
            <div className="flex items-center justify-between mb-1">
              <div className="flex items-center gap-2">
                <Target size={15} className="text-primary" />
                <h3 className="text-sm font-bold text-foreground">{t("backtest.vsLive.coneTitle")}</h3>
              </div>
              {liveInBand != null && (
                <span className={`text-[10px] font-bold uppercase tracking-wider px-2 py-1 rounded ${
                  liveInBand ? "bg-bull/10 text-bull" : "bg-bear/10 text-bear"
                }`}>
                  {liveInBand ? t("backtest.vsLive.inBandChip") : t("backtest.vsLive.outOfBandChip")}
                </span>
              )}
            </div>
            <p className="text-[10px] text-muted-foreground mb-3">{t("backtest.vsLive.coneSubtitle")}</p>
            {chartData.length >= 2 ? (
              <ResponsiveContainer width="100%" height={300}>
                <ComposedChart data={chartData} margin={{ left: 0, right: 6, top: 4, bottom: 0 }}>
                  <XAxis dataKey="date" tick={{ fontSize: 9, fill: "var(--muted-foreground)" }} interval="preserveStartEnd" minTickGap={48} />
                  <YAxis tick={{ fontSize: 9, fill: "var(--muted-foreground)" }} tickFormatter={(v: number) => `${(v * 100).toFixed(1)}%`} domain={["auto", "auto"]} width={46} />
                  <Tooltip
                    contentStyle={{ background: "var(--popover)", border: "1px solid var(--border)", borderRadius: 8, fontSize: 11 }}
                    formatter={(value: number | [number, number], name: string) => {
                      if (Array.isArray(value)) {
                        const label = name === "band2" ? t("backtest.vsLive.band2") : t("backtest.vsLive.band1");
                        return [`${(value[0] * 100).toFixed(2)}% ~ ${(value[1] * 100).toFixed(2)}%`, label];
                      }
                      const label = name === "expected" ? t("backtest.vsLive.expected") : name === "live" ? t("backtest.vsLive.liveExcess") : name;
                      return [`${(value * 100).toFixed(2)}%`, label];
                    }}
                  />
                  <ReferenceLine y={0} stroke="var(--border)" strokeDasharray="2 2" />
                  <Area dataKey="band2" stroke="none" fill="#8884d8" fillOpacity={0.08} isAnimationActive={false} />
                  <Area dataKey="band1" stroke="none" fill="#8884d8" fillOpacity={0.15} isAnimationActive={false} />
                  <Line dataKey="expected" stroke="#888888" strokeWidth={1.4} strokeDasharray="5 4" dot={false} isAnimationActive={false} />
                  <Line dataKey="live" stroke={liveColor} strokeWidth={2.4} dot={false} isAnimationActive={false} connectNulls />
                </ComposedChart>
              </ResponsiveContainer>
            ) : (
              <div className="flex items-center justify-center h-[300px] text-xs text-muted-foreground border border-dashed border-border rounded-lg">
                {t("backtest.vsLive.noConeData")}
              </div>
            )}
            <div className="flex justify-between items-center text-[9px] text-muted-foreground mt-2 px-1">
              <span className="uppercase tracking-wider font-bold">{t("backtest.vsLive.coneLegend")}</span>
              <span>
                {data.live.length} {t("backtest.vsLive.tradingDays")}
                {data.expectation && ` · μ=${fmtPct(data.expectation.weekly_excess_mean)}/wk σ=${fmtPct(data.expectation.weekly_excess_std)}/wk`}
              </span>
            </div>
          </div>

          {/* Section 3: Metric comparison */}
          <div className="bg-card rounded-xl border border-border p-5 fade-in-up stagger-3">
            <div className="flex items-center gap-2 mb-4">
              <TrendingUp size={15} className="text-primary" />
              <h3 className="text-sm font-bold text-foreground">{t("backtest.vsLive.comparisonTitle")}</h3>
            </div>
            <div className="flex items-center px-3 py-2 border-b border-border bg-muted/10 text-[10px] font-black text-muted-foreground uppercase tracking-[0.15em]">
              <div className="flex-1">{t("backtest.vsLive.metric")}</div>
              <div className="w-32 text-right">{t("backtest.vsLive.backtestCol")}</div>
              <div className="w-32 text-right">{t("backtest.vsLive.liveCol")}</div>
              <div className="w-36 text-right">{t("backtest.vsLive.statusCol")}</div>
            </div>
            {(["weekly_excess", "turnover", "max_drawdown", "ic"] as const).map((key) => {
              const c = comparison[key];
              if (!c) return null;
              const chip = statusChip(key, c);
              return (
                <div key={key} className="flex items-center px-3 py-3 border-b border-border/50 last:border-0 text-xs">
                  <div className="flex-1">
                    <span className="text-foreground font-semibold">{t(`backtest.vsLive.metrics.${key}`)}</span>
                    {c.note && <span className="ml-2 text-[10px] text-muted-foreground">{c.note}</span>}
                  </div>
                  <div className="w-32 text-right font-mono text-muted-foreground">{fmtPct(c.backtest)}</div>
                  <div className="w-32 text-right font-mono font-bold text-foreground">{fmtPct(c.live)}</div>
                  <div className="w-36 text-right">
                    <span className={`text-[9px] font-bold uppercase tracking-wider px-1.5 py-0.5 rounded ${chip.cls}`}>{chip.label}</span>
                  </div>
                </div>
              );
            })}
          </div>
        </>
      )}
    </div>
  );
};

export default Backtest;
