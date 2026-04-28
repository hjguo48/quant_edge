import { useState } from "react";
import { Play, Settings2, BarChart2, AlertTriangle } from "lucide-react";
import { useTranslation } from "react-i18next";
import StatCard from "../components/StatCard";
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, BarChart, Bar, ReferenceLine, Cell } from "recharts";

function generateEquityCurve(n: number) {
  let equity = 1;
  let benchmark = 1;
  return Array.from({ length: n }, (_, i) => {
    const ret = (Math.random() - 0.46) * 0.025;
    const bret = (Math.random() - 0.48) * 0.015;
    equity = Math.max(0.3, equity * (1 + ret));
    benchmark = Math.max(0.3, benchmark * (1 + bret));
    return {
      day: i + 1,
      equity: parseFloat(equity.toFixed(4)),
      benchmark: parseFloat(benchmark.toFixed(4)),
      drawdown: parseFloat((Math.min(0, ret * 4) * 100).toFixed(2)),
    };
  });
}

function generateMonthlyReturns() {
  const months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
  return months.map((month) => ({
    month,
    ret: parseFloat(((Math.random() - 0.4) * 8).toFixed(2)),
  }));
}

const equityData = generateEquityCurve(252);
const monthlyData = generateMonthlyReturns();

const UNIVERSE = ["S&P 500", "NASDAQ 100", "Russell 2000", "Custom List"];
const FACTORS = ["Momentum", "Value + Momentum", "Quality", "Multi-Factor", "Custom"];

const Backtest = () => {
  const { t } = useTranslation();
  const [running, setRunning] = useState(false);
  const [, setRan] = useState(true);
  const [universe, setUniverse] = useState("S&P 500");
  const [factor, setFactor] = useState("Multi-Factor");
  const [startYear, setStartYear] = useState("2020");
  const [endYear, setEndYear] = useState("2024");
  const [rebalance, setRebalance] = useState("Monthly");

  const handleRun = () => {
    setRunning(true);
    setTimeout(() => { setRunning(false); setRan(true); }, 2200);
  };

  const finalEquity = equityData[equityData.length - 1].equity;
  const totalReturn = ((finalEquity - 1) * 100).toFixed(2);
  const isPositive = parseFloat(totalReturn) >= 0;

  return (
    <div className="flex-1 overflow-y-auto p-6 space-y-5">
      {/* Header */}
      <div className="flex items-center justify-between fade-in-up">
        <div>
          <h2 className="text-xl font-bold text-foreground">{t("backtest.title")}</h2>
          <p className="text-xs text-muted-foreground mt-1">
            {t("backtest.subtitle")}
          </p>
        </div>
        <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-muted border border-border text-xs text-muted-foreground">
          <AlertTriangle size={13} className="text-yellow-500" />
          {t("backtest.disclaimer")}
        </div>
      </div>

      <div className="flex gap-5">
        {/* Config Panel */}
        <div className="w-64 flex-shrink-0 space-y-4 fade-in-up stagger-1">
          <div className="bg-card rounded-xl border border-border p-5 space-y-4">
            <div className="flex items-center gap-2 pb-2 border-b border-border">
              <Settings2 size={14} className="text-primary" />
              <h3 className="text-sm font-semibold text-foreground">{t("backtest.configuration")}</h3>
            </div>

            {[
              { key: "universe", label: t("backtest.fields.universe"), value: universe, setValue: setUniverse, options: UNIVERSE },
              { key: "strategy", label: t("backtest.fields.strategy"), value: factor, setValue: setFactor, options: FACTORS },
              { key: "startYear", label: t("backtest.fields.startYear"), value: startYear, setValue: setStartYear, options: ["2015","2016","2017","2018","2019","2020","2021"] },
              { key: "endYear", label: t("backtest.fields.endYear"), value: endYear, setValue: setEndYear, options: ["2022","2023","2024","2025"] },
              { key: "rebalance", label: t("backtest.fields.rebalance"), value: rebalance, setValue: setRebalance, options: ["Daily","Weekly","Monthly","Quarterly"] },
            ].map(({ key, label, value, setValue, options }) => (
              <div key={key}>
                <label className="text-xs text-muted-foreground mb-1.5 block">{label}</label>
                <select
                  value={value}
                  onChange={(e) => setValue(e.target.value)}
                  className="w-full bg-muted text-sm text-foreground px-3 py-2 rounded-lg border border-transparent outline-none cursor-pointer hover:bg-accent transition-colors"
                >
                  {options.map((o) => <option key={o} value={o} className="bg-popover">{o}</option>)}
                </select>
              </div>
            ))}

            <div>
              <label className="text-xs text-muted-foreground mb-1.5 block">{t("backtest.fields.maxPositionSize")}</label>
              <div className="flex items-center gap-2">
                <input type="range" min={1} max={20} defaultValue={5} className="flex-1 accent-primary" />
                <span className="text-xs font-mono text-foreground w-6">5%</span>
              </div>
            </div>

            <div>
              <label className="text-xs text-muted-foreground mb-1.5 block">{t("backtest.fields.stopLoss")}</label>
              <div className="flex items-center gap-2">
                <input type="range" min={1} max={15} defaultValue={8} className="flex-1 accent-primary" />
                <span className="text-xs font-mono text-bear w-7">-8%</span>
              </div>
            </div>

            <button
              onClick={handleRun}
              disabled={running}
              className={`w-full flex items-center justify-center gap-2 py-2.5 rounded-xl text-sm font-bold transition-all duration-300 ${
                running
                  ? "bg-primary/50 text-primary-foreground cursor-not-allowed"
                  : "btn-primary text-primary-foreground"
              }`}
            >
              {running ? (
                <>
                  <div className="w-4 h-4 border-2 border-primary-foreground/30 border-t-primary-foreground rounded-full animate-spin" />
                  {t("backtest.running")}
                </>
              ) : (
                <>
                  <Play size={14} fill="currentColor" />
                  {t("backtest.runButton")}
                </>
              )}
            </button>
          </div>
        </div>

        {/* Results */}
        <div className="flex-1 space-y-5">
          {/* Stats */}
          <div className="flex gap-4 fade-in-up stagger-1">
            {[
              { label: t("backtest.results.totalReturn"), value: `${isPositive ? "+" : ""}${totalReturn}%`, change: parseFloat(totalReturn), changeLabel: t("backtest.results.vsBenchmark", { value: (parseFloat(totalReturn) * 0.6).toFixed(1) }), trend: isPositive ? "up" as const : "down" as const },
              { label: t("backtest.results.sharpe"), value: "1.87", change: 0.12, changeLabel: t("backtest.results.annualized"), trend: "up" as const },
              { label: t("backtest.results.maxDrawdown"), value: "-12.4%", change: -1.2, changeLabel: t("backtest.results.fromPeak"), trend: "down" as const },
              { label: t("backtest.results.calmar"), value: "1.42", change: 0.08, changeLabel: t("backtest.results.returnPerDrawdown"), trend: "up" as const },
            ].map((s, i) => (
              <div key={s.label} className="flex-1">
                <StatCard {...s} delay={i * 60} />
              </div>
            ))}
          </div>

          {/* Equity Curve */}
          <div className="bg-card rounded-xl border border-border p-5 fade-in-up stagger-2">
            <div className="flex items-center justify-between mb-4">
              <div>
                <h3 className="text-sm font-semibold text-foreground">{t("backtest.equityCurve.title")}</h3>
                <p className="text-xs text-muted-foreground mt-0.5">{t("backtest.equityCurve.subtitle", { start: startYear, end: endYear })}</p>
              </div>
              <div className="flex items-center gap-3">
                <div className="flex items-center gap-1.5"><div className="w-2.5 h-0.5 bg-primary" /><span className="text-xs text-muted-foreground">{t("backtest.equityCurve.strategy")}</span></div>
                <div className="flex items-center gap-1.5"><div className="w-2.5 h-0.5" style={{ backgroundColor: "#607B96" }} /><span className="text-xs text-muted-foreground">{t("backtest.equityCurve.benchmark")}</span></div>
              </div>
            </div>
            <ResponsiveContainer width="100%" height={200}>
              <AreaChart data={equityData} margin={{ top: 5, right: 0, bottom: 0, left: 0 }}>
                <defs>
                  <linearGradient id="eqGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#00C805" stopOpacity={0.18} />
                    <stop offset="95%" stopColor="#00C805" stopOpacity={0} />
                  </linearGradient>
                  <linearGradient id="bmGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#607B96" stopOpacity={0.1} />
                    <stop offset="95%" stopColor="#607B96" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <XAxis dataKey="day" tick={{ fill: "#607B96", fontSize: 10 }} axisLine={false} tickLine={false} interval={40} />
                <YAxis tick={{ fill: "#607B96", fontSize: 10 }} axisLine={false} tickLine={false} />
                <Tooltip
                  cursor={{ stroke: "rgba(255,255,255,0.1)" }}
                  content={({ active, payload }: { active?: boolean; payload?: { name: string; value: number }[] }) => {
                    if (!active || !payload?.length) return null;
                    return (
                      <div className="bg-popover border border-border rounded-lg px-3 py-2 shadow-custom space-y-1">
                        {payload.map((p) => (
                          <div key={p.name} className="flex gap-2">
                            <span className="text-xs text-muted-foreground capitalize">{p.name}:</span>
                            <span className={`text-xs font-bold ${p.name === "equity" ? "text-bull" : "text-muted-foreground"}`}>
                              {(p.value).toFixed(3)}x
                            </span>
                          </div>
                        ))}
                      </div>
                    );
                  }}
                />
                <Area type="monotone" dataKey="benchmark" stroke="#607B96" strokeWidth={1.5} fill="url(#bmGrad)" strokeDasharray="4 2" />
                <Area type="monotone" dataKey="equity" stroke="#00C805" strokeWidth={2} fill="url(#eqGrad)" />
              </AreaChart>
            </ResponsiveContainer>
          </div>

          {/* Monthly Returns */}
          <div className="flex gap-5">
            <div className="flex-1 bg-card rounded-xl border border-border p-5 fade-in-up stagger-3">
              <h3 className="text-sm font-semibold text-foreground mb-1">{t("backtest.monthlyReturns.title")}</h3>
              <p className="text-xs text-muted-foreground mb-4">{t("backtest.monthlyReturns.subtitle")}</p>
              <ResponsiveContainer width="100%" height={150}>
                <BarChart data={monthlyData} margin={{ top: 5, right: 0, bottom: 0, left: 0 }}>
                  <XAxis dataKey="month" tick={{ fill: "#607B96", fontSize: 10 }} axisLine={false} tickLine={false} />
                  <YAxis tick={{ fill: "#607B96", fontSize: 10 }} axisLine={false} tickLine={false} />
                  <ReferenceLine y={0} stroke="rgba(255,255,255,0.1)" />
                  <Tooltip
                    cursor={{ fill: "rgba(255,255,255,0.03)" }}
                    content={({ active, payload }: { active?: boolean; payload?: { value: number }[] }) => {
                      if (!active || !payload?.length) return null;
                      const v = payload[0].value;
                      return (
                        <div className="bg-popover border border-border rounded-lg px-2.5 py-1.5 shadow-custom">
                          <p className={`text-xs font-bold ${v >= 0 ? "text-bull" : "text-bear"}`}>{v >= 0 ? "+" : ""}{v.toFixed(2)}%</p>
                        </div>
                      );
                    }}
                  />
                  <Bar dataKey="ret" radius={[3, 3, 0, 0]}>
                    {monthlyData.map((entry, i) => (
                      <Cell key={i} fill={entry.ret >= 0 ? "#00C805" : "#FF5252"} fillOpacity={0.8} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
            {/* Risk Metrics */}
            <div className="w-64 bg-card rounded-xl border border-border p-5 flex-shrink-0 fade-in-up stagger-4">
              <div className="flex items-center gap-2 mb-4">
                <BarChart2 size={14} className="text-primary" />
                <h3 className="text-sm font-semibold text-foreground">{t("backtest.riskMetrics.title")}</h3>
              </div>
              <div className="space-y-3">
                {[
                  { key: "annualizedVol", label: t("backtest.riskMetrics.annualizedVol"), value: "14.2%", color: "text-foreground" },
                  { key: "var95", label: t("backtest.riskMetrics.var95"), value: "-1.8%", color: "text-bear" },
                  { key: "cvar95", label: t("backtest.riskMetrics.cvar95"), value: "-2.4%", color: "text-bear" },
                  { key: "winRate", label: t("backtest.riskMetrics.winRate"), value: "58.3%", color: "text-bull" },
                  { key: "profitFactor", label: t("backtest.riskMetrics.profitFactor"), value: "1.74", color: "text-bull" },
                  { key: "avgWinLoss", label: t("backtest.riskMetrics.avgWinLoss"), value: "1.82", color: "text-bull" },
                  { key: "betaSpx", label: t("backtest.riskMetrics.betaSpx"), value: "0.72", color: "text-foreground" },
                  { key: "turnoverMonthly", label: t("backtest.riskMetrics.turnoverMonthly"), value: "18.4%", color: "text-muted-foreground" },
                ].map(({ key, label, value, color }) => (
                  <div key={key} className="flex items-center justify-between">
                    <span className="text-xs text-muted-foreground">{label}</span>
                    <span className={`text-xs font-bold font-mono ${color}`}>{value}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Backtest;
