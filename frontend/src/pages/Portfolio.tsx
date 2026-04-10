import { useState } from "react";
import { TrendingUp, TrendingDown, PieChart, DollarSign } from "lucide-react";
import StatCard from "../components/StatCard";
import MiniSparkline from "../components/MiniSparkline";
import { PieChart as RePieChart, Pie, Cell, Tooltip, ResponsiveContainer, AreaChart, Area, XAxis, YAxis } from "recharts";

const holdings = [
  { ticker: "NVDA", name: "NVIDIA Corp.", weight: 18.4, value: 524120, pnl: 87230, pnlPct: 19.96, signal: "long", sparkData: [18, 22, 20, 28, 25, 32, 30, 38, 35, 42] },
  { ticker: "AAPL", name: "Apple Inc.", weight: 15.2, value: 432600, pnl: 32400, pnlPct: 8.09, signal: "long", sparkData: [10, 12, 11, 15, 14, 18, 16, 20, 19, 24] },
  { ticker: "MSFT", name: "Microsoft Corp.", weight: 13.7, value: 389740, pnl: 41200, pnlPct: 11.81, signal: "long", sparkData: [15, 17, 16, 20, 19, 22, 21, 25, 23, 27] },
  { ticker: "GOOGL", name: "Alphabet Inc.", weight: 11.1, value: 315890, pnl: 28900, pnlPct: 10.07, signal: "long", sparkData: [8, 11, 10, 13, 12, 16, 14, 18, 16, 20] },
  { ticker: "TSLA", name: "Tesla Inc.", weight: -8.3, value: 236240, pnl: -18400, pnlPct: -7.22, signal: "short", sparkData: [32, 28, 30, 25, 27, 22, 24, 19, 21, 16] },
  { ticker: "JPM", name: "JPMorgan Chase", weight: 7.6, value: 216290, pnl: 12800, pnlPct: 6.29, signal: "long", sparkData: [12, 14, 13, 16, 15, 18, 17, 20, 19, 22] },
  { ticker: "UNH", name: "UnitedHealth Group", weight: 6.8, value: 193480, pnl: 22100, pnlPct: 12.90, signal: "long", sparkData: [14, 16, 15, 19, 17, 22, 20, 24, 22, 27] },
  { ticker: "XOM", name: "Exxon Mobil", weight: -4.2, value: 119520, pnl: -8700, pnlPct: -6.79, signal: "short", sparkData: [28, 25, 26, 22, 24, 20, 22, 18, 20, 16] },
];

const sectorAlloc = [
  { name: "Technology", value: 58 },
  { name: "Finance", value: 14 },
  { name: "Healthcare", value: 7 },
  { name: "Energy", value: 8 },
  { name: "Consumer", value: 8 },
  { name: "Cash", value: 5 },
];
const PIE_COLORS = ["#00C805", "#3B82F6", "#8B5CF6", "#F59E0B", "#FF5252", "#607B96"];

const perfData = Array.from({ length: 90 }, (_, i) => {
  const base = 2400000 + i * 4800;
  return {
    day: i + 1,
    portfolio: base + Math.sin(i * 0.4) * 40000 + Math.random() * 20000,
    benchmark: 2400000 + i * 2800 + Math.sin(i * 0.3) * 20000,
  };
});

const Portfolio = () => {
  const [tab, setTab] = useState("holdings");

  return (
    <div className="flex-1 overflow-y-auto p-6 space-y-5">
      {/* Stats */}
      <div className="flex gap-4 fade-in-up">
        {[
          { label: "Total Portfolio Value", value: "$2847320", change: 3.21, changeLabel: "MTD return", trend: "up" as const },
          { label: "Realized P&L (YTD)", value: "$198420", change: 8.12, changeLabel: "vs. prior year", trend: "up" as const },
          { label: "Unrealized P&L", value: "$247330", change: 2.41, changeLabel: "open positions", trend: "up" as const },
          { label: "Portfolio Beta", value: "0.82", change: -0.05, changeLabel: "vs. SPX", trend: "up" as const },
        ].map((s, i) => (
          <div key={s.label} className="flex-1">
            <StatCard {...s} delay={i * 60} />
          </div>
        ))}
      </div>

      {/* Main Area */}
      <div className="flex gap-5">
        {/* Performance Chart */}
        <div className="flex-1 bg-card rounded-xl border border-border p-5 fade-in-up stagger-2">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h3 className="text-sm font-semibold text-foreground">Portfolio vs. Benchmark</h3>
              <p className="text-xs text-muted-foreground mt-0.5">Model portfolio · SPX benchmark · 90d</p>
            </div>
            <div className="flex items-center gap-3">
              <div className="flex items-center gap-1.5">
                <div className="w-2.5 h-2.5 rounded-full bg-primary" />
                <span className="text-xs text-muted-foreground">Portfolio</span>
              </div>
              <div className="flex items-center gap-1.5">
                <div className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: "#607B96" }} />
                <span className="text-xs text-muted-foreground">Benchmark</span>
              </div>
            </div>
          </div>
          <ResponsiveContainer width="100%" height={200}>
            <AreaChart data={perfData} margin={{ top: 5, right: 0, bottom: 0, left: 0 }}>
              <defs>
                <linearGradient id="portGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#00C805" stopOpacity={0.18} />
                  <stop offset="95%" stopColor="#00C805" stopOpacity={0} />
                </linearGradient>
                <linearGradient id="benchGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#607B96" stopOpacity={0.12} />
                  <stop offset="95%" stopColor="#607B96" stopOpacity={0} />
                </linearGradient>
              </defs>
              <XAxis dataKey="day" tick={{ fill: "#607B96", fontSize: 10 }} axisLine={false} tickLine={false} interval={14} />
              <YAxis hide />
              <Tooltip
                cursor={{ stroke: "rgba(255,255,255,0.1)", strokeWidth: 1 }}
                content={({ active, payload }: { active?: boolean; payload?: { name: string; value: number }[] }) => {
                  if (!active || !payload?.length) return null;
                  return (
                    <div className="bg-popover border border-border rounded-lg px-3 py-2 shadow-custom space-y-1">
                      {payload.map((p) => (
                        <div key={p.name} className="flex items-center gap-2">
                          <div className="w-2 h-2 rounded-full" style={{ backgroundColor: p.name === "portfolio" ? "#00C805" : "#607B96" }} />
                          <span className="text-xs text-muted-foreground capitalize">{p.name}:</span>
                          <span className="text-xs font-bold text-foreground">${(p.value / 1000000).toFixed(2)}M</span>
                        </div>
                      ))}
                    </div>
                  );
                }}
              />
              <Area type="monotone" dataKey="benchmark" stroke="#607B96" strokeWidth={1.5} fill="url(#benchGrad)" strokeDasharray="4 2" />
              <Area type="monotone" dataKey="portfolio" stroke="#00C805" strokeWidth={2} fill="url(#portGrad)" />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        {/* Pie */}
        <div className="w-72 bg-card rounded-xl border border-border p-5 flex-shrink-0 fade-in-up stagger-3">
          <div className="flex items-center gap-2 mb-2">
            <PieChart size={14} className="text-primary" />
            <h3 className="text-sm font-semibold text-foreground">Sector Allocation</h3>
          </div>
          <ResponsiveContainer width="100%" height={160}>
            <RePieChart>
              <Pie data={sectorAlloc} dataKey="value" cx="50%" cy="50%" outerRadius={65} innerRadius={35} strokeWidth={0}>
                {sectorAlloc.map((_, i) => (
                  <Cell key={i} fill={PIE_COLORS[i % PIE_COLORS.length]} fillOpacity={0.85} />
                ))}
              </Pie>
              <Tooltip
                content={({ active, payload }: { active?: boolean; payload?: { name: string; value: number }[] }) => {
                  if (!active || !payload?.length) return null;
                  return (
                    <div className="bg-popover border border-border rounded-lg px-2.5 py-1.5 shadow-custom">
                      <p className="text-xs font-bold text-foreground">{payload[0].name}</p>
                      <p className="text-xs text-primary">{payload[0].value}%</p>
                    </div>
                  );
                }}
              />
            </RePieChart>
          </ResponsiveContainer>
          <div className="space-y-1.5 mt-2">
            {sectorAlloc.map((s, i) => (
              <div key={s.name} className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 rounded-full flex-shrink-0" style={{ backgroundColor: PIE_COLORS[i] }} />
                  <span className="text-xs text-muted-foreground">{s.name}</span>
                </div>
                <span className="text-xs font-semibold text-foreground">{s.value}%</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Holdings Table */}
      <div className="bg-card rounded-xl border border-border overflow-hidden fade-in-up stagger-4">
        <div className="flex items-center justify-between px-5 py-4 border-b border-border">
          <div className="flex items-center gap-2">
            <DollarSign size={14} className="text-primary" />
            <h3 className="text-sm font-semibold text-foreground">Holdings</h3>
          </div>
          <div className="flex gap-1 bg-muted rounded-lg p-0.5">
            {["holdings", "trades", "risk"].map((t) => (
              <button key={t} onClick={() => setTab(t)} className={`px-3 py-1.5 rounded-md text-xs font-medium capitalize transition-all duration-200 ${tab === t ? "bg-card text-foreground" : "text-muted-foreground hover:text-foreground"}`}>
                {t}
              </button>
            ))}
          </div>
        </div>
        <div>
          {/* Col Headers */}
          <div className="flex items-center px-5 py-2.5 border-b border-border bg-muted/20 text-xs font-semibold text-muted-foreground uppercase tracking-wider">
            <div className="w-28">Ticker</div>
            <div className="flex-1">Weight</div>
            <div className="w-28 text-right">Market Value</div>
            <div className="w-24 text-right">P&L</div>
            <div className="w-24 text-right">Return</div>
            <div className="w-24 text-center">Signal</div>
            <div className="w-20 text-center">Trend</div>
          </div>
          {holdings.map((h, i) => {
            const isLong = h.signal === "long";
            const isPos = h.pnl >= 0;
            return (
              <div
                key={h.ticker}
                className="flex items-center px-5 py-3.5 border-b border-border last:border-0 hover:bg-accent/40 transition-colors cursor-pointer fade-in-up"
                style={{ animationDelay: `${i * 40}ms` }}
              >
                <div className="w-28">
                  <div className="text-sm font-bold text-foreground">{h.ticker}</div>
                  <div className="text-xs text-muted-foreground truncate max-w-24">{h.name}</div>
                </div>
                <div className="flex-1">
                  <div className="flex items-center gap-2">
                    <div className="w-20 h-1.5 bg-muted rounded-full overflow-hidden">
                      <div
                        className="h-full rounded-full"
                        style={{
                          width: `${Math.min(100, Math.abs(h.weight) * 4)}%`,
                          backgroundColor: isLong ? "#00C805" : "#FF5252",
                        }}
                      />
                    </div>
                    <span className={`text-xs font-semibold ${isLong ? "text-bull" : "text-bear"}`}>
                      {isLong ? "" : "-"}{Math.abs(h.weight).toFixed(1)}%
                    </span>
                  </div>
                </div>
                <div className="w-28 text-right">
                  <span className="text-sm font-semibold text-foreground">${(h.value / 1000).toFixed(0)}K</span>
                </div>
                <div className="w-24 text-right">
                  <span className={`text-sm font-semibold ${isPos ? "text-bull" : "text-bear"}`}>
                    {isPos ? "+" : ""}{h.pnl < 0 ? "-$" : "$"}{Math.abs(h.pnl / 1000).toFixed(1)}K
                  </span>
                </div>
                <div className="w-24 text-right">
                  <span className={`text-sm font-bold ${isPos ? "text-bull" : "text-bear"} font-mono`}>
                    {isPos ? "+" : ""}{h.pnlPct.toFixed(2)}%
                  </span>
                </div>
                <div className="w-24 flex justify-center">
                  <span className={`text-xs font-semibold px-2 py-0.5 rounded-md ${isLong ? "tag-bull" : "tag-bear"}`}>
                    {isLong ? "LONG" : "SHORT"}
                  </span>
                </div>
                <div className="w-20 flex justify-center">
                  <MiniSparkline data={h.sparkData} positive={isPos} width={60} height={26} animated={false} />
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};

export default Portfolio;
