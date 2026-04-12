import { useState, useMemo } from "react";
import { TrendingUp, TrendingDown, PieChart, DollarSign, RefreshCw, Calculator, ShoppingCart, ShieldCheck } from "lucide-react";
import { useQuery } from "@tanstack/react-query";
import StatCard from "../components/StatCard";
import { fetchApi } from "../hooks/useApi";

interface PortfolioHolding {
  ticker: string;
  weight: number;
  score: number;
}

interface PortfolioCurrentResponse {
  signal_date: string;
  week_number: number;
  holding_count: number;
  gross_exposure: number;
  cash_weight: number;
  portfolio_beta: number;
  cvar_95: number;
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
  portfolio_beta: number;
  cvar_95: number;
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

const Portfolio = () => {
  const [tab, setTab] = useState("holdings");
  const [totalBudget, setTotalBudget] = useState(100000);

  const currentQuery = useQuery<PortfolioCurrentResponse>({
    queryKey: ["portfolioCurrent"],
    queryFn: () => fetchApi<PortfolioCurrentResponse>("/api/portfolio/current"),
    retry: false,
  });

  const summaryQuery = useQuery<PortfolioSummaryResponse>({
    queryKey: ["portfolioSummary"],
    queryFn: () => fetchApi<PortfolioSummaryResponse>("/api/portfolio/summary"),
    retry: false,
  });

  const budgetQuery = useQuery<BudgetResponse>({
    queryKey: ["portfolioBudget", totalBudget],
    queryFn: () => fetchApi<BudgetResponse>(`/api/portfolio/budget?total_budget=${totalBudget}`),
    retry: false,
  });

  const rebalanceQuery = useQuery<RebalanceResponse>({
    queryKey: ["portfolioRebalance"],
    queryFn: () => fetchApi<RebalanceResponse>("/api/portfolio/rebalance"),
    retry: false,
  });

  const isLoading = currentQuery.isLoading || summaryQuery.isLoading;
  const isError = currentQuery.isError || summaryQuery.isError;

  const current = currentQuery.data;
  const summary = summaryQuery.data;

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
        value: summary.portfolio_beta.toFixed(2), 
        change: 0, 
        changeLabel: "vs. Benchmark", 
        trend: "neutral" as const 
      },
      { 
        label: "CVaR (95%)", 
        value: `${(summary.cvar_95 * 100).toFixed(2)}%`, 
        change: summary.risk_pass ? 0 : 1, 
        changeLabel: summary.risk_pass ? "Risk check passed" : "Risk limit breach", 
        trend: summary.risk_pass ? "up" as const : "down" as const 
      },
    ];
  }, [summary]);

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
    <div className="flex-1 overflow-y-auto p-6 space-y-5">
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
          <div className="flex flex-col items-center justify-center h-full text-center p-8">
            <TrendingUp size={32} className="text-muted-foreground mb-4 opacity-20" />
            <h3 className="text-sm font-semibold text-foreground mb-1">Performance Tracking</h3>
            <p className="text-xs text-muted-foreground max-w-xs">
              Historical portfolio performance and benchmark comparison is currently being migrated to the new Alpha Engine.
            </p>
            <span className="mt-4 px-2.5 py-1 rounded-full bg-muted text-[10px] font-bold uppercase tracking-wider text-muted-foreground border border-border">
              Under Migration
            </span>
          </div>
        </div>

        <div className="w-72 bg-card rounded-xl border border-border p-5 flex-shrink-0 fade-in-up stagger-3">
          <div className="flex flex-col items-center justify-center h-full text-center">
            <PieChart size={32} className="text-muted-foreground mb-4 opacity-20" />
            <h3 className="text-sm font-semibold text-foreground mb-1">Sector Weights</h3>
            <p className="text-xs text-muted-foreground px-4">
              Sector classification for the current universe is processing.
            </p>
            <div className="mt-6 w-full space-y-3">
              {[1, 2, 3].map(i => (
                <div key={i} className="flex items-center justify-between opacity-30">
                  <div className="flex items-center gap-2">
                    <div className="w-2 h-2 rounded-full bg-muted" />
                    <div className="w-16 h-2 bg-muted rounded" />
                  </div>
                  <div className="w-8 h-2 bg-muted rounded" />
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Holdings Table */}
      <div className="bg-card rounded-xl border border-border overflow-hidden fade-in-up stagger-4">
        <div className="flex items-center justify-between px-5 py-4 border-b border-border">
          <div className="flex items-center gap-2">
            <DollarSign size={14} className="text-primary" />
            <h3 className="text-sm font-semibold text-foreground">
              {tab === "holdings" ? "Current Holdings" : tab === "trades" ? "Rebalance Orders" : "Capital Allocation"}
            </h3>
          </div>
          <div className="flex gap-1 bg-muted rounded-lg p-0.5">
            {[
              { id: "holdings", label: "Holdings", icon: ShieldCheck },
              { id: "trades", label: "Trades", icon: ShoppingCart },
              { id: "budget", label: "Budget", icon: Calculator },
            ].map((t) => (
              <button 
                key={t.id} 
                onClick={() => setTab(t.id)} 
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-all duration-200 ${tab === t.id ? "bg-card text-foreground shadow-sm" : "text-muted-foreground hover:text-foreground"}`}
              >
                <t.icon size={12} />
                {t.label}
              </button>
            ))}
          </div>
        </div>

        {tab === "holdings" && (
          <div>
            <div className="flex items-center px-5 py-2.5 border-b border-border bg-muted/20 text-xs font-semibold text-muted-foreground uppercase tracking-wider">
              <div className="w-28">Ticker</div>
              <div className="flex-1">Optimal Weight</div>
              <div className="w-32 text-right">Model Score</div>
              <div className="w-32 text-center">Direction</div>
              <div className="w-10" />
            </div>
            {isLoading ? (
              Array.from({ length: 6 }).map((_, i) => (
                <div key={i} className="px-5 py-4 border-b border-border flex items-center gap-4 animate-pulse">
                  <div className="w-28 h-4 bg-muted rounded" />
                  <div className="flex-1 h-2 bg-muted rounded" />
                  <div className="w-32 h-4 bg-muted rounded" />
                  <div className="w-32 h-4 bg-muted rounded" />
                </div>
              ))
            ) : current?.holdings.map((h, i) => {
              const isLong = h.score > 0;
              return (
                <div
                  key={h.ticker}
                  className="flex items-center px-5 py-3.5 border-b border-border last:border-0 hover:bg-accent/40 transition-colors cursor-pointer fade-in-up"
                  style={{ animationDelay: `${i * 30}ms` }}
                >
                  <div className="w-28">
                    <div className="text-sm font-bold text-foreground">{h.ticker}</div>
                    <div className="text-[10px] text-muted-foreground uppercase font-mono tracking-tighter">Target Allocation</div>
                  </div>
                  <div className="flex-1">
                    <div className="flex items-center gap-3">
                      <div className="w-32 h-1.5 bg-muted rounded-full overflow-hidden">
                        <div
                          className="h-full rounded-full transition-all duration-1000"
                          style={{
                            width: `${Math.min(100, h.weight * 100 * 5)}%`,
                            backgroundColor: isLong ? "#00C805" : "#FF5252",
                          }}
                        />
                      </div>
                      <span className={`text-xs font-bold font-mono ${isLong ? "text-bull" : "text-bear"}`}>
                        {(h.weight * 100).toFixed(2)}%
                      </span>
                    </div>
                  </div>
                  <div className="w-32 text-right">
                    <span className="text-xs font-mono font-bold text-foreground">{h.score.toFixed(4)}</span>
                  </div>
                  <div className="w-32 flex justify-center">
                    <span className={`text-[10px] font-black px-2 py-0.5 rounded border ${isLong ? "bg-bull/5 border-bull/20 text-bull" : "bg-bear/5 border-bear/20 text-bear"}`}>
                      {isLong ? "LONG" : "SHORT"}
                    </span>
                  </div>
                  <div className="w-10 flex justify-end">
                    <RefreshCw size={12} className="text-muted-foreground opacity-20" />
                  </div>
                </div>
              );
            })}
          </div>
        )}

        {tab === "trades" && (
          <div className="p-0">
            <div className="flex items-center px-5 py-2.5 border-b border-border bg-muted/20 text-xs font-semibold text-muted-foreground uppercase tracking-wider">
              <div className="w-28">Ticker</div>
              <div className="w-32 text-center">Action</div>
              <div className="w-32 text-right">Prev Weight</div>
              <div className="w-32 text-right">New Weight</div>
              <div className="flex-1 text-right">Delta</div>
            </div>
            {rebalanceQuery.isLoading ? (
              <div className="p-8 text-center text-xs text-muted-foreground animate-pulse">Calculating rebalance orders...</div>
            ) : rebalanceQuery.isError ? (
              <div className="py-12 text-center">
                <p className="text-sm text-muted-foreground">Unable to load rebalance orders right now.</p>
                <button
                  onClick={() => rebalanceQuery.refetch()}
                  className="mt-3 text-xs text-primary hover:underline"
                >
                  Try again
                </button>
              </div>
            ) : rebalanceQuery.data?.orders.map((order, i) => (
              <div key={order.ticker} className="flex items-center px-5 py-3.5 border-b border-border last:border-0 hover:bg-accent/40 transition-colors">
                <div className="w-28 text-sm font-bold text-foreground">{order.ticker}</div>
                <div className="w-32 flex justify-center">
                  <span className={`text-[10px] font-bold px-2 py-0.5 rounded uppercase ${
                    order.action === "buy" ? "bg-bull text-white" : order.action === "sell" ? "bg-bear text-white" : "bg-muted text-muted-foreground"
                  }`}>
                    {order.action}
                  </span>
                </div>
                <div className="w-32 text-right text-xs font-mono text-muted-foreground">{(order.weight_prev * 100).toFixed(2)}%</div>
                <div className="w-32 text-right text-xs font-mono text-foreground font-bold">{(order.weight_new * 100).toFixed(2)}%</div>
                <div className="flex-1 text-right">
                  <span className={`text-xs font-mono font-bold ${order.weight_delta > 0 ? "text-bull" : order.weight_delta < 0 ? "text-bear" : "text-muted-foreground"}`}>
                    {order.weight_delta > 0 ? "+" : ""}{(order.weight_delta * 100).toFixed(2)}%
                  </span>
                </div>
              </div>
            ))}
            {!rebalanceQuery.isLoading && (!rebalanceQuery.data || rebalanceQuery.data.orders.length === 0) && (
              <div className="py-12 text-center">
                <p className="text-sm text-muted-foreground">No rebalance required for the current period.</p>
              </div>
            )}
          </div>
        )}

        {tab === "budget" && (
          <div className="p-5">
            <div className="flex items-center gap-4 mb-6 p-4 rounded-xl bg-muted/50 border border-border">
              <Calculator size={20} className="text-primary" />
              <div className="flex-1">
                <label className="text-xs font-bold text-muted-foreground uppercase tracking-widest block mb-1">Total Trading Budget (USD)</label>
                <input 
                  type="number" 
                  value={totalBudget} 
                  onChange={(e) => setTotalBudget(Number(e.target.value))}
                  className="bg-transparent text-xl font-mono font-bold text-foreground outline-none border-b border-primary/20 focus:border-primary transition-colors w-full"
                />
              </div>
              <div className="text-right">
                <p className="text-[10px] text-muted-foreground uppercase">Estimated Holdings</p>
                <p className="text-lg font-bold text-foreground">{budgetQuery.data?.allocations.length || 0}</p>
              </div>
            </div>

            <div className="space-y-2">
              <div className="flex items-center px-4 py-2 text-[10px] font-bold text-muted-foreground uppercase tracking-widest">
                <div className="w-24">Ticker</div>
                <div className="w-32 text-right">Weight</div>
                <div className="flex-1 text-right">Dollar Allocation</div>
              </div>
              {budgetQuery.isLoading ? (
                <div className="py-8 text-center text-xs text-muted-foreground animate-pulse">Calculating dollar amounts...</div>
              ) : budgetQuery.isError ? (
                <div className="py-12 text-center">
                  <p className="text-sm text-muted-foreground">Unable to calculate budget allocations.</p>
                  <button
                    onClick={() => budgetQuery.refetch()}
                    className="mt-3 text-xs text-primary hover:underline"
                  >
                    Try again
                  </button>
                </div>
              ) : budgetQuery.data?.allocations.map((alloc) => (
                <div key={alloc.ticker} className="flex items-center px-4 py-3 rounded-lg border border-border bg-card hover:border-primary/30 transition-colors">
                  <div className="w-24 text-sm font-bold text-foreground">{alloc.ticker}</div>
                  <div className="w-32 text-right text-xs font-mono text-muted-foreground">{(alloc.weight * 100).toFixed(2)}%</div>
                  <div className="flex-1 text-right text-sm font-mono font-black text-bull">
                    ${alloc.dollar_amount.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
      
      <div className="flex items-center justify-center gap-2 py-4">
        <ShieldCheck size={12} className="text-muted-foreground" />
        <p className="text-[10px] text-muted-foreground uppercase tracking-widest">
          SEC Compliant Model Output · Not Investment Advice · All weights reflect optimal theoretical allocation
        </p>
      </div>
    </div>
  );
};

export default Portfolio;
