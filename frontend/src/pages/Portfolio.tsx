import { useState, useMemo, useRef, useEffect } from "react";
import { TrendingUp, TrendingDown, PieChart, DollarSign, RefreshCw, Calculator, ShoppingCart, ShieldCheck, ArrowRight, AlertCircle } from "lucide-react";
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
  const [budgetStr, setBudgetStr] = useState("100000");
  const [prevBudgetStr, setBudgetStrPrev] = useState("100000");
  const totalBudget = parseInt(budgetStr) || 0;
  const [debouncedTotalBudget, setDebouncedTotalBudget] = useState(100000);

  // Debounce budget calculation to avoid excessive API calls and layout flickering
  useEffect(() => {
    if (totalBudget < 1000) {
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
    retry: false,
  });

  const summaryQuery = useQuery<PortfolioSummaryResponse>({
    queryKey: ["portfolioSummary"],
    queryFn: () => fetchApi<PortfolioSummaryResponse>("/api/portfolio/summary"),
    retry: false,
  });

  const budgetQuery = useQuery<BudgetResponse>({
    queryKey: ["portfolioBudget", debouncedTotalBudget],
    queryFn: () => fetchApi<BudgetResponse>(`/api/portfolio/budget?total_budget=${debouncedTotalBudget}`),
    retry: false,
    enabled: debouncedTotalBudget >= 1000,
    staleTime: 10000, // Cache for 10 seconds
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
    
    let formatted: (string | number)[] = [];
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

        {tab === "holdings" && (
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
            ) : current?.holdings.map((h, i) => {
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
          </div>
        )}

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
            ) : rebalanceQuery.data?.orders.map((order, i) => (
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
