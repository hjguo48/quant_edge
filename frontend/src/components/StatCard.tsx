import { useEffect, useRef, useState } from "react";
import { TrendingUp, TrendingDown, Minus } from "lucide-react";

interface StatCardProps {
  label?: string;
  value?: string;
  change?: number;
  changeLabel?: string;
  prefix?: string;
  suffix?: string;
  trend?: "up" | "down" | "neutral";
  animateValue?: boolean;
  delay?: number;
}

function useCountUp(target: number, duration: number, start: boolean) {
  const [current, setCurrent] = useState(0);
  useEffect(() => {
    if (!start) return;
    let startTime: number | null = null;
    const step = (ts: number) => {
      if (!startTime) startTime = ts;
      const progress = Math.min((ts - startTime) / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3);
      setCurrent(eased * target);
      if (progress < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }, [target, duration, start]);
  return current;
}

const StatCard = ({
  label = "Total P&L",
  value = "$0",
  change = 2.4,
  changeLabel = "vs. yesterday",
  prefix = "",
  suffix = "",
  trend = "up",
  animateValue = true,
  delay = 0,
}: StatCardProps) => {
  const ref = useRef<HTMLDivElement>(null);
  const [visible, setVisible] = useState(false);

  const numericValue = parseFloat(value.replace(/[^0-9.-]/g, "")) || 0;
  const animated = useCountUp(numericValue, 1200, visible && animateValue);

  useEffect(() => {
    const timer = setTimeout(() => setVisible(true), delay);
    return () => clearTimeout(timer);
  }, [delay]);

  const formatDisplay = () => {
    if (!animateValue || !visible) return value;
    const hasDollar = value.includes("$");
    const hasPercent = value.includes("%");
    const hasK = value.includes("K") || value.includes("k");
    const hasM = value.includes("M");
    if (hasK) return `${hasDollar ? "$" : ""}${(animated / 1000).toFixed(1)}K`;
    if (hasM) return `${hasDollar ? "$" : ""}${(animated / 1000000).toFixed(2)}M`;
    if (hasDollar) return `$${animated.toFixed(0).replace(/\B(?=(\d{3})+(?!\d))/g, ",")}`;
    if (hasPercent) return `${animated.toFixed(2)}%`;
    return `${animated.toFixed(0)}`;
  };

  const trendColor =
    trend === "up" ? "text-bull" : trend === "down" ? "text-bear" : "text-muted-foreground";

  const TrendIcon =
    trend === "up" ? TrendingUp : trend === "down" ? TrendingDown : Minus;

  return (
    <div
      ref={ref}
      data-cmp="StatCard"
      className="bg-card rounded-xl p-5 border border-border card-hover shine-effect fade-in-up"
      style={{ animationDelay: `${delay}ms` }}
    >
      <div className="flex items-start justify-between mb-3">
        <span className="text-xs font-medium text-muted-foreground uppercase tracking-wider">{label}</span>
        <div className={`flex items-center gap-1 text-xs font-semibold ${trendColor}`}>
          <TrendIcon size={12} />
          <span>{change > 0 ? "+" : ""}{change.toFixed(2)}%</span>
        </div>
      </div>
      <div className="flex items-end gap-1">
        {prefix && <span className="text-sm text-muted-foreground mb-0.5">{prefix}</span>}
        <span className="text-2xl font-bold text-foreground number-animate font-mono">
          {formatDisplay()}
        </span>
        {suffix && <span className="text-sm text-muted-foreground mb-0.5">{suffix}</span>}
      </div>
      <p className="text-xs text-muted-foreground mt-2">{changeLabel}</p>
    </div>
  );
};

export default StatCard;
