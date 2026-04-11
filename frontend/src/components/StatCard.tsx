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

interface ParsedAnimatedValue {
  currencySymbol: string;
  decimals: number;
  scale: number;
  showPositiveSign: boolean;
  unitSuffix: string;
  value: number;
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

function parseAnimatedValue(value: string): ParsedAnimatedValue | null {
  const trimmed = value.trim();
  const match = trimmed.match(/^(\$)?([+-]?\d[\d,]*(?:\.\d+)?)(%|[KMBTkmbt])?$/);

  if (!match) return null;

  const [, currencySymbol = "", numericToken, rawSuffix = ""] = match;
  const numericValue = Number(numericToken.replace(/,/g, ""));
  if (!Number.isFinite(numericValue)) return null;

  const normalizedSuffix = rawSuffix.toUpperCase();
  const scaleMap: Record<string, number> = {
    "": 1,
    "%": 1,
    K: 1_000,
    M: 1_000_000,
    B: 1_000_000_000,
    T: 1_000_000_000_000,
  };

  return {
    currencySymbol,
    decimals: numericToken.includes(".") ? numericToken.split(".")[1].length : 0,
    scale: scaleMap[normalizedSuffix] ?? 1,
    showPositiveSign: trimmed.startsWith("+"),
    unitSuffix: normalizedSuffix,
    value: numericValue * (scaleMap[normalizedSuffix] ?? 1),
  };
}

function formatAnimatedValue(value: number, parsed: ParsedAnimatedValue): string {
  const scaledValue = value / parsed.scale;
  const absoluteValue = Math.abs(scaledValue);
  const sign = scaledValue < 0 ? "-" : parsed.showPositiveSign ? "+" : "";
  const formattedNumber = absoluteValue.toLocaleString("en-US", {
    minimumFractionDigits: parsed.decimals,
    maximumFractionDigits: parsed.decimals,
  });

  return `${sign}${parsed.currencySymbol}${formattedNumber}${parsed.unitSuffix}`;
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
  const parsedValue = parseAnimatedValue(value);
  const shouldAnimateValue = animateValue && parsedValue !== null;

  const animated = useCountUp(parsedValue?.value ?? 0, 1200, visible && shouldAnimateValue);

  useEffect(() => {
    const timer = setTimeout(() => setVisible(true), delay);
    return () => clearTimeout(timer);
  }, [delay]);

  const formatDisplay = () => {
    if (!visible || !shouldAnimateValue || parsedValue === null) return value;
    return formatAnimatedValue(animated, parsedValue);
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
