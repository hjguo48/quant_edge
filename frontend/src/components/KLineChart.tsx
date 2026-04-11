import { useEffect, useMemo, useRef, useState, type MouseEvent } from "react";
import { useQuery } from "@tanstack/react-query";

interface Candle {
  time: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

interface StockPricesResponse {
  ticker: string;
  days: number;
  prices: Array<{
    trade_date: string;
    open?: number | null;
    high?: number | null;
    low?: number | null;
    close?: number | null;
    adj_close?: number | null;
    volume?: number | null;
  }>;
}

type RangeKey = "1D" | "1W" | "1M" | "3M" | "1Y";

interface RangeOption {
  key: RangeKey;
  label: string;
  days: number;
}

interface HoverState {
  index: number;
  x: number;
  y: number;
}

interface KLineChartProps {
  ticker?: string;
  candles?: Candle[];
  height?: number;
  defaultRange?: RangeKey;
}

const RANGE_OPTIONS: RangeOption[] = [
  { key: "1D", label: "1D", days: 1 },
  { key: "1W", label: "1W", days: 7 },
  { key: "1M", label: "1M", days: 30 },
  { key: "3M", label: "3M", days: 90 },
  { key: "1Y", label: "1Y", days: 365 },
];

function generateCandles(n: number): Candle[] {
  const generated: Candle[] = [];
  let price = 185;
  const now = new Date();

  for (let i = n; i >= 0; i -= 1) {
    const date = new Date(now);
    date.setDate(date.getDate() - i);
    const open = price;
    const change = (Math.random() - 0.48) * 4;
    const close = Math.max(100, open + change);
    const high = Math.max(open, close) + Math.random() * 2;
    const low = Math.min(open, close) - Math.random() * 2;
    const volume = Math.floor(20_000_000 + Math.random() * 30_000_000);

    generated.push({
      time: date.toISOString().split("T")[0],
      open: parseFloat(open.toFixed(2)),
      high: parseFloat(high.toFixed(2)),
      low: parseFloat(low.toFixed(2)),
      close: parseFloat(close.toFixed(2)),
      volume,
    });

    price = close;
  }

  return generated;
}

function getDaysForRange(range: RangeKey): number {
  return RANGE_OPTIONS.find((option) => option.key === range)?.days ?? 30;
}

async function fetchJson<T>(url: string): Promise<T> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Request failed with status ${response.status}`);
  }

  return response.json() as Promise<T>;
}

function mapPricesToCandles(payload?: StockPricesResponse): Candle[] {
  return (payload?.prices ?? [])
    .filter((price) => {
      const open = price.open ?? price.close;
      const high = price.high ?? price.close;
      const low = price.low ?? price.close;
      const close = price.close ?? price.adj_close;

      return [open, high, low, close].every((value) => typeof value === "number" && Number.isFinite(value));
    })
    .map((price) => ({
      time: price.trade_date,
      open: Number(price.open ?? price.close ?? 0),
      high: Number(price.high ?? price.close ?? 0),
      low: Number(price.low ?? price.close ?? 0),
      close: Number(price.close ?? price.adj_close ?? 0),
      volume: Number(price.volume ?? 0),
    }));
}

function formatTradeDate(value: string): string {
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return value;

  return parsed.toLocaleDateString("en-US", {
    year: "numeric",
    month: "short",
    day: "numeric",
  });
}

function formatVolume(value: number): string {
  return new Intl.NumberFormat("en-US", {
    notation: "compact",
    maximumFractionDigits: 2,
  }).format(value);
}

const KLineChart = ({
  ticker = "AAPL",
  candles,
  height = 280,
  defaultRange = "1M",
}: KLineChartProps) => {
  const chartAreaRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const fallbackCandlesRef = useRef<Candle[] | null>(null);
  const [selectedRange, setSelectedRange] = useState<RangeKey>(defaultRange);
  const [chartWidth, setChartWidth] = useState(0);
  const [hoverState, setHoverState] = useState<HoverState | null>(null);

  useEffect(() => {
    setSelectedRange(defaultRange);
    setHoverState(null);
  }, [defaultRange, ticker]);

  useEffect(() => {
    if (!chartAreaRef.current) return undefined;

    const updateWidth = () => {
      setChartWidth(chartAreaRef.current?.clientWidth ?? 0);
    };

    updateWidth();

    const resizeObserver = new ResizeObserver(updateWidth);
    resizeObserver.observe(chartAreaRef.current);
    window.addEventListener("resize", updateWidth);

    return () => {
      resizeObserver.disconnect();
      window.removeEventListener("resize", updateWidth);
    };
  }, []);

  useEffect(() => {
    setHoverState(null);
  }, [selectedRange]);

  if (!fallbackCandlesRef.current) {
    fallbackCandlesRef.current = generateCandles(60);
  }

  const selectedDays = getDaysForRange(selectedRange);
  const fallbackCandles = !ticker && !candles?.length ? fallbackCandlesRef.current ?? [] : [];

  const pricesQuery = useQuery<StockPricesResponse>({
    queryKey: ["klineChartPrices", ticker, selectedDays],
    queryFn: () => fetchJson(`/api/stocks/${ticker}/prices?days=${selectedDays}`),
    enabled: Boolean(ticker),
    staleTime: 60_000,
    placeholderData: (previousData) => previousData,
  });

  const liveCandles = useMemo(() => mapPricesToCandles(pricesQuery.data), [pricesQuery.data]);
  const data = liveCandles.length > 0 ? liveCandles : candles?.length ? candles : fallbackCandles;

  const metrics = useMemo(() => {
    if (!chartWidth || data.length === 0) return null;

    const width = chartWidth;
    const padding = { left: 12, right: 56, top: 16, bottom: 4 };
    const volumeHeight = 40;
    const chartHeight = height - volumeHeight - 10;
    const plotWidth = Math.max(0, width - padding.left - padding.right);
    const count = Math.max(data.length, 1);
    const spacing = plotWidth / count;
    const candleWidth = Math.max(3, Math.floor(plotWidth / count) - 2);
    const prices = data.flatMap((candle) => [candle.high, candle.low]);
    const minPrice = Math.min(...prices);
    const maxPrice = Math.max(...prices);
    const priceRange = Math.max(maxPrice - minPrice, maxPrice * 0.01, 1);
    const maxVolume = Math.max(...data.map((candle) => candle.volume), 1);

    return {
      width,
      padding,
      volumeHeight,
      chartHeight,
      plotWidth,
      spacing,
      candleWidth,
      minPrice,
      priceRange,
      maxVolume,
    };
  }, [chartWidth, data, height]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !metrics || data.length === 0) return;

    const dpr = window.devicePixelRatio || 1;
    const width = metrics.width;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, width, height);

    const toY = (price: number) =>
      metrics.padding.top +
      (1 - (price - metrics.minPrice) / metrics.priceRange) *
        (metrics.chartHeight - metrics.padding.top - metrics.padding.bottom);

    const volumeTop = metrics.chartHeight + 10;

    ctx.strokeStyle = "rgba(255,255,255,0.04)";
    ctx.lineWidth = 1;
    for (let index = 0; index <= 4; index += 1) {
      const y = metrics.padding.top + (index / 4) * (metrics.chartHeight - metrics.padding.top - metrics.padding.bottom);
      ctx.beginPath();
      ctx.moveTo(metrics.padding.left, y);
      ctx.lineTo(width - metrics.padding.right, y);
      ctx.stroke();
    }

    data.forEach((candle, index) => {
      const x = metrics.padding.left + index * metrics.spacing + metrics.spacing / 2;
      const isBullish = candle.close >= candle.open;
      const color = isBullish ? "#00C805" : "#FF5252";

      ctx.strokeStyle = color;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x, toY(candle.high));
      ctx.lineTo(x, toY(candle.low));
      ctx.stroke();

      const bodyTop = toY(Math.max(candle.open, candle.close));
      const bodyBottom = toY(Math.min(candle.open, candle.close));
      const bodyHeight = Math.max(1, bodyBottom - bodyTop);

      ctx.fillStyle = color;
      ctx.fillRect(x - metrics.candleWidth / 2, bodyTop, metrics.candleWidth, bodyHeight);

      const volumeBarHeight = (candle.volume / metrics.maxVolume) * (metrics.volumeHeight - 8);
      ctx.globalAlpha = 0.35;
      ctx.fillRect(
        x - metrics.candleWidth / 2,
        volumeTop + (metrics.volumeHeight - 8 - volumeBarHeight),
        metrics.candleWidth,
        volumeBarHeight,
      );
      ctx.globalAlpha = 1;
    });

    ctx.fillStyle = "rgba(96,123,150,0.85)";
    ctx.font = "10px Inter, sans-serif";
    ctx.textAlign = "right";
    for (let index = 0; index <= 4; index += 1) {
      const price = metrics.minPrice + (metrics.priceRange * (4 - index)) / 4;
      const y = metrics.padding.top + (index / 4) * (metrics.chartHeight - metrics.padding.top - metrics.padding.bottom);
      ctx.fillText(`$${price.toFixed(1)}`, width - metrics.padding.right + 24, y - 2);
    }

    const activeIndex = hoverState?.index ?? data.length - 1;
    const activeCandle = data[activeIndex];
    const activeX = metrics.padding.left + activeIndex * metrics.spacing + metrics.spacing / 2;
    const activeY = toY(activeCandle.close);

    if (hoverState) {
      ctx.strokeStyle = "rgba(96,123,150,0.7)";
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 4]);

      ctx.beginPath();
      ctx.moveTo(activeX, metrics.padding.top);
      ctx.lineTo(activeX, height - 4);
      ctx.stroke();

      ctx.beginPath();
      ctx.moveTo(metrics.padding.left, activeY);
      ctx.lineTo(width - metrics.padding.right, activeY);
      ctx.stroke();

      ctx.setLineDash([]);

      ctx.fillStyle = activeCandle.close >= activeCandle.open ? "#00C805" : "#FF5252";
      ctx.beginPath();
      ctx.arc(activeX, activeY, 3, 0, Math.PI * 2);
      ctx.fill();
    } else {
      const isBullish = activeCandle.close >= activeCandle.open;
      ctx.strokeStyle = isBullish ? "rgba(0,200,5,0.5)" : "rgba(255,82,82,0.5)";
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(metrics.padding.left, activeY);
      ctx.lineTo(width - metrics.padding.right, activeY);
      ctx.stroke();
      ctx.setLineDash([]);

      const badgeWidth = 56;
      ctx.fillStyle = isBullish ? "#00C805" : "#FF5252";
      ctx.beginPath();
      ctx.roundRect(width - metrics.padding.right - badgeWidth + 6, activeY - 9, badgeWidth, 18, 4);
      ctx.fill();
      ctx.fillStyle = "#0D1421";
      ctx.font = "bold 10px Inter, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText(`$${activeCandle.close.toFixed(2)}`, width - metrics.padding.right - badgeWidth / 2 + 6, activeY + 4);
    }
  }, [data, height, hoverState, metrics]);

  const handleMouseMove = (event: MouseEvent<HTMLCanvasElement>) => {
    if (!metrics) return;

    const rect = event.currentTarget.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    const minX = metrics.padding.left;
    const maxX = metrics.width - metrics.padding.right;

    if (x < minX || x > maxX || y < 0 || y > height) {
      setHoverState(null);
      return;
    }

    const rawIndex = Math.round((x - metrics.padding.left - metrics.spacing / 2) / metrics.spacing);
    const nextIndex = Math.min(Math.max(rawIndex, 0), data.length - 1);

    setHoverState({ index: nextIndex, x, y });
  };

  const isInitialLoading = pricesQuery.isLoading && liveCandles.length === 0 && !candles?.length && Boolean(ticker);
  const hasError = pricesQuery.isError && data.length === 0 && Boolean(ticker);
  const activeCandle = hoverState ? data[hoverState.index] : null;
  const lastCandle = data.length > 0 ? data[data.length - 1] : null;
  const firstCandle = data.length > 0 ? data[0] : null;
  const totalChange =
    lastCandle && firstCandle ? ((lastCandle.close - firstCandle.open) / firstCandle.open) * 100 : 0;
  const isPositive = totalChange >= 0;
  const tooltipLeft =
    hoverState && metrics ? Math.max(12, Math.min(hoverState.x + 14, metrics.width - 176)) : 0;
  const tooltipTop = hoverState ? Math.max(12, Math.min(hoverState.y + 14, height - 126)) : 0;

  return (
    <div data-cmp="KLineChart" className="bg-card rounded-xl border border-border p-4">
      <div className="flex items-center justify-between mb-3 gap-3">
        <div className="flex items-center gap-3 flex-wrap">
          <span className="text-base font-bold text-foreground">{ticker}</span>
          <span className={`text-sm font-bold ${isPositive ? "text-bull" : "text-bear"} font-mono`}>
            {lastCandle ? `$${lastCandle.close.toFixed(2)}` : "—"}
          </span>
          <span className={`text-xs font-semibold px-2 py-0.5 rounded-md ${isPositive ? "tag-bull" : "tag-bear"}`}>
            {totalChange >= 0 ? "+" : ""}{totalChange.toFixed(2)}%
          </span>
          {pricesQuery.isFetching && !isInitialLoading && (
            <span className="text-[10px] font-semibold uppercase tracking-[0.16em] text-muted-foreground">
              Refreshing
            </span>
          )}
        </div>

        <div className="flex items-center gap-1.5 flex-wrap">
          {RANGE_OPTIONS.map((option) => (
            <button
              key={option.key}
              onClick={() => setSelectedRange(option.key)}
              className={`text-xs px-2 py-1 rounded-md transition-all duration-200 ${
                selectedRange === option.key
                  ? "bg-primary text-primary-foreground font-semibold"
                  : "text-muted-foreground hover:text-foreground hover:bg-accent"
              }`}
            >
              {option.label}
            </button>
          ))}
        </div>
      </div>

      <div ref={chartAreaRef} className="relative">
        {isInitialLoading ? (
          <div className="flex items-center justify-center rounded-lg bg-surface animate-pulse" style={{ height }}>
            <span className="text-sm text-muted-foreground">Loading price history…</span>
          </div>
        ) : hasError ? (
          <div className="flex items-center justify-center rounded-lg border border-dashed border-border bg-surface text-sm text-bear" style={{ height }}>
            Failed to load price history.
          </div>
        ) : data.length === 0 ? (
          <div className="flex items-center justify-center rounded-lg border border-dashed border-border bg-surface text-sm text-muted-foreground" style={{ height }}>
            No price data available.
          </div>
        ) : (
          <>
            <canvas
              ref={canvasRef}
              onMouseMove={handleMouseMove}
              onMouseLeave={() => setHoverState(null)}
              style={{ width: "100%", height, cursor: "crosshair" }}
              className="block w-full"
            />

            {activeCandle && (
              <div
                className="pointer-events-none absolute z-10 rounded-lg border border-border px-3 py-2 shadow-custom"
                style={{
                  left: tooltipLeft,
                  top: tooltipTop,
                  backgroundColor: "#131C2E",
                  minWidth: 164,
                }}
              >
                <div className="text-[10px] font-semibold uppercase tracking-[0.16em] text-muted-foreground mb-1.5">
                  {formatTradeDate(activeCandle.time)}
                </div>
                <div className="space-y-1">
                  <div className="flex items-center justify-between gap-3 text-xs">
                    <span className="text-muted-foreground">Open</span>
                    <span className="font-mono text-foreground">${activeCandle.open.toFixed(2)}</span>
                  </div>
                  <div className="flex items-center justify-between gap-3 text-xs">
                    <span className="text-muted-foreground">High</span>
                    <span className="font-mono text-foreground">${activeCandle.high.toFixed(2)}</span>
                  </div>
                  <div className="flex items-center justify-between gap-3 text-xs">
                    <span className="text-muted-foreground">Low</span>
                    <span className="font-mono text-foreground">${activeCandle.low.toFixed(2)}</span>
                  </div>
                  <div className="flex items-center justify-between gap-3 text-xs">
                    <span className="text-muted-foreground">Close</span>
                    <span className="font-mono text-foreground">${activeCandle.close.toFixed(2)}</span>
                  </div>
                  <div className="flex items-center justify-between gap-3 text-xs">
                    <span className="text-muted-foreground">Volume</span>
                    <span className="font-mono text-foreground">{formatVolume(activeCandle.volume)}</span>
                  </div>
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
};

export default KLineChart;
