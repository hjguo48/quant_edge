import { useEffect, useLayoutEffect, useMemo, useRef, useState, type CSSProperties, type MouseEvent } from "react";
import { useQuery } from "@tanstack/react-query";
import { CandlestickChart, TrendingUp, RefreshCcw } from "lucide-react";
import { fetchApi } from "../hooks/useApi";

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

interface IntradayAggregate {
  t: number;
  o?: number | null;
  h?: number | null;
  l?: number | null;
  c?: number | null;
  v?: number | null;
}

interface IntradayResponse {
  ticker?: string;
  status?: string;
  results?: IntradayAggregate[];
  resultsCount?: number;
}

interface IntradayRequestConfig {
  path: string;
  isIntraday: boolean;
}

type RangeKey = "1D" | "1W" | "1M" | "3M" | "1Y" | "5Y" | "All";
type ChartMode = "candlestick" | "line";

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
  { key: "1W", label: "1W", days: 5 },
  { key: "1M", label: "1M", days: 21 },
  { key: "3M", label: "3M", days: 63 },
  { key: "1Y", label: "1Y", days: 252 },
  { key: "5Y", label: "5Y", days: 1260 },
  { key: "All", label: "All", days: 2520 },
];

const MOTION_CSS = `
@keyframes kline-live-flash {
  0% { opacity: 0; transform: scale(0.88); }
  25% { opacity: 0.85; transform: scale(1.08); }
  100% { opacity: 0; transform: scale(1.22); }
}
`;

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

function mapPricesToCandles(payload?: StockPricesResponse): Candle[] {
  if (!payload?.prices) return [];
  
  return payload.prices
    .filter((price) => {
      const close = price.adj_close ?? price.close;
      return typeof close === "number" && Number.isFinite(close);
    })
    .map((price) => {
      const adjRatio = (price.adj_close && price.close && price.close !== 0)
        ? Number(price.adj_close) / Number(price.close)
        : 1;
      return {
        time: price.trade_date,
        open: Number((price.open ?? price.close ?? 0)) * adjRatio,
        high: Number((price.high ?? price.close ?? 0)) * adjRatio,
        low: Number((price.low ?? price.close ?? 0)) * adjRatio,
        close: Number(price.adj_close ?? price.close ?? 0),
        volume: Number(price.volume ?? 0),
      };
    });
}

function mapIntradayToCandles(payload?: IntradayResponse): Candle[] {
  if (!payload?.results) return [];

  return payload.results
    .filter((bar) => typeof bar.t === "number" && typeof bar.c === "number" && Number.isFinite(bar.c))
    .map((bar) => ({
      time: new Date(bar.t).toISOString(),
      open: Number(bar.o ?? bar.c ?? 0),
      high: Number(bar.h ?? bar.c ?? 0),
      low: Number(bar.l ?? bar.c ?? 0),
      close: Number(bar.c ?? 0),
      volume: Number(bar.v ?? 0),
    }));
}

function formatTradeDate(value: string, range: RangeKey): string {
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return value;

  if (range === "1D") {
    return parsed.toLocaleTimeString("en-US", {
      hour: "numeric",
      minute: "2-digit",
      hour12: true,
    });
  }

  if (range === "1W" || range === "1M") {
    const m = parsed.getMonth() + 1;
    const d = parsed.getDate();
    const time = parsed.toLocaleTimeString("en-US", {
      hour: "numeric",
      minute: "2-digit",
      hour12: true,
    });
    return `${m}/${d} ${time}`;
  }

  return parsed.toLocaleDateString("en-US", {
    year: "numeric",
    month: "short",
    day: "numeric",
  });
}

function easeOutCubic(progress: number): number {
  return 1 - Math.pow(1 - progress, 3);
}

function getRefetchInterval(range: RangeKey): number {
  if (range === "1D") return 30_000;
  if (range === "1W" || range === "1M") return 60_000;
  return 300_000;
}

function formatMarketDate(date: Date): string {
  const formatter = new Intl.DateTimeFormat("en-CA", {
    timeZone: "America/New_York",
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  });
  return formatter.format(date);
}

function subtractTradingDays(reference: Date, tradingDays: number): Date {
  const result = new Date(reference);
  let remaining = tradingDays;

  while (remaining > 0) {
    result.setDate(result.getDate() - 1);
    const weekday = result.getDay();
    if (weekday !== 0 && weekday !== 6) {
      remaining -= 1;
    }
  }

  return result;
}

function buildIntradayRequestConfig(ticker: string, range: RangeKey): IntradayRequestConfig | null {
  const normalizedTicker = ticker.trim().toUpperCase();
  if (!normalizedTicker) return null;

  if (range === "1D") {
    return {
      isIntraday: true,
      path: `/api/market/intraday?ticker=${normalizedTicker}&multiplier=1&timespan=minute`,
    };
  }

  if (range === "1W" || range === "1M") {
    const today = new Date();
    const lookbackDays = range === "1W" ? 4 : 20;
    const fromDate = formatMarketDate(subtractTradingDays(today, lookbackDays));
    const toDate = formatMarketDate(today);
    return {
      isIntraday: true,
      path:
        `/api/market/intraday?ticker=${normalizedTicker}` +
        `&multiplier=1&timespan=hour&from_date=${fromDate}&to_date=${toDate}`,
    };
  }

  return null;
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
  const rangeSelectorRef = useRef<HTMLDivElement>(null);
  const rangeButtonRefs = useRef<Array<HTMLButtonElement | null>>([]);
  const [selectedRange, setSelectedRange] = useState<RangeKey>(defaultRange);
  const [chartMode, setChartMode] = useState<ChartMode>("candlestick");
  const [chartWidth, setChartWidth] = useState(0);
  const [hoverState, setHoverState] = useState<HoverState | null>(null);
  const [isTransitioning, setIsTransitioning] = useState(false);
  const [intradayRevealProgress, setIntradayRevealProgress] = useState(1);
  const [intradayFlash, setIntradayFlash] = useState(false);
  const [rangeSliderStyle, setRangeSliderStyle] = useState<CSSProperties>({
    width: 0,
    transform: "translateX(0px)",
    opacity: 0,
  });
  const intradaySignatureRef = useRef<string | null>(null);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect -- reset on prop change
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
    // eslint-disable-next-line react-hooks/set-state-in-effect -- clear hover on range change
    setHoverState(null);
  }, [selectedRange]);

  useLayoutEffect(() => {
    const updateRangeSlider = () => {
      const selectedIndex = RANGE_OPTIONS.findIndex((option) => option.key === selectedRange);
      const button = rangeButtonRefs.current[selectedIndex];

      if (!button) {
        setRangeSliderStyle((current) => ({ ...current, opacity: 0 }));
        return;
      }

      setRangeSliderStyle({
        width: button.offsetWidth,
        transform: `translateX(${Math.max(0, button.offsetLeft - 4)}px)`,
        opacity: 1,
      });
    };

    updateRangeSlider();

    const resizeObserver = new ResizeObserver(updateRangeSlider);
    const container = rangeSelectorRef.current;
    if (container) {
      resizeObserver.observe(container);
    }
    rangeButtonRefs.current.forEach((button) => {
      if (button) {
        resizeObserver.observe(button);
      }
    });
    window.addEventListener("resize", updateRangeSlider);

    return () => {
      resizeObserver.disconnect();
      window.removeEventListener("resize", updateRangeSlider);
    };
  }, [selectedRange]);

  const fallbackCandlesData = useMemo(() => generateCandles(60), []);

  const selectedDays = getDaysForRange(selectedRange);
  const intradayRequest = ticker ? buildIntradayRequestConfig(ticker, selectedRange) : null;
  const isIntradayMode = intradayRequest?.isIntraday ?? false;
  const fallbackCandles = useMemo(
    () => (!ticker && !candles?.length ? fallbackCandlesData : []),
    [ticker, candles, fallbackCandlesData],
  );

  const pricesQuery = useQuery<StockPricesResponse, Error>({
    queryKey: ["klineChartPrices", ticker, selectedDays],
    queryFn: () => fetchApi<StockPricesResponse>(`/api/stocks/${ticker}/prices?days=${selectedDays}`),
    enabled: Boolean(ticker) && !isIntradayMode,
    staleTime: 60_000,
    retry: false,
    placeholderData: (previousData) => previousData,
    refetchInterval: !isIntradayMode ? getRefetchInterval(selectedRange) : false,
    refetchIntervalInBackground: false,
  });

  const intradayQuery = useQuery<IntradayResponse, Error>({
    queryKey: ["klineChartIntraday", ticker, selectedRange, intradayRequest?.path],
    queryFn: () => fetchApi<IntradayResponse>(intradayRequest?.path ?? ""),
    enabled: Boolean(ticker) && isIntradayMode,
    staleTime: 10_000,
    retry: false,
    placeholderData: (previousData) => previousData,
    refetchInterval: isIntradayMode ? getRefetchInterval(selectedRange) : false,
    refetchIntervalInBackground: false,
  });

  const liveCandles = useMemo(
    () => (isIntradayMode ? mapIntradayToCandles(intradayQuery.data) : mapPricesToCandles(pricesQuery.data)),
    [intradayQuery.data, isIntradayMode, pricesQuery.data],
  );
  const activeQuery = isIntradayMode ? intradayQuery : pricesQuery;
  const data = useMemo(
    () => (liveCandles.length > 0 ? liveCandles : (candles?.length ? candles : (ticker ? [] : fallbackCandles))),
    [liveCandles, candles, ticker, fallbackCandles],
  );

  useEffect(() => {
    if (liveCandles.length === 0) {
      intradaySignatureRef.current = null;
      // eslint-disable-next-line react-hooks/set-state-in-effect -- reset flash when stream empty
      setIntradayFlash(false);
      return;
    }

    const lastCandle = liveCandles[liveCandles.length - 1];
    const signature = `${liveCandles.length}:${lastCandle.time}:${lastCandle.close}:${lastCandle.volume}`;
    if (intradaySignatureRef.current === null) {
      intradaySignatureRef.current = signature;
      return;
    }
    if (intradaySignatureRef.current === signature) {
      return;
    }

    intradaySignatureRef.current = signature;
    setIntradayFlash(true);
    const timeoutId = window.setTimeout(() => setIntradayFlash(false), 650);
    return () => window.clearTimeout(timeoutId);
  }, [liveCandles]);

  useEffect(() => {
    if (data.length === 0) {
      // eslint-disable-next-line react-hooks/set-state-in-effect -- finalize reveal when no data
      setIntradayRevealProgress(1);
      return;
    }

    let animationFrame = 0;
    const start = performance.now();
    const durationMs = 900;
    setIntradayRevealProgress(0);

    const tick = (now: number) => {
      const progress = Math.min(1, (now - start) / durationMs);
      setIntradayRevealProgress(easeOutCubic(progress));
      if (progress < 1) {
        animationFrame = window.requestAnimationFrame(tick);
      }
    };

    animationFrame = window.requestAnimationFrame(tick);
    return () => window.cancelAnimationFrame(animationFrame);
  }, [chartMode, data.length, liveCandles.length, selectedRange, ticker]);

  const metrics = useMemo(() => {
    if (!chartWidth || data.length === 0) return null;

    const width = chartWidth;
    const padding = { left: 12, right: 56, top: 16, bottom: 22 };
    const volumeHeight = 44;
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

  const lastCandle = data.length > 0 ? data[data.length - 1] : null;
  const firstCandle = data.length > 0 ? data[0] : null;
  const totalChange =
    lastCandle && firstCandle ? ((lastCandle.close - firstCandle.open) / firstCandle.open) * 100 : 0;
  const isPositive = totalChange >= 0;

  const latestMarker = useMemo(() => {
    if (!metrics || !lastCandle || data.length === 0) return null;

    const lastIndex = data.length - 1;
    const x = metrics.padding.left + lastIndex * metrics.spacing + metrics.spacing / 2;
    const priceToY = (price: number) =>
      metrics.padding.top +
      (1 - (price - metrics.minPrice) / metrics.priceRange) *
        (metrics.chartHeight - metrics.padding.top - metrics.padding.bottom);
    const y = priceToY(lastCandle.close);
    const openY = priceToY(lastCandle.open);

    return {
      x,
      y,
      bodyTop: Math.min(y, openY),
      bodyHeight: Math.max(6, Math.abs(openY - y)),
      color: isPositive ? "#00C805" : "#FF5252",
    };
  }, [data.length, isPositive, lastCandle, metrics]);

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

    const volumeTop = metrics.chartHeight + 8;
    const revealSweep = intradayRevealProgress * data.length;

    // Draw Grid
    ctx.strokeStyle = "rgba(255,255,255,0.04)";
    ctx.lineWidth = 1;
    for (let index = 0; index <= 4; index += 1) {
      const y = metrics.padding.top + (index / 4) * (metrics.chartHeight - metrics.padding.top - metrics.padding.bottom);
      ctx.beginPath();
      ctx.moveTo(metrics.padding.left, y);
      ctx.lineTo(width - metrics.padding.right, y);
      ctx.stroke();
    }

    const firstPoint = data[0];
    const lastPoint = data[data.length - 1];
    const isOverallPositive = lastPoint.close >= firstPoint.open;
    const themeColor = isOverallPositive ? "#00C805" : "#FF5252";

    if (chartMode === "line") {
      const visibleCount = Math.max(2, Math.ceil(intradayRevealProgress * data.length));
      const visibleData = data.slice(0, visibleCount);

      // Draw Area
      const gradient = ctx.createLinearGradient(0, metrics.padding.top, 0, metrics.chartHeight);
      gradient.addColorStop(0, `${themeColor}33`);
      gradient.addColorStop(1, `${themeColor}00`);
      
      ctx.beginPath();
      visibleData.forEach((candle, index) => {
        const x = metrics.padding.left + index * metrics.spacing + metrics.spacing / 2;
        const y = toY(candle.close);
        if (index === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      
      // Close the path for filling
      const firstX = metrics.padding.left + metrics.spacing / 2;
      const lastX = metrics.padding.left + (visibleData.length - 1) * metrics.spacing + metrics.spacing / 2;
      ctx.lineTo(lastX, metrics.chartHeight);
      ctx.lineTo(firstX, metrics.chartHeight);
      ctx.closePath();
      ctx.fillStyle = gradient;
      ctx.globalAlpha = 0.55 + intradayRevealProgress * 0.45;
      ctx.fill();
      ctx.globalAlpha = 1;

      // Draw Line
      ctx.beginPath();
      ctx.strokeStyle = themeColor;
      ctx.lineWidth = 2;
      ctx.lineJoin = "round";
      ctx.lineCap = "round";
      visibleData.forEach((candle, index) => {
        const x = metrics.padding.left + index * metrics.spacing + metrics.spacing / 2;
        const y = toY(candle.close);
        if (index === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
    } else {
      // Draw Candlesticks
      data.forEach((candle, index) => {
        const reveal = Math.max(0, Math.min(1, revealSweep - index));
        if (reveal <= 0) return;
        const x = metrics.padding.left + index * metrics.spacing + metrics.spacing / 2;
        const isBullish = candle.close >= candle.open;
        const color = isBullish ? "#00C805" : "#FF5252";
        const easedReveal = easeOutCubic(reveal);

        ctx.strokeStyle = color;
        ctx.lineWidth = 1;
        ctx.globalAlpha = 0.35 + easedReveal * 0.65;
        ctx.beginPath();
        const lowY = toY(candle.low);
        const highY = toY(candle.high);
        const animatedHighY = lowY - (lowY - highY) * easedReveal;
        ctx.moveTo(x, animatedHighY);
        ctx.lineTo(x, lowY);
        ctx.stroke();

        const bodyTop = toY(Math.max(candle.open, candle.close));
        const bodyBottom = toY(Math.min(candle.open, candle.close));
        const bodyHeight = Math.max(1, bodyBottom - bodyTop);

        ctx.fillStyle = color;
        const animatedBodyTop = bodyBottom - bodyHeight * easedReveal;
        ctx.fillRect(
          x - metrics.candleWidth / 2,
          animatedBodyTop,
          metrics.candleWidth,
          Math.max(1, bodyBottom - animatedBodyTop),
        );
        ctx.globalAlpha = 1;
      });
    }

    // Draw Volume (Always)
    data.forEach((candle, index) => {
      const reveal = Math.max(0, Math.min(1, revealSweep - index));
      if (reveal <= 0) return;
      const x = metrics.padding.left + index * metrics.spacing + metrics.spacing / 2;
      const isBullish = candle.close >= candle.open;
      const color = isBullish ? "#00C805" : "#FF5252";
      
      const volumeBarHeight = (candle.volume / metrics.maxVolume) * (metrics.volumeHeight - 8) * easeOutCubic(reveal);
      ctx.fillStyle = color;
      ctx.globalAlpha = 0.35;
      ctx.fillRect(
        x - metrics.candleWidth / 2,
        volumeTop + (metrics.volumeHeight - 8 - volumeBarHeight),
        metrics.candleWidth,
        volumeBarHeight,
      );
      ctx.globalAlpha = 1;
    });

    // Draw Price Labels
    ctx.fillStyle = "rgba(96,123,150,0.85)";
    ctx.font = "10px Inter, sans-serif";
    ctx.textAlign = "right";
    for (let index = 0; index <= 4; index += 1) {
      const price = metrics.minPrice + (metrics.priceRange * (4 - index)) / 4;
      const y = metrics.padding.top + (index / 4) * (metrics.chartHeight - metrics.padding.top - metrics.padding.bottom);
      ctx.fillText(`$${price.toFixed(1)}`, width - metrics.padding.right + 24, y - 2);
    }

    // X-axis labels removed per user request

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

      ctx.fillStyle = activeCandle.close >= (chartMode === "candlestick" ? activeCandle.open : activeCandle.close) ? "#00C805" : "#FF5252";
      // In line mode, we can just use the themeColor or point specific color. 
      // Requirement says "线条颜色: 如果整体涨 → bull, 跌 → bear". 
      // For hover dot, let's use themeColor for consistency in line mode.
      ctx.fillStyle = chartMode === "line" ? themeColor : (activeCandle.close >= activeCandle.open ? "#00C805" : "#FF5252");
      
      ctx.beginPath();
      ctx.arc(activeX, activeY, 3, 0, Math.PI * 2);
      ctx.fill();
    } else {
      const isBullish = activeCandle.close >= activeCandle.open;
      const currentThemeColor = chartMode === "line" ? themeColor : (isBullish ? "#00C805" : "#FF5252");

      if (chartMode === "candlestick") {
        ctx.strokeStyle = currentThemeColor;
        ctx.lineWidth = 1;
        ctx.setLineDash([3, 3]);
        ctx.beginPath();
        ctx.moveTo(metrics.padding.left, activeY);
        ctx.lineTo(width - metrics.padding.right, activeY);
        ctx.stroke();
        ctx.setLineDash([]);

        const badgeWidth = 56;
        ctx.fillStyle = currentThemeColor;
        ctx.beginPath();
        ctx.roundRect(width - metrics.padding.right - badgeWidth + 6, activeY - 9, badgeWidth, 18, 4);
        ctx.fill();
        ctx.fillStyle = "#0D1421";
        ctx.font = "bold 10px Inter, sans-serif";
        ctx.textAlign = "center";
        ctx.fillText(`$${activeCandle.close.toFixed(2)}`, width - metrics.padding.right - badgeWidth / 2 + 6, activeY + 4);
      }
    }
  }, [chartMode, data, height, hoverState, intradayRevealProgress, isIntradayMode, metrics, selectedRange]);

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

  const isInitialLoading = activeQuery.isLoading && liveCandles.length === 0 && !candles?.length && Boolean(ticker);
  const hasError = activeQuery.isError && data.length === 0 && Boolean(ticker);
  const activeCandle = hoverState ? data[hoverState.index] : null;
  const chartModeTranslate = chartMode === "candlestick" ? "translateX(0%)" : "translateX(100%)";
  const tooltipLeft =
    hoverState && metrics ? Math.max(12, Math.min(hoverState.x + 14, metrics.width - 176)) : 0;
  const tooltipTop = hoverState ? Math.max(12, Math.min(hoverState.y + 14, height - 126)) : 0;
  const displayedPrice = lastCandle?.close ?? null;

  return (
    <div data-cmp="KLineChart" className="bg-card rounded-xl border border-border p-4">
      <style>{MOTION_CSS}</style>
      <div className="flex items-center justify-between mb-3 gap-3">
        <div className="flex items-center gap-3 flex-wrap">
          <span className="text-base font-bold text-foreground">{ticker}</span>
          <span
            className={`text-sm font-bold ${isPositive ? "text-bull" : "text-bear"} font-mono tracking-tight`}
          >
            {displayedPrice !== null ? `$${displayedPrice.toFixed(2)}` : "—"}
          </span>
          <span className={`text-xs font-semibold px-2 py-0.5 rounded-md ${isPositive ? "tag-bull" : "tag-bear"}`}>
            {totalChange >= 0 ? "+" : ""}{totalChange.toFixed(2)}%
          </span>
          {activeQuery.isFetching && !isInitialLoading && (
            <span className="text-[10px] font-semibold uppercase tracking-[0.16em] text-muted-foreground animate-pulse">
              Refreshing
            </span>
          )}
        </div>

        <div className="flex items-center gap-3 flex-wrap">
          {/* Chart Type Toggle */}
          <div className="relative grid grid-cols-2 items-center rounded-lg bg-accent/50 p-1">
            <div
              aria-hidden="true"
              className="absolute bottom-1 left-1 top-1 w-[calc(50%-4px)] rounded-md bg-card shadow-sm transition-transform duration-300 ease-out"
              style={{ transform: chartModeTranslate }}
            />
            <button
              onClick={() => setChartMode("candlestick")}
              className={`relative z-10 flex h-6 w-7 items-center justify-center rounded-md transition-colors duration-300 ${
                chartMode === "candlestick" ? "text-primary" : "text-muted-foreground hover:text-foreground"
              }`}
              title="Candlestick"
              aria-label="Candlestick chart"
            >
              <CandlestickChart size={12} />
            </button>
            <button
              onClick={() => setChartMode("line")}
              className={`relative z-10 flex h-6 w-7 items-center justify-center rounded-md transition-colors duration-300 ${
                chartMode === "line" ? "text-primary" : "text-muted-foreground hover:text-foreground"
              }`}
              title="Line Chart"
              aria-label="Line chart"
            >
              <TrendingUp size={12} />
            </button>
          </div>

          <div className="w-px h-4 bg-border" />

          {/* Range Selector */}
          <div ref={rangeSelectorRef} className="relative flex items-center gap-1 bg-accent/50 p-1 rounded-lg">
            <div
              aria-hidden="true"
              className="absolute bottom-1 top-1 left-1 rounded-md bg-card shadow-sm transition-all duration-300 ease-out"
              style={rangeSliderStyle}
            />
            {RANGE_OPTIONS.map((option, index) => (
              <button
                key={option.key}
                ref={(node) => {
                  rangeButtonRefs.current[index] = node;
                }}
                onClick={() => {
                  if (option.key === selectedRange) return;
                  setIsTransitioning(true);
                  setSelectedRange(option.key);
                  setTimeout(() => setIsTransitioning(false), 350);
                }}
                className={`relative z-10 px-3 py-1 text-[10px] font-bold rounded-md transition-colors duration-300 ${
                  selectedRange === option.key
                    ? "text-foreground"
                    : "text-muted-foreground hover:text-foreground"
                }`}
              >
                {option.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      <div ref={chartAreaRef} className={`relative transition-all duration-500 ease-out origin-center ${isTransitioning ? "opacity-20 scale-[0.985]" : "opacity-100 scale-100"}`}>
        {isInitialLoading ? (
          <div className="flex items-center justify-center rounded-lg bg-surface animate-pulse" style={{ height }}>
            <span className="text-sm text-muted-foreground">Loading price history…</span>
          </div>
        ) : hasError ? (
          <div className="flex flex-col items-center justify-center rounded-lg border border-dashed border-border bg-surface px-4 text-center" style={{ height }}>
            <span className="text-sm text-bear mb-2">Failed to load price history.</span>
            <p className="text-[10px] text-muted-foreground mb-3 max-w-[200px]">
              {activeQuery.error?.message || "Internal server error"}
            </p>
            <button 
              onClick={() => activeQuery.refetch()}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-muted hover:bg-accent transition-colors"
            >
              <RefreshCcw size={12} />
              Retry
            </button>
          </div>
        ) : data.length === 0 ? (
          <div className="flex items-center justify-center rounded-lg border border-dashed border-border bg-surface text-sm text-muted-foreground" style={{ height }}>
            No price data available.
          </div>
        ) : (
          <>
            {chartMode === "line" && latestMarker && (
              <>
                <div
                  aria-hidden="true"
                  className="pointer-events-none absolute rounded-full"
                  style={{
                    left: latestMarker.x - 4,
                    top: latestMarker.y - 4,
                    width: 8,
                    height: 8,
                    background: latestMarker.color,
                    boxShadow: `0 0 10px ${latestMarker.color}`,
                    opacity: 0.92,
                  }}
                />
                {intradayFlash && (
                  <div
                    aria-hidden="true"
                    className="pointer-events-none absolute rounded-full"
                    style={{
                      left: latestMarker.x - 10,
                      top: latestMarker.y - 10,
                      width: 20,
                      height: 20,
                      border: `1px solid ${latestMarker.color}`,
                      animation: "kline-live-flash 650ms ease-out forwards",
                    }}
                  />
                )}
              </>
            )}
            <canvas
              key={chartMode}
              ref={canvasRef}
              onMouseMove={handleMouseMove}
              onMouseLeave={() => setHoverState(null)}
              style={{ width: "100%", height, cursor: "crosshair" }}
              className="block w-full animate-in fade-in duration-500"
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
                  {formatTradeDate(activeCandle.time, selectedRange)}
                </div>
                <div className="space-y-1">
                  {chartMode === "candlestick" && (
                    <>
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
                    </>
                  )}
                  <div className="flex items-center justify-between gap-3 text-xs">
                    <span className="text-muted-foreground">{chartMode === "candlestick" ? "Close" : "Price"}</span>
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
