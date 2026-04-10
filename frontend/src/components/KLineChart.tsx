import { useEffect, useRef } from "react";

interface Candle {
  time: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

interface KLineChartProps {
  ticker?: string;
  candles?: Candle[];
  height?: number;
}

function generateCandles(n: number): Candle[] {
  const candles: Candle[] = [];
  let price = 185;
  const now = new Date();
  for (let i = n; i >= 0; i--) {
    const d = new Date(now);
    d.setDate(d.getDate() - i);
    const open = price;
    const change = (Math.random() - 0.48) * 4;
    const close = Math.max(100, open + change);
    const high = Math.max(open, close) + Math.random() * 2;
    const low = Math.min(open, close) - Math.random() * 2;
    const volume = Math.floor(20000000 + Math.random() * 30000000);
    candles.push({
      time: d.toISOString().split("T")[0],
      open: parseFloat(open.toFixed(2)),
      close: parseFloat(close.toFixed(2)),
      high: parseFloat(high.toFixed(2)),
      low: parseFloat(low.toFixed(2)),
      volume,
    });
    price = close;
  }
  return candles;
}

const KLineChart = ({
  ticker = "AAPL",
  candles,
  height = 280,
}: KLineChartProps) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const data = candles ?? generateCandles(60);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    const W = canvas.offsetWidth;
    const H = height;
    canvas.width = W * dpr;
    canvas.height = H * dpr;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.scale(dpr, dpr);

    const volH = 40;
    const chartH = H - volH - 10;
    const padding = { left: 8, right: 8, top: 16, bottom: 4 };
    const chartW = W - padding.left - padding.right;
    const n = data.length;
    const candleW = Math.max(3, Math.floor(chartW / n) - 1);
    const spacing = chartW / n;

    const prices = data.flatMap((c) => [c.high, c.low]);
    const minP = Math.min(...prices);
    const maxP = Math.max(...prices);
    const priceRange = maxP - minP;

    const toY = (p: number) =>
      padding.top + (1 - (p - minP) / priceRange) * (chartH - padding.top - padding.bottom);

    const maxVol = Math.max(...data.map((c) => c.volume));
    const volTop = chartH + 10;

    // Background grid
    ctx.strokeStyle = "rgba(255,255,255,0.04)";
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const y = padding.top + (i / 4) * (chartH - padding.top - padding.bottom);
      ctx.beginPath();
      ctx.moveTo(padding.left, y);
      ctx.lineTo(W - padding.right, y);
      ctx.stroke();
    }

    // Draw candles
    data.forEach((c, i) => {
      const x = padding.left + i * spacing + spacing / 2;
      const isGreen = c.close >= c.open;
      const color = isGreen ? "#00C805" : "#FF5252";

      // Wick
      ctx.strokeStyle = color;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x, toY(c.high));
      ctx.lineTo(x, toY(c.low));
      ctx.stroke();

      // Body
      const bodyTop = toY(Math.max(c.open, c.close));
      const bodyBot = toY(Math.min(c.open, c.close));
      const bodyH = Math.max(1, bodyBot - bodyTop);
      ctx.fillStyle = color;
      ctx.fillRect(x - candleW / 2, bodyTop, candleW, bodyH);

      // Volume bar
      const vH = (c.volume / maxVol) * (volH - 8);
      ctx.globalAlpha = 0.4;
      ctx.fillStyle = color;
      ctx.fillRect(x - candleW / 2, volTop + (volH - 8 - vH), candleW, vH);
      ctx.globalAlpha = 1;
    });

    // Price labels
    ctx.fillStyle = "rgba(96,123,150,0.8)";
    ctx.font = `${10 * dpr > 12 ? 10 : 10}px Inter, sans-serif`;
    ctx.textAlign = "right";
    for (let i = 0; i <= 4; i++) {
      const p = minP + (priceRange * (4 - i)) / 4;
      const y = padding.top + (i / 4) * (chartH - padding.top - padding.bottom);
      ctx.fillText(`$${p.toFixed(1)}`, W - padding.right - 2, y - 2);
    }

    // Latest price line
    const last = data[data.length - 1];
    const lastY = toY(last.close);
    const isLastGreen = last.close >= last.open;
    ctx.strokeStyle = isLastGreen ? "rgba(0,200,5,0.5)" : "rgba(255,82,82,0.5)";
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(padding.left, lastY);
    ctx.lineTo(W - padding.right, lastY);
    ctx.stroke();
    ctx.setLineDash([]);

    // Price badge
    const badgeW = 56;
    ctx.fillStyle = isLastGreen ? "#00C805" : "#FF5252";
    ctx.beginPath();
    ctx.roundRect(W - padding.right - badgeW - 2, lastY - 9, badgeW, 18, 4);
    ctx.fill();
    ctx.fillStyle = "#0D1421";
    ctx.font = "bold 10px Inter, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(`$${last.close.toFixed(2)}`, W - padding.right - badgeW / 2 - 2, lastY + 4);
  }, [data, height]);

  const last = data[data.length - 1];
  const first = data[0];
  const totalChange = ((last.close - first.open) / first.open * 100);
  const isPositive = totalChange >= 0;

  return (
    <div data-cmp="KLineChart" className="bg-card rounded-xl border border-border p-4">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-3">
          <span className="text-base font-bold text-foreground">{ticker}</span>
          <span className={`text-sm font-bold ${isPositive ? "text-bull" : "text-bear"} font-mono`}>
            ${last.close.toFixed(2)}
          </span>
          <span className={`text-xs font-semibold px-2 py-0.5 rounded-md ${isPositive ? "tag-bull" : "tag-bear"}`}>
            {isPositive ? "+" : ""}{totalChange.toFixed(2)}%
          </span>
        </div>
        <div className="flex items-center gap-1.5">
          {["1D","1W","1M","3M","YTD"].map((t) => (
            <button key={t} className={`text-xs px-2 py-1 rounded-md transition-all duration-200 ${t === "1M" ? "bg-primary text-primary-foreground font-semibold" : "text-muted-foreground hover:text-foreground hover:bg-accent"}`}>
              {t}
            </button>
          ))}
        </div>
      </div>
      <canvas
        ref={canvasRef}
        style={{ width: "100%", height }}
        className="block w-full"
      />
    </div>
  );
};

export default KLineChart;
