import { useEffect, useRef } from "react";

interface MiniSparklineProps {
  data?: number[];
  positive?: boolean;
  width?: number;
  height?: number;
  animated?: boolean;
}

const MiniSparkline = ({
  data = [10, 12, 11, 15, 14, 18, 16, 20, 19, 24, 22, 26],
  positive = true,
  width = 80,
  height = 32,
  animated = true,
}: MiniSparklineProps) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animRef = useRef<number>(0);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.scale(dpr, dpr);

    const color = positive ? "#00C805" : "#FF5252";
    const min = Math.min(...data);
    const max = Math.max(...data);
    const range = max - min || 1;

    const getX = (i: number) => (i / (data.length - 1)) * width;
    const getY = (v: number) => height - ((v - min) / range) * (height - 4) - 2;

    const totalFrames = animated ? 45 : 1;

    const draw = (frame: number) => {
      ctx.clearRect(0, 0, width, height);
      const p = animated ? Math.min(frame / totalFrames, 1) : 1;
      const pointCount = Math.max(2, Math.floor(p * data.length));

      ctx.beginPath();
      ctx.moveTo(getX(0), getY(data[0]));
      for (let i = 1; i < pointCount; i++) {
        const cp1x = getX(i - 0.5);
        const cp1y = getY(data[i - 1]);
        const cp2x = getX(i - 0.5);
        const cp2y = getY(data[i]);
        ctx.bezierCurveTo(cp1x, cp1y, cp2x, cp2y, getX(i), getY(data[i]));
      }

      ctx.strokeStyle = color;
      ctx.lineWidth = 1.5;
      ctx.stroke();

      // Gradient fill
      const last = pointCount - 1;
      ctx.lineTo(getX(last), height);
      ctx.lineTo(getX(0), height);
      ctx.closePath();
      const grad = ctx.createLinearGradient(0, 0, 0, height);
      grad.addColorStop(0, positive ? "rgba(0,200,5,0.18)" : "rgba(255,82,82,0.18)");
      grad.addColorStop(1, "transparent");
      ctx.fillStyle = grad;
      ctx.fill();

      if (animated && frame < totalFrames) {
        animRef.current = requestAnimationFrame(() => draw(frame + 1));
      }
    };

    animRef.current = requestAnimationFrame(() => draw(0));
    return () => cancelAnimationFrame(animRef.current);
  }, [data, positive, width, height, animated]);

  return (
    <canvas
      ref={canvasRef}
      style={{ width, height }}
      className="block"
    />
  );
};

export default MiniSparkline;
