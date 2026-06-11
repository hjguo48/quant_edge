import { useMemo } from "react";
import { useTranslation } from "react-i18next";
import { getSectorColor } from "../constants/sectorColors";

export interface SectorSlice {
  name: string;
  weight: number;
  tickerCount: number;
  isOther: boolean;
}

interface SectorDonutProps {
  slices: SectorSlice[];
  totalTickers: number;
  hovered: string | null;
  onHover: (sector: string | null) => void;
}

const SIZE = 210;
const CX = SIZE / 2;
const CY = SIZE / 2;
const R_INNER = 66;
const R_OUTER = 90;
const TOTAL_BARS = 64;
const BAR_WIDTH = 5;
const SWEEP_TOTAL_MS = 420;
const OTHER_COLOR = "#94a3b8";

function polar(r: number, angleDeg: number): [number, number] {
  const rad = ((angleDeg - 90) * Math.PI) / 180;
  return [CX + r * Math.cos(rad), CY + r * Math.sin(rad)];
}

/** Allocate TOTAL_BARS bars across slices proportionally (largest remainder). */
function allocateBars(slices: SectorSlice[], totalWeight: number): number[] {
  if (totalWeight <= 0 || slices.length === 0) return slices.map(() => 0);
  const exact = slices.map((s) => (s.weight / totalWeight) * TOTAL_BARS);
  const floors = exact.map((v) => Math.max(1, Math.floor(v)));
  let used = floors.reduce((a, b) => a + b, 0);
  const remainders = exact.map((v, i) => ({ i, frac: v - Math.floor(v) }));
  remainders.sort((a, b) => b.frac - a.frac);
  let cursor = 0;
  while (used < TOTAL_BARS && cursor < remainders.length) {
    floors[remainders[cursor].i] += 1;
    used += 1;
    cursor += 1;
  }
  while (used > TOTAL_BARS) {
    const maxIdx = floors.indexOf(Math.max(...floors));
    if (floors[maxIdx] <= 1) break;
    floors[maxIdx] -= 1;
    used -= 1;
  }
  return floors;
}

const SectorDonut = ({ slices, totalTickers, hovered, onHover }: SectorDonutProps) => {
  const { t } = useTranslation();

  const totalWeight = slices.reduce((sum, s) => sum + s.weight, 0);

  const bars = useMemo(() => {
    const counts = allocateBars(slices, totalWeight);
    const step = 360 / TOTAL_BARS;
    const result: {
      key: string;
      sector: string;
      color: string;
      x1: number;
      y1: number;
      x2: number;
      y2: number;
      idxInSector: number;
      sectorBarCount: number;
    }[] = [];
    let barCursor = 0;
    slices.forEach((s, si) => {
      const color = s.isOther ? OTHER_COLOR : getSectorColor(s.name).text;
      for (let j = 0; j < counts[si]; j++) {
        const angle = (barCursor + j) * step;
        const [x1, y1] = polar(R_INNER, angle);
        const [x2, y2] = polar(R_OUTER, angle);
        result.push({
          key: `${s.name}-${j}`,
          sector: s.name,
          color,
          x1,
          y1,
          x2,
          y2,
          idxInSector: j,
          sectorBarCount: counts[si],
        });
      }
      barCursor += counts[si];
    });
    return result;
  }, [slices, totalWeight]);

  const hoveredSlice = slices.find((s) => s.name === hovered) ?? null;
  const hoveredPct =
    hoveredSlice && totalWeight > 0 ? (hoveredSlice.weight / totalWeight) * 100 : null;

  return (
    <div className="relative mx-auto" style={{ width: SIZE, height: SIZE }}>
      <svg width={SIZE} height={SIZE} viewBox={`0 0 ${SIZE} ${SIZE}`}>
        {bars.map((bar) => {
          const isActive = hovered === bar.sector;
          const isDimmed = hovered != null && !isActive;
          const delayMs = isActive
            ? (bar.idxInSector / Math.max(bar.sectorBarCount - 1, 1)) * SWEEP_TOTAL_MS
            : 0;
          return (
            <line
              key={bar.key}
              x1={bar.x1}
              y1={bar.y1}
              x2={bar.x2}
              y2={bar.y2}
              stroke={bar.color}
              strokeWidth={BAR_WIDTH}
              strokeLinecap="round"
              style={
                isActive
                  ? {
                      // Sequential fill: bars light up one after another along the arc
                      opacity: 0.25,
                      animation: `sectorBarFill 220ms ease-out ${delayMs.toFixed(0)}ms forwards`,
                      cursor: "pointer",
                    }
                  : {
                      opacity: isDimmed ? 0.22 : 0.9,
                      transition: "opacity 250ms ease",
                      cursor: "pointer",
                    }
              }
              onMouseEnter={() => onHover(bar.sector)}
              onMouseLeave={() => onHover(null)}
            />
          );
        })}
      </svg>
      {/* Center label */}
      <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none">
        {hoveredSlice && hoveredPct != null ? (
          <>
            <span className="text-xl font-black text-foreground font-mono leading-none">
              {hoveredPct.toFixed(1)}%
            </span>
            <span className="text-[10px] text-muted-foreground font-semibold mt-1 max-w-[100px] truncate text-center">
              {hoveredSlice.isOther
                ? t("portfolio.sectorWeights.other", { defaultValue: "Other" })
                : t(`sectors.${hoveredSlice.name}`, { defaultValue: hoveredSlice.name })}
            </span>
          </>
        ) : (
          <>
            <span className="text-2xl font-black text-foreground font-mono leading-none">{totalTickers}</span>
            <span className="text-[10px] text-muted-foreground font-semibold mt-1">
              {t("portfolio.sectorWeights.totalTickers", { defaultValue: "Total Tickers" })}
            </span>
          </>
        )}
      </div>
    </div>
  );
};

export default SectorDonut;
