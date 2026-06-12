import { useMemo, useState } from "react";
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
const R_MID = (R_INNER + R_OUTER) / 2;
const TOTAL_BARS = 80;
const BAR_WIDTH = 2;
const SWEEP_TOTAL_MS = 420;
const OTHER_COLOR = "#94a3b8";

function polar(r: number, angleDeg: number): [number, number] {
  const rad = ((angleDeg - 90) * Math.PI) / 180;
  return [CX + r * Math.cos(rad), CY + r * Math.sin(rad)];
}

function arcPath(r: number, startDeg: number, endDeg: number, reverse = false): string {
  const clampedEnd = Math.min(endDeg, startDeg + 359.9);
  const [sx, sy] = polar(r, startDeg);
  const [ex, ey] = polar(r, clampedEnd);
  const largeArc = clampedEnd - startDeg > 180 ? 1 : 0;
  if (reverse) {
    // Path starts at the clockwise end and draws back: dash animation runs CCW
    return `M ${ex.toFixed(3)} ${ey.toFixed(3)} A ${r} ${r} 0 ${largeArc} 0 ${sx.toFixed(3)} ${sy.toFixed(3)}`;
  }
  return `M ${sx.toFixed(3)} ${sy.toFixed(3)} A ${r} ${r} 0 ${largeArc} 1 ${ex.toFixed(3)} ${ey.toFixed(3)}`;
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

  const { bars, hitArcs, sectorMidDeg } = useMemo(() => {
    const counts = allocateBars(slices, totalWeight);
    const step = 360 / TOTAL_BARS;
    const barList: {
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
    const hitList: { sector: string; path: string; pathReverse: string; arcLen: number }[] = [];
    const midDegBySector: Record<string, number> = {};
    let barCursor = 0;
    slices.forEach((s, si) => {
      const color = s.isOther ? OTHER_COLOR : getSectorColor(s.name).text;
      for (let j = 0; j < counts[si]; j++) {
        const angle = (barCursor + j) * step;
        const [x1, y1] = polar(R_INNER, angle);
        const [x2, y2] = polar(R_OUTER, angle);
        barList.push({
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
      // Invisible hit arc covering the sector's full angular span (incl. gaps between bars)
      const startDeg = barCursor * step - step / 2;
      const endDeg = (barCursor + counts[si]) * step - step / 2;
      hitList.push({
        sector: s.name,
        path: arcPath(R_MID, startDeg, endDeg),
        pathReverse: arcPath(R_MID, startDeg, endDeg, true),
        arcLen: ((endDeg - startDeg) * Math.PI * R_MID) / 180,
      });
      midDegBySector[s.name] = (startDeg + endDeg) / 2;
      barCursor += counts[si];
    });
    return { bars: barList, hitArcs: hitList, sectorMidDeg: midDegBySector };
  }, [slices, totalWeight]);

  const hoveredSlice = slices.find((s) => s.name === hovered) ?? null;
  const hoveredPct =
    hoveredSlice && totalWeight > 0 ? (hoveredSlice.weight / totalWeight) * 100 : null;

  // Direction-aware sweep: the fill runs clockwise when the newly hovered sector
  // sits clockwise of the previous reference sector, counter-clockwise otherwise.
  // Initial reference = the largest sector (slices[0]); afterwards = last hovered.
  const [sweep, setSweep] = useState<{ ref: string | null; dir: 1 | -1 }>({ ref: null, dir: 1 });
  if (hovered != null && hovered !== sweep.ref) {
    const refName = sweep.ref ?? slices[0]?.name ?? hovered;
    const refMid = sectorMidDeg[refName] ?? 0;
    const newMid = sectorMidDeg[hovered] ?? 0;
    const deltaCw = (newMid - refMid + 360) % 360;
    const dir: 1 | -1 = refName === hovered ? 1 : deltaCw <= 180 ? 1 : -1;
    setSweep({ ref: hovered, dir });
  }

  return (
    <div className="relative mx-auto" style={{ width: SIZE, height: SIZE }}>
      <svg width={SIZE} height={SIZE} viewBox={`0 0 ${SIZE} ${SIZE}`}>
        {bars.map((bar) => {
          const isDimmed = hovered != null && hovered !== bar.sector;
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
              pointerEvents="none"
              style={{
                opacity: isDimmed ? 0.22 : 0.9,
                transition: "opacity 250ms ease",
              }}
            />
          );
        })}
        {/* Solid fill sweep: hovered sector's stripes get covered by a solid
            ring segment drawing itself from one end to the other.
            dir=1 → clockwise draw, dir=-1 → counter-clockwise draw */}
        {(() => {
          const hit = hitArcs.find((h) => h.sector === hovered);
          if (!hit) return null;
          const seg = bars.find((b) => b.sector === hovered);
          const color = seg?.color ?? OTHER_COLOR;
          return (
            <path
              key={`fill-${hit.sector}-${sweep.dir}`}
              d={sweep.dir === 1 ? hit.path : hit.pathReverse}
              fill="none"
              stroke={color}
              strokeWidth={R_OUTER - R_INNER}
              strokeLinecap="butt"
              pointerEvents="none"
              style={
                {
                  "--sweep-len": `${hit.arcLen.toFixed(2)}px`,
                  strokeDasharray: `${hit.arcLen.toFixed(2)}px`,
                  animation: `sectorSweep ${SWEEP_TOTAL_MS}ms ease-out forwards`,
                } as React.CSSProperties
              }
            />
          );
        })()}
        {/* Transparent hit arcs: hovering anywhere within a sector's ring span
            (including gaps between bars) triggers that sector */}
        {hitArcs.map((hit) => (
          <path
            key={`hit-${hit.sector}`}
            d={hit.path}
            fill="none"
            stroke="transparent"
            strokeWidth={R_OUTER - R_INNER + 14}
            pointerEvents="stroke"
            style={{ cursor: "pointer" }}
            onMouseEnter={() => onHover(hit.sector)}
            onMouseLeave={() => onHover(null)}
          />
        ))}
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
