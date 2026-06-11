import { useMemo } from "react";
import { useTranslation } from "react-i18next";
import { TrendingUp } from "lucide-react";

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

// Monochrome orange ramp (largest slice = most vivid), shadcn chart style
const RAMP = ["#f97316", "#fb923c", "#fdba74", "#ea580c", "#c2410c", "#9a3412", "#78716c"];

const SIZE = 180;
const CX = SIZE / 2;
const CY = SIZE / 2;
const R = 64;
const STROKE = 21;
const HOVER_STROKE = 25;
const GAP_DEG = 3.2;

function polar(r: number, angleDeg: number): [number, number] {
  const rad = ((angleDeg - 90) * Math.PI) / 180;
  return [CX + r * Math.cos(rad), CY + r * Math.sin(rad)];
}

function arcPath(r: number, startDeg: number, endDeg: number): string {
  const [sx, sy] = polar(r, startDeg);
  const [ex, ey] = polar(r, endDeg);
  const largeArc = endDeg - startDeg > 180 ? 1 : 0;
  return `M ${sx.toFixed(3)} ${sy.toFixed(3)} A ${r} ${r} 0 ${largeArc} 1 ${ex.toFixed(3)} ${ey.toFixed(3)}`;
}

const SectorDonut = ({ slices, totalTickers, hovered, onHover }: SectorDonutProps) => {
  const { t } = useTranslation();

  const totalWeight = slices.reduce((sum, s) => sum + s.weight, 0);

  const segments = useMemo(() => {
    let cursor = 0;
    return slices.map((s, i) => {
      const sweep = totalWeight > 0 ? (s.weight / totalWeight) * 360 : 0;
      const start = cursor + GAP_DEG / 2;
      const end = cursor + Math.max(sweep - GAP_DEG / 2, GAP_DEG / 2 + 0.5);
      cursor += sweep;
      const arcLen = ((end - start) * Math.PI * R) / 180;
      return {
        ...s,
        color: s.isOther ? RAMP[RAMP.length - 1] : RAMP[Math.min(i, RAMP.length - 2)],
        path: arcPath(R, start, end),
        arcLen,
        pct: totalWeight > 0 ? (s.weight / totalWeight) * 100 : 0,
      };
    });
  }, [slices, totalWeight]);

  const hoveredSeg = segments.find((s) => s.name === hovered) ?? null;
  const top = segments[0];

  return (
    <div className="flex flex-col h-full">
      {/* Chart + Legend, side by side like the reference */}
      <div className="flex items-center gap-4 flex-1">
        <div className="relative flex-shrink-0" style={{ width: SIZE, height: SIZE }}>
          <svg width={SIZE} height={SIZE} viewBox={`0 0 ${SIZE} ${SIZE}`}>
            {segments.map((seg) => {
              const isActive = hovered === seg.name;
              const isDimmed = hovered != null && !isActive;
              return (
                <path
                  key={seg.name}
                  d={seg.path}
                  fill="none"
                  stroke={seg.color}
                  strokeWidth={isActive ? HOVER_STROKE : STROKE}
                  strokeLinecap="butt"
                  style={{
                    // Active base dims so the sweep overlay reads as "filling in"
                    opacity: isActive ? 0.25 : isDimmed ? 0.3 : 1,
                    transition: "stroke-width 250ms ease, opacity 250ms ease",
                    cursor: "pointer",
                  }}
                  onMouseEnter={() => onHover(seg.name)}
                  onMouseLeave={() => onHover(null)}
                />
              );
            })}
            {/* Sweep overlay: same-color arc draws itself from one end to the other */}
            {hoveredSeg && (
              <path
                key={`sweep-${hoveredSeg.name}`}
                d={hoveredSeg.path}
                fill="none"
                stroke={hoveredSeg.color}
                strokeWidth={HOVER_STROKE}
                strokeLinecap="butt"
                pointerEvents="none"
                style={
                  {
                    "--sweep-len": `${hoveredSeg.arcLen.toFixed(2)}px`,
                    strokeDasharray: `${hoveredSeg.arcLen.toFixed(2)}px`,
                    animation: "sectorSweep 550ms ease-out forwards",
                    filter: "brightness(1.15)",
                  } as React.CSSProperties
                }
              />
            )}
          </svg>
          {/* Center label */}
          <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none">
            {hoveredSeg ? (
              <>
                <span className="text-xl font-black text-foreground font-mono leading-none">
                  {hoveredSeg.pct.toFixed(1)}%
                </span>
                <span className="text-[10px] text-muted-foreground font-semibold mt-1 max-w-[90px] truncate text-center">
                  {t(`sectors.${hoveredSeg.name}`, { defaultValue: hoveredSeg.name })}
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

        {/* Legend list */}
        <div className="flex-1 min-w-0 space-y-1">
          {segments.map((seg) => {
            const isActive = hovered === seg.name;
            const isDimmed = hovered != null && !isActive;
            return (
              <div
                key={seg.name}
                onMouseEnter={() => onHover(seg.name)}
                onMouseLeave={() => onHover(null)}
                className="flex items-center justify-between gap-2 px-1.5 py-1 rounded-md cursor-pointer transition-all duration-200"
                style={{
                  opacity: isDimmed ? 0.4 : 1,
                  background: isActive ? "rgba(255,255,255,0.05)" : "transparent",
                }}
              >
                <div className="flex items-center gap-2 min-w-0">
                  <span
                    className="w-2 h-2 rounded-[2px] flex-shrink-0 transition-transform duration-200"
                    style={{ background: seg.color, transform: isActive ? "scale(1.4)" : "scale(1)" }}
                  />
                  <span className="text-[11px] text-muted-foreground font-medium truncate">
                    {seg.isOther
                      ? t("portfolio.sectorWeights.other", { defaultValue: "Other" })
                      : t(`sectors.${seg.name}`, { defaultValue: seg.name })}
                  </span>
                </div>
                <span className="text-[11px] font-mono font-bold text-foreground flex-shrink-0">
                  {seg.pct.toFixed(1)}%
                </span>
              </div>
            );
          })}
        </div>
      </div>

      {/* Footer, like the reference trend lines */}
      <div className="mt-4 pt-3 border-t border-border/40 space-y-1 flex-shrink-0">
        {top && (
          <div className="flex items-center gap-1.5 text-xs font-semibold text-foreground">
            {t("portfolio.sectorWeights.leadsAt", {
              defaultValue: "{{sector}} leads at {{pct}}%",
              sector: t(`sectors.${top.name}`, { defaultValue: top.name }),
              pct: top.pct.toFixed(1),
            })}
            <TrendingUp size={13} className="text-bull" />
          </div>
        )}
        <p className="text-[10px] text-muted-foreground">
          {t("portfolio.sectorWeights.showing", {
            defaultValue: "Showing sector weights for current holdings · {{sectors}} sectors · {{tickers}} tickers",
            sectors: slices.length,
            tickers: totalTickers,
          })}
        </p>
      </div>
    </div>
  );
};

export default SectorDonut;
