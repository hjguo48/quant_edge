import { useState } from "react";
import { ChevronRight } from "lucide-react";
import MiniSparkline from "./MiniSparkline";
import { getSectorColor } from "../constants/sectorColors";

type Conviction = "strong" | "long" | "watch" | "buffer" | "short";

interface SignalRowProps {
  ticker?: string;
  name?: string;
  direction?: "long" | "short" | "neutral";
  tier?: Conviction;
  confidence?: number;
  score?: number;
  sparkData?: number[];
  sector?: string;
  onClick?: () => void;
}

function deriveConviction(direction: SignalRowProps["direction"], confidence: number): Conviction {
  if (direction === "neutral") return "buffer";
  if (direction === "short") return "short";
  if (confidence >= 75) return "strong";
  if (confidence < 25) return "watch";
  return "long";
}

const TIER_STYLE: Record<Conviction, { tagClass: string; tagLabel: string; scoreClass: string; barFrom: string; barTo: string; sparkPositive: boolean }> = {
  strong: { tagClass: "tag-bull-strong", tagLabel: "STRONG LONG", scoreClass: "text-bull-strong", barFrom: "#00FF7F", barTo: "#00D060", sparkPositive: true },
  long:   { tagClass: "tag-bull",        tagLabel: "LONG",        scoreClass: "text-bull",        barFrom: "#00C805", barTo: "#00A804", sparkPositive: true },
  watch:  { tagClass: "tag-bull-watch",  tagLabel: "WATCH LONG",  scoreClass: "text-bull-watch",  barFrom: "#C9A445", barTo: "#9A7E32", sparkPositive: true },
  buffer: { tagClass: "tag-neutral",     tagLabel: "BUFFER",      scoreClass: "text-muted-foreground", barFrom: "#607B96", barTo: "#475569", sparkPositive: false },
  short:  { tagClass: "tag-bear",        tagLabel: "SHORT SIGNAL", scoreClass: "text-bear",       barFrom: "#FF5252", barTo: "#E04040", sparkPositive: false },
};

const SignalRow = ({
  ticker = "AAPL",
  name = "Apple Inc.",
  direction = "long",
  tier,
  confidence = 78,
  score = 0.0,
  sparkData = [],
  sector = "—",
  onClick = () => {},
}: SignalRowProps) => {
  const [hovered, setHovered] = useState(false);

  const conviction: Conviction = tier ?? deriveConviction(direction, confidence);
  const style = TIER_STYLE[conviction];
  const directionClass = style.tagClass;
  const directionLabel = style.tagLabel;
  const isPositive = style.sparkPositive;

  const sColor = getSectorColor(sector);

  return (
    <div
      data-cmp="SignalRow"
      onClick={onClick}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      className="flex items-center gap-4 px-5 py-3.5 border-b border-border last:border-0 cursor-pointer transition-all duration-200"
    >
      {/* Ticker */}
      <div className="w-24 flex-shrink-0">
        <div className="text-sm font-bold text-foreground">{ticker}</div>
        <div className="text-xs text-muted-foreground truncate">{name}</div>
      </div>

      {/* Direction */}
      <div className="w-32 flex-shrink-0">
        <span className={`text-xs font-semibold px-2 py-0.5 rounded-md ${directionClass}`}>
          {directionLabel}
        </span>
      </div>

      {/* Confidence Bar */}
      <div className="flex-1 min-w-0">
        <div className="flex items-center justify-between mb-1">
          <span className="text-xs text-muted-foreground">Confidence</span>
          <span className="text-xs font-bold text-foreground">{confidence}%</span>
        </div>
        <div className="h-1.5 bg-muted rounded-full overflow-hidden">
          <div
            className="h-full rounded-full transition-all duration-700"
            style={{
              width: `${confidence}%`,
              background: `linear-gradient(90deg, ${style.barFrom} 0%, ${style.barTo} 100%)`,
            }}
          />
        </div>
      </div>

      {/* Score */}
      <div className="w-20 text-right flex-shrink-0">
        <div className={`text-sm font-bold ${style.scoreClass}`}>
          {score > 0 ? "+" : ""}{score.toFixed(4)}
        </div>
        <div className="text-xs text-muted-foreground">Score</div>
      </div>

      {/* Trend */}
      <div className="w-24 flex justify-center flex-shrink-0">
        <MiniSparkline data={sparkData} positive={isPositive} width={80} height={32} animated={hovered} />
      </div>

      {/* Sector */}
      <div className="w-32 flex justify-end flex-shrink-0">
        <span 
          className="text-[10px] font-semibold px-2 py-0.5 rounded-md border truncate max-w-full"
          style={{ 
            backgroundColor: sColor.bg, 
            color: sColor.text, 
            borderColor: sColor.border 
          }}
        >
          {sector || "—"}
        </span>
      </div>

      {/* Arrow */}
      <ChevronRight
        size={16}
        className={`flex-shrink-0 transition-all duration-200 ${hovered ? "text-primary translate-x-0.5" : "text-border"}`}
      />
    </div>
  );
};

export default SignalRow;
