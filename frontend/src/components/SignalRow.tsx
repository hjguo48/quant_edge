import { useState } from "react";
import { ChevronRight } from "lucide-react";
import MiniSparkline from "./MiniSparkline";
import { getSectorColor } from "../constants/sectorColors";

interface SignalRowProps {
  ticker?: string;
  name?: string;
  direction?: "long" | "short" | "neutral";
  confidence?: number;
  score?: number;
  sparkData?: number[];
  sector?: string;
  onClick?: () => void;
}

const SignalRow = ({
  ticker = "AAPL",
  name = "Apple Inc.",
  direction = "long",
  confidence = 78,
  score = 0.0,
  sparkData = [],
  sector = "—",
  onClick = () => {},
}: SignalRowProps) => {
  const [hovered, setHovered] = useState(false);

  const directionClass =
    direction === "long" ? "tag-bull" : direction === "short" ? "tag-bear" : "tag-neutral";
  const directionLabel =
    direction === "long" ? "LONG SIGNAL" : direction === "short" ? "SHORT SIGNAL" : "NEUTRAL";
  const isPositive = direction === "long";

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
      <div className="w-28 flex-shrink-0">
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
              background: isPositive
                ? `linear-gradient(90deg, #00C805 0%, #00A804 100%)`
                : `linear-gradient(90deg, #FF5252 0%, #E04040 100%)`,
            }}
          />
        </div>
      </div>

      {/* Score */}
      <div className="w-20 text-right flex-shrink-0">
        <div className={`text-sm font-bold ${isPositive ? "text-bull" : "text-bear"}`}>
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
