import { useState } from "react";
import { ChevronRight } from "lucide-react";
import MiniSparkline from "./MiniSparkline";

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

const SECTOR_COLORS: Record<string, { bg: string; text: string; border: string }> = {
  "Technology":             { bg: "rgba(59,130,246,0.12)",  text: "#60A5FA",  border: "rgba(59,130,246,0.25)" },
  "Healthcare":             { bg: "rgba(168,85,247,0.12)",  text: "#C084FC",  border: "rgba(168,85,247,0.25)" },
  "Financials":             { bg: "rgba(234,179,8,0.12)",   text: "#FACC15",  border: "rgba(234,179,8,0.25)" },
  "Consumer Discretionary": { bg: "rgba(249,115,22,0.12)",  text: "#FB923C",  border: "rgba(249,115,22,0.25)" },
  "Consumer Staples":       { bg: "rgba(34,197,94,0.12)",   text: "#4ADE80",  border: "rgba(34,197,94,0.25)" },
  "Energy":                 { bg: "rgba(239,68,68,0.12)",   text: "#F87171",  border: "rgba(239,68,68,0.25)" },
  "Industrials":            { bg: "rgba(148,163,184,0.12)", text: "#94A3B8",  border: "rgba(148,163,184,0.25)" },
  "Materials":              { bg: "rgba(45,212,191,0.12)",   text: "#2DD4BF",  border: "rgba(45,212,191,0.25)" },
  "Real Estate":            { bg: "rgba(244,114,182,0.12)", text: "#F472B6",  border: "rgba(244,114,182,0.25)" },
  "Utilities":              { bg: "rgba(163,230,53,0.12)",  text: "#A3E635",  border: "rgba(163,230,53,0.25)" },
  "Communication Services": { bg: "rgba(56,189,248,0.12)",  text: "#38BDF8",  border: "rgba(56,189,248,0.25)" },
  "Default":                { bg: "rgba(148,163,184,0.08)", text: "#64748B",  border: "rgba(148,163,184,0.15)" },
};

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

  const sColor = (sector && SECTOR_COLORS[sector]) || SECTOR_COLORS["Default"];

  return (
    <div
      data-cmp="SignalRow"
      onClick={onClick}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      className={`flex items-center gap-4 px-5 py-3.5 border-b border-border cursor-pointer transition-all duration-200 ${hovered ? "bg-accent/50" : ""}`}
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
        <MiniSparkline data={sparkData} positive={isPositive} width={80} height={32} />
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
