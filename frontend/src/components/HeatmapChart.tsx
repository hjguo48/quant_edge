import { useMemo } from "react";

interface HeatmapChartProps {
  title?: string;
  subtitle?: string;
  rows?: string[];
  cols?: string[];
  values?: number[][];
  valueFormatter?: (value: number) => string;
  tooltipFormatter?: (value: number) => string;
}

const SECTORS = ["Tech", "Finance", "Healthcare", "Energy", "Consumer", "Utilities", "Materials"];
const FACTORS = ["Momentum", "Value", "Quality", "Low Vol", "Growth", "Carry", "Reversal"];

function generateHeatmap(rows: number, cols: number): number[][] {
  const seeded = [];
  for (let r = 0; r < rows; r++) {
    const row = [];
    for (let c = 0; c < cols; c++) {
      const v = Math.sin(r * 3.7 + c * 2.1) * 0.5 + Math.cos(r * 1.3 + c * 4.2) * 0.5;
      row.push(parseFloat(v.toFixed(3)));
    }
    seeded.push(row);
  }
  return seeded;
}

function valueToColor(v: number): string {
  if (v > 0.4) return "rgba(0,200,5,0.85)";
  if (v > 0.2) return "rgba(0,200,5,0.55)";
  if (v > 0.05) return "rgba(0,200,5,0.25)";
  if (v > -0.05) return "rgba(96,123,150,0.3)";
  if (v > -0.2) return "rgba(255,82,82,0.25)";
  if (v > -0.4) return "rgba(255,82,82,0.55)";
  return "rgba(255,82,82,0.85)";
}

const HeatmapChart = ({
  title = "Factor × Sector IC Heatmap",
  subtitle = "Rolling 20-day IC",
  rows = FACTORS,
  cols = SECTORS,
  values,
  valueFormatter = (value) => value.toFixed(2),
  tooltipFormatter = (value) => value.toFixed(3),
}: HeatmapChartProps) => {
  const data = useMemo(() => {
    const hasValidShape =
      Array.isArray(values) &&
      values.length === rows.length &&
      values.every((row) => Array.isArray(row) && row.length === cols.length);

    return hasValidShape ? values : generateHeatmap(rows.length, cols.length);
  }, [cols, rows, values]);

  const maxAbsValue = useMemo(() => {
    const flattened = data.flat();
    const maxAbs = flattened.reduce((current, value) => Math.max(current, Math.abs(value)), 0);
    return maxAbs || 1;
  }, [data]);

  return (
    <div data-cmp="HeatmapChart" className="bg-card rounded-xl border border-border p-5">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-semibold text-foreground">{title}</h3>
        <span className="text-xs text-muted-foreground">{subtitle}</span>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full border-separate" style={{ borderSpacing: "2px" }}>
          <thead>
            <tr>
              <th className="w-20" />
              {cols.map((col) => (
                <th key={col} className="text-xs font-medium text-muted-foreground pb-2 text-center min-w-14">
                  {col}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, ri) => (
              <tr key={row}>
                <td className="text-xs font-medium text-muted-foreground pr-3 text-right whitespace-nowrap">
                  {row}
                </td>
                {cols.map((col, ci) => {
                  const v = data[ri][ci];
                  const normalizedValue = v / maxAbsValue;
                  return (
                    <td key={col} className="h-9 rounded text-center transition-all duration-200 hover:opacity-80 cursor-default"
                      style={{ backgroundColor: valueToColor(normalizedValue) }}
                      title={`${row} × ${col}: ${tooltipFormatter(v)}`}
                    >
                      <span className="text-xs font-mono font-semibold text-foreground">
                        {valueFormatter(v)}
                      </span>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Legend */}
      <div className="flex items-center gap-2 mt-4 justify-end">
        <span className="text-xs text-muted-foreground">Low IC</span>
        <div className="flex gap-0.5">
          {["rgba(255,82,82,0.85)","rgba(255,82,82,0.4)","rgba(96,123,150,0.3)","rgba(0,200,5,0.4)","rgba(0,200,5,0.85)"].map((c, i) => (
            <div key={i} className="w-5 h-3 rounded-sm" style={{ backgroundColor: c }} />
          ))}
        </div>
        <span className="text-xs text-muted-foreground">High IC</span>
      </div>
    </div>
  );
};

export default HeatmapChart;
