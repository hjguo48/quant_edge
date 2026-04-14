import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Cell,
  CartesianGrid,
} from "recharts";

interface ShapFeature {
  feature: string;
  shap_value: number;
}

interface ShapWaterfallProps {
  features: ShapFeature[];
  height?: number;
}

const formatFeatureName = (name: string) => {
  return name
    .split("_")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
};

const ShapWaterfall = ({ features, height = 400 }: ShapWaterfallProps) => {
  const data = features
    .slice(0, 15)
    .map((f) => ({
      name: formatFeatureName(f.feature),
      value: f.shap_value,
    }))
    .sort((a, b) => Math.abs(b.value) - Math.abs(a.value));

  return (
    <div className="w-full" style={{ height }}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart
          data={data}
          layout="vertical"
          margin={{ top: 5, right: 30, left: 40, bottom: 5 }}
        >
          <CartesianGrid strokeDasharray="3 3" horizontal={true} vertical={false} stroke="rgba(255,255,255,0.05)" />
          <XAxis type="number" hide />
          <YAxis
            type="category"
            dataKey="name"
            tick={{ fill: "#607B96", fontSize: 11 }}
            width={150}
            axisLine={false}
            tickLine={false}
          />
          <Tooltip
            cursor={{ fill: "rgba(255,255,255,0.03)" }}
            content={({ active, payload }) => {
              if (active && payload && payload.length) {
                const data = payload[0].payload;
                return (
                  <div className="bg-popover border border-border rounded-lg px-3 py-2 shadow-custom">
                    <p className="text-xs font-semibold text-foreground mb-1">{data.name}</p>
                    <p className={`text-xs font-mono font-bold ${data.value >= 0 ? "text-bull" : "text-bear"}`}>
                      {data.value > 0 ? "+" : ""}{data.value.toFixed(4)}
                    </p>
                  </div>
                );
              }
              return null;
            }}
          />
          <Bar dataKey="value" radius={[0, 4, 4, 0]}>
            {data.map((entry, index) => (
              <Cell
                key={`cell-${index}`}
                fill={entry.value >= 0 ? "#00C805" : "#FF5252"}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
};

export default ShapWaterfall;
