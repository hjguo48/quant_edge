import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
} from "recharts";

interface HistoryPoint {
  week: number;
  signal_date: string;
  score: number;
  rank: number;
  total: number;
}

interface SignalHistoryProps {
  history: HistoryPoint[];
  height?: number;
}

const formatDate = (dateStr: string) => {
  const date = new Date(dateStr);
  return `${(date.getMonth() + 1).toString().padStart(2, '0')}-${date.getDate().toString().padStart(2, '0')}`;
};

const SignalHistory = ({ history, height = 300 }: SignalHistoryProps) => {
  const data = [...history].sort((a, b) => new Date(a.signal_date).getTime() - new Date(b.signal_date).getTime());

  return (
    <div className="w-full" style={{ height }}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 20, right: 30, left: 10, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="rgba(255,255,255,0.05)" />
          <XAxis
            dataKey="signal_date"
            tick={{ fill: "#607B96", fontSize: 10 }}
            tickFormatter={formatDate}
            axisLine={false}
            tickLine={false}
          />
          <YAxis
            tick={{ fill: "#607B96", fontSize: 10 }}
            axisLine={false}
            tickLine={false}
            domain={['auto', 'auto']}
          />
          <Tooltip
            content={({ active, payload }) => {
              if (active && payload && payload.length) {
                const point = payload[0].payload as HistoryPoint;
                return (
                  <div className="bg-popover border border-border rounded-lg px-3 py-2 shadow-custom">
                    <p className="text-xs font-semibold text-foreground mb-1">{point.signal_date}</p>
                    <div className="space-y-1">
                      <p className="text-xs text-muted-foreground flex items-center justify-between gap-4">
                        Score: <span className={`font-mono font-bold ${point.score >= 0 ? "text-bull" : "text-bear"}`}>{point.score.toFixed(4)}</span>
                      </p>
                      <p className="text-xs text-muted-foreground flex items-center justify-between gap-4">
                        Rank: <span className="text-foreground font-mono font-bold">#{point.rank} / {point.total}</span>
                      </p>
                    </div>
                  </div>
                );
              }
              return null;
            }}
          />
          <Line
            type="monotone"
            dataKey="score"
            stroke="#3B82F6"
            strokeWidth={2}
            dot={{ r: 4, fill: "#3B82F6", strokeWidth: 0 }}
            activeDot={{ r: 6, fill: "#3B82F6", strokeWidth: 0 }}
            animationDuration={1500}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

export default SignalHistory;
