export interface GreyscaleHorizonWeek {
  status: string;
  portfolio_return: number | null;
  spy_return: number | null;
  excess: number | null;
  tickers_used: number;
  tickers_missing: number;
  horizon_end_date: string | null;
}

export interface GreyscaleWeeklyCurvePoint {
  signal_date: string;
  weekly_return: number | null;
  weekly_spy: number | null;
  weekly_excess: number | null;
  cumulative_return: number | null;
  cumulative_spy: number | null;
  cumulative_excess: number | null;
}

export interface GreyscaleHorizonCumulative {
  return: number | null;
  spy_return: number | null;
  excess: number | null;
  max_drawdown: number | null;
  weeks_realized: number;
  winrate_vs_spy: number | null;
  weekly_curve: GreyscaleWeeklyCurvePoint[];
}

export interface GreyscalePerWeek {
  week_number: number;
  signal_date: string | null;
  horizons: Record<string, GreyscaleHorizonWeek>;
}

export interface GreyscalePerformanceResponse {
  as_of_utc: string | null;
  today: string | null;
  benchmark: string;
  horizons_supported: number[];
  per_week: GreyscalePerWeek[];
  cumulative: Record<string, GreyscaleHorizonCumulative>;
}

export const GREYSCALE_HORIZONS = ["1d", "5d", "20d", "60d"] as const;
export type GreyscaleHorizonKey = typeof GREYSCALE_HORIZONS[number];
