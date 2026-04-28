export interface GreyscaleHeartbeat {
  status: string | null;
  bundle_version: string | null;
  signal_date: string | null;
  generated_at_utc: string | null;
  layer3_enforcement_mode: boolean | null;
  ticker_count: number | null;
  actual_holding_count: number | null;
  shadow_holding_count: number | null;
  shadow_cvar_triggered: boolean | null;
  layer1_pass: boolean | null;
  layer2_pass: boolean | null;
  layer3_pass: boolean | null;
  shadow_layer3_pass: boolean | null;
  layer4_pass: boolean | null;
  gate_status: string | null;
  matured_weeks: number | null;
}

export interface GreyscaleWeekSummary {
  week_number: number;
  signal_date: string | null;
  holding_count: number | null;
  turnover: number | null;
  layer1_pass: boolean | null;
  layer2_pass: boolean | null;
  layer3_pass: boolean | null;
  layer4_pass: boolean | null;
  weight_source: string | null;
  realized_ic_mean: number | null;
}

export interface GreyscaleGateCheck {
  threshold: string | null;
  value: number | null;
  passed: boolean | null;
  skipped_reason: string | null;
}

export interface GreyscaleGate {
  gate_rule: string | null;
  gate_status: string | null;
  matured_weeks: number;
  required_weeks: number;
  reports_seen: number;
  layer12_halt_count: number;
  mean_live_ic: number | null;
  mean_pairwise_rank_correlation: number | null;
  mean_turnover: number | null;
  positive_live_ic_weeks: number;
  rolling_live_ic_std: number | null;
  checks: Record<string, GreyscaleGateCheck>;
}

export interface GreyscaleShadowReduceEntry {
  ticker: string;
  raw_weight: number | null;
  shadow_weight: number | null;
}

export interface GreyscaleShadowDiagnostics {
  enforcement_mode: boolean | null;
  shadow_holding_count: number | null;
  shadow_gross_exposure: number | null;
  shadow_cvar_99: number | null;
  shadow_cash_weight: number | null;
  shadow_turnover_vs_previous: number | null;
  cvar_triggered: boolean | null;
  cvar_haircut_rounds: number | null;
  tickers_layer3_would_remove: string[];
  tickers_layer3_would_reduce: GreyscaleShadowReduceEntry[];
  warnings: string[];
  audit_trail: Record<string, unknown>[];
}

export interface GreyscaleLayer1Diagnostics {
  warning_triggered: boolean;
  warning_name: string | null;
  warn_threshold: number;
  latest_trade_date: string | null;
  max_null_rate: number;
  features_over_threshold: Record<string, number>;
  per_feature_null_rates: Record<string, number>;
}

export interface GreyscaleMonitorResponse {
  heartbeat: GreyscaleHeartbeat | null;
  weeks: GreyscaleWeekSummary[];
  gate: GreyscaleGate | null;
  shadow_diagnostics: GreyscaleShadowDiagnostics | null;
  layer1_diagnostics: GreyscaleLayer1Diagnostics | null;
}
