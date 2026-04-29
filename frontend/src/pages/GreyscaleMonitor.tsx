import { useTranslation } from "react-i18next";
import { useQuery } from "@tanstack/react-query";
import { Eye, Clock, ShieldCheck, Activity, AlertTriangle, CheckCircle2, XCircle, Hourglass } from "lucide-react";
import { fetchApi } from "../hooks/useApi";
import type {
  GreyscaleMonitorResponse,
  GreyscaleWeekSummary,
} from "../types/greyscaleMonitor";

const fmtPct = (v: number | null | undefined, digits = 2) =>
  v == null ? "—" : `${(v * 100).toFixed(digits)}%`;

const fmtFloat = (v: number | null | undefined, digits = 4) =>
  v == null ? "—" : v.toFixed(digits);

interface LayerPillProps {
  label: string;
  pass: boolean | null;
  shadow?: boolean;
}

const LayerPill = ({ label, pass, shadow }: LayerPillProps) => {
  let color: string;
  let icon: React.ReactNode;
  if (pass === true) {
    color = "bg-bull/10 text-bull border-bull/20";
    icon = <CheckCircle2 size={11} />;
  } else if (pass === false) {
    color = "bg-bear/10 text-bear border-bear/20";
    icon = <XCircle size={11} />;
  } else {
    color = "bg-muted/50 text-muted-foreground border-border";
    icon = <Hourglass size={11} />;
  }
  if (shadow) {
    color = "bg-amber-500/10 text-amber-500 border-amber-500/20";
  }
  return (
    <span
      className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider border ${color}`}
    >
      {icon}
      <span>{label}</span>
    </span>
  );
};

const StatusDot = ({ value }: { value: boolean | null }) => {
  if (value === true) return <span className="inline-block w-1.5 h-1.5 rounded-full bg-bull" />;
  if (value === false) return <span className="inline-block w-1.5 h-1.5 rounded-full bg-bear" />;
  return <span className="inline-block w-1.5 h-1.5 rounded-full bg-muted-foreground/50" />;
};

const GateStatusBadge = ({ status }: { status: string | null }) => {
  const { t } = useTranslation();
  const normalized = (status ?? "PENDING").toUpperCase();
  const labelKey =
    normalized === "PENDING" ? "greyscale.gate.pending" :
    normalized === "PROVISIONAL" ? "greyscale.gate.provisional" :
    normalized === "MATURE" ? "greyscale.gate.mature" :
    normalized === "PASS" ? "greyscale.gate.passed" :
    normalized === "FAIL" ? "greyscale.gate.failed" :
    "greyscale.gate.pending";
  const color =
    normalized === "PASS" ? "bg-bull/15 text-bull border-bull/30" :
    normalized === "FAIL" ? "bg-bear/15 text-bear border-bear/30" :
    "bg-muted/50 text-muted-foreground border-border";
  return (
    <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-[10px] font-black uppercase tracking-wider border ${color}`}>
      {t(labelKey)}
    </span>
  );
};

const WeeksTable = ({ weeks }: { weeks: GreyscaleWeekSummary[] }) => {
  const { t } = useTranslation();
  if (weeks.length === 0) {
    return (
      <div className="p-8 text-center text-xs text-muted-foreground">
        {t("common.noData")}
      </div>
    );
  }
  return (
    <div>
      <div className="flex items-center px-4 py-2 border-b border-border bg-muted/10 text-[10px] font-black text-muted-foreground uppercase tracking-[0.15em]">
        <div className="w-24">#</div>
        <div className="w-28">{t("greyscale.weeks.signalDate")}</div>
        <div className="w-20 text-right">{t("greyscale.weeks.tickers")}</div>
        <div className="flex-1 text-center">{t("greyscale.weeks.layerStatus")}</div>
        <div className="w-24 text-right">IC</div>
      </div>
      {[...weeks].reverse().map((w) => (
        <div
          key={w.week_number}
          className="flex items-center px-4 py-3 border-b border-border/50 last:border-0 hover:bg-accent/40 transition-colors text-xs"
        >
          <div className="w-24 text-foreground font-bold">{t("greyscale.weeks.weekNumber", { n: w.week_number })}</div>
          <div className="w-28 text-foreground font-mono">{w.signal_date ?? "—"}</div>
          <div className="w-20 text-right font-mono text-foreground">{w.holding_count ?? "—"}</div>
          <div className="flex-1 flex flex-wrap gap-1 justify-center">
            <LayerPill label="L1" pass={w.layer1_pass} />
            <LayerPill label="L2" pass={w.layer2_pass} />
            <LayerPill label="L3" pass={w.layer3_pass} />
            <LayerPill label="L4" pass={w.layer4_pass} />
          </div>
          <div className="w-24 text-right font-mono text-muted-foreground">
            {w.realized_ic_mean == null ? "—" : w.realized_ic_mean.toFixed(4)}
          </div>
        </div>
      ))}
    </div>
  );
};

const GreyscaleMonitor = () => {
  const { t, i18n } = useTranslation();
  const lang = (i18n.resolvedLanguage ?? i18n.language ?? "en").split("-")[0];
  const localeTag = lang === "zh" ? "zh-CN" : "en-US";

  const monitorQuery = useQuery<GreyscaleMonitorResponse>({
    queryKey: ["greyscaleMonitor"],
    queryFn: () => fetchApi<GreyscaleMonitorResponse>("/api/greyscale/monitor"),
    retry: false,
    staleTime: 30_000,
  });

  if (monitorQuery.isLoading) {
    return (
      <div className="flex-1 overflow-y-auto p-6 space-y-5">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-primary/10 border border-primary/20 flex items-center justify-center">
            <Eye size={18} className="text-primary animate-pulse" />
          </div>
          <div>
            <h2 className="text-xl font-bold text-foreground">{t("greyscale.title")}</h2>
            <p className="text-xs text-muted-foreground mt-0.5">{t("common.loading")}</p>
          </div>
        </div>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
          {[1, 2, 3, 4].map((i) => (
            <div key={i} className="h-40 bg-card rounded-xl border border-border animate-pulse" />
          ))}
        </div>
      </div>
    );
  }

  if (monitorQuery.isError || !monitorQuery.data) {
    return (
      <div className="flex-1 overflow-y-auto p-6">
        <div className="flex flex-col items-center justify-center min-h-[400px] text-center text-xs text-muted-foreground">
          <AlertTriangle size={36} className="text-muted-foreground/40 mb-3" />
          <p className="font-bold uppercase tracking-widest mb-1">{t("common.error")}</p>
          <p className="opacity-70 max-w-md">
            {(monitorQuery.error as Error | undefined)?.message ?? t("common.noData")}
          </p>
        </div>
      </div>
    );
  }

  const data = monitorQuery.data;
  const heartbeat = data.heartbeat;
  const gate = data.gate;
  const shadow = data.shadow_diagnostics;
  const layer1 = data.layer1_diagnostics;

  const generatedAtLabel = heartbeat?.generated_at_utc
    ? new Date(heartbeat.generated_at_utc).toLocaleString(localeTag, {
        year: "numeric",
        month: "short",
        day: "numeric",
        hour: "2-digit",
        minute: "2-digit",
      })
    : "—";

  return (
    <div className="flex-1 overflow-y-auto p-6 space-y-5">
      {/* Header */}
      <div className="flex items-center justify-between fade-in-up">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-primary/10 border border-primary/20 flex items-center justify-center">
            <Eye size={18} className="text-primary" />
          </div>
          <div>
            <h2 className="text-xl font-bold text-foreground">{t("greyscale.title")}</h2>
            <p className="text-xs text-muted-foreground mt-0.5">{t("greyscale.subtitle")}</p>
          </div>
        </div>
        <div className="flex items-center gap-3 text-xs text-muted-foreground">
          <Clock size={12} />
          <span>{generatedAtLabel}</span>
        </div>
      </div>

      {/* Heartbeat — latest run summary */}
      {heartbeat && (
        <div className="bg-card rounded-xl border border-border p-5 fade-in-up stagger-1">
          <div className="flex items-start justify-between mb-4">
            <div>
              <div className="flex items-center gap-2 mb-1">
                <Activity size={14} className="text-primary" />
                <h3 className="text-sm font-bold text-foreground">
                  {t("greyscale.weeks.signalDate")}: <span className="font-mono">{heartbeat.signal_date ?? "—"}</span>
                </h3>
              </div>
              <p className="text-[10px] text-muted-foreground font-medium">
                {heartbeat.bundle_version ?? "—"}
              </p>
            </div>
            <GateStatusBadge status={heartbeat.gate_status} />
          </div>

          <div className="grid grid-cols-4 gap-3 mb-4">
            <div className="bg-muted/20 rounded-lg p-3 border border-border/50">
              <p className="text-[10px] text-muted-foreground uppercase tracking-wider mb-1">
                {t("portfolio.stats.holdings")}
              </p>
              <p className="text-lg font-black font-mono text-foreground">
                {heartbeat.actual_holding_count ?? "—"}
              </p>
            </div>
            <div className="bg-muted/20 rounded-lg p-3 border border-border/50">
              <p className="text-[10px] text-muted-foreground uppercase tracking-wider mb-1">
                {t("greyscale.shadowDiagnostics.shadowHoldings")}
              </p>
              <p className="text-lg font-black font-mono text-amber-500">
                {heartbeat.shadow_holding_count ?? "—"}
              </p>
            </div>
            <div className="bg-muted/20 rounded-lg p-3 border border-border/50">
              <p className="text-[10px] text-muted-foreground uppercase tracking-wider mb-1">
                Universe
              </p>
              <p className="text-lg font-black font-mono text-foreground">
                {heartbeat.ticker_count ?? "—"}
              </p>
            </div>
            <div className="bg-muted/20 rounded-lg p-3 border border-border/50">
              <p className="text-[10px] text-muted-foreground uppercase tracking-wider mb-1">
                {t("greyscale.gate.weeksMatured")}
              </p>
              <p className="text-lg font-black font-mono text-foreground">
                {heartbeat.matured_weeks ?? 0} / {gate?.required_weeks ?? "—"}
              </p>
            </div>
          </div>

          <div className="flex flex-wrap gap-2">
            <LayerPill label={t("greyscale.layers.layer1")} pass={heartbeat.layer1_pass} />
            <LayerPill label={t("greyscale.layers.layer2")} pass={heartbeat.layer2_pass} />
            {heartbeat.layer3_enforcement_mode ? (
              <LayerPill label={t("greyscale.layers.layer3")} pass={heartbeat.layer3_pass} />
            ) : (
              <LayerPill
                label={`${t("greyscale.layers.layer3")} (${t("greyscale.layers.shadow")})`}
                pass={heartbeat.shadow_layer3_pass}
                shadow
              />
            )}
            <LayerPill label={t("greyscale.layers.layer4")} pass={heartbeat.layer4_pass} />
          </div>
        </div>
      )}

      {/* Two-column: Gate + Shadow */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        {/* Gate */}
        {gate && (
          <div className="bg-card rounded-xl border border-border p-5 fade-in-up stagger-2">
            <div className="flex items-start justify-between mb-3">
              <div>
                <h3 className="text-sm font-bold text-foreground flex items-center gap-2">
                  <ShieldCheck size={14} className="text-primary" />
                  {t("greyscale.gate.title")}
                </h3>
                <p className="text-[10px] text-muted-foreground mt-0.5 font-medium">
                  {gate.gate_rule ?? ""}
                </p>
              </div>
              <GateStatusBadge status={gate.gate_status} />
            </div>

            <div className="grid grid-cols-2 gap-3 mb-4">
              <div className="bg-muted/20 rounded-lg p-3 border border-border/50">
                <p className="text-[10px] text-muted-foreground uppercase tracking-wider mb-1">
                  {t("greyscale.gate.weeksMatured")}
                </p>
                <p className="text-base font-black font-mono text-foreground">
                  {gate.matured_weeks} / {gate.required_weeks}
                </p>
              </div>
              <div className="bg-muted/20 rounded-lg p-3 border border-border/50">
                <p className="text-[10px] text-muted-foreground uppercase tracking-wider mb-1">
                  Live IC (mean)
                </p>
                <p className="text-base font-black font-mono text-foreground">
                  {fmtFloat(gate.mean_live_ic)}
                </p>
              </div>
            </div>

            <div className="space-y-1.5">
              {Object.entries(gate.checks).map(([name, check]) => (
                <div key={name} className="flex items-center justify-between text-xs px-2 py-1.5 rounded bg-muted/10 border border-border/30">
                  <div className="flex items-center gap-2 min-w-0">
                    <StatusDot value={check.passed} />
                    <span className="text-foreground font-medium truncate">{name}</span>
                  </div>
                  <div className="flex items-center gap-3 text-muted-foreground font-mono text-[10px] flex-shrink-0">
                    <span>{check.threshold ?? "—"}</span>
                    <span className="text-foreground font-bold">
                      {check.value == null ? "—" : check.value.toFixed(3)}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Shadow Diagnostics */}
        {shadow && (
          <div className="bg-card rounded-xl border border-border p-5 fade-in-up stagger-3">
            <div className="mb-3">
              <h3 className="text-sm font-bold text-foreground flex items-center gap-2">
                <Activity size={14} className="text-amber-500" />
                {t("greyscale.shadowDiagnostics.title")}
              </h3>
              <p className="text-[10px] text-muted-foreground mt-0.5 font-medium">
                {t("greyscale.shadowDiagnostics.subtitle")}
              </p>
            </div>

            <div className="grid grid-cols-2 gap-3 mb-3">
              <div className="bg-muted/20 rounded-lg p-3 border border-border/50">
                <p className="text-[10px] text-muted-foreground uppercase tracking-wider mb-1">
                  {t("greyscale.shadowDiagnostics.shadowHoldings")}
                </p>
                <p className="text-base font-black font-mono text-amber-500">
                  {shadow.shadow_holding_count ?? "—"}
                </p>
              </div>
              <div className="bg-muted/20 rounded-lg p-3 border border-border/50">
                <p className="text-[10px] text-muted-foreground uppercase tracking-wider mb-1">
                  Shadow CVaR 99%
                </p>
                <p className="text-base font-black font-mono text-foreground">
                  {fmtPct(shadow.shadow_cvar_99)}
                </p>
              </div>
              <div className="bg-muted/20 rounded-lg p-3 border border-border/50">
                <p className="text-[10px] text-muted-foreground uppercase tracking-wider mb-1">
                  {t("greyscale.shadowDiagnostics.wouldRemove")}
                </p>
                <p className="text-base font-black font-mono text-foreground">
                  {shadow.tickers_layer3_would_remove.length}
                </p>
              </div>
              <div className="bg-muted/20 rounded-lg p-3 border border-border/50">
                <p className="text-[10px] text-muted-foreground uppercase tracking-wider mb-1">
                  {t("greyscale.shadowDiagnostics.wouldReduce")}
                </p>
                <p className="text-base font-black font-mono text-foreground">
                  {shadow.tickers_layer3_would_reduce.length}
                </p>
              </div>
            </div>

            {shadow.tickers_layer3_would_remove.length > 0 && (
              <div className="mb-3">
                <p className="text-[10px] text-muted-foreground uppercase tracking-wider mb-1.5 font-bold">
                  {t("greyscale.shadowDiagnostics.wouldRemove")}
                </p>
                <div className="flex flex-wrap gap-1.5">
                  {shadow.tickers_layer3_would_remove.slice(0, 12).map((tk) => (
                    <span
                      key={tk}
                      className="px-2 py-0.5 rounded text-[10px] font-bold font-mono bg-bear/10 text-bear border border-bear/20"
                    >
                      {tk}
                    </span>
                  ))}
                  {shadow.tickers_layer3_would_remove.length > 12 && (
                    <span className="px-2 py-0.5 rounded text-[10px] font-mono text-muted-foreground">
                      +{shadow.tickers_layer3_would_remove.length - 12}
                    </span>
                  )}
                </div>
              </div>
            )}

            {shadow.cvar_triggered && (
              <div className="flex items-center gap-2 text-[10px] text-amber-500 bg-amber-500/10 border border-amber-500/20 rounded px-2 py-1.5">
                <AlertTriangle size={11} />
                <span className="font-bold uppercase tracking-wider">
                  {t("greyscale.shadowDiagnostics.cvarTriggered")}
                </span>
                {shadow.cvar_haircut_rounds != null && (
                  <span className="font-mono">({shadow.cvar_haircut_rounds} rounds)</span>
                )}
              </div>
            )}
          </div>
        )}
      </div>

      {/* Layer 1 diagnostics */}
      {layer1 && (
        <div className="bg-card rounded-xl border border-border p-5 fade-in-up stagger-4">
          <div className="flex items-start justify-between mb-3">
            <div>
              <h3 className="text-sm font-bold text-foreground flex items-center gap-2">
                <Activity size={14} className={layer1.warning_triggered ? "text-amber-500" : "text-primary"} />
                {t("greyscale.layers.layer1")} · per-feature dropout
              </h3>
              <p className="text-[10px] text-muted-foreground mt-0.5 font-medium">
                trade_date {layer1.latest_trade_date ?? "—"} · threshold {fmtPct(layer1.warn_threshold, 0)}
              </p>
            </div>
            {layer1.warning_triggered ? (
              <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-md text-[10px] font-black uppercase tracking-wider bg-amber-500/10 text-amber-500 border border-amber-500/20">
                <AlertTriangle size={11} />
                {t("greyscale.layers.warning")}
              </span>
            ) : (
              <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-md text-[10px] font-black uppercase tracking-wider bg-bull/10 text-bull border border-bull/20">
                <CheckCircle2 size={11} />
                {t("greyscale.layers.pass")}
              </span>
            )}
          </div>
          {layer1.warning_triggered && Object.keys(layer1.features_over_threshold).length > 0 ? (
            <div className="space-y-1">
              {Object.entries(layer1.features_over_threshold).slice(0, 10).map(([name, rate]) => (
                <div key={name} className="flex items-center justify-between text-xs px-3 py-1.5 rounded bg-amber-500/5 border border-amber-500/15">
                  <span className="text-foreground font-mono">{name}</span>
                  <span className="text-amber-500 font-bold font-mono">{fmtPct(rate)}</span>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-xs text-muted-foreground">
              max null rate <span className="font-mono text-foreground">{fmtPct(layer1.max_null_rate)}</span> — under threshold
            </p>
          )}
        </div>
      )}

      {/* Weeks history */}
      <div className="bg-card rounded-xl border border-border overflow-hidden fade-in-up stagger-5">
        <div className="px-5 py-4 border-b border-border bg-muted/10">
          <h3 className="text-sm font-bold text-foreground">{t("greyscale.weeks.header")}</h3>
        </div>
        <WeeksTable weeks={data.weeks} />
      </div>
    </div>
  );
};

export default GreyscaleMonitor;
