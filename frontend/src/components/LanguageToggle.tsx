import { useTranslation } from "react-i18next";
import { Languages } from "lucide-react";
import type { SupportedLanguage } from "../i18n/config";

const NEXT_LABEL: Record<SupportedLanguage, string> = {
  en: "中",
  zh: "EN",
};

const LanguageToggle = () => {
  const { i18n, t } = useTranslation();
  const current = (i18n.resolvedLanguage ?? i18n.language ?? "en").split("-")[0] as SupportedLanguage;
  const nextLang: SupportedLanguage = current === "en" ? "zh" : "en";

  const toggle = () => {
    void i18n.changeLanguage(nextLang);
  };

  return (
    <button
      onClick={toggle}
      title={t("common.language")}
      aria-label={t("common.language")}
      className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg bg-muted/40 border border-border/60 hover:bg-accent/40 transition-colors text-[10px] font-bold uppercase tracking-widest text-muted-foreground hover:text-foreground"
    >
      <Languages size={12} />
      <span className="font-mono text-foreground">{NEXT_LABEL[current] ?? "EN"}</span>
    </button>
  );
};

export default LanguageToggle;
