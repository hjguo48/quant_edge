import { useState } from "react";
import { useTranslation } from "react-i18next";
import {
  LayoutDashboard,
  Zap,
  FileText,
  Briefcase,
  FlaskConical,
  Eye,
  ChevronRight,
  TrendingUp,
  Bell,
  Settings,
} from "lucide-react";

interface NavItem {
  id: string;
  i18nKey: string;
  iconName: string;
  badge?: string;
}

interface SidebarProps {
  activePage?: string;
  onNavigate?: (page: string) => void;
}

const navItems: NavItem[] = [
  { id: "dashboard", i18nKey: "sidebar.dashboard", iconName: "dashboard" },
  { id: "signals", i18nKey: "sidebar.signals", iconName: "zap", badge: "12" },
  { id: "portfolio", i18nKey: "sidebar.portfolio", iconName: "briefcase" },
  { id: "backtest", i18nKey: "sidebar.backtest", iconName: "flask" },
  { id: "greyscale", i18nKey: "sidebar.greyscale", iconName: "eye" },
];

const iconMap: Record<string, React.ElementType> = {
  dashboard: LayoutDashboard,
  zap: Zap,
  file: FileText,
  briefcase: Briefcase,
  flask: FlaskConical,
  eye: Eye,
};

const Sidebar = ({
  activePage = "dashboard",
  onNavigate = () => {},
}: SidebarProps) => {
  const { t } = useTranslation();
  const [collapsed, setCollapsed] = useState(false);

  return (
    <div
      data-cmp="Sidebar"
      className={`flex flex-col h-screen bg-sidebar border-r border-border transition-all duration-300 ease-in-out ${collapsed ? "w-16" : "w-60"} flex-shrink-0`}
    >
      {/* Logo */}
      <div className="flex items-center justify-between px-4 py-5 border-b border-border">
        {!collapsed && (
          <div className="flex items-center gap-2">
            <div className="w-7 h-7 rounded-lg bg-primary flex items-center justify-center flex-shrink-0">
              <TrendingUp className="w-4 h-4 text-primary-foreground" />
            </div>
            <span className="text-sm font-bold text-foreground tracking-wide">QuantEdge</span>
          </div>
        )}
        {collapsed && (
          <div className="w-7 h-7 rounded-lg bg-primary flex items-center justify-center mx-auto">
            <TrendingUp className="w-4 h-4 text-primary-foreground" />
          </div>
        )}
        {!collapsed && (
          <button
            onClick={() => setCollapsed(true)}
            className="text-muted-foreground hover:text-foreground transition-colors p-0.5 rounded"
          >
            <ChevronRight className="w-4 h-4 rotate-180" />
          </button>
        )}
      </div>

      {collapsed && (
        <button
          onClick={() => setCollapsed(false)}
          className="mx-auto mt-2 text-muted-foreground hover:text-foreground transition-colors p-1.5 rounded"
        >
          <ChevronRight className="w-4 h-4" />
        </button>
      )}

      {/* Nav */}
      <nav className="flex-1 px-2 py-4 space-y-0.5 overflow-y-auto">
        {navItems.map((item) => {
          const Icon = iconMap[item.iconName];
          const isActive = activePage === item.id || (item.id === "signals" && activePage === "signal-detail");
          const label = t(item.i18nKey);
          return (
            <button
              key={item.id}
              onClick={() => onNavigate(item.id)}
              className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-all duration-200 ${
                isActive
                  ? "nav-item-active text-primary"
                  : "text-muted-foreground hover:text-foreground hover:bg-accent"
              }`}
              title={collapsed ? label : undefined}
            >
              <Icon className={`w-4.5 h-4.5 flex-shrink-0 ${collapsed ? "mx-auto" : ""}`} size={18} />
              {!collapsed && (
                <>
                  <span className="flex-1 text-left">{label}</span>
                  {item.badge && (
                    <span className="text-xs font-semibold px-1.5 py-0.5 rounded-full tag-bull">
                      {item.badge}
                    </span>
                  )}
                </>
              )}
            </button>
          );
        })}
      </nav>

      {/* Bottom */}
      <div className="px-2 pb-4 border-t border-border pt-4 space-y-0.5">
        {[
          { icon: Bell, key: "sidebar.alerts", fallback: "Alerts" },
          { icon: Settings, key: "sidebar.settings", fallback: "Settings" },
        ].map(({ icon: Icon, key, fallback }) => {
          const label = t(key, { defaultValue: fallback });
          return (
            <button
              key={key}
              className="w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium text-muted-foreground hover:text-foreground hover:bg-accent transition-all duration-200"
              title={collapsed ? label : undefined}
            >
              <Icon className={`flex-shrink-0 ${collapsed ? "mx-auto" : ""}`} size={18} />
              {!collapsed && <span>{label}</span>}
            </button>
          );
        })}

        {/* User Avatar */}
        <div className={`flex items-center gap-3 px-3 py-2.5 mt-2 ${collapsed ? "justify-center" : ""}`}>
          <div className="w-7 h-7 rounded-full bg-primary/20 border border-primary/30 flex items-center justify-center flex-shrink-0">
            <span className="text-xs font-bold text-primary">QE</span>
          </div>
          {!collapsed && (
            <div className="min-w-0">
              <div className="text-xs font-semibold text-foreground truncate">{t("sidebar.userName", { defaultValue: "Quant User" })}</div>
              <div className="text-xs text-muted-foreground truncate">{t("sidebar.userRole", { defaultValue: "Model Output Only" })}</div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Sidebar;
