import { Component, type ErrorInfo, type ReactNode } from "react";
import { AlertTriangle, RefreshCw } from "lucide-react";
import { withTranslation, type WithTranslation } from "react-i18next";

interface ErrorBoundaryProps extends WithTranslation {
  children: ReactNode;
  /** Reset the boundary when this key changes (e.g. active page). */
  resetKey?: string;
}

interface ErrorBoundaryState {
  error: Error | null;
}

class ErrorBoundaryBase extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  state: ErrorBoundaryState = { error: null };

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    console.error("[ErrorBoundary]", error, info.componentStack);
  }

  componentDidUpdate(prevProps: ErrorBoundaryProps): void {
    if (this.state.error !== null && prevProps.resetKey !== this.props.resetKey) {
      this.setState({ error: null });
    }
  }

  render() {
    const { t } = this.props;
    if (this.state.error !== null) {
      return (
        <div className="flex-1 flex flex-col items-center justify-center p-8 text-center">
          <AlertTriangle size={36} className="text-bear opacity-60 mb-4" />
          <h3 className="text-sm font-bold text-foreground mb-1">
            {t("errorBoundary.title", "Something went wrong on this page")}
          </h3>
          <p className="text-xs text-muted-foreground max-w-md mb-1 font-mono">
            {this.state.error.message}
          </p>
          <p className="text-[10px] text-muted-foreground/70 max-w-md mb-5">
            {t("errorBoundary.detail", "The rest of the app is unaffected. Try reloading this page.")}
          </p>
          <button
            onClick={() => this.setState({ error: null })}
            className="flex items-center gap-2 px-4 py-2 rounded-lg bg-muted hover:bg-accent text-xs font-bold text-foreground transition-colors"
          >
            <RefreshCw size={13} />
            {t("errorBoundary.retry", "Reload page")}
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}

const ErrorBoundary = withTranslation()(ErrorBoundaryBase);
export default ErrorBoundary;
