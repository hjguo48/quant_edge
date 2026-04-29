import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import Index from "./pages/Index";

// fetchApi (hooks/useApi.ts) 内部已处理 network/timeout (3 次重试 ~1s 内),
// React Query 再加 1 次重试 (~1.5s 后) 来扛 API 重启 (~5s)。
// 关键: refetchInterval-on-error — 失败状态下每 8s 自动重试,
// 让用户不需要手动硬刷新就能从 API 重启之类的瞬时故障中恢复。
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      retryDelay: (attemptIndex) => Math.min(1500 * 2 ** attemptIndex, 5000),
      refetchOnWindowFocus: false,
      refetchOnReconnect: false,
      staleTime: 30_000,
      // Auto-recover from cached errors — poll every 8s while query is errored.
      refetchInterval: (query) => (query.state.error ? 8_000 : false),
      refetchIntervalInBackground: false,
    },
  },
});

const App = () => (
  <QueryClientProvider client={queryClient}>
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Index />} />
        <Route path="/signals" element={<Index />} />
        <Route path="/signals/:ticker" element={<Index />} />
        <Route path="/portfolio" element={<Index />} />
        <Route path="/backtest" element={<Index />} />
        <Route path="/greyscale" element={<Index />} />
      </Routes>
    </BrowserRouter>
  </QueryClientProvider>
);

export default App;
