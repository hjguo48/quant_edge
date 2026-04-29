import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import Index from "./pages/Index";

// fetchApi (hooks/useApi.ts) 内部已处理 network/timeout (3 次重试 ~1s 内),
// React Query 再加 1 次重试 (~1.5s 后) 来扛 API 重启 (~5s)。
// 总最坏: 4 (fetchApi 一轮) × 2 (RQ 0+1) = 8 次 HTTP — 比 retry:3 时的 16 次少一半。
// 关 focus/reconnect 拉取避免切 tab 风暴。
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      retryDelay: (attemptIndex) => Math.min(1500 * 2 ** attemptIndex, 5000),
      refetchOnWindowFocus: false,
      refetchOnReconnect: false,
      staleTime: 30_000,
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
