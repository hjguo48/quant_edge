import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import Index from "./pages/Index";

// fetchApi (hooks/useApi.ts) 已经处理 network/timeout 重试 (3 次内部重试),
// 所以 React Query 的 retry 设 0 避免 4×4 = 16 次叠加放大。
// 真正需要重试的 HTTP 错误 (5xx) fetchApi 不重试，但单次失败立即报错对 UX 更直观。
// 关掉 focus/reconnect 重新拉取,避免切 tab 时 query 风暴。
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 0,
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
