import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import Index from "./pages/Index";

// WSL2 ↔ Windows 浏览器网络偶发 RST,
// 默认 retry: 3 (~ 7s 内 4 次尝试) 容忍单次抖动，
// 关掉 focus/reconnect 重新拉取,避免切 tab 时 query 风暴。
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 3,
      retryDelay: (attemptIndex) => Math.min(1000 * 2 ** attemptIndex, 5000),
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
