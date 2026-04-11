import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import Index from "./pages/Index";

const queryClient = new QueryClient();

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
