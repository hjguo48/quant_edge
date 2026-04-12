interface FetchOptions {
  retries?: number;
  timeout?: number;
}

export async function fetchApi<T>(
  path: string,
  options: FetchOptions = {},
): Promise<T> {
  const { retries = 2, timeout = 15000 } = options;
  let lastError: Error | null = null;

  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), timeout);

      const response = await fetch(path, { 
        signal: controller.signal 
      });
      
      clearTimeout(timeoutId);

      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        throw new Error(body?.detail || `Request failed (${response.status})`);
      }

      return response.json();
    } catch (err) {
      lastError = err instanceof Error ? err : new Error(String(err));
      
      const isNetworkError = err instanceof TypeError;
      const isAbortError = err instanceof DOMException && err.name === 'AbortError';

      // Only retry on network errors or timeouts
      if (isNetworkError || isAbortError) {
        if (attempt < retries) {
          // Exponential backoff: 300ms, 600ms
          await new Promise(r => setTimeout(r, 300 * (attempt + 1)));
          continue;
        }
      }
      
      // For HTTP errors or if we've exhausted retries, break and throw
      break;
    }
  }

  throw lastError ?? new Error('Request failed');
}
