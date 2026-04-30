interface FetchOptions {
  retries?: number;
  timeout?: number;
}

/**
 * Error subclass that preserves the HTTP status code.
 * Lets callers do `if (err instanceof ApiError && err.status === 404)`
 * instead of fragile string-matching against the message.
 */
export class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

export async function fetchApi<T>(
  path: string,
  options: FetchOptions = {},
): Promise<T> {
  const { retries = 3, timeout = 15000 } = options;
  let lastError: Error | null = null;

  for (let attempt = 0; attempt <= retries; attempt++) {
    const controller = new AbortController();
    const timeoutId = window.setTimeout(() => controller.abort(), timeout);

    try {
      console.log(`[fetchApi] Attempt ${attempt + 1}/${retries + 1} for: ${path}`);
      const response = await fetch(path, {
        signal: controller.signal,
      });

      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        throw new ApiError(
          body?.detail || `Request failed (${response.status})`,
          response.status,
        );
      }

      return response.json();
    } catch (err) {
      lastError = err instanceof Error ? err : new Error(String(err));
      const isNetworkError = err instanceof TypeError;
      const isAbortError = err instanceof DOMException && err.name === "AbortError";

      // Only retry on network errors or timeouts.
      if (isNetworkError || isAbortError) {
        if (attempt < retries) {
          await new Promise((resolve) => window.setTimeout(resolve, 300 * (attempt + 1)));
          continue;
        }
      }

      break;
    } finally {
      window.clearTimeout(timeoutId);
    }
  }

  if (lastError) {
    console.error(`[fetchApi] Failed after ${retries} retries for path: ${path}. Error:`, lastError);
  }
  throw lastError ?? new Error("Request failed");
}
