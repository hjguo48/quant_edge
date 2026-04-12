export async function fetchApi<T>(path: string): Promise<T> {
  const response = await fetch(path);
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body?.detail || 'Request failed');
  }
  return response.json();
}
