import type { Source } from '@/types/chat'

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || '/api/v1'

export interface SSECallbacks {
  onSources?: (sources: Source[]) => void
  onToken?: (token: string) => void
  onDone?: (traceId: string) => void
  onError?: (error: Error) => void
}

export function createSSEStream(
  payload: { kb_id: number; query: string; history: { role: string; content: string }[] },
  callbacks: SSECallbacks,
  signal?: AbortSignal
): () => void {
  const token = typeof window !== 'undefined' ? localStorage.getItem('token') : null
  let aborted = false

  const run = async () => {
    try {
      const response = await fetch(`${API_BASE}/chat/ask/stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
        },
        body: JSON.stringify(payload),
        signal,
      })

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`)
      }

      const reader = response.body!.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      while (!aborted) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n\n')
        buffer = lines.pop() || ''

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue
          try {
            const json = JSON.parse(line.slice(6))
            if (json.type === 'sources') callbacks.onSources?.(json.data)
            else if (json.type === 'token') callbacks.onToken?.(json.data)
            else if (json.type === 'done') callbacks.onDone?.(json.trace_id)
          } catch {
            // skip malformed JSON
          }
        }
      }
    } catch (e: any) {
      if (e.name !== 'AbortError') {
        callbacks.onError?.(e)
      }
    }
  }

  run()

  return () => {
    aborted = true
  }
}
