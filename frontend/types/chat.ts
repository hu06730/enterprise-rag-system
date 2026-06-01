export interface ChatRequest {
  kb_id: number
  query: string
  history: { role: string; content: string }[]
}

export interface Source {
  doc_title: string
  chunk_text: string
  score: number
}

export interface SSEEvent {
  type: 'sources' | 'token' | 'done'
  data?: any
  trace_id?: string
}

export interface ChatMessage {
  role: 'user' | 'assistant'
  content: string
  sources?: Source[]
  traceId?: string
  loading?: boolean
}

export interface FeedbackRequest {
  trace_id: string
  feedback: number
  feedback_text?: string
}

export interface Conversation {
  id: string
  title: string
  kbId: number
  messages: ChatMessage[]
  createdAt: number
}
