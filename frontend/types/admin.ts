export interface AdminStats {
  knowledge_bases: number
  documents: number
  chunks: number
  total_queries: number
  feedback_up: number
  feedback_down: number
}

export interface ConfigUpdate {
  chunk_strategy?: string
  chunk_size?: number
  chunk_overlap?: number
  retrieval_mode?: string
  top_k?: number
  min_score?: number
  rerank_enabled?: boolean
  vector_weight?: number
  bm25_weight?: number
  prompt_template?: string
}

export interface AuditLog {
  id: number
  user_id: number
  username: string
  kb_id: number
  query: string
  answer: string
  feedback: number
  feedback_text: string
  trace_id: string
  created_at: string
}

export interface EvalData {
  id: number
  kb_id: number
  question: string
  reference_answer: string
  relevant_doc_ids: string
}

export interface EvalDataCreate {
  kb_id: number
  question: string
  reference_answer?: string
  relevant_doc_ids?: number[]
}

export interface EvalReport {
  kb_id: number
  retrieval: {
    'avg_recall@5': number
    'avg_recall@10': number
    avg_mrr: number
    details: any[]
  }
  generation: any[]
}
