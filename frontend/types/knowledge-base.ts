export interface KB {
  id: number
  name: string
  description: string
  kb_type: string
  access_level: string
  allowed_departments: string
  allowed_users: string
  created_at: string
  updated_at: string
}

export interface KBConfig {
  kb_id: number
  chunk_strategy: string
  chunk_size: number
  chunk_overlap: number
  retrieval_mode: string
  top_k: number
  min_score: number
  rerank_enabled: number
  vector_weight: number
  bm25_weight: number
  prompt_template: string
}

export interface KBCreate {
  name: string
  description?: string
  kb_type?: string
  access_level?: string
  allowed_departments?: string[]
  allowed_users?: number[]
  chunk_strategy?: string
  chunk_size?: number
  chunk_overlap?: number
  retrieval_mode?: string
  top_k?: number
  vector_weight?: number
  bm25_weight?: number
}

export interface KBDetail {
  kb: KB
  config: KBConfig
}
