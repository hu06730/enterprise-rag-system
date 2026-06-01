export interface DocItem {
  id: number
  kb_id: number
  filename: string
  file_path: string
  file_type: string
  status: 'pending' | 'processing' | 'completed' | 'failed'
  access_level: string
  created_at: string
}

export interface DocStatus {
  id: number
  filename: string
  status: string
}
