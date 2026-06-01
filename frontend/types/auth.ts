export interface LoginRequest {
  username: string
  password: string
}

export interface LoginResponse {
  token: string
  username: string
  role: string
}

export interface UserProfile {
  id: number
  username: string
  role: 'admin' | 'editor' | 'viewer'
  departments: string[]
}

export interface RegisterRequest {
  username: string
  password: string
  role?: string
  departments?: string[]
}
