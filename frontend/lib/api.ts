const API_BASE = process.env.NEXT_PUBLIC_API_BASE || '/api/v1'

interface ApiResponse<T = any> {
  code: number
  data: T
  message: string
}

class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message)
    this.name = 'ApiError'
  }
}

function getToken(): string | null {
  if (typeof window === 'undefined') return null
  return localStorage.getItem('token')
}

async function request<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
  const token = getToken()
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    ...(options.headers as Record<string, string>),
  }
  if (token) headers['Authorization'] = `Bearer ${token}`

  const res = await fetch(`${API_BASE}${endpoint}`, {
    ...options,
    headers,
  })

  if (res.status === 401) {
    if (typeof window !== 'undefined') {
      localStorage.removeItem('token')
      window.location.href = '/login'
    }
    throw new ApiError(401, '登录已过期')
  }

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: '请求失败' }))
    throw new ApiError(res.status, err.detail || '请求失败')
  }

  const json: ApiResponse<T> = await res.json()
  if (json.code !== 0) throw new ApiError(0, json.message)
  return json.data
}

export const api = {
  get: <T>(endpoint: string) => request<T>(endpoint),

  post: <T>(endpoint: string, data?: any) =>
    request<T>(endpoint, { method: 'POST', body: data ? JSON.stringify(data) : undefined }),

  put: <T>(endpoint: string, data?: any) =>
    request<T>(endpoint, { method: 'PUT', body: data ? JSON.stringify(data) : undefined }),

  delete: <T>(endpoint: string) => request<T>(endpoint, { method: 'DELETE' }),

  upload: <T>(endpoint: string, formData: FormData) => {
    const token = getToken()
    const headers: Record<string, string> = {}
    if (token) headers['Authorization'] = `Bearer ${token}`
    return fetch(`${API_BASE}${endpoint}`, {
      method: 'POST',
      headers,
      body: formData,
    }).then(async (res) => {
      if (!res.ok) throw new ApiError(res.status, '上传失败')
      const json: ApiResponse<T> = await res.json()
      if (json.code !== 0) throw new ApiError(0, json.message)
      return json.data
    })
  },
}
