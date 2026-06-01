import { create } from 'zustand'
import { api } from '@/lib/api'
import type { UserProfile, LoginResponse } from '@/types/auth'

interface AuthState {
  token: string | null
  user: UserProfile | null
  isAuthenticated: boolean
  isAdmin: boolean
  isEditor: boolean
  login: (username: string, password: string) => Promise<void>
  logout: () => void
  fetchProfile: () => Promise<void>
  init: () => Promise<void>
}

export const useAuthStore = create<AuthState>((set, get) => ({
  token: typeof window !== 'undefined' ? localStorage.getItem('token') : null,
  user: null,
  isAuthenticated: false,
  isAdmin: false,
  isEditor: false,

  login: async (username, password) => {
    const res = await api.post<LoginResponse>('/auth/login', { username, password })
    localStorage.setItem('token', res.token)
    set({ token: res.token, isAuthenticated: true })
    await get().fetchProfile()
  },

  logout: () => {
    localStorage.removeItem('token')
    set({ token: null, user: null, isAuthenticated: false, isAdmin: false, isEditor: false })
  },

  fetchProfile: async () => {
    try {
      const user = await api.get<UserProfile>('/auth/me')
      set({
        user,
        isAuthenticated: true,
        isAdmin: user.role === 'admin',
        isEditor: user.role === 'editor' || user.role === 'admin',
      })
    } catch {
      get().logout()
    }
  },

  init: async () => {
    const token = localStorage.getItem('token')
    if (token) {
      set({ token, isAuthenticated: true })
      await get().fetchProfile()
    }
  },
}))
