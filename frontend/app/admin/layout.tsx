'use client'

import { useEffect, useState } from 'react'
import { useRouter } from 'next/navigation'
import { useAuthStore } from '@/stores/auth'
import { AdminSidebar } from '@/components/admin/admin-sidebar'
import { Button } from '@/components/ui/button'
import { LogOut } from 'lucide-react'

export default function AdminLayout({ children }: { children: React.ReactNode }) {
  const router = useRouter()
  const { user, init, logout } = useAuthStore()
  const [initialized, setInitialized] = useState(false)

  useEffect(() => {
    const initApp = async () => {
      await init()
      setInitialized(true)
    }
    initApp()
  }, [])

  useEffect(() => {
    if (initialized && !useAuthStore.getState().isAuthenticated) {
      router.push('/login')
    }
  }, [initialized])

  if (!initialized) return null

  return (
    <div className="flex h-screen overflow-hidden">
      <AdminSidebar userRole={user?.role} />
      <div className="flex flex-1 flex-col overflow-hidden">
        <header className="flex h-14 items-center justify-between border-b bg-white px-6">
          <h1 className="text-lg font-semibold text-slate-800">管理后台</h1>
          <div className="flex items-center gap-4">
            <span className="text-sm text-slate-500">{user?.username} ({user?.role})</span>
            <Button variant="ghost" size="sm" onClick={() => router.push('/chat')}>
              返回聊天
            </Button>
            <Button variant="ghost" size="icon" onClick={() => { logout(); router.push('/login') }}>
              <LogOut className="h-4 w-4" />
            </Button>
          </div>
        </header>
        <main className="flex-1 overflow-y-auto p-6">{children}</main>
      </div>
    </div>
  )
}
