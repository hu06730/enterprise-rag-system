'use client'

import { useEffect, useState } from 'react'
import { useRouter } from 'next/navigation'
import { Button } from '@/components/ui/button'
import { Settings, Menu } from 'lucide-react'
import { useAuthStore } from '@/stores/auth'
import { useChatStore } from '@/stores/chat'
import { ChatSidebar } from '@/components/chat/chat-sidebar'
import { ChatMessages } from '@/components/chat/chat-messages'
import { ChatInput } from '@/components/chat/chat-input'

export default function ChatPage() {
  const router = useRouter()
  const { isAuthenticated, user, init } = useAuthStore()
  const { fetchKnowledgeBases } = useChatStore()
  const [sidebarOpen, setSidebarOpen] = useState(true)
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

  useEffect(() => {
    if (isAuthenticated) fetchKnowledgeBases()
  }, [isAuthenticated])

  if (!initialized) return null

  return (
    <div className="flex h-screen overflow-hidden">
      {/* Sidebar */}
      <div className={`${sidebarOpen ? 'w-[280px]' : 'w-0'} transition-all duration-300 overflow-hidden`}>
        <ChatSidebar />
      </div>

      {/* Main Area */}
      <div className="flex flex-1 flex-col">
        {/* Header */}
        <header className="flex h-14 items-center justify-between border-b bg-white px-4">
          <div className="flex items-center gap-3">
            <Button
              variant="ghost"
              size="icon"
              className="h-9 w-9 md:hidden"
              onClick={() => setSidebarOpen(!sidebarOpen)}
            >
              <Menu className="h-5 w-5" />
            </Button>
            <Button
              variant="ghost"
              size="icon"
              className="h-9 w-9"
              onClick={() => setSidebarOpen(!sidebarOpen)}
            >
              <Menu className="h-5 w-5" />
            </Button>
            <h1 className="text-lg font-semibold text-slate-800">RAG 知识库问答</h1>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-sm text-slate-500">{user?.username}</span>
            {user?.role === 'admin' && (
              <Button variant="ghost" size="sm" onClick={() => router.push('/admin/dashboard')}>
                <Settings className="mr-1 h-4 w-4" />
                管理
              </Button>
            )}
          </div>
        </header>

        {/* Messages */}
        <ChatMessages />

        {/* Input */}
        <ChatInput />
      </div>
    </div>
  )
}
