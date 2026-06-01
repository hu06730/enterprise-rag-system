'use client'

import { useState, useRef } from 'react'
import { Button } from '@/components/ui/button'
import { Textarea } from '@/components/ui/textarea'
import { Send, Square, Paperclip } from 'lucide-react'
import { useChatStore } from '@/stores/chat'
import { api } from '@/lib/api'
import { useAuthStore } from '@/stores/auth'

export function ChatInput() {
  const [query, setQuery] = useState('')
  const { isStreaming, sendMessage, stopGeneration, selectedKbId } = useChatStore()
  const fileInputRef = useRef<HTMLInputElement>(null)
  const user = useAuthStore((s) => s.user)
  const canUpload = user?.role === 'admin' || user?.role === 'editor'

  const handleSend = () => {
    const text = query.trim()
    if (!text || isStreaming) return
    sendMessage(text)
    setQuery('')
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file || !selectedKbId) return
    const formData = new FormData()
    formData.append('kb_id', String(selectedKbId))
    formData.append('file', file)
    formData.append('access_level', 'internal')
    await api.upload('/documents/upload', formData)
    e.target.value = ''
  }

  return (
    <div className="border-t bg-white px-4 py-4">
      <div className="mx-auto max-w-3xl">
        <div className="flex items-end gap-2 rounded-2xl border bg-slate-50 p-2 shadow-sm focus-within:ring-2 focus-within:ring-blue-200">
          {canUpload && (
            <>
              <input
                ref={fileInputRef}
                type="file"
                className="hidden"
                accept=".pdf,.doc,.docx,.md,.txt"
                onChange={handleFileUpload}
              />
              <Button
                variant="ghost"
                size="icon"
                className="h-9 w-9 shrink-0 text-slate-400 hover:text-slate-600"
                onClick={() => fileInputRef.current?.click()}
                title="上传文档"
              >
                <Paperclip className="h-4 w-4" />
              </Button>
            </>
          )}

          <Textarea
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="输入你的问题..."
            className="min-h-[40px] max-h-[120px] resize-none border-0 bg-transparent p-2 text-sm shadow-none focus-visible:ring-0"
            rows={1}
          />

          {isStreaming ? (
            <Button
              variant="ghost"
              size="icon"
              className="h-9 w-9 shrink-0 text-red-400 hover:text-red-600"
              onClick={stopGeneration}
            >
              <Square className="h-4 w-4" />
            </Button>
          ) : (
            <Button
              size="icon"
              className="h-9 w-9 shrink-0 rounded-xl bg-blue-500 hover:bg-blue-600"
              onClick={handleSend}
              disabled={!query.trim()}
            >
              <Send className="h-4 w-4" />
            </Button>
          )}
        </div>
        <p className="mt-2 text-center text-xs text-slate-400">
          按 Enter 发送，Shift + Enter 换行
        </p>
      </div>
    </div>
  )
}
