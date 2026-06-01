'use client'

import { Button } from '@/components/ui/button'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Separator } from '@/components/ui/separator'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Plus, MessageSquare, Trash2 } from 'lucide-react'
import { useChatStore } from '@/stores/chat'
import { cn } from '@/lib/utils'

export function ChatSidebar() {
  const {
    conversations,
    currentConversationId,
    knowledgeBases,
    selectedKbId,
    setSelectedKb,
    createConversation,
    switchConversation,
    deleteConversation,
  } = useChatStore()

  return (
    <div className="flex h-full w-[280px] flex-col border-r bg-slate-50">
      {/* KB Selector */}
      <div className="p-3">
        <Select value={selectedKbId?.toString()} onValueChange={(v) => setSelectedKb(Number(v))}>
          <SelectTrigger className="h-9 text-sm">
            <SelectValue placeholder="选择知识库" />
          </SelectTrigger>
          <SelectContent>
            {knowledgeBases.map((kb) => (
              <SelectItem key={kb.id} value={kb.id.toString()}>
                {kb.name}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      <Separator />

      {/* New Chat Button */}
      <div className="p-3">
        <Button
          variant="outline"
          className="w-full justify-start gap-2 text-sm"
          onClick={() => createConversation()}
        >
          <Plus className="h-4 w-4" />
          新对话
        </Button>
      </div>

      {/* Conversation List */}
      <ScrollArea className="flex-1 px-3">
        <div className="space-y-1 pb-3">
          {conversations.map((conv) => (
            <div
              key={conv.id}
              className={cn(
                'group flex cursor-pointer items-center gap-2 rounded-lg px-3 py-2 text-sm transition-colors',
                conv.id === currentConversationId
                  ? 'bg-blue-50 text-blue-700'
                  : 'text-slate-600 hover:bg-slate-100'
              )}
              onClick={() => switchConversation(conv.id)}
            >
              <MessageSquare className="h-4 w-4 shrink-0" />
              <span className="flex-1 truncate">{conv.title}</span>
              <Button
                variant="ghost"
                size="icon"
                className="h-6 w-6 shrink-0 opacity-0 group-hover:opacity-100"
                onClick={(e) => {
                  e.stopPropagation()
                  deleteConversation(conv.id)
                }}
              >
                <Trash2 className="h-3 w-3" />
              </Button>
            </div>
          ))}
        </div>
      </ScrollArea>
    </div>
  )
}
