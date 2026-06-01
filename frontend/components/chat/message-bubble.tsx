'use client'

import { useState } from 'react'
import { Avatar, AvatarFallback } from '@/components/ui/avatar'
import { Button } from '@/components/ui/button'
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog'
import { ThumbsUp, ThumbsDown, BookOpen, User, Bot } from 'lucide-react'
import { SourceCard } from './source-card'
import { ThinkingIndicator } from './thinking-indicator'
import { cn } from '@/lib/utils'
import type { ChatMessage } from '@/types/chat'

interface MessageBubbleProps {
  message: ChatMessage
  onFeedback?: (traceId: string, feedback: number) => void
}

export function MessageBubble({ message, onFeedback }: MessageBubbleProps) {
  const [feedbackGiven, setFeedbackGiven] = useState<number | null>(null)
  const isUser = message.role === 'user'

  const handleFeedback = (type: 1 | -1) => {
    if (message.traceId && onFeedback) {
      onFeedback(message.traceId, type)
      setFeedbackGiven(type)
    }
  }

  return (
    <div className={cn('flex gap-3', isUser ? 'flex-row-reverse' : 'flex-row')}>
      <Avatar className="h-8 w-8 shrink-0">
        <AvatarFallback className={cn(isUser ? 'bg-blue-100 text-blue-600' : 'bg-slate-100 text-slate-600')}>
          {isUser ? <User className="h-4 w-4" /> : <Bot className="h-4 w-4" />}
        </AvatarFallback>
      </Avatar>

      <div className={cn('max-w-[70%] space-y-2', isUser && 'items-end')}>
        {/* Message bubble */}
        <div
          className={cn(
            'rounded-2xl px-4 py-3 text-sm leading-relaxed shadow-sm',
            isUser
              ? 'rounded-tr-md bg-blue-50 text-slate-800'
              : 'rounded-tl-md bg-white text-slate-700'
          )}
        >
          {message.loading && !message.content ? (
            <ThinkingIndicator />
          ) : (
            <span className="whitespace-pre-wrap">{message.content}</span>
          )}
        </div>

        {/* Sources */}
        {!isUser && message.sources && message.sources.length > 0 && (
          <Dialog>
            <DialogTrigger
              render={
                <Button variant="ghost" size="sm" className="h-7 gap-1 text-xs text-slate-400 hover:text-slate-600" />
              }
            >
              <BookOpen className="h-3 w-3" />
              查看引用来源 ({message.sources.length})
            </DialogTrigger>
            <DialogContent className="max-w-2xl">
              <DialogHeader>
                <DialogTitle>引用来源</DialogTitle>
              </DialogHeader>
              <div className="max-h-[60vh] space-y-3 overflow-y-auto">
                {message.sources.map((src, i) => (
                  <SourceCard key={i} source={src} />
                ))}
              </div>
            </DialogContent>
          </Dialog>
        )}

        {/* Feedback buttons */}
        {!isUser && message.traceId && !message.loading && (
          <div className="flex gap-1">
            <Button
              variant="ghost"
              size="icon"
              className={cn('h-7 w-7', feedbackGiven === 1 && 'text-green-500')}
              onClick={() => handleFeedback(1)}
              disabled={feedbackGiven !== null}
            >
              <ThumbsUp className="h-3.5 w-3.5" />
            </Button>
            <Button
              variant="ghost"
              size="icon"
              className={cn('h-7 w-7', feedbackGiven === -1 && 'text-red-500')}
              onClick={() => handleFeedback(-1)}
              disabled={feedbackGiven !== null}
            >
              <ThumbsDown className="h-3.5 w-3.5" />
            </Button>
          </div>
        )}
      </div>
    </div>
  )
}
