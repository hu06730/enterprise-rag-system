'use client'

import { useEffect, useRef } from 'react'
import { MessageBubble } from './message-bubble'
import { WelcomeCard } from './welcome-card'
import { useChatStore } from '@/stores/chat'

export function ChatMessages() {
  const { messages, sendFeedback } = useChatStore()
  const scrollRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [messages])

  const handleQuestionClick = (question: string) => {
    useChatStore.getState().sendMessage(question)
  }

  if (messages.length === 0) {
    return <WelcomeCard onQuestionClick={handleQuestionClick} />
  }

  return (
    <div ref={scrollRef} className="flex-1 overflow-y-auto px-4 py-6">
      <div className="mx-auto max-w-3xl space-y-6">
        {messages.map((msg, i) => (
          <MessageBubble key={i} message={msg} onFeedback={sendFeedback} />
        ))}
      </div>
    </div>
  )
}
