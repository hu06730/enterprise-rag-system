import { create } from 'zustand'
import { api } from '@/lib/api'
import { createSSEStream } from '@/lib/sse'
import type { ChatMessage, Conversation, Source, FeedbackRequest } from '@/types/chat'
import type { KB } from '@/types/knowledge-base'

interface ChatState {
  conversations: Conversation[]
  currentConversationId: string | null
  messages: ChatMessage[]
  isStreaming: boolean
  currentSources: Source[]
  knowledgeBases: KB[]
  selectedKbId: number | null
  stopStream: (() => void) | null

  fetchKnowledgeBases: () => Promise<void>
  setSelectedKb: (id: number) => void
  createConversation: () => string
  switchConversation: (id: string) => void
  deleteConversation: (id: string) => void
  sendMessage: (query: string) => Promise<void>
  stopGeneration: () => void
  sendFeedback: (traceId: string, feedback: number, text?: string) => Promise<void>
  clearCurrentChat: () => void
}

function generateId() {
  return Date.now().toString(36) + Math.random().toString(36).slice(2)
}

function getTitle(messages: ChatMessage[]): string {
  const firstUser = messages.find((m) => m.role === 'user')
  return firstUser?.content.slice(0, 30) || '新对话'
}

export const useChatStore = create<ChatState>((set, get) => ({
  conversations: [],
  currentConversationId: null,
  messages: [],
  isStreaming: false,
  currentSources: [],
  knowledgeBases: [],
  selectedKbId: null,
  stopStream: null,

  fetchKnowledgeBases: async () => {
    const kbs = await api.get<KB[]>('/knowledge-bases/')
    set({ knowledgeBases: kbs })
    if (kbs.length > 0 && !get().selectedKbId) {
      set({ selectedKbId: kbs[0].id })
    }
  },

  setSelectedKb: (id) => set({ selectedKbId: id }),

  createConversation: () => {
    const id = generateId()
    const conv: Conversation = { id, title: '新对话', kbId: get().selectedKbId || 0, messages: [], createdAt: Date.now() }
    set((s) => ({ conversations: [conv, ...s.conversations], currentConversationId: id, messages: [] }))
    return id
  },

  switchConversation: (id) => {
    const conv = get().conversations.find((c) => c.id === id)
    if (conv) set({ currentConversationId: id, messages: conv.messages })
  },

  deleteConversation: (id) => {
    set((s) => {
      const convs = s.conversations.filter((c) => c.id !== id)
      const isCurrent = s.currentConversationId === id
      return {
        conversations: convs,
        currentConversationId: isCurrent ? (convs[0]?.id || null) : s.currentConversationId,
        messages: isCurrent ? (convs[0]?.messages || []) : s.messages,
      }
    })
  },

  sendMessage: async (query) => {
    const state = get()
    let convId = state.currentConversationId
    if (!convId) convId = get().createConversation()
    if (!state.selectedKbId) return

    const userMsg: ChatMessage = { role: 'user', content: query }
    const assistantMsg: ChatMessage = { role: 'assistant', content: '', loading: true, sources: [] }
    const newMessages = [...state.messages, userMsg, assistantMsg]

    set({ messages: newMessages, isStreaming: true, currentSources: [] })

    const history = state.messages.map((m) => ({ role: m.role, content: m.content }))

    const controller = new AbortController()
    set({
      stopStream: () => {
        controller.abort()
        set({ isStreaming: false, stopStream: null })
      },
    })

    let accumulated = ''
    let sources: Source[] = []
    let traceId = ''

    createSSEStream(
      { kb_id: state.selectedKbId, query, history },
      {
        onSources: (s) => { sources = s; set({ currentSources: s }) },
        onToken: (token) => {
          accumulated += token
          set((s) => {
            const msgs = [...s.messages]
            msgs[msgs.length - 1] = { ...msgs[msgs.length - 1], content: accumulated }
            return { messages: msgs }
          })
        },
        onDone: (tid) => {
          traceId = tid
          set((s) => {
            const msgs = [...s.messages]
            msgs[msgs.length - 1] = { ...msgs[msgs.length - 1], content: accumulated, sources, traceId, loading: false }
            const convs = s.conversations.map((c) =>
              c.id === convId ? { ...c, messages: msgs, title: getTitle(msgs) } : c
            )
            return { messages: msgs, conversations: convs, isStreaming: false, stopStream: null }
          })
        },
        onError: () => {
          set((s) => {
            const msgs = [...s.messages]
            msgs[msgs.length - 1] = { ...msgs[msgs.length - 1], content: '抱歉，发生了错误，请重试。', loading: false }
            return { messages: msgs, isStreaming: false, stopStream: null }
          })
        },
      },
      controller.signal
    )
  },

  stopGeneration: () => get().stopStream?.(),

  sendFeedback: async (traceId, feedback, text) => {
    await api.post('/chat/feedback', { trace_id: traceId, feedback, feedback_text: text || '' })
  },

  clearCurrentChat: () => {
    const id = get().currentConversationId
    if (id) {
      set((s) => ({
        messages: [],
        conversations: s.conversations.map((c) => (c.id === id ? { ...c, messages: [], title: '新对话' } : c)),
      }))
    }
  },
}))
