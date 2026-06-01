import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import { TooltipProvider } from '@/components/ui/tooltip'
import './globals.css'

const inter = Inter({ subsets: ['latin'], variable: '--font-sans' })

export const metadata: Metadata = {
  title: '企业知识库 RAG 系统',
  description: '基于 RAG 的智能问答系统',
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="zh-CN" className={inter.variable}>
      <body className="min-h-screen bg-slate-50 font-sans antialiased">
        <TooltipProvider>{children}</TooltipProvider>
      </body>
    </html>
  )
}
