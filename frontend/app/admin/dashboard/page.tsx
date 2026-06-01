'use client'

import { useEffect, useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { StatsCard } from '@/components/admin/stats-card'
import { Database, FileText, Layers, MessageSquare, ThumbsUp, ThumbsDown } from 'lucide-react'
import { api } from '@/lib/api'
import type { AdminStats } from '@/types/admin'

export default function DashboardPage() {
  const [stats, setStats] = useState<AdminStats | null>(null)

  useEffect(() => {
    api.get<AdminStats>('/admin/stats').then(setStats).catch(() => {})
  }, [])

  if (!stats) return <div className="flex items-center justify-center p-8 text-slate-400">加载中...</div>

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold text-slate-800">仪表盘</h2>
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <StatsCard title="知识库" value={stats.knowledge_bases} icon={<Database className="h-5 w-5" />} />
        <StatsCard title="文档" value={stats.documents} icon={<FileText className="h-5 w-5" />} />
        <StatsCard title="分块" value={stats.chunks} icon={<Layers className="h-5 w-5" />} />
        <StatsCard title="查询次数" value={stats.total_queries} icon={<MessageSquare className="h-5 w-5" />} />
      </div>
      <div className="grid gap-4 sm:grid-cols-2">
        <Card>
          <CardHeader><CardTitle className="text-base">用户反馈</CardTitle></CardHeader>
          <CardContent className="flex justify-around text-center">
            <div>
              <div className="text-3xl font-bold text-green-500">{stats.feedback_up}</div>
              <div className="text-sm text-slate-500">好评</div>
            </div>
            <div>
              <div className="text-3xl font-bold text-red-500">{stats.feedback_down}</div>
              <div className="text-sm text-slate-500">差评</div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
