'use client'

import { Card } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { FileText } from 'lucide-react'
import type { Source } from '@/types/chat'

interface SourceCardProps {
  source: Source
}

export function SourceCard({ source }: SourceCardProps) {
  return (
    <Card className="p-3 transition-shadow hover:shadow-md">
      <div className="mb-2 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <FileText className="h-4 w-4 text-slate-400" />
          <span className="text-sm font-medium text-slate-700">{source.doc_title}</span>
        </div>
        <Badge variant="secondary" className="text-xs">
          {Math.round(source.score * 100)}% 相关
        </Badge>
      </div>
      <p className="text-xs leading-relaxed text-slate-500">{source.chunk_text}</p>
    </Card>
  )
}
