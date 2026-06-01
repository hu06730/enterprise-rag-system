'use client'

import { Button } from '@/components/ui/button'
import { Card } from '@/components/ui/card'
import { MessageSquare } from 'lucide-react'

const EXAMPLE_QUESTIONS = [
  '总结上月销售报告',
  '如何配置 VPN',
  '公司的年假政策是什么',
  '产品退货流程是怎样的',
]

interface WelcomeCardProps {
  onQuestionClick: (question: string) => void
}

export function WelcomeCard({ onQuestionClick }: WelcomeCardProps) {
  return (
    <div className="flex h-full items-center justify-center">
      <Card className="max-w-lg p-8 text-center">
        <div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-full bg-slate-100">
          <MessageSquare className="h-8 w-8 text-slate-400" />
        </div>
        <h2 className="mb-2 text-xl font-semibold text-slate-800">有什么可以帮助你？</h2>
        <p className="mb-6 text-sm text-slate-500">
          选择一个知识库，开始提问
        </p>
        <div className="flex flex-wrap justify-center gap-2">
          {EXAMPLE_QUESTIONS.map((q) => (
            <Button
              key={q}
              variant="outline"
              size="sm"
              className="rounded-full text-slate-600"
              onClick={() => onQuestionClick(q)}
            >
              {q}
            </Button>
          ))}
        </div>
      </Card>
    </div>
  )
}
