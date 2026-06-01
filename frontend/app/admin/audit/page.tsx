'use client'

import { useEffect, useState } from 'react'
import { Card, CardContent } from '@/components/ui/card'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { api } from '@/lib/api'
import type { AuditLog } from '@/types/admin'

export default function AuditLogsPage() {
  const [logs, setLogs] = useState<AuditLog[]>([])
  const [kbId, setKbId] = useState('')
  const [offset, setOffset] = useState(0)
  const limit = 50

  const fetchLogs = () => {
    const params = new URLSearchParams()
    if (kbId) params.set('kb_id', kbId)
    params.set('limit', String(limit))
    params.set('offset', String(offset))
    api.get<AuditLog[]>(`/audit/logs?${params}`).then(setLogs).catch(() => {})
  }

  useEffect(() => { fetchLogs() }, [offset])

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold text-slate-800">审计日志</h2>

      <div className="flex items-center gap-3">
        <Input
          placeholder="知识库 ID"
          value={kbId}
          onChange={(e) => setKbId(e.target.value)}
          className="w-[150px]"
        />
        <Button onClick={() => { setOffset(0); fetchLogs() }}>搜索</Button>
      </div>

      <Card>
        <CardContent className="p-0">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="w-16">ID</TableHead>
                <TableHead>用户</TableHead>
                <TableHead className="w-16">KB</TableHead>
                <TableHead>问题</TableHead>
                <TableHead>回答</TableHead>
                <TableHead className="w-16">反馈</TableHead>
                <TableHead>时间</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {logs.map((log) => (
                <TableRow key={log.id}>
                  <TableCell>{log.id}</TableCell>
                  <TableCell>{log.username}</TableCell>
                  <TableCell>{log.kb_id}</TableCell>
                  <TableCell className="max-w-[200px] truncate">{log.query}</TableCell>
                  <TableCell className="max-w-[300px] truncate">{log.answer}</TableCell>
                  <TableCell>
                    {log.feedback === 1 && <Badge className="bg-green-100 text-green-700">好评</Badge>}
                    {log.feedback === -1 && <Badge variant="destructive">差评</Badge>}
                  </TableCell>
                  <TableCell className="text-sm text-slate-500">{log.created_at}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>

      <div className="flex justify-center gap-3">
        <Button variant="outline" disabled={offset === 0} onClick={() => setOffset((o) => Math.max(0, o - limit))}>
          上一页
        </Button>
        <Button variant="outline" disabled={logs.length < limit} onClick={() => setOffset((o) => o + limit)}>
          下一页
        </Button>
      </div>
    </div>
  )
}
