'use client'

import { useEffect, useState } from 'react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter } from '@/components/ui/dialog'
import { api } from '@/lib/api'
import type { KB } from '@/types/knowledge-base'
import type { EvalData, EvalReport } from '@/types/admin'

export default function EvaluationPage() {
  const [kbs, setKbs] = useState<KB[]>([])
  const [selectedKb, setSelectedKb] = useState<string>('')
  const [datasets, setDatasets] = useState<EvalData[]>([])
  const [report, setReport] = useState<EvalReport | null>(null)
  const [running, setRunning] = useState(false)
  const [dialogOpen, setDialogOpen] = useState(false)
  const [newEval, setNewEval] = useState({ question: '', reference_answer: '' })

  useEffect(() => {
    api.get<KB[]>('/knowledge-bases/').then((data) => {
      setKbs(data)
      if (data.length > 0) setSelectedKb(data[0].id.toString())
    })
  }, [])

  useEffect(() => {
    if (selectedKb) {
      api.get<EvalData[]>(`/evaluation/dataset/${selectedKb}`).then(setDatasets).catch(() => {})
    }
  }, [selectedKb])

  const handleAdd = async () => {
    if (!selectedKb || !newEval.question) return
    await api.post('/evaluation/dataset', { kb_id: Number(selectedKb), ...newEval })
    setDialogOpen(false)
    setNewEval({ question: '', reference_answer: '' })
    api.get<EvalData[]>(`/evaluation/dataset/${selectedKb}`).then(setDatasets)
  }

  const handleRun = async () => {
    if (!selectedKb) return
    setRunning(true)
    try {
      const res = await api.post<EvalReport>(`/evaluation/run/${selectedKb}`)
      setReport(res)
    } finally {
      setRunning(false)
    }
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold text-slate-800">评估系统</h2>
        <div className="flex items-center gap-3">
          <Select value={selectedKb} onValueChange={(v) => setSelectedKb(v || '')}>
            <SelectTrigger className="w-[200px]"><SelectValue placeholder="选择知识库" /></SelectTrigger>
            <SelectContent>
              {kbs.map((kb) => <SelectItem key={kb.id} value={kb.id.toString()}>{kb.name}</SelectItem>)}
            </SelectContent>
          </Select>
          <Button variant="outline" onClick={() => setDialogOpen(true)}>添加测试数据</Button>
          <Button onClick={handleRun} disabled={running}>{running ? '运行中...' : '运行评估'}</Button>
        </div>
      </div>

      <Card>
        <CardHeader><CardTitle className="text-base">测试数据集</CardTitle></CardHeader>
        <CardContent className="p-0">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="w-16">ID</TableHead>
                <TableHead>问题</TableHead>
                <TableHead>参考答案</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {datasets.map((d) => (
                <TableRow key={d.id}>
                  <TableCell>{d.id}</TableCell>
                  <TableCell>{d.question}</TableCell>
                  <TableCell className="max-w-[300px] truncate">{d.reference_answer}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>

      {report && (
        <Card>
          <CardHeader><CardTitle className="text-base">评估报告</CardTitle></CardHeader>
          <CardContent>
            <div className="grid grid-cols-3 gap-4 text-center">
              <div className="rounded-lg bg-slate-50 p-4">
                <div className="text-2xl font-bold text-blue-600">{(report.retrieval['avg_recall@5'] * 100).toFixed(1)}%</div>
                <div className="text-sm text-slate-500">Recall@5</div>
              </div>
              <div className="rounded-lg bg-slate-50 p-4">
                <div className="text-2xl font-bold text-blue-600">{(report.retrieval['avg_recall@10'] * 100).toFixed(1)}%</div>
                <div className="text-sm text-slate-500">Recall@10</div>
              </div>
              <div className="rounded-lg bg-slate-50 p-4">
                <div className="text-2xl font-bold text-blue-600">{(report.retrieval.avg_mrr * 100).toFixed(1)}%</div>
                <div className="text-sm text-slate-500">MRR</div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      <Dialog open={dialogOpen} onOpenChange={setDialogOpen}>
        <DialogContent>
          <DialogHeader><DialogTitle>添加测试数据</DialogTitle></DialogHeader>
          <div className="space-y-4">
            <div>
              <Label>问题 *</Label>
              <Input value={newEval.question} onChange={(e) => setNewEval((f) => ({ ...f, question: e.target.value }))} />
            </div>
            <div>
              <Label>参考答案</Label>
              <Input value={newEval.reference_answer} onChange={(e) => setNewEval((f) => ({ ...f, reference_answer: e.target.value }))} />
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setDialogOpen(false)}>取消</Button>
            <Button onClick={handleAdd}>添加</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  )
}
