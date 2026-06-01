'use client'

import { useState, useEffect } from 'react'
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter } from '@/components/ui/dialog'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import type { KBCreate } from '@/types/knowledge-base'

interface KBDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  onSubmit: (data: KBCreate) => void
  initialData?: KBCreate
}

export function KBDialog({ open, onOpenChange, onSubmit, initialData }: KBDialogProps) {
  const [form, setForm] = useState<KBCreate>({
    name: '',
    description: '',
    kb_type: 'employee',
    chunk_strategy: 'recursive',
    chunk_size: 512,
    chunk_overlap: 50,
    retrieval_mode: 'hybrid',
    top_k: 10,
    vector_weight: 0.5,
    bm25_weight: 0.5,
  })

  useEffect(() => {
    if (initialData) setForm((f) => ({ ...f, ...initialData }))
  }, [initialData])

  const handleSubmit = () => {
    if (!form.name) return
    onSubmit(form)
    onOpenChange(false)
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-lg">
        <DialogHeader>
          <DialogTitle>{initialData ? '编辑知识库' : '创建知识库'}</DialogTitle>
        </DialogHeader>
        <div className="space-y-4">
          <div>
            <Label>名称 *</Label>
            <Input value={form.name} onChange={(e) => setForm((f) => ({ ...f, name: e.target.value }))} />
          </div>
          <div>
            <Label>描述</Label>
            <Input value={form.description} onChange={(e) => setForm((f) => ({ ...f, description: e.target.value }))} />
          </div>
          <div>
            <Label>类型</Label>
            <Select value={form.kb_type} onValueChange={(v) => setForm((f) => ({ ...f, kb_type: v }))}>
              <SelectTrigger><SelectValue /></SelectTrigger>
              <SelectContent>
                <SelectItem value="employee">员工知识库</SelectItem>
                <SelectItem value="customer">客服知识库</SelectItem>
                <SelectItem value="compliance">合规知识库</SelectItem>
              </SelectContent>
            </Select>
          </div>
          <div>
            <Label>分块策略</Label>
            <Select value={form.chunk_strategy} onValueChange={(v) => setForm((f) => ({ ...f, chunk_strategy: v }))}>
              <SelectTrigger><SelectValue /></SelectTrigger>
              <SelectContent>
                <SelectItem value="recursive">递归分块</SelectItem>
                <SelectItem value="markdown">Markdown 分块</SelectItem>
                <SelectItem value="semantic">语义分块</SelectItem>
              </SelectContent>
            </Select>
          </div>
          <div>
            <Label>检索模式</Label>
            <Select value={form.retrieval_mode} onValueChange={(v) => setForm((f) => ({ ...f, retrieval_mode: v }))}>
              <SelectTrigger><SelectValue /></SelectTrigger>
              <SelectContent>
                <SelectItem value="hybrid">混合检索</SelectItem>
                <SelectItem value="vector">向量检索</SelectItem>
                <SelectItem value="keyword">关键词检索</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </div>
        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>取消</Button>
          <Button onClick={handleSubmit}>{initialData ? '保存' : '创建'}</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
