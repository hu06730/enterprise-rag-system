'use client'

import { useEffect, useState } from 'react'
import { useRouter, useParams } from 'next/navigation'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { api } from '@/lib/api'
import type { KBDetail, KBCreate } from '@/types/knowledge-base'

export default function KnowledgeBaseDetailPage() {
  const router = useRouter()
  const params = useParams()
  const id = Number(params.id)
  const [form, setForm] = useState<KBCreate>({ name: '' })
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    api.get<KBDetail>(`/knowledge-bases/${id}`).then((detail) => {
      setForm({
        name: detail.kb.name,
        description: detail.kb.description,
        kb_type: detail.kb.kb_type,
        chunk_strategy: detail.config.chunk_strategy,
        chunk_size: detail.config.chunk_size,
        chunk_overlap: detail.config.chunk_overlap,
        retrieval_mode: detail.config.retrieval_mode,
        top_k: detail.config.top_k,
        vector_weight: detail.config.vector_weight,
        bm25_weight: detail.config.bm25_weight,
      })
    })
  }, [id])

  const handleSave = async () => {
    setLoading(true)
    try {
      await api.put(`/knowledge-bases/${id}`, form)
      router.push('/admin/knowledge-bases')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold text-slate-800">编辑知识库</h2>
        <Button variant="outline" onClick={() => router.push('/admin/knowledge-bases')}>返回列表</Button>
      </div>

      <Card>
        <CardContent className="space-y-4 p-6">
          <div>
            <Label>名称</Label>
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
          <div className="flex gap-4">
            <div className="flex-1">
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
            <div className="flex-1">
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
          <Button onClick={handleSave} disabled={loading}>{loading ? '保存中...' : '保存'}</Button>
        </CardContent>
      </Card>
    </div>
  )
}
