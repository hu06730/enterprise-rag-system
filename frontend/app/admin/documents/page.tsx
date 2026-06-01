'use client'

import { useEffect, useState, useRef } from 'react'
import { Button } from '@/components/ui/button'
import { Card, CardContent } from '@/components/ui/card'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { Badge } from '@/components/ui/badge'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Upload, RefreshCw, Trash2 } from 'lucide-react'
import { api } from '@/lib/api'
import type { DocItem } from '@/types/document'
import type { KB } from '@/types/knowledge-base'

const STATUS_MAP: Record<string, { label: string; variant: 'default' | 'secondary' | 'destructive' | 'outline' }> = {
  pending: { label: '等待处理', variant: 'secondary' },
  processing: { label: '处理中', variant: 'default' },
  completed: { label: '已完成', variant: 'outline' },
  failed: { label: '失败', variant: 'destructive' },
}

export default function DocumentsPage() {
  const [docs, setDocs] = useState<DocItem[]>([])
  const [kbs, setKbs] = useState<KB[]>([])
  const [selectedKb, setSelectedKb] = useState<string>('')
  const fileInputRef = useRef<HTMLInputElement>(null)

  const fetchDocs = (kbId?: string) => {
    const params = kbId ? `?kb_id=${kbId}` : ''
    api.get<DocItem[]>(`/documents/${params}`).then(setDocs).catch(() => {})
  }

  useEffect(() => {
    api.get<KB[]>('/knowledge-bases/').then((data) => {
      setKbs(data)
      if (data.length > 0) {
        setSelectedKb(data[0].id.toString())
        fetchDocs(data[0].id.toString())
      }
    })
  }, [])

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file || !selectedKb) return
    const formData = new FormData()
    formData.append('kb_id', selectedKb)
    formData.append('file', file)
    formData.append('access_level', 'internal')
    await api.upload('/documents/upload', formData)
    fetchDocs(selectedKb)
    e.target.value = ''
  }

  const handleDelete = async (id: number) => {
    if (!confirm('确定删除此文档？')) return
    await api.delete(`/documents/${id}`)
    fetchDocs(selectedKb)
  }

  const handleReprocess = async (id: number) => {
    await api.put(`/documents/${id}/reprocess`)
    fetchDocs(selectedKb)
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold text-slate-800">文档管理</h2>
        <div className="flex items-center gap-3">
          <Select value={selectedKb} onValueChange={(v) => { setSelectedKb(v || ''); fetchDocs(v || '') }}>
            <SelectTrigger className="w-[200px]"><SelectValue placeholder="选择知识库" /></SelectTrigger>
            <SelectContent>
              {kbs.map((kb) => <SelectItem key={kb.id} value={kb.id.toString()}>{kb.name}</SelectItem>)}
            </SelectContent>
          </Select>
          <input ref={fileInputRef} type="file" className="hidden" accept=".pdf,.doc,.docx,.md,.txt" onChange={handleUpload} />
          <Button onClick={() => fileInputRef.current?.click()}>
            <Upload className="mr-2 h-4 w-4" /> 上传文档
          </Button>
        </div>
      </div>

      <Card>
        <CardContent className="p-0">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="w-16">ID</TableHead>
                <TableHead>文件名</TableHead>
                <TableHead>类型</TableHead>
                <TableHead>状态</TableHead>
                <TableHead>访问级别</TableHead>
                <TableHead>上传时间</TableHead>
                <TableHead className="w-32">操作</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {docs.map((doc) => (
                <TableRow key={doc.id}>
                  <TableCell>{doc.id}</TableCell>
                  <TableCell className="font-medium">{doc.filename}</TableCell>
                  <TableCell><Badge variant="outline">{doc.file_type}</Badge></TableCell>
                  <TableCell>
                    <Badge variant={STATUS_MAP[doc.status]?.variant || 'secondary'}>
                      {STATUS_MAP[doc.status]?.label || doc.status}
                    </Badge>
                  </TableCell>
                  <TableCell>{doc.access_level}</TableCell>
                  <TableCell className="text-sm text-slate-500">{doc.created_at}</TableCell>
                  <TableCell>
                    <div className="flex gap-1">
                      <Button variant="ghost" size="icon" onClick={() => handleReprocess(doc.id)} disabled={doc.status === 'processing'}>
                        <RefreshCw className="h-4 w-4" />
                      </Button>
                      <Button variant="ghost" size="icon" onClick={() => handleDelete(doc.id)}>
                        <Trash2 className="h-4 w-4 text-red-500" />
                      </Button>
                    </div>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
    </div>
  )
}
