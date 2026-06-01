'use client'

import { useEffect, useState } from 'react'
import { useRouter } from 'next/navigation'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { Badge } from '@/components/ui/badge'
import { Plus, Pencil, Trash2 } from 'lucide-react'
import { KBDialog } from '@/components/admin/kb-dialog'
import { api } from '@/lib/api'
import type { KB, KBCreate } from '@/types/knowledge-base'

export default function KnowledgeBasesPage() {
  const router = useRouter()
  const [kbs, setKbs] = useState<KB[]>([])
  const [dialogOpen, setDialogOpen] = useState(false)

  const fetchKbs = () => api.get<KB[]>('/knowledge-bases/').then(setKbs).catch(() => {})

  useEffect(() => { fetchKbs() }, [])

  const handleCreate = async (data: KBCreate) => {
    await api.post('/knowledge-bases/', data)
    fetchKbs()
  }

  const handleDelete = async (id: number) => {
    if (!confirm('确定删除此知识库？')) return
    await api.delete(`/knowledge-bases/${id}`)
    fetchKbs()
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold text-slate-800">知识库管理</h2>
        <Button onClick={() => setDialogOpen(true)}>
          <Plus className="mr-2 h-4 w-4" /> 创建知识库
        </Button>
      </div>

      <Card>
        <CardContent className="p-0">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="w-16">ID</TableHead>
                <TableHead>名称</TableHead>
                <TableHead>类型</TableHead>
                <TableHead>描述</TableHead>
                <TableHead>创建时间</TableHead>
                <TableHead className="w-32">操作</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {kbs.map((kb) => (
                <TableRow key={kb.id}>
                  <TableCell>{kb.id}</TableCell>
                  <TableCell className="font-medium">{kb.name}</TableCell>
                  <TableCell><Badge variant="outline">{kb.kb_type}</Badge></TableCell>
                  <TableCell className="max-w-[200px] truncate">{kb.description}</TableCell>
                  <TableCell className="text-sm text-slate-500">{kb.created_at}</TableCell>
                  <TableCell>
                    <div className="flex gap-1">
                      <Button variant="ghost" size="icon" onClick={() => router.push(`/admin/knowledge-bases/${kb.id}`)}>
                        <Pencil className="h-4 w-4" />
                      </Button>
                      <Button variant="ghost" size="icon" onClick={() => handleDelete(kb.id)}>
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

      <KBDialog open={dialogOpen} onOpenChange={setDialogOpen} onSubmit={handleCreate} />
    </div>
  )
}
