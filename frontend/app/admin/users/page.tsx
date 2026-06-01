'use client'

import { useState } from 'react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Badge } from '@/components/ui/badge'
import { api } from '@/lib/api'

export default function UserManagementPage() {
  const [form, setForm] = useState({ username: '', password: '', role: 'viewer', departments: [] as string[] })
  const [loading, setLoading] = useState(false)
  const [success, setSuccess] = useState('')
  const [deptInput, setDeptInput] = useState('')

  const handleRegister = async () => {
    if (!form.username || !form.password) return
    setLoading(true)
    setSuccess('')
    try {
      await api.post('/auth/register', form)
      setSuccess('用户创建成功')
      setForm({ username: '', password: '', role: 'viewer', departments: [] })
    } finally {
      setLoading(false)
    }
  }

  const addDept = () => {
    if (deptInput.trim()) {
      setForm((f) => ({ ...f, departments: [...f.departments, deptInput.trim()] }))
      setDeptInput('')
    }
  }

  const removeDept = (i: number) => {
    setForm((f) => ({ ...f, departments: f.departments.filter((_, idx) => idx !== i) }))
  }

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold text-slate-800">用户管理</h2>

      <Card className="max-w-md">
        <CardHeader><CardTitle className="text-base">注册新用户</CardTitle></CardHeader>
        <CardContent className="space-y-4">
          <div>
            <Label>用户名 *</Label>
            <Input value={form.username} onChange={(e) => setForm((f) => ({ ...f, username: e.target.value }))} />
          </div>
          <div>
            <Label>密码 *</Label>
            <Input type="password" value={form.password} onChange={(e) => setForm((f) => ({ ...f, password: e.target.value }))} />
          </div>
          <div>
            <Label>角色</Label>
            <Select value={form.role} onValueChange={(v) => setForm((f) => ({ ...f, role: v || 'viewer' }))}>
              <SelectTrigger><SelectValue /></SelectTrigger>
              <SelectContent>
                <SelectItem value="admin">管理员</SelectItem>
                <SelectItem value="editor">编辑者</SelectItem>
                <SelectItem value="viewer">查看者</SelectItem>
              </SelectContent>
            </Select>
          </div>
          <div>
            <Label>部门</Label>
            <div className="flex gap-2">
              <Input value={deptInput} onChange={(e) => setDeptInput(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && addDept()} placeholder="输入部门名称" />
              <Button variant="outline" onClick={addDept}>添加</Button>
            </div>
            <div className="mt-2 flex flex-wrap gap-2">
              {form.departments.map((d, i) => (
                <Badge key={i} variant="secondary" className="cursor-pointer" onClick={() => removeDept(i)}>{d} ×</Badge>
              ))}
            </div>
          </div>
          {success && <p className="text-sm text-green-600">{success}</p>}
          <Button onClick={handleRegister} disabled={loading}>{loading ? '创建中...' : '创建用户'}</Button>
        </CardContent>
      </Card>
    </div>
  )
}
