'use client'

import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { cn } from '@/lib/utils'
import {
  LayoutDashboard,
  Database,
  FileText,
  ClipboardList,
  BarChart3,
  Users,
  MessageSquare,
} from 'lucide-react'

const NAV_ITEMS = [
  { href: '/chat', label: '智能问答', icon: MessageSquare, roles: ['admin', 'editor', 'viewer'] },
  { href: '/admin/dashboard', label: '仪表盘', icon: LayoutDashboard, roles: ['admin', 'editor'] },
  { href: '/admin/knowledge-bases', label: '知识库管理', icon: Database, roles: ['admin', 'editor'] },
  { href: '/admin/documents', label: '文档管理', icon: FileText, roles: ['admin', 'editor'] },
  { href: '/admin/audit', label: '审计日志', icon: ClipboardList, roles: ['admin', 'editor'] },
  { href: '/admin/evaluation', label: '评估系统', icon: BarChart3, roles: ['admin', 'editor'] },
  { href: '/admin/users', label: '用户管理', icon: Users, roles: ['admin'] },
]

interface AdminSidebarProps {
  userRole?: string
}

export function AdminSidebar({ userRole }: AdminSidebarProps) {
  const pathname = usePathname()

  const filteredItems = NAV_ITEMS.filter((item) => userRole && item.roles.includes(userRole))

  return (
    <aside className="flex h-full w-[220px] flex-col bg-slate-900">
      <div className="flex h-14 items-center justify-center border-b border-slate-800">
        <span className="text-lg font-bold text-white">企业知识库</span>
      </div>
      <nav className="flex-1 space-y-1 p-3">
        {filteredItems.map((item) => {
          const isActive = pathname === item.href || pathname.startsWith(item.href + '/')
          return (
            <Link
              key={item.href}
              href={item.href}
              className={cn(
                'flex items-center gap-3 rounded-lg px-3 py-2 text-sm transition-colors',
                isActive
                  ? 'bg-blue-600 text-white'
                  : 'text-slate-400 hover:bg-slate-800 hover:text-white'
              )}
            >
              <item.icon className="h-4 w-4" />
              {item.label}
            </Link>
          )
        })}
      </nav>
    </aside>
  )
}
