"use client";

import React from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import {
  BarChart3,
  TrendingUp,
  Settings,
  Target,
  Home
} from 'lucide-react';

interface SidebarLayoutProps {
  children: React.ReactNode;
}

const navigation = [
  { name: 'Dashboard', href: '/', icon: Home },
  { name: 'Sensitivity Analysis', href: '/sensitivity', icon: Settings },
  { name: 'Benchmark Results', href: '/benchmark', icon: BarChart3 },
  { name: 'Model Comparison', href: '/comparison', icon: Target },
];

function cn(...classes: string[]) {
  return classes.filter(Boolean).join(' ');
}

export default function SidebarLayout({ children }: SidebarLayoutProps) {
  const pathname = usePathname();

  return (
    <div className="min-h-screen bg-zinc-50">
      {/* Sidebar */}
      <div className="fixed inset-y-0 left-0 w-64 bg-white border-r border-zinc-200">
        <div className="flex flex-col h-full">
          {/* Logo */}
          <div className="flex items-center gap-3 px-6 py-4 border-b border-zinc-200">
            <TrendingUp className="w-8 h-8 text-blue-600" />
            <div>
              <span className="text-xl font-bold tracking-tight">IE-Capstone</span>
              <p className="text-xs text-zinc-500">Supply Chain DSS</p>
            </div>
          </div>

          {/* Navigation */}
          <nav className="flex-1 px-4 py-6 space-y-2">
            {navigation.map((item) => {
              const Icon = item.icon;
              const isActive = pathname === item.href;
              return (
                <Link
                  key={item.name}
                  href={item.href}
                  className={cn(
                    'flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-semibold transition-all',
                    isActive
                      ? 'bg-blue-600 text-white shadow-sm'
                      : 'text-zinc-600 hover:bg-zinc-100 hover:text-zinc-900'
                  )}
                >
                  <Icon className="w-5 h-5" />
                  {item.name}
                </Link>
              );
            })}
          </nav>

          {/* Project Status */}
          <div className="px-4 pb-6">
            <div className="bg-zinc-100 rounded-xl p-4">
              <p className="text-xs font-semibold text-zinc-500 mb-1">PROJECT PHASE</p>
              <p className="text-sm font-bold text-zinc-800 mb-2">IE 4198 Midterm</p>
              <div className="w-full bg-zinc-200 h-1.5 rounded-full overflow-hidden">
                <div className="bg-blue-600 h-full w-3/4" />
              </div>
              <p className="text-xs text-zinc-500 mt-1">75% Complete</p>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="pl-64">
        <main className="min-h-screen">
          {children}
        </main>
      </div>
    </div>
  );
}