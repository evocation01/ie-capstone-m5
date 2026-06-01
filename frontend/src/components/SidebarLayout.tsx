"use client";

import React from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import {
  BarChart3,
  TrendingUp,
  Settings,
  Target,
  Home,
  BookOpen
} from 'lucide-react';

interface SidebarLayoutProps {
  children: React.ReactNode;
}

const navigation = [
  { name: 'Dashboard', href: '/', icon: Home },
  { name: 'Sensitivity Analysis', href: '/sensitivity', icon: Settings },
  { name: 'Benchmark Results', href: '/benchmark', icon: BarChart3 },
  { name: 'Model Comparison', href: '/comparison', icon: Target },
  { name: 'Executive Summary', href: '/about', icon: BookOpen },
];

function cn(...classes: string[]) {
  return classes.filter(Boolean).join(' ');
}

export default function SidebarLayout({ children }: SidebarLayoutProps) {
  const pathname = usePathname();

  return (
    <div className="min-h-screen bg-slate-50 flex">
      {/* Sidebar */}
      <div className="fixed inset-y-0 left-0 w-64 glass-panel border-r border-slate-200/50 z-10 flex flex-col">
        {/* Logo */}
        <div className="flex items-center gap-3 px-6 py-6 border-b border-slate-200/50">
          <div className="bg-gradient-to-br from-blue-500 to-indigo-600 text-white p-2 rounded-xl shadow-md">
            <TrendingUp className="w-6 h-6" />
          </div>
          <div>
            <span className="text-xl font-extrabold tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-slate-900 to-slate-700">IE-Capstone</span>
            <p className="text-[10px] uppercase font-bold text-slate-400 tracking-wider">Supply Chain DSS</p>
          </div>
        </div>

        {/* Navigation */}
        <nav className="flex-1 px-4 py-6 space-y-1.5 overflow-y-auto">
          {navigation.map((item) => {
            const Icon = item.icon;
            const isActive = pathname === item.href;
            return (
              <Link
                key={item.name}
                href={item.href}
                className={cn(
                  'flex items-center gap-3 px-3 py-3 rounded-xl text-sm font-semibold transition-all duration-200',
                  isActive
                    ? 'bg-gradient-to-r from-blue-600 to-indigo-600 text-white shadow-md transform scale-[1.02]'
                    : 'text-slate-600 hover:bg-slate-100 hover:text-slate-900'
                )}
              >
                <Icon className={cn("w-5 h-5", isActive ? "text-white" : "text-slate-400")} />
                {item.name}
              </Link>
            );
          })}
        </nav>

        {/* Project Status */}
        <div className="px-4 pb-6">
          <div className="bg-slate-100/80 backdrop-blur-sm rounded-xl p-4 border border-slate-200/50 shadow-sm">
            <p className="text-[10px] font-bold text-slate-400 mb-1 tracking-wider uppercase">Project Phase</p>
            <p className="text-sm font-extrabold text-slate-800 mb-2">Final Presentation</p>
            <div className="w-full bg-slate-200 h-2 rounded-full overflow-hidden">
              <div className="bg-gradient-to-r from-blue-500 to-emerald-400 h-full w-[95%] animate-pulse" />
            </div>
            <p className="text-[10px] font-semibold text-slate-500 mt-2 text-right">95% Complete</p>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 pl-64">
        <main className="min-h-screen relative">
          {/* Decorative background blob */}
          <div className="absolute top-0 right-0 -z-10 w-96 h-96 bg-blue-400/10 rounded-full blur-3xl pointer-events-none" />
          {children}
        </main>
      </div>
    </div>
  );
}