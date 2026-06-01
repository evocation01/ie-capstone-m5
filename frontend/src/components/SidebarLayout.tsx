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
    <div className="min-h-screen bg-slate-50 flex overflow-hidden">
      {/* Sidebar Navigation - Hidden during PDF Print */}
      <nav className="w-64 bg-slate-50 border-r border-slate-200 text-slate-900 flex-shrink-0 flex flex-col print-hidden relative z-20">
        <div className="p-6 h-full flex flex-col">
          {/* Logo */}
          <div className="flex items-center gap-3 px-4 py-6 border-b border-slate-200/80">
            <div className="bg-gradient-to-br from-blue-500 to-indigo-600 text-white p-2 rounded-xl shadow-md">
              <TrendingUp className="w-6 h-6" />
            </div>
            <div>
              <span className="text-xl font-extrabold tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-slate-900 to-slate-700">IE-Capstone</span>
              <p className="text-[10px] uppercase font-bold text-slate-500 tracking-wider">Supply Chain DSS</p>
            </div>
          </div>

          {/* Navigation */}
          <div className="flex-1 px-2 py-6 space-y-1.5 overflow-y-auto">
            {navigation.map((item) => {
              const Icon = item.icon;
              const isActive = pathname === item.href;
              return (
                <Link
                  key={item.name}
                  href={item.href}
                  className={cn(
                    'flex items-center gap-3 px-4 py-3 rounded-xl text-sm font-semibold transition-all duration-200',
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
          </div>

          {/* Project Status */}
          <div className="px-2 pb-2">
            <div className="bg-white rounded-xl p-4 border border-slate-200 shadow-sm">
              <p className="text-[10px] font-bold text-slate-400 mb-1 tracking-wider uppercase">Project Phase</p>
              <p className="text-sm font-extrabold text-slate-800 mb-2">Final Presentation</p>
              <div className="w-full bg-slate-100 h-2 rounded-full overflow-hidden">
                <div className="bg-gradient-to-r from-blue-500 to-emerald-400 h-full w-[95%] animate-pulse" />
              </div>
              <p className="text-[10px] font-semibold text-slate-500 mt-2 text-right">95% Complete</p>
            </div>
          </div>
        </div>
      </nav>

      {/* Main Content Area - Expanded during PDF Print */}
      <main className="flex-1 overflow-y-auto print-expanded relative">
        <div className="absolute top-0 inset-x-0 h-96 bg-gradient-to-b from-blue-50/50 to-transparent -z-10 pointer-events-none print-hidden" />
        {children}
      </main>
    </div>
  );
}