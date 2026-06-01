import React from 'react';

export default function Skeleton({ className }: { className?: string }) {
  return (
    <div className={`animate-pulse bg-slate-200/60 rounded-2xl ${className || ''}`} />
  );
}

export function DashboardSkeleton() {
  return (
    <div className="p-8 max-w-7xl mx-auto space-y-10">
      <div className="space-y-4">
        <Skeleton className="h-12 w-1/3" />
        <Skeleton className="h-6 w-1/2" />
      </div>
      
      <Skeleton className="h-32 w-full" />
      
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <Skeleton className="h-36 w-full" />
        <Skeleton className="h-36 w-full" />
        <Skeleton className="h-36 w-full" />
        <Skeleton className="h-36 w-full" />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <Skeleton className="h-[400px] lg:col-span-2 w-full" />
        <Skeleton className="h-[400px] w-full" />
      </div>
    </div>
  );
}

export function PageSkeleton() {
  return (
    <div className="p-8 max-w-7xl mx-auto space-y-10">
      <div className="space-y-4">
        <Skeleton className="h-10 w-1/4" />
        <Skeleton className="h-6 w-1/3" />
      </div>
      <Skeleton className="h-[500px] w-full" />
    </div>
  );
}
