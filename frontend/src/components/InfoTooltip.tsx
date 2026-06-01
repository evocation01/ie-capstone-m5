import React, { useState } from 'react';
import { Info } from 'lucide-react';

interface InfoTooltipProps {
  title: string;
  content: React.ReactNode;
  children?: React.ReactNode;
}

export default function InfoTooltip({ title, content, children }: InfoTooltipProps) {
  const [show, setShow] = useState(false);

  return (
    <div 
      className="relative inline-flex items-center gap-1 group z-50 cursor-help"
      onMouseEnter={() => setShow(true)}
      onMouseLeave={() => setShow(false)}
    >
      {children}
      <Info className="w-4 h-4 text-slate-400 group-hover:text-blue-500 transition-colors" />
      
      {show && (
        <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 w-64 p-4 rounded-xl glass-panel bg-white/95 shadow-xl border border-slate-200/60 animate-fade-in z-50 pointer-events-none text-left">
          <h4 className="text-xs font-bold text-slate-900 mb-1 uppercase tracking-wider">{title}</h4>
          <div className="text-xs text-slate-600 font-medium leading-relaxed">
            {content}
          </div>
          {/* Triangle Pointer */}
          <div className="absolute top-full left-1/2 -translate-x-1/2 -mt-px w-0 h-0 border-l-[6px] border-l-transparent border-r-[6px] border-r-transparent border-t-[6px] border-t-white" />
        </div>
      )}
    </div>
  );
}
