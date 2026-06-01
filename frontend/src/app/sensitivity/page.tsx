"use client";

import React, { useState, useMemo } from 'react';
import SidebarLayout from '@/components/SidebarLayout';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { Settings2, AlertTriangle, ArrowRightLeft, BookOpen } from 'lucide-react';
import InfoTooltip from '@/components/InfoTooltip';

const models = [
  { name: 'LightGBM', rmse: 2.10, color: '#3b82f6' },
  { name: 'DeepAR', rmse: 2.15, color: '#8b5cf6' },
  { name: 'LSTM', rmse: 3.58, color: '#f43f5e' }
];

export default function SensitivityPage() {
  const [holdingCost, setHoldingCost] = useState<number>(1.0);
  const [stockoutCost, setStockoutCost] = useState<number>(10.0);
  const [leadTime, setLeadTime] = useState<number>(1);
  const criticalRatio = useMemo(() => {
    return stockoutCost / (holdingCost + stockoutCost);
  }, [holdingCost, stockoutCost]);

  // Generate data curve
  const curveData = useMemo(() => {
    const data = [];
    // Calculate cost curves for service levels from 50% to 99%
    for (let sl = 0.5; sl <= 0.99; sl += 0.05) {
      // Approximate Z-score mapping
      let z = 0;
      if (sl >= 0.99) z = 2.33;
      else if (sl >= 0.95) z = 1.645;
      else if (sl >= 0.90) z = 1.28;
      else if (sl >= 0.85) z = 1.04;
      else if (sl >= 0.80) z = 0.84;
      else if (sl >= 0.70) z = 0.52;
      else if (sl >= 0.60) z = 0.25;
      else z = 0;

      const point: Record<string, string | number> = { serviceLevel: (sl * 100).toFixed(0) + '%' };
      
      models.forEach(m => {
        // Mock math to show curve shape (Holding + Expected Shortage)
        const holding = z * m.rmse * Math.sqrt(leadTime) * holdingCost;
        // The shortage decreases as SL increases
        const shortage = ((1 - sl) * m.rmse * 2) * stockoutCost; 
        point[m.name] = Math.round(holding + shortage);
      });
      data.push(point);
    }
    return data;
  }, [holdingCost, stockoutCost, leadTime]);

  return (
    <SidebarLayout>
      <div className="p-8">
        <div className="max-w-7xl mx-auto space-y-10">
          
          <header className="animate-fade-in">
            <h1 className="text-4xl font-extrabold tracking-tight text-slate-900">Sensitivity Analysis</h1>
            <p className="text-slate-500 mt-2 font-medium max-w-2xl">
              Simulate the financial impact of changing inventory policy parameters on total logistics costs.
            </p>
          </header>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            
            {/* Control Panel */}
            <div className="glass-panel p-6 rounded-3xl animate-fade-in" style={{ animationDelay: '100ms' }}>
              <div className="flex items-center gap-2 mb-6">
                <Settings2 className="w-5 h-5 text-slate-600" />
                <h2 className="text-lg font-bold text-slate-900">Cost Parameters</h2>
              </div>
              
              <div className="space-y-6">
                <div>
                  <InfoTooltip title="Holding Cost Unit ($)" content="The cost to store one unit of inventory for one day. Higher holding costs push optimal service levels down (keep less stock).">
                    <div className="flex justify-between mb-2">
                      <label className="text-sm font-semibold text-slate-700">Holding Cost Unit ($)</label>
                      <span className="text-sm font-bold text-blue-600">${holdingCost.toFixed(2)}</span>
                    </div>
                  </InfoTooltip>
                  <input
                    type="range" min="0.1" max="5.0" step="0.1"
                    value={holdingCost}
                    onChange={(e) => setHoldingCost(parseFloat(e.target.value))}
                    className="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
                  />
                </div>

                <div>
                  <InfoTooltip title="Stockout Penalty ($)" content="The cost incurred when a customer wants to buy an item, but it's out of stock (e.g. lost profit margin or customer goodwill). Higher stockout penalties push optimal service levels up.">
                    <div className="flex justify-between mb-2">
                      <label className="text-sm font-semibold text-slate-700">Stockout Penalty ($)</label>
                      <span className="text-sm font-bold text-rose-600">${stockoutCost.toFixed(2)}</span>
                    </div>
                  </InfoTooltip>
                  <input
                    type="range" min="1.0" max="50.0" step="1.0"
                    value={stockoutCost}
                    onChange={(e) => setStockoutCost(parseFloat(e.target.value))}
                    className="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-rose-600"
                  />
                </div>

                <div>
                  <InfoTooltip title="Lead Time (Days)" content="The number of days it takes for an order to arrive after being placed. Longer lead times require exponentially more safety stock to cover uncertainty.">
                    <div className="flex justify-between mb-2">
                      <label className="text-sm font-semibold text-slate-700">Lead Time (Days)</label>
                      <span className="text-sm font-bold text-emerald-600">{leadTime}</span>
                    </div>
                  </InfoTooltip>
                  <input
                    type="range" min="1" max="14" step="1"
                    value={leadTime}
                    onChange={(e) => setLeadTime(parseInt(e.target.value))}
                    className="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-emerald-600"
                  />
                </div>
              </div>

              <div className="mt-8 bg-blue-50 border border-blue-100 rounded-2xl p-5">
                <h3 className="text-xs font-bold text-blue-800 uppercase tracking-wider mb-2">Theoretical Optimal</h3>
                <div className="flex items-end gap-2">
                  <span className="text-3xl font-extrabold text-blue-600">{(criticalRatio * 100).toFixed(1)}%</span>
                  <span className="text-sm text-blue-600 font-medium mb-1">Service Level</span>
                </div>
                <p className="text-xs text-blue-700 mt-2">
                  Based on the Newsvendor Critical Ratio: Cu / (Cu + Co)
                </p>
              </div>
            </div>

            {/* Chart Area */}
            <div className="lg:col-span-2 glass-panel p-6 rounded-3xl flex flex-col animate-fade-in" style={{ animationDelay: '200ms' }}>
              <div className="flex items-center justify-between mb-6">
                <div>
                  <h2 className="text-lg font-bold text-slate-900">Cost Trade-off Curve</h2>
                  <p className="text-sm text-slate-500">Total cost vs Target Service Level</p>
                </div>
                <div className="flex gap-4">
                  {models.map(m => (
                    <div key={m.name} className="flex items-center gap-1.5">
                      <div className="w-3 h-3 rounded-full" style={{ backgroundColor: m.color }} />
                      <span className="text-xs font-bold text-slate-600">{m.name}</span>
                    </div>
                  ))}
                </div>
              </div>

              <div className="flex-1 min-h-[300px]">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={curveData} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e2e8f0" />
                    <XAxis dataKey="serviceLevel" tick={{ fill: '#64748b', fontSize: 12 }} tickLine={false} />
                    <YAxis tick={{ fill: '#64748b', fontSize: 12 }} tickLine={false} axisLine={false} tickFormatter={(value) => `$${value}`} />
                    <Tooltip 
                      contentStyle={{ borderRadius: '12px', border: 'none', boxShadow: '0 10px 15px -3px rgb(0 0 0 / 0.1)' }}
                      // eslint-disable-next-line @typescript-eslint/no-explicit-any
                      formatter={(value: any) => [`$${value}`, "Total Cost"]}
                    />
                    {models.map(m => (
                      <Line 
                        key={m.name}
                        type="monotone" 
                        dataKey={m.name} 
                        stroke={m.color} 
                        strokeWidth={3} 
                        dot={false}
                        activeDot={{ r: 6, strokeWidth: 0 }}
                      />
                    ))}
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>

          </div>

          {/* Insights Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 animate-fade-in" style={{ animationDelay: '300ms' }}>
            <div className="bg-white/60 border border-slate-200 rounded-2xl p-6 flex gap-4">
              <div className="bg-amber-100 p-3 rounded-xl h-fit">
                <AlertTriangle className="w-6 h-6 text-amber-600" />
              </div>
              <div>
                <h3 className="text-sm font-bold text-slate-900 mb-1">High Stockout Environments</h3>
                <p className="text-sm text-slate-600 leading-relaxed">
                  When stockout penalties are severe (e.g., $50 vs $1), the cost curve becomes highly asymmetric. The penalty for under-forecasting dwarfs over-forecasting, pushing the optimal service level near 99%. In these environments, models that slightly over-forecast (lower bias) may outperform mathematically &quot;more accurate&quot; models.
                </p>
              </div>
            </div>

            <div className="bg-white/60 border border-slate-200 rounded-2xl p-6 flex gap-4">
              <div className="bg-emerald-100 p-3 rounded-xl h-fit">
                <ArrowRightLeft className="w-6 h-6 text-emerald-600" />
              </div>
              <div>
                <h3 className="text-sm font-bold text-slate-900 mb-1">The Margin of Error</h3>
                <p className="text-sm text-slate-600 leading-relaxed">
                  Notice the gap between LightGBM and LSTM at the optimal point (lowest point on the curve). The vertical difference represents the literal financial impact of model accuracy. A model with lower RMSE flattens the curve, making the business more resilient to sub-optimal parameter choices.
                </p>
              </div>
            </div>

            <div className="md:col-span-2 glass-panel border border-blue-200/50 bg-gradient-to-r from-slate-50/50 to-blue-50/50 rounded-2xl p-6 flex gap-4">
              <div className="bg-blue-100 p-3 rounded-xl h-fit">
                <BookOpen className="w-6 h-6 text-blue-600" />
              </div>
              <div>
                <h3 className="text-sm font-bold text-slate-900 mb-2">Theory: The Newsvendor Critical Ratio</h3>
                <p className="text-sm text-slate-600 leading-relaxed mb-3">
                  In inventory management, deciding exactly how much to stock is a balance between the cost of ordering too much (Holding Cost, <strong>C<sub className="text-[10px]">h</sub></strong>) and ordering too little (Stockout Cost, <strong>C<sub className="text-[10px]">s</sub></strong>). The <strong>Critical Ratio</strong> gives us the mathematically optimal service level.
                </p>
                <div className="bg-white/80 px-4 py-3 rounded-lg border border-blue-100 font-mono text-sm text-center text-blue-800 font-semibold inline-block">
                  Optimal Service Level = C<sub className="text-xs">s</sub> / (C<sub className="text-xs">s</sub> + C<sub className="text-xs">h</sub>)
                </div>
                <p className="text-sm text-slate-600 leading-relaxed mt-3">
                  This theory dictates that if a stockout is 10x more expensive than holding an item, your service level should naturally be extremely high (over 90%), and your models should be evaluated on how well they perform <em>at that specific threshold</em> rather than just their general mean error.
                </p>
              </div>
            </div>
          </div>

        </div>
      </div>
    </SidebarLayout>
  );
}