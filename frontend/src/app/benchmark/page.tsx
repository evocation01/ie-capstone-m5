"use client";

import React, { useState, useEffect } from 'react';
import SidebarLayout from '@/components/SidebarLayout';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer, Cell } from 'recharts';
import { Trophy, ArrowDownRight, ArrowUpRight, Zap, Target } from 'lucide-react';
import InfoTooltip from '@/components/InfoTooltip';

interface SummaryItem {
  Model: string;
  RMSE: number;
  W1_RMSE: number;
  W4_RMSE: number;
  Holding_Cost: number;
  Stockout_Cost: number;
  Total_Cost: number;
}

export default function BenchmarkPage() {
  const [summary, setSummary] = useState<SummaryItem[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch('/data/summary.json')
      .then(res => res.json())
      .then(data => {
        // Sort by RMSE ascending
        const sorted = [...data].sort((a, b) => a.RMSE - b.RMSE);
        setSummary(sorted);
        setLoading(false);
      })
      .catch(err => {
        console.error(err);
        setLoading(false);
      });
  }, []);

  if (loading) {
    return (
      <SidebarLayout>
        <div className="flex items-center justify-center h-screen">
          <div className="animate-pulse flex flex-col items-center gap-4">
            <div className="w-12 h-12 border-4 border-blue-500 border-t-transparent rounded-full animate-spin"></div>
            <p className="text-slate-500 font-semibold">Loading Benchmark Data...</p>
          </div>
        </div>
      </SidebarLayout>
    );
  }

  const bestModel = summary[0];

  return (
    <SidebarLayout>
      <div className="p-8">
        <div className="max-w-7xl mx-auto space-y-10">
          
          <header className="animate-fade-in">
            <h1 className="text-4xl font-extrabold tracking-tight text-slate-900">Benchmark Results</h1>
            <p className="text-slate-500 mt-2 font-medium max-w-2xl">
              Comprehensive evaluation of 10+ forecasting models across 3 tiers (Classical, ML, Deep Learning) 
              evaluated on the CA_1 store subset over a 28-day validation period.
            </p>
          </header>

          {/* Top Level Insights */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 animate-fade-in" style={{ animationDelay: '100ms' }}>
            <div className="glass-panel p-6 rounded-2xl relative overflow-hidden group">
              <div className="absolute top-0 right-0 -mr-4 -mt-4 w-24 h-24 bg-gradient-to-br from-amber-200 to-yellow-400 rounded-full blur-2xl opacity-50 group-hover:opacity-70 transition-opacity" />
              <Trophy className="w-8 h-8 text-amber-500 mb-4 relative z-10" />
              <h3 className="text-sm font-bold text-slate-500 uppercase tracking-wider relative z-10">Overall Champion</h3>
              <p className="text-3xl font-extrabold text-slate-900 mt-1 relative z-10">{bestModel?.Model}</p>
              <p className="text-sm font-medium text-slate-600 mt-2 relative z-10">Achieved lowest RMSE ({bestModel?.RMSE.toFixed(3)})</p>
            </div>
            
            <div className="glass-panel p-6 rounded-2xl relative overflow-hidden group">
              <div className="absolute top-0 right-0 -mr-4 -mt-4 w-24 h-24 bg-gradient-to-br from-emerald-200 to-green-400 rounded-full blur-2xl opacity-50 group-hover:opacity-70 transition-opacity" />
              <Zap className="w-8 h-8 text-emerald-500 mb-4 relative z-10" />
              <h3 className="text-sm font-bold text-slate-500 uppercase tracking-wider relative z-10">Max Cost Reduction</h3>
              <p className="text-3xl font-extrabold text-slate-900 mt-1 relative z-10">19.5%</p>
              <p className="text-sm font-medium text-slate-600 mt-2 relative z-10">Savings vs Naive Baseline</p>
            </div>

            <div className="glass-panel p-6 rounded-2xl relative overflow-hidden group">
              <div className="absolute top-0 right-0 -mr-4 -mt-4 w-24 h-24 bg-gradient-to-br from-purple-200 to-indigo-400 rounded-full blur-2xl opacity-50 group-hover:opacity-70 transition-opacity" />
              <Target className="w-8 h-8 text-purple-500 mb-4 relative z-10" />
              <h3 className="text-sm font-bold text-slate-500 uppercase tracking-wider relative z-10">Deep Learning Focus</h3>
              <p className="text-3xl font-extrabold text-slate-900 mt-1 relative z-10">DeepAR</p>
              <p className="text-sm font-medium text-slate-600 mt-2 relative z-10">Solved the sparsity penalty</p>
            </div>
          </div>

          {/* Full Table */}
          <div className="glass-panel rounded-2xl overflow-hidden animate-fade-in" style={{ animationDelay: '200ms' }}>
            <div className="px-6 py-5 border-b border-slate-200/50 bg-white/50 flex justify-between items-center">
              <div>
                <h2 className="text-xl font-bold text-slate-900">The &quot;Drag Race&quot; Leaderboard</h2>
                <p className="text-sm text-slate-500 font-medium mt-1">Ranked by Root Mean Square Error (RMSE)</p>
              </div>
              <InfoTooltip title="Root Mean Square Error" content="RMSE measures the average magnitude of the forecasting errors. A lower RMSE means the model's predictions were closer to the actual sales.">
                <span className="text-xs font-bold text-blue-600 bg-blue-100 px-3 py-1 rounded-full">What is RMSE?</span>
              </InfoTooltip>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-left border-collapse">
                <thead>
                  <tr className="bg-slate-50/50">
                    <th className="px-6 py-4 text-xs font-bold text-slate-500 uppercase tracking-wider border-b border-slate-200">Rank</th>
                    <th className="px-6 py-4 text-xs font-bold text-slate-500 uppercase tracking-wider border-b border-slate-200">Model Name</th>
                    <th className="px-6 py-4 text-xs font-bold text-slate-500 uppercase tracking-wider border-b border-slate-200">Tier</th>
                    <th className="px-6 py-4 text-xs font-bold text-slate-500 uppercase tracking-wider border-b border-slate-200">RMSE</th>
                    <th className="px-6 py-4 text-xs font-bold text-slate-500 uppercase tracking-wider border-b border-slate-200">Total Cost</th>
                    <th className="px-6 py-4 text-xs font-bold text-slate-500 uppercase tracking-wider border-b border-slate-200">Performance vs Baseline</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {summary.map((item, idx) => {
                    const isBaseline = item.Model === 'Naive';
                    const baselineCost = summary.find(s => s.Model === 'Naive')?.Total_Cost || 640000;
                    const diff = ((baselineCost - item.Total_Cost) / baselineCost) * 100;
                    
                    let tier = 'Classical';
                    if (['LightGBM', 'XGBoost', 'RandomForest'].includes(item.Model)) tier = 'Machine Learning';
                    if (['LSTM', 'DeepAR'].includes(item.Model)) tier = 'Deep Learning';
                    if (isBaseline) tier = 'Baseline';

                    return (
                      <tr key={item.Model} className={`hover:bg-slate-50/50 transition-colors ${idx === 0 ? 'bg-blue-50/30' : ''}`}>
                        <td className="px-6 py-4">
                          <span className={`inline-flex items-center justify-center w-6 h-6 rounded-full text-xs font-bold ${idx === 0 ? 'bg-blue-100 text-blue-700' : 'bg-slate-100 text-slate-600'}`}>
                            {idx + 1}
                          </span>
                        </td>
                        <td className="px-6 py-4 font-bold text-slate-900">
                          {item.Model} {idx === 0 && <span className="ml-2 text-amber-500 text-sm">🏆</span>}
                        </td>
                        <td className="px-6 py-4">
                          <span className="px-2.5 py-1 rounded-md bg-slate-100 text-slate-600 text-xs font-semibold">
                            {tier}
                          </span>
                        </td>
                        <td className="px-6 py-4 font-mono font-semibold text-slate-700">
                          {item.RMSE.toFixed(3)}
                        </td>
                        <td className="px-6 py-4 font-mono font-semibold text-slate-700">
                          ${Math.round(item.Total_Cost).toLocaleString()}
                        </td>
                        <td className="px-6 py-4">
                          {isBaseline ? (
                            <span className="text-slate-400 font-semibold text-sm">--</span>
                          ) : (
                            <div className={`flex items-center gap-1 font-bold text-sm ${diff > 0 ? 'text-emerald-600' : 'text-red-500'}`}>
                              {diff > 0 ? <ArrowUpRight className="w-4 h-4" /> : <ArrowDownRight className="w-4 h-4" />}
                              {Math.abs(diff).toFixed(1)}%
                            </div>
                          )}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>

          {/* Bar Chart Visualization */}
          <div className="glass-panel rounded-2xl p-6 animate-fade-in" style={{ animationDelay: '300ms' }}>
            <h2 className="text-lg font-bold text-slate-900 mb-6">RMSE Comparison Chart</h2>
            <div className="h-80 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={summary} layout="vertical" margin={{ top: 5, right: 30, left: 40, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" horizontal={true} vertical={false} stroke="#e2e8f0" />
                  <XAxis type="number" domain={[0, 'dataMax + 0.5']} />
                  <YAxis dataKey="Model" type="category" axisLine={false} tickLine={false} tick={{ fill: '#475569', fontWeight: 600 }} />
                  <RechartsTooltip 
                    cursor={{fill: '#f1f5f9'}}
                    contentStyle={{ borderRadius: '12px', border: 'none', boxShadow: '0 10px 15px -3px rgb(0 0 0 / 0.1)' }}
                  />
                  <Bar dataKey="RMSE" radius={[0, 4, 4, 0]} barSize={24}>
                    {summary.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={index === 0 ? '#3b82f6' : entry.Model === 'Naive' ? '#94a3b8' : ['LSTM', 'DeepAR'].includes(entry.Model) ? '#8b5cf6' : '#cbd5e1'} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
          
        </div>
      </div>
    </SidebarLayout>
  );
}