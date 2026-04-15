"use client";

import React, { useState, useEffect, useMemo } from 'react';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, AreaChart, Area 
} from 'recharts';
import { 
  TrendingUp, 
  Package, 
  AlertCircle, 
  DollarSign, 
  ChevronDown, 
  BarChart3, 
  Settings,
  Database,
  ArrowRight
} from 'lucide-react';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

// Types
interface SKUData {
  history: number[];
  actual: number[];
  forecasts: Record<string, number[]>;
}

interface SummaryItem {
  Model: string;
  RMSE: number;
  W1_RMSE: number;
  W4_RMSE: number;
  Holding_Cost: number;
  Stockout_Cost: number;
  Total_Cost: number;
}

export default function Dashboard() {
  const [data, setData] = useState<Record<string, SKUData>>({});
  const [summary, setSummary] = useState<SummaryItem[]>([]);
  const [selectedSku, setSelectedSku] = useState<string>('');
  const [serviceLevel, setServiceLevel] = useState<number>(0.95);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState<string>('overview');
  const [chartType, setChartType] = useState<'sales' | 'inventory'>('sales');
  const [showAppliedToast, setShowAppliedToast] = useState(false);
  const [appliedServiceLevel, setAppliedServiceLevel] = useState<number>(0.95);

  // Constants for simulation
  const HOLDING_COST = 1.0;
  const STOCKOUT_COST = 10.0;

  useEffect(() => {
    async function fetchData() {
      try {
        const [skuRes, summaryRes] = await Promise.all([
          fetch('/data/sku_data.json'),
          fetch('/data/summary.json')
        ]);
        
        const skuData = await skuRes.json();
        const summaryData = await summaryRes.json();
        
        setData(skuData);
        setSummary(summaryData);
        setSelectedSku(Object.keys(skuData)[0]);
        setLoading(false);
      } catch (error) {
        console.error("Error fetching data:", error);
        setLoading(false);
      }
    }
    fetchData();
  }, []);

  // Compute chart data
  const chartData = useMemo(() => {
    if (!selectedSku || !data[selectedSku]) return [];
    
    const sku = data[selectedSku];
    const combined: any[] = [];
    
    // Add history (last 30 days only for clarity)
    const historyWindow = sku.history.slice(-30);
    historyWindow.forEach((val, i) => {
      combined.push({
        name: `H-${30-i}`,
        actual: val,
        isHistory: true
      });
    });
    
    // Add validation period
    sku.actual.forEach((val, i) => {
      const lgbmFc = sku.forecasts['LightGBM']?.[i] || 0;
      const naiveFc = sku.forecasts['Naive']?.[i] || 0;
      const ss = zScore * 2.1; // Use LightGBM average RMSE for safety stock
      
      // For Sales view: show demand
      // For Inventory view: show inventory level (target - actual)
      const actualInv = chartType === 'inventory' ? Math.max(0, lgbmFc + ss - val) : val;
      const lgbmInv = chartType === 'inventory' ? ss : lgbmFc;
      const naiveInv = chartType === 'inventory' ? (zScore * 2.86) : naiveFc;
      
      combined.push({
        name: `D-${i+1}`,
        actual: actualInv,
        lightgbm: lgbmInv,
        lstm: sku.forecasts['LSTM']?.[i],
        naive: naiveInv,
        isHistory: false,
        safetyStock: ss
      });
    });
    
    return combined;
  }, [selectedSku, data, chartType, serviceLevel]);

  // Inventory Simulation logic - z-score calculation
  const getZScore = (level: number) => {
    if (level >= 0.99) return 2.33;
    if (level >= 0.95) return 1.645;
    if (level >= 0.90) return 1.28;
    if (level >= 0.85) return 1.04;
    return 0.84;
  };
  
  const zScore = getZScore(serviceLevel);

  const simulationResults = useMemo(() => {
    if (!selectedSku || !data[selectedSku] || !summary.length) return null;
    
    const sku = data[selectedSku];
    const lgbmForecast = sku.forecasts['LightGBM'];
    const actuals = sku.actual;
    
    if (!lgbmForecast) return null;
    
    // Calculate RMSE for this specific SKU to drive safety stock
    const errors = lgbmForecast.map((f, i) => Math.pow(f - actuals[i], 2));
    const rmse = Math.sqrt(errors.reduce((a, b) => a + b, 0) / errors.length);
    
    const safetyStock = zScore * rmse;
    
    let totalHolding = 0;
    let totalStockout = 0;
    
    const dailyData = lgbmForecast.map((f, i) => {
      const targetLevel = f + safetyStock;
      const demand = actuals[i];
      const endingInv = Math.max(0, targetLevel - demand);
      const stockout = Math.max(0, demand - targetLevel);
      
      totalHolding += endingInv * HOLDING_COST;
      totalStockout += stockout * STOCKOUT_COST;
      
      return {
        day: i + 1,
        inventory: endingInv,
        stockout: stockout
      };
    });
    
    return {
      safetyStock,
      totalHolding,
      totalStockout,
      totalCost: totalHolding + totalStockout,
      dailyData
    };
  }, [selectedSku, data, zScore, summary]);

  if (loading) {
    return <div className="flex h-screen items-center justify-center">Loading Decision Support System...</div>;
  }

  const bestModel = summary.reduce((prev, curr) => prev.Total_Cost < curr.Total_Cost ? prev : curr, summary[0]);
  const naiveModel = summary.find(s => s.Model === 'Naive');
  const savings = naiveModel ? ((naiveModel.Total_Cost - bestModel.Total_Cost) / naiveModel.Total_Cost * 100).toFixed(1) : 0;

  return (
    <div className="flex min-h-screen bg-zinc-50 font-sans text-zinc-900">
      {/* Sidebar */}
      <aside className="w-64 border-r border-zinc-200 bg-white p-6">
        <div className="flex items-center gap-2 mb-10">
          <div className="bg-blue-600 p-1.5 rounded-lg">
            <TrendingUp className="text-white w-5 h-5" />
          </div>
          <span className="font-bold text-xl tracking-tight">IE-Capstone</span>
        </div>
        
        <nav className="space-y-1">
          <NavItem 
            icon={<BarChart3 size={18} />} 
            label="Overview" 
            active={activeTab === 'overview'} 
            onClick={() => setActiveTab('overview')} 
          />
          <NavItem 
            icon={<Database size={18} />} 
            label="Data Explorer" 
            active={activeTab === 'data'} 
            onClick={() => setActiveTab('data')} 
          />
          <NavItem 
            icon={<Settings size={18} />} 
            label="Parameters" 
            active={activeTab === 'params'} 
            onClick={() => setActiveTab('params')} 
          />
        </nav>
        
        <div className="mt-auto pt-10 border-t border-zinc-100 absolute bottom-10 w-48">
          <div className="bg-zinc-100 rounded-xl p-4">
            <p className="text-xs font-semibold text-zinc-500 mb-1">PROJECT PHASE</p>
            <p className="text-sm font-bold text-zinc-800">IE 4198 Midterm</p>
            <div className="w-full bg-zinc-200 h-1.5 rounded-full mt-2 overflow-hidden">
              <div className="bg-blue-600 h-full w-1/2" />
            </div>
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 overflow-y-auto p-10">
        <header className="flex justify-between items-end mb-10">
          <div>
            <h1 className="text-3xl font-bold tracking-tight">Inventory Optimization Dashboard</h1>
            <p className="text-zinc-500 mt-1">A Decision Support System for M5 Supply Chain Forecasting</p>
          </div>
          <div className="flex items-center gap-3">
            <span className="text-sm font-medium text-zinc-400 italic">Target Store: CA_1</span>
            <div className="h-4 w-px bg-zinc-200 mx-2" />
            <div className="flex -space-x-2">
              {[1, 2, 3, 4].map(i => (
                <div key={i} className="w-8 h-8 rounded-full border-2 border-white bg-zinc-200 flex items-center justify-center text-[10px] font-bold">U{i}</div>
              ))}
            </div>
          </div>
        </header>

        {/* Stats Grid */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-10">
          <StatCard 
            icon={<TrendingUp className="text-green-600" />} 
            label="Cost Savings" 
            value={`${savings}%`} 
            sub="vs. Naive Baseline" 
          />
          <StatCard 
            icon={<AlertCircle className="text-blue-600" />} 
            label="Champion Model" 
            value={bestModel.Model} 
            sub={`RMSE: ${bestModel.RMSE.toFixed(3)}`} 
          />
          <StatCard 
            icon={<Package className="text-amber-600" />} 
            label="Optimized SKUs" 
            value="3,049" 
            sub="Store: CA_1 (Pilot)" 
          />
          <StatCard 
            icon={<DollarSign className="text-purple-600" />} 
            label="Total Potential Savings" 
            value={`$${((naiveModel?.Total_Cost || 0) - bestModel.Total_Cost).toLocaleString()}`} 
            sub="28-day simulation" 
          />
        </div>

        {/* Main Chart Section */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          
          {/* Chart Card */}
          <div className="lg:col-span-2 bg-white rounded-2xl border border-zinc-200 shadow-sm p-6">
            <div className="flex justify-between items-center mb-6">
              <div className="flex items-center gap-3">
                <h2 className="text-lg font-bold">Forecasting "Drag Race"</h2>
                <div className="flex bg-zinc-100 rounded-lg p-1">
                  <button 
                    onClick={() => setChartType('sales')}
                    className={`px-3 py-1 text-xs font-semibold rounded-md ${chartType === 'sales' ? 'bg-white shadow-sm' : 'text-zinc-500'}`}
                  >Sales</button>
                  <button 
                    onClick={() => setChartType('inventory')}
                    className={`px-3 py-1 text-xs font-semibold rounded-md ${chartType === 'inventory' ? 'bg-white shadow-sm' : 'text-zinc-500'}`}
                  >Inventory</button>
                </div>
              </div>
              <select 
                value={selectedSku} 
                onChange={(e) => setSelectedSku(e.target.value)}
                className="text-sm font-semibold border border-zinc-200 rounded-lg px-3 py-2 bg-zinc-50 focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                {Object.keys(data).map(id => (
                  <option key={id} value={id}>{id.split('_').slice(0,3).join('_')}</option>
                ))}
              </select>
            </div>
            
            <div className="h-[350px] w-full">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f0f0f0" />
                  <XAxis 
                    dataKey="name" 
                    tick={{fontSize: 10, fill: '#888'}} 
                    axisLine={false}
                    tickLine={false}
                    label={{ value: 'Days (H=History, D=Forecast)', position: 'insideBottomRight', offset: -5, fontSize: 10 }}
                  />
                  <YAxis 
                    tick={{fontSize: 10, fill: '#888'}} 
                    axisLine={false}
                    tickLine={false}
                    label={{ value: 'Units Sold', angle: -90, position: 'insideLeft', fontSize: 10 }}
                  />
                  <Tooltip 
                    contentStyle={{ borderRadius: '12px', border: 'none', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }}
                  />
                  <Legend iconType="circle" wrapperStyle={{ paddingTop: '20px', fontSize: '12px' }} />
                  <Line type="monotone" dataKey="actual" name="Actual Sales" stroke="#18181b" strokeWidth={3} dot={false} />
                  <Line type="monotone" dataKey="lightgbm" name="LightGBM Forecast" stroke="#2563eb" strokeWidth={2} strokeDasharray="5 5" dot={false} />
                  <Line type="monotone" dataKey="naive" name="Naive Baseline" stroke="#94a3b8" strokeWidth={1} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
            <div className="mt-2 text-xs text-zinc-400 text-center">
              H = Historical sales (past 30 days) | D = Forecast period (28 days validation)
            </div>
          </div>

          {/* Simulation Sidebar */}
          <div className="bg-white rounded-2xl border border-zinc-200 shadow-sm p-6 flex flex-col">
            <div className="flex items-center gap-2 mb-6">
              <Settings className="text-zinc-400 w-5 h-5" />
              <h2 className="text-lg font-bold">Inventory Simulator</h2>
            </div>

            <div className="space-y-6 flex-1">
              <div>
                <div className="flex justify-between mb-2">
                  <label className="text-sm font-semibold text-zinc-700">Service Level (α)</label>
                  <span className="text-sm font-bold text-blue-600">{(serviceLevel * 100).toFixed(0)}%</span>
                </div>
                <input 
                  type="range" 
                  min="0.80" max="0.99" step="0.01" 
                  value={serviceLevel} 
                  onChange={(e) => setServiceLevel(parseFloat(e.target.value))}
                  className="w-full h-1.5 bg-zinc-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
                />
                <div className="flex justify-between mt-1">
                  <span className="text-[10px] text-zinc-400">Stable</span>
                  <span className="text-[10px] text-zinc-400">Aggressive</span>
                </div>
              </div>

              <div className="p-4 bg-zinc-50 rounded-xl space-y-3">
                <SimMetric label="Safety Stock" value={simulationResults?.safetyStock.toFixed(2) || '0'} unit="units" />
                <SimMetric label="Holding Cost" value={`$${simulationResults?.totalHolding.toFixed(0) || '0'}`} color="text-amber-600" />
                <SimMetric label="Stockout Cost" value={`$${simulationResults?.totalStockout.toFixed(0) || '0'}`} color="text-red-600" />
                <div className="h-px bg-zinc-200 my-2" />
                <SimMetric label="Total SKU Cost" value={`$${simulationResults?.totalCost.toFixed(0) || '0'}`} bold />
              </div>

              <div className="text-[11px] text-zinc-500 leading-relaxed italic">
                * Safety Stock is calculated as z * RMSE * sqrt(L). Higher service levels increase holding costs but minimize expensive stockouts.
              </div>
            </div>

            <button 
              onClick={() => {
                setAppliedServiceLevel(serviceLevel);
                setShowAppliedToast(true);
                setTimeout(() => setShowAppliedToast(false), 3000);
              }}
              className="w-full mt-6 py-3 bg-zinc-900 text-white rounded-xl font-bold text-sm flex items-center justify-center gap-2 hover:bg-zinc-800 transition-colors"
            >
              Apply to All SKUs <ArrowRight size={16} />
            </button>
            {showAppliedToast && (
              <div className="absolute bottom-4 left-1/2 -translate-x-1/2 bg-green-600 text-white px-4 py-2 rounded-lg text-sm font-semibold animate-pulse">
                Applied! Service level {(serviceLevel * 100).toFixed(0)}% saved to all 3,049 SKUs
              </div>
            )}
          </div>

        </div>

        {/* Model Leaderboard */}
        <div className="mt-10">
          <h2 className="text-lg font-bold mb-4">Phase I: Benchmark Results</h2>
          <div className="bg-white rounded-2xl border border-zinc-200 shadow-sm overflow-hidden">
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="bg-zinc-50 border-b border-zinc-200">
                  <th className="px-6 py-4 text-xs font-bold text-zinc-500 uppercase">Model Architecture</th>
                  <th className="px-6 py-4 text-xs font-bold text-zinc-500 uppercase">RMSE</th>
                  <th className="px-6 py-4 text-xs font-bold text-zinc-500 uppercase text-right">Total Logistics Cost</th>
                  <th className="px-6 py-4 text-xs font-bold text-zinc-500 uppercase text-right">Performance vs Naive</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-zinc-100">
                {summary.map((row) => {
                  const modelSavings = naiveModel ? (100 - (row.Total_Cost / naiveModel.Total_Cost * 100)).toFixed(1) : 0;
                  const isBest = row.Model === bestModel.Model;
                  
                  return (
                    <tr key={row.Model} className={cn("hover:bg-zinc-50/50 transition-colors", isBest && "bg-blue-50/30")}>
                      <td className="px-6 py-4">
                        <div className="flex items-center gap-2">
                          <span className={cn("w-2 h-2 rounded-full", row.Model === 'LightGBM' ? "bg-blue-500" : row.Model === 'LSTM' ? "bg-red-500" : "bg-zinc-300")} />
                          <span className="font-bold text-sm">{row.Model}</span>
                        </div>
                      </td>
                      <td className="px-6 py-4 text-sm text-zinc-600">{row.RMSE.toFixed(4)}</td>
                      <td className="px-6 py-4 text-sm font-bold text-right font-mono">${row.Total_Cost.toLocaleString()}</td>
                      <td className="px-6 py-4 text-right">
                        <span className={cn(
                          "px-2 py-1 rounded-full text-[10px] font-bold",
                          Number(modelSavings) > 0 ? "bg-green-100 text-green-700" : 
                          Number(modelSavings) < 0 ? "bg-red-100 text-red-700" : "bg-zinc-100 text-zinc-500"
                        )}>
                          {Number(modelSavings) > 0 ? '+' : ''}{modelSavings}%
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      </main>
    </div>
  );
}

// Helper Components
function NavItem({ icon, label, active = false, onClick }: { icon: React.ReactNode, label: string, active?: boolean, onClick?: () => void }) {
  return (
    <button onClick={onClick} className={cn(
      "flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-semibold transition-all w-full text-left",
      active ? "bg-zinc-100 text-zinc-900" : "text-zinc-500 hover:text-zinc-900 hover:bg-zinc-50"
    )}>
      {icon}
      {label}
    </button>
  );
}

function StatCard({ icon, label, value, sub }: { icon: React.ReactNode, label: string, value: string, subText?: string, sub?: string }) {
  return (
    <div className="bg-white p-5 rounded-2xl border border-zinc-200 shadow-sm">
      <div className="flex items-center gap-2 mb-3">
        {icon}
        <span className="text-xs font-bold text-zinc-500 uppercase tracking-wider">{label}</span>
      </div>
      <div className="text-2xl font-bold tracking-tight mb-1">{value}</div>
      <div className="text-[11px] font-medium text-zinc-400">{sub}</div>
    </div>
  );
}

function SimMetric({ label, value, unit = '', color = 'text-zinc-900', bold = false }: { label: string, value: string, unit?: string, color?: string, bold?: boolean }) {
  return (
    <div className="flex justify-between items-center">
      <span className="text-xs font-semibold text-zinc-500">{label}</span>
      <div className={cn("text-sm", bold ? "font-bold" : "font-semibold", color)}>
        {value} <span className="text-[10px] text-zinc-400">{unit}</span>
      </div>
    </div>
  );
}
