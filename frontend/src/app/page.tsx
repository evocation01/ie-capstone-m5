"use client";

import React, { useState, useEffect, useMemo } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer
} from 'recharts';
import {
  TrendingUp,
  Package,
  AlertCircle,
  DollarSign,
  Settings,
  ArrowRight,
  BookOpen
} from 'lucide-react';
import InfoTooltip from '@/components/InfoTooltip';
import SidebarLayout from '@/components/SidebarLayout';
import { DashboardSkeleton } from '@/components/Skeleton';

function cn(...inputs: string[]) {
  return inputs.filter(Boolean).join(' ');
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
  Overstock_Units?: number;
  Understock_Units?: number;
  Note?: string;
}

export default function DashboardPage() {
  const [data, setData] = useState<Record<string, SKUData>>({});
  const [summary, setSummary] = useState<SummaryItem[]>([]);
  const [selectedSku, setSelectedSku] = useState<string>('');
  const [serviceLevel, setServiceLevel] = useState<number>(0.95);
  const [holdingCost, setHoldingCost] = useState<number>(1.00);
  const [stockoutCost, setStockoutCost] = useState<number>(10.00);
  const [leadTime] = useState<number>(1);
  const [loading, setLoading] = useState(true);
  const [chartType, setChartType] = useState<'sales' | 'inventory'>('sales');
  const [shapData, setShapData] = useState<Array<{feature: string, importance: number}>>([]);

  // Load data on mount
  useEffect(() => {
    const loadData = async () => {
      try {
        const [skuRes, summaryRes, shapRes] = await Promise.all([
          fetch('/data/sku_data.json'),
          fetch('/data/summary.json'),
          fetch('/data/shap_importance.json').catch(() => ({ json: () => [] }))
        ]);

        const skuData = await skuRes.json();
        const summaryData = await summaryRes.json();
        const shapJson = await (shapRes as Response).json();

        setData(skuData);
        setSummary(summaryData);
        setShapData(shapJson);

        // Set first SKU as default
        const firstSku = Object.keys(skuData)[0];
        setSelectedSku(firstSku);
      } catch (error) {
        console.error('Error loading data:', error);
      } finally {
        setLoading(false);
      }
    };

    loadData();
  }, []);

  // Compute simulation results
  const simulationResults = useMemo(() => {
    if (!selectedSku || !data[selectedSku] || !summary.length) return null;

    const sku = data[selectedSku];
    const lgbmForecast = sku.forecasts['LightGBM'];

    // Calculate safety stock using LightGBM RMSE
    const lgbmModel = summary.find(s => s.Model === 'LightGBM');
    if (!lgbmModel || !lgbmForecast) return null;

    const zScore = serviceLevel === 0.8 ? 1.28 : serviceLevel === 0.9 ? 1.645 : serviceLevel === 0.95 ? 1.645 : serviceLevel === 0.99 ? 2.33 : 1.645;
    const safetyStock = zScore * lgbmModel.RMSE * Math.sqrt(leadTime);

    // Calculate costs
    const totalHolding = safetyStock * holdingCost;
    const expectedUnitsShort = (1 - serviceLevel) * lgbmForecast.reduce((a, b) => a + b, 0) / lgbmForecast.length;
    const totalStockout = expectedUnitsShort * stockoutCost;
    const totalCost = totalHolding + totalStockout;

    return {
      safetyStock,
      totalHolding,
      totalStockout,
      totalCost
    };
  }, [selectedSku, data, summary, serviceLevel, holdingCost, stockoutCost, leadTime]);

  // Compute chart data
  const chartData = useMemo(() => {
    if (!selectedSku || !data[selectedSku]) return [];

    const sku = data[selectedSku];
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const combined: Array<Record<string, any>> = [];

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
    const getLocalZScore = (level: number) => {
      if (level >= 0.99) return 2.33;
      if (level >= 0.95) return 1.645;
      if (level >= 0.90) return 1.645;
      return 1.28;
    };

    const ss = getLocalZScore(serviceLevel) * 2.1; // Use LightGBM average RMSE for safety stock

    if (sku.actual && sku.actual.length > 0) {
      sku.actual.forEach((act, i) => {
        combined.push({
          name: `D+${i+1}`,
          actual: act,
          lightgbm: sku.forecasts['LightGBM']?.[i] ?? 0,
          lstm: sku.forecasts['LSTM']?.[i] ?? 0,
          deepar: sku.forecasts['DeepAR']?.[i] ?? 0,
          naive: sku.forecasts['Naive']?.[i] ?? 0,
          isHistory: false,
          safetyStock: ss
        });
      });
    }

    return combined;
  }, [selectedSku, data, serviceLevel]);

  // Recalculate summary based on dynamic holding/stockout costs
  const dynamicSummary = useMemo(() => {
    return summary.map(item => {
      // If the backend didn't provide units, fallback to original cost
      if (item.Overstock_Units === undefined || item.Understock_Units === undefined) {
        return item;
      }
      
      const newHoldingCost = item.Overstock_Units * holdingCost;
      const newStockoutCost = item.Understock_Units * stockoutCost;
      
      return {
        ...item,
        Holding_Cost: newHoldingCost,
        Stockout_Cost: newStockoutCost,
        Total_Cost: newHoldingCost + newStockoutCost
      };
    });
  }, [summary, holdingCost, stockoutCost]);

  const bestModel = dynamicSummary.reduce((prev, curr) => prev.Total_Cost < curr.Total_Cost ? prev : curr, dynamicSummary[0]);
  const naiveModel = dynamicSummary.find(s => s.Model === 'Naive');
  const savings = naiveModel ? ((naiveModel.Total_Cost - bestModel.Total_Cost) / naiveModel.Total_Cost * 100) : 0;
  
  const sortedSummary = [...dynamicSummary].sort((a, b) => a.Total_Cost - b.Total_Cost);
  
  const simMetrics = simulationResults || { safetyStock: 0, totalHolding: 0, totalStockout: 0, totalCost: 0 };

  if (loading) {
    return (
      <SidebarLayout>
        <DashboardSkeleton />
      </SidebarLayout>
    );
  }

  return (
    <SidebarLayout>
      <div className="p-8">
        <div className="max-w-7xl mx-auto">
          <header className="animate-fade-in flex flex-col md:flex-row md:items-start justify-between gap-4 mb-10">
            <div>
              <h1 className="text-4xl font-extrabold tracking-tight text-slate-900">Decision Support System</h1>
              <p className="text-slate-500 mt-2 font-medium max-w-2xl">
                Evaluate model predictions dynamically transformed into actionable inventory costs for the CA_1 store subset.
              </p>
            </div>
          </header>

          {/* New Educational Panel */}
          <div className="glass-panel p-6 rounded-2xl border border-blue-100/50 bg-gradient-to-r from-blue-50/50 to-indigo-50/50 animate-fade-in mb-10" style={{ animationDelay: '50ms' }}>
            <div className="flex items-start gap-4">
              <div className="bg-blue-100 p-3 rounded-xl mt-1">
                <BookOpen className="w-6 h-6 text-blue-600" />
              </div>
              <div>
                <h3 className="text-lg font-bold text-slate-900">How to Read This Dashboard</h3>
                <p className="text-sm text-slate-600 mt-1 max-w-4xl leading-relaxed">
                  This system connects <strong>statistical forecasting</strong> with <strong>financial impact</strong>. 
                  Use the <span className="font-semibold text-slate-800">Parameters</span> below to set the business environment (e.g. how expensive is a stockout vs holding stock?). 
                  The simulation then runs a 28-day inventory test using each model&apos;s predictions as the daily order quantity. 
                  The <strong>Champion Model</strong> is the one that results in the lowest Total Cost (Holding + Stockouts).
                </p>
              </div>
            </div>
          </div>

          {/* Stats Grid */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-10">
            <StatCard
              icon={<TrendingUp className="text-green-600" />}
              label="Cost Savings"
              value={`${savings.toFixed(1)}%`}
              sub="vs. Naive Baseline"
            />
            <StatCard
              icon={<AlertCircle className="text-blue-600" />}
              label="Champion Model"
              value={bestModel?.Model || 'N/A'}
              sub={`RMSE: ${bestModel?.RMSE.toFixed(3) || 'N/A'}`}
            />
            <StatCard
              icon={<Package className="text-purple-600" />}
              label="Items Optimized"
              value="3,049"
              sub="CA_1 Store SKUs"
            />
            <div className="glass-panel p-6 rounded-2xl relative overflow-hidden group">
              <div className="absolute top-0 right-0 -mr-4 -mt-4 w-24 h-24 bg-gradient-to-br from-emerald-200 to-teal-400 rounded-full blur-2xl opacity-50 group-hover:opacity-70 transition-opacity" />
              <DollarSign className="w-8 h-8 text-emerald-500 mb-4 relative z-10" />
              <InfoTooltip title="Total Cost" content="The sum of holding unsold inventory and penalties for missed sales over the 28-day period. Lower is better.">
                <h3 className="text-sm font-bold text-slate-500 uppercase tracking-wider relative z-10">Total Cost</h3>
              </InfoTooltip>
              <p className="text-3xl font-extrabold text-slate-900 mt-1 relative z-10">${Math.round(bestModel.Total_Cost).toLocaleString()}</p>
            </div>
          </div>

          {/* SKU Selector and Chart */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-10">
            {/* Chart */}
            <div className="lg:col-span-2 bg-white rounded-2xl border border-zinc-200 shadow-sm p-6">
              <div className="flex justify-between items-center mb-6">
                <div className="flex items-center gap-3">
                  <h2 className="text-lg font-bold text-slate-900">Forecasting &ldquo;Drag Race&rdquo;</h2>
                  <div className="flex bg-zinc-100 rounded-lg p-1">
                    <button
                      onClick={() => setChartType('sales')}
                      className={`px-3 py-1 text-xs font-semibold rounded-md ${chartType === 'sales' ? 'bg-white text-zinc-900 shadow-sm' : 'text-zinc-500 hover:text-zinc-700'}`}
                    >
                      Sales
                    </button>
                    <button
                      onClick={() => setChartType('inventory')}
                      className={`px-3 py-1 text-xs font-semibold rounded-md ${chartType === 'inventory' ? 'bg-white text-zinc-900 shadow-sm' : 'text-zinc-500 hover:text-zinc-700'}`}
                    >
                      Inventory
                    </button>
                  </div>
                </div>

                <select
                  value={selectedSku}
                  onChange={(e) => setSelectedSku(e.target.value)}
                  className="px-3 py-2 border border-zinc-200 rounded-lg text-sm bg-white text-zinc-900 font-medium"
                >
                  {Object.keys(data).map(sku => (
                    <option key={sku} value={sku}>{sku}</option>
                  ))}
                </select>
              </div>

              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" tick={{ fill: '#334155', fontWeight: 600 }} />
                  <YAxis tick={{ fill: '#334155', fontWeight: 600 }} />
                  <Tooltip />
                  <Legend />
                  <Line type="monotone" dataKey="actual" name="Actual Sales" stroke="#0f172a" strokeWidth={2.5} />
                  <Line type="monotone" dataKey="lightgbm" name="LightGBM" stroke="#2563eb" strokeWidth={2.5} strokeDasharray="5 5" dot={false} />
                  <Line type="monotone" dataKey="deepar" name="DeepAR" stroke="#ea580c" strokeWidth={2.5} strokeDasharray="5 5" dot={false} />
                  <Line type="monotone" dataKey="lstm" name="LSTM" stroke="#7c3aed" strokeWidth={2} strokeDasharray="8 8" dot={false} opacity={0.6} />
                  <Line type="monotone" dataKey="naive" name="Naive" stroke="#94a3b8" strokeWidth={1.5} dot={false} opacity={0.5} />
                </LineChart>
              </ResponsiveContainer>
            </div>

            {/* Inventory Simulator */}
            <div className="bg-white rounded-2xl border border-zinc-200 shadow-sm p-6 flex flex-col">
              <div className="flex items-center gap-2 mb-6">
                <Settings className="text-zinc-700 w-5 h-5" />
                <h2 className="text-lg font-bold text-slate-900">Inventory Simulator</h2>
              </div>

              <div className="space-y-6 flex-1">
                <div>
                  <InfoTooltip title="Service Level" content="The probability of not stocking out during the lead time. Higher values require more safety stock, increasing holding costs but reducing stockouts.">
                    <label className="text-sm font-semibold text-slate-700 block mb-3">Service Level: <span className="text-blue-600 font-bold">{Math.round(serviceLevel * 100)}%</span></label>
                  </InfoTooltip>
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

                <div className="grid grid-cols-2 gap-4 mt-6">
                  <div className="p-4 bg-slate-50 rounded-xl">
                    <InfoTooltip title="Holding Cost" content="The daily cost of storing one unsold unit. High holding costs penalize models that over-forecast.">
                      <div className="flex justify-between items-center mb-1">
                        <span className="text-xs font-bold text-slate-500 uppercase">Holding Cost</span>
                        <span className="text-sm font-bold text-slate-900">${holdingCost.toFixed(2)}</span>
                      </div>
                    </InfoTooltip>
                    <input
                      type="range"
                      min="0.1"
                      max="5.0"
                      step="0.1"
                      value={holdingCost}
                      onChange={(e) => setHoldingCost(parseFloat(e.target.value))}
                      className="w-full h-1.5 bg-zinc-200 rounded-lg appearance-none cursor-pointer accent-emerald-600"
                    />
                  </div>
                  <div className="p-4 bg-slate-50 rounded-xl">
                    <InfoTooltip title="Stockout Cost" content="The financial penalty for missing a sale (e.g. lost profit margin). High stockout costs penalize models that under-forecast.">
                      <div className="flex justify-between items-center mb-1">
                        <span className="text-xs font-bold text-slate-500 uppercase">Stockout Cost</span>
                        <span className="text-sm font-bold text-slate-900">${stockoutCost.toFixed(2)}</span>
                      </div>
                    </InfoTooltip>
                    <input
                      type="range"
                      min="1.0"
                      max="25.0"
                      step="1.0"
                      value={stockoutCost}
                      onChange={(e) => setStockoutCost(parseFloat(e.target.value))}
                      className="w-full h-1.5 bg-zinc-200 rounded-lg appearance-none cursor-pointer accent-emerald-600"
                    />
                  </div>
                </div>

                <div className="p-4 bg-zinc-50 rounded-xl space-y-3">
                  <SimMetric label="Safety Stock" value={simMetrics.safetyStock.toFixed(2)} unit="units" />
                  <div className="h-px bg-zinc-200 my-2" />
                  <SimMetric label="Total SKU Cost" value={`$${Math.round(simMetrics.totalCost)}`} bold />
                </div>
              </div>

              <button
                onClick={() => {
                  // Apply to all SKUs logic would go here
                }}
                className="w-full mt-6 py-3 bg-zinc-900 text-white rounded-xl font-bold text-sm flex items-center justify-center gap-2 hover:bg-zinc-800 transition-colors"
              >
                Apply to All SKUs <ArrowRight size={16} />
              </button>
            </div>
          </div>

          {/* Benchmark Results */}
          <div className="bg-white rounded-2xl border border-zinc-200 shadow-sm overflow-hidden mb-10">
            <div className="px-6 py-4 border-b border-zinc-200">
              <h3 className="text-lg font-semibold text-zinc-900">Model Performance Comparison</h3>
              <p className="text-sm text-zinc-600">28-day forecast accuracy and financial impact on CA_1 store</p>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full">
                <thead className="bg-zinc-50">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-semibold text-zinc-500 uppercase tracking-wider">Model</th>
                    <th className="px-6 py-3 text-left text-xs font-semibold text-zinc-500 uppercase tracking-wider">RMSE</th>
                    <th className="px-6 py-3 text-left text-xs font-semibold text-zinc-500 uppercase tracking-wider">Total Cost</th>
                    <th className="px-6 py-3 text-left text-xs font-semibold text-zinc-500 uppercase tracking-wider">Savings</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-zinc-200">
                  {sortedSummary.map((item, index) => (
                    <tr key={item.Model} className={index === 0 ? 'bg-green-50' : ''}>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="flex items-center">
                          {index === 0 && <span className="text-yellow-500 mr-2">🏆</span>}
                          <span className="text-sm font-medium text-zinc-900">{item.Model}</span>
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-zinc-900">{item.RMSE.toFixed(2)}</td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-zinc-900">${item.Total_Cost.toLocaleString()}</td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <span className={`text-sm font-semibold ${item.Total_Cost < (dynamicSummary.find(s => s.Model === 'Naive')?.Total_Cost || Infinity) ? 'text-green-600' : 'text-zinc-600'}`}>
                          {(() => {
                            if (item.Model === 'Naive') return 'Baseline';
                            const naiveCost = dynamicSummary.find(s => s.Model === 'Naive')?.Total_Cost || 1;
                            const savings = ((naiveCost - item.Total_Cost) / naiveCost * 100).toFixed(1);
                            return `+${savings}%`;
                          })()}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Feature Importance and Sensitivity */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-2 bg-amber-50 border border-amber-200 rounded-xl p-6">
              <h3 className="text-lg font-bold text-amber-800 mb-2">Top Predictive Features (SHAP)</h3>
              <div className="text-sm text-amber-700 space-y-3 mb-3 mt-3">
                {shapData.slice(0, 5).map((d, i) => {
                  const featureDescriptions: Record<string, string> = {
                    'rolling_mean_28': '4-week sales average (Captures monthly trends)',
                    'lag_7': 'Sales exactly 1 week ago (Captures weekly seasonality)',
                    'lag_14': 'Sales exactly 2 weeks ago (Validates weekly patterns)',
                    'item_id': 'Product identity (Model learns item-specific baselines)',
                    'rolling_mean_7': '1-week sales average (Captures immediate momentum)',
                    'sell_price': 'Current price of the item',
                    'event_name_1': 'Special events or holidays (e.g., SuperBowl)',
                  };
                  const desc = featureDescriptions[d.feature] || d.feature;
                  
                  return (
                    <div key={d.feature} className="flex flex-col gap-1">
                      <div className="flex items-center gap-2">
                        <span className="font-bold w-40 truncate">{i + 1}. {d.feature}</span>
                        <div className="flex-1 h-2 bg-amber-200 rounded-full overflow-hidden">
                          <div 
                            className="h-full bg-amber-500" 
                            style={{ width: `${(d.importance / shapData[0].importance) * 100}%` }}
                          />
                        </div>
                      </div>
                      <span className="text-xs text-amber-600/90 ml-30 pl-[170px] leading-tight">↳ {desc}</span>
                    </div>
                  );
                })}
                {shapData.length === 0 && <p className="text-amber-600/70 italic">Loading SHAP Explanations...</p>}
              </div>
              <p className="text-xs text-amber-600 mt-2 font-medium">SHAP values use Game Theory to prove exactly which variables drive LightGBM predictions.</p>
            </div>

            <div className="bg-purple-50 border border-purple-200 rounded-xl p-6 flex flex-col justify-center">
              <h3 className="text-lg font-bold text-purple-800 mb-3">Newsvendor Economics: Why did {bestModel.Model} win?</h3>
              {(() => {
                const cr = stockoutCost / (stockoutCost + holdingCost);
                if (cr < 0.5) {
                  return (
                    <p className="text-sm text-purple-700 leading-relaxed">
                      <strong>Low Critical Ratio ({(cr*100).toFixed(1)}%):</strong> Holding inventory is more expensive than missing sales. 
                      Models that systematically <strong>under-forecast</strong> (like LSTM) artificially win because they refuse to hold stock, entirely avoiding the massive holding penalty. 
                      This is a "Conservative" supply chain strategy.
                    </p>
                  );
                } else if (cr > 0.90) {
                  return (
                    <p className="text-sm text-purple-700 leading-relaxed">
                      <strong>High Critical Ratio ({(cr*100).toFixed(1)}%):</strong> Stockouts are incredibly expensive (e.g. high-margin electronics). 
                      Models that systematically <strong>over-forecast</strong> (like DeepAR) gain an advantage because missing a sale is heavily penalized. 
                      This is an "Aggressive" supply chain strategy.
                    </p>
                  );
                } else {
                  return (
                    <p className="text-sm text-purple-700 leading-relaxed">
                      <strong>Balanced Critical Ratio ({(cr*100).toFixed(1)}%):</strong> Costs are proportional. 
                      In this scenario, statistical accuracy matters most. The model with the lowest error (<strong>{bestModel.Model}</strong>) wins because it perfectly balances the risks of overstocking and understocking.
                    </p>
                  );
                }
              })()}
            </div>
          </div>
        </div>
      </div>
    </SidebarLayout>
  );
}

// Helper Components
function StatCard({ icon, label, value, sub }: { icon: React.ReactNode; label: string; value: string; sub: string }) {
  return (
    <div className="glass-panel rounded-2xl p-6 transition-all duration-300 hover:-translate-y-1 hover:shadow-lg">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm font-semibold text-slate-500 mb-1">{label}</p>
          <p className="text-3xl font-extrabold text-slate-900 tracking-tight">{value}</p>
          <p className="text-xs font-medium text-slate-400 mt-2">{sub}</p>
        </div>
        <div className="p-3 bg-white/80 rounded-xl shadow-sm border border-slate-100">
          {icon}
        </div>
      </div>
    </div>
  );
}

function SimMetric({ label, value, color = "text-slate-900", unit = "", bold = false }: {
  label: string;
  value: string;
  color?: string;
  unit?: string;
  bold?: boolean;
}) {
  return (
    <div className="flex justify-between items-center py-1">
      <span className="text-sm font-medium text-slate-600">{label}</span>
      <span className={cn("text-sm tracking-tight", color, bold ? "font-bold text-lg" : "font-semibold")}>
        {value}{unit && <span className="text-xs text-slate-400 font-normal ml-1">{unit}</span>}
      </span>
    </div>
  );
}