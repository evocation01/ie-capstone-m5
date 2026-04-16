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
  BarChart3,
  Settings,
  ArrowRight
} from 'lucide-react';
import SidebarLayout from '@/components/SidebarLayout';

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
  Note?: string;
}

export default function DashboardPage() {
  const [data, setData] = useState<Record<string, SKUData>>({});
  const [summary, setSummary] = useState<SummaryItem[]>([]);
  const [selectedSku, setSelectedSku] = useState<string>('');
  const [serviceLevel, setServiceLevel] = useState<number>(0.95);
  const [holdingCost, setHoldingCost] = useState<number>(1.00);
  const [stockoutCost, setStockoutCost] = useState<number>(10.00);
  const [leadTime, setLeadTime] = useState<number>(1);
  const [loading, setLoading] = useState(true);
  const [chartType, setChartType] = useState<'sales' | 'inventory'>('sales');

  // Load data on mount
  useEffect(() => {
    const loadData = async () => {
      try {
        const [skuRes, summaryRes] = await Promise.all([
          fetch('/data/sku_data.json'),
          fetch('/data/summary.json')
        ]);

        const skuData = await skuRes.json();
        const summaryData = await summaryRes.json();

        setData(skuData);
        setSummary(summaryData);

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

    const lgbmFc = sku.forecasts['LightGBM']?.[0] || 0;
    const naiveFc = sku.forecasts['Naive']?.[0] || 0;
    const ss = getLocalZScore(serviceLevel) * 2.1; // Use LightGBM average RMSE for safety stock

    combined.push({
      name: `D-1`,
      actual: sku.actual?.[0] || 0,
      lightgbm: lgbmFc,
      lstm: sku.forecasts['LSTM']?.[0],
      naive: naiveFc,
      isHistory: false,
      safetyStock: ss
    });

    return combined;
  }, [selectedSku, data, serviceLevel]);

  const bestModel = summary.reduce((prev, curr) => prev.Total_Cost < curr.Total_Cost ? prev : curr, summary[0]);
  const naiveModel = summary.find(s => s.Model === 'Naive');
  const savings = naiveModel ? ((naiveModel.Total_Cost - bestModel.Total_Cost) / naiveModel.Total_Cost * 100) : 0;

  if (loading) {
    return (
      <SidebarLayout>
        <div className="p-8">
          <div className="max-w-7xl mx-auto">
            <div className="flex items-center justify-center h-64">
              <div className="text-zinc-500">Loading dashboard...</div>
            </div>
          </div>
        </div>
      </SidebarLayout>
    );
  }

  return (
    <SidebarLayout>
      <div className="p-8">
        <div className="max-w-7xl mx-auto">
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
            <StatCard
              icon={<DollarSign className="text-amber-600" />}
              label="Theoretical Impact"
              value={`$${(bestModel?.Total_Cost / 1000).toFixed(0) || '0'}K`}
              sub="28-day cost savings"
            />
          </div>

          {/* SKU Selector and Chart */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-10">
            {/* Chart */}
            <div className="lg:col-span-2 bg-white rounded-2xl border border-zinc-200 shadow-sm p-6">
              <div className="flex justify-between items-center mb-6">
                <div className="flex items-center gap-3">
                  <h2 className="text-lg font-bold">Forecasting &ldquo;Drag Race&rdquo;</h2>
                  <div className="flex bg-zinc-100 rounded-lg p-1">
                    <button
                      onClick={() => setChartType('sales')}
                      className={`px-3 py-1 text-xs font-semibold rounded-md ${chartType === 'sales' ? 'bg-white shadow-sm' : 'text-zinc-500'}`}
                    >
                      Sales
                    </button>
                    <button
                      onClick={() => setChartType('inventory')}
                      className={`px-3 py-1 text-xs font-semibold rounded-md ${chartType === 'inventory' ? 'bg-white shadow-sm' : 'text-zinc-500'}`}
                    >
                      Inventory
                    </button>
                  </div>
                </div>

                <select
                  value={selectedSku}
                  onChange={(e) => setSelectedSku(e.target.value)}
                  className="px-3 py-2 border border-zinc-200 rounded-lg text-sm"
                >
                  {Object.keys(data).map(sku => (
                    <option key={sku} value={sku}>{sku}</option>
                  ))}
                </select>
              </div>

              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Line type="monotone" dataKey="actual" name="Actual Sales" stroke="#000000" strokeWidth={2} />
                  <Line type="monotone" dataKey="lightgbm" name="LightGBM Forecast" stroke="#2563eb" strokeWidth={2} strokeDasharray="5 5" dot={false} />
                  <Line type="monotone" dataKey="lstm" name="LSTM Forecast" stroke="#7c3aed" strokeWidth={2} strokeDasharray="8 8" dot={false} />
                  <Line type="monotone" dataKey="naive" name="Naive Baseline" stroke="#94a3b8" strokeWidth={1} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>

            {/* Inventory Simulator */}
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
                  <SimMetric label="Stockout Risk" value={`$${simulationResults?.totalStockout.toFixed(0) || '0'}`} color="text-red-600" />
                  <div className="h-px bg-zinc-200 my-2" />
                  <div className="flex justify-between text-xs text-zinc-500">
                    <span>Lower service ↓ holding cost but ↑ stockout risk</span>
                  </div>
                  <div className="h-px bg-zinc-200 my-2" />
                  <SimMetric label="Total SKU Cost" value={`$${simulationResults?.totalCost.toFixed(0) || '0'}`} bold />
                </div>

                <div className="text-[11px] text-zinc-500 leading-relaxed italic">
                  * Safety Stock is calculated as z * RMSE * sqrt(L). Higher service levels increase holding costs but minimize expensive stockouts.
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
                  {summary.map((item, index) => (
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
                        <span className={`text-sm font-semibold ${item.Total_Cost < (summary.find(s => s.Model === 'Naive')?.Total_Cost || Infinity) ? 'text-green-600' : 'text-zinc-600'}`}>
                          {(() => {
                            if (item.Model === 'Naive') return 'Baseline';
                            const naiveCost = summary.find(s => s.Model === 'Naive')?.Total_Cost || 1;
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
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="bg-amber-50 border border-amber-200 rounded-xl p-6">
              <h3 className="text-sm font-bold text-amber-800 mb-1">Top Predictive Features</h3>
              <div className="text-xs text-amber-700 space-y-1">
                <p><span className="font-semibold">1. lag_7</span> - Last week's sales</p>
                <p><span className="font-semibold">2. rolling_mean_28</span> - 4-week average</p>
                <p><span className="font-semibold">3. item_id</span> - Product identity</p>
                <p><span className="font-semibold">4. lag_14</span> - 2-week lag</p>
                <p className="text-[10px] text-amber-600 mt-2">Recent history dominates → model learns real patterns</p>
              </div>
            </div>

            <div className="bg-purple-50 border border-purple-200 rounded-xl p-6">
              <h3 className="text-sm font-bold text-purple-800 mb-1">Sensitivity: Service Level Trade-off</h3>
              <p className="text-xs text-purple-700 mb-2">Cost impact by service level:</p>
              <table className="w-full text-xs text-purple-700">
                <thead>
                  <tr className="border-b border-purple-200">
                    <th className="text-left py-1">Service</th>
                    <th className="text-right py-1">Total</th>
                  </tr>
                </thead>
                <tbody>
                  <tr><td className="py-1">80%</td><td className="text-right font-bold">$132k</td></tr>
                  <tr><td className="py-1">95%</td><td className="text-right font-bold">$72k</td></tr>
                  <tr><td className="py-1">99%</td><td className="text-right font-bold">$60k</td></tr>
                </tbody>
              </table>
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
    <div className="bg-white rounded-xl border border-zinc-200 p-6">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm font-medium text-zinc-500">{label}</p>
          <p className="text-2xl font-bold text-zinc-900 mt-1">{value}</p>
          <p className="text-xs text-zinc-400 mt-1">{sub}</p>
        </div>
        <div className="text-zinc-400">
          {icon}
        </div>
      </div>
    </div>
  );
}

function SimMetric({ label, value, color = "text-zinc-900", unit = "", bold = false }: {
  label: string;
  value: string;
  color?: string;
  unit?: string;
  bold?: boolean;
}) {
  return (
    <div className="flex justify-between items-center">
      <span className="text-sm text-zinc-600">{label}</span>
      <span className={cn("text-sm", color, bold ? "font-bold" : "")}>
        {value}{unit && ` ${unit}`}
      </span>
    </div>
  );
}