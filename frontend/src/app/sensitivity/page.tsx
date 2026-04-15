import SidebarLayout from '@/components/SidebarLayout';
import { Settings, TrendingUp, DollarSign, AlertTriangle } from 'lucide-react';

export default function SensitivityAnalysisPage() {
  return (
    <SidebarLayout>
      <div className="p-8">
        <div className="max-w-7xl mx-auto">
          <div className="mb-8">
            <h1 className="text-2xl font-bold text-zinc-900 mb-2">Sensitivity Analysis</h1>
            <p className="text-zinc-600">Explore how forecast accuracy affects inventory costs across different parameters</p>
          </div>

          {/* Parameter Controls */}
          <div className="grid grid-cols-1 lg:grid-cols-4 gap-6 mb-8">
            <div className="bg-white rounded-xl border border-zinc-200 p-6">
              <div className="flex items-center gap-3 mb-4">
                <Settings className="w-5 h-5 text-blue-600" />
                <h3 className="font-semibold text-zinc-900">Service Level</h3>
              </div>
              <div className="space-y-3">
                <input
                  type="range"
                  min="0.80"
                  max="0.99"
                  step="0.01"
                  defaultValue="0.95"
                  className="w-full h-2 bg-zinc-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
                />
                <div className="text-center">
                  <span className="text-lg font-bold text-blue-600">95%</span>
                </div>
              </div>
            </div>

            <div className="bg-white rounded-xl border border-zinc-200 p-6">
              <div className="flex items-center gap-3 mb-4">
                <DollarSign className="w-5 h-5 text-amber-600" />
                <h3 className="font-semibold text-zinc-900">Holding Cost</h3>
              </div>
              <div className="space-y-3">
                <input
                  type="range"
                  min="0.5"
                  max="2.0"
                  step="0.1"
                  defaultValue="1.0"
                  className="w-full h-2 bg-zinc-200 rounded-lg appearance-none cursor-pointer accent-amber-600"
                />
                <div className="text-center">
                  <span className="text-lg font-bold text-amber-600">$1.00</span>
                </div>
              </div>
            </div>

            <div className="bg-white rounded-xl border border-zinc-200 p-6">
              <div className="flex items-center gap-3 mb-4">
                <AlertTriangle className="w-5 h-5 text-red-600" />
                <h3 className="font-semibold text-zinc-900">Stockout Cost</h3>
              </div>
              <div className="space-y-3">
                <input
                  type="range"
                  min="5"
                  max="20"
                  step="1"
                  defaultValue="10"
                  className="w-full h-2 bg-zinc-200 rounded-lg appearance-none cursor-pointer accent-red-600"
                />
                <div className="text-center">
                  <span className="text-lg font-bold text-red-600">$10.00</span>
                </div>
              </div>
            </div>

            <div className="bg-white rounded-xl border border-zinc-200 p-6">
              <div className="flex items-center gap-3 mb-4">
                <TrendingUp className="w-5 h-5 text-purple-600" />
                <h3 className="font-semibold text-zinc-900">Lead Time</h3>
              </div>
              <div className="space-y-3">
                <input
                  type="range"
                  min="1"
                  max="7"
                  step="1"
                  defaultValue="1"
                  className="w-full h-2 bg-zinc-200 rounded-lg appearance-none cursor-pointer accent-purple-600"
                />
                <div className="text-center">
                  <span className="text-lg font-bold text-purple-600">1 day</span>
                </div>
              </div>
            </div>
          </div>

          {/* Impact Analysis */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
            <div className="bg-white rounded-xl border border-zinc-200 p-6">
              <h3 className="text-lg font-semibold text-zinc-900 mb-4">Cost Impact Analysis</h3>
              <div className="space-y-4">
                <div className="flex justify-between items-center p-3 bg-zinc-50 rounded-lg">
                  <span className="text-sm text-zinc-600">Service Level +10%</span>
                  <span className="text-sm font-bold text-blue-600">+$15,000 cost impact</span>
                </div>
                <div className="flex justify-between items-center p-3 bg-zinc-50 rounded-lg">
                  <span className="text-sm text-zinc-600">Holding Cost +$0.50</span>
                  <span className="text-sm font-bold text-amber-600">+$2,300 cost impact</span>
                </div>
                <div className="flex justify-between items-center p-3 bg-zinc-50 rounded-lg">
                  <span className="text-sm text-zinc-600">Stockout Cost +$5</span>
                  <span className="text-sm font-bold text-red-600">+$1,800 cost impact</span>
                </div>
                <div className="flex justify-between items-center p-3 bg-zinc-50 rounded-lg">
                  <span className="text-sm text-zinc-600">Lead Time +1 day</span>
                  <span className="text-sm font-bold text-purple-600">+15% safety stock</span>
                </div>
              </div>
            </div>

            <div className="bg-white rounded-xl border border-zinc-200 p-6">
              <h3 className="text-lg font-semibold text-zinc-900 mb-4">Parameter Sensitivity</h3>
              <div className="space-y-4">
                <div>
                  <div className="flex justify-between text-sm mb-1">
                    <span className="text-zinc-600">Service Level</span>
                    <span className="font-semibold text-zinc-900">High Impact</span>
                  </div>
                  <div className="w-full bg-zinc-200 rounded-full h-2">
                    <div className="bg-red-500 h-2 rounded-full w-4/5"></div>
                  </div>
                </div>
                <div>
                  <div className="flex justify-between text-sm mb-1">
                    <span className="text-zinc-600">Holding Cost</span>
                    <span className="font-semibold text-zinc-900">Medium Impact</span>
                  </div>
                  <div className="w-full bg-zinc-200 rounded-full h-2">
                    <div className="bg-yellow-500 h-2 rounded-full w-3/5"></div>
                  </div>
                </div>
                <div>
                  <div className="flex justify-between text-sm mb-1">
                    <span className="text-zinc-600">Stockout Cost</span>
                    <span className="font-semibold text-zinc-900">High Impact</span>
                  </div>
                  <div className="w-full bg-zinc-200 rounded-full h-2">
                    <div className="bg-red-500 h-2 rounded-full w-4/5"></div>
                  </div>
                </div>
                <div>
                  <div className="flex justify-between text-sm mb-1">
                    <span className="text-zinc-600">Lead Time</span>
                    <span className="font-semibold text-zinc-900">Medium Impact</span>
                  </div>
                  <div className="w-full bg-zinc-200 rounded-full h-2">
                    <div className="bg-yellow-500 h-2 rounded-full w-3/5"></div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Recommendations */}
          <div className="bg-gradient-to-r from-blue-50 to-purple-50 rounded-xl border border-blue-200 p-6">
            <h3 className="text-lg font-semibold text-blue-900 mb-4">Optimization Recommendations</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="bg-white p-4 rounded-lg">
                <h4 className="font-semibold text-zinc-900 mb-2">High Priority</h4>
                <ul className="text-sm text-zinc-600 space-y-1">
                  <li>• Focus on service level optimization (80-95% range)</li>
                  <li>• Stockout cost has highest impact - negotiate better terms</li>
                  <li>• Consider demand forecasting improvements over cost tuning</li>
                </ul>
              </div>
              <div className="bg-white p-4 rounded-lg">
                <h4 className="font-semibold text-zinc-900 mb-2">Implementation Strategy</h4>
                <ul className="text-sm text-zinc-600 space-y-1">
                  <li>• Start with 90% service level as baseline</li>
                  <li>• Monitor actual vs. predicted stockouts quarterly</li>
                  <li>• Use sensitivity analysis for scenario planning</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </div>
    </SidebarLayout>
  );
}