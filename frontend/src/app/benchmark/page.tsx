import SidebarLayout from '@/components/SidebarLayout';
import { BarChart3, TrendingUp, Trophy, AlertCircle } from 'lucide-react';

export default function BenchmarkResultsPage() {
  const benchmarkData = [
    { model: 'LightGBM', rmse: 2.10, cost: 515513, savings: 19.5, category: 'ML' },
    { model: 'Holt-Winters', rmse: 2.31, cost: 529096, savings: 17.4, category: 'Classical' },
    { model: 'LSTM (Log Transform)', rmse: 3.62, cost: 550000, savings: 14.0, category: 'DL' },
    { model: 'Naive', rmse: 2.86, cost: 640703, savings: 0, category: 'Baseline' },
    { model: 'LSTM (Original)', rmse: 3.58, cost: 1229646, savings: -91.0, category: 'DL' },
  ];

  return (
    <SidebarLayout>
      <div className="p-8">
        <div className="max-w-7xl mx-auto">
          <div className="mb-8">
            <h1 className="text-2xl font-bold text-zinc-900 mb-2">Benchmark Results</h1>
            <p className="text-zinc-600">Comprehensive evaluation of 10+ forecasting models on the M5 retail dataset</p>
          </div>

          {/* Summary Cards */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
            <div className="bg-white rounded-xl border border-zinc-200 p-6">
              <div className="flex items-center gap-3 mb-4">
                <Trophy className="w-6 h-6 text-yellow-500" />
                <div>
                  <p className="text-sm text-zinc-500">Champion Model</p>
                  <p className="text-lg font-bold text-zinc-900">LightGBM</p>
                </div>
              </div>
              <p className="text-2xl font-bold text-green-600">19.5%</p>
              <p className="text-sm text-zinc-500">Cost Savings</p>
            </div>

            <div className="bg-white rounded-xl border border-zinc-200 p-6">
              <div className="flex items-center gap-3 mb-4">
                <BarChart3 className="w-6 h-6 text-blue-500" />
                <div>
                  <p className="text-sm text-zinc-500">Best RMSE</p>
                  <p className="text-lg font-bold text-zinc-900">2.10</p>
                </div>
              </div>
              <p className="text-2xl font-bold text-blue-600">LightGBM</p>
              <p className="text-sm text-zinc-500">Units</p>
            </div>

            <div className="bg-white rounded-xl border border-zinc-200 p-6">
              <div className="flex items-center gap-3 mb-4">
                <TrendingUp className="w-6 h-6 text-purple-500" />
                <div>
                  <p className="text-sm text-zinc-500">Deep Learning</p>
                  <p className="text-lg font-bold text-zinc-900">LSTM</p>
                </div>
              </div>
              <p className="text-2xl font-bold text-purple-600">14.0%</p>
              <p className="text-sm text-zinc-500">Best DL Savings</p>
            </div>

            <div className="bg-white rounded-xl border border-zinc-200 p-6">
              <div className="flex items-center gap-3 mb-4">
                <AlertCircle className="w-6 h-6 text-red-500" />
                <div>
                  <p className="text-sm text-zinc-500">Worst Performer</p>
                  <p className="text-lg font-bold text-zinc-900">LSTM</p>
                </div>
              </div>
              <p className="text-2xl font-bold text-red-600">-91%</p>
              <p className="text-sm text-zinc-500">Original Model</p>
            </div>
          </div>

          {/* Detailed Results Table */}
          <div className="bg-white rounded-xl border border-zinc-200 overflow-hidden mb-8">
            <div className="px-6 py-4 border-b border-zinc-200">
              <h3 className="text-lg font-semibold text-zinc-900">Model Performance Comparison</h3>
              <p className="text-sm text-zinc-600">28-day forecast accuracy and financial impact on CA_1 store (3,049 SKUs)</p>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full">
                <thead className="bg-zinc-50">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-semibold text-zinc-500 uppercase tracking-wider">Model</th>
                    <th className="px-6 py-3 text-left text-xs font-semibold text-zinc-500 uppercase tracking-wider">Category</th>
                    <th className="px-6 py-3 text-left text-xs font-semibold text-zinc-500 uppercase tracking-wider">RMSE</th>
                    <th className="px-6 py-3 text-left text-xs font-semibold text-zinc-500 uppercase tracking-wider">Total Cost</th>
                    <th className="px-6 py-3 text-left text-xs font-semibold text-zinc-500 uppercase tracking-wider">Savings</th>
                    <th className="px-6 py-3 text-left text-xs font-semibold text-zinc-500 uppercase tracking-wider">Status</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-zinc-200">
                  {benchmarkData.map((item, index) => (
                    <tr key={item.model} className={index === 0 ? 'bg-green-50' : ''}>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="flex items-center">
                          {index === 0 && <Trophy className="w-4 h-4 text-yellow-500 mr-2" />}
                          <span className="text-sm font-medium text-zinc-900">{item.model}</span>
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${
                          item.category === 'ML' ? 'bg-blue-100 text-blue-800' :
                          item.category === 'DL' ? 'bg-purple-100 text-purple-800' :
                          item.category === 'Classical' ? 'bg-green-100 text-green-800' :
                          'bg-gray-100 text-gray-800'
                        }`}>
                          {item.category}
                        </span>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-zinc-900">{item.rmse.toFixed(2)}</td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-zinc-900">${item.cost.toLocaleString()}</td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <span className={`text-sm font-semibold ${
                          item.savings > 0 ? 'text-green-600' : item.savings === 0 ? 'text-zinc-600' : 'text-red-600'
                        }`}>
                          {item.savings > 0 ? '+' : ''}{item.savings.toFixed(1)}%
                        </span>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${
                          index === 0 ? 'bg-green-100 text-green-800' :
                          item.savings > 10 ? 'bg-blue-100 text-blue-800' :
                          item.savings > 0 ? 'bg-yellow-100 text-yellow-800' :
                          'bg-red-100 text-red-800'
                        }`}>
                          {index === 0 ? 'Champion' :
                           item.savings > 10 ? 'Strong' :
                           item.savings > 0 ? 'Moderate' : 'Poor'}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Category Analysis */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
            <div className="bg-white rounded-xl border border-zinc-200 p-6">
              <h3 className="text-lg font-semibold text-zinc-900 mb-4">Classical Methods</h3>
              <div className="space-y-3">
                <div className="flex justify-between">
                  <span className="text-sm text-zinc-600">Best Performance</span>
                  <span className="text-sm font-semibold text-zinc-900">Holt-Winters (17.4%)</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-zinc-600">Average RMSE</span>
                  <span className="text-sm font-semibold text-zinc-900">2.59</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-zinc-600">Strengths</span>
                  <span className="text-sm font-semibold text-green-600">Stable, Interpretable</span>
                </div>
              </div>
            </div>

            <div className="bg-white rounded-xl border border-zinc-200 p-6">
              <h3 className="text-lg font-semibold text-zinc-900 mb-4">Machine Learning</h3>
              <div className="space-y-3">
                <div className="flex justify-between">
                  <span className="text-sm text-zinc-600">Best Performance</span>
                  <span className="text-sm font-semibold text-zinc-900">LightGBM (19.5%)</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-zinc-600">Average RMSE</span>
                  <span className="text-sm font-semibold text-zinc-900">2.10</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-zinc-600">Strengths</span>
                  <span className="text-sm font-semibold text-blue-600">Accuracy, Speed</span>
                </div>
              </div>
            </div>

            <div className="bg-white rounded-xl border border-zinc-200 p-6">
              <h3 className="text-lg font-semibold text-zinc-900 mb-4">Deep Learning</h3>
              <div className="space-y-3">
                <div className="flex justify-between">
                  <span className="text-sm text-zinc-600">Best Performance</span>
                  <span className="text-sm font-semibold text-zinc-900">LSTM Log (14.0%)</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-zinc-600">Average RMSE</span>
                  <span className="text-sm font-semibold text-zinc-900">3.60</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-zinc-600">Challenge</span>
                  <span className="text-sm font-semibold text-red-600">Sparsity Penalty</span>
                </div>
              </div>
            </div>
          </div>

          {/* Key Insights */}
          <div className="bg-gradient-to-r from-blue-50 to-green-50 rounded-xl border border-blue-200 p-6">
            <h3 className="text-lg font-semibold text-blue-900 mb-4">Key Insights & Recommendations</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <h4 className="font-semibold text-zinc-900 mb-2">What Works</h4>
                <ul className="text-sm text-zinc-600 space-y-1">
                  <li>• LightGBM provides best balance of accuracy and efficiency</li>
                  <li>• Classical methods remain competitive for stability</li>
                  <li>• Log transformation helps deep learning handle zeros</li>
                  <li>• Feature engineering is crucial for all approaches</li>
                </ul>
              </div>
              <div>
                <h4 className="font-semibold text-zinc-900 mb-2">Future Research</h4>
                <ul className="text-sm text-zinc-600 space-y-1">
                  <li>• Further optimization of deep learning architectures</li>
                  <li>• Hybrid approaches combining ML and classical methods</li>
                  <li>• Advanced feature engineering and external data integration</li>
                  <li>• Real-time adaptation and online learning</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </div>
    </SidebarLayout>
  );
}