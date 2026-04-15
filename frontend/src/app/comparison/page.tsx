import SidebarLayout from '@/components/SidebarLayout';
import { Target, TrendingUp, BarChart3, Zap } from 'lucide-react';

export default function ModelComparisonPage() {
  const modelDetails = [
    {
      name: 'LightGBM',
      category: 'Machine Learning',
      rmse: 2.10,
      cost: 515513,
      savings: 19.5,
      strengths: ['High accuracy', 'Fast training', 'Handles missing data', 'Feature importance'],
      weaknesses: ['Less interpretable than linear models', 'Requires tuning'],
      useCase: 'Production deployment, high accuracy needed',
      icon: BarChart3,
      color: 'blue'
    },
    {
      name: 'Holt-Winters',
      category: 'Classical',
      rmse: 2.31,
      cost: 529096,
      savings: 17.4,
      strengths: ['Interpretable', 'Stable predictions', 'Handles seasonality', 'Low computational cost'],
      weaknesses: ['Limited to time series patterns', 'No exogenous variables'],
      useCase: 'Stable environments, interpretability required',
      icon: TrendingUp,
      color: 'green'
    },
    {
      name: 'LSTM (Log Transform)',
      category: 'Deep Learning',
      rmse: 3.62,
      cost: 550000,
      savings: 14.0,
      strengths: ['Handles complex patterns', 'Learns from sequences', 'Addresses sparsity', 'Scalable'],
      weaknesses: ['High computational cost', 'Requires large datasets', 'Black box nature'],
      useCase: 'Complex patterns, sufficient data available',
      icon: Zap,
      color: 'purple'
    },
    {
      name: 'Naive',
      category: 'Baseline',
      rmse: 2.86,
      cost: 640703,
      savings: 0,
      strengths: ['Simple to implement', 'No training required', 'Always available'],
      weaknesses: ['Poor accuracy', 'No intelligence', 'High costs'],
      useCase: 'Minimum viable forecasting, comparison baseline',
      icon: Target,
      color: 'gray'
    }
  ];

  return (
    <SidebarLayout>
      <div className="p-8">
        <div className="max-w-7xl mx-auto">
          <div className="mb-8">
            <h1 className="text-2xl font-bold text-zinc-900 mb-2">Model Comparison</h1>
            <p className="text-zinc-600">Detailed analysis of top-performing forecasting models</p>
          </div>

          {/* Model Cards */}
          <div className="space-y-6 mb-8">
            {modelDetails.map((model, index) => {
              const Icon = model.icon;
              const isChampion = index === 0;

              return (
                <div key={model.name} className={`bg-white rounded-xl border ${isChampion ? 'border-yellow-300 bg-gradient-to-r from-yellow-50 to-white' : 'border-zinc-200'} p-6`}>
                  <div className="flex items-start justify-between mb-6">
                    <div className="flex items-center gap-4">
                      <div className={`p-3 rounded-xl ${model.color === 'blue' ? 'bg-blue-100' : model.color === 'green' ? 'bg-green-100' : model.color === 'purple' ? 'bg-purple-100' : 'bg-gray-100'}`}>
                        <Icon className={`w-6 h-6 ${model.color === 'blue' ? 'text-blue-600' : model.color === 'green' ? 'text-green-600' : model.color === 'purple' ? 'text-purple-600' : 'text-gray-600'}`} />
                      </div>
                      <div>
                        <div className="flex items-center gap-2">
                          <h3 className="text-xl font-bold text-zinc-900">{model.name}</h3>
                          {isChampion && (
                            <span className="px-2 py-1 bg-yellow-100 text-yellow-800 text-xs font-semibold rounded-full">
                              🏆 Champion
                            </span>
                          )}
                        </div>
                        <p className="text-sm text-zinc-500">{model.category}</p>
                      </div>
                    </div>

                    <div className="text-right">
                      <div className="text-2xl font-bold text-zinc-900">{model.rmse.toFixed(2)}</div>
                      <div className="text-sm text-zinc-500">RMSE</div>
                    </div>
                  </div>

                  {/* Performance Metrics */}
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                    <div className="bg-zinc-50 rounded-lg p-4">
                      <div className="text-sm text-zinc-500 mb-1">Total Cost</div>
                      <div className="text-lg font-semibold text-zinc-900">${model.cost.toLocaleString()}</div>
                    </div>
                    <div className="bg-zinc-50 rounded-lg p-4">
                      <div className="text-sm text-zinc-500 mb-1">Cost Savings</div>
                      <div className={`text-lg font-semibold ${model.savings > 0 ? 'text-green-600' : 'text-zinc-600'}`}>
                        {model.savings > 0 ? '+' : ''}{model.savings.toFixed(1)}%
                      </div>
                    </div>
                    <div className="bg-zinc-50 rounded-lg p-4">
                      <div className="text-sm text-zinc-500 mb-1">Use Case</div>
                      <div className="text-sm font-semibold text-zinc-900">{model.useCase}</div>
                    </div>
                  </div>

                  {/* Strengths and Weaknesses */}
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div>
                      <h4 className="text-sm font-semibold text-green-700 mb-3 flex items-center gap-2">
                        <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                        Strengths
                      </h4>
                      <ul className="space-y-1">
                        {model.strengths.map((strength, i) => (
                          <li key={i} className="text-sm text-zinc-600 flex items-start gap-2">
                            <span className="text-green-500 mt-1.5">•</span>
                            {strength}
                          </li>
                        ))}
                      </ul>
                    </div>

                    <div>
                      <h4 className="text-sm font-semibold text-red-700 mb-3 flex items-center gap-2">
                        <div className="w-2 h-2 bg-red-500 rounded-full"></div>
                        Weaknesses
                      </h4>
                      <ul className="space-y-1">
                        {model.weaknesses.map((weakness, i) => (
                          <li key={i} className="text-sm text-zinc-600 flex items-start gap-2">
                            <span className="text-red-500 mt-1.5">•</span>
                            {weakness}
                          </li>
                        ))}
                      </ul>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>

          {/* Decision Framework */}
          <div className="bg-gradient-to-r from-blue-50 to-indigo-50 rounded-xl border border-blue-200 p-6 mb-8">
            <h3 className="text-lg font-semibold text-blue-900 mb-4">Model Selection Framework</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <h4 className="font-semibold text-zinc-900 mb-3">When to Choose LightGBM</h4>
                <ul className="text-sm text-zinc-600 space-y-1">
                  <li>• High accuracy requirements</li>
                  <li>• Production deployment with speed constraints</li>
                  <li>• Feature engineering and interpretability are priorities</li>
                  <li>• Limited computational resources</li>
                </ul>
              </div>
              <div>
                <h4 className="font-semibold text-zinc-900 mb-3">When to Choose Deep Learning</h4>
                <ul className="text-sm text-zinc-600 space-y-1">
                  <li>• Complex, non-linear patterns in data</li>
                  <li>• Sufficient training data available</li>
                  <li>• Computational resources are not limited</li>
                  <li>• Research and experimentation focus</li>
                </ul>
              </div>
            </div>
          </div>

          {/* Implementation Recommendations */}
          <div className="bg-white rounded-xl border border-zinc-200 p-6">
            <h3 className="text-lg font-semibold text-zinc-900 mb-4">Implementation Recommendations</h3>
            <div className="space-y-4">
              <div className="border-l-4 border-green-500 pl-4">
                <h4 className="font-semibold text-zinc-900">Immediate Deployment</h4>
                <p className="text-sm text-zinc-600 mt-1">
                  Deploy LightGBM as the primary forecasting model for production use. It provides the best balance of accuracy, speed, and reliability.
                </p>
              </div>

              <div className="border-l-4 border-blue-500 pl-4">
                <h4 className="font-semibold text-zinc-900">Research Continuation</h4>
                <p className="text-sm text-zinc-600 mt-1">
                  Continue development of deep learning approaches, particularly the log-transform LSTM, as they show promise for handling complex retail patterns.
                </p>
              </div>

              <div className="border-l-4 border-purple-500 pl-4">
                <h4 className="font-semibold text-zinc-900">Hybrid Approach</h4>
                <p className="text-sm text-zinc-600 mt-1">
                  Consider ensemble methods combining LightGBM predictions with classical approaches for improved stability and accuracy.
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </SidebarLayout>
  );
}