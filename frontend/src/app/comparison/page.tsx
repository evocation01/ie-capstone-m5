"use client";

import React from 'react';
import SidebarLayout from '@/components/SidebarLayout';
import { Layers, BrainCircuit, Activity, CheckCircle2, XCircle } from 'lucide-react';

export default function ComparisonPage() {
  return (
    <SidebarLayout>
      <div className="p-8">
        <div className="max-w-7xl mx-auto space-y-10">
          
          <header className="animate-fade-in">
            <h1 className="text-4xl font-extrabold tracking-tight text-slate-900">Model Methodology Comparison</h1>
            <p className="text-slate-500 mt-2 font-medium max-w-3xl">
              A deep dive into the algorithmic approaches tested in Phase I & Phase II, 
              exploring why certain architectures succeeded while others struggled with the M5 dataset&apos;s sparsity.
            </p>
          </header>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            
            {/* Classical Tier */}
            <div className="glass-panel p-8 rounded-3xl relative overflow-hidden group animate-fade-in" style={{ animationDelay: '100ms' }}>
              <div className="absolute -top-10 -right-10 w-40 h-40 bg-slate-200/50 rounded-full blur-3xl group-hover:bg-slate-300/50 transition-colors" />
              <Activity className="w-10 h-10 text-slate-600 mb-6 relative z-10" />
              <h2 className="text-2xl font-bold text-slate-900 mb-2 relative z-10">Tier 1: Classical</h2>
              <p className="text-sm font-semibold text-slate-500 mb-6 uppercase tracking-wider relative z-10">Statistical Methods</p>
              
              <div className="space-y-6 relative z-10">
                <p className="text-sm text-slate-600 leading-relaxed">
                  Traditional time-series models (ARIMA, Holt-Winters, ETS) rely on explicit mathematical formulas to capture trend, seasonality, and level.
                </p>
                
                <div className="bg-white/60 rounded-xl p-4 border border-slate-100">
                  <h4 className="text-xs font-bold text-slate-900 mb-3 uppercase tracking-wide">Key Findings</h4>
                  <ul className="space-y-2 text-sm text-slate-700">
                    <li className="flex items-start gap-2">
                      <CheckCircle2 className="w-4 h-4 text-emerald-500 mt-0.5 shrink-0" />
                      <span>Extremely fast to compute and explain.</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <XCircle className="w-4 h-4 text-red-500 mt-0.5 shrink-0" />
                      <span>Cannot learn cross-series relationships (trains 1 model per item).</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <XCircle className="w-4 h-4 text-red-500 mt-0.5 shrink-0" />
                      <span>Failed entirely on intermittent/sparse data.</span>
                    </li>
                  </ul>
                </div>
              </div>
            </div>

            {/* ML Tier */}
            <div className="glass-panel p-8 rounded-3xl relative overflow-hidden group animate-fade-in" style={{ animationDelay: '200ms' }}>
              <div className="absolute -top-10 -right-10 w-40 h-40 bg-blue-200/50 rounded-full blur-3xl group-hover:bg-blue-300/50 transition-colors" />
              <Layers className="w-10 h-10 text-blue-600 mb-6 relative z-10" />
              <h2 className="text-2xl font-bold text-slate-900 mb-2 relative z-10">Tier 2: Machine Learning</h2>
              <p className="text-sm font-semibold text-blue-500 mb-6 uppercase tracking-wider relative z-10">Gradient Boosting</p>
              
              <div className="space-y-6 relative z-10">
                <p className="text-sm text-slate-600 leading-relaxed">
                  Tree-based ensemble models (LightGBM, XGBoost) treat forecasting as a supervised regression problem using sliding windows and lag features.
                </p>
                
                <div className="bg-white/60 rounded-xl p-4 border border-slate-100">
                  <h4 className="text-xs font-bold text-slate-900 mb-3 uppercase tracking-wide">Key Findings</h4>
                  <ul className="space-y-2 text-sm text-slate-700">
                    <li className="flex items-start gap-2">
                      <CheckCircle2 className="w-4 h-4 text-emerald-500 mt-0.5 shrink-0" />
                      <span className="font-semibold text-slate-900">Current Champion (19.5% savings).</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <CheckCircle2 className="w-4 h-4 text-emerald-500 mt-0.5 shrink-0" />
                      <span>Global model learns patterns across all items simultaneously.</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <CheckCircle2 className="w-4 h-4 text-emerald-500 mt-0.5 shrink-0" />
                      <span>Handles tabular categorical data (events, prices) natively.</span>
                    </li>
                  </ul>
                </div>
              </div>
            </div>

            {/* DL Tier */}
            <div className="glass-panel p-8 rounded-3xl relative overflow-hidden group animate-fade-in" style={{ animationDelay: '300ms' }}>
              <div className="absolute -top-10 -right-10 w-40 h-40 bg-purple-200/50 rounded-full blur-3xl group-hover:bg-purple-300/50 transition-colors" />
              <BrainCircuit className="w-10 h-10 text-purple-600 mb-6 relative z-10" />
              <h2 className="text-2xl font-bold text-slate-900 mb-2 relative z-10">Tier 3: Deep Learning</h2>
              <p className="text-sm font-semibold text-purple-500 mb-6 uppercase tracking-wider relative z-10">Sequence Models</p>
              
              <div className="space-y-6 relative z-10">
                <p className="text-sm text-slate-600 leading-relaxed">
                  Neural Networks (LSTM, DeepAR) process raw sequential data, learning complex temporal dynamics and non-linear relationships.
                </p>
                
                <div className="bg-white/60 rounded-xl p-4 border border-slate-100">
                  <h4 className="text-xs font-bold text-slate-900 mb-3 uppercase tracking-wide">Key Findings</h4>
                  <ul className="space-y-2 text-sm text-slate-700">
                    <li className="flex items-start gap-2">
                      <CheckCircle2 className="w-4 h-4 text-emerald-500 mt-0.5 shrink-0" />
                      <span>DeepAR effectively modeled zero-inflated (intermittent) demand.</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <XCircle className="w-4 h-4 text-red-500 mt-0.5 shrink-0" />
                      <span>Standard LSTMs suffered from the &quot;Sparsity Penalty&quot;.</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <XCircle className="w-4 h-4 text-red-500 mt-0.5 shrink-0" />
                      <span>Requires significantly higher computational resources.</span>
                    </li>
                  </ul>
                </div>
              </div>
            </div>

          </div>
          
          {/* Detailed Analysis Section */}
          <div className="glass-panel rounded-3xl p-8 mt-10 animate-fade-in" style={{ animationDelay: '400ms' }}>
            <h3 className="text-xl font-bold text-slate-900 mb-4">The &quot;Sparsity Penalty&quot; Explained</h3>
            <div className="prose prose-slate max-w-none text-sm text-slate-600">
              <p className="mb-4">
                During Phase I, we discovered that standard Deep Learning models (like a basic LSTM) severely underperformed in financial terms despite having reasonable statistical error metrics. We dubbed this the <strong>&quot;Sparsity Penalty&quot;</strong>.
              </p>
              <p className="mb-4">
                Retail data (especially at the SKU-Store level) is highly sparse. Many items sell 0 units on most days, and perhaps 1-3 units occasionally. A standard LSTM optimizing for Mean Squared Error (MSE) learns that the safest mathematical prediction is a constant fractional number (e.g., 0.4 units per day) to minimize average penalty.
              </p>
              <p>
                <strong>The Financial Consequence:</strong> In an inventory simulator, if you predict 0.4 units of demand, your system orders stock. Because the prediction is never 0, stock continuously accumulates over 28 days for an item that is rarely sold. This leads to massive Holding Costs, making the LSTM artificially expensive in real-world applications. Our solution in Phase II was the implementation of <strong>DeepAR</strong>, an autoregressive probabilistic model that uses a Negative Binomial likelihood specifically designed to output zero-heavy, intermittent demand distributions.
              </p>
            </div>
          </div>

        </div>
      </div>
    </SidebarLayout>
  );
}