"use client";

import React from 'react';
import SidebarLayout from '@/components/SidebarLayout';
import { Target, Layers, TrendingUp, CheckCircle2 } from 'lucide-react';

export default function AboutPage() {
  return (
    <SidebarLayout>
      <div className="p-8">
        <div className="max-w-4xl mx-auto space-y-10">
          
          <header className="animate-fade-in text-center space-y-4">
            <div className="inline-flex items-center px-3 py-1 rounded-full bg-blue-100 text-blue-700 text-xs font-bold uppercase tracking-wider">
              Final Report
            </div>
            <h1 className="text-5xl font-extrabold tracking-tight text-slate-900">Executive Summary</h1>
            <p className="text-xl text-slate-500 font-medium">
              Bridging the gap between statistical accuracy and financial viability in retail supply chains.
            </p>
          </header>

          <div className="glass-panel p-8 rounded-3xl animate-fade-in" style={{ animationDelay: '100ms' }}>
            <div className="flex items-start gap-4 mb-6">
              <div className="bg-amber-100 p-3 rounded-xl">
                <Target className="w-6 h-6 text-amber-600" />
              </div>
              <div>
                <h2 className="text-2xl font-bold text-slate-900">The Business Problem</h2>
              </div>
            </div>
            <div className="prose prose-slate max-w-none text-slate-600 space-y-4">
              <p>
                In large-scale retail environments, accurately forecasting daily sales at the granular <strong>Item-Store level</strong> is notoriously difficult. The data is highly intermittent (sparse), meaning an item might not sell for days, and then sell 2 units unexpectedly.
              </p>
              <p>
                When forecasts are wrong, businesses suffer financially in two ways:
              </p>
              <ul className="list-disc pl-5 space-y-2 font-medium">
                <li><strong>Over-forecasting</strong> leads to excess inventory, tying up capital and incurring high <span className="text-amber-600">Holding Costs</span>.</li>
                <li><strong>Under-forecasting</strong> leads to empty shelves, lost sales, and severe <span className="text-red-500">Stockout Penalties</span>.</li>
              </ul>
              <p>
                The goal of this Capstone project was to evaluate state-of-the-art forecasting models not just on their mathematical error (RMSE), but on their <strong>bottom-line financial impact</strong> through a simulated inventory environment.
              </p>
            </div>
          </div>

          <div className="glass-panel p-8 rounded-3xl animate-fade-in" style={{ animationDelay: '200ms' }}>
            <div className="flex items-start gap-4 mb-6">
              <div className="bg-blue-100 p-3 rounded-xl">
                <Layers className="w-6 h-6 text-blue-600" />
              </div>
              <div>
                <h2 className="text-2xl font-bold text-slate-900">Methodology & Modeling</h2>
              </div>
            </div>
            <div className="space-y-6">
              <p className="text-slate-600">
                We evaluated 10+ models across three distinct architectural tiers on the M5 Forecasting Dataset (Walmart hierarchical sales data).
              </p>
              
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="bg-white/60 p-5 rounded-2xl border border-slate-100">
                  <h3 className="font-bold text-slate-900 mb-2">1. Classical</h3>
                  <p className="text-xs text-slate-500 mb-3">ARIMA, Holt-Winters</p>
                  <p className="text-sm text-slate-600">Fast but failed to capture cross-series relationships. Severely penalized by intermittent demand.</p>
                </div>
                <div className="bg-blue-50/50 p-5 rounded-2xl border border-blue-100 relative overflow-hidden">
                  <div className="absolute top-0 right-0 bg-blue-500 text-white text-[10px] font-bold px-2 py-1 rounded-bl-lg">CHAMPION</div>
                  <h3 className="font-bold text-slate-900 mb-2">2. Machine Learning</h3>
                  <p className="text-xs text-slate-500 mb-3">LightGBM, XGBoost</p>
                  <p className="text-sm text-slate-600">Global models that learned from all items simultaneously using lag and rolling-mean features. Achieved the lowest overall RMSE.</p>
                </div>
                <div className="bg-purple-50/50 p-5 rounded-2xl border border-purple-100">
                  <h3 className="font-bold text-slate-900 mb-2">3. Deep Learning</h3>
                  <p className="text-xs text-slate-500 mb-3">LSTM, DeepAR</p>
                  <p className="text-sm text-slate-600">Standard LSTMs suffered from the &quot;Sparsity Penalty&quot;, but DeepAR solved this using a Negative Binomial likelihood distribution.</p>
                </div>
              </div>
            </div>
          </div>

          <div className="glass-panel p-8 rounded-3xl animate-fade-in" style={{ animationDelay: '300ms' }}>
            <div className="flex items-start gap-4 mb-6">
              <div className="bg-emerald-100 p-3 rounded-xl">
                <TrendingUp className="w-6 h-6 text-emerald-600" />
              </div>
              <div>
                <h2 className="text-2xl font-bold text-slate-900">Key Outcomes & Value</h2>
              </div>
            </div>
            
            <div className="space-y-4">
              <div className="flex gap-3">
                <CheckCircle2 className="w-5 h-5 text-emerald-500 shrink-0 mt-0.5" />
                <p className="text-slate-600"><strong className="text-slate-900">19.5% Cost Reduction:</strong> The LightGBM champion model successfully reduced simulated inventory costs by 19.5% compared to the Naive baseline.</p>
              </div>
              <div className="flex gap-3">
                <CheckCircle2 className="w-5 h-5 text-emerald-500 shrink-0 mt-0.5" />
                <p className="text-slate-600"><strong className="text-slate-900">The Sparsity Penalty Discovery:</strong> We proved that optimizing for Mean Squared Error (MSE) on intermittent data causes standard neural networks (LSTMs) to predict constant fractional values. While mathematically safe, this destroys inventory systems by triggering constant stock re-orders. We mitigated this by implementing DeepAR.</p>
              </div>
              <div className="flex gap-3">
                <CheckCircle2 className="w-5 h-5 text-emerald-500 shrink-0 mt-0.5" />
                <p className="text-slate-600"><strong className="text-slate-900">Dynamic Decision Support:</strong> By building this interactive Next.js dashboard, we translated static data science metrics into a dynamic business tool, allowing stakeholders to adjust risk parameters (Critical Ratio) in real-time.</p>
              </div>
            </div>
          </div>

        </div>
      </div>
    </SidebarLayout>
  );
}
