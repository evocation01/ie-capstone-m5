import SidebarLayout from '@/components/SidebarLayout';

export default function DashboardPage() {
  return (
    <SidebarLayout>
      <div className="p-8">
        <div className="max-w-7xl mx-auto">
          <h1 className="text-2xl font-bold text-zinc-900 mb-6">Inventory Optimization Dashboard</h1>
          <p className="text-zinc-600 mb-8">A Decision Support System for M5 Supply Chain Forecasting</p>

          {/* Placeholder for the main dashboard content */}
          <div className="bg-white rounded-xl border border-zinc-200 p-8">
            <h2 className="text-lg font-semibold text-zinc-900 mb-4">Forecasting Overview</h2>
            <p className="text-zinc-600">Main forecasting dashboard content will be moved here from the original page.</p>
          </div>
        </div>
      </div>
    </SidebarLayout>
  );
}