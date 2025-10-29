import { type Component } from 'solid-js';
import { TradingDashboard } from './TradingDashboardLive';

const TradingPage: Component = () => {
  return (
    <div class="min-h-screen bg-slate-950">
      <TradingDashboard />
    </div>
  );
};

export default TradingPage;
