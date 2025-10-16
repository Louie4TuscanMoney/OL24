/**
 * System Status Component
 * Shows status of all backend systems
 */

import { Component, createSignal, onMount } from 'solid-js';

interface SystemHealth {
  nba_api: 'online' | 'offline';
  ml_model: 'online' | 'offline';
  betonline: 'online' | 'offline';
  risk_system: 'online' | 'offline';
}

const SystemStatus: Component = () => {
  const [health, setHealth] = createSignal<SystemHealth>({
    nba_api: 'offline',
    ml_model: 'offline',
    betonline: 'offline',
    risk_system: 'online' // Always online (local)
  });

  const [bankroll, setBankroll] = createSignal(5000);
  const [totalBets, setTotalBets] = createSignal(0);
  const [winRate, setWinRate] = createSignal(0.62);

  const getStatusColor = (status: 'online' | 'offline') => 
    status === 'online' ? 'green' : 'red';

  const getStatusEmoji = (status: 'online' | 'offline') => 
    status === 'online' ? '✅' : '❌';

  return (
    <div class="bg-gray-900 rounded-lg p-6 border border-gray-800">
      <h2 class="text-lg font-bold mb-4">System Status</h2>

      {/* Backend Systems */}
      <div class="space-y-3 mb-6">
        <div class="flex items-center justify-between text-sm">
          <span class="text-gray-400">NBA API</span>
          <span class={`text-${getStatusColor(health().nba_api)}-400`}>
            {getStatusEmoji(health().nba_api)} {health().nba_api}
          </span>
        </div>

        <div class="flex items-center justify-between text-sm">
          <span class="text-gray-400">ML Model (5.39 MAE)</span>
          <span class={`text-${getStatusColor(health().ml_model)}-400`}>
            {getStatusEmoji(health().ml_model)} {health().ml_model}
          </span>
        </div>

        <div class="flex items-center justify-between text-sm">
          <span class="text-gray-400">BetOnline Scraper</span>
          <span class={`text-${getStatusColor(health().betonline)}-400`}>
            {getStatusEmoji(health().betonline)} {health().betonline}
          </span>
        </div>

        <div class="flex items-center justify-between text-sm">
          <span class="text-gray-400">Risk System (5 layers)</span>
          <span class={`text-${getStatusColor(health().risk_system)}-400`}>
            {getStatusEmoji(health().risk_system)} {health().risk_system}
          </span>
        </div>
      </div>

      {/* Bankroll Stats */}
      <div class="border-t border-gray-800 pt-4">
        <div class="space-y-3">
          <div>
            <div class="text-xs text-gray-500 mb-1">Bankroll</div>
            <div class="text-2xl font-bold">${bankroll().toLocaleString()}</div>
          </div>

          <div class="grid grid-cols-2 gap-4">
            <div>
              <div class="text-xs text-gray-500 mb-1">Total Bets</div>
              <div class="text-lg font-semibold">{totalBets()}</div>
            </div>

            <div>
              <div class="text-xs text-gray-500 mb-1">Win Rate</div>
              <div class="text-lg font-semibold">{(winRate() * 100).toFixed(1)}%</div>
            </div>
          </div>

          <div>
            <div class="text-xs text-gray-500 mb-1">Max Bet (Safety Limit)</div>
            <div class="text-lg font-semibold text-amber-400">$750</div>
            <div class="text-xs text-gray-500">15% of original $5,000</div>
          </div>
        </div>
      </div>

      {/* Risk Limits */}
      <div class="border-t border-gray-800 pt-4 mt-4">
        <div class="text-xs text-gray-500 mb-2">Safety Limits</div>
        <div class="space-y-1 text-xs text-gray-400">
          <div>• Max single bet: $750</div>
          <div>• Max portfolio: $2,500</div>
          <div>• Reserve held: $2,500</div>
          <div>• Max progression: 3 levels</div>
        </div>
      </div>
    </div>
  );
};

export default SystemStatus;

