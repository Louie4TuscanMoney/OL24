/**
 * Main Dashboard Component
 * Real-time NBA predictions and betting dashboard
 * 
 * COMPLETE INTEGRATION:
 * - NBA API (live scores)
 * - ML Model (5.39 MAE predictions)
 * - BetOnline (market odds)
 * - Risk Management (5-layer system)
 */

import { Component, For, Show, onMount, onCleanup } from 'solid-js';
import { wsService } from '../services/websocket';
import GameCardExpanded from './GameCardExpanded';
import SystemStatus from './SystemStatus';

const Dashboard: Component = () => {
  // Get signals from WebSocket service (reactive!)
  const [connected] = wsService.connected;
  const [games] = wsService.games;
  const [patterns] = wsService.patterns;
  const [predictions] = wsService.predictions;
  const [edges] = wsService.edges;
  const [recommendations] = wsService.recommendations;
  const [lastUpdate] = wsService.lastUpdate;

  // Connect on mount
  onMount(() => {
    wsService.connect();
  });

  // Disconnect on unmount
  onCleanup(() => {
    wsService.disconnect();
  });

  // Convert Map to Array for iteration
  const gamesList = () => Array.from(games().values());

  // Live games count
  const liveGamesCount = () => 
    gamesList().filter(g => g.is_live).length;

  // Total edges detected
  const edgesCount = () => 
    Array.from(edges().values()).filter(e => e.has_edge).length;

  return (
    <div class="min-h-screen bg-gray-950 text-gray-100">
      {/* Header */}
      <header class="bg-gray-900 border-b border-gray-800 sticky top-0 z-50">
        <div class="max-w-7xl mx-auto px-4 py-4">
          <div class="flex items-center justify-between">
            <div>
              <h1 class="text-2xl font-bold">🏀 NBA Betting Dashboard</h1>
              <p class="text-sm text-gray-400">Real-time predictions + Risk management</p>
            </div>

            {/* Connection Status */}
            <div class="flex items-center gap-6">
              <div class="text-right">
                <div class="text-sm text-gray-400">Live Games</div>
                <div class="text-xl font-bold">{liveGamesCount()}</div>
              </div>

              <div class="text-right">
                <div class="text-sm text-gray-400">Edges</div>
                <div class="text-xl font-bold text-amber-400">{edgesCount()}</div>
              </div>

              <div class="flex items-center gap-2">
                <span class={`w-3 h-3 rounded-full ${connected() ? 'bg-green-500' : 'bg-red-500'}`}></span>
                <span class="text-sm">
                  {connected() ? 'Connected' : 'Disconnected'}
                </span>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main class="max-w-7xl mx-auto px-4 py-8">
        {/* Connection Message */}
        <Show when={!connected()}>
          <div class="bg-yellow-500/10 border border-yellow-500/30 rounded-lg p-4 mb-6">
            <div class="flex items-center gap-3">
              <span class="text-yellow-500 text-xl">⚠️</span>
              <div>
                <div class="font-semibold">Connecting to NBA API...</div>
                <div class="text-sm text-gray-400">
                  Make sure NBA_API WebSocket is running on port 8765
                </div>
              </div>
            </div>
          </div>
        </Show>

        {/* No Games Message */}
        <Show when={connected() && gamesList().length === 0}>
          <div class="bg-gray-900 rounded-lg p-12 text-center">
            <div class="text-6xl mb-4">🏀</div>
            <div class="text-xl font-semibold mb-2">No live games</div>
            <div class="text-gray-400">
              Waiting for NBA games to start...
            </div>
          </div>
        </Show>

        {/* System Status Sidebar + Games Grid */}
        <Show when={gamesList().length > 0}>
          <div class="grid grid-cols-1 lg:grid-cols-4 gap-6">
            {/* System Status (Sidebar) */}
            <div class="lg:col-span-1">
              <div class="lg:sticky lg:top-24">
                <SystemStatus />
              </div>
            </div>

            {/* Games Grid */}
            <div class="lg:col-span-3">
              <div class="grid grid-cols-1 xl:grid-cols-2 gap-6">
                <For each={gamesList()}>
                  {(game) => (
                    <GameCardExpanded
                      game={game}
                      pattern={patterns().get(game.game_id)}
                      prediction={predictions().get(game.game_id)}
                      edge={edges().get(game.game_id)}
                      recommendation={recommendations().get(game.game_id)}
                    />
                  )}
                </For>
              </div>
            </div>
          </div>
        </Show>

        {/* Footer Info */}
        <div class="mt-8 text-center text-sm text-gray-500">
          Last update: {lastUpdate().toLocaleTimeString()}
          <span class="mx-2">•</span>
          ML MAE: 5.39 points
          <span class="mx-2">•</span>
          Coverage: 94.6%
          <span class="mx-2">•</span>
          Max bet: $750
        </div>
      </main>
    </div>
  );
};

export default Dashboard;

