/**
 * LIVE SNAPSHOTS WIDGET
 * 
 * Displays minute-by-minute scoring progress for a live game
 * Shows score differential trends and momentum
 */

import { createSignal, createEffect, onCleanup, Show, For } from 'solid-js';
import { Line } from 'solid-chartjs';

interface LiveSnapshotsWidgetProps {
  gameId: string;
}

interface Snapshot {
  event_num: number;
  period: number;
  clock: string;
  time_elapsed: number;
  home_score: number;
  away_score: number;
  margin: number;
  timestamp: string;
}

export function LiveSnapshotsWidget(props: LiveSnapshotsWidgetProps) {
  const [snapshots, setSnapshots] = createSignal<Snapshot[]>([]);
  const [loading, setLoading] = createSignal(true);
  const [error, setError] = createSignal<string | null>(null);
  
  const API_BASE = 'https://ol24-production.up.railway.app';

  const fetchSnapshots = async () => {
    try {
      const response = await fetch(`${API_BASE}/api/game/${props.gameId}/play-by-play`);
      const data = await response.json();
      
      if (data.error) {
        setError(data.error);
        return;
      }
      
      setSnapshots(data.snapshots || []);
      setLoading(false);
    } catch (err) {
      setError('Failed to fetch snapshots');
      console.error('Error fetching snapshots:', err);
    }
  };

  // Fetch snapshots immediately
  createEffect(() => {
    fetchSnapshots();
    
    // Refresh every 30 seconds
    const interval = setInterval(fetchSnapshots, 30000);
    onCleanup(() => clearInterval(interval));
  });

  // Chart data for score progression
  const chartData = () => {
    const data = snapshots();
    if (!data.length) {
      return {
        labels: ['No data'],
        datasets: [
          {
            label: 'Score Margin',
            data: [0],
            borderColor: 'rgba(139, 92, 246, 0.8)',
            backgroundColor: 'rgba(139, 92, 246, 0.1)',
            tension: 0.4
          }
        ]
      };
    }

    return {
      labels: data.map(s => `Min ${s.event_num}`),
      datasets: [
        {
          label: 'Score Margin (Home - Away)',
          data: data.map(s => s.margin),
          borderColor: 'rgba(139, 92, 246, 0.8)',
          backgroundColor: 'rgba(139, 92, 246, 0.1)',
          tension: 0.4,
          pointRadius: 4,
          pointHoverRadius: 6,
          pointBackgroundColor: 'rgba(139, 92, 246, 1)',
          pointBorderColor: '#fff',
          pointBorderWidth: 2
        }
      ]
    };
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: true,
        position: 'top' as const,
        labels: {
          color: '#e5e7eb',
          font: {
            size: 12,
            weight: 'bold' as const
          }
        }
      },
      tooltip: {
        backgroundColor: 'rgba(17, 24, 39, 0.9)',
        titleColor: '#e5e7eb',
        bodyColor: '#e5e7eb',
        borderColor: 'rgba(139, 92, 246, 0.5)',
        borderWidth: 1,
        padding: 12,
        callbacks: {
          label: (context: any) => {
            const margin = context.parsed.y;
            const idx = context.dataIndex;
            const snapshot = snapshots()[idx];
            if (!snapshot) return '';
            return `Margin: ${margin > 0 ? '+' : ''}${margin} (${snapshot.away_score}-${snapshot.home_score})`;
          }
        }
      }
    },
    scales: {
      x: {
        ticks: {
          color: '#9ca3af',
          font: {
            size: 10
          }
        },
        grid: {
          color: 'rgba(75, 85, 99, 0.3)'
        }
      },
      y: {
        ticks: {
          color: '#9ca3af',
          font: {
            size: 10
          },
          callback: (value: number) => value > 0 ? `+${value}` : value
        },
        grid: {
          color: 'rgba(75, 85, 99, 0.3)'
        },
        zeroLineColor: 'rgba(139, 92, 246, 0.5)',
        zeroLineWidth: 2
      }
    }
  };

  const latestSnapshot = () => {
    const data = snapshots();
    return data.length > 0 ? data[data.length - 1] : null;
  };

  const momentum = () => {
    const data = snapshots();
    if (data.length < 2) return 0;
    
    const recent = data.slice(-3);
    const marginDelta = recent[recent.length - 1].margin - recent[0].margin;
    return marginDelta;
  };

  const momentumDirection = () => {
    const mom = momentum();
    if (mom > 2) return 'up';
    if (mom < -2) return 'down';
    return 'neutral';
  };

  return (
    <div class="bg-gray-900 rounded-lg p-6 border border-purple-900/30">
      <div class="flex items-center justify-between mb-4">
        <h3 class="text-xl font-bold text-white flex items-center gap-2">
          <span class="text-purple-400">📊</span>
          Minute-by-Minute Analysis
        </h3>
        <button
          onClick={fetchSnapshots}
          class="px-3 py-1 bg-purple-600 hover:bg-purple-700 text-white text-xs rounded transition-colors"
        >
          Refresh
        </button>
      </div>

      <Show when={loading()}>
        <div class="text-center py-8">
          <div class="inline-block animate-spin text-4xl">⏳</div>
          <p class="text-gray-400 mt-2">Loading snapshots...</p>
        </div>
      </Show>

      <Show when={error()}>
        <div class="text-center py-8">
          <p class="text-red-400">{error()}</p>
          <p class="text-gray-500 text-sm mt-2">Snapshots will appear once the cron starts collecting data</p>
        </div>
      </Show>

      <Show when={!loading() && !error() && snapshots().length > 0}>
        <div class="space-y-4">
          {/* Latest Snapshot Info */}
          <Show when={latestSnapshot()}>
            {(snapshot) => (
              <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div class="bg-gray-800/50 rounded-lg p-3">
                  <div class="text-gray-400 text-xs mb-1">Current Snapshot</div>
                  <div class="text-white font-bold text-lg">
                    Minute {snapshot().event_num}
                  </div>
                  <div class="text-gray-500 text-xs mt-1">
                    Q{snapshot().period} {snapshot().clock}
                  </div>
                </div>
                
                <div class="bg-gray-800/50 rounded-lg p-3">
                  <div class="text-gray-400 text-xs mb-1">Score</div>
                  <div class="text-white font-bold text-lg">
                    {snapshot().away_score}-{snapshot().home_score}
                  </div>
                </div>
                
                <div class="bg-gray-800/50 rounded-lg p-3">
                  <div class="text-gray-400 text-xs mb-1">Margin</div>
                  <div class={`font-bold text-lg ${
                    snapshot().margin > 0 ? 'text-green-400' : 
                    snapshot().margin < 0 ? 'text-red-400' : 'text-gray-400'
                  }`}>
                    {snapshot().margin > 0 ? '+' : ''}{snapshot().margin}
                  </div>
                </div>
                
                <div class="bg-gray-800/50 rounded-lg p-3">
                  <div class="text-gray-400 text-xs mb-1">Momentum</div>
                  <div class={`font-bold text-lg flex items-center gap-1 ${
                    momentumDirection() === 'up' ? 'text-green-400' :
                    momentumDirection() === 'down' ? 'text-red-400' :
                    'text-gray-400'
                  }`}>
                    <span>
                      {momentumDirection() === 'up' ? '📈' :
                       momentumDirection() === 'down' ? '📉' : '➡️'}
                    </span>
                    {momentum() > 0 ? '+' : ''}{momentum()}
                  </div>
                </div>
              </div>
            )}
          </Show>

          {/* Chart */}
          <div class="bg-gray-800/30 rounded-lg p-4">
            <div style="height: 250px;">
              <Line data={chartData()} options={chartOptions} />
            </div>
          </div>

          {/* Recent Snapshots Table */}
          <div class="bg-gray-800/30 rounded-lg p-4">
            <h4 class="text-white font-semibold mb-3">Recent Snapshots</h4>
            <div class="overflow-x-auto">
              <table class="w-full">
                <thead>
                  <tr class="text-left text-xs text-gray-400 border-b border-gray-700">
                    <th class="pb-2 px-2">Min</th>
                    <th class="pb-2 px-2">Q</th>
                    <th class="pb-2 px-2">Clock</th>
                    <th class="pb-2 px-2">Score</th>
                    <th class="pb-2 px-2">Margin</th>
                  </tr>
                </thead>
                <tbody>
                  <For each={snapshots().slice(-10).reverse()}>
                    {(snapshot) => (
                      <tr class="text-sm border-b border-gray-700/50 hover:bg-gray-700/30">
                        <td class="py-2 px-2 text-gray-300 font-mono">
                          {snapshot.event_num}
                        </td>
                        <td class="py-2 px-2 text-gray-400">
                          Q{snapshot.period}
                        </td>
                        <td class="py-2 px-2 text-gray-400 font-mono">
                          {snapshot.clock}
                        </td>
                        <td class="py-2 px-2 text-white font-semibold">
                          {snapshot.away_score}-{snapshot.home_score}
                        </td>
                        <td class={`py-2 px-2 font-semibold ${
                          snapshot.margin > 0 ? 'text-green-400' :
                          snapshot.margin < 0 ? 'text-red-400' : 'text-gray-400'
                        }`}>
                          {snapshot.margin > 0 ? '+' : ''}{snapshot.margin}
                        </td>
                      </tr>
                    )}
                  </For>
                </tbody>
              </table>
            </div>
            
            <Show when={snapshots().length === 0}>
              <div class="text-center py-8 text-gray-400">
                No snapshots yet. Waiting for data collection...
              </div>
            </Show>
          </div>
        </div>
      </Show>

      <style>{`
        .bg-gray-900 {
          background-color: #111827;
        }
        .bg-gray-800\/50 {
          background-color: rgba(31, 41, 55, 0.5);
        }
        .bg-gray-800\/30 {
          background-color: rgba(31, 41, 55, 0.3);
        }
        .bg-gray-700\/30 {
          background-color: rgba(55, 65, 81, 0.3);
        }
      `}</style>
    </div>
  );
}

