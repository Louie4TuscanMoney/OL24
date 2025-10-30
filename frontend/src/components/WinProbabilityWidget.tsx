/**
 * WIN PROBABILITY WIDGET
 * 
 * Displays minute-by-minute win probability timeline
 * Shows how win probability changes throughout the game
 */

import { createSignal, createEffect, onCleanup, Show, For } from 'solid-js';
import { Line } from 'solid-chartjs';

interface WinProbabilityWidgetProps {
  gameId: string;
}

interface WinProbabilityData {
  period: number;
  time_elapsed: number;
  home_win_prob: number;
  away_win_prob: number;
  margin_prediction: number;
  confidence: number;
  timestamp: string;
}

export function WinProbabilityWidget(props: WinProbabilityWidgetProps) {
  const [probabilities, setProbabilities] = createSignal<WinProbabilityData[]>([]);
  const [loading, setLoading] = createSignal(true);
  const [error, setError] = createSignal<string | null>(null);
  
  const API_BASE = 'https://ol24-production.up.railway.app';

  const fetchProbabilities = async () => {
    try {
      const response = await fetch(`${API_BASE}/api/game/${props.gameId}/win-probability`);
      const data = await response.json();
      
      if (data.error) {
        setError(data.error);
        setLoading(false);
        return;
      }
      
      setProbabilities(data.timeline || []);
      setLoading(false);
    } catch (err) {
      setError('Failed to fetch win probabilities');
      console.error('Error fetching probabilities:', err);
    }
  };

  // Fetch probabilities immediately
  createEffect(() => {
    fetchProbabilities();
    
    // Refresh every 30 seconds
    const interval = setInterval(fetchProbabilities, 30000);
    onCleanup(() => clearInterval(interval));
  });

  // Chart data for win probability over time
  const chartData = () => {
    const data = probabilities();
    if (!data.length) {
      return {
        labels: ['No data'],
        datasets: [
          {
            label: 'Home Win Probability',
            data: [50],
            borderColor: 'rgba(59, 130, 246, 0.8)',
            backgroundColor: 'rgba(59, 130, 246, 0.1)',
            tension: 0.4
          },
          {
            label: 'Away Win Probability',
            data: [50],
            borderColor: 'rgba(239, 68, 68, 0.8)',
            backgroundColor: 'rgba(239, 68, 68, 0.1)',
            tension: 0.4
          }
        ]
      };
    }

    return {
      labels: data.map(d => `${d.time_elapsed}s`),
      datasets: [
        {
          label: 'Home Win Probability',
          data: data.map(d => d.home_win_prob),
          borderColor: 'rgba(59, 130, 246, 0.8)',
          backgroundColor: 'rgba(59, 130, 246, 0.1)',
          tension: 0.4,
          fill: true
        },
        {
          label: 'Away Win Probability',
          data: data.map(d => d.away_win_prob),
          borderColor: 'rgba(239, 68, 68, 0.8)',
          backgroundColor: 'rgba(239, 68, 68, 0.1)',
          tension: 0.4,
          fill: true
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
        padding: 12
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
        min: 0,
        max: 100,
        ticks: {
          color: '#9ca3af',
          font: {
            size: 10
          },
          callback: (value: number) => `${value}%`
        },
        grid: {
          color: 'rgba(75, 85, 99, 0.3)'
        }
      }
    }
  };

  const latest = () => {
    const data = probabilities();
    return data.length > 0 ? data[data.length - 1] : null;
  };

  const homeFavored = () => {
    const latest_data = latest();
    return latest_data ? latest_data.home_win_prob > 50 : false;
  };

  return (
    <div class="bg-gray-900 rounded-lg p-6 border border-blue-900/30">
      <div class="flex items-center justify-between mb-4">
        <h3 class="text-xl font-bold text-white flex items-center gap-2">
          <span class="text-blue-400">📈</span>
          Minute-by-Minute Win Probability
        </h3>
        <button
          onClick={fetchProbabilities}
          class="px-3 py-1 bg-blue-600 hover:bg-blue-700 text-white text-xs rounded transition-colors"
        >
          Refresh
        </button>
      </div>

      <Show when={loading()}>
        <div class="text-center py-8">
          <div class="inline-block animate-spin text-4xl">⏳</div>
          <p class="text-gray-400 mt-2">Loading win probabilities...</p>
        </div>
      </Show>

      <Show when={error()}>
        <div class="text-center py-8">
          <p class="text-red-400">{error()}</p>
          <p class="text-gray-500 text-sm mt-2">
            Probabilities will appear after 6+ minutes of data collection
          </p>
        </div>
      </Show>

      <Show when={!loading() && !error() && probabilities().length > 0}>
        <div class="space-y-4">
          {/* Latest Probability */}
          <Show when={latest()}>
            {(latest_data) => (
              <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div class="bg-blue-950/30 rounded-lg p-3 border border-blue-800/50">
                  <div class="text-gray-400 text-xs mb-1">Home Win Prob</div>
                  <div class="text-2xl font-bold text-blue-400">
                    {latest_data().home_win_prob}%
                  </div>
                </div>
                
                <div class="bg-red-950/30 rounded-lg p-3 border border-red-800/50">
                  <div class="text-gray-400 text-xs mb-1">Away Win Prob</div>
                  <div class="text-2xl font-bold text-red-400">
                    {latest_data().away_win_prob}%
                  </div>
                </div>
                
                <div class="bg-gray-800/30 rounded-lg p-3">
                  <div class="text-gray-400 text-xs mb-1">Margin Pred</div>
                  <div class={`text-2xl font-bold ${
                    latest_data().margin_prediction > 0 ? 'text-green-400' : 'text-orange-400'
                  }`}>
                    {latest_data().margin_prediction > 0 ? '+' : ''}{latest_data().margin_prediction.toFixed(1)}
                  </div>
                </div>
                
                <div class="bg-gray-800/30 rounded-lg p-3">
                  <div class="text-gray-400 text-xs mb-1">Confidence</div>
                  <div class="text-2xl font-bold text-purple-400">
                    {latest_data().confidence}%
                  </div>
                </div>
              </div>
            )}
          </Show>

          {/* Chart */}
          <div class="bg-gray-800/30 rounded-lg p-4">
            <div style="height: 300px;">
              <Line data={chartData()} options={chartOptions} />
            </div>
          </div>

          {/* Favored Team Indicator */}
          <Show when={homeFavored() !== null}>
            <div class={`rounded-lg p-4 ${
              homeFavored() ? 'bg-blue-950/30 border border-blue-800/50' : 'bg-red-950/30 border border-red-800/50'
            }`}>
              <div class="flex items-center gap-2">
                <span class="text-2xl">
                  {homeFavored() ? '🏠' : '✈️'}
                </span>
                <div>
                  <div class="text-white font-semibold">
                    {homeFavored() ? 'Home' : 'Away'} Team Currently Favored
                  </div>
                  <div class="text-gray-400 text-sm">
                    {latest()?.home_win_prob || 50}% vs {latest()?.away_win_prob || 50}%
                  </div>
                </div>
              </div>
            </div>
          </Show>
        </div>
      </Show>

      <Show when={!loading() && !error() && probabilities().length === 0}>
        <div class="text-center py-8 text-gray-400">
          No win probability data yet.
          <br />
          <span class="text-sm">Waiting for 6+ minutes of game data...</span>
        </div>
      </Show>

      <style>{`
        .bg-gray-900 {
          background-color: #111827;
        }
        .bg-gray-800\/30 {
          background-color: rgba(31, 41, 55, 0.3);
        }
        .bg-blue-950\/30 {
          background-color: rgba(0, 0, 60, 0.3);
        }
        .bg-red-950\/30 {
          background-color: rgba(60, 0, 0, 0.3);
        }
      `}</style>
    </div>
  );
}

