/**
 * MAMBA FEATURES WIDGET
 * 
 * Displays the 33 Mamba features extracted from game data
 * Shows feature values and their significance
 */

import { createSignal, createEffect, onCleanup, Show } from 'solid-js';

interface MambaFeaturesWidgetProps {
  gameId: string;
}

interface MambaFeatures {
  game_id: string;
  features: number[];
  prediction: number;
  confidence: number;
  triggered_at: string;
  period: number;
  clock: string;
}

export function MambaFeaturesWidget(props: MambaFeaturesWidgetProps) {
  const [features, setFeatures] = createSignal<MambaFeatures | null>(null);
  const [loading, setLoading] = createSignal(true);
  const [error, setError] = createSignal<string | null>(null);
  
  const API_BASE = 'https://ol24-production.up.railway.app';

  const fetchFeatures = async () => {
    try {
      const response = await fetch(`${API_BASE}/api/ml/prediction/${props.gameId}`);
      const data = await response.json();
      
      if (data.error) {
        setError(data.error);
        setLoading(false);
        return;
      }
      
      setFeatures({
        game_id: props.gameId,
        features: data.features || [],
        prediction: data.prediction || 0,
        confidence: data.confidence || 0,
        triggered_at: data.triggered_at || '',
        period: data.period || 0,
        clock: data.clock || ''
      });
      setLoading(false);
    } catch (err) {
      setError('Failed to fetch Mamba features');
      console.error('Error fetching features:', err);
    }
  };

  // Fetch features immediately
  createEffect(() => {
    fetchFeatures();
    
    // Refresh every 30 seconds
    const interval = setInterval(fetchFeatures, 30000);
    onCleanup(() => clearInterval(interval));
  });

  const featureNames = [
    'Pattern Mean', 'Pattern Std', 'Pattern Min', 'Pattern Max', 'Pattern Range',
    'Trend', 'Velocity', 'Acceleration', 'Volatility', 'Momentum', 'Lead Changes', 'Max Swing',
    'Spectral 1', 'Spectral 2', 'Spectral 3', 'Spectral 4', 'Spectral 5', 'Spectral 6',
    'Autocorr 1', 'Autocorr 2', 'Autocorr 3', 'Autocorr 4', 'Autocorr 5', 'Autocorr 6',
    'Freq Peak', 'Freq Power', 'Energy', 'Entropy', 'Complexity', 'Chaos', 'Predictability', 'Stability', 'Dominance'
  ];

  const groupFeatures = (feat: number[]) => {
    return {
      pattern: feat.slice(0, 12),
      spectral: feat.slice(12, 18),
      autocorr: feat.slice(18, 24),
      frequency: feat.slice(24, 33)
    };
  };

  return (
    <div class="bg-gray-900 rounded-lg p-6 border border-purple-900/30">
      <div class="flex items-center justify-between mb-4">
        <h3 class="text-xl font-bold text-white flex items-center gap-2">
          <span class="text-purple-400">🧠</span>
          Mamba 33 Features
        </h3>
        <button
          onClick={fetchFeatures}
          class="px-3 py-1 bg-purple-600 hover:bg-purple-700 text-white text-xs rounded transition-colors"
        >
          Refresh
        </button>
      </div>

      <Show when={loading()}>
        <div class="text-center py-8">
          <div class="inline-block animate-spin text-4xl">⏳</div>
          <p class="text-gray-400 mt-2">Loading Mamba features...</p>
        </div>
      </Show>

      <Show when={error()}>
        <div class="text-center py-8">
          <p class="text-red-400">{error()}</p>
          <p class="text-gray-500 text-sm mt-2">
            Features will appear once Mamba triggers at Q2 6:00
          </p>
        </div>
      </Show>

      <Show when={!loading() && !error() && features()}>
        {(data) => {
          const grouped = groupFeatures(data().features);
          
          return (
            <div class="space-y-6">
              {/* Prediction Summary */}
              <div class="bg-purple-950/30 rounded-lg p-4 border border-purple-800/50">
                <div class="grid grid-cols-3 gap-4">
                  <div>
                    <div class="text-gray-400 text-xs mb-1">Prediction</div>
                    <div class={`text-2xl font-bold ${
                      data().prediction > 0 ? 'text-green-400' : 'text-red-400'
                    }`}>
                      {data().prediction > 0 ? '+' : ''}{data().prediction.toFixed(1)}
                    </div>
                  </div>
                  <div>
                    <div class="text-gray-400 text-xs mb-1">Confidence</div>
                    <div class="text-2xl font-bold text-purple-400">
                      {data().confidence}%
                    </div>
                  </div>
                  <div>
                    <div class="text-gray-400 text-xs mb-1">Triggered At</div>
                    <div class="text-white font-bold text-sm">
                      Q{data().period} {data().clock}
                    </div>
                    <div class="text-gray-500 text-xs mt-1">
                      {new Date(data().triggered_at).toLocaleTimeString()}
                    </div>
                  </div>
                </div>
              </div>

              {/* Feature Groups */}
              <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                {/* Pattern Statistics */}
                <div class="bg-gray-800/30 rounded-lg p-4">
                  <h4 class="text-white font-semibold mb-3 text-sm">Pattern Statistics</h4>
                  <div class="space-y-2">
                    {grouped.pattern.slice(0, 8).map((val, idx) => (
                      <div class="flex items-center justify-between">
                        <span class="text-gray-400 text-xs">{featureNames[idx]}:</span>
                        <span class="text-white font-mono text-sm">{val.toFixed(2)}</span>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Spectral Features */}
                <div class="bg-gray-800/30 rounded-lg p-4">
                  <h4 class="text-white font-semibold mb-3 text-sm">Spectral Analysis</h4>
                  <div class="space-y-2">
                    {grouped.spectral.map((val, idx) => (
                      <div class="flex items-center justify-between">
                        <span class="text-gray-400 text-xs">{featureNames[idx + 12]}:</span>
                        <span class="text-white font-mono text-sm">{val.toFixed(3)}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              {/* Feature Visualization Bars */}
              <div class="bg-gray-800/30 rounded-lg p-4">
                <h4 class="text-white font-semibold mb-3 text-sm">Feature Importance (Top 10)</h4>
                <div class="space-y-2">
                  {data().features
                    .map((val, idx) => ({ val, idx }))
                    .sort((a, b) => Math.abs(b.val) - Math.abs(a.val))
                    .slice(0, 10)
                    .map(({ val, idx }) => (
                      <div>
                        <div class="flex items-center justify-between mb-1">
                          <span class="text-gray-400 text-xs">{featureNames[idx]}:</span>
                          <span class="text-white font-mono text-xs">{val.toFixed(3)}</span>
                        </div>
                        <div class="w-full bg-gray-700 rounded-full h-1.5">
                          <div
                            class={`h-1.5 rounded-full ${
                              val > 0 ? 'bg-green-500' : 'bg-red-500'
                            }`}
                            style={`width: ${Math.min(100, Math.abs(val) * 10)}%`}
                          ></div>
                        </div>
                      </div>
                    ))}
                </div>
              </div>
            </div>
          );
        }}
      </Show>

      <style>{`
        .bg-gray-900 {
          background-color: #111827;
        }
        .bg-gray-800\/30 {
          background-color: rgba(31, 41, 55, 0.3);
        }
        .bg-purple-950\/30 {
          background-color: rgba(30, 0, 60, 0.3);
        }
      `}</style>
    </div>
  );
}

