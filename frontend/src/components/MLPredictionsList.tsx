import { type Component, For, Show, createSignal, onMount, onCleanup } from 'solid-js';

interface MLPrediction {
  game_id: string;
  home_team: string;
  away_team: string;
  point_forecast: number;
  interval_lower: number;
  interval_upper: number;
  model_confidence: number;
  edge_detected: boolean;
  quarter: number;
  time_remaining: string;
  prediction_timestamp: string;
}

const MLPredictionsList: Component = () => {
  const [predictions, setPredictions] = createSignal<MLPrediction[]>([]);
  const [loading, setLoading] = createSignal(true);
  const API_BASE = 'https://ol24-production.up.railway.app';

  const fetchPredictions = async () => {
    try {
      const response = await fetch(`${API_BASE}/api/ml/predictions/active`);
      const data = await response.json();
      setPredictions(data);
      setLoading(false);
    } catch (error) {
      console.error('Error fetching predictions:', error);
      setLoading(false);
    }
  };

  onMount(() => {
    fetchPredictions();
    // Refresh every 30 seconds
    const interval = setInterval(fetchPredictions, 30000);
    onCleanup(() => clearInterval(interval));
  });

  return (
    <div class="modern-card">
      <div class="flex items-center justify-between mb-4">
        <h2 class="text-xl font-bold text-white">Active ML Predictions</h2>
        <div class="px-3 py-1 bg-purple-500/20 border border-purple-500/30 rounded-full">
          <span class="text-purple-400 text-xs font-semibold">{predictions().length} Active</span>
        </div>
      </div>

      <Show when={!loading()} fallback={
        <div class="text-center py-8 text-gray-400">
          Loading predictions...
        </div>
      }>
        <Show when={predictions().length > 0} fallback={
          <div class="text-center py-8 text-gray-500">
            No active predictions. Waiting for live games...
          </div>
        }>
          <div class="space-y-3">
            <For each={predictions()}>
              {(pred) => (
                <div class="bg-gray-800/50 border border-gray-700/50 rounded-lg p-4 hover:bg-gray-800/70 transition-colors">
                  {/* Game Header */}
                  <div class="flex items-center justify-between mb-3">
                    <div class="text-sm font-semibold text-white">
                      {pred.away_team} @ {pred.home_team}
                    </div>
                    <div class="text-xs text-gray-400">
                      Q{pred.quarter} {pred.time_remaining}
                    </div>
                  </div>

                  {/* Prediction Details */}
                  <div class="grid grid-cols-3 gap-3 mb-3">
                    <div class="text-center">
                      <div class="text-xs text-gray-500 mb-1">Forecast</div>
                      <div class="text-lg font-bold text-purple-400">
                        {pred.point_forecast >= 0 ? '+' : ''}{pred.point_forecast.toFixed(1)}
                      </div>
                    </div>
                    <div class="text-center">
                      <div class="text-xs text-gray-500 mb-1">Confidence</div>
                      <div class={`text-lg font-bold ${
                        pred.model_confidence > 0.8 ? 'text-green-400' :
                        pred.model_confidence > 0.6 ? 'text-yellow-400' :
                        'text-red-400'
                      }`}>
                        {(pred.model_confidence * 100).toFixed(0)}%
                      </div>
                    </div>
                    <div class="text-center">
                      <div class="text-xs text-gray-500 mb-1">Edge</div>
                      <div class={`text-lg font-bold ${pred.edge_detected ? 'text-green-400' : 'text-gray-600'}`}>
                        {pred.edge_detected ? '✓' : '—'}
                      </div>
                    </div>
                  </div>

                  {/* Interval */}
                  <div class="text-xs text-center text-gray-500">
                    90% CI: [{pred.interval_lower.toFixed(1)}, {pred.interval_upper.toFixed(1)}]
                  </div>

                  {/* Edge Badge */}
                  <Show when={pred.edge_detected}>
                    <div class="mt-2 bg-green-500/20 border border-green-500/30 rounded px-2 py-1 text-center">
                      <span class="text-green-400 text-xs font-semibold">Edge Detected</span>
                    </div>
                  </Show>
                </div>
              )}
            </For>
          </div>
        </Show>
      </Show>
    </div>
  );
};

export default MLPredictionsList;

