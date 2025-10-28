import { type Component, Show } from 'solid-js';

interface MLPrediction {
  point_forecast: number;
  interval_lower: number;
  interval_upper: number;
  model_confidence: number;
  edge_detected: boolean;
  edge_magnitude?: number;
  prediction_timestamp: string;
  quarter: number;
  time_remaining: string;
}

interface Props {
  gameId: string;
  prediction?: MLPrediction;
  isQ26Min?: boolean;
}

const MLPredictionBox: Component<Props> = (props) => {
  return (
    <Show when={props.prediction} fallback={
      <div class="modern-card bg-gray-800/30 border-gray-700/50 text-center py-3">
        <div class="text-gray-500 text-xs">Waiting for Q2 6:00...</div>
      </div>
    }>
      {(pred) => (
        <div class={`modern-card transition-all duration-300 ${
          props.isQ26Min 
            ? 'bg-gradient-to-br from-green-900/40 to-emerald-900/40 border-green-500/50 ring-2 ring-green-500/30 shadow-lg shadow-green-500/20' 
            : 'bg-gray-800/30 border-gray-700/30'
        }`}>
          {/* Header */}
          <div class="flex items-center justify-between mb-3">
            <div class="flex items-center gap-2">
              <div class={`w-2 h-2 rounded-full ${props.isQ26Min ? 'bg-green-500 live-pulse' : 'bg-blue-500'}`}></div>
              <span class={`text-xs font-bold uppercase tracking-wider ${
                props.isQ26Min ? 'text-green-400' : 'text-blue-400'
              }`}>
                {props.isQ26Min ? '🎯 Trade Signal' : 'ML Prediction'}
              </span>
            </div>
            <div class="text-xs text-gray-400">
              Q{pred().quarter} {pred().time_remaining}
            </div>
          </div>

          {/* Forecast */}
          <div class="text-center mb-3">
            <div class="text-xs text-gray-400 mb-1">Point Spread Forecast</div>
            <div class={`text-3xl font-bold ${props.isQ26Min ? 'text-green-400' : 'text-white'}`}>
              {pred().point_forecast >= 0 ? '+' : ''}{pred().point_forecast.toFixed(1)}
            </div>
            <div class="text-xs text-gray-500 mt-1">
              90% CI: [{pred().interval_lower.toFixed(1)}, {pred().interval_upper.toFixed(1)}]
            </div>
          </div>

          {/* Edge Detection */}
          <Show when={pred().edge_detected}>
            <div class="bg-green-500/20 border border-green-500/30 rounded-lg p-2 mb-3">
              <div class="flex items-center justify-between text-xs">
                <span class="text-green-400 font-semibold">Edge Detected</span>
                <span class="text-green-300 font-bold">
                  {pred().edge_magnitude ? `+${pred().edge_magnitude?.toFixed(1)}` : '✓'}
                </span>
              </div>
            </div>
          </Show>

          {/* Confidence */}
          <div class="flex items-center justify-between text-xs">
            <span class="text-gray-400">Model Confidence</span>
            <div class="flex items-center gap-2">
              <div class="w-16 h-1.5 bg-gray-700 rounded-full overflow-hidden">
                <div 
                  class={`h-full ${
                    pred().model_confidence > 0.8 ? 'bg-green-500' :
                    pred().model_confidence > 0.6 ? 'bg-yellow-500' :
                    'bg-red-500'
                  } transition-all duration-300`}
                  style={{ width: `${pred().model_confidence * 100}%` }}
                ></div>
              </div>
              <span class={`font-semibold ${
                pred().model_confidence > 0.8 ? 'text-green-400' :
                pred().model_confidence > 0.6 ? 'text-yellow-400' :
                'text-red-400'
              }`}>
                {(pred().model_confidence * 100).toFixed(0)}%
              </span>
            </div>
          </div>

          {/* Trade Window (only for Q2 6:00) */}
          <Show when={props.isQ26Min}>
            <div class="mt-3 pt-3 border-t border-green-500/30">
              <div class="text-center">
                <div class="text-xs text-green-400 font-semibold mb-1">⏰ Trade Window Active</div>
                <div class="text-xs text-gray-400">Valid until Q2 5:00</div>
              </div>
            </div>
          </Show>

          {/* Timestamp */}
          <div class="mt-2 text-center text-xs text-gray-600">
            Updated: {new Date(pred().prediction_timestamp).toLocaleTimeString()}
          </div>
        </div>
      )}
    </Show>
  );
};

export default MLPredictionBox;

