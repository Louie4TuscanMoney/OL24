/**
 * Feature Details Modal - Shows ALL 33 Mamba Features + Math
 * Opens when user clicks on a prediction
 */

import { Component, Show, For } from 'solid-js';
import type { EnhancedPrediction, ScorePattern } from '../types';

interface Props {
  isOpen: boolean;
  onClose: () => void;
  prediction: EnhancedPrediction;
  pattern: ScorePattern[];
  gameInfo: {
    home_team: string;
    away_team: string;
    game_id: string;
  };
}

const FeatureDetailsModal: Component<Props> = (props) => {
  const formatNumber = (num: number, decimals: number = 2) => {
    return num.toFixed(decimals);
  };

  const formatPercent = (num: number) => {
    return `${(num * 100).toFixed(1)}%`;
  };

  return (
    <Show when={props.isOpen}>
      <div class="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80" onClick={props.onClose}>
        <div 
          class="bg-gray-900 rounded-xl border border-gray-700 max-w-6xl w-full max-h-[90vh] overflow-y-auto"
          onClick={(e) => e.stopPropagation()}
        >
          {/* Header */}
          <div class="sticky top-0 bg-gray-800 border-b border-gray-700 px-6 py-4 flex justify-between items-center">
            <div>
              <h2 class="text-2xl font-bold text-white">
                🤖 ML Prediction Details
              </h2>
              <p class="text-sm text-gray-400 mt-1">
                {props.gameInfo.away_team} @ {props.gameInfo.home_team} • {props.gameInfo.game_id}
              </p>
            </div>
            <button 
              onClick={props.onClose}
              class="text-gray-400 hover:text-white transition-colors"
            >
              <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"/>
              </svg>
            </button>
          </div>

          <div class="p-6 space-y-6">
            {/* Prediction Summary */}
            <div class="bg-blue-500/10 border border-blue-500/30 rounded-lg p-6">
              <h3 class="text-lg font-semibold text-blue-400 mb-4">Prediction Summary</h3>
              <div class="grid grid-cols-3 gap-6">
                <div>
                  <div class="text-sm text-gray-400">Point Forecast</div>
                  <div class="text-3xl font-bold text-white mt-1">
                    {formatNumber(props.prediction.point_forecast, 1)}
                  </div>
                </div>
                <div>
                  <div class="text-sm text-gray-400">Confidence Interval (95%)</div>
                  <div class="text-xl font-semibold text-white mt-1">
                    [{formatNumber(props.prediction.interval_lower, 1)}, {formatNumber(props.prediction.interval_upper, 1)}]
                  </div>
                </div>
                <div>
                  <div class="text-sm text-gray-400">Model Confidence</div>
                  <div class="text-3xl font-bold text-green-400 mt-1">
                    {formatPercent(props.prediction.model_confidence)}
                  </div>
                </div>
              </div>
            </div>

            {/* 18-Minute Pattern Visualization */}
            <div class="bg-gray-800 rounded-lg p-6">
              <h3 class="text-lg font-semibold text-white mb-4">📊 18-Minute Pattern</h3>
              <div class="grid grid-cols-6 gap-2">
                <For each={props.pattern}>
                  {(point) => (
                    <div class="bg-gray-700 rounded p-2 text-center">
                      <div class="text-xs text-gray-400">Min {point.minute}</div>
                      <div class={`text-lg font-bold ${point.differential > 0 ? 'text-green-400' : point.differential < 0 ? 'text-red-400' : 'text-gray-300'}`}>
                        {point.differential > 0 ? '+' : ''}{point.differential}
                      </div>
                    </div>
                  )}
                </For>
              </div>
            </div>

            <Show when={props.prediction.features}>
              {/* CATEGORY 1: Pattern Analysis (10 features) */}
              <div class="bg-gray-800 rounded-lg p-6">
                <h3 class="text-lg font-semibold text-purple-400 mb-4">
                  📈 Pattern Analysis (10 Features)
                </h3>
                <div class="grid grid-cols-2 gap-4">
                  <FeatureItem 
                    name="Mean Differential" 
                    value={formatNumber(props.prediction.features!.pattern_analysis.mean_diff)} 
                    description="Average score differential over 18 minutes"
                  />
                  <FeatureItem 
                    name="Standard Deviation" 
                    value={formatNumber(props.prediction.features!.pattern_analysis.std_diff)} 
                    description="Volatility of scoring pattern"
                  />
                  <FeatureItem 
                    name="Trend (Slope)" 
                    value={formatNumber(props.prediction.features!.pattern_analysis.trend, 4)} 
                    description="Linear trend direction (positive = pulling away)"
                  />
                  <FeatureItem 
                    name="Volatility" 
                    value={formatNumber(props.prediction.features!.pattern_analysis.volatility)} 
                    description="Variance in differential changes"
                  />
                  <FeatureItem 
                    name="Velocity" 
                    value={formatNumber(props.prediction.features!.pattern_analysis.velocity)} 
                    description="Rate of change in differential"
                  />
                  <FeatureItem 
                    name="Acceleration" 
                    value={formatNumber(props.prediction.features!.pattern_analysis.acceleration)} 
                    description="Change in velocity (momentum shift)"
                  />
                  <FeatureItem 
                    name="Recent Momentum" 
                    value={formatNumber(props.prediction.features!.pattern_analysis.recent_momentum)} 
                    description="Last 5-minute trend strength"
                  />
                  <FeatureItem 
                    name="Lead Changes" 
                    value={props.prediction.features!.pattern_analysis.lead_changes.toString()} 
                    description="Number of times lead changed hands"
                  />
                  <FeatureItem 
                    name="Max Swing" 
                    value={formatNumber(props.prediction.features!.pattern_analysis.max_swing)} 
                    description="Largest point swing in single play"
                  />
                  <FeatureItem 
                    name="Comeback Potential" 
                    value={formatNumber(props.prediction.features!.pattern_analysis.comeback_potential, 3)} 
                    description="Likelihood of comeback (0-1)"
                  />
                </div>
              </div>

              {/* CATEGORY 2: Spectral Features (6 features) */}
              <div class="bg-gray-800 rounded-lg p-6">
                <h3 class="text-lg font-semibold text-cyan-400 mb-4">
                  🌊 Spectral Analysis (6 Features)
                </h3>
                <div class="grid grid-cols-2 gap-4">
                  <FeatureItem 
                    name="Spectral Energy" 
                    value={formatNumber(props.prediction.features!.spectral.spectral_energy, 0)} 
                    description="Total energy in frequency domain (FFT)"
                  />
                  <FeatureItem 
                    name="Spectral Entropy" 
                    value={formatNumber(props.prediction.features!.spectral.spectral_entropy)} 
                    description="Spread of frequencies (high = chaotic)"
                  />
                  <FeatureItem 
                    name="Low Frequency Power" 
                    value={formatNumber(props.prediction.features!.spectral.low_freq_power, 0)} 
                    description="Slow/steady momentum trends"
                    highlight={true}
                  />
                  <FeatureItem 
                    name="Mid Frequency Power" 
                    value={formatNumber(props.prediction.features!.spectral.mid_freq_power, 0)} 
                    description="Medium runs and counter-runs"
                  />
                  <FeatureItem 
                    name="High Frequency Power" 
                    value={formatNumber(props.prediction.features!.spectral.high_freq_power, 0)} 
                    description="Rapid back-and-forth scoring"
                  />
                  <FeatureItem 
                    name="Dominant Frequency" 
                    value={formatNumber(props.prediction.features!.spectral.dominant_freq, 0)} 
                    description="Most prominent frequency component"
                  />
                </div>
                <div class="mt-4 p-3 bg-blue-500/10 border border-blue-500/30 rounded text-sm text-gray-300">
                  💡 <strong>What this means:</strong> High low_freq_power = steady dominance. High high_freq_power = competitive back-and-forth game.
                </div>
              </div>

              {/* CATEGORY 3: Autocorrelation (3 features) */}
              <div class="bg-gray-800 rounded-lg p-6">
                <h3 class="text-lg font-semibold text-green-400 mb-4">
                  🔁 Autocorrelation (3 Features)
                </h3>
                <div class="grid grid-cols-3 gap-4">
                  <FeatureItem 
                    name="Lag-1 Correlation" 
                    value={formatNumber(props.prediction.features!.autocorrelation.lag1, 3)} 
                    description="Correlation with 1-minute lag"
                  />
                  <FeatureItem 
                    name="Lag-2 Correlation" 
                    value={formatNumber(props.prediction.features!.autocorrelation.lag2, 3)} 
                    description="Correlation with 2-minute lag"
                  />
                  <FeatureItem 
                    name="Lag-3 Correlation" 
                    value={formatNumber(props.prediction.features!.autocorrelation.lag3, 3)} 
                    description="Correlation with 3-minute lag"
                  />
                </div>
                <div class="mt-4 p-3 bg-blue-500/10 border border-blue-500/30 rounded text-sm text-gray-300">
                  💡 <strong>What this means:</strong> High values (close to 1) = pattern is predictable and continues. Low values = random/chaotic game.
                </div>
              </div>

              {/* CATEGORY 4: Advanced Stats (8 features) */}
              <div class="bg-gray-800 rounded-lg p-6">
                <h3 class="text-lg font-semibold text-yellow-400 mb-4">
                  ⚡ Advanced Stats (8 Features)
                </h3>
                <div class="grid grid-cols-2 gap-4">
                  <FeatureItem 
                    name="Pace Proxy" 
                    value={formatNumber(props.prediction.features!.advanced_stats.pace_proxy)} 
                    description="Actions per minute (game speed)"
                  />
                  <FeatureItem 
                    name="Effective FG%" 
                    value={formatPercent(props.prediction.features!.advanced_stats.efg_proxy)} 
                    description="Shooting efficiency (proxy)"
                  />
                  <FeatureItem 
                    name="True Shooting%" 
                    value={formatPercent(props.prediction.features!.advanced_stats.ts_proxy)} 
                    description="Overall scoring efficiency"
                  />
                  <FeatureItem 
                    name="Net Rating Proxy" 
                    value={formatNumber(props.prediction.features!.advanced_stats.netrtg_proxy)} 
                    description="Point differential per minute"
                  />
                  <FeatureItem 
                    name="Usage Proxy" 
                    value={formatPercent(props.prediction.features!.advanced_stats.usg_proxy)} 
                    description="Possession usage rate"
                  />
                  <FeatureItem 
                    name="Plus/Minus Proxy" 
                    value={formatNumber(props.prediction.features!.advanced_stats.pm_proxy)} 
                    description="Current differential"
                  />
                  <FeatureItem 
                    name="PIE Proxy" 
                    value={formatPercent(props.prediction.features!.advanced_stats.pie_proxy)} 
                    description="Player Impact Estimate"
                  />
                  <FeatureItem 
                    name="Four Factors" 
                    value={formatPercent(props.prediction.features!.advanced_stats.four_factors)} 
                    description="Combined efficiency metric"
                  />
                </div>
              </div>

              {/* CATEGORY 5: Team Form (6 features) */}
              <div class="bg-gray-800 rounded-lg p-6">
                <h3 class="text-lg font-semibold text-orange-400 mb-4">
                  📅 Team Form (6 Features)
                </h3>
                <div class="grid grid-cols-2 gap-4">
                  <FeatureItem 
                    name="Last Game Differential" 
                    value={formatNumber(props.prediction.features!.team_form.team_diff_lag1)} 
                    description="Point differential in most recent game"
                  />
                  <FeatureItem 
                    name="Last Game Average" 
                    value={formatNumber(props.prediction.features!.team_form.team_mean_lag1)} 
                    description="Average score in last game"
                  />
                  <FeatureItem 
                    name="3-Game Rolling Diff" 
                    value={formatNumber(props.prediction.features!.team_form.team_diff_rolling3)} 
                    description="Average differential over last 3 games"
                  />
                  <FeatureItem 
                    name="3-Game Volatility" 
                    value={formatNumber(props.prediction.features!.team_form.team_volatility_rolling3)} 
                    description="Consistency of performance"
                  />
                  <FeatureItem 
                    name="10-Game Form" 
                    value={formatNumber(props.prediction.features!.team_form.team_form_10games)} 
                    description="Season trend (last 10 games)"
                  />
                  <FeatureItem 
                    name="Team Consistency" 
                    value={formatNumber(props.prediction.features!.team_form.team_consistency, 3)} 
                    description="Reliability metric (higher = more consistent)"
                  />
                </div>
              </div>

              {/* Extraction Metadata */}
              <div class="bg-gray-800 rounded-lg p-4">
                <h3 class="text-sm font-semibold text-gray-400 mb-2">Extraction Metadata</h3>
                <div class="grid grid-cols-3 gap-4 text-sm">
                  <div>
                    <span class="text-gray-500">Extraction Time:</span>
                    <span class="text-white ml-2 font-semibold">{props.prediction.features!.extraction_time_ms}ms</span>
                  </div>
                  <div>
                    <span class="text-gray-500">PBP Events:</span>
                    <span class="text-white ml-2 font-semibold">{props.prediction.features!.pbp_events_count} events</span>
                  </div>
                  <div>
                    <span class="text-gray-500">Pattern Length:</span>
                    <span class="text-white ml-2 font-semibold">{props.prediction.features!.pattern_length} points</span>
                  </div>
                </div>
              </div>
            </Show>

            {/* Close Button */}
            <div class="flex justify-end pt-4">
              <button 
                onClick={props.onClose}
                class="px-6 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      </div>
    </Show>
  );
};

// Helper component for individual feature items
const FeatureItem: Component<{
  name: string;
  value: string;
  description: string;
  highlight?: boolean;
}> = (props) => {
  return (
    <div class={`p-3 rounded ${props.highlight ? 'bg-yellow-500/10 border border-yellow-500/30' : 'bg-gray-700/50'}`}>
      <div class="flex justify-between items-start mb-1">
        <span class="text-sm text-gray-400">{props.name}</span>
        <span class={`text-lg font-bold ${props.highlight ? 'text-yellow-400' : 'text-white'}`}>
          {props.value}
        </span>
      </div>
      <p class="text-xs text-gray-500">{props.description}</p>
    </div>
  );
};

export default FeatureDetailsModal;

