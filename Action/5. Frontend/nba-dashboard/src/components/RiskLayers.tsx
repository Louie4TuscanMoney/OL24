/**
 * Risk Layers Component
 * Shows bet progression through all 5 risk layers
 */

import { Component } from 'solid-js';
import type { BettingRecommendation } from '../types';

interface Props {
  recommendation: BettingRecommendation;
}

const RiskLayers: Component<Props> = (props) => {
  const layers = [
    { name: 'Kelly', value: props.recommendation.kelly_bet, color: 'blue' },
    { name: 'Delta', value: props.recommendation.delta_bet, color: 'purple' },
    { name: 'Portfolio', value: props.recommendation.portfolio_bet, color: 'indigo' },
    { name: 'Decision', value: props.recommendation.decision_tree_bet, color: 'pink' },
    { name: 'Final', value: props.recommendation.final_bet, color: 'green' }
  ];

  const getBarWidth = (value: number) => {
    const max = Math.max(...layers.map(l => l.value));
    return (value / max) * 100;
  };

  return (
    <div class="bg-gray-800/30 rounded-lg p-4">
      <div class="text-xs text-gray-400 mb-3 flex justify-between">
        <span>5-Layer Risk System</span>
        <span class={`px-2 py-1 rounded text-${props.recommendation.safety_mode === 'GREEN' ? 'green' : props.recommendation.safety_mode === 'YELLOW' ? 'yellow' : 'red'}-400 bg-${props.recommendation.safety_mode === 'GREEN' ? 'green' : props.recommendation.safety_mode === 'YELLOW' ? 'yellow' : 'red'}-500/20`}>
          {props.recommendation.safety_mode}
        </span>
      </div>

      {/* Layer progression bars */}
      <div class="space-y-2">
        {layers.map((layer, i) => (
          <div>
            <div class="flex justify-between text-xs mb-1">
              <span class="text-gray-400">{layer.name}</span>
              <span class="font-mono">${layer.value.toFixed(0)}</span>
            </div>
            <div class="h-2 bg-gray-700 rounded-full overflow-hidden">
              <div
                class={`h-full bg-${layer.color}-500 transition-all duration-500`}
                style={{ width: `${getBarWidth(layer.value)}%` }}
              ></div>
            </div>
          </div>
        ))}
      </div>

      {/* Final bet callout */}
      <div class="mt-4 p-3 bg-green-500/10 border border-green-500/30 rounded">
        <div class="text-xs text-gray-400 mb-1">FINAL BET (After all layers)</div>
        <div class="text-2xl font-bold text-green-400">
          ${props.recommendation.final_bet.toFixed(0)}
        </div>
        <div class="text-xs text-gray-500 mt-1">
          EV: ${props.recommendation.expected_value > 0 ? '+' : ''}
          {props.recommendation.expected_value.toFixed(0)}
        </div>
      </div>

      {/* Reasoning */}
      <div class="mt-3 text-xs text-gray-500">
        {props.recommendation.reasoning.slice(0, 2).map(reason => (
          <div class="truncate">• {reason}</div>
        ))}
      </div>
    </div>
  );
};

export default RiskLayers;

