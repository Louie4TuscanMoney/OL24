/**
 * Game Card Component
 * Displays single NBA game with ML prediction and betting edge
 */

import { Component, Show } from 'solid-js';
import type { NBAGame, Prediction, Edge, BettingRecommendation } from '../types';

interface Props {
  game: NBAGame;
  prediction?: Prediction;
  edge?: Edge;
  recommendation?: BettingRecommendation;
}

const GameCard: Component<Props> = (props) => {
  const diff = () => props.game.score_home - props.game.score_away;
  
  const diffColor = () => {
    const d = diff();
    if (d > 5) return 'text-green-400';
    if (d < -5) return 'text-red-400';
    return 'text-gray-300';
  };

  const safetyModeColor = () => {
    if (!props.recommendation) return 'gray';
    switch (props.recommendation.safety_mode) {
      case 'GREEN': return 'green';
      case 'YELLOW': return 'yellow';
      case 'RED': return 'red';
      default: return 'gray';
    }
  };

  return (
    <div class="bg-gray-900 rounded-lg p-6 border border-gray-800 hover:border-gray-700 transition-colors">
      {/* Header */}
      <div class="flex justify-between items-center text-sm text-gray-400 mb-4">
        <div class="flex items-center gap-2">
          <Show when={props.game.is_live}>
            <span class="flex items-center gap-1">
              <span class="w-2 h-2 bg-red-500 rounded-full live-pulse"></span>
              <span class="text-red-500 font-semibold">LIVE</span>
            </span>
          </Show>
          <span>Q{props.game.quarter} • {props.game.time_remaining}</span>
        </div>
        <span class="text-xs">{props.game.game_id}</span>
      </div>

      {/* Score Display */}
      <div class="flex justify-between items-center mb-6">
        {/* Away Team */}
        <div class="text-left flex-1">
          <div class="text-gray-400 text-sm mb-1">{props.game.away_team}</div>
          <div class="text-4xl font-bold">{props.game.score_away}</div>
        </div>

        {/* Differential */}
        <div class="text-center px-6">
          <div class="text-xs text-gray-500 mb-1">DIFF</div>
          <div class={`text-3xl font-bold ${diffColor()} score-update`}>
            {diff() > 0 ? '+' : ''}{diff()}
          </div>
        </div>

        {/* Home Team */}
        <div class="text-right flex-1">
          <div class="text-gray-400 text-sm mb-1">{props.game.home_team}</div>
          <div class="text-4xl font-bold">{props.game.score_home}</div>
        </div>
      </div>

      {/* ML Prediction */}
      <Show when={props.prediction}>
        <div class="border-t border-gray-800 pt-4 mb-4">
          <div class="text-xs text-gray-500 mb-2">ML PREDICTION (Halftime)</div>
          <div class="flex items-center justify-between">
            <div>
              <div class="text-2xl font-bold text-blue-400">
                {props.prediction!.point_forecast > 0 ? '+' : ''}
                {props.prediction!.point_forecast.toFixed(1)}
              </div>
              <div class="text-xs text-gray-500">
                95% CI: [{props.prediction!.interval_lower.toFixed(1)}, {props.prediction!.interval_upper.toFixed(1)}]
              </div>
            </div>
            <div class="text-right">
              <div class="text-xs text-gray-500">MAE: 5.39</div>
              <div class="text-xs text-gray-500">Coverage: 94.6%</div>
            </div>
          </div>
        </div>
      </Show>

      {/* Edge Detection */}
      <Show when={props.edge && props.edge.has_edge}>
        <div class="border-t border-gray-800 pt-4 mb-4">
          <div class="flex items-center justify-between">
            <div>
              <div class="text-xs text-gray-500 mb-1">🎯 EDGE DETECTED</div>
              <div class="text-lg font-semibold text-amber-400">
                {props.edge!.edge_size.toFixed(1)} points
              </div>
              <div class="text-xs text-gray-400 capitalize">
                {props.edge!.direction} • {props.edge!.confidence} confidence
              </div>
            </div>
            <div class="text-right">
              <div class="text-xs text-gray-500">ML: {props.edge!.ml_forecast.toFixed(1)}</div>
              <div class="text-xs text-gray-500">Market: {props.edge!.market_spread.toFixed(1)}</div>
            </div>
          </div>
        </div>
      </Show>

      {/* Betting Recommendation */}
      <Show when={props.recommendation}>
        <div class="border-t border-gray-800 pt-4">
          <div class="flex items-center justify-between mb-2">
            <div class="text-xs text-gray-500">BET RECOMMENDATION</div>
            <div class={`text-xs px-2 py-1 rounded bg-${safetyModeColor()}-500/20 text-${safetyModeColor()}-400`}>
              {props.recommendation!.safety_mode}
            </div>
          </div>
          <div class="flex items-center justify-between">
            <div>
              <div class="text-2xl font-bold text-green-400">
                ${props.recommendation!.final_bet.toFixed(0)}
              </div>
              <div class="text-xs text-gray-500">
                EV: ${props.recommendation!.expected_value > 0 ? '+' : ''}
                {props.recommendation!.expected_value.toFixed(0)}
              </div>
            </div>
            <div class="text-right text-xs text-gray-500">
              <div>Kelly: ${props.recommendation!.kelly_bet.toFixed(0)}</div>
              <div>Final: ${props.recommendation!.final_bet.toFixed(0)}</div>
            </div>
          </div>
        </div>
      </Show>
    </div>
  );
};

export default GameCard;

