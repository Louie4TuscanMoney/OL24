/**
 * Expanded Game Card - Shows ALL data layers
 * NBA scores + 18-min pattern + ML prediction + BetOnline odds + Risk layers
 */

import { Component, Show } from 'solid-js';
import type { NBAGame, Prediction, Edge, BettingRecommendation, ScorePattern } from '../types';
import PredictionChart from './PredictionChart';
import RiskLayers from './RiskLayers';

interface Props {
  game: NBAGame;
  pattern?: ScorePattern[];
  prediction?: Prediction;
  edge?: Edge;
  recommendation?: BettingRecommendation;
}

const GameCardExpanded: Component<Props> = (props) => {
  const diff = () => props.game.score_home - props.game.score_away;
  
  const diffColor = () => {
    const d = diff();
    if (d > 5) return 'text-green-400';
    if (d < -5) return 'text-red-400';
    return 'text-gray-300';
  };

  return (
    <div class="bg-gray-900 rounded-lg border border-gray-800 overflow-hidden">
      {/* Header */}
      <div class="bg-gray-800/50 px-6 py-4 flex justify-between items-center">
        <div class="flex items-center gap-3">
          <Show when={props.game.is_live}>
            <span class="flex items-center gap-1.5">
              <span class="w-2.5 h-2.5 bg-red-500 rounded-full live-pulse"></span>
              <span class="text-red-500 font-semibold text-sm">LIVE</span>
            </span>
          </Show>
          <span class="text-gray-400 text-sm">Q{props.game.quarter} • {props.game.time_remaining}</span>
        </div>
        <span class="text-xs text-gray-500">{props.game.game_id}</span>
      </div>

      <div class="p-6">
        {/* Score Display */}
        <div class="flex justify-between items-center mb-6">
          {/* Away Team */}
          <div class="text-left flex-1">
            <div class="text-gray-400 text-sm mb-1">{props.game.away_team}</div>
            <div class="text-5xl font-bold">{props.game.score_away}</div>
          </div>

          {/* Differential */}
          <div class="text-center px-8">
            <div class="text-xs text-gray-500 mb-1">DIFFERENTIAL</div>
            <div class={`text-4xl font-bold ${diffColor()} score-update`}>
              {diff() > 0 ? '+' : ''}{diff()}
            </div>
          </div>

          {/* Home Team */}
          <div class="text-right flex-1">
            <div class="text-gray-400 text-sm mb-1">{props.game.home_team}</div>
            <div class="text-5xl font-bold">{props.game.score_home}</div>
          </div>
        </div>

        {/* 18-Minute Pattern + Prediction Chart */}
        <Show when={props.pattern && props.pattern.length > 0}>
          <div class="mb-6">
            <PredictionChart 
              pattern={props.pattern!}
              prediction={props.prediction}
            />
          </div>
        </Show>

        {/* ML Prediction Details */}
        <Show when={props.prediction}>
          <div class="bg-blue-500/10 border border-blue-500/30 rounded-lg p-4 mb-4">
            <div class="flex justify-between items-start mb-2">
              <div>
                <div class="text-xs text-gray-400 mb-1">ML ENSEMBLE PREDICTION</div>
                <div class="text-3xl font-bold text-blue-400">
                  {props.prediction!.point_forecast > 0 ? '+' : ''}
                  {props.prediction!.point_forecast.toFixed(1)} pts
                </div>
                <div class="text-xs text-gray-500 mt-1">
                  {props.prediction!.point_forecast > 0 ? props.game.home_team : props.game.away_team} leads at halftime
                </div>
              </div>
              <div class="text-right">
                <div class="text-xs text-gray-500">Dejavu (6.17) + LSTM (5.24)</div>
                <div class="text-xs text-gray-500">Ensemble MAE: 5.39</div>
                <div class="text-xs text-gray-500">Conformal: 94.6% coverage</div>
              </div>
            </div>

            {/* Confidence Interval */}
            <div class="mt-3 pt-3 border-t border-blue-500/30">
              <div class="text-xs text-gray-400 mb-2">95% Confidence Interval</div>
              <div class="flex items-center justify-between text-sm">
                <span class="text-gray-300">
                  {props.prediction!.interval_lower.toFixed(1)}
                </span>
                <div class="flex-1 mx-3 h-2 bg-gray-700 rounded-full relative">
                  <div class="absolute inset-0 bg-gradient-to-r from-blue-600 to-blue-400 rounded-full opacity-50"></div>
                </div>
                <span class="text-gray-300">
                  {props.prediction!.interval_upper.toFixed(1)}
                </span>
              </div>
            </div>
          </div>
        </Show>

        {/* Edge Detection */}
        <Show when={props.edge && props.edge.has_edge}>
          <div class="bg-amber-500/10 border border-amber-500/30 rounded-lg p-4 mb-4">
            <div class="flex justify-between items-start">
              <div>
                <div class="text-xs text-gray-400 mb-1">🎯 BETTING EDGE DETECTED</div>
                <div class="text-3xl font-bold text-amber-400">
                  {props.edge!.edge_size.toFixed(1)} pts
                </div>
                <div class="text-sm text-gray-300 mt-1 capitalize">
                  {props.edge!.direction} • {props.edge!.confidence} confidence
                </div>
              </div>
              <div class="text-right">
                <div class="text-xs text-gray-500">ML Forecast</div>
                <div class="text-sm font-semibold">{props.edge!.ml_forecast.toFixed(1)}</div>
                <div class="text-xs text-gray-500 mt-2">Market Spread</div>
                <div class="text-sm font-semibold">{props.edge!.market_spread.toFixed(1)}</div>
              </div>
            </div>
          </div>
        </Show>

        {/* Risk Management Recommendation */}
        <Show when={props.recommendation}>
          <RiskLayers recommendation={props.recommendation!} />
        </Show>
      </div>
    </div>
  );
};

export default GameCardExpanded;

