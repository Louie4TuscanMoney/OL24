/**
 * Prediction Chart Component
 * Visualizes 18-minute score differential pattern + ML prediction
 */

import { Component, For } from 'solid-js';
import type { ScorePattern, Prediction } from '../types';

interface Props {
  pattern: ScorePattern[];
  prediction?: Prediction;
}

const PredictionChart: Component<Props> = (props) => {
  const maxDiff = () => Math.max(...props.pattern.map(p => Math.abs(p.differential)), 10);
  
  const getY = (diff: number) => {
    // Scale to chart height (200px)
    const scale = 100 / maxDiff();
    return 100 - (diff * scale);
  };

  return (
    <div class="bg-gray-800/50 rounded-lg p-4">
      <div class="text-xs text-gray-400 mb-3 flex justify-between">
        <span>18-Minute Pattern</span>
        <span>{props.pattern.length} minutes</span>
      </div>
      
      {/* Chart */}
      <div class="relative h-48 bg-gray-900 rounded">
        <svg viewBox="0 0 180 200" class="w-full h-full">
          {/* Zero line */}
          <line
            x1="0"
            y1="100"
            x2="180"
            y2="100"
            stroke="rgb(75, 85, 99)"
            stroke-width="1"
            stroke-dasharray="4"
          />
          
          {/* Pattern line */}
          <polyline
            points={props.pattern.map((p, i) => 
              `${i * 10},${getY(p.differential)}`
            ).join(' ')}
            fill="none"
            stroke="rgb(59, 130, 246)"
            stroke-width="2"
          />
          
          {/* Pattern points */}
          <For each={props.pattern}>
            {(point, i) => (
              <circle
                cx={i() * 10}
                cy={getY(point.differential)}
                r="3"
                fill="rgb(59, 130, 246)"
              />
            )}
          </For>
          
          {/* Prediction point (at minute 24) */}
          {props.prediction && (
            <>
              <circle
                cx="180"
                cy={getY(props.prediction.point_forecast)}
                r="5"
                fill="rgb(34, 197, 94)"
                stroke="rgb(22, 163, 74)"
                stroke-width="2"
              />
              
              {/* Confidence interval */}
              <line
                x1="180"
                y1={getY(props.prediction.interval_lower)}
                x2="180"
                y2={getY(props.prediction.interval_upper)}
                stroke="rgb(34, 197, 94)"
                stroke-width="3"
                opacity="0.5"
              />
            </>
          )}
        </svg>
      </div>
      
      {/* Legend */}
      <div class="mt-3 flex justify-between text-xs">
        <div class="flex items-center gap-2">
          <span class="w-3 h-3 bg-blue-500 rounded"></span>
          <span class="text-gray-400">Historical (18 min)</span>
        </div>
        <div class="flex items-center gap-2">
          <span class="w-3 h-3 bg-green-500 rounded"></span>
          <span class="text-gray-400">ML Prediction (halftime)</span>
        </div>
      </div>
    </div>
  );
};

export default PredictionChart;

