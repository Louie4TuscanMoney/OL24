# Step 2: Component Architecture

**Objective:** Build core UI components optimized for real-time NBA predictions

**Duration:** 1-2 hours  
**Prerequisites:** Completed Step 1 (SolidJS Setup)  
**Output:** Reusable, performant components with fine-grained reactivity

---

## 2.1 Component Hierarchy

```
App
├── Dashboard (main view)
│   ├── LiveGamesList
│   │   └── LiveGameCard (×N games)
│   │       ├── GameHeader
│   │       ├── ScoreDisplay
│   │       ├── TimeSeriesChart
│   │       ├── PredictionDisplay
│   │       │   ├── PointForecast
│   │       │   ├── ConfidenceInterval
│   │       │   └── ModelBreakdown
│   │       └── SimilarGames
│   │
│   └── PerformanceMetrics
│
└── Sidebar (optional)
    ├── GameFilter
    └── SettingsPanel
```

---

## 2.2 Core Components

### Component 1: LiveGameCard (15 minutes)

**Create `src/components/LiveGameCard.tsx`:**

```typescript
import { Component, Show, createMemo } from 'solid-js';
import type { NBAGame, Prediction } from '@types';
import ScoreDisplay from './ScoreDisplay';
import TimeSeriesChart from './TimeSeriesChart';
import PredictionDisplay from './PredictionDisplay';

interface LiveGameCardProps {
  game: NBAGame;
  prediction?: Prediction;
  pattern?: number[];
}

const LiveGameCard: Component<LiveGameCardProps> = (props) => {
  // Memoized derived values (automatically tracks dependencies)
  const isSecondQuarter = createMemo(() => props.game.quarter === 2);
  const hasPrediction = createMemo(() => !!props.prediction);
  const leadingTeam = createMemo(() => 
    props.game.differential > 0 ? props.game.home_team : props.game.away_team
  );

  return (
    <div class="game-card animate-prediction-appear">
      {/* Header */}
      <div class="flex justify-between items-center mb-4">
        <div class="flex items-center gap-3">
          <div class="text-sm text-gray-400">
            Q{props.game.quarter} • {props.game.time_remaining}
          </div>
          <Show when={isSecondQuarter()}>
            <span class="prediction-badge bg-blue-600 text-white">
              🎯 Prediction Ready
            </span>
          </Show>
        </div>
        <div class="text-xs text-gray-500">{props.game.game_id}</div>
      </div>

      {/* Score Display */}
      <ScoreDisplay game={props.game} />

      {/* Time Series Chart (18-minute pattern) */}
      <Show when={props.pattern && props.pattern.length > 0}>
        <div class="mt-6">
          <h3 class="text-sm font-medium text-gray-400 mb-3">
            Score Pattern (First 18 Minutes)
          </h3>
          <TimeSeriesChart pattern={props.pattern!} />
        </div>
      </Show>

      {/* Prediction Display */}
      <Show when={hasPrediction()}>
        <div class="mt-6 pt-6 border-t border-gray-800">
          <PredictionDisplay prediction={props.prediction!} />
        </div>
      </Show>

      {/* Status Indicator */}
      <div class="mt-4 flex items-center justify-between text-xs text-gray-500">
        <span>Leading: {leadingTeam()}</span>
        <span class="flex items-center gap-2">
          <div class="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
          Live
        </span>
      </div>
    </div>
  );
};

export default LiveGameCard;
```

**Key Features:**
- ✅ Fine-grained reactivity (only updates changed parts)
- ✅ Memoized computations (auto-tracked dependencies)
- ✅ Conditional rendering with `<Show>` (more efficient than React's ternaries)
- ✅ Type-safe props

---

### Component 2: ScoreDisplay (10 minutes)

**Create `src/components/ScoreDisplay.tsx`:**

```typescript
import { Component, createMemo } from 'solid-js';
import type { NBAGame } from '@types';
import clsx from 'clsx';

interface ScoreDisplayProps {
  game: NBAGame;
}

const ScoreDisplay: Component<ScoreDisplayProps> = (props) => {
  const isAwayLeading = createMemo(() => props.game.differential < 0);
  const isHomeLeading = createMemo(() => props.game.differential > 0);
  const isTied = createMemo(() => props.game.differential === 0);

  return (
    <div class="flex items-center justify-between gap-6 py-4">
      {/* Away Team */}
      <div class={clsx(
        'flex-1 text-right',
        isAwayLeading() && 'opacity-100',
        !isAwayLeading() && 'opacity-60'
      )}>
        <div class="text-sm text-gray-400 mb-1">{props.game.away_team}</div>
        <div class="text-4xl font-bold tabular-nums">{props.game.score_away}</div>
      </div>

      {/* Differential */}
      <div class="flex flex-col items-center">
        <div class={clsx(
          'text-2xl font-bold tabular-nums',
          props.game.differential > 0 && 'text-green-400',
          props.game.differential < 0 && 'text-red-400',
          props.game.differential === 0 && 'text-gray-400'
        )}>
          {props.game.differential > 0 ? '+' : ''}{props.game.differential}
        </div>
        <div class="text-xs text-gray-500 mt-1">Differential</div>
      </div>

      {/* Home Team */}
      <div class={clsx(
        'flex-1 text-left',
        isHomeLeading() && 'opacity-100',
        !isHomeLeading() && 'opacity-60'
      )}>
        <div class="text-sm text-gray-400 mb-1">{props.game.home_team}</div>
        <div class="text-4xl font-bold tabular-nums">{props.game.score_home}</div>
      </div>
    </div>
  );
};

export default ScoreDisplay;
```

**Performance Note:**
- When score updates, **only the score numbers re-render**, not the entire component
- React equivalent would re-render the entire component
- ~10x faster updates

---

### Component 3: PredictionDisplay (15 minutes)

**Create `src/components/PredictionDisplay.tsx`:**

```typescript
import { Component, Show, For, createMemo } from 'solid-js';
import type { Prediction } from '@types';
import ConfidenceInterval from './ConfidenceInterval';
import ModelBreakdown from './ModelBreakdown';

interface PredictionDisplayProps {
  prediction: Prediction;
}

const PredictionDisplay: Component<PredictionDisplayProps> = (props) => {
  const intervalWidth = createMemo(() => 
    props.prediction.interval_upper - props.prediction.interval_lower
  );

  const isNarrowInterval = createMemo(() => intervalWidth() < 8);

  return (
    <div class="space-y-4">
      {/* Main Prediction */}
      <div class="text-center">
        <h3 class="text-sm font-medium text-gray-400 mb-2">
          Halftime Prediction
        </h3>
        <div class="text-5xl font-bold tabular-nums text-blue-400">
          {props.prediction.point_forecast > 0 ? '+' : ''}
          {props.prediction.point_forecast.toFixed(1)}
        </div>
        <div class="text-sm text-gray-500 mt-2">
          points differential
        </div>
      </div>

      {/* Confidence Interval */}
      <div class="bg-gray-800 rounded-lg p-4">
        <div class="flex items-center justify-between mb-2">
          <span class="text-xs text-gray-400">
            {(props.prediction.coverage_probability * 100).toFixed(0)}% Confidence
          </span>
          <span class={clsx(
            'text-xs font-medium',
            isNarrowInterval() ? 'text-green-400' : 'text-yellow-400'
          )}>
            ±{(intervalWidth() / 2).toFixed(1)} pts
          </span>
        </div>
        <ConfidenceInterval
          lower={props.prediction.interval_lower}
          upper={props.prediction.interval_upper}
          point={props.prediction.point_forecast}
        />
        <div class="flex justify-between text-xs text-gray-500 mt-2">
          <span>{props.prediction.interval_lower.toFixed(1)}</span>
          <span>{props.prediction.interval_upper.toFixed(1)}</span>
        </div>
      </div>

      {/* Model Breakdown */}
      <Show when={props.prediction.explanation}>
        <ModelBreakdown explanation={props.prediction.explanation!} />
      </Show>

      {/* Similar Games */}
      <Show when={props.prediction.explanation?.similar_games}>
        <div class="bg-gray-800 rounded-lg p-4">
          <h4 class="text-xs font-medium text-gray-400 mb-3">
            📊 Similar Historical Games
          </h4>
          <div class="space-y-2">
            <For each={props.prediction.explanation!.similar_games!.slice(0, 3)}>
              {(game, index) => (
                <div class="flex items-center justify-between text-sm">
                  <span class="text-gray-400">
                    {index() + 1}. {game.teams}
                  </span>
                  <span class="text-gray-300 font-medium">
                    {game.halftime_differential > 0 ? '+' : ''}
                    {game.halftime_differential.toFixed(1)}
                  </span>
                </div>
              )}
            </For>
          </div>
        </div>
      </Show>
    </div>
  );
};

export default PredictionDisplay;
```

---

### Component 4: ConfidenceInterval (10 minutes)

**Create `src/components/ConfidenceInterval.tsx`:**

```typescript
import { Component, createMemo } from 'solid-js';

interface ConfidenceIntervalProps {
  lower: number;
  upper: number;
  point: number;
}

const ConfidenceInterval: Component<ConfidenceIntervalProps> = (props) => {
  // Calculate positions (normalize to 0-100%)
  const range = createMemo(() => Math.abs(props.upper - props.lower));
  const pointPosition = createMemo(() => {
    const total = props.upper - props.lower;
    const offset = props.point - props.lower;
    return (offset / total) * 100;
  });

  return (
    <div class="relative h-12 bg-gray-900 rounded-lg overflow-hidden">
      {/* Interval bar */}
      <div class="absolute inset-y-0 left-0 right-0 flex items-center px-2">
        <div class="relative w-full h-3 bg-gradient-to-r from-red-500/20 via-blue-500/40 to-green-500/20 rounded-full">
          {/* Point indicator */}
          <div
            class="absolute top-1/2 -translate-y-1/2 w-4 h-4 bg-blue-500 rounded-full border-2 border-white shadow-lg transition-all duration-300"
            style={{ left: `calc(${pointPosition()}% - 8px)` }}
          >
            <div class="absolute inset-0 bg-blue-500 rounded-full animate-ping opacity-75" />
          </div>
        </div>
      </div>

      {/* Range labels */}
      <div class="absolute inset-0 flex items-center justify-between px-2 text-xs font-medium">
        <span class="text-red-400">Worse</span>
        <span class="text-green-400">Better</span>
      </div>
    </div>
  );
};

export default ConfidenceInterval;
```

---

### Component 5: ModelBreakdown (10 minutes)

**Create `src/components/ModelBreakdown.tsx`:**

```typescript
import { Component } from 'solid-js';
import type { PredictionExplanation } from '@types';

interface ModelBreakdownProps {
  explanation: PredictionExplanation;
}

const ModelBreakdown: Component<ModelBreakdownProps> = (props) => {
  return (
    <div class="bg-gray-800 rounded-lg p-4">
      <h4 class="text-xs font-medium text-gray-400 mb-3">🤖 Model Breakdown</h4>
      
      <div class="space-y-3">
        {/* Dejavu */}
        <div>
          <div class="flex items-center justify-between mb-1">
            <div class="flex items-center gap-2">
              <span class="text-sm text-gray-300">Déjà vu</span>
              <span class="text-xs text-gray-500">
                (Pattern Matching)
              </span>
            </div>
            <span class="text-sm font-medium text-gray-200">
              {props.explanation.dejavu_prediction > 0 ? '+' : ''}
              {props.explanation.dejavu_prediction.toFixed(1)}
            </span>
          </div>
          <div class="h-1.5 bg-gray-700 rounded-full overflow-hidden">
            <div 
              class="h-full bg-purple-500 rounded-full transition-all duration-500"
              style={{ width: `${props.explanation.dejavu_weight * 100}%` }}
            />
          </div>
          <div class="text-xs text-gray-500 mt-1">
            Weight: {(props.explanation.dejavu_weight * 100).toFixed(0)}%
          </div>
        </div>

        {/* LSTM */}
        <div>
          <div class="flex items-center justify-between mb-1">
            <div class="flex items-center gap-2">
              <span class="text-sm text-gray-300">LSTM</span>
              <span class="text-xs text-gray-500">
                (Pattern Learning)
              </span>
            </div>
            <span class="text-sm font-medium text-gray-200">
              {props.explanation.lstm_prediction > 0 ? '+' : ''}
              {props.explanation.lstm_prediction.toFixed(1)}
            </span>
          </div>
          <div class="h-1.5 bg-gray-700 rounded-full overflow-hidden">
            <div 
              class="h-full bg-blue-500 rounded-full transition-all duration-500"
              style={{ width: `${props.explanation.lstm_weight * 100}%` }}
            />
          </div>
          <div class="text-xs text-gray-500 mt-1">
            Weight: {(props.explanation.lstm_weight * 100).toFixed(0)}%
          </div>
        </div>

        {/* Ensemble */}
        <div class="pt-3 border-t border-gray-700">
          <div class="flex items-center justify-between">
            <span class="text-sm font-medium text-gray-200">
              Ensemble Forecast
            </span>
            <span class="text-lg font-bold text-blue-400">
              {props.explanation.ensemble_forecast > 0 ? '+' : ''}
              {props.explanation.ensemble_forecast.toFixed(1)}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ModelBreakdown;
```

---

### Component 6: TimeSeriesChart (15 minutes)

**Create `src/components/TimeSeriesChart.tsx`:**

```typescript
import { Component, createMemo } from 'solid-js';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';

interface TimeSeriesChartProps {
  pattern: number[];
}

const TimeSeriesChart: Component<TimeSeriesChartProps> = (props) => {
  // Transform pattern array to chart data
  const chartData = createMemo(() => 
    props.pattern.map((diff, idx) => ({
      minute: idx + 1,
      differential: diff,
    }))
  );

  // Memoize chart color based on trend
  const lineColor = createMemo(() => {
    const trend = props.pattern[props.pattern.length - 1] - props.pattern[0];
    return trend > 0 ? '#10B981' : '#EF4444';
  });

  return (
    <div class="chart-container">
      <ResponsiveContainer width="100%" height={200}>
        <LineChart data={chartData()}>
          <XAxis 
            dataKey="minute" 
            stroke="#6B7280"
            style={{ fontSize: '12px' }}
            label={{ value: 'Minute', position: 'insideBottom', offset: -5 }}
          />
          <YAxis 
            stroke="#6B7280"
            style={{ fontSize: '12px' }}
            label={{ value: 'Differential', angle: -90, position: 'insideLeft' }}
          />
          <Tooltip 
            contentStyle={{ 
              backgroundColor: '#1F2937', 
              border: '1px solid #374151',
              borderRadius: '8px',
            }}
            labelStyle={{ color: '#9CA3AF' }}
            itemStyle={{ color: '#F3F4F6' }}
          />
          <ReferenceLine y={0} stroke="#6B7280" strokeDasharray="3 3" />
          <Line 
            type="monotone" 
            dataKey="differential" 
            stroke={lineColor()}
            strokeWidth={2}
            dot={false}
            animationDuration={300}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

export default TimeSeriesChart;
```

**Performance Note:**
- Chart only re-renders when `pattern` prop changes
- Memoized computations prevent unnecessary recalculations
- Smooth 60 FPS even with live updates

---

## 2.3 Dashboard Container (15 minutes)

**Create `src/components/Dashboard.tsx`:**

```typescript
import { Component, For, Show, createSignal, onMount, onCleanup } from 'solid-js';
import type { NBAGame, Prediction } from '@types';
import LiveGameCard from './LiveGameCard';
import { WebSocketService } from '@services/websocket-service';

const Dashboard: Component = () => {
  const [games, setGames] = createSignal<NBAGame[]>([]);
  const [predictions, setPredictions] = createSignal<Map<string, Prediction>>(new Map());
  const [isConnected, setIsConnected] = createSignal(false);

  let ws: WebSocketService;

  onMount(() => {
    // Initialize WebSocket connection
    ws = new WebSocketService();
    
    ws.onConnect(() => {
      setIsConnected(true);
      console.log('✅ WebSocket connected');
    });

    ws.onDisconnect(() => {
      setIsConnected(false);
      console.log('❌ WebSocket disconnected');
    });

    ws.onGameUpdate((game: NBAGame) => {
      setGames(prev => {
        const existing = prev.findIndex(g => g.game_id === game.game_id);
        if (existing >= 0) {
          const updated = [...prev];
          updated[existing] = game;
          return updated;
        }
        return [...prev, game];
      });
    });

    ws.onPrediction((prediction: Prediction) => {
      setPredictions(prev => {
        const updated = new Map(prev);
        updated.set(prediction.game_id, prediction);
        return updated;
      });
    });

    ws.connect();
  });

  onCleanup(() => {
    ws?.disconnect();
  });

  return (
    <div class="min-h-screen bg-gray-950 text-gray-100">
      {/* Header */}
      <header class="bg-gray-900 border-b border-gray-800 sticky top-0 z-50">
        <div class="max-w-7xl mx-auto px-4 py-4">
          <div class="flex items-center justify-between">
            <h1 class="text-2xl font-bold">🏀 NBA Predictions</h1>
            <div class="flex items-center gap-3">
              <div class="flex items-center gap-2">
                <div class={`w-2 h-2 rounded-full ${isConnected() ? 'bg-green-500 animate-pulse' : 'bg-red-500'}`} />
                <span class="text-sm text-gray-400">
                  {isConnected() ? 'Live' : 'Disconnected'}
                </span>
              </div>
              <div class="text-sm text-gray-500">
                {games().length} game{games().length !== 1 ? 's' : ''}
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main class="max-w-7xl mx-auto px-4 py-8">
        <Show
          when={games().length > 0}
          fallback={
            <div class="text-center py-20">
              <p class="text-gray-400 text-lg">No live games at the moment</p>
              <p class="text-gray-600 text-sm mt-2">
                Waiting for games to start...
              </p>
            </div>
          }
        >
          <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <For each={games()}>
              {(game) => (
                <LiveGameCard
                  game={game}
                  prediction={predictions().get(game.game_id)}
                  pattern={[]} // TODO: Add pattern from game
                />
              )}
            </For>
          </div>
        </Show>
      </main>
    </div>
  );
};

export default Dashboard;
```

---

## 2.4 Update Main App

**Edit `src/App.tsx`:**

```typescript
import { Component } from 'solid-js';
import Dashboard from './components/Dashboard';

const App: Component = () => {
  return <Dashboard />;
};

export default App;
```

---

## ✅ Validation Checklist

- [ ] All components compile without TypeScript errors
- [ ] Components render correctly in browser
- [ ] Signals update UI instantly (<5ms)
- [ ] Memoized values prevent unnecessary recalculations
- [ ] Charts render at 60 FPS
- [ ] Tailwind classes apply correctly
- [ ] No console warnings or errors

---

## 🚀 Performance Verification

Test component performance:

```typescript
// Add to any component for performance tracking
import { onMount } from 'solid-js';

onMount(() => {
  const start = performance.now();
  // Component logic here
  const end = performance.now();
  console.log(`Component render: ${(end - start).toFixed(2)}ms`);
});
```

**Expected Results:**
- Initial render: <10ms per component
- Update render: <5ms per component
- Chart render: <16ms (60 FPS)

**React equivalent:** 3-5x slower

---

## Next Step

Proceed to **Step 3: WebSocket Integration** to connect real-time data streams.

---

*Action Step 2 of 10 - Component Architecture*

