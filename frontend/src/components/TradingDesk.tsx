/**
 * TRADING DESK - THE CENTERPIECE
 * Professional trading terminal for manual odds entry + real-time value calculations
 * 
 * This is where money is made - optimized for speed and clarity
 */

import { type Component, createSignal, createEffect, Show, For, onMount } from 'solid-js';

interface Game {
  game_id: string;
  home_team: string;
  away_team: string;
  home_score: number;
  away_score: number;
  period: number;
  time_remaining: string;
  status: number;
}

interface MLPrediction {
  game_id: string;
  predicted_spread: number;
  confidence_interval_90: [number, number];
  win_probability: number;
  model_confidence: number;
  timestamp: string;
}

interface Calculation {
  // Edge
  spread_edge: number;
  spread_edge_pct: number;
  
  // Probabilities
  ml_win_prob: number;
  market_implied_prob: number;
  true_edge_prob: number;
  
  // Value
  expected_value: number;
  kelly_criterion: number;
  
  // Odds
  fair_american: number;
  market_american: number;
  
  // Confidence
  ci_lower: number;
  ci_upper: number;
  model_confidence: number;
}

const TradingDesk: Component<{
  games: Game[];
  backendUrl: string;
}> = (props) => {
  // State
  const [selectedGame, setSelectedGame] = createSignal<Game | null>(null);
  const [mlPrediction, setMlPrediction] = createSignal<MLPrediction | null>(null);
  const [loading, setLoading] = createSignal(false);
  
  // Manual Entry Inputs
  const [spread, setSpread] = createSignal('');
  const [total, setTotal] = createSignal('');
  const [homeML, setHomeML] = createSignal('');
  const [awayML, setAwayML] = createSignal('');
  
  // Live Calculation
  const [calc, setCalc] = createSignal<Calculation | null>(null);
  
  // Refs for autofocus
  let spreadInput: HTMLInputElement | undefined;

  // Fetch ML prediction when game selected
  createEffect(() => {
    const game = selectedGame();
    if (game) {
      fetchMLPrediction(game.game_id);
      // Auto-focus spread input
      setTimeout(() => spreadInput?.focus(), 100);
    }
  });

  // INSTANT calculation on any input change
  createEffect(() => {
    const s = spread();
    const ml = mlPrediction();
    
    if (s && ml) {
      calculateInstant();
    } else {
      setCalc(null);
    }
  });

  const fetchMLPrediction = async (gameId: string) => {
    setLoading(true);
    try {
      const response = await fetch(`${props.backendUrl}/api/ml/prediction/${gameId}`);
      if (response.ok) {
        const data = await response.json();
        setMlPrediction(data);
      } else {
        setMlPrediction(null);
      }
    } catch (error) {
      console.error('ML fetch error:', error);
      setMlPrediction(null);
    } finally {
      setLoading(false);
    }
  };

  const calculateInstant = () => {
    const ml = mlPrediction();
    const marketSpread = parseFloat(spread());
    const marketML = parseInt(homeML() || '-110');

    if (!ml || isNaN(marketSpread)) {
      setCalc(null);
      return;
    }

    // EDGE CALCULATION
    const spreadEdge = ml.predicted_spread - marketSpread;
    const spreadEdgePct = (spreadEdge / (Math.abs(marketSpread) || 1)) * 100;

    // PROBABILITY CONVERSIONS
    const mlWinProb = ml.win_probability;
    const marketImpliedProb = americanToProb(marketML);
    const trueEdgeProb = mlWinProb - marketImpliedProb;

    // VALUE METRICS
    const expectedValue = trueEdgeProb * 100;
    const kellyFraction = trueEdgeProb / (1 / mlWinProb - 1);
    const kellyCriterion = Math.max(0, Math.min(25, kellyFraction * 100)); // Cap 0-25%

    // ODDS CONVERSIONS
    const fairAmerican = probToAmerican(mlWinProb);

    setCalc({
      spread_edge: spreadEdge,
      spread_edge_pct: spreadEdgePct,
      
      ml_win_prob: mlWinProb,
      market_implied_prob: marketImpliedProb,
      true_edge_prob: trueEdgeProb,
      
      expected_value: expectedValue,
      kelly_criterion: kellyCriterion,
      
      fair_american: fairAmerican,
      market_american: marketML,
      
      ci_lower: ml.confidence_interval_90[0],
      ci_upper: ml.confidence_interval_90[1],
      model_confidence: ml.model_confidence,
    });
  };

  const americanToProb = (american: number): number => {
    if (american > 0) {
      return 100 / (american + 100);
    } else {
      return Math.abs(american) / (Math.abs(american) + 100);
    }
  };

  const probToAmerican = (prob: number): number => {
    if (prob >= 0.5) {
      return Math.round(-100 * prob / (1 - prob));
    } else {
      return Math.round(100 * (1 - prob) / prob);
    }
  };

  const submitToOntoRisk = async () => {
    const game = selectedGame();
    const c = calc();
    
    if (!game || !c) return;

    try {
      const response = await fetch(`${props.backendUrl}/api/betonline/manual-entry`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          game_id: game.game_id,
          home_team: game.home_team,
          away_team: game.away_team,
          spread: parseFloat(spread()),
          total: parseFloat(total() || '225'),
          home_ml: parseInt(homeML() || '-110'),
          away_ml: parseInt(awayML() || '-110'),
        }),
      });

      if (response.ok) {
        const result = await response.json();
        alert(`✅ Odds submitted!\n\nEdge: ${c.spread_edge > 0 ? '+' : ''}${c.spread_edge.toFixed(1)} pts\nKelly: ${c.kelly_criterion.toFixed(1)}%`);
      } else {
        alert('❌ Submission failed');
      }
    } catch (error) {
      console.error('Submit error:', error);
      alert('❌ Network error');
    }
  };

  // Keyboard shortcuts
  onMount(() => {
    const handleKeyPress = (e: KeyboardEvent) => {
      if (e.key === 'Enter' && e.metaKey && calc()) {
        submitToOntoRisk();
      }
    };
    document.addEventListener('keydown', handleKeyPress);
    return () => document.removeEventListener('keydown', handleKeyPress);
  });

  return (
    <div class="min-h-screen bg-black p-4 sm:p-6">
      {/* Terminal Header */}
      <div class="max-w-7xl mx-auto mb-6">
        <div class="flex items-center justify-between">
          <div>
            <h1 class="text-3xl font-bold text-white mb-2">
              💰 Live Trading Terminal
            </h1>
            <p class="text-gray-400 text-sm">
              Manual odds entry → Instant edge calculations → OntoRisk execution
            </p>
          </div>
          <div class="text-right">
            <div class="text-xs text-gray-500 uppercase tracking-wider mb-1">System Status</div>
            <div class="text-green-400 font-bold">🟢 LIVE</div>
          </div>
        </div>
      </div>

      {/* Game Selector - Full Width */}
      <div class="max-w-7xl mx-auto mb-6">
        <div class="bg-gray-900/50 backdrop-blur-xl border border-gray-800 rounded-2xl p-6">
          <label class="block text-gray-400 text-xs uppercase tracking-wider font-semibold mb-3">
            Select Live Game to Trade:
          </label>
          <select
            value={selectedGame()?.game_id || ''}
            onChange={(e) => {
              const game = props.games.find(g => g.game_id === e.target.value);
              setSelectedGame(game || null);
              setSpread('');
              setTotal('');
              setHomeML('');
              setAwayML('');
              setCalc(null);
              setMlPrediction(null);
            }}
            class="w-full bg-black/50 text-white text-lg font-semibold px-6 py-4 rounded-xl border-2 border-gray-700 focus:border-blue-500 focus:outline-none transition-all"
          >
            <option value="" class="bg-gray-900">-- Select a live game to start trading --</option>
            <For each={props.games.filter(g => g.status === 2)}>
              {(game) => (
                <option value={game.game_id} class="bg-gray-900">
                  {game.away_team} @ {game.home_team} • Q{game.period} {game.time_remaining} • {game.away_score}-{game.home_score}
                </option>
              )}
            </For>
          </select>
        </div>
      </div>

      {/* Main Trading Interface */}
      <Show when={selectedGame()}>
        <div class="max-w-7xl mx-auto">
          {/* Loading ML Prediction */}
          <Show when={loading()}>
            <div class="text-center py-12">
              <div class="inline-block animate-spin rounded-full h-12 w-12 border-4 border-blue-500 border-t-transparent mb-4"></div>
              <p class="text-gray-400">Loading ML prediction...</p>
            </div>
          </Show>

          {/* No ML Prediction Available */}
          <Show when={!loading() && !mlPrediction()}>
            <div class="bg-yellow-900/20 border border-yellow-600/30 rounded-2xl p-8 text-center">
              <div class="text-5xl mb-4">⚠️</div>
              <h3 class="text-yellow-400 text-xl font-bold mb-2">No ML Prediction Available</h3>
              <p class="text-gray-400">ML model prediction not available for this game yet</p>
              <p class="text-gray-500 text-sm mt-2">Prediction available at Q2 6:00 mark</p>
            </div>
          </Show>

          {/* TRADING TERMINAL */}
          <Show when={!loading() && mlPrediction()}>
            <div class="grid lg:grid-cols-2 gap-6">
              
              {/* LEFT PANEL: Odds Entry Terminal */}
              <div class="bg-gradient-to-br from-blue-950 via-purple-950 to-blue-950 border-2 border-blue-500/30 rounded-2xl p-8 shadow-2xl">
                <div class="flex items-center justify-between mb-6">
                  <h2 class="text-white text-xl font-bold">📊 Market Odds Entry</h2>
                  <div class="text-xs text-blue-400 font-mono">BetOnline</div>
                </div>

                {/* Game Info */}
                <div class="bg-black/30 rounded-xl p-4 mb-6 border border-blue-500/20">
                  <div class="flex justify-between items-center mb-2">
                    <span class="text-gray-400 text-sm">{selectedGame()!.away_team}</span>
                    <span class="text-white text-2xl font-bold">{selectedGame()!.away_score}</span>
                  </div>
                  <div class="flex justify-between items-center">
                    <span class="text-gray-400 text-sm">{selectedGame()!.home_team}</span>
                    <span class="text-white text-2xl font-bold">{selectedGame()!.home_score}</span>
                  </div>
                  <div class="text-center mt-3 text-blue-400 text-sm font-semibold">
                    Q{selectedGame()!.period} • {selectedGame()!.time_remaining}
                  </div>
                </div>

                {/* Odds Inputs - Trading Style */}
                <div class="space-y-4">
                  
                  {/* SPREAD - Primary Input */}
                  <div>
                    <label class="block text-blue-400 text-xs uppercase tracking-wider font-bold mb-2">
                      Spread ({selectedGame()!.home_team})
                    </label>
                    <input
                      ref={spreadInput}
                      type="number"
                      step="0.5"
                      placeholder="-3.5"
                      value={spread()}
                      onInput={(e) => setSpread(e.target.value)}
                      class="w-full bg-black/50 text-white text-3xl font-bold text-center px-6 py-5 rounded-xl border-2 border-blue-500/50 focus:border-blue-400 focus:ring-4 focus:ring-blue-500/20 focus:outline-none transition-all placeholder-gray-600"
                      autofocus
                    />
                  </div>

                  {/* Grid: Total + Moneylines */}
                  <div class="grid grid-cols-3 gap-3">
                    
                    {/* Total */}
                    <div>
                      <label class="block text-purple-400 text-xs uppercase tracking-wider font-bold mb-2">
                        Total
                      </label>
                      <input
                        type="number"
                        step="0.5"
                        placeholder="225.5"
                        value={total()}
                        onInput={(e) => setTotal(e.target.value)}
                        class="w-full bg-black/50 text-white text-xl font-bold text-center px-3 py-3 rounded-xl border-2 border-purple-500/50 focus:border-purple-400 focus:ring-4 focus:ring-purple-500/20 focus:outline-none transition-all placeholder-gray-600"
                      />
                    </div>

                    {/* Home ML */}
                    <div>
                      <label class="block text-green-400 text-xs uppercase tracking-wider font-bold mb-2">
                        {selectedGame()!.home_team} ML
                      </label>
                      <input
                        type="number"
                        placeholder="-160"
                        value={homeML()}
                        onInput={(e) => setHomeML(e.target.value)}
                        class="w-full bg-black/50 text-white text-xl font-bold text-center px-3 py-3 rounded-xl border-2 border-green-500/50 focus:border-green-400 focus:ring-4 focus:ring-green-500/20 focus:outline-none transition-all placeholder-gray-600"
                      />
                    </div>

                    {/* Away ML */}
                    <div>
                      <label class="block text-red-400 text-xs uppercase tracking-wider font-bold mb-2">
                        {selectedGame()!.away_team} ML
                      </label>
                      <input
                        type="number"
                        placeholder="+140"
                        value={awayML()}
                        onInput={(e) => setAwayML(e.target.value)}
                        class="w-full bg-black/50 text-white text-xl font-bold text-center px-3 py-3 rounded-xl border-2 border-red-500/50 focus:border-red-400 focus:ring-4 focus:ring-red-500/20 focus:outline-none transition-all placeholder-gray-600"
                      />
                    </div>
                  </div>
                </div>

                {/* Quick Tips */}
                <div class="mt-6 p-4 bg-blue-500/10 rounded-lg border border-blue-500/20">
                  <div class="text-blue-300 text-xs space-y-1">
                    <p>💡 <strong>Tab</strong> to navigate inputs</p>
                    <p>⌘ <strong>Cmd+Enter</strong> to submit to OntoRisk</p>
                    <p>🎯 Calculations update <strong>instantly</strong> as you type</p>
                  </div>
                </div>
              </div>

              {/* RIGHT PANEL: Real-Time Calculations */}
              <div class="space-y-6">
                
                <Show
                  when={calc()}
                  fallback={
                    <div class="bg-gray-900/50 backdrop-blur-xl border border-gray-800 rounded-2xl p-12 text-center">
                      <div class="text-6xl mb-4">📈</div>
                      <h3 class="text-gray-400 text-xl font-bold mb-2">Waiting for Odds Entry</h3>
                      <p class="text-gray-600">Enter spread to see calculations</p>
                    </div>
                  }
                >
                  {/* EDGE DISPLAY - Hero Section */}
                  <div 
                    class={`
                      rounded-2xl p-8 text-center border-4 transition-all duration-300
                      ${calc()!.spread_edge > 2 
                        ? 'bg-gradient-to-br from-yellow-900/40 to-orange-900/40 border-yellow-500 shadow-2xl shadow-yellow-500/20' 
                        : calc()!.spread_edge > 0
                        ? 'bg-gradient-to-br from-green-900/40 to-emerald-900/40 border-green-500/50'
                        : 'bg-gradient-to-br from-gray-900/40 to-gray-800/40 border-gray-700'}
                    `}
                  >
                    <div class="text-xs uppercase tracking-widest font-bold mb-3" 
                         style={`color: ${calc()!.spread_edge > 2 ? '#fbbf24' : calc()!.spread_edge > 0 ? '#10b981' : '#9ca3af'}`}>
                      {calc()!.spread_edge > 2 ? '🚨 STRONG EDGE DETECTED' : 
                       calc()!.spread_edge > 0 ? '✅ POSITIVE EDGE' : 
                       '⚠️ NO EDGE - MARKET EFFICIENT'}
                    </div>
                    <div class="text-7xl font-black mb-3" 
                         style={`color: ${calc()!.spread_edge > 2 ? '#fbbf24' : calc()!.spread_edge > 0 ? '#10b981' : '#9ca3af'}`}>
                      {calc()!.spread_edge > 0 ? '+' : ''}{calc()!.spread_edge.toFixed(1)}
                    </div>
                    <div class="text-xl font-bold" 
                         style={`color: ${calc()!.spread_edge > 2 ? '#fbbf24' : calc()!.spread_edge > 0 ? '#10b981' : '#9ca3af'}`}>
                      {calc()!.spread_edge > 0 ? '+' : ''}{calc()!.spread_edge_pct.toFixed(1)}% point edge
                    </div>
                  </div>

                  {/* VALUE METRICS - Trading Grid */}
                  <div class="grid grid-cols-2 gap-4">
                    
                    {/* Expected Value */}
                    <div class="bg-gray-900/80 backdrop-blur-xl border border-gray-700 rounded-xl p-5">
                      <div class="text-xs text-gray-500 uppercase tracking-wider font-bold mb-2">Expected Value</div>
                      <div class={`text-4xl font-black ${calc()!.expected_value > 0 ? 'text-green-400' : 'text-red-400'}`}>
                        {calc()!.expected_value > 0 ? '+' : ''}{calc()!.expected_value.toFixed(2)}%
                      </div>
                      <div class="text-xs text-gray-500 mt-1">Per $100 wagered</div>
                    </div>

                    {/* Kelly Criterion */}
                    <div class="bg-gray-900/80 backdrop-blur-xl border border-gray-700 rounded-xl p-5">
                      <div class="text-xs text-gray-500 uppercase tracking-wider font-bold mb-2">Kelly Criterion</div>
                      <div class="text-4xl font-black text-yellow-400">
                        {calc()!.kelly_criterion.toFixed(1)}%
                      </div>
                      <div class="text-xs text-gray-500 mt-1">Of bankroll</div>
                    </div>

                    {/* ML Win Probability */}
                    <div class="bg-gray-900/80 backdrop-blur-xl border border-gray-700 rounded-xl p-5">
                      <div class="text-xs text-gray-500 uppercase tracking-wider font-bold mb-2">ML Win Prob</div>
                      <div class="text-4xl font-black text-blue-400">
                        {(calc()!.ml_win_prob * 100).toFixed(1)}%
                      </div>
                      <div class="text-xs text-gray-500 mt-1">Model confidence: {(calc()!.model_confidence * 100).toFixed(0)}%</div>
                    </div>

                    {/* Market Implied */}
                    <div class="bg-gray-900/80 backdrop-blur-xl border border-gray-700 rounded-xl p-5">
                      <div class="text-xs text-gray-500 uppercase tracking-wider font-bold mb-2">Market Implied</div>
                      <div class="text-4xl font-black text-purple-400">
                        {(calc()!.market_implied_prob * 100).toFixed(1)}%
                      </div>
                      <div class="text-xs text-gray-500 mt-1">From {calc()!.market_american} odds</div>
                    </div>
                  </div>

                  {/* Odds Comparison */}
                  <div class="bg-gray-900/80 backdrop-blur-xl border border-gray-700 rounded-xl p-6">
                    <h3 class="text-white text-sm font-bold mb-4 uppercase tracking-wider">Odds Comparison</h3>
                    <div class="grid grid-cols-2 gap-6">
                      <div>
                        <div class="text-xs text-gray-500 mb-1">ML Fair Odds</div>
                        <div class="text-3xl font-bold text-green-400">
                          {calc()!.fair_american > 0 ? '+' : ''}{calc()!.fair_american}
                        </div>
                      </div>
                      <div>
                        <div class="text-xs text-gray-500 mb-1">Market Odds</div>
                        <div class="text-3xl font-bold text-red-400">
                          {calc()!.market_american > 0 ? '+' : ''}{calc()!.market_american}
                        </div>
                      </div>
                    </div>
                    <div class="mt-4 pt-4 border-t border-gray-700">
                      <div class="flex justify-between items-center">
                        <span class="text-gray-400 text-sm">ML Spread Prediction:</span>
                        <span class="text-white text-xl font-bold">
                          {mlPrediction()!.predicted_spread > 0 ? '+' : ''}{mlPrediction()!.predicted_spread.toFixed(1)}
                        </span>
                      </div>
                      <div class="flex justify-between items-center mt-2">
                        <span class="text-gray-400 text-sm">90% Confidence Interval:</span>
                        <span class="text-gray-300 text-sm font-mono">
                          [{calc()!.ci_lower.toFixed(1)}, {calc()!.ci_upper.toFixed(1)}]
                        </span>
                      </div>
                    </div>
                  </div>

                  {/* EXECUTION BUTTON */}
                  <button
                    onClick={submitToOntoRisk}
                    disabled={!calc() || calc()!.spread_edge <= 0}
                    class={`
                      w-full py-6 rounded-xl text-xl font-black uppercase tracking-wider
                      transition-all duration-200 transform
                      ${calc() && calc()!.spread_edge > 0
                        ? 'bg-gradient-to-r from-green-600 to-emerald-600 text-white shadow-2xl shadow-green-500/30 hover:scale-[1.02] active:scale-[0.98] cursor-pointer'
                        : 'bg-gray-800 text-gray-600 cursor-not-allowed opacity-50'}
                    `}
                  >
                    {calc() && calc()!.spread_edge > 0 
                      ? `🎯 Execute Trade • Bet ${calc()!.kelly_criterion.toFixed(1)}% of Bankroll`
                      : '⛔ No Edge - Hold'}
                  </button>

                  {/* Trade Explanation */}
                  <Show when={calc() && calc()!.spread_edge > 0}>
                    <div class="bg-green-900/20 border border-green-500/30 rounded-xl p-4">
                      <div class="text-green-400 text-sm font-semibold mb-2">📚 Trade Rationale:</div>
                      <div class="text-gray-300 text-xs leading-relaxed space-y-1">
                        <p>
                          • ML predicts <strong>{mlPrediction()!.predicted_spread > 0 ? '+' : ''}{mlPrediction()!.predicted_spread.toFixed(1)}</strong>, 
                          market offers <strong>{parseFloat(spread()) > 0 ? '+' : ''}{parseFloat(spread()).toFixed(1)}</strong>
                        </p>
                        <p>
                          • Edge of <strong>{Math.abs(calc()!.spread_edge).toFixed(1)} points</strong> in our favor
                        </p>
                        <p>
                          • ML gives {selectedGame()!.home_team} <strong>{(calc()!.ml_win_prob * 100).toFixed(1)}%</strong> win probability
                        </p>
                        <p>
                          • Market implies only <strong>{(calc()!.market_implied_prob * 100).toFixed(1)}%</strong> 
                          → <strong>{(calc()!.true_edge_prob * 100).toFixed(1)}%</strong> true edge
                        </p>
                        <p>
                          • Expected value: <strong>+{calc()!.expected_value.toFixed(2)}%</strong> long-term profit
                        </p>
                        <p>
                          • Kelly says bet <strong>{calc()!.kelly_criterion.toFixed(1)}%</strong> for optimal growth
                        </p>
                      </div>
                    </div>
                  </Show>
                </div>
              </div>

              {/* RIGHT PANEL: Price Ladder (Trading Desk Style) */}
              <div class="space-y-6">
                
                {/* Spread Ladder */}
                <div class="bg-gray-900/80 backdrop-blur-xl border border-gray-700 rounded-2xl p-6">
                  <h3 class="text-white text-sm font-bold mb-4 uppercase tracking-wider">📊 Spread Ladder</h3>
                  <div class="space-y-2">
                    <For each={[-7, -5.5, -4, -2.5, -1, 0, +1, +2.5, +4, +5.5, +7]}>
                      {(testSpread) => {
                        const isCurrent = Math.abs(parseFloat(spread() || '999') - testSpread) < 0.1;
                        const isMLPrediction = Math.abs(mlPrediction()!.predicted_spread - testSpread) < 1;
                        
                        return (
                          <div 
                            class={`
                              flex justify-between items-center px-4 py-2 rounded-lg cursor-pointer transition-all
                              ${isCurrent 
                                ? 'bg-blue-600 text-white font-bold ring-2 ring-blue-400' 
                                : isMLPrediction
                                ? 'bg-green-600/20 text-green-400 border border-green-500/30'
                                : 'bg-black/30 text-gray-400 hover:bg-black/50'}
                            `}
                            onClick={() => setSpread(testSpread.toString())}
                          >
                            <span class="font-mono text-sm">
                              {testSpread > 0 ? '+' : ''}{testSpread.toFixed(1)}
                            </span>
                            <span class="text-xs">
                              {isMLPrediction ? '🎯 ML' : isCurrent ? '← MARKET' : '-110'}
                            </span>
                          </div>
                        );
                      }}
                    </For>
                  </div>
                </div>

                {/* ML Prediction Box */}
                <div class="bg-gradient-to-br from-green-950 to-emerald-950 border-2 border-green-500/30 rounded-2xl p-6">
                  <h3 class="text-green-400 text-sm font-bold mb-4 uppercase tracking-wider">🤖 ML Model Prediction</h3>
                  <div class="space-y-3">
                    <div class="flex justify-between items-center">
                      <span class="text-gray-400 text-sm">Predicted Spread:</span>
                      <span class="text-white text-2xl font-black">
                        {mlPrediction()!.predicted_spread > 0 ? '+' : ''}{mlPrediction()!.predicted_spread.toFixed(1)}
                      </span>
                    </div>
                    <div class="flex justify-between items-center">
                      <span class="text-gray-400 text-sm">Win Probability:</span>
                      <span class="text-green-400 text-xl font-bold">
                        {(mlPrediction()!.win_probability * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div class="flex justify-between items-center">
                      <span class="text-gray-400 text-sm">Model Confidence:</span>
                      <span class="text-blue-400 text-xl font-bold">
                        {(mlPrediction()!.model_confidence * 100).toFixed(0)}%
                      </span>
                    </div>
                    <div class="pt-3 border-t border-green-500/20">
                      <div class="text-gray-500 text-xs mb-2">90% Confidence Interval:</div>
                      <div class="flex justify-between items-center bg-black/30 rounded-lg px-4 py-2">
                        <span class="text-gray-400 font-mono">{calc()!.ci_lower.toFixed(1)}</span>
                        <span class="text-gray-600">→</span>
                        <span class="text-white font-bold text-lg">{mlPrediction()!.predicted_spread.toFixed(1)}</span>
                        <span class="text-gray-600">→</span>
                        <span class="text-gray-400 font-mono">{calc()!.ci_upper.toFixed(1)}</span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </Show>
        </div>
      </Show>

      {/* No Games Available */}
      <Show when={!selectedGame() && props.games.filter(g => g.status === 2).length === 0}>
        <div class="max-w-4xl mx-auto mt-12">
          <div class="bg-gray-900/50 backdrop-blur-xl border border-gray-800 rounded-2xl p-12 text-center">
            <div class="text-6xl mb-4">🏀</div>
            <h2 class="text-white text-2xl font-bold mb-3">No Live Games Right Now</h2>
            <p class="text-gray-400 mb-6">Trading desk activates when NBA games are live</p>
            <p class="text-gray-600 text-sm">Check back during game times for live trading</p>
          </div>
        </div>
      </Show>
    </div>
  );
};

export default TradingDesk;
