/**
 * TRADING DESK - Professional Terminal for Manual Odds Entry
 * Real-time edge calculations as you type
 */

import { type Component, createSignal, createEffect, Show, For } from 'solid-js';
import type { NBAGame } from '../types';

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
}

const TradingDesk: Component<{
  games: NBAGame[];
  backendUrl: string;
}> = (props) => {
  
  // Convert NBAGame to Game format
  const gamesList = () => props.games.map(g => ({
    game_id: g.game_id,
    home_team: g.home_team,
    away_team: g.away_team,
    home_score: g.score_home,
    away_score: g.score_away,
    period: g.quarter,
    time_remaining: g.time_remaining,
    status: g.is_live ? 2 : 1
  }));
  const [selectedGame, setSelectedGame] = createSignal<Game | null>(null);
  const [mlPred, setMlPred] = createSignal<MLPrediction | null>(null);
  const [loading, setLoading] = createSignal(false);
  
  // Odds inputs
  const [spread, setSpread] = createSignal('');
  const [total, setTotal] = createSignal('');
  const [homeML, setHomeML] = createSignal('');
  
  let spreadInput: HTMLInputElement | undefined;

  // Fetch ML when game selected
  createEffect(() => {
    const game = selectedGame();
    if (game) {
      setLoading(true);
      fetch(`${props.backendUrl}/api/ml/prediction/${game.game_id}`)
        .then(r => r.ok ? r.json() : null)
        .then(data => {
          setMlPred(data);
          setLoading(false);
          setTimeout(() => spreadInput?.focus(), 100);
        })
        .catch(() => {
          setMlPred(null);
          setLoading(false);
        });
    }
  });

  // Calculate edge (reactive)
  const edge = () => {
    const ml = mlPred();
    const s = parseFloat(spread());
    if (!ml || isNaN(s)) return null;
    return ml.predicted_spread - s;
  };

  const edgePct = () => {
    const e = edge();
    const s = parseFloat(spread());
    if (e === null || isNaN(s) || s === 0) return null;
    return (e / Math.abs(s)) * 100;
  };

  const mlWinProb = () => mlPred()?.win_probability || 0;
  
  const marketImpliedProb = () => {
    const ml = parseInt(homeML() || '-110');
    return ml < 0 ? Math.abs(ml) / (Math.abs(ml) + 100) : 100 / (ml + 100);
  };

  const expectedValue = () => {
    const ml = mlWinProb();
    const market = marketImpliedProb();
    return (ml - market) * 100;
  };

  const kellyCriterion = () => {
    const ev = expectedValue();
    const ml = mlWinProb();
    if (ml === 0 || ev <= 0) return 0;
    const kelly = (ml - marketImpliedProb()) / (1 / ml - 1);
    return Math.max(0, Math.min(25, kelly * 100));
  };

  const fairOdds = () => {
    const prob = mlWinProb();
    if (prob === 0) return 0;
    return prob >= 0.5 ? Math.round(-100 * prob / (1 - prob)) : Math.round(100 * (1 - prob) / prob);
  };

  const submitTrade = async () => {
    const game = selectedGame();
    if (!game || !edge() || edge()! <= 0) return;

    try {
      await fetch(`${props.backendUrl}/api/betonline/manual-entry`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          game_id: game.game_id,
          home_team: game.home_team,
          away_team: game.away_team,
          spread: parseFloat(spread()),
          total: parseFloat(total() || '225'),
          home_ml: parseInt(homeML() || '-110'),
          away_ml: -110,
        }),
      });
      alert(`✅ Trade submitted!\nEdge: ${edge()!.toFixed(1)} pts\nKelly: ${kellyCriterion().toFixed(1)}%`);
    } catch (err) {
      alert('❌ Submission failed');
    }
  };

  const liveGames = () => gamesList().filter((g) => g.status === 2);

  return (
    <div class="min-h-screen bg-black p-4 sm:p-6">
      <div class="max-w-7xl mx-auto">
        
        {/* Header */}
        <div class="mb-6">
          <h1 class="text-3xl font-bold text-white mb-2">💰 Live Trading Terminal</h1>
          <p class="text-gray-400">Enter BetOnline odds → See edge calculations instantly</p>
        </div>

        {/* Game Selector */}
        <div class="bg-gray-900/50 border border-gray-800 rounded-2xl p-6 mb-6">
          <select
            value={selectedGame()?.game_id || ''}
            onChange={(e) => {
              const game = gamesList().find((g) => g.game_id === e.target.value);
              setSelectedGame(game || null);
              setSpread('');
              setTotal('');
              setHomeML('');
            }}
            class="w-full bg-black text-white text-lg font-semibold px-6 py-4 rounded-xl border-2 border-gray-700 focus:border-blue-500 focus:outline-none"
          >
            <option value="">-- Select Live Game --</option>
            <For each={liveGames()}>
              {(game) => (
                <option value={game.game_id}>
                  {game.away_team} @ {game.home_team} • Q{game.period} {game.time_remaining} • {game.away_score}-{game.home_score}
                </option>
              )}
            </For>
          </select>
        </div>

        <Show when={selectedGame() && !loading()}>
          <Show when={!mlPred()}>
            <div class="bg-yellow-900/20 border border-yellow-600/30 rounded-2xl p-8 text-center">
              <div class="text-5xl mb-4">⚠️</div>
              <h3 class="text-yellow-400 text-xl font-bold">No ML Prediction Available</h3>
              <p class="text-gray-400">Available at Q2 6:00 mark</p>
            </div>
          </Show>

          <Show when={mlPred()}>
            <div class="grid lg:grid-cols-2 gap-6">
              
              {/* LEFT: Odds Entry */}
              <div class="bg-gradient-to-br from-blue-950 to-purple-950 border-2 border-blue-500/30 rounded-2xl p-8">
                <h2 class="text-white text-xl font-bold mb-6">📊 Market Odds</h2>
                
                <div class="space-y-4">
                  <div>
                    <label class="block text-blue-400 text-xs font-bold mb-2">SPREAD ({selectedGame()!.home_team})</label>
                    <input
                      ref={spreadInput}
                      type="number"
                      step="0.5"
                      placeholder="-3.5"
                      value={spread()}
                      onInput={(e) => setSpread(e.target.value)}
                      class="w-full bg-black/50 text-white text-3xl font-bold text-center px-6 py-5 rounded-xl border-2 border-blue-500 focus:border-blue-400 focus:ring-4 focus:ring-blue-500/20 focus:outline-none"
                    />
                  </div>

                  <div class="grid grid-cols-2 gap-3">
                    <div>
                      <label class="block text-purple-400 text-xs font-bold mb-2">TOTAL</label>
                      <input
                        type="number"
                        step="0.5"
                        placeholder="225.5"
                        value={total()}
                        onInput={(e) => setTotal(e.target.value)}
                        class="w-full bg-black/50 text-white text-xl font-bold text-center px-3 py-3 rounded-xl border-2 border-purple-500 focus:outline-none"
                      />
                    </div>
                    <div>
                      <label class="block text-green-400 text-xs font-bold mb-2">HOME ML</label>
                      <input
                        type="number"
                        placeholder="-160"
                        value={homeML()}
                        onInput={(e) => setHomeML(e.target.value)}
                        class="w-full bg-black/50 text-white text-xl font-bold text-center px-3 py-3 rounded-xl border-2 border-green-500 focus:outline-none"
                      />
                    </div>
                  </div>
                </div>
              </div>

              {/* RIGHT: Calculations */}
              <div class="space-y-4">
                
                {/* EDGE - Hero Display */}
                <div class={`rounded-2xl p-8 text-center border-4 ${edge() !== null && edge()! > 2 ? 'bg-yellow-900/40 border-yellow-500' : edge() !== null && edge()! > 0 ? 'bg-green-900/40 border-green-500' : 'bg-gray-900/40 border-gray-700'}`}>
                  <div class="text-xs uppercase font-bold mb-2" style={`color: ${edge() !== null && edge()! > 2 ? '#fbbf24' : edge() !== null && edge()! > 0 ? '#10b981' : '#9ca3af'}`}>
                    {edge() !== null && edge()! > 2 ? '🚨 STRONG EDGE' : edge() !== null && edge()! > 0 ? '✅ POSITIVE EDGE' : '⚠️ NO EDGE'}
                  </div>
                  <div class="text-6xl font-black" style={`color: ${edge() !== null && edge()! > 2 ? '#fbbf24' : edge() !== null && edge()! > 0 ? '#10b981' : '#9ca3af'}`}>
                    {edge() !== null ? (edge()! > 0 ? '+' : '') + edge()!.toFixed(1) : '--'}
                  </div>
                  <div class="text-lg font-bold mt-2" style={`color: ${edge() !== null && edge()! > 2 ? '#fbbf24' : edge() !== null && edge()! > 0 ? '#10b981' : '#9ca3af'}`}>
                    {edgePct() !== null ? (edgePct()! > 0 ? '+' : '') + edgePct()!.toFixed(1) + '% edge' : 'Enter spread'}
                  </div>
                </div>

                {/* Metrics Grid */}
                <div class="grid grid-cols-2 gap-3">
                  <div class="bg-gray-900 border border-gray-700 rounded-xl p-4 text-center">
                    <div class="text-xs text-gray-500 uppercase font-bold mb-1">Expected Value</div>
                    <div class="text-3xl font-black text-green-400">{expectedValue() > 0 ? '+' : ''}{expectedValue().toFixed(1)}%</div>
                  </div>
                  <div class="bg-gray-900 border border-gray-700 rounded-xl p-4 text-center">
                    <div class="text-xs text-gray-500 uppercase font-bold mb-1">Kelly Criterion</div>
                    <div class="text-3xl font-black text-yellow-400">{kellyCriterion().toFixed(1)}%</div>
                  </div>
                  <div class="bg-gray-900 border border-gray-700 rounded-xl p-4 text-center">
                    <div class="text-xs text-gray-500 uppercase font-bold mb-1">ML Win%</div>
                    <div class="text-3xl font-black text-blue-400">{(mlWinProb() * 100).toFixed(1)}%</div>
                  </div>
                  <div class="bg-gray-900 border border-gray-700 rounded-xl p-4 text-center">
                    <div class="text-xs text-gray-500 uppercase font-bold mb-1">Fair Odds</div>
                    <div class="text-3xl font-black text-green-400">{fairOdds() > 0 ? '+' : ''}{fairOdds()}</div>
                  </div>
                </div>

                {/* Trade Button */}
                <button
                  onClick={submitTrade}
                  disabled={edge() === null || edge()! <= 0}
                  class={`w-full py-6 rounded-xl text-xl font-black uppercase ${edge() !== null && edge()! > 0 ? 'bg-green-600 text-white hover:bg-green-700 cursor-pointer' : 'bg-gray-800 text-gray-600 cursor-not-allowed'}`}
                >
                  {edge() !== null && edge()! > 0 ? `🎯 EXECUTE • ${kellyCriterion().toFixed(1)}% Bankroll` : '⛔ NO EDGE'}
                </button>
              </div>
            </div>
          </Show>
        </Show>

        {/* No Games */}
        <Show when={liveGames().length === 0}>
          <div class="bg-gray-900/50 border border-gray-800 rounded-2xl p-12 text-center mt-6">
            <div class="text-6xl mb-4">🏀</div>
            <h2 class="text-white text-2xl font-bold mb-3">No Live Games</h2>
            <p class="text-gray-400">Trading desk activates during live NBA games</p>
          </div>
        </Show>
      </div>
    </div>
  );
};

export default TradingDesk;
