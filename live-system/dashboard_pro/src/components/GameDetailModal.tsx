import { createSignal, onMount, onCleanup, Show } from 'solid-js';
import axios from 'axios';

interface GameDetailModalProps {
  game: any;
  onClose: () => void;
}

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8001';

export default function GameDetailModal(props: GameDetailModalProps) {
  const [gameData, setGameData] = createSignal<any>(props.game);
  const [betOnlineData, setBetOnlineData] = createSignal<any>(null);
  const [lastUpdate, setLastUpdate] = createSignal<string>(new Date().toLocaleTimeString());
  let interval: number;

  const fetchLiveGameData = async () => {
    try {
      // Cache-busting for real-time data
      const timestamp = Date.now();
      const config = {
        headers: {
          'Cache-Control': 'no-cache, no-store, must-revalidate',
          'Pragma': 'no-cache',
          'Expires': '0'
        }
      };
      
      const [gameRes, oddsRes] = await Promise.all([
        axios.get(`${API_URL}/api/live-games?t=${timestamp}`, config),
        axios.get(`${API_URL}/api/betonline/live/${props.game.game_id}?t=${timestamp}`, config).catch(() => ({ data: null }))
      ]);
      
      const updatedGame = gameRes.data.games.find((g: any) => g.game_id === props.game.game_id);
      if (updatedGame) {
        setGameData(updatedGame);
      }
      
      if (oddsRes.data) {
        setBetOnlineData(oddsRes.data);
      }
      
      setLastUpdate(new Date().toLocaleTimeString());
    } catch (error) {
      console.error('Error fetching game data:', error);
    }
  };

  onMount(() => {
    fetchLiveGameData();
    interval = setInterval(fetchLiveGameData, 2000) as unknown as number; // Update every 2 seconds - ULTRA FAST!
  });

  onCleanup(() => {
    clearInterval(interval);
  });

  const game = () => gameData();

  const getStatusColor = () => {
    const status = game().status_text;
    if (status === 'LIVE') return 'text-green-400';
    if (status === 'FINAL') return 'text-red-400';
    return 'text-yellow-400';
  };

  const getLeadingTeam = () => {
    const homeScore = game().home_score || 0;
    const awayScore = game().away_score || 0;
    if (homeScore > awayScore) return game().home_team;
    if (awayScore > homeScore) return game().away_team;
    return 'TIED';
  };

  return (
    <div class="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div class="bg-gradient-to-br from-slate-900 to-slate-800 rounded-3xl border-2 border-white/20 shadow-2xl max-w-4xl w-full max-h-[90vh] overflow-y-auto">
        {/* Header */}
        <div class="sticky top-0 bg-gradient-to-r from-purple-900/90 to-blue-900/90 backdrop-blur-xl p-6 border-b border-white/20 flex justify-between items-center">
          <div>
            <h2 class="text-3xl font-black text-white mb-2">
              {game().away_team} @ {game().home_team}
            </h2>
            <div class="flex gap-4 items-center">
              <span class={`text-lg font-bold ${getStatusColor()}`}>
                {game().status_text}
              </span>
              <Show when={game().period > 0}>
                <span class="text-white/70">
                  Q{game().period} • {game().clock || '12:00'}
                </span>
              </Show>
              <span class="text-white/50 text-sm">
                Updates every 5s • Last: {lastUpdate()}
              </span>
            </div>
          </div>
          <button 
            onClick={props.onClose}
            class="text-white/60 hover:text-white text-3xl font-bold"
          >
            ✕
          </button>
        </div>

        {/* Live Score */}
        <div class="p-6">
          <div class="bg-gradient-to-br from-black/60 to-black/40 rounded-2xl p-8 border border-white/10 mb-6">
            <div class="grid grid-cols-2 gap-8">
              {/* Away Team */}
              <div class={`text-center p-6 rounded-xl ${game().away_score > game().home_score ? 'bg-green-500/20 border-2 border-green-500' : 'bg-white/5'}`}>
                <div class="text-6xl font-black text-white mb-2">
                  {game().away_score || 0}
                </div>
                <div class="text-2xl font-bold text-white/80">
                  {game().away_team}
                </div>
                <Show when={game().away_score > game().home_score}>
                  <div class="text-green-400 text-sm font-semibold mt-2">
                    LEADING
                  </div>
                </Show>
              </div>

              {/* Home Team */}
              <div class={`text-center p-6 rounded-xl ${game().home_score > game().away_score ? 'bg-green-500/20 border-2 border-green-500' : 'bg-white/5'}`}>
                <div class="text-6xl font-black text-white mb-2">
                  {game().home_score || 0}
                </div>
                <div class="text-2xl font-bold text-white/80">
                  {game().home_team}
                </div>
                <Show when={game().home_score > game().away_score}>
                  <div class="text-green-400 text-sm font-semibold mt-2">
                    LEADING
                  </div>
                </Show>
              </div>
            </div>

            {/* Score Differential */}
            <div class="mt-6 text-center">
              <div class="text-white/50 text-sm mb-1">CURRENT DIFFERENTIAL</div>
              <div class="text-4xl font-black text-white">
                {game().current_diff > 0 ? '+' : ''}{game().current_diff || 0}
              </div>
              <Show when={getLeadingTeam() !== 'TIED'}>
                <div class="text-white/60 text-sm mt-2">
                  {getLeadingTeam()} by {Math.abs(game().current_diff || 0)}
                </div>
              </Show>
            </div>
          </div>

          {/* Game Info */}
          <div class="grid grid-cols-3 gap-4 mb-6">
            <div class="bg-black/40 rounded-xl p-4 border border-white/10">
              <div class="text-white/50 text-xs mb-1">GAME ID</div>
              <div class="text-white font-semibold text-sm">{game().game_id}</div>
            </div>
            <div class="bg-black/40 rounded-xl p-4 border border-white/10">
              <div class="text-white/50 text-xs mb-1">PERIOD</div>
              <div class="text-white font-semibold text-sm">
                {game().period === 0 ? 'PREGAME' : `Q${game().period}`}
              </div>
            </div>
            <div class="bg-black/40 rounded-xl p-4 border border-white/10">
              <div class="text-white/50 text-xs mb-1">CLOCK</div>
              <div class="text-white font-semibold text-sm">
                {game().clock || 'Not Started'}
              </div>
            </div>
          </div>

          {/* Q2 6:00 Status */}
          <Show when={game().period === 2}>
            <div class={`rounded-xl p-6 mb-6 border-2 ${game().is_q2_6min ? 'bg-green-500/20 border-green-500' : 'bg-yellow-500/20 border-yellow-500/50'}`}>
              <div class="flex items-center justify-between">
                <div>
                  <div class="text-white font-bold text-lg mb-1">
                    {game().is_q2_6min ? '🔥 Q2 6:00 - PREDICTION TIME!' : '⏰ Waiting for Q2 6:00...'}
                  </div>
                  <div class="text-white/70 text-sm">
                    {game().is_q2_6min 
                      ? 'ML prediction active! Check opportunities section.' 
                      : `Current: ${game().clock || '12:00'} remaining in Q2`}
                  </div>
                </div>
                <Show when={game().is_q2_6min}>
                  <div class="text-5xl animate-pulse">🎯</div>
                </Show>
              </div>
            </div>
          </Show>

          {/* Prediction Status */}
          <div class="bg-black/40 rounded-xl p-6 border border-purple-500/30 mb-6">
            <div class="text-white font-bold mb-4">📊 ML Prediction Status</div>
            <div class="space-y-2 text-sm">
              <div class="flex justify-between">
                <span class="text-white/60">Can Predict:</span>
                <span class={`font-semibold ${game().can_predict ? 'text-green-400' : 'text-red-400'}`}>
                  {game().can_predict ? '✅ YES' : '❌ NOT YET'}
                </span>
              </div>
              <div class="flex justify-between">
                <span class="text-white/60">At Q2 6:00:</span>
                <span class={`font-semibold ${game().is_q2_6min ? 'text-green-400' : 'text-yellow-400'}`}>
                  {game().is_q2_6min ? '✅ ACTIVE NOW' : '⏰ WAITING'}
                </span>
              </div>
              <div class="flex justify-between">
                <span class="text-white/60">System:</span>
                <span class="text-white font-semibold">Mamba Mentality</span>
              </div>
            </div>
          </div>

          {/* BetOnline Live Odds */}
          <Show when={betOnlineData()}>
            <div class="bg-gradient-to-br from-yellow-500/20 to-orange-500/20 rounded-xl p-6 border border-yellow-500/30 mb-6">
              <div class="text-yellow-400 font-bold mb-4 flex items-center gap-2">
                <span>💰</span>
                <span>BetOnline Live Odds</span>
              </div>
              <div class="grid grid-cols-3 gap-4">
                <div class="bg-black/40 rounded-lg p-4">
                  <div class="text-white/50 text-xs mb-1">SPREAD</div>
                  <div class="text-2xl font-bold text-white">
                    {betOnlineData()?.spread > 0 ? '+' : ''}{betOnlineData()?.spread}
                  </div>
                </div>
                <div class="bg-black/40 rounded-lg p-4">
                  <div class="text-white/50 text-xs mb-1">TOTAL</div>
                  <div class="text-2xl font-bold text-white">
                    {betOnlineData()?.total || 'N/A'}
                  </div>
                </div>
                <div class="bg-black/40 rounded-lg p-4">
                  <div class="text-white/50 text-xs mb-1">SOURCE</div>
                  <div class="text-sm font-bold text-yellow-400">
                    {betOnlineData()?.source || 'BetOnline'}
                  </div>
                </div>
              </div>
              <div class="mt-4 text-xs text-white/60">
                ✅ Live odds updating every 5 seconds
              </div>
            </div>
          </Show>

          <Show when={!betOnlineData()}>
            <div class="bg-gray-500/10 rounded-xl p-6 border border-gray-500/30 mb-6">
              <div class="text-white/50 font-bold mb-2 flex items-center gap-2">
                <span>💰</span>
                <span>BetOnline Odds</span>
              </div>
              <div class="text-white/40 text-sm">
                Waiting for live odds... (available when game is active)
              </div>
            </div>
          </Show>

          {/* Raw JSON Toggle */}
          <details class="bg-black/40 rounded-xl border border-white/10 mb-4">
            <summary class="p-4 text-white/70 font-semibold cursor-pointer hover:text-white">
              🔍 Show Raw API Data (Debug)
            </summary>
            <div class="p-4 bg-black/60 space-y-4">
              <div>
                <div class="text-green-400 font-bold mb-2 text-sm">🏀 NBA API Response:</div>
                <pre class="text-xs text-white/90 font-mono overflow-auto max-h-64 bg-black/50 rounded p-3">
{JSON.stringify(game(), null, 2)}
                </pre>
              </div>
              <Show when={betOnlineData()}>
                <div>
                  <div class="text-yellow-400 font-bold mb-2 text-sm">💰 BetOnline Response:</div>
                  <pre class="text-xs text-white/90 font-mono overflow-auto max-h-64 bg-black/50 rounded p-3">
{JSON.stringify(betOnlineData(), null, 2)}
                  </pre>
                </div>
              </Show>
            </div>
          </details>
        </div>

        {/* Footer */}
        <div class="sticky bottom-0 bg-gradient-to-r from-purple-900/90 to-blue-900/90 backdrop-blur-xl p-4 border-t border-white/20 flex justify-between items-center">
          <div class="text-white/60 text-sm">
            Auto-refreshing every 5 seconds
          </div>
          <button 
            onClick={props.onClose}
            class="px-6 py-2 bg-white/10 hover:bg-white/20 text-white font-semibold rounded-lg transition-all"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
}

