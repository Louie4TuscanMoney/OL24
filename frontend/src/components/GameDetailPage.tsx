import { type Component, createSignal, createEffect, onCleanup, Show } from 'solid-js';
import { wsService } from '../services/websocket';
import { MambaLiveWidget } from './MambaLiveWidget';
import { LiveSnapshotsWidget } from './LiveSnapshotsWidget';
import { MambaFeaturesWidget } from './MambaFeaturesWidget';
import { WinProbabilityWidget } from './WinProbabilityWidget';

interface GameDetailPageProps {
  gameId: string;
}

// Team ID mapping
const getTeamId = (teamIdentifier: string): string => {
  const teamMap: Record<string, string> = {
    'Atlanta Hawks': '1610612737', 'Boston Celtics': '1610612738', 'Brooklyn Nets': '1610612751',
    'Charlotte Hornets': '1610612766', 'Chicago Bulls': '1610612741', 'Cleveland Cavaliers': '1610612739',
    'Dallas Mavericks': '1610612742', 'Denver Nuggets': '1610612743', 'Detroit Pistons': '1610612765',
    'Golden State Warriors': '1610612744', 'Houston Rockets': '1610612745', 'Indiana Pacers': '1610612754',
    'Los Angeles Clippers': '1610612746', 'LA Clippers': '1610612746', 'Los Angeles Lakers': '1610612747',
    'Memphis Grizzlies': '1610612763', 'Miami Heat': '1610612748', 'Milwaukee Bucks': '1610612749',
    'Minnesota Timberwolves': '1610612750', 'New Orleans Pelicans': '1610612740', 'New York Knicks': '1610612752',
    'Oklahoma City Thunder': '1610612760', 'Orlando Magic': '1610612753', 'Philadelphia 76ers': '1610612755',
    'Phoenix Suns': '1610612756', 'Portland Trail Blazers': '1610612757', 'Sacramento Kings': '1610612758',
    'San Antonio Spurs': '1610612759', 'Toronto Raptors': '1610612761', 'Utah Jazz': '1610612762',
    'Washington Wizards': '1610612764',
    'ATL': '1610612737', 'BOS': '1610612738', 'BKN': '1610612751', 'CHA': '1610612766',
    'CHI': '1610612741', 'CLE': '1610612739', 'DAL': '1610612742', 'DEN': '1610612743',
    'DET': '1610612765', 'GSW': '1610612744', 'HOU': '1610612745', 'IND': '1610612754',
    'LAC': '1610612746', 'LAL': '1610612747', 'MEM': '1610612763', 'MIA': '1610612748',
    'MIL': '1610612749', 'MIN': '1610612750', 'NOP': '1610612740', 'NYK': '1610612752',
    'OKC': '1610612760', 'ORL': '1610612753', 'PHI': '1610612755', 'PHX': '1610612756',
    'POR': '1610612757', 'SAC': '1610612758', 'SAS': '1610612759', 'TOR': '1610612761',
    'UTA': '1610612762', 'WAS': '1610612764'
  };
  return teamMap[teamIdentifier] || teamMap[teamIdentifier?.toUpperCase()] || '1610612738';
};

// Convert probability to American odds
const probToAmericanOdds = (prob: number): string => {
  if (prob >= 0.5) {
    return '' + Math.round(-100 * prob / (1 - prob));
  } else {
    return '+' + Math.round(100 * (1 - prob) / prob);
  }
};

// Calculate implied probability from spread
const impliedProbFromSpread = (spread: number): number => {
  // Simplified: spread of -3 ≈ 60% win probability
  return 0.5 + (spread * 0.033);
};

const GameDetailPage: Component<GameDetailPageProps> = (props) => {
  const [mlPrediction, setMlPrediction] = createSignal<any>(null);
  const [q2Prediction, setQ2Prediction] = createSignal<any>(null);
  const [showJson, setShowJson] = createSignal(false);
  const API_BASE = 'https://ol24-production.up.railway.app';

  // Get game from WebSocket
  const [games] = wsService.games;
  const game = () => games().get(props.gameId);

  // Fetch ML prediction for this game
  const fetchPrediction = async () => {
    try {
      const response = await fetch(`${API_BASE}/api/ml/prediction/${props.gameId}`);
      const data = await response.json();
      if (!data.error) {
        setMlPrediction(data);
        
        // If this is Q2 6:00 prediction, store it separately
        if (data.is_q2_6min) {
          setQ2Prediction(data);
        }
      }
    } catch (error) {
      console.error('Error fetching prediction:', error);
    }
  };

  // Auto-refresh prediction every 30 seconds
  createEffect(() => {
    if (game()) {
      fetchPrediction();
      const interval = setInterval(fetchPrediction, 30000);
      onCleanup(() => clearInterval(interval));
    }
  });

  // Check if Q2 6:00 trade window is active
  const isTradeWindowActive = () => {
    if (!game() || !q2Prediction()) return false;
    const g = game()!;
    // Active if Q2 and between 6:00 and 5:00
    if (g.quarter !== 2) return false;
    const timeRemaining = g.time_remaining || g.clock || '';
    const minutes = parseInt(timeRemaining.split(':')[0] || '0');
    return minutes >= 5 && minutes <= 6;
  };

  // Calculate betting metrics
  const bettingMetrics = () => {
    const pred = mlPrediction();
    if (!pred || !pred.point_forecast) return null;

    const forecast = pred.point_forecast;
    const confidence = pred.model_confidence || 0.7;
    const edge = pred.edge_magnitude || 0;

    // Calculate probabilities
    const winProb = impliedProbFromSpread(forecast);
    const loseProb = 1 - winProb;

    // American odds
    const americanOdds = probToAmericanOdds(winProb);
    
    // Expected value
    const ev = (winProb * 100) - (loseProb * 100);

    // Kelly Criterion (simplified)
    const kelly = ((winProb * 2) - 1) / 1;
    const kellyPercent = Math.max(0, Math.min(kelly * 100, 25)); // Cap at 25%

    return {
      forecast,
      winProb,
      loseProb,
      americanOdds,
      ev,
      kelly: kellyPercent,
      confidence,
      edge
    };
  };

  return (
    <div class="min-h-screen bg-gradient-to-b from-black to-gray-900 p-4 md:p-6 lg:p-8">
      <Show when={game()} fallback={
        <div class="flex items-center justify-center min-h-screen">
          <div class="text-white text-xl">Loading game...</div>
        </div>
      }>
        <div class="max-w-7xl mx-auto space-y-6">
          {/* Game Scoreboard */}
          <div class="modern-card">
            <div class="flex flex-col md:flex-row items-center justify-between gap-6">
              {/* Away Team */}
              <div class="text-center flex-1">
                <img 
                  src={`https://cdn.nba.com/logos/nba/${getTeamId(game()!.away_team)}/primary/L/logo.svg`}
                  alt={game()!.away_team}
                  class="w-20 h-20 md:w-24 md:h-24 mx-auto mb-3"
                  onError={(e) => {
                    e.currentTarget.src = 'https://cdn.nba.com/logos/nba/1610612738/primary/L/logo.svg';
                  }}
                />
                <div class="text-white font-bold text-lg md:text-xl mb-2">{game()!.away_team}</div>
                <div class="text-5xl md:text-6xl font-bold text-white">{game()!.score_away}</div>
              </div>

              {/* Game Status */}
              <div class="text-center px-6">
                <Show when={game()!.is_live}>
                  <div class="flex items-center justify-center gap-2 mb-2">
                    <div class="w-3 h-3 bg-red-500 rounded-full live-pulse"></div>
                    <span class="text-red-500 font-bold text-lg uppercase">Live</span>
                  </div>
                  <div class="text-gray-300 text-xl font-semibold">Q{game()!.quarter}</div>
                  <div class="text-gray-400 text-lg">{game()!.time_remaining || game()!.clock}</div>
                </Show>
                <Show when={!game()!.is_live}>
                  <div class="text-gray-500 text-xl font-semibold">Scheduled</div>
                </Show>
              </div>

              {/* Home Team */}
              <div class="text-center flex-1">
                <img 
                  src={`https://cdn.nba.com/logos/nba/${getTeamId(game()!.home_team)}/primary/L/logo.svg`}
                  alt={game()!.home_team}
                  class="w-20 h-20 md:w-24 md:h-24 mx-auto mb-3"
                  onError={(e) => {
                    e.currentTarget.src = 'https://cdn.nba.com/logos/nba/1610612738/primary/L/logo.svg';
                  }}
                />
                <div class="text-white font-bold text-lg md:text-xl mb-2">{game()!.home_team}</div>
                <div class="text-5xl md:text-6xl font-bold text-white">{game()!.score_home}</div>
              </div>
            </div>
          </div>

          {/* 🏆 BIG GOLD Q2 6:00 OFFICIAL TRADE SIGNAL */}
          <Show when={isTradeWindowActive() && q2Prediction()}>
            <div class="modern-card bg-gradient-to-br from-yellow-900/60 via-amber-900/60 to-yellow-900/60 border-4 border-yellow-500 shadow-2xl shadow-yellow-500/50 relative overflow-hidden">
              {/* Animated background */}
              <div class="absolute inset-0 bg-gradient-to-r from-yellow-500/10 via-amber-500/10 to-yellow-500/10 animate-pulse"></div>
              
              <div class="relative z-10 p-6">
                <div class="text-center mb-6">
                  <div class="text-6xl mb-3">🏆</div>
                  <div class="text-yellow-400 text-3xl font-bold mb-2">OFFICIAL TRADE SIGNAL</div>
                  <div class="text-yellow-300 text-xl">Q2 6:00 Mark - Mamba Prediction Window</div>
                  <div class="text-amber-400 text-sm mt-2">⏰ Valid Until Q2 5:00</div>
                </div>

                {/* Main Prediction */}
                <div class="bg-black/40 rounded-2xl p-6 mb-6">
                  <div class="text-center mb-4">
                    <div class="text-gray-300 text-sm mb-2">Final Score Spread Prediction</div>
                    <div class="text-yellow-400 text-6xl font-bold">
                      {q2Prediction().point_forecast >= 0 ? '+' : ''}{q2Prediction().point_forecast.toFixed(1)}
                    </div>
                    <div class="text-amber-400 text-sm mt-2">
                      90% Confidence Interval: [{q2Prediction().interval_lower.toFixed(1)}, {q2Prediction().interval_upper.toFixed(1)}]
                    </div>
                  </div>

                  {/* Betting Metrics */}
                  <Show when={bettingMetrics()}>
                    {(metrics) => (
                      <div class="grid grid-cols-2 md:grid-cols-4 gap-4 mt-6">
                        {/* Win Probability */}
                        <div class="bg-yellow-500/20 rounded-lg p-4 text-center border border-yellow-500/30">
                          <div class="text-yellow-300 text-xs mb-1">Win Probability</div>
                          <div class="text-yellow-400 text-2xl font-bold">{(metrics().winProb * 100).toFixed(1)}%</div>
                        </div>

                        {/* American Odds */}
                        <div class="bg-yellow-500/20 rounded-lg p-4 text-center border border-yellow-500/30">
                          <div class="text-yellow-300 text-xs mb-1">American Odds</div>
                          <div class="text-yellow-400 text-2xl font-bold">{metrics().americanOdds}</div>
                        </div>

                        {/* Expected Value */}
                        <div class="bg-yellow-500/20 rounded-lg p-4 text-center border border-yellow-500/30">
                          <div class="text-yellow-300 text-xs mb-1">Expected Value</div>
                          <div class="text-yellow-400 text-2xl font-bold">{metrics().ev >= 0 ? '+' : ''}{metrics().ev.toFixed(1)}%</div>
                        </div>

                        {/* Kelly Criterion */}
                        <div class="bg-yellow-500/20 rounded-lg p-4 text-center border border-yellow-500/30">
                          <div class="text-yellow-300 text-xs mb-1">Kelly Bet Size</div>
                          <div class="text-yellow-400 text-2xl font-bold">{metrics().kelly.toFixed(1)}%</div>
                        </div>
                      </div>
                    )}
                  </Show>

                  {/* Edge Detection */}
                  <Show when={q2Prediction().edge_detected}>
                    <div class="mt-4 bg-green-500/30 border-2 border-green-400 rounded-xl p-4 text-center">
                      <div class="text-green-300 text-lg font-bold mb-1">🎯 EDGE DETECTED</div>
                      <div class="text-green-400 text-3xl font-bold">
                        +{q2Prediction().edge_magnitude?.toFixed(1)} points
                      </div>
                      <div class="text-green-300 text-sm mt-1">Market inefficiency confirmed</div>
                    </div>
                  </Show>
                </div>

                {/* Trading Instructions */}
                <div class="bg-yellow-500/10 border border-yellow-500/30 rounded-lg p-4">
                  <div class="text-yellow-400 text-sm font-semibold mb-2">Trading Parameters:</div>
                  <div class="grid grid-cols-1 md:grid-cols-2 gap-2 text-xs text-yellow-300">
                    <div>• Model: MAMBA_MENTALITY v1.0.0</div>
                    <div>• MAE: 5.39 points</div>
                    <div>• Confidence: {((q2Prediction().model_confidence || 0.7) * 100).toFixed(0)}%</div>
                    <div>• Coverage: 90% interval</div>
                  </div>
                </div>
              </div>
            </div>
          </Show>

          {/* Continuous ML Predictions (Small Box - Every 30s) */}
          <Show when={game()!.is_live && mlPrediction() && !isTradeWindowActive()}>
            <div class="modern-card bg-gray-800/50 border-purple-500/30">
              <div class="flex items-center justify-between mb-4">
                <div class="flex items-center gap-2">
                  <div class="w-2 h-2 bg-purple-500 rounded-full animate-pulse"></div>
                  <h3 class="text-lg font-bold text-white">Real-Time ML Prediction</h3>
                </div>
                <div class="text-xs text-gray-400">Updates every 30s</div>
              </div>

              <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
                {/* Forecast */}
                <div class="text-center">
                  <div class="text-gray-400 text-xs mb-1">Spread Forecast</div>
                  <div class="text-purple-400 text-3xl font-bold">
                    {mlPrediction().point_forecast >= 0 ? '+' : ''}{mlPrediction().point_forecast.toFixed(1)}
                  </div>
                  <div class="text-gray-500 text-xs mt-1">
                    [{mlPrediction().interval_lower.toFixed(1)}, {mlPrediction().interval_upper.toFixed(1)}]
                  </div>
                </div>

                {/* Probability */}
                <div class="text-center">
                  <div class="text-gray-400 text-xs mb-1">Win Probability</div>
                  <div class="text-blue-400 text-3xl font-bold">
                    {bettingMetrics() ? (bettingMetrics()!.winProb * 100).toFixed(1) : 'N/A'}%
                  </div>
                  <div class="text-gray-500 text-xs mt-1">
                    Odds: {bettingMetrics() ? bettingMetrics()!.americanOdds : 'N/A'}
                  </div>
                </div>

                {/* Confidence */}
                <div class="text-center">
                  <div class="text-gray-400 text-xs mb-1">Model Confidence</div>
                  <div class="text-green-400 text-3xl font-bold">
                    {((mlPrediction().model_confidence || 0.7) * 100).toFixed(0)}%
                  </div>
                  <div class="text-gray-500 text-xs mt-1">
                    MAE: 5.39 pts
                  </div>
                </div>
              </div>

              {/* Edge */}
              <Show when={mlPrediction().edge_detected}>
                <div class="mt-4 pt-4 border-t border-gray-700 text-center">
                  <span class="text-green-400 text-sm font-semibold">Edge Detected: </span>
                  <span class="text-green-300 text-sm">+{mlPrediction().edge_magnitude?.toFixed(1)} points</span>
                </div>
              </Show>
            </div>
          </Show>

          {/* Waiting for Q2 6:00 Message */}
          <Show when={game()!.is_live && !mlPrediction() && game()!.quarter < 2}>
            <div class="modern-card bg-blue-900/20 border-blue-500/30 text-center py-8">
              <div class="text-4xl mb-3">⏰</div>
              <div class="text-blue-400 text-xl font-bold mb-2">Waiting for Q2 6:00...</div>
              <div class="text-gray-400">Mamba model requires 18 minutes of play-by-play data</div>
              <div class="text-gray-500 text-sm mt-2">Current: Q{game()!.quarter} {game()!.time_remaining}</div>
            </div>
          </Show>

          {/* 🎯 MAMBA LIVE PATTERN VISUALIZATION */}
          <Show when={game()!.is_live}>
            <MambaLiveWidget gameId={props.gameId} />
          </Show>

          {/* 📊 LIVE SNAPSHOTS ANALYSIS */}
          <Show when={game()!.is_live}>
            <div class="mb-6">
              <LiveSnapshotsWidget gameId={props.gameId} />
            </div>
          </Show>

          {/* 📈 WIN PROBABILITY TIMELINE */}
          <Show when={game()!.is_live}>
            <div class="mb-6">
              <WinProbabilityWidget gameId={props.gameId} />
            </div>
          </Show>

          {/* 🧠 MAMBA FEATURES */}
          <Show when={game()!.is_live}>
            <div class="mb-6">
              <MambaFeaturesWidget gameId={props.gameId} />
            </div>
          </Show>

          {/* BACKEND STATUS - What's happening right now */}
          <div class="modern-card bg-gradient-to-br from-indigo-950/50 to-purple-950/50 border-indigo-500/30 mb-6">
            <h2 class="text-xl font-bold text-white mb-4">🔬 Backend ML Feed Status</h2>
            
            <div class="space-y-3">
              {/* ESPN API Status */}
              <div class="flex items-center justify-between p-3 bg-black/30 rounded-lg">
                <div class="flex items-center gap-2">
                  <div class="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                  <span class="text-white font-medium">ESPN API</span>
                </div>
                <span class="text-green-400 text-sm">1-second updates ⚡</span>
              </div>

              {/* ML Model Status */}
              <div class="flex items-center justify-between p-3 bg-black/30 rounded-lg">
                <div class="flex items-center gap-2">
                  <div class={`w-2 h-2 rounded-full ${mlPrediction() ? 'bg-purple-500 animate-pulse' : 'bg-gray-500'}`}></div>
                  <span class="text-white font-medium">Mamba ML Model</span>
                </div>
                <span class={`text-sm ${mlPrediction() ? 'text-purple-400' : 'text-gray-500'}`}>
                  {mlPrediction() ? 'Predicting 🤖' : 'Waiting for Q2 6:00'}
                </span>
              </div>

              {/* Prediction Frequency */}
              <div class="flex items-center justify-between p-3 bg-black/30 rounded-lg">
                <div class="flex items-center gap-2">
                  <div class="w-2 h-2 bg-blue-500 rounded-full"></div>
                  <span class="text-white font-medium">Update Frequency</span>
                </div>
                <span class="text-blue-400 text-sm">
                  {mlPrediction() ? 'Every 30 seconds' : 'Not running'}
                </span>
              </div>

              {/* Features Extracted */}
              <div class="flex items-center justify-between p-3 bg-black/30 rounded-lg">
                <div class="flex items-center gap-2">
                  <div class="w-2 h-2 bg-yellow-500 rounded-full"></div>
                  <span class="text-white font-medium">Features</span>
                </div>
                <span class="text-yellow-400 text-sm">33 Mamba features from PBP</span>
              </div>

              {/* API Endpoint */}
              <div class="p-3 bg-black/30 rounded-lg">
                <div class="text-xs text-gray-500 mb-1">API Endpoint:</div>
                <div class="text-xs text-blue-400 font-mono break-all">
                  GET {API_BASE}/api/ml/prediction/{props.gameId}
                </div>
              </div>
            </div>
          </div>

          {/* API Testing Panel */}
          <div class="modern-card">
            <div class="flex items-center justify-between mb-4">
              <h2 class="text-xl font-bold text-white">📊 Live JSON Data Inspector</h2>
              <button
                onClick={() => setShowJson(!showJson())}
                class="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-xl text-sm font-semibold transition-all"
              >
                {showJson() ? 'Hide JSON' : 'Show Raw JSON'}
              </button>
            </div>

            <Show when={showJson()}>
              <div class="space-y-4">
                {/* Connection Status */}
                <div class="flex items-center gap-3 p-3 bg-green-900/20 border border-green-700/30 rounded-lg">
                  <div class="w-3 h-3 bg-green-500 rounded-full live-pulse"></div>
                  <span class="text-green-400 font-medium">Connected to NBA API</span>
                </div>

                {/* Live Game Data */}
                <div class="bg-gray-900 border border-gray-700 rounded-lg p-4 overflow-x-auto">
                  <div class="text-sm text-gray-400 mb-2 font-semibold">Game Data (WebSocket):</div>
                  <pre class="text-xs text-gray-300">
{JSON.stringify(game(), null, 2)}
                  </pre>
                </div>

                {/* ML Prediction Data */}
                <Show when={mlPrediction()}>
                  <div class="bg-gray-900 border border-gray-700 rounded-lg p-4 overflow-x-auto">
                    <div class="text-sm text-gray-400 mb-2 font-semibold">ML Prediction (30s updates):</div>
                    <pre class="text-xs text-gray-300">
{JSON.stringify(mlPrediction(), null, 2)}
                    </pre>
                  </div>
                </Show>

                {/* Q2 6:00 Prediction */}
                <Show when={q2Prediction()}>
                  <div class="bg-yellow-900/20 border border-yellow-700/30 rounded-lg p-4 overflow-x-auto">
                    <div class="text-sm text-yellow-400 mb-2 font-semibold">Q2 6:00 Official Prediction:</div>
                    <pre class="text-xs text-yellow-300">
{JSON.stringify(q2Prediction(), null, 2)}
                    </pre>
                  </div>
                </Show>

                {/* Refresh Button */}
                <button
                  onClick={fetchPrediction}
                  class="w-full px-4 py-2 bg-gray-700 hover:bg-gray-600 text-white rounded-lg font-medium transition-all"
                >
                  Refresh Prediction Data
                </button>
              </div>
            </Show>
          </div>
        </div>
      </Show>
    </div>
  );
};

export default GameDetailPage;
