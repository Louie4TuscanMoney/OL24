import { type Component, createSignal, createEffect, onCleanup, Show } from 'solid-js';
import { wsService } from '../services/websocket';
import MLPredictionBox from './MLPredictionBox';

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

const GameDetailPage: Component<GameDetailPageProps> = (props) => {
  const [mlPrediction, setMlPrediction] = createSignal<any>(null);
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

  // Check if we're in Q2 6:00 trade window
  const isTradeWindow = () => {
    const pred = mlPrediction();
    if (!pred) return false;
    return pred.is_q2_6min && pred.quarter === 2 && pred.time_remaining >= '05:00' && pred.time_remaining <= '06:00';
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
                    console.error('Logo failed to load:', game()!.away_team, getTeamId(game()!.away_team));
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
                  <div class="text-gray-500 text-xl font-semibold">Final</div>
                </Show>
              </div>

              {/* Home Team */}
              <div class="text-center flex-1">
                <img 
                  src={`https://cdn.nba.com/logos/nba/${getTeamId(game()!.home_team)}/primary/L/logo.svg`}
                  alt={game()!.home_team}
                  class="w-20 h-20 md:w-24 md:h-24 mx-auto mb-3"
                  onError={(e) => {
                    console.error('Logo failed to load:', game()!.home_team, getTeamId(game()!.home_team));
                    e.currentTarget.src = 'https://cdn.nba.com/logos/nba/1610612738/primary/L/logo.svg';
                  }}
                />
                <div class="text-white font-bold text-lg md:text-xl mb-2">{game()!.home_team}</div>
                <div class="text-5xl md:text-6xl font-bold text-white">{game()!.score_home}</div>
              </div>
            </div>
          </div>

          {/* Q2 6:00 TRADE SIGNAL (Special Green Box) */}
          <Show when={isTradeWindow()}>
            <div class="modern-card bg-gradient-to-br from-green-900/50 to-emerald-900/50 border-2 border-green-500 shadow-2xl shadow-green-500/30 animate-pulse">
              <div class="text-center py-6">
                <div class="text-4xl mb-4">🎯</div>
                <div class="text-green-400 text-2xl font-bold mb-2">TRADE SIGNAL ACTIVE</div>
                <div class="text-green-300 text-lg mb-4">Q2 6:00 Mark - Optimal Prediction Window</div>
                <div class="text-gray-300 text-sm">Valid until Q2 5:00</div>
              </div>
            </div>
          </Show>

          {/* ML Prediction Box (Every 30 Seconds) */}
          <Show when={game()!.is_live}>
            <div>
              <h2 class="text-2xl font-bold text-white mb-4">Real-Time ML Prediction (30s updates)</h2>
              <MLPredictionBox 
                gameId={props.gameId} 
                prediction={mlPrediction()}
                isQ26Min={isTradeWindow()}
              />
            </div>
          </Show>

          {/* API Testing Panel */}
          <div class="modern-card">
            <div class="flex items-center justify-between mb-4">
              <h2 class="text-xl font-bold text-white">API Testing & Connection</h2>
              <button
                onClick={() => setShowJson(!showJson())}
                class="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-xl text-sm font-semibold"
              >
                {showJson() ? 'Hide JSON' : 'Show Live JSON'}
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
                <div class="bg-gray-900 border border-gray-700 rounded-lg p-4">
                  <div class="text-sm text-gray-400 mb-2 font-semibold">Game Data:</div>
                  <pre class="text-xs text-gray-300 overflow-x-auto">
                    {JSON.stringify(game(), null, 2)}
                  </pre>
                </div>

                {/* ML Prediction Data */}
                <Show when={mlPrediction()}>
                  <div class="bg-gray-900 border border-gray-700 rounded-lg p-4">
                    <div class="text-sm text-gray-400 mb-2 font-semibold">ML Prediction:</div>
                    <pre class="text-xs text-gray-300 overflow-x-auto">
                      {JSON.stringify(mlPrediction(), null, 2)}
                    </pre>
                  </div>
                </Show>

                {/* Refresh Button */}
                <button
                  onClick={fetchPrediction}
                  class="w-full px-4 py-2 bg-gray-700 hover:bg-gray-600 text-white rounded-lg font-medium"
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
