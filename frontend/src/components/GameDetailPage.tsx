import { type Component, createSignal, createEffect, onCleanup, Show } from 'solid-js';

interface GameDetailPageProps {
  gameId: string;
}

const GameDetailPage: Component<GameDetailPageProps> = (props) => {
  const [gameData, setGameData] = createSignal<any>(null);
  const [liveJson, setLiveJson] = createSignal<any>(null);
  const [showJson, setShowJson] = createSignal(false);
  const [loading, setLoading] = createSignal(true);
  const [countdown, setCountdown] = createSignal('');
  const API_BASE = 'https://ol24-production.up.railway.app';

  // Fetch game data
  const fetchGameData = async () => {
    try {
      const response = await fetch(`${API_BASE}/api/game/${props.gameId}/live-data`);
      const data = await response.json();
      setGameData(data);
      setLiveJson(data);
      setLoading(false);
      
      // Calculate countdown if game is scheduled
      if (data.status === 'Scheduled' && data.gameTime) {
        updateCountdown(data.gameTime);
      }
    } catch (error) {
      console.error('Error fetching game data:', error);
      setLoading(false);
    }
  };

  // Update countdown timer
  const updateCountdown = (gameTime: string) => {
    const now = new Date().getTime();
    const gameDate = new Date(gameTime).getTime();
    const distance = gameDate - now;

    if (distance < 0) {
      setCountdown('Game Started');
      return;
    }

    const days = Math.floor(distance / (1000 * 60 * 60 * 24));
    const hours = Math.floor((distance % (1000 * 60 * 60 * 24)) / (1000 * 60 * 60));
    const minutes = Math.floor((distance % (1000 * 60 * 60)) / (1000 * 60));
    const seconds = Math.floor((distance % (1000 * 60)) / 1000);

    if (days > 0) {
      setCountdown(`${days}d ${hours}h ${minutes}m`);
    } else if (hours > 0) {
      setCountdown(`${hours}h ${minutes}m ${seconds}s`);
    } else {
      setCountdown(`${minutes}m ${seconds}s`);
    }
  };

  // Auto-refresh for live games
  createEffect(() => {
    fetchGameData();
    const interval = setInterval(() => {
      fetchGameData();
    }, 5000); // Update every 5 seconds

    onCleanup(() => clearInterval(interval));
  });

  // Countdown timer
  createEffect(() => {
    const game = gameData();
    if (game && game.status === 'Scheduled' && game.gameTime) {
      const timer = setInterval(() => {
        updateCountdown(game.gameTime);
      }, 1000);
      
      onCleanup(() => clearInterval(timer));
    }
  });

  return (
    <div class="min-h-screen bg-black p-4 md:p-6 lg:p-8">
      <Show when={!loading()} fallback={
        <div class="flex items-center justify-center min-h-screen">
          <div class="text-white text-xl">Loading game...</div>
        </div>
      }>
        <div class="max-w-7xl mx-auto space-y-6">
          {/* Game Header */}
          <div class="modern-card">
            <div class="flex flex-col md:flex-row items-center justify-between gap-4">
              {/* Teams */}
              <div class="flex items-center gap-6 flex-1">
                {/* Away Team */}
                <div class="text-center">
                  <img 
                    src={`https://cdn.nba.com/logos/nba/${gameData()?.awayTeam?.id}/primary/L/logo.svg`}
                    alt={gameData()?.awayTeam?.name}
                    class="w-16 h-16 md:w-20 md:h-20 mx-auto mb-2"
                  />
                  <div class="text-white font-semibold text-sm md:text-base">{gameData()?.awayTeam?.name}</div>
                  <div class="text-4xl md:text-5xl font-bold text-white mt-2">{gameData()?.awayTeam?.score || '-'}</div>
                </div>

                {/* VS / Status */}
                <div class="text-center px-4">
                  <Show when={gameData()?.status === 'Live'}>
                    <div class="text-red-500 text-sm font-semibold mb-1">LIVE</div>
                    <div class="text-gray-400 text-sm">{gameData()?.quarter}</div>
                    <div class="text-gray-400 text-sm">{gameData()?.gameClock}</div>
                  </Show>
                  <Show when={gameData()?.status === 'Scheduled'}>
                    <div class="text-blue-500 text-sm font-semibold mb-1">SCHEDULED</div>
                    <div class="text-2xl md:text-3xl font-bold text-white countdown-pulse">{countdown()}</div>
                    <div class="text-gray-400 text-xs mt-1">{new Date(gameData()?.gameTime).toLocaleString()}</div>
                  </Show>
                  <Show when={gameData()?.status === 'Final'}>
                    <div class="text-gray-500 text-sm font-semibold">FINAL</div>
                  </Show>
                </div>

                {/* Home Team */}
                <div class="text-center">
                  <img 
                    src={`https://cdn.nba.com/logos/nba/${gameData()?.homeTeam?.id}/primary/L/logo.svg`}
                    alt={gameData()?.homeTeam?.name}
                    class="w-16 h-16 md:w-20 md:h-20 mx-auto mb-2"
                  />
                  <div class="text-white font-semibold text-sm md:text-base">{gameData()?.homeTeam?.name}</div>
                  <div class="text-4xl md:text-5xl font-bold text-white mt-2">{gameData()?.homeTeam?.score || '-'}</div>
                </div>
              </div>

              {/* Game Info */}
              <div class="text-right text-sm text-gray-400">
                <div>{gameData()?.arena}</div>
                <div>{gameData()?.tv}</div>
              </div>
            </div>
          </div>

          {/* Box Score Tabs */}
          <div class="modern-card">
            <h2 class="text-2xl font-bold text-white mb-4">Box Score</h2>
            <div class="overflow-x-auto">
              <table class="w-full text-sm">
                <thead>
                  <tr class="border-b border-gray-700">
                    <th class="text-left text-gray-400 font-semibold pb-2">Player</th>
                    <th class="text-center text-gray-400 font-semibold pb-2">MIN</th>
                    <th class="text-center text-gray-400 font-semibold pb-2">PTS</th>
                    <th class="text-center text-gray-400 font-semibold pb-2">REB</th>
                    <th class="text-center text-gray-400 font-semibold pb-2">AST</th>
                    <th class="text-center text-gray-400 font-semibold pb-2">FG</th>
                    <th class="text-center text-gray-400 font-semibold pb-2">3PT</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td colspan="7" class="text-center text-gray-500 py-8">
                      Box score will appear when game starts
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>

          {/* ML Prediction (when available) */}
          <Show when={gameData()?.mlPrediction}>
            <div class="modern-card bg-gradient-to-br from-purple-900/20 to-blue-900/20">
              <h2 class="text-2xl font-bold text-white mb-4">ML Prediction (Q2 6:00)</h2>
              <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div class="text-center">
                  <div class="text-gray-400 text-sm mb-1">Win Probability</div>
                  <div class="text-3xl font-bold text-green-400">{gameData()?.mlPrediction?.winProb}%</div>
                </div>
                <div class="text-center">
                  <div class="text-gray-400 text-sm mb-1">Predicted Spread</div>
                  <div class="text-3xl font-bold text-blue-400">{gameData()?.mlPrediction?.spread}</div>
                </div>
                <div class="text-center">
                  <div class="text-gray-400 text-sm mb-1">Confidence</div>
                  <div class="text-3xl font-bold text-purple-400">{gameData()?.mlPrediction?.confidence}%</div>
                </div>
              </div>
            </div>
          </Show>

          {/* API Testing Panel */}
          <div class="modern-card">
            <div class="flex items-center justify-between mb-4">
              <h2 class="text-xl font-bold text-white">API Testing & Connection</h2>
              <button
                onClick={() => setShowJson(!showJson())}
                class="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg text-sm font-medium"
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

                {/* Live JSON Data */}
                <div class="bg-gray-900 border border-gray-700 rounded-lg p-4 overflow-x-auto">
                  <pre class="text-xs text-gray-300">
                    {JSON.stringify(liveJson(), null, 2)}
                  </pre>
                </div>

                {/* Refresh Button */}
                <button
                  onClick={fetchGameData}
                  class="w-full px-4 py-2 bg-gray-700 hover:bg-gray-600 text-white rounded-lg font-medium"
                >
                  Refresh API Data
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

