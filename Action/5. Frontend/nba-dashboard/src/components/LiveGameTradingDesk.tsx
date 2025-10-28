import { Component, createSignal, onMount, onCleanup, For, Show } from 'solid-js';
import { Line } from 'solid-chartjs';
import { Chart, Title, Tooltip, Legend, Colors } from 'chart.js';

// Register Chart.js components
Chart.register(Title, Tooltip, Legend, Colors);

interface GameScore {
  game_id: string;
  home_team: string;
  away_team: string;
  home_score: number;
  away_score: number;
  quarter: number;
  time_remaining: string;
  game_status: string;
}

interface SpreadLevel {
  spread: number;
  price: number;
  size: number;
  side: 'home' | 'away';
}

interface Props {
  gameId: string;
  onClose: () => void;
}

const LiveGameTradingDesk: Component<Props> = (props) => {
  const [game, setGame] = createSignal<GameScore | null>(null);
  const [scoreHistory, setScoreHistory] = createSignal<{time: string, homeDiff: number}[]>([]);
  const [spreadLadder, setSpreadLadder] = createSignal<SpreadLevel[]>([]);
  
  const API_BASE = 'https://ol24-production.up.railway.app';

  // Update every 1 second
  let interval: number;

  const fetchGameData = async () => {
    try {
      // Fetch live game data
      const response = await fetch(`${API_BASE}/ws`); // Will need dedicated endpoint
      // For now, using WebSocket data
      
      // Mock data for now - replace with real data
      const currentDiff = (game()?.home_score || 0) - (game()?.away_score || 0);
      const now = new Date().toLocaleTimeString();
      
      setScoreHistory(prev => [
        ...prev.slice(-60), // Keep last 60 data points (1 minute)
        { time: now, homeDiff: currentDiff }
      ]);
      
      // Generate spread ladder (mock - will connect to BetOnline)
      const spreadLadderData: SpreadLevel[] = [];
      for (let i = -10; i <= 10; i += 0.5) {
        spreadLadderData.push({
          spread: i,
          price: -110 + (Math.random() * 10 - 5), // Mock pricing
          size: Math.floor(Math.random() * 1000) + 100,
          side: i < currentDiff ? 'away' : 'home'
        });
      }
      setSpreadLadder(spreadLadderData);
      
    } catch (error) {
      console.error('Error fetching game data:', error);
    }
  };

  onMount(() => {
    fetchGameData();
    interval = setInterval(fetchGameData, 1000) as unknown as number; // 1-second updates!
  });

  onCleanup(() => {
    clearInterval(interval);
  });

  const chartData = () => ({
    labels: scoreHistory().map(d => d.time),
    datasets: [{
      label: 'Score Differential (Home - Away)',
      data: scoreHistory().map(d => d.homeDiff),
      borderColor: 'rgb(59, 130, 246)',
      backgroundColor: 'rgba(59, 130, 246, 0.1)',
      tension: 0.4,
      fill: true
    }]
  });

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    scales: {
      y: {
        grid: { color: 'rgba(255, 255, 255, 0.1)' },
        ticks: { color: '#9CA3AF' }
      },
      x: {
        grid: { color: 'rgba(255, 255, 255, 0.1)' },
        ticks: { color: '#9CA3AF', maxTicksLimit: 10 }
      }
    },
    plugins: {
      legend: { labels: { color: '#FFF' } }
    }
  };

  return (
    <div class="fixed inset-0 z-50 bg-black bg-opacity-95 overflow-auto">
      {/* Close Button */}
      <button
        onClick={props.onClose}
        class="absolute top-4 right-4 text-white text-3xl hover:text-red-500 z-50"
      >
        ✕
      </button>

      <div class="min-h-screen p-6">
        <div class="max-w-[95vw] mx-auto">
          {/* Game Header */}
          <div class="text-center mb-6">
            <h1 class="text-4xl font-bold text-white mb-2">
              {game()?.away_team || 'AWAY'} @ {game()?.home_team || 'HOME'}
            </h1>
            <div class="text-2xl text-gray-400">
              Q{game()?.quarter || 1} {game()?.time_remaining || '12:00'}
            </div>
          </div>

          <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* LEFT: Score & Graph */}
            <div class="lg:col-span-2 space-y-6">
              {/* Live Score */}
              <div class="bg-gray-900 rounded-lg p-8 border border-gray-800">
                <div class="grid grid-cols-2 gap-8">
                  {/* Away Score */}
                  <div class="text-center">
                    <div class="text-gray-400 text-xl mb-2">{game()?.away_team || 'AWAY'}</div>
                    <div class="text-7xl font-bold text-white">{game()?.away_score || 0}</div>
                  </div>
                  {/* Home Score */}
                  <div class="text-center">
                    <div class="text-gray-400 text-xl mb-2">{game()?.home_team || 'HOME'}</div>
                    <div class="text-7xl font-bold text-white">{game()?.home_score || 0}</div>
                  </div>
                </div>
                
                {/* Score Differential */}
                <div class="text-center mt-6 pt-6 border-t border-gray-800">
                  <div class="text-gray-400 text-sm mb-1">SCORE DIFFERENTIAL</div>
                  <div class={`text-4xl font-bold ${
                    ((game()?.home_score || 0) - (game()?.away_score || 0)) > 0 
                      ? 'text-green-400' 
                      : 'text-red-400'
                  }`}>
                    {((game()?.home_score || 0) - (game()?.away_score || 0)) > 0 ? '+' : ''}
                    {(game()?.home_score || 0) - (game()?.away_score || 0)}
                  </div>
                </div>
              </div>

              {/* Scoring Graph */}
              <div class="bg-gray-900 rounded-lg p-6 border border-gray-800">
                <h2 class="text-white text-2xl font-bold mb-4">📈 Live Score Differential</h2>
                <div class="h-[400px]">
                  <Show when={scoreHistory().length > 0}>
                    <Line data={chartData()} options={chartOptions} />
                  </Show>
                  <Show when={scoreHistory().length === 0}>
                    <div class="flex items-center justify-center h-full text-gray-500">
                      Collecting data...
                    </div>
                  </Show>
                </div>
              </div>
            </div>

            {/* RIGHT: Pricing Ladder (Order Book Style) */}
            <div class="bg-gray-900 rounded-lg p-6 border border-gray-800">
              <h2 class="text-white text-2xl font-bold mb-4">💰 Spread Pricing Ladder</h2>
              <div class="text-gray-400 text-sm mb-4 text-center">
                Live Market Depth (BetOnline)
              </div>

              {/* Current Line Indicator */}
              <div class="bg-blue-900/30 border border-blue-500 rounded p-3 mb-4 text-center">
                <div class="text-blue-400 text-xs mb-1">CURRENT SPREAD</div>
                <div class="text-white text-xl font-bold">
                  {((game()?.home_score || 0) - (game()?.away_score || 0)).toFixed(1)}
                </div>
              </div>

              {/* Order Book Style Ladder */}
              <div class="space-y-px">
                <For each={spreadLadder().slice().reverse()}>
                  {(level) => {
                    const currentDiff = (game()?.home_score || 0) - (game()?.away_score || 0);
                    const isAtMarket = Math.abs(level.spread - currentDiff) < 0.5;
                    
                    return (
                      <div 
                        class={`grid grid-cols-3 gap-2 p-2 text-sm ${
                          isAtMarket ? 'bg-yellow-900/30 border-l-4 border-yellow-500' :
                          level.side === 'home' ? 'bg-green-900/20 hover:bg-green-900/40' : 
                          'bg-red-900/20 hover:bg-red-900/40'
                        } transition-all cursor-pointer`}
                      >
                        {/* Spread */}
                        <div class={`font-mono font-bold ${
                          level.side === 'home' ? 'text-green-400' : 'text-red-400'
                        }`}>
                          {level.spread > 0 ? '+' : ''}{level.spread.toFixed(1)}
                        </div>
                        
                        {/* Price */}
                        <div class="text-white text-center font-semibold">
                          {level.price > 0 ? '+' : ''}{level.price.toFixed(0)}
                        </div>
                        
                        {/* Size */}
                        <div class="text-gray-400 text-right text-xs">
                          ${(level.size).toLocaleString()}
                        </div>
                      </div>
                    );
                  }}
                </For>
              </div>

              {/* Legend */}
              <div class="mt-6 pt-4 border-t border-gray-800 grid grid-cols-2 gap-2 text-xs">
                <div class="flex items-center gap-2">
                  <div class="w-3 h-3 bg-green-600 rounded"></div>
                  <span class="text-gray-400">HOME (Ask)</span>
                </div>
                <div class="flex items-center gap-2">
                  <div class="w-3 h-3 bg-red-600 rounded"></div>
                  <span class="text-gray-400">AWAY (Bid)</span>
                </div>
              </div>

              {/* Live Update Indicator */}
              <div class="mt-4 text-center">
                <div class="inline-flex items-center gap-2 px-3 py-1 bg-green-900/30 border border-green-500 rounded-full">
                  <div class="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                  <span class="text-green-400 text-xs font-semibold">LIVE · 1s Updates</span>
                </div>
              </div>
            </div>
          </div>

          {/* Bottom: Additional Stats */}
          <div class="mt-6 grid grid-cols-1 md:grid-cols-3 gap-6">
            {/* Quarter-by-Quarter */}
            <div class="bg-gray-900 rounded-lg p-6 border border-gray-800">
              <h3 class="text-white text-xl font-bold mb-4">📊 Quarter Breakdown</h3>
              {/* Will add quarter-by-quarter scores */}
              <div class="text-gray-500 text-sm">
                Q1, Q2, Q3, Q4 scores coming soon...
              </div>
            </div>

            {/* Team Stats */}
            <div class="bg-gray-900 rounded-lg p-6 border border-gray-800">
              <h3 class="text-white text-xl font-bold mb-4">📈 Team Stats</h3>
              <div class="text-gray-500 text-sm">
                FG%, 3P%, Rebounds, etc.
              </div>
            </div>

            {/* Betting Insights */}
            <div class="bg-gray-900 rounded-lg p-6 border border-gray-800">
              <h3 class="text-white text-xl font-bold mb-4">🎯 ML Insights</h3>
              <div class="text-gray-500 text-sm">
                Model prediction, confidence, edge
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default LiveGameTradingDesk;

