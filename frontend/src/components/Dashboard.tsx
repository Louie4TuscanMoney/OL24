/**
 * Main Dashboard Component
 * Real-time NBA predictions and betting dashboard
 * Professional sports trading interface
 */

import { type Component, For, Show, onMount, onCleanup } from 'solid-js';
import { wsService } from '../services/websocket';
import GameCardExpanded from './GameCardExpanded';
import SystemStatus from './SystemStatus';

interface DashboardProps {
  onGameClick?: (gameId: string) => void;
  onTeamClick?: (teamAbbr: string) => void;
}

// Team name to abbreviation mapping
const teamNameToAbbr: Record<string, string> = {
  'Boston Celtics': 'BOS', 'Brooklyn Nets': 'BKN', 'New York Knicks': 'NYK',
  'Philadelphia 76ers': 'PHI', 'Toronto Raptors': 'TOR',
  'Chicago Bulls': 'CHI', 'Cleveland Cavaliers': 'CLE', 'Detroit Pistons': 'DET',
  'Indiana Pacers': 'IND', 'Milwaukee Bucks': 'MIL',
  'Atlanta Hawks': 'ATL', 'Charlotte Hornets': 'CHA', 'Miami Heat': 'MIA',
  'Orlando Magic': 'ORL', 'Washington Wizards': 'WAS',
  'Denver Nuggets': 'DEN', 'Minnesota Timberwolves': 'MIN', 'Oklahoma City Thunder': 'OKC',
  'Portland Trail Blazers': 'POR', 'Utah Jazz': 'UTA',
  'Golden State Warriors': 'GSW', 'LA Clippers': 'LAC', 'Los Angeles Lakers': 'LAL',
  'Phoenix Suns': 'PHX', 'Sacramento Kings': 'SAC',
  'Dallas Mavericks': 'DAL', 'Houston Rockets': 'HOU', 'Memphis Grizzlies': 'MEM',
  'New Orleans Pelicans': 'NOP', 'San Antonio Spurs': 'SAS'
};

// Team ID for logos (simplified - using team names)
const getTeamLogo = (teamName: string) => {
  const teamIds: Record<string, string> = {
    'Boston Celtics': '1610612738', 'Brooklyn Nets': '1610612751', 'New York Knicks': '1610612752',
    'Philadelphia 76ers': '1610612755', 'Toronto Raptors': '1610612761',
    'Chicago Bulls': '1610612741', 'Cleveland Cavaliers': '1610612739', 'Detroit Pistons': '1610612765',
    'Indiana Pacers': '1610612754', 'Milwaukee Bucks': '1610612749',
    'Atlanta Hawks': '1610612737', 'Charlotte Hornets': '1610612766', 'Miami Heat': '1610612748',
    'Orlando Magic': '1610612753', 'Washington Wizards': '1610612764',
    'Denver Nuggets': '1610612743', 'Minnesota Timberwolves': '1610612750', 'Oklahoma City Thunder': '1610612760',
    'Portland Trail Blazers': '1610612757', 'Utah Jazz': '1610612762',
    'Golden State Warriors': '1610612744', 'LA Clippers': '1610612746', 'Los Angeles Lakers': '1610612747',
    'Phoenix Suns': '1610612756', 'Sacramento Kings': '1610612758',
    'Dallas Mavericks': '1610612742', 'Houston Rockets': '1610612745', 'Memphis Grizzlies': '1610612763',
    'New Orleans Pelicans': '1610612740', 'San Antonio Spurs': '1610612759'
  };
  return teamIds[teamName] || '1610612738';
};

const Dashboard: Component<DashboardProps> = (props) => {
  // Get signals from WebSocket service (reactive!)
  const [connected] = wsService.connected;
  const [games] = wsService.games;
  const [patterns] = wsService.patterns;
  const [predictions] = wsService.predictions;
  const [edges] = wsService.edges;
  const [recommendations] = wsService.recommendations;

  // Connect on mount
  onMount(() => {
    wsService.connect();
  });

  // Disconnect on unmount
  onCleanup(() => {
    wsService.disconnect();
  });

  // Convert Map to Array for iteration
  const gamesList = () => Array.from(games().values());

  // Live games
  const liveGames = () => gamesList().filter(g => g.is_live);

  // Upcoming games (not live)
  const upcomingGames = () => gamesList().filter(g => !g.is_live);

  // Total edges detected
  const edgesCount = () => 
    Array.from(edges().values()).filter(e => e.has_edge).length;
  
  // Handle game tile click
  const handleGameClick = (gameId: string) => {
    if (props.onGameClick) {
      props.onGameClick(gameId);
    }
  };

  // Handle team click
  const handleTeamClick = (e: Event, teamName: string) => {
    e.stopPropagation(); // Prevent game card click
    const abbr = teamNameToAbbr[teamName];
    if (abbr && props.onTeamClick) {
      props.onTeamClick(abbr);
    }
  };

  return (
    <div class="min-h-screen bg-gradient-to-b from-black to-gray-900 text-gray-100">
      {/* Professional Header */}
      <header class="bg-black/50 border-b border-gray-800 sticky top-0 z-50 backdrop-blur-xl">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div class="flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
            <div>
              <h1 class="text-3xl md:text-4xl font-bold tracking-tight text-white">Live Games</h1>
              <p class="text-sm md:text-base text-gray-400 mt-1">Real-time analytics for sports traders</p>
            </div>
            
            {/* Status Pills */}
            <div class="flex flex-wrap gap-3">
              <div class="px-4 py-2.5 bg-red-500/10 border border-red-500/20 rounded-xl backdrop-blur-sm">
                <span class="text-red-400 font-semibold text-sm">{liveGames().length} Live</span>
              </div>
              <div class="px-4 py-2.5 bg-blue-500/10 border border-blue-500/20 rounded-xl backdrop-blur-sm">
                <span class="text-blue-400 font-semibold text-sm">{upcomingGames().length} Upcoming</span>
              </div>
              <Show when={edgesCount() > 0}>
                <div class="px-4 py-2.5 bg-green-500/10 border border-green-500/20 rounded-xl backdrop-blur-sm">
                  <span class="text-green-400 font-semibold text-sm">{edgesCount()} Edges</span>
                </div>
              </Show>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Connection Status */}
        <Show when={!connected()}>
          <div class="modern-card bg-yellow-900/10 border-yellow-700/20 mb-6">
            <div class="flex items-center gap-3">
              <div class="w-3 h-3 bg-yellow-500 rounded-full animate-pulse"></div>
              <span class="text-yellow-400 font-medium">Connecting to NBA API...</span>
            </div>
          </div>
        </Show>

        {/* Live Games Section */}
        <Show when={liveGames().length > 0}>
          <section class="mb-10">
            <div class="flex items-center gap-3 mb-5">
              <div class="w-2 h-2 bg-red-500 rounded-full live-pulse"></div>
              <h2 class="text-2xl font-bold text-white">Live Now</h2>
            </div>
            <div class="grid grid-cols-1 lg:grid-cols-2 gap-5">
              <For each={liveGames()}>
                {(game) => (
                  <div 
                    onClick={() => handleGameClick(game.game_id)}
                    class="modern-card cursor-pointer hover:border-red-500/40 transition-all duration-300 hover:shadow-lg hover:shadow-red-500/20"
                  >
                    {/* Live Status Bar */}
                    <div class="flex items-center justify-between mb-5 pb-3 border-b border-gray-800">
                      <div class="flex items-center gap-2">
                        <div class="w-2 h-2 bg-red-500 rounded-full live-pulse"></div>
                        <span class="text-red-500 font-bold text-xs uppercase tracking-wider">Live</span>
                      </div>
                      <div class="text-gray-300 text-sm font-medium">
                        Q{game.quarter} • {game.time_remaining || game.clock || 'Live'}
                      </div>
                    </div>

                    {/* Teams & Scores */}
                    <div class="space-y-4 mb-5">
                      {/* Away Team */}
                      <div class="flex items-center justify-between group">
                        <div 
                          onClick={(e) => handleTeamClick(e, game.away_team)}
                          class="flex items-center gap-4 flex-1 hover:bg-white/5 rounded-lg p-2 -m-2 transition-colors cursor-pointer"
                        >
                          <img 
                            src={`https://cdn.nba.com/logos/nba/${getTeamLogo(game.away_team)}/primary/L/logo.svg`}
                            alt={game.away_team}
                            class="w-12 h-12 sm:w-14 sm:h-14 transition-transform group-hover:scale-110"
                          />
                          <div class="flex-1 min-w-0">
                            <div class="text-white font-semibold text-base sm:text-lg truncate">
                              {game.away_team}
                            </div>
                            <div class="text-gray-500 text-xs">Away</div>
                          </div>
                        </div>
                        <div class="text-3xl sm:text-4xl font-bold text-white ml-4">{game.score_away}</div>
                      </div>

                      {/* Home Team */}
                      <div class="flex items-center justify-between group">
                        <div 
                          onClick={(e) => handleTeamClick(e, game.home_team)}
                          class="flex items-center gap-4 flex-1 hover:bg-white/5 rounded-lg p-2 -m-2 transition-colors cursor-pointer"
                        >
                          <img 
                            src={`https://cdn.nba.com/logos/nba/${getTeamLogo(game.home_team)}/primary/L/logo.svg`}
                            alt={game.home_team}
                            class="w-12 h-12 sm:w-14 sm:h-14 transition-transform group-hover:scale-110"
                          />
                          <div class="flex-1 min-w-0">
                            <div class="text-white font-semibold text-base sm:text-lg truncate">
                              {game.home_team}
                            </div>
                            <div class="text-gray-500 text-xs">Home</div>
                          </div>
                        </div>
                        <div class="text-3xl sm:text-4xl font-bold text-white ml-4">{game.score_home}</div>
                      </div>
                    </div>

                    {/* ML Prediction (if available) */}
                    <Show when={predictions().get(game.game_id)}>
                      {(pred) => (
                        <div class="mt-4 pt-4 border-t border-gray-800">
                          <div class="grid grid-cols-3 gap-3 text-center">
                            <div>
                              <div class="text-xs text-gray-500 mb-1">Forecast</div>
                              <div class="text-sm font-bold text-purple-400">
                                {pred().point_forecast?.toFixed(1) || 'N/A'}
                              </div>
                            </div>
                            <div>
                              <div class="text-xs text-gray-500 mb-1">Interval</div>
                              <div class="text-sm font-bold text-blue-400">
                                {pred().interval_lower?.toFixed(1) || 'N/A'} - {pred().interval_upper?.toFixed(1) || 'N/A'}
                              </div>
                            </div>
                            <div>
                              <div class="text-xs text-gray-500 mb-1">Edge</div>
                              <div class="text-sm font-bold text-green-400">
                                {edges().get(game.game_id)?.has_edge ? '✓' : '-'}
                              </div>
                            </div>
                          </div>
                        </div>
                      )}
                    </Show>

                    {/* Click hint */}
                    <div class="mt-4 pt-3 border-t border-gray-800 text-center">
                      <span class="text-gray-500 text-xs">Click for full game analysis →</span>
                    </div>
                  </div>
                )}
              </For>
            </div>
          </section>
        </Show>

        {/* Upcoming Games Section */}
        <Show when={upcomingGames().length > 0}>
          <section class="mb-10">
            <div class="flex items-center gap-3 mb-5">
              <div class="w-2 h-2 bg-blue-500 rounded-full"></div>
              <h2 class="text-2xl font-bold text-white">Scheduled Games</h2>
            </div>
            <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              <For each={upcomingGames()}>
                {(game) => (
                  <div 
                    onClick={() => handleGameClick(game.game_id)}
                    class="modern-card cursor-pointer hover:border-blue-500/40 transition-all duration-300"
                  >
                    <div class="text-center mb-4 pb-4 border-b border-gray-800">
                      <div class="text-blue-400 text-xs font-semibold uppercase mb-2">Scheduled</div>
                      <div class="text-sm text-gray-400">Game ID: {game.game_id}</div>
                    </div>

                    {/* Teams */}
                    <div class="space-y-3">
                      {/* Away Team */}
                      <div 
                        onClick={(e) => handleTeamClick(e, game.away_team)}
                        class="flex items-center gap-3 hover:bg-white/5 rounded-lg p-2 -m-2 transition-colors cursor-pointer"
                      >
                        <img 
                          src={`https://cdn.nba.com/logos/nba/${getTeamLogo(game.away_team)}/primary/L/logo.svg`}
                          alt={game.away_team}
                          class="w-10 h-10"
                        />
                        <div class="flex-1 min-w-0">
                          <div class="text-white font-semibold text-sm truncate">{game.away_team}</div>
                          <div class="text-gray-500 text-xs">Away</div>
                        </div>
                      </div>

                      {/* Home Team */}
                      <div 
                        onClick={(e) => handleTeamClick(e, game.home_team)}
                        class="flex items-center gap-3 hover:bg-white/5 rounded-lg p-2 -m-2 transition-colors cursor-pointer"
                      >
                        <img 
                          src={`https://cdn.nba.com/logos/nba/${getTeamLogo(game.home_team)}/primary/L/logo.svg`}
                          alt={game.home_team}
                          class="w-10 h-10"
                        />
                        <div class="flex-1 min-w-0">
                          <div class="text-white font-semibold text-sm truncate">{game.home_team}</div>
                          <div class="text-gray-500 text-xs">Home</div>
                        </div>
                      </div>
                    </div>

                    {/* Click hint */}
                    <div class="mt-4 pt-3 border-t border-gray-800 text-center">
                      <span class="text-gray-500 text-xs">Click for pre-game analysis →</span>
                    </div>
                  </div>
                )}
              </For>
            </div>
          </section>
        </Show>

        {/* No Games Message */}
        <Show when={gamesList().length === 0 && connected()}>
          <div class="modern-card text-center py-16">
            <div class="text-6xl mb-6 opacity-50">🏀</div>
            <h3 class="text-2xl font-bold text-white mb-3">No Games Right Now</h3>
            <p class="text-gray-400 text-lg">Check back soon for live games and predictions</p>
          </div>
        </Show>

        {/* Full Game Analysis Cards (for games with predictions) */}
        <Show when={predictions().size > 0}>
          <section class="mt-10">
            <h2 class="text-2xl font-bold text-white mb-5">Detailed Analysis</h2>
            <div class="space-y-6">
              <For each={Array.from(predictions().keys())}>
                {(gameId) => {
                  const game = games().get(gameId);
                  const prediction = predictions().get(gameId);
                  const edge = edges().get(gameId);
                  const recommendation = recommendations().get(gameId);
                  const pattern = patterns().get(gameId);
                  
                  return game ? (
                    <GameCardExpanded 
                      game={game}
                      prediction={prediction as any}
                      edge={edge}
                      recommendation={recommendation}
                      pattern={pattern}
                    />
                  ) : null;
                }}
              </For>
            </div>
          </section>
        </Show>

        {/* System Status */}
        <div class="mt-10">
          <SystemStatus />
        </div>
      </main>
    </div>
  );
};

export default Dashboard;
