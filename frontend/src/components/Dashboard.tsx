/**
 * Main Dashboard Component
 * Real-time NBA predictions and betting dashboard
 * Professional sports trading interface
 */

import { type Component, For, Show, onMount, onCleanup, createSignal } from 'solid-js';
import { wsService } from '../services/websocket';
import GameCardExpanded from './GameCardExpanded';
import SystemStatus from './SystemStatus';

interface DashboardProps {
  onGameClick?: (gameId: string) => void;
  onTeamClick?: (teamAbbr: string) => void;
}

// Comprehensive team name/abbreviation to NBA team ID mapping
const getTeamId = (teamIdentifier: string): string => {
  // Map of all possible team name variations + abbreviations → NBA team ID
  const teamMap: Record<string, string> = {
    // Full names
    'Atlanta Hawks': '1610612737',
    'Boston Celtics': '1610612738',
    'Brooklyn Nets': '1610612751',
    'Charlotte Hornets': '1610612766',
    'Chicago Bulls': '1610612741',
    'Cleveland Cavaliers': '1610612739',
    'Dallas Mavericks': '1610612742',
    'Denver Nuggets': '1610612743',
    'Detroit Pistons': '1610612765',
    'Golden State Warriors': '1610612744',
    'Houston Rockets': '1610612745',
    'Indiana Pacers': '1610612754',
    'Los Angeles Clippers': '1610612746',
    'LA Clippers': '1610612746',
    'Los Angeles Lakers': '1610612747',
    'Memphis Grizzlies': '1610612763',
    'Miami Heat': '1610612748',
    'Milwaukee Bucks': '1610612749',
    'Minnesota Timberwolves': '1610612750',
    'New Orleans Pelicans': '1610612740',
    'New York Knicks': '1610612752',
    'Oklahoma City Thunder': '1610612760',
    'Orlando Magic': '1610612753',
    'Philadelphia 76ers': '1610612755',
    'Phoenix Suns': '1610612756',
    'Portland Trail Blazers': '1610612757',
    'Sacramento Kings': '1610612758',
    'San Antonio Spurs': '1610612759',
    'Toronto Raptors': '1610612761',
    'Utah Jazz': '1610612762',
    'Washington Wizards': '1610612764',
    
    // Abbreviations
    'ATL': '1610612737', 'BOS': '1610612738', 'BKN': '1610612751', 'CHA': '1610612766',
    'CHI': '1610612741', 'CLE': '1610612739', 'DAL': '1610612742', 'DEN': '1610612743',
    'DET': '1610612765', 'GSW': '1610612744', 'HOU': '1610612745', 'IND': '1610612754',
    'LAC': '1610612746', 'LAL': '1610612747', 'MEM': '1610612763', 'MIA': '1610612748',
    'MIL': '1610612749', 'MIN': '1610612750', 'NOP': '1610612740', 'NYK': '1610612752',
    'OKC': '1610612760', 'ORL': '1610612753', 'PHI': '1610612755', 'PHX': '1610612756',
    'POR': '1610612757', 'SAC': '1610612758', 'SAS': '1610612759', 'TOR': '1610612761',
    'UTA': '1610612762', 'WAS': '1610612764'
  };
  
  return teamMap[teamIdentifier] || teamMap[teamIdentifier.toUpperCase()] || '1610612738';
};

// Team name to abbreviation mapping (for navigation)
const teamNameToAbbr: Record<string, string> = {
  'Atlanta Hawks': 'ATL', 'Boston Celtics': 'BOS', 'Brooklyn Nets': 'BKN', 'Charlotte Hornets': 'CHA',
  'Chicago Bulls': 'CHI', 'Cleveland Cavaliers': 'CLE', 'Dallas Mavericks': 'DAL', 'Denver Nuggets': 'DEN',
  'Detroit Pistons': 'DET', 'Golden State Warriors': 'GSW', 'Houston Rockets': 'HOU', 'Indiana Pacers': 'IND',
  'LA Clippers': 'LAC', 'Los Angeles Clippers': 'LAC', 'Los Angeles Lakers': 'LAL',
  'Memphis Grizzlies': 'MEM', 'Miami Heat': 'MIA', 'Milwaukee Bucks': 'MIL', 'Minnesota Timberwolves': 'MIN',
  'New Orleans Pelicans': 'NOP', 'New York Knicks': 'NYK', 'Oklahoma City Thunder': 'OKC',
  'Orlando Magic': 'ORL', 'Philadelphia 76ers': 'PHI', 'Phoenix Suns': 'PHX', 'Portland Trail Blazers': 'POR',
  'Sacramento Kings': 'SAC', 'San Antonio Spurs': 'SAS', 'Toronto Raptors': 'TOR', 'Utah Jazz': 'UTA',
  'Washington Wizards': 'WAS'
};

const Dashboard: Component<DashboardProps> = (props) => {
  // Get signals from WebSocket service (reactive!)
  const [connected] = wsService.connected;
  const [games] = wsService.games;
  const [patterns] = wsService.patterns;
  const [predictions] = wsService.predictions;
  const [edges] = wsService.edges;
  const [recommendations] = wsService.recommendations;
  
  // Scheduled games from API
  const [scheduledGames, setScheduledGames] = createSignal<any[]>([]);
  
  const API_BASE = 'https://ol24-production.up.railway.app';

  // Fetch scheduled games
  const fetchScheduledGames = async () => {
    try {
      const response = await fetch(`${API_BASE}/api/schedule?days=3`);
      const data = await response.json();
      setScheduledGames(data.games || []);
      console.log('📅 Fetched scheduled games:', data.games?.length || 0);
    } catch (error) {
      console.error('Error fetching schedule:', error);
    }
  };

  // Connect on mount
  onMount(() => {
    wsService.connect();
    fetchScheduledGames();
    
    // Refresh scheduled games every 5 minutes
    const scheduleInterval = setInterval(fetchScheduledGames, 5 * 60 * 1000);
    
    onCleanup(() => {
      clearInterval(scheduleInterval);
    });
  });

  // Disconnect on unmount
  onCleanup(() => {
    wsService.disconnect();
  });

  // Convert Map to Array for iteration
  const gamesList = () => {
    const gamesArray = Array.from(games().values());
    // Debug: Log team names to verify mapping
    if (gamesArray.length > 0 && gamesArray[0]) {
      console.log('🏀 Team names from API:', {
        away: gamesArray[0].away_team,
        home: gamesArray[0].home_team,
        away_id: getTeamId(gamesArray[0].away_team),
        home_id: getTeamId(gamesArray[0].home_team)
      });
    }
    return gamesArray;
  };

  // Live games
  const liveGames = () => gamesList().filter(g => g.is_live);

  // Upcoming games - use scheduledGames from API, not WebSocket, sorted by time
  const upcomingGames = () => {
    const filtered = scheduledGames().filter(g => g.status !== 'Final' && g.status !== 'Live');
    
    // Sort by date and time (earliest first)
    return filtered.sort((a, b) => {
      // First sort by date
      if (a.date !== b.date) {
        return a.date.localeCompare(b.date);
      }
      
      // Then by time
      const timeA = a.time_pst || a.time || '';
      const timeB = b.time_pst || b.time || '';
      return timeA.localeCompare(timeB);
    });
  };

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
                        <Show when={game.game_date}>
                          <span class="text-gray-500 text-xs">• {game.game_date}</span>
                        </Show>
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
                            src={`https://cdn.nba.com/logos/nba/${getTeamId(game.away_team)}/primary/L/logo.svg`}
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
                            src={`https://cdn.nba.com/logos/nba/${getTeamId(game.home_team)}/primary/L/logo.svg`}
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

                    {/* ML PREDICTION FEED (if available) */}
                    <Show when={predictions().get(game.game_id)}>
                      {(pred) => {
                        const [showJSON, setShowJSON] = createSignal(false);
                        return (
                          <div class="mt-4 pt-4 border-t border-purple-900/30">
                            {/* ML Header */}
                            <div class="flex items-center justify-between mb-3">
                              <div class="flex items-center gap-2">
                                <div class="w-2 h-2 bg-purple-500 rounded-full animate-pulse"></div>
                                <span class="text-purple-400 font-bold text-xs uppercase tracking-wide">🤖 Mamba Prediction</span>
                              </div>
                              <button 
                                onClick={(e) => {
                                  e.stopPropagation();
                                  setShowJSON(!showJSON());
                                }}
                                class="text-xs text-gray-500 hover:text-purple-400 transition-colors"
                              >
                                {showJSON() ? 'Hide' : 'Show'} JSON
                              </button>
                            </div>

                            {/* ML Metrics Grid */}
                            <div class="grid grid-cols-4 gap-2 mb-3">
                              <div class="bg-purple-950/30 rounded-lg p-2 text-center">
                                <div class="text-xs text-gray-500 mb-1">Spread</div>
                                <div class="text-lg font-black text-white">
                                  {pred().point_forecast > 0 ? '+' : ''}{pred().point_forecast?.toFixed(1) || '--'}
                                </div>
                              </div>
                              <div class="bg-blue-950/30 rounded-lg p-2 text-center">
                                <div class="text-xs text-gray-500 mb-1">90% CI</div>
                                <div class="text-xs font-bold text-blue-400">
                                  {pred().interval_lower?.toFixed(1) || '--'} to {pred().interval_upper?.toFixed(1) || '--'}
                                </div>
                              </div>
                              <div class="bg-green-950/30 rounded-lg p-2 text-center">
                                <div class="text-xs text-gray-500 mb-1">Win%</div>
                                <div class="text-lg font-black text-green-400">
                                  {pred().win_probability ? (pred().win_probability! * 100).toFixed(0) : '--'}%
                                </div>
                              </div>
                              <div class="bg-yellow-950/30 rounded-lg p-2 text-center">
                                <div class="text-xs text-gray-500 mb-1">Edge</div>
                                <div class="text-lg font-black text-yellow-400">
                                  {edges().get(game.game_id)?.has_edge ? '✓' : '—'}
                                </div>
                              </div>
                            </div>

                            {/* Detailed Status */}
                            <div class="text-xs text-gray-500">
                              <Show when={pred().is_q2_6min}>
                                <div class="text-yellow-400 font-bold">⭐ Q2 6:00 TRADE SIGNAL ACTIVE</div>
                              </Show>
                              <Show when={!pred().is_q2_6min && pred().prediction_type === 'continuous'}>
                                <div class="text-purple-400">📊 Continuous 30s prediction</div>
                              </Show>
                              <Show when={pred().timestamp}>
                                <div>Last updated: {new Date(pred().timestamp).toLocaleTimeString()}</div>
                              </Show>
                            </div>

                            {/* RAW JSON (toggleable for transparency) */}
                            <Show when={showJSON()}>
                              <div 
                                onClick={(e) => e.stopPropagation()}
                                class="mt-3 bg-black/50 rounded-lg p-3 overflow-x-auto"
                              >
                                <pre class="text-xs text-green-400 font-mono">
                                  {JSON.stringify(pred(), null, 2)}
                                </pre>
                              </div>
                            </Show>
                          </div>
                        );
                      }}
                    </Show>

                    {/* ML Status if NOT available */}
                    <Show when={!predictions().get(game.game_id)}>
                      <div class="mt-4 pt-4 border-t border-gray-800">
                        <div class="text-center text-gray-500 text-xs">
                          {game.quarter >= 2 ? '⏳ Waiting for Q2 6:00 prediction window...' : '🏀 Collecting play-by-play data...'}
            </div>
          </div>
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
                      <div class="text-sm text-white font-medium mb-1">
                        {game.date || game.game_date || 'TBD'}
                      </div>
                      <div class="text-lg text-blue-400 font-bold">
                        {game.time_pst || game.time || game.game_time || 'Time TBD'}
                      </div>
                    </div>

                    {/* Teams */}
                    <div class="space-y-3">
                      {/* Away Team */}
                      <div 
                        onClick={(e) => handleTeamClick(e, game.away_team?.abbr || game.away_team?.name || game.away_team)}
                        class="flex items-center gap-3 hover:bg-white/5 rounded-lg p-2 -m-2 transition-colors cursor-pointer"
                      >
                        <Show 
                          when={game.away_team?.logo}
                          fallback={
                            <img 
                              src={`https://cdn.nba.com/logos/nba/${getTeamId(game.away_team?.abbr || game.away_team?.name || game.away_team)}/primary/L/logo.svg`}
                              alt={game.away_team?.abbr}
                              class="w-10 h-10"
                            />
                          }
                        >
                          <img 
                            src={game.away_team.logo}
                            alt={game.away_team.abbr}
                            class="w-10 h-10"
                          />
                        </Show>
                        <div class="flex-1 min-w-0">
                          <div class="text-white font-semibold text-sm truncate">
                            {game.away_team?.abbr || game.away_team?.name || game.away_team}
                          </div>
                          <div class="text-gray-500 text-xs">Away</div>
                        </div>
                      </div>

                      {/* Home Team */}
                      <div 
                        onClick={(e) => handleTeamClick(e, game.home_team?.abbr || game.home_team?.name || game.home_team)}
                        class="flex items-center gap-3 hover:bg-white/5 rounded-lg p-2 -m-2 transition-colors cursor-pointer"
                      >
                        <Show 
                          when={game.home_team?.logo}
                          fallback={
                            <img 
                              src={`https://cdn.nba.com/logos/nba/${getTeamId(game.home_team?.abbr || game.home_team?.name || game.home_team)}/primary/L/logo.svg`}
                              alt={game.home_team?.abbr}
                              class="w-10 h-10"
                            />
                          }
                        >
                          <img 
                            src={game.home_team.logo}
                            alt={game.home_team.abbr}
                            class="w-10 h-10"
                          />
                        </Show>
                        <div class="flex-1 min-w-0">
                          <div class="text-white font-semibold text-sm truncate">
                            {game.home_team?.abbr || game.home_team?.name || game.home_team}
                          </div>
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
