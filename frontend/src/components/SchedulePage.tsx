import { type Component, createSignal, onMount, For, Show } from 'solid-js';

interface ScheduledGame {
  game_id: string;
  date: string;
  time: string | null;
  home_team: {
    abbr: string;
    name: string;
    logo: string;
  };
  away_team: {
    abbr: string;
    name: string;
    logo: string;
  };
  status: string;
  arena: string;
  tv: string;
  score?: {
    home: number;
    away: number;
  };
}

const SchedulePage: Component = () => {
  const [games, setGames] = createSignal<ScheduledGame[]>([]);
  const [daysAhead, setDaysAhead] = createSignal(30);
  const [loading, setLoading] = createSignal(true);
  const [selectedTeam, setSelectedTeam] = createSignal<string>('');
  const [viewMode, setViewMode] = createSignal<'upcoming' | 'today' | 'week' | 'month'>('upcoming');

  const API_BASE = 'https://ol24-production.up.railway.app';

  const fetchSchedule = async () => {
    setLoading(true);
    try {
      const response = await fetch(`${API_BASE}/api/schedule?days_ahead=${daysAhead()}`);
      const data = await response.json();
      
      let gamesList = data.games || [];
      
      // Filter by team if selected
      if (selectedTeam()) {
        gamesList = gamesList.filter((g: ScheduledGame) => 
          g.home_team.abbr === selectedTeam() || g.away_team.abbr === selectedTeam()
        );
      }
      
      setGames(gamesList);
    } catch (error) {
      console.error('Error fetching schedule:', error);
    }
    setLoading(false);
  };

  onMount(() => {
    fetchSchedule();
  });

  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr);
    return date.toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric' });
  };

  const groupGamesByDate = () => {
    const grouped: { [key: string]: ScheduledGame[] } = {};
    games().forEach(game => {
      if (!grouped[game.date]) {
        grouped[game.date] = [];
      }
      grouped[game.date].push(game);
    });
    return grouped;
  };

  return (
    <div class="min-h-screen bg-gradient-to-br from-gray-900 via-purple-900 to-gray-900 p-6">
      <div class="max-w-6xl mx-auto">
        {/* Header */}
        <div class="text-center mb-8">
          <h1 class="text-5xl font-bold text-white mb-2">
            📅 NBA Schedule
          </h1>
          <p class="text-gray-400 text-lg">
            Upcoming games, depth charts, and predictions
          </p>
        </div>

        {/* Professional Filter Bar */}
        <div class="bg-gray-800 rounded-lg p-6 mb-6 border border-gray-700">
          {/* View Mode Buttons */}
          <div class="flex gap-2 mb-4 justify-center flex-wrap">
            <button
              onClick={() => { setViewMode('today'); setDaysAhead(1); fetchSchedule(); }}
              class={`px-6 py-3 rounded-lg font-semibold transition-all ${
                viewMode() === 'today' ? 'bg-blue-600 text-white' : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }`}
            >
              📅 Today
            </button>
            <button
              onClick={() => { setViewMode('week'); setDaysAhead(7); fetchSchedule(); }}
              class={`px-6 py-3 rounded-lg font-semibold transition-all ${
                viewMode() === 'week' ? 'bg-blue-600 text-white' : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }`}
            >
              📅 This Week
            </button>
            <button
              onClick={() => { setViewMode('month'); setDaysAhead(30); fetchSchedule(); }}
              class={`px-6 py-3 rounded-lg font-semibold transition-all ${
                viewMode() === 'month' ? 'bg-blue-600 text-white' : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }`}
            >
              📅 This Month
            </button>
            <button
              onClick={() => { setViewMode('upcoming'); setDaysAhead(180); fetchSchedule(); }}
              class={`px-6 py-3 rounded-lg font-semibold transition-all ${
                viewMode() === 'upcoming' ? 'bg-blue-600 text-white' : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }`}
            >
              📅 Full Season
            </button>
          </div>

          {/* Team Filter */}
          <div class="flex items-center justify-center gap-3">
            <label class="text-gray-400 font-semibold">Filter by Team:</label>
            <select 
              value={selectedTeam()}
              onChange={(e) => {
                setSelectedTeam(e.target.value);
                fetchSchedule();
              }}
              class="bg-gray-700 text-white px-4 py-2 rounded-lg border border-gray-600 font-semibold min-w-[200px]"
            >
              <option value="">🏀 All Teams</option>
              <option value="ATL">Hawks</option>
              <option value="BOS">Celtics</option>
              <option value="BKN">Nets</option>
              <option value="CHA">Hornets</option>
              <option value="CHI">Bulls</option>
              <option value="CLE">Cavaliers</option>
              <option value="DAL">Mavericks</option>
              <option value="DEN">Nuggets</option>
              <option value="DET">Pistons</option>
              <option value="GSW">Warriors</option>
              <option value="HOU">Rockets</option>
              <option value="IND">Pacers</option>
              <option value="LAC">Clippers</option>
              <option value="LAL">Lakers</option>
              <option value="MEM">Grizzlies</option>
              <option value="MIA">Heat</option>
              <option value="MIL">Bucks</option>
              <option value="MIN">Timberwolves</option>
              <option value="NOP">Pelicans</option>
              <option value="NYK">Knicks</option>
              <option value="OKC">Thunder</option>
              <option value="ORL">Magic</option>
              <option value="PHI">76ers</option>
              <option value="PHX">Suns</option>
              <option value="POR">Trail Blazers</option>
              <option value="SAC">Kings</option>
              <option value="SAS">Spurs</option>
              <option value="TOR">Raptors</option>
              <option value="UTA">Jazz</option>
              <option value="WAS">Wizards</option>
            </select>
            
            <Show when={selectedTeam()}>
              <button
                onClick={() => { setSelectedTeam(''); fetchSchedule(); }}
                class="px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded-lg font-semibold"
              >
                Clear Filter
              </button>
            </Show>
          </div>

          {/* Stats Summary */}
          <div class="text-center mt-4 text-gray-400 text-sm">
            Showing {games().length} games
            {selectedTeam() && ` for ${selectedTeam()}`}
          </div>
        </div>

        <Show when={loading()}>
          <div class="text-center text-white text-xl">Loading schedule...</div>
        </Show>

        <Show when={!loading()}>
          {/* Games grouped by date */}
          <For each={Object.entries(groupGamesByDate())}>
            {([date, gamesOnDate]) => (
              <div class="mb-8">
                <h2 class="text-2xl font-bold text-white mb-4">
                  {formatDate(date)}
                </h2>
                <div class="space-y-4">
                  <For each={gamesOnDate}>
                    {(game) => (
                      <div class="bg-gray-800 rounded-lg p-6 border border-gray-700 hover:border-blue-500 transition-all">
                        <div class="flex items-center justify-between">
                          {/* Away Team */}
                          <div class="flex items-center gap-3 flex-1">
                            <Show when={game.away_team.logo}>
                              <img src={game.away_team.logo} alt={game.away_team.abbr} class="w-12 h-12" />
                            </Show>
                            <div>
                              <h3 class="text-white font-bold text-lg">{game.away_team.abbr}</h3>
                              <p class="text-gray-400 text-sm">{game.away_team.name}</p>
                            </div>
                            <Show when={game.score}>
                              <div class="text-2xl font-bold text-white ml-4">
                                {game.score!.away}
                              </div>
                            </Show>
                          </div>

                          {/* Game Info */}
                          <div class="text-center px-6">
                            <div class="text-gray-400 text-sm mb-1">
                              {game.time || 'TBD'}
                            </div>
                            <div class={`px-3 py-1 rounded text-xs font-semibold ${
                              game.status === 'Final' ? 'bg-gray-600' :
                              game.status === 'Live' ? 'bg-red-600 animate-pulse' :
                              'bg-blue-600'
                            } text-white`}>
                              {game.status}
                            </div>
                            <Show when={game.tv}>
                              <div class="text-gray-500 text-xs mt-1">{game.tv}</div>
                            </Show>
                          </div>

                          {/* Home Team */}
                          <div class="flex items-center gap-3 flex-1 justify-end">
                            <Show when={game.score}>
                              <div class="text-2xl font-bold text-white mr-4">
                                {game.score!.home}
                              </div>
                            </Show>
                            <div class="text-right">
                              <h3 class="text-white font-bold text-lg">{game.home_team.abbr}</h3>
                              <p class="text-gray-400 text-sm">{game.home_team.name}</p>
                            </div>
                            <Show when={game.home_team.logo}>
                              <img src={game.home_team.logo} alt={game.home_team.abbr} class="w-12 h-12" />
                            </Show>
                          </div>
                        </div>

                        {/* Arena */}
                        <div class="text-center text-gray-500 text-sm mt-3 pt-3 border-t border-gray-700">
                          📍 {game.arena}
                        </div>
                      </div>
                    )}
                  </For>
                </div>
              </div>
            )}
          </For>

          <Show when={games().length === 0}>
            <div class="text-center text-gray-500 text-xl py-12">
              No games scheduled for this period
            </div>
          </Show>
        </Show>
      </div>
    </div>
  );
};

export default SchedulePage;

