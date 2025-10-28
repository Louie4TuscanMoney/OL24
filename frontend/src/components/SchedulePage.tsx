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
  const [daysAhead, setDaysAhead] = createSignal(7);
  const [loading, setLoading] = createSignal(true);
  const [selectedTeam, setSelectedTeam] = createSignal<string>('');

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

        {/* Filters */}
        <div class="bg-gray-800 rounded-lg p-4 mb-6 border border-gray-700 flex gap-4 items-center justify-center">
          <div>
            <label class="text-gray-400 text-sm mr-2">Days Ahead:</label>
            <select 
              value={daysAhead()}
              onChange={(e) => {
                setDaysAhead(parseInt(e.target.value));
                fetchSchedule();
              }}
              class="bg-gray-700 text-white px-3 py-2 rounded border border-gray-600"
            >
              <option value="1">Today</option>
              <option value="3">Next 3 Days</option>
              <option value="7">Next Week</option>
              <option value="14">Next 2 Weeks</option>
              <option value="30">Next Month</option>
            </select>
          </div>
          
          <div>
            <label class="text-gray-400 text-sm mr-2">Team Filter:</label>
            <select 
              value={selectedTeam()}
              onChange={(e) => {
                setSelectedTeam(e.target.value);
                fetchSchedule();
              }}
              class="bg-gray-700 text-white px-3 py-2 rounded border border-gray-600"
            >
              <option value="">All Teams</option>
              <option value="LAL">Lakers</option>
              <option value="GSW">Warriors</option>
              <option value="BOS">Celtics</option>
              <option value="MIA">Heat</option>
              {/* Add all 30 teams */}
            </select>
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

