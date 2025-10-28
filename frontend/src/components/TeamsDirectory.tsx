import { type Component, createSignal, onMount, For, Show } from 'solid-js';

interface Team {
  team_id: string;
  abbreviation: string;
  full_name: string;
  logo_url: string;
  primary_color: string;
  secondary_color: string;
  wins: number;
  losses: number;
  ppg: number;
  games_played: number;
}

interface Props {
  onTeamClick: (teamAbbr: string) => void;
}

const TeamsDirectory: Component<Props> = (props) => {
  const [teams, setTeams] = createSignal<Team[]>([]);
  const [loading, setLoading] = createSignal(true);
  const [sortBy, setSortBy] = createSignal<'name' | 'record' | 'ppg'>('ppg');

  const API_BASE = 'https://ol24-production.up.railway.app';

  onMount(async () => {
    try {
      const response = await fetch(`${API_BASE}/api/stats/teams`);
      const data = await response.json();
      setTeams(data.teams || []);
      setLoading(false);
    } catch (error) {
      console.error('Error fetching teams:', error);
      setLoading(false);
    }
  });

  const sortedTeams = () => {
    const teamsList = [...teams()];
    if (sortBy() === 'ppg') {
      return teamsList.sort((a, b) => (b.ppg || 0) - (a.ppg || 0));
    } else if (sortBy() === 'record') {
      const winPct = (t: Team) => t.wins / (t.wins + t.losses || 1);
      return teamsList.sort((a, b) => winPct(b) - winPct(a));
    } else {
      return teamsList.sort((a, b) => a.full_name.localeCompare(b.full_name));
    }
  };

  return (
    <div class="min-h-screen bg-gradient-to-br from-gray-900 via-blue-900 to-gray-900 p-6">
      <div class="max-w-7xl mx-auto">
        {/* Header */}
        <div class="text-center mb-8">
          <h1 class="text-5xl font-bold text-white mb-2">
            🏀 All NBA Teams
          </h1>
          <p class="text-gray-400 text-lg">
            Click any team to view depth chart, stats & schedule
          </p>
        </div>

        {/* Sort Options */}
        <div class="bg-gray-800 rounded-lg p-4 mb-6 border border-gray-700 flex gap-3 justify-center">
          <button
            onClick={() => setSortBy('ppg')}
            class={`px-4 py-2 rounded-lg font-semibold ${
              sortBy() === 'ppg' ? 'bg-blue-600 text-white' : 'bg-gray-700 text-gray-300'
            }`}
          >
            Sort by PPG
          </button>
          <button
            onClick={() => setSortBy('record')}
            class={`px-4 py-2 rounded-lg font-semibold ${
              sortBy() === 'record' ? 'bg-blue-600 text-white' : 'bg-gray-700 text-gray-300'
            }`}
          >
            Sort by Record
          </button>
          <button
            onClick={() => setSortBy('name')}
            class={`px-4 py-2 rounded-lg font-semibold ${
              sortBy() === 'name' ? 'bg-blue-600 text-white' : 'bg-gray-700 text-gray-300'
            }`}
          >
            Sort A-Z
          </button>
        </div>

        <Show when={loading()}>
          <div class="text-center text-white text-xl">Loading teams...</div>
        </Show>

        <Show when={!loading()}>
          {/* Teams Grid */}
          <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
            <For each={sortedTeams()}>
              {(team) => (
                <div
                  onClick={() => props.onTeamClick(team.abbreviation)}
                  class="bg-gray-800 rounded-lg p-6 border-2 border-gray-700 hover:border-blue-500 transition-all cursor-pointer hover:scale-105 transform hover:shadow-2xl"
                >
                  {/* Team Logo */}
                  <div class="text-center mb-4">
                    <Show when={team.logo_url}>
                      <img 
                        src={team.logo_url} 
                        alt={team.abbreviation} 
                        class="w-24 h-24 mx-auto mb-3"
                      />
                    </Show>
                    <h3 class="text-white font-bold text-xl mb-1">{team.abbreviation}</h3>
                    <p class="text-gray-400 text-sm">{team.full_name}</p>
                  </div>

                  {/* Team Stats */}
                  <div class="space-y-2">
                    {/* Record */}
                    <div class="flex justify-between items-center bg-gray-900 rounded p-2">
                      <span class="text-gray-400 text-sm">Record</span>
                      <span class="text-white font-bold text-lg">
                        {team.wins}-{team.losses}
                      </span>
                    </div>

                    {/* PPG */}
                    <div class="flex justify-between items-center bg-gray-900 rounded p-2">
                      <span class="text-gray-400 text-sm">PPG</span>
                      <span class="text-blue-400 font-bold text-lg">
                        {team.ppg?.toFixed(1) || 'N/A'}
                      </span>
                    </div>

                    {/* Win % */}
                    <div class="flex justify-between items-center bg-gray-900 rounded p-2">
                      <span class="text-gray-400 text-sm">Win %</span>
                      <span class="text-green-400 font-bold text-lg">
                        {team.wins > 0 ? ((team.wins / (team.wins + team.losses)) * 100).toFixed(1) : '0.0'}%
                      </span>
                    </div>
                  </div>

                  {/* Click Indicator */}
                  <div class="mt-4 pt-3 border-t border-gray-700 text-center">
                    <span class="text-gray-500 text-xs">Click to view details →</span>
                  </div>
                </div>
              )}
            </For>
          </div>
        </Show>
      </div>
    </div>
  );
};

export default TeamsDirectory;

