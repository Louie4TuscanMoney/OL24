import { Component, createSignal, onMount, For, Show } from 'solid-js';

interface PlayerStats {
  player_id: string;
  name: string;
  position: string;
  team_abbr: string;
  team_name: string;
  ppg: number;
  rpg: number;
  apg: number;
  fg_pct: number;
  ts_pct: number;
  efg_pct: number;
  pts_100: number;
  gp: number;
}

interface TeamStats {
  team_id: string;
  abbreviation: string;
  full_name: string;
  logo_url: string;
  wins: number;
  losses: number;
  ppg: number;
  net_rating: number;
}

interface Injury {
  player_id: string;
  name: string;
  team_abbr: string;
  status: string;
  injury_type: string;
  description: string;
}

const StatsPage: Component = () => {
  const [topScorers, setTopScorers] = createSignal<PlayerStats[]>([]);
  const [topRebounders, setTopRebounders] = createSignal<PlayerStats[]>([]);
  const [topAssists, setTopAssists] = createSignal<PlayerStats[]>([]);
  const [teams, setTeams] = createSignal<TeamStats[]>([]);
  const [injuries, setInjuries] = createSignal<Injury[]>([]);
  const [loading, setLoading] = createSignal(true);

  const API_BASE = 'https://ol24-production.up.railway.app';

  onMount(async () => {
    try {
      // Fetch teams
      const teamsRes = await fetch(`${API_BASE}/api/stats/teams`);
      const teamsData = await teamsRes.json();
      setTeams(teamsData.teams || []);

      // Fetch injuries
      const injuriesRes = await fetch(`${API_BASE}/api/injuries`);
      const injuriesData = await injuriesRes.json();
      setInjuries(injuriesData.injuries || []);

      // TODO: Add player leaderboard endpoints to backend
      // For now, we'll show teams and injuries

      setLoading(false);
    } catch (error) {
      console.error('Error fetching stats:', error);
      setLoading(false);
    }
  });

  return (
    <div class="min-h-screen bg-gradient-to-br from-gray-900 via-blue-900 to-gray-900 p-6">
      <div class="max-w-7xl mx-auto">
        {/* Header */}
        <div class="text-center mb-8">
          <h1 class="text-5xl font-bold text-white mb-2">
            🏀 NBA Analytics
          </h1>
          <p class="text-gray-400 text-lg">
            Real-time stats, depth charts, injuries & more
          </p>
        </div>

        <Show when={loading()}>
          <div class="text-center text-white text-xl">Loading...</div>
        </Show>

        <Show when={!loading()}>
          {/* Team Standings */}
          <div class="mb-8">
            <h2 class="text-3xl font-bold text-white mb-4">📊 Team Standings</h2>
            <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              <For each={teams()}>
                {(team) => (
                  <div class="bg-gray-800 rounded-lg p-4 border border-gray-700 hover:border-blue-500 transition-all cursor-pointer">
                    <div class="flex items-center gap-3 mb-2">
                      <Show when={team.logo_url}>
                        <img src={team.logo_url} alt={team.abbreviation} class="w-12 h-12" />
                      </Show>
                      <div>
                        <h3 class="text-white font-bold text-lg">{team.full_name}</h3>
                        <p class="text-gray-400 text-sm">{team.wins}-{team.losses}</p>
                      </div>
                    </div>
                    <div class="grid grid-cols-2 gap-2 text-sm">
                      <div class="text-gray-400">PPG:</div>
                      <div class="text-white font-semibold">{team.ppg?.toFixed(1) || 'N/A'}</div>
                      <div class="text-gray-400">Net Rating:</div>
                      <div class={`font-semibold ${(team.net_rating || 0) > 0 ? 'text-green-400' : 'text-red-400'}`}>
                        {team.net_rating > 0 ? '+' : ''}{team.net_rating?.toFixed(1) || 'N/A'}
                      </div>
                    </div>
                  </div>
                )}
              </For>
            </div>
          </div>

          {/* Active Injuries */}
          <div class="mb-8">
            <h2 class="text-3xl font-bold text-white mb-4">🏥 Active Injuries</h2>
            <div class="bg-gray-800 rounded-lg border border-gray-700 overflow-hidden">
              <table class="w-full">
                <thead class="bg-gray-900">
                  <tr class="text-left text-gray-400 text-sm">
                    <th class="p-3">Player</th>
                    <th class="p-3">Team</th>
                    <th class="p-3">Status</th>
                    <th class="p-3">Injury</th>
                    <th class="p-3">Description</th>
                  </tr>
                </thead>
                <tbody>
                  <For each={injuries()}>
                    {(injury) => (
                      <tr class="border-t border-gray-700 hover:bg-gray-750 transition-colors">
                        <td class="p-3 text-white font-semibold">{injury.name}</td>
                        <td class="p-3 text-gray-300">{injury.team_abbr}</td>
                        <td class="p-3">
                          <span class={`px-2 py-1 rounded text-xs font-semibold ${
                            injury.status === 'Out' ? 'bg-red-600 text-white' :
                            injury.status === 'Questionable' ? 'bg-yellow-600 text-white' :
                            injury.status === 'Probable' ? 'bg-green-600 text-white' :
                            'bg-gray-600 text-white'
                          }`}>
                            {injury.status}
                          </span>
                        </td>
                        <td class="p-3 text-gray-300">{injury.injury_type}</td>
                        <td class="p-3 text-gray-400 text-sm">{injury.description}</td>
                      </tr>
                    )}
                  </For>
                  <Show when={injuries().length === 0}>
                    <tr>
                      <td colspan="5" class="p-6 text-center text-gray-500">
                        No active injuries reported
                      </td>
                    </tr>
                  </Show>
                </tbody>
              </table>
            </div>
          </div>
        </Show>
      </div>
    </div>
  );
};

export default StatsPage;

