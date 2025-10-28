import { type Component, createSignal, onMount, For, Show } from 'solid-js';

interface Player {
  player_id: string;
  name: string;
  position: string;
  jersey: string;
  depth_rank: number;
  mpg: number;
  is_starter: boolean;
  ppg: number;
  rpg: number;
  apg: number;
  injury_status: string | null;
}

interface TeamScheduleGame {
  game_id: string;
  date: string;
  time: string | null;
  opponent: string;
  location: string;
  status: string;
  arena: string;
}

interface Props {
  teamAbbr: string;
}

const TeamPage: Component<Props> = (props) => {
  const [starters, setStarters] = createSignal<Player[]>([]);
  const [depthChart, setDepthChart] = createSignal<{ [key: string]: Player[] }>({});
  const [schedule, setSchedule] = createSignal<TeamScheduleGame[]>([]);
  const [teamName, setTeamName] = createSignal('');
  const [loading, setLoading] = createSignal(true);

  const API_BASE = 'https://ol24-production.up.railway.app';

  const [teamStats, setTeamStats] = createSignal<any>(null);

  onMount(async () => {
    try {
      // Fetch team info from teams API
      const teamsRes = await fetch(`${API_BASE}/api/stats/teams`);
      const teamsData = await teamsRes.json();
      const teamInfo = (teamsData.teams || []).find((t: any) => t.abbreviation === props.teamAbbr);
      setTeamStats(teamInfo);
      setTeamName(teamInfo?.full_name || props.teamAbbr);

      // Fetch depth chart
      const depthRes = await fetch(`${API_BASE}/api/team/${props.teamAbbr}/depth-chart`);
      const depthData = await depthRes.json();
      
      setStarters(depthData.starters || []);
      setDepthChart(depthData.depth_chart || {});

      // Fetch team schedule
      const schedRes = await fetch(`${API_BASE}/api/team/${props.teamAbbr}/schedule?days_ahead=14`);
      const schedData = await schedRes.json();
      
      setSchedule(schedData.games || []);

      setLoading(false);
    } catch (error) {
      console.error('Error fetching team data:', error);
      setLoading(false);
    }
  });

  return (
    <div class="min-h-screen bg-gradient-to-br from-gray-900 via-indigo-900 to-gray-900 p-6">
      <div class="max-w-7xl mx-auto">
        {/* Header with Team Logo */}
        <div class="text-center mb-8">
          <Show when={teamStats()?.logo_url}>
            <img 
              src={teamStats()!.logo_url} 
              alt={props.teamAbbr} 
              class="w-32 h-32 mx-auto mb-4"
            />
          </Show>
          <h1 class="text-5xl font-bold text-white mb-2">
            {teamName() || props.teamAbbr}
          </h1>
          <Show when={teamStats()}>
            <div class="flex items-center justify-center gap-6 text-gray-400 text-lg">
              <span class="text-2xl font-bold text-white">{teamStats()!.wins}-{teamStats()!.losses}</span>
              <span>•</span>
              <span>{teamStats()!.ppg?.toFixed(1) || 'N/A'} PPG</span>
              <span>•</span>
              <span>{teamStats()!.games_played || 0} GP</span>
            </div>
          </Show>
        </div>

        {/* Team Stats Dashboard */}
        <Show when={teamStats() && !loading()}>
          <div class="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
            <div class="bg-gray-800 rounded-lg p-4 border border-gray-700">
              <div class="text-gray-400 text-sm mb-1">Points Per Game</div>
              <div class="text-3xl font-bold text-white">{teamStats()!.ppg?.toFixed(1) || 'N/A'}</div>
            </div>
            <div class="bg-gray-800 rounded-lg p-4 border border-gray-700">
              <div class="text-gray-400 text-sm mb-1">Win Percentage</div>
              <div class="text-3xl font-bold text-green-400">
                {((teamStats()!.wins / (teamStats()!.wins + teamStats()!.losses)) * 100).toFixed(1) || 0}%
              </div>
            </div>
            <div class="bg-gray-800 rounded-lg p-4 border border-gray-700">
              <div class="text-gray-400 text-sm mb-1">Games Played</div>
              <div class="text-3xl font-bold text-white">{teamStats()!.games_played || 0}</div>
            </div>
            <div class="bg-gray-800 rounded-lg p-4 border border-gray-700">
              <div class="text-gray-400 text-sm mb-1">Record</div>
              <div class="text-3xl font-bold text-white">{teamStats()!.wins}-{teamStats()!.losses}</div>
            </div>
          </div>
        </Show>

        <Show when={loading()}>
          <div class="text-center text-white text-xl">Loading...</div>
        </Show>

        <Show when={!loading()}>
          {/* Starting Lineup */}
          <div class="mb-8">
            <h2 class="text-3xl font-bold text-white mb-4">🏀 Starting Lineup</h2>
            <div class="grid grid-cols-1 md:grid-cols-5 gap-4">
              <For each={starters()}>
                {(player) => (
                  <div class="bg-gray-800 rounded-lg p-4 border-2 border-green-500">
                    <div class="text-center mb-3">
                      <div class="text-white font-bold text-lg">{player.name}</div>
                      <div class="text-gray-400 text-sm">#{player.jersey} | {player.position}</div>
                      <Show when={player.injury_status}>
                        <div class="mt-2 px-2 py-1 bg-red-600 text-white text-xs rounded">
                          {player.injury_status}
                        </div>
                      </Show>
                    </div>
                    <div class="space-y-1 text-sm">
                      <div class="flex justify-between">
                        <span class="text-gray-400">MPG:</span>
                        <span class="text-white font-semibold">{player.mpg.toFixed(1)}</span>
                      </div>
                      <div class="flex justify-between">
                        <span class="text-gray-400">PPG:</span>
                        <span class="text-white font-semibold">{player.ppg.toFixed(1)}</span>
                      </div>
                      <div class="flex justify-between">
                        <span class="text-gray-400">RPG:</span>
                        <span class="text-white font-semibold">{player.rpg.toFixed(1)}</span>
                      </div>
                      <div class="flex justify-between">
                        <span class="text-gray-400">APG:</span>
                        <span class="text-white font-semibold">{player.apg.toFixed(1)}</span>
                      </div>
                    </div>
                  </div>
                )}
              </For>
            </div>
          </div>

          {/* Full Depth Chart */}
          <div class="mb-8">
            <h2 class="text-3xl font-bold text-white mb-4">📊 Depth Chart</h2>
            <div class="space-y-4">
              <For each={Object.entries(depthChart())}>
                {([position, players]) => (
                  <div class="bg-gray-800 rounded-lg p-4 border border-gray-700">
                    <h3 class="text-white font-bold text-xl mb-3">{position}</h3>
                    <div class="space-y-2">
                      <For each={players}>
                        {(player) => (
                          <div class={`flex items-center justify-between p-3 rounded ${
                            player.is_starter ? 'bg-green-900/30 border border-green-500' : 'bg-gray-700'
                          }`}>
                            <div class="flex items-center gap-3">
                              <div class="text-gray-400 font-bold text-sm w-8">{player.depth_rank}</div>
                              <div>
                                <div class="text-white font-semibold">{player.name}</div>
                                <div class="text-gray-400 text-xs">#{player.jersey}</div>
                              </div>
                              <Show when={player.injury_status}>
                                <span class="px-2 py-1 bg-red-600 text-white text-xs rounded">
                                  {player.injury_status}
                                </span>
                              </Show>
                            </div>
                            <div class="flex gap-6 text-sm">
                              <div>
                                <span class="text-gray-400">MPG: </span>
                                <span class="text-white font-semibold">{player.mpg.toFixed(1)}</span>
                              </div>
                              <div>
                                <span class="text-gray-400">PPG: </span>
                                <span class="text-white font-semibold">{player.ppg.toFixed(1)}</span>
                              </div>
                              <div>
                                <span class="text-gray-400">RPG: </span>
                                <span class="text-white font-semibold">{player.rpg.toFixed(1)}</span>
                              </div>
                              <div>
                                <span class="text-gray-400">APG: </span>
                                <span class="text-white font-semibold">{player.apg.toFixed(1)}</span>
                              </div>
                            </div>
                          </div>
                        )}
                      </For>
                    </div>
                  </div>
                )}
              </For>
            </div>
          </div>

          {/* Team Schedule */}
          <div>
            <h2 class="text-3xl font-bold text-white mb-4">📅 Upcoming Games</h2>
            <div class="space-y-3">
              <For each={schedule()}>
                {(game) => (
                  <div class="bg-gray-800 rounded-lg p-4 border border-gray-700 hover:border-purple-500 transition-all">
                    <div class="flex items-center justify-between">
                      <div class="flex items-center gap-4">
                        <div class="text-gray-400 text-sm min-w-[100px]">
                          {new Date(game.date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}
                        </div>
                        <div class="text-white font-semibold">
                          vs. {game.opponent}
                        </div>
                        <span class={`px-3 py-1 rounded text-xs font-semibold ${
                          game.location === 'Home' ? 'bg-blue-600' : 'bg-purple-600'
                        } text-white`}>
                          {game.location}
                        </span>
                      </div>
                      <div class="text-gray-400 text-sm">
                        {game.time || 'TBD'}
                      </div>
                    </div>
                  </div>
                )}
              </For>
            </div>
          </div>
        </Show>
      </div>
    </div>
  );
};

export default TeamPage;

