import { type Component, createSignal, onMount, For, Show } from 'solid-js';

interface GameDetails {
  game_id: string;
  date: string;
  time: string | null;
  home_team: { abbr: string; name: string; logo: string };
  away_team: { abbr: string; name: string; logo: string };
  status: string;
  arena: string;
  tv: string;
  score?: { home: number; away: number };
}

interface Props {
  game: GameDetails;
  onClose: () => void;
}

const ScheduleGameModal: Component<Props> = (props) => {
  const [homeDepth, setHomeDepth] = createSignal<any[]>([]);
  const [awayDepth, setAwayDepth] = createSignal<any[]>([]);
  const [loading, setLoading] = createSignal(true);

  const API_BASE = 'https://ol24-production.up.railway.app';

  onMount(async () => {
    try {
      // Fetch depth charts for both teams
      const [homeRes, awayRes] = await Promise.all([
        fetch(`${API_BASE}/api/team/${props.game.home_team.abbr}/depth-chart`),
        fetch(`${API_BASE}/api/team/${props.game.away_team.abbr}/depth-chart`)
      ]);

      const homeData = await homeRes.json();
      const awayData = await awayRes.json();

      setHomeDepth(homeData.starters || []);
      setAwayDepth(awayData.starters || []);
      setLoading(false);
    } catch (error) {
      console.error('Error fetching game details:', error);
      setLoading(false);
    }
  });

  return (
    <div class="fixed inset-0 z-50 bg-black bg-opacity-90 overflow-auto">
      {/* Close Button */}
      <button
        onClick={props.onClose}
        class="absolute top-4 right-4 text-white text-3xl hover:text-red-500 z-50"
      >
        ✕
      </button>

      <div class="min-h-screen p-6">
        <div class="max-w-6xl mx-auto">
          {/* Game Header */}
          <div class="text-center mb-8">
            <div class="text-gray-400 text-lg mb-2">
              {new Date(props.game.date).toLocaleDateString('en-US', { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' })}
              {props.game.time && ` • ${props.game.time}`}
            </div>
            <h1 class="text-5xl font-bold text-white mb-4">
              {props.game.away_team.name} @ {props.game.home_team.name}
            </h1>
            <div class="flex items-center justify-center gap-6 mb-4">
              <span class={`px-4 py-2 rounded-lg font-semibold ${
                props.game.status === 'Final' ? 'bg-gray-600' :
                props.game.status === 'Live' ? 'bg-red-600 animate-pulse' :
                'bg-blue-600'
              } text-white`}>
                {props.game.status}
              </span>
              <Show when={props.game.tv}>
                <span class="text-gray-400">📺 {props.game.tv}</span>
              </Show>
            </div>
            <div class="text-gray-400">📍 {props.game.arena}</div>
          </div>

          {/* Score (if available) */}
          <Show when={props.game.score}>
            <div class="bg-gray-900 rounded-lg p-8 border border-gray-800 mb-8">
              <div class="grid grid-cols-2 gap-12">
                <div class="text-center">
                  <div class="text-gray-400 text-xl mb-2">{props.game.away_team.abbr}</div>
                  <div class="text-7xl font-bold text-white">{props.game.score!.away}</div>
                </div>
                <div class="text-center">
                  <div class="text-gray-400 text-xl mb-2">{props.game.home_team.abbr}</div>
                  <div class="text-7xl font-bold text-white">{props.game.score!.home}</div>
                </div>
              </div>
            </div>
          </Show>

          <Show when={loading()}>
            <div class="text-center text-white">Loading lineups...</div>
          </Show>

          <Show when={!loading()}>
            {/* Projected Lineups */}
            <div class="grid grid-cols-1 md:grid-cols-2 gap-8 mb-8">
              {/* Away Team Lineup */}
              <div class="bg-gray-900 rounded-lg p-6 border border-gray-800">
                <div class="flex items-center gap-3 mb-6">
                  <Show when={props.game.away_team.logo}>
                    <img src={props.game.away_team.logo} alt={props.game.away_team.abbr} class="w-16 h-16" />
                  </Show>
                  <div>
                    <h2 class="text-2xl font-bold text-white">{props.game.away_team.name}</h2>
                    <p class="text-gray-400">Projected Lineup</p>
                  </div>
                </div>
                <div class="space-y-3">
                  <For each={awayDepth().slice(0, 5)}>
                    {(player) => (
                      <div class="bg-gray-800 rounded p-3 flex justify-between items-center">
                        <div>
                          <div class="text-white font-semibold">{player.name}</div>
                          <div class="text-gray-400 text-sm">{player.position} • #{player.jersey}</div>
                          <Show when={player.injury_status}>
                            <span class="px-2 py-1 bg-red-600 text-white text-xs rounded mt-1 inline-block">
                              {player.injury_status}
                            </span>
                          </Show>
                        </div>
                        <div class="text-right text-sm">
                          <div class="text-white font-bold">{player.ppg?.toFixed(1) || '0.0'} PPG</div>
                          <div class="text-gray-400">{player.mpg?.toFixed(1) || '0.0'} MPG</div>
                        </div>
                      </div>
                    )}
                  </For>
                </div>
              </div>

              {/* Home Team Lineup */}
              <div class="bg-gray-900 rounded-lg p-6 border border-gray-800">
                <div class="flex items-center gap-3 mb-6">
                  <Show when={props.game.home_team.logo}>
                    <img src={props.game.home_team.logo} alt={props.game.home_team.abbr} class="w-16 h-16" />
                  </Show>
                  <div>
                    <h2 class="text-2xl font-bold text-white">{props.game.home_team.name}</h2>
                    <p class="text-gray-400">Projected Lineup</p>
                  </div>
                </div>
                <div class="space-y-3">
                  <For each={homeDepth().slice(0, 5)}>
                    {(player) => (
                      <div class="bg-gray-800 rounded p-3 flex justify-between items-center">
                        <div>
                          <div class="text-white font-semibold">{player.name}</div>
                          <div class="text-gray-400 text-sm">{player.position} • #{player.jersey}</div>
                          <Show when={player.injury_status}>
                            <span class="px-2 py-1 bg-red-600 text-white text-xs rounded mt-1 inline-block">
                              {player.injury_status}
                            </span>
                          </Show>
                        </div>
                        <div class="text-right text-sm">
                          <div class="text-white font-bold">{player.ppg?.toFixed(1) || '0.0'} PPG</div>
                          <div class="text-gray-400">{player.mpg?.toFixed(1) || '0.0'} MPG</div>
                        </div>
                      </div>
                    )}
                  </For>
                </div>
              </div>
            </div>

            {/* Additional Stats */}
            <div class="bg-gray-900 rounded-lg p-6 border border-gray-800">
              <h3 class="text-2xl font-bold text-white mb-4">📊 Matchup Insights</h3>
              <div class="text-gray-400">
                <p class="mb-2">• Projected starters shown above</p>
                <p class="mb-2">• Injury status updated daily at 3:30 AM UTC</p>
                <p class="mb-2">• Stats based on season averages (2025-26)</p>
                <Show when={props.game.status === 'Scheduled'}>
                  <p class="mt-4 text-blue-400 font-semibold">ML prediction available at Q2 6:00 during live game</p>
                </Show>
              </div>
            </div>
          </Show>
        </div>
      </div>
    </div>
  );
};

export default ScheduleGameModal;

