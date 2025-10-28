/**
 * PROFESSIONAL PLAYER PROFILE CARD
 * Shows player as independent probability element with comprehensive stats
 * 
 * Displays:
 * - Player image + team logo
 * - Bio information
 * - Season stats with graphs
 * - Advanced metrics (self-computed!)
 * - Probability analysis
 * - Last 10 games trend
 */

import { type Component, Show, For, createSignal, onMount } from 'solid-js';

interface PlayerProfile {
  player_id: string;
  name: string;
  first_name: string;
  last_name: string;
  headshot_url: string;
  action_photo_url: string;
  jersey: string;
  position: string;
  height: string;
  weight: number;
  age: number;
  country: string;
  experience: number;
  draft_year: number;
  draft_round: number;
  draft_number: number;
  college: string;
  team: {
    abbreviation: string;
    full_name: string;
    logo_url: string;
    primary_color: string;
    secondary_color: string;
  };
  season_stats: {
    games_played: number;
    ppg: number;
    rpg: number;
    apg: number;
    fg_pct: number;
    fg3_pct: number;
    ft_pct: number;
  };
  advanced_stats: {
    ts_pct: number;
    efg_pct: number;
    pts_100: number;
    reb_100: number;
    ast_100: number;
    pts_36: number;
    reb_36: number;
    ast_36: number;
    lebron: number;
    lebron_offense: number;
    lebron_defense: number;
    rapm: number;
  };
  last10_games: Array<{
    date: string;
    opponent: string;
    pts: number;
    reb: number;
    ast: number;
    minutes: number;
    plus_minus: number;
    pts_100: number;
    ts_pct: number;
  }>;
  probability_metrics: {
    sample_size: number;
    scoring_variance: number;
    consistency_score: number;
  };
}

interface Props {
  playerId: string;
}

const PlayerProfileCard: Component<Props> = (props) => {
  const [player, setPlayer] = createSignal<PlayerProfile | null>(null);
  const [loading, setLoading] = createSignal(true);

  onMount(async () => {
    try {
      // Fetch from Railway backend
      const response = await fetch(`https://ol24-production.up.railway.app/api/stats/player/${props.playerId}`);
      const data = await response.json();
      setPlayer(data);
    } catch (error) {
      console.error('Failed to load player:', error);
    } finally {
      setLoading(false);
    }
  });

  const getStatColor = (value: number, good_threshold: number, great_threshold: number) => {
    if (value >= great_threshold) return 'text-green-400';
    if (value >= good_threshold) return 'text-yellow-400';
    return 'text-gray-400';
  };

  const getLebronColor = (lebron: number) => {
    if (lebron >= 5) return 'text-green-400';
    if (lebron >= 0) return 'text-yellow-400';
    return 'text-red-400';
  };

  return (
    <Show when={!loading() && player()} fallback={<div class="text-white">Loading player...</div>}>
      <div class="bg-gray-900 rounded-2xl overflow-hidden border border-gray-800 shadow-2xl">
        {/* Header with Team Colors */}
        <div 
          class="relative h-48 flex items-end p-6"
          style={{
            background: `linear-gradient(135deg, ${player()!.team.primary_color} 0%, ${player()!.team.secondary_color} 100%)`
          }}
        >
          {/* Player Headshot */}
          <div class="absolute top-4 right-4 w-32 h-32 rounded-full overflow-hidden border-4 border-white shadow-lg">
            <img 
              src={player()!.headshot_url} 
              alt={player()!.name}
              class="w-full h-full object-cover"
              onError={(e) => {
                // Fallback to placeholder
                e.currentTarget.src = 'https://via.placeholder.com/128?text=NBA';
              }}
            />
          </div>

          {/* Team Logo */}
          <div class="absolute top-4 left-4 w-16 h-16">
            <img 
              src={player()!.team.logo_url} 
              alt={player()!.team.abbreviation}
              class="w-full h-full object-contain drop-shadow-lg"
            />
          </div>

          {/* Player Name + Jersey */}
          <div class="text-white">
            <div class="text-sm font-semibold opacity-90">{player()!.team.full_name}</div>
            <h1 class="text-4xl font-bold">{player()!.name}</h1>
            <div class="flex items-center gap-3 mt-1">
              <span class="text-lg">#{player()!.jersey}</span>
              <span class="text-lg">•</span>
              <span class="text-lg">{player()!.position}</span>
              <span class="text-lg">•</span>
              <span class="text-lg">{player()!.height} • {player()!.weight} lbs</span>
            </div>
          </div>
        </div>

        {/* Stats Grid */}
        <div class="p-6 grid grid-cols-3 gap-4">
          {/* Per-Game Stats */}
          <div class="bg-gray-800/50 rounded-lg p-4">
            <div class="text-xs text-gray-400 mb-2">PER GAME</div>
            <div class="space-y-2">
              <div class="flex justify-between">
                <span class="text-gray-300">PPG</span>
                <span class={`font-bold ${getStatColor(player()!.season_stats.ppg, 15, 25)}`}>
                  {player()!.season_stats.ppg.toFixed(1)}
                </span>
              </div>
              <div class="flex justify-between">
                <span class="text-gray-300">RPG</span>
                <span class="text-white font-bold">{player()!.season_stats.rpg.toFixed(1)}</span>
              </div>
              <div class="flex justify-between">
                <span class="text-gray-300">APG</span>
                <span class="text-white font-bold">{player()!.season_stats.apg.toFixed(1)}</span>
              </div>
            </div>
          </div>

          {/* Per-100 Stats (Self-Computed!) */}
          <div class="bg-gray-800/50 rounded-lg p-4">
            <div class="text-xs text-gray-400 mb-2">PER 100 POSS</div>
            <div class="space-y-2">
              <div class="flex justify-between">
                <span class="text-gray-300">PTS</span>
                <span class="text-blue-400 font-bold">{player()!.advanced_stats.pts_100.toFixed(1)}</span>
              </div>
              <div class="flex justify-between">
                <span class="text-gray-300">REB</span>
                <span class="text-blue-400 font-bold">{player()!.advanced_stats.reb_100.toFixed(1)}</span>
              </div>
              <div class="flex justify-between">
                <span class="text-gray-300">AST</span>
                <span class="text-blue-400 font-bold">{player()!.advanced_stats.ast_100.toFixed(1)}</span>
              </div>
            </div>
          </div>

          {/* Impact Metrics (LEBRON!) */}
          <div class="bg-gray-800/50 rounded-lg p-4">
            <div class="text-xs text-gray-400 mb-2">IMPACT (LEBRON)</div>
            <div class="space-y-2">
              <div class="flex justify-between">
                <span class="text-gray-300">Total</span>
                <span class={`font-bold text-xl ${getLebronColor(player()!.advanced_stats.lebron)}`}>
                  {player()!.advanced_stats.lebron > 0 ? '+' : ''}
                  {player()!.advanced_stats.lebron.toFixed(1)}
                </span>
              </div>
              <div class="flex justify-between text-sm">
                <span class="text-gray-400">Offense</span>
                <span class="text-green-400">+{player()!.advanced_stats.lebron_offense.toFixed(1)}</span>
              </div>
              <div class="flex justify-between text-sm">
                <span class="text-gray-400">Defense</span>
                <span class="text-red-400">+{player()!.advanced_stats.lebron_defense.toFixed(1)}</span>
              </div>
            </div>
          </div>
        </div>

        {/* Shooting Efficiency */}
        <div class="px-6 pb-4">
          <div class="bg-gray-800/50 rounded-lg p-4">
            <div class="text-xs text-gray-400 mb-3">EFFICIENCY (SELF-COMPUTED)</div>
            <div class="grid grid-cols-3 gap-4">
              <div class="text-center">
                <div class="text-2xl font-bold text-purple-400">
                  {(player()!.advanced_stats.ts_pct * 100).toFixed(1)}%
                </div>
                <div class="text-xs text-gray-400">TS%</div>
              </div>
              <div class="text-center">
                <div class="text-2xl font-bold text-purple-400">
                  {(player()!.advanced_stats.efg_pct * 100).toFixed(1)}%
                </div>
                <div class="text-xs text-gray-400">eFG%</div>
              </div>
              <div class="text-center">
                <div class="text-2xl font-bold text-purple-400">
                  {(player()!.season_stats.fg_pct * 100).toFixed(1)}%
                </div>
                <div class="text-xs text-gray-400">FG%</div>
              </div>
            </div>
          </div>
        </div>

        {/* Probability Analysis (Independent Math Problem!) */}
        <div class="px-6 pb-4">
          <div class="bg-gradient-to-r from-blue-900/50 to-purple-900/50 rounded-lg p-4 border border-blue-500/30">
            <div class="text-xs text-gray-300 mb-3">📊 PROBABILITY ANALYSIS</div>
            <div class="grid grid-cols-3 gap-4 text-center">
              <div>
                <div class="text-lg font-bold text-white">{player()!.probability_metrics.sample_size}</div>
                <div class="text-xs text-gray-400">Sample Size</div>
              </div>
              <div>
                <div class="text-lg font-bold text-yellow-400">
                  σ² = {player()!.probability_metrics.scoring_variance.toFixed(1)}
                </div>
                <div class="text-xs text-gray-400">Variance</div>
              </div>
              <div>
                <div class="text-lg font-bold text-green-400">
                  {(player()!.probability_metrics.consistency_score * 100).toFixed(0)}%
                </div>
                <div class="text-xs text-gray-400">Consistency</div>
              </div>
            </div>
            <div class="mt-3 text-xs text-gray-400 leading-relaxed">
              <strong>Independent Probability Model:</strong> With n={player()!.probability_metrics.sample_size} games, 
              variance σ²={player()!.probability_metrics.scoring_variance.toFixed(1)}, and consistency score {(player()!.probability_metrics.consistency_score * 100).toFixed(0)}%, 
              we can model this player's scoring as a random variable X ~ N(μ={player()!.season_stats.ppg.toFixed(1)}, σ²={player()!.probability_metrics.scoring_variance.toFixed(1)}).
            </div>
          </div>
        </div>

        {/* Last 10 Games Chart */}
        <div class="px-6 pb-6">
          <div class="bg-gray-800/50 rounded-lg p-4">
            <div class="text-xs text-gray-400 mb-3">LAST 10 GAMES (TREND)</div>
            <div class="grid grid-cols-10 gap-1 items-end h-24">
              <For each={player()!.last10_games.slice(0, 10).reverse()}>
                {(game) => {
                  const maxPts = Math.max(...player()!.last10_games.map(g => g.pts));
                  const height = (game.pts / maxPts) * 100;
                  const isGood = game.pts >= player()!.season_stats.ppg;
                  
                  return (
                    <div 
                      class={`rounded-t relative group ${isGood ? 'bg-green-500' : 'bg-red-500'}`}
                      style={{ height: `${height}%` }}
                      title={`vs ${game.opponent}: ${game.pts} pts, +${game.plus_minus}`}
                    >
                      {/* Tooltip on hover */}
                      <div class="absolute bottom-full mb-1 left-1/2 transform -translate-x-1/2 hidden group-hover:block bg-black text-white text-xs rounded py-1 px-2 whitespace-nowrap z-10">
                        <div>vs {game.opponent}</div>
                        <div>{game.pts} / {game.reb} / {game.ast}</div>
                        <div>+/- {game.plus_minus}</div>
                      </div>
                    </div>
                  );
                }}
              </For>
            </div>
            <div class="flex justify-between mt-2 text-xs text-gray-500">
              <span>10 games ago</span>
              <span>Recent</span>
            </div>
          </div>
        </div>

        {/* Detailed Stats Table */}
        <div class="px-6 pb-6">
          <div class="bg-gray-800/50 rounded-lg overflow-hidden">
            <table class="w-full text-sm">
              <thead class="bg-gray-700/50">
                <tr>
                  <th class="text-left p-3 text-gray-400">Date</th>
                  <th class="text-left p-3 text-gray-400">OPP</th>
                  <th class="text-right p-3 text-gray-400">PTS</th>
                  <th class="text-right p-3 text-gray-400">REB</th>
                  <th class="text-right p-3 text-gray-400">AST</th>
                  <th class="text-right p-3 text-gray-400">+/-</th>
                  <th class="text-right p-3 text-gray-400">PTS/100</th>
                </tr>
              </thead>
              <tbody>
                <For each={player()!.last10_games.slice(0, 10)}>
                  {(game) => (
                    <tr class="border-t border-gray-700/50 hover:bg-gray-700/30">
                      <td class="p-3 text-gray-300">{new Date(game.date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}</td>
                      <td class="p-3 text-gray-300">{game.opponent}</td>
                      <td class="p-3 text-right text-white font-bold">{game.pts}</td>
                      <td class="p-3 text-right text-white">{game.reb}</td>
                      <td class="p-3 text-right text-white">{game.ast}</td>
                      <td class={`p-3 text-right font-bold ${game.plus_minus >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                        {game.plus_minus > 0 ? '+' : ''}{game.plus_minus}
                      </td>
                      <td class="p-3 text-right text-blue-400">{game.pts_100.toFixed(1)}</td>
                    </tr>
                  )}
                </For>
              </tbody>
            </table>
          </div>
        </div>

        {/* Bio Footer */}
        <div class="px-6 pb-6 text-sm text-gray-400 border-t border-gray-800 pt-4">
          <div class="grid grid-cols-2 gap-2">
            <div><strong>Draft:</strong> {player()!.draft_year} R{player()!.draft_round} #{player()!.draft_number}</div>
            <div><strong>College:</strong> {player()!.college || 'N/A'}</div>
            <div><strong>Experience:</strong> {player()!.experience} years</div>
            <div><strong>Country:</strong> {player()!.country}</div>
          </div>
        </div>
      </div>
    </Show>
  );
};

export default PlayerProfileCard;

