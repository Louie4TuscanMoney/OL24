import { createSignal, onMount, onCleanup, For, Show } from 'solid-js';
import axios from 'axios';
import './App.css';
import BasketballCourt3D from './components/BasketballCourt3D';
import BetTrackingModal from './components/BetTrackingModal';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8001';

interface Game {
  game_id: string;
  home_team: string;
  away_team: string;
  home_score: number;
  away_score: number;
  period: number;
  clock: string;
  status_text: string;
  current_diff: number;
  can_predict: boolean;
  is_q2_6min: boolean;
}

interface Opportunity {
  matchup: string;
  current_score: string;
  period: string;
  prediction: number;
  market_spread: number;
  edge: number;
  p_win: number;
  recommended_stake: number;
  bet_line: string;
  confidence_interval: [number, number];
}

interface RiskStatus {
  current_bankroll: number;
  peak_bankroll: number;
  current_drawdown: number;
  daily_loss: number;
  can_bet: boolean;
  alerts: string[];
  open_positions: number;
}

interface PortfolioSummary {
  total_bets: number;
  wins: number;
  losses: number;
  win_rate: number;
  total_profit: number;
  roi: number;
}

function App() {
  const [liveGames, setLiveGames] = createSignal<Game[]>([]);
  const [opportunities, setOpportunities] = createSignal<Opportunity[]>([]);
  const [riskStatus, setRiskStatus] = createSignal<RiskStatus | null>(null);
  const [portfolioSummary, setPortfolioSummary] = createSignal<PortfolioSummary | null>(null);
  const [lastUpdate, setLastUpdate] = createSignal<string>('');
  const [isConnected, setIsConnected] = createSignal(false);
  const [showBetModal, setShowBetModal] = createSignal(false);
  const [selectedOpp, setSelectedOpp] = createSignal<Opportunity | null>(null);
  const [show3DCourt, setShow3DCourt] = createSignal(false);
  const [current3DData, setCurrent3DData] = createSignal<any>(null);

  const fetchData = async () => {
    try {
      const [gamesRes, oppsRes, riskRes, portfolioRes] = await Promise.all([
        axios.get(`${API_URL}/api/live-games`).catch(() => ({ data: { games: [] } })),
        axios.get(`${API_URL}/api/opportunities`).catch(() => ({ data: { opportunities: [] } })),
        axios.get(`${API_URL}/api/risk-status`).catch(() => ({ data: { risk_status: null } })),
        axios.get(`${API_URL}/api/bets/summary`).catch(() => ({ data: null }))
      ]);

      setLiveGames(gamesRes.data.games || []);
      setOpportunities(oppsRes.data.opportunities || []);
      setRiskStatus(riskRes.data.risk_status);
      setPortfolioSummary(portfolioRes.data);
      setLastUpdate(new Date().toLocaleTimeString());
      setIsConnected(true);
    } catch (error) {
      console.error('Connection error:', error);
      setIsConnected(false);
    }
  };

  const fetch3DData = async (gameId: string) => {
    try {
      const res = await axios.get(`${API_URL}/api/court/3d/${gameId}`);
      setCurrent3DData(res.data.current_frame);
    } catch (error) {
      console.error('Error fetching 3D data:', error);
    }
  };

  let interval: number;
  onMount(() => {
    fetchData();
    interval = setInterval(fetchData, 10000) as unknown as number;
  });

  onCleanup(() => clearInterval(interval));

  const openBetModal = (opp: Opportunity) => {
    setSelectedOpp(opp);
    setShowBetModal(true);
  };

  const toggle3DCourt = (game: Game) => {
    setShow3DCourt(!show3DCourt());
    if (!show3DCourt() && game.game_id) {
      fetch3DData(game.game_id);
    }
  };

  const profitColor = () => {
    const status = riskStatus();
    if (!status) return '#10b981';
    const profit = -status.daily_loss;
    return profit >= 0 ? '#10b981' : '#ef4444';
  };

  return (
    <div class="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      {/* Animated Background */}
      <div class="fixed inset-0 overflow-hidden pointer-events-none">
        <div class="absolute top-0 left-1/4 w-96 h-96 bg-purple-500 rounded-full mix-blend-multiply filter blur-3xl opacity-20 animate-blob"></div>
        <div class="absolute top-0 right-1/4 w-96 h-96 bg-yellow-500 rounded-full mix-blend-multiply filter blur-3xl opacity-20 animate-blob animation-delay-2000"></div>
        <div class="absolute bottom-0 left-1/3 w-96 h-96 bg-pink-500 rounded-full mix-blend-multiply filter blur-3xl opacity-20 animate-blob animation-delay-4000"></div>
      </div>

      {/* Main Container */}
      <div class="relative z-10 container mx-auto px-4 py-6 max-w-7xl">
        
        {/* Header */}
        <header class="mb-8">
          <div class="bg-white/10 backdrop-blur-xl rounded-3xl p-8 border border-white/20 shadow-2xl">
            <div class="flex justify-between items-start flex-wrap gap-4">
              <div>
                <h1 class="text-5xl font-black text-white mb-2 tracking-tight">
                  <span class="bg-gradient-to-r from-yellow-400 via-orange-500 to-red-500 bg-clip-text text-transparent">
                    ONTOLOGIC XYZ
                  </span>
                </h1>
                <p class="text-white/70 text-lg">
                  NBA Live Trading • Mamba Mentality System • 9.0 MAE
                </p>
                <div class="flex gap-4 mt-3 flex-wrap">
                  <span class={`px-3 py-1 rounded-full text-sm font-semibold ${
                    isConnected() ? 'bg-green-500/20 text-green-300' : 'bg-red-500/20 text-red-300'
                  }`}>
                    {isConnected() ? '🟢 LIVE' : '🔴 OFFLINE'}
                  </span>
                  <span class="px-3 py-1 bg-blue-500/20 text-blue-300 rounded-full text-sm">
                    Updated {lastUpdate()}
                  </span>
                  <button 
                    onClick={() => setShowBetModal(true)}
                    class="px-4 py-1 bg-gradient-to-r from-purple-500 to-pink-500 text-white rounded-full text-sm font-semibold hover:shadow-lg transition-all"
                  >
                    📝 Log Bet
                  </button>
                </div>
              </div>
              
              <div class="text-right">
                <div class="text-white/60 text-sm mb-1">SYSTEM STATUS</div>
                <div class="text-3xl font-black text-white">AUTONOMOUS</div>
                <div class="text-white/50 text-xs mt-1">Running 24/7</div>
              </div>
            </div>
          </div>
        </header>

        {/* 3D Court Visualization */}
        <Show when={show3DCourt()}>
          <section class="mb-8">
            <div class="bg-white/10 backdrop-blur-xl rounded-3xl p-8 border border-white/20 shadow-2xl">
              <div class="flex justify-between items-center mb-6">
                <h2 class="text-3xl font-black text-white flex items-center gap-3">
                  <span class="text-4xl">🏀</span>
                  <span>Live 3D Court View</span>
                </h2>
                <button 
                  onClick={() => setShow3DCourt(false)}
                  class="text-white/60 hover:text-white text-2xl"
                >
                  ✕
                </button>
              </div>
              <div class="h-[600px]">
                <BasketballCourt3D gameData={current3DData()} />
              </div>
            </div>
          </section>
        </Show>

        {/* Portfolio Summary */}
        <Show when={portfolioSummary() && portfolioSummary()!.total_bets > 0}>
          <section class="mb-8">
            <div class="bg-gradient-to-br from-blue-500/20 to-purple-500/20 backdrop-blur-xl rounded-3xl p-8 border-2 border-blue-500/50 shadow-2xl">
              <h2 class="text-2xl font-bold text-white mb-6 flex items-center gap-3">
                <span>📊</span>
                <span>My Bet Portfolio</span>
              </h2>
              
              <div class="grid grid-cols-2 md:grid-cols-5 gap-6">
                <div class="bg-black/30 rounded-2xl p-5 border border-white/10">
                  <div class="text-white/60 text-sm mb-2">TOTAL BETS</div>
                  <div class="text-4xl font-black text-white">
                    {portfolioSummary()!.total_bets}
                  </div>
                </div>
                <div class="bg-black/30 rounded-2xl p-5 border border-white/10">
                  <div class="text-white/60 text-sm mb-2">WIN RATE</div>
                  <div class="text-4xl font-black text-green-400">
                    {(portfolioSummary()!.win_rate * 100).toFixed(1)}%
                  </div>
                </div>
                <div class="bg-black/30 rounded-2xl p-5 border border-white/10">
                  <div class="text-white/60 text-sm mb-2">TOTAL PROFIT</div>
                  <div class={`text-4xl font-black ${portfolioSummary()!.total_profit >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                    ${portfolioSummary()!.total_profit.toLocaleString()}
                  </div>
                </div>
                <div class="bg-black/30 rounded-2xl p-5 border border-white/10">
                  <div class="text-white/60 text-sm mb-2">ROI</div>
                  <div class={`text-4xl font-black ${portfolioSummary()!.roi >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                    {(portfolioSummary()!.roi * 100).toFixed(1)}%
                  </div>
                </div>
                <div class="bg-black/30 rounded-2xl p-5 border border-white/10">
                  <div class="text-white/60 text-sm mb-2">RECORD</div>
                  <div class="text-2xl font-black text-white">
                    {portfolioSummary()!.wins}-{portfolioSummary()!.losses}
                  </div>
                </div>
              </div>
            </div>
          </section>
        </Show>

        {/* Risk Dashboard */}
        <Show when={riskStatus()}>
          <section class="mb-8">
            <div class={`bg-gradient-to-br ${
              riskStatus()!.can_bet 
                ? 'from-green-500/20 to-emerald-500/20 border-green-500/50' 
                : 'from-red-500/20 to-rose-500/20 border-red-500/50'
            } backdrop-blur-xl rounded-3xl p-8 border-2 shadow-2xl`}>
              <h2 class="text-2xl font-bold text-white mb-6 flex items-center gap-3">
                <span>🛡️</span>
                <span>Risk Management</span>
                <span class={`ml-auto text-xl ${riskStatus()!.can_bet ? 'text-green-400' : 'text-red-400'}`}>
                  {riskStatus()!.can_bet ? '✅ CLEARED TO BET' : '🚨 BETTING HALTED'}
                </span>
              </h2>
              
              <div class="grid grid-cols-2 md:grid-cols-4 gap-6">
                <div class="bg-black/30 rounded-2xl p-5 border border-white/10">
                  <div class="text-white/60 text-sm mb-2">BANKROLL</div>
                  <div class="text-4xl font-black text-white">
                    ${riskStatus()!.current_bankroll.toLocaleString()}
                  </div>
                  <div class="text-white/40 text-xs mt-1">
                    Peak: ${riskStatus()!.peak_bankroll.toLocaleString()}
                  </div>
                </div>
                
                <div class="bg-black/30 rounded-2xl p-5 border border-white/10">
                  <div class="text-white/60 text-sm mb-2">DRAWDOWN</div>
                  <div class={`text-4xl font-black ${
                    riskStatus()!.current_drawdown > 0.15 ? 'text-red-400' : 'text-white'
                  }`}>
                    {(riskStatus()!.current_drawdown * 100).toFixed(1)}%
                  </div>
                  <div class="text-white/40 text-xs mt-1">Limit: 30%</div>
                </div>
                
                <div class="bg-black/30 rounded-2xl p-5 border border-white/10">
                  <div class="text-white/60 text-sm mb-2">DAILY P&L</div>
                  <div class="text-4xl font-black" style={{ color: profitColor() }}>
                    ${(-riskStatus()!.daily_loss).toLocaleString()}
                  </div>
                  <div class="text-white/40 text-xs mt-1">Limit: $1,000</div>
                </div>
                
                <div class="bg-black/30 rounded-2xl p-5 border border-white/10">
                  <div class="text-white/60 text-sm mb-2">POSITIONS</div>
                  <div class="text-4xl font-black text-white">
                    {riskStatus()!.open_positions || 0}
                  </div>
                  <div class="text-white/40 text-xs mt-1">Max: 5</div>
                </div>
              </div>

              <Show when={riskStatus()!.alerts && riskStatus()!.alerts.length > 0}>
                <div class="mt-6 bg-red-500/20 border border-red-500/50 rounded-2xl p-4">
                  <div class="font-bold text-red-300 mb-2">⚠️ RISK ALERTS:</div>
                  <For each={riskStatus()!.alerts}>
                    {(alert) => <div class="text-red-200 text-sm">{alert}</div>}
                  </For>
                </div>
              </Show>
            </div>
          </section>
        </Show>

        {/* Betting Opportunities */}
        <section class="mb-8">
          <div class="bg-white/10 backdrop-blur-xl rounded-3xl p-8 border border-white/20 shadow-2xl">
            <div class="flex justify-between items-center mb-6">
              <h2 class="text-3xl font-black text-white flex items-center gap-3">
                <span class="text-4xl">🎯</span>
                <span>Betting Opportunities</span>
              </h2>
              <div class="bg-gradient-to-r from-yellow-500 to-orange-500 text-white font-black px-4 py-2 rounded-full text-xl">
                {opportunities().length}
              </div>
            </div>
            
            <Show when={opportunities().length === 0}>
              <div class="text-center py-16">
                <div class="text-6xl mb-4">🎲</div>
                <div class="text-2xl font-bold text-white/80 mb-2">No Opportunities Yet</div>
                <div class="text-white/50">
                  Waiting for games with Edge ≥ 5 pts and P(Win) ≥ 55%
                </div>
                <div class="text-white/40 text-sm mt-4">
                  System checking autonomously every 30 seconds...
                </div>
              </div>
            </Show>

            <div class="grid gap-6">
              <For each={opportunities()}>
                {(opp) => (
                  <div class="bg-gradient-to-br from-green-500/20 to-blue-500/20 backdrop-blur-lg rounded-2xl p-6 border-2 border-green-500/50 shadow-xl hover:shadow-2xl transition-all duration-300 hover:scale-[1.02]">
                    {/* Header */}
                    <div class="flex justify-between items-start mb-6">
                      <div>
                        <div class="text-2xl font-black text-white mb-2">
                          {opp.matchup}
                        </div>
                        <div class="text-white/70">
                          {opp.current_score} • {opp.period}
                        </div>
                      </div>
                      <div class="text-right">
                        <div class="text-5xl font-black bg-gradient-to-r from-yellow-400 to-orange-500 bg-clip-text text-transparent">
                          {opp.edge.toFixed(1)}
                        </div>
                        <div class="text-white/60 font-semibold">EDGE PTS</div>
                      </div>
                    </div>

                    {/* Metrics Grid */}
                    <div class="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                      <div class="bg-black/40 rounded-xl p-4 border border-white/10">
                        <div class="text-white/50 text-xs mb-1">OUR PREDICTION</div>
                        <div class="text-2xl font-bold text-white">
                          {opp.prediction > 0 ? '+' : ''}{opp.prediction.toFixed(1)}
                        </div>
                      </div>
                      <div class="bg-black/40 rounded-xl p-4 border border-white/10">
                        <div class="text-white/50 text-xs mb-1">MARKET SPREAD</div>
                        <div class="text-2xl font-bold text-white">
                          {opp.market_spread > 0 ? '+' : ''}{opp.market_spread.toFixed(1)}
                        </div>
                      </div>
                      <div class="bg-black/40 rounded-xl p-4 border border-white/10">
                        <div class="text-white/50 text-xs mb-1">WIN PROBABILITY</div>
                        <div class="text-2xl font-bold text-green-400">
                          {(opp.p_win * 100).toFixed(1)}%
                        </div>
                      </div>
                      <div class="bg-black/40 rounded-xl p-4 border border-white/10">
                        <div class="text-white/50 text-xs mb-1">CONFIDENCE</div>
                        <div class="text-2xl font-bold text-blue-400">
                          {opp.confidence_interval ? 
                            `±${((opp.confidence_interval[1] - opp.confidence_interval[0]) / 2).toFixed(1)}` 
                            : 'N/A'}
                        </div>
                      </div>
                    </div>

                    {/* Recommended Bet */}
                    <div class="bg-gradient-to-r from-yellow-500/30 to-orange-500/30 rounded-2xl p-6 mb-6 border border-yellow-500/50">
                      <div class="text-white/70 text-sm mb-2">✅ RECOMMENDED BET</div>
                      <div class="text-3xl font-black text-white">
                        {opp.bet_line}
                      </div>
                    </div>

                    {/* Action */}
                    <div class="flex justify-between items-center">
                      <div>
                        <div class="text-white/60 text-sm">KELLY-OPTIMAL STAKE</div>
                        <div class="text-3xl font-black text-white">
                          ${opp.recommended_stake.toLocaleString()}
                        </div>
                      </div>
                      <button 
                        onClick={() => openBetModal(opp)}
                        class="bg-gradient-to-r from-green-500 to-emerald-600 hover:from-green-600 hover:to-emerald-700 text-white font-black px-8 py-4 rounded-xl text-xl shadow-lg hover:shadow-xl transition-all duration-200 hover:scale-105"
                      >
                        PLACE BET 💰
                      </button>
                    </div>
                  </div>
                )}
              </For>
            </div>
          </div>
        </section>

        {/* Live Games */}
        <section>
          <div class="bg-white/10 backdrop-blur-xl rounded-3xl p-8 border border-white/20 shadow-2xl">
            <div class="flex justify-between items-center mb-6">
              <h2 class="text-3xl font-black text-white flex items-center gap-3">
                <span class="text-4xl">🏀</span>
                <span>Live NBA Games</span>
              </h2>
              <div class="flex items-center gap-3">
                <span class="bg-purple-500/30 text-purple-200 px-4 py-1 rounded-full text-xl font-black">
                  {liveGames().length}
                </span>
              </div>
            </div>
            
            <Show when={liveGames().length === 0}>
              <div class="text-center py-16">
                <div class="text-6xl mb-4">🌙</div>
                <div class="text-2xl font-bold text-white/80 mb-2">No Live Games</div>
                <div class="text-white/50">
                  Dashboard will activate when NBA games start
                </div>
              </div>
            </Show>

            <div class="grid gap-4">
              <For each={liveGames()}>
                {(game) => (
                  <div class={`
                    bg-gradient-to-r rounded-2xl p-6 border-2 transition-all duration-300
                    ${game.status_text === 'LIVE' 
                      ? 'from-red-500/20 to-orange-500/20 border-red-500/50 shadow-lg shadow-red-500/20' 
                      : 'from-slate-500/20 to-slate-600/20 border-slate-500/30'}
                    ${game.is_q2_6min ? 'ring-4 ring-yellow-500/50 shadow-xl shadow-yellow-500/30' : ''}
                    hover:scale-[1.02]
                  `}>
                    <div class="flex justify-between items-center">
                      <div class="flex items-center gap-4">
                        <div>
                          {game.status_text === 'LIVE' && <span class="text-2xl">🔴</span>}
                          {game.is_q2_6min && <span class="text-2xl">⭐</span>}
                          {game.can_predict && <span class="text-2xl">🎯</span>}
                        </div>
                        <div>
                          <div class="text-2xl font-black text-white">
                            {game.away_team} @ {game.home_team}
                          </div>
                          <div class="text-white/70 mt-1 flex items-center gap-3">
                            <span>Q{game.period} {game.clock} • {game.status_text}</span>
                            {game.is_q2_6min && <span class="text-yellow-400 font-bold">⚡ PREDICTION POINT</span>}
                            {game.status_text === 'LIVE' && (
                              <button 
                                onClick={() => toggle3DCourt(game)}
                                class="text-sm bg-purple-500/30 hover:bg-purple-500/50 px-3 py-1 rounded-full transition-all"
                              >
                                🏀 View 3D Court
                              </button>
                            )}
                          </div>
                        </div>
                      </div>
                      
                      <div class="text-right">
                        <div class="text-4xl font-black text-white">
                          {game.away_score} - {game.home_score}
                        </div>
                        <div class={`text-lg font-semibold mt-1 ${
                          game.current_diff > 0 ? 'text-green-400' : game.current_diff < 0 ? 'text-red-400' : 'text-white/60'
                        }`}>
                          {game.current_diff > 0 ? '+' : ''}{game.current_diff}
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </For>
            </div>
          </div>
        </section>

        {/* Footer */}
        <footer class="mt-8 text-center">
          <div class="bg-white/5 backdrop-blur-lg rounded-2xl p-6 border border-white/10">
            <div class="text-white/60 text-sm">
              <div class="font-semibold mb-2">ONTOLOGIC XYZ • MAMBA MENTALITY SYSTEM</div>
              <div class="text-white/40">
                9.0 MAE • OntoRisk • Real-time ML Predictions • Autonomous 24/7
              </div>
              <div class="text-white/30 mt-3 text-xs">
                Powered by ML + OntoRisk + SolidJS + Three.js • Production Ready
              </div>
            </div>
          </div>
        </footer>
      </div>

      {/* Bet Tracking Modal */}
      <BetTrackingModal 
        isOpen={showBetModal()}
        onClose={() => setShowBetModal(false)}
        opportunity={selectedOpp()}
        apiUrl={API_URL}
      />
    </div>
  );
}

export default App;

