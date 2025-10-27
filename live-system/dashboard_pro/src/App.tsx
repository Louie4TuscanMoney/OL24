import { createSignal, onMount, onCleanup, For, Show } from 'solid-js';
import axios from 'axios';
import './App.css';
import BasketballCourt3D from './components/BasketballCourt3D';
import BetTrackingModal from './components/BetTrackingModal';
import MLModelVisualization3D from './components/MLModelVisualization3D';
import OpportunityCard from './components/OpportunityCard';
import GameDetailModal from './components/GameDetailModal';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8001';
const CORRECT_PASSWORD = 'rwwc2018';

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
  // Authentication
  const [isAuthenticated, setIsAuthenticated] = createSignal(false);
  const [passwordInput, setPasswordInput] = createSignal('');
  const [showError, setShowError] = createSignal(false);
  const [showSignup, setShowSignup] = createSignal(false);
  const [signupPhone, setSignupPhone] = createSignal('');
  const [signupPassword, setSignupPassword] = createSignal('');
  const [signupSuccess, setSignupSuccess] = createSignal(false);
  const [signupError, setSignupError] = createSignal('');

  // Declare interval variable
  let interval: number;

  const handleLogin = async (e: Event) => {
    e.preventDefault();
    const password = passwordInput().trim();
    
    // Check admin password first
    if (password === CORRECT_PASSWORD) {
      setPasswordInput('');
      setShowError(false);
      setIsAuthenticated(true);
      localStorage.setItem('ontologic_auth', 'true');
      
      // Start fetching data after successful login
      setTimeout(() => {
        fetchData();
        interval = setInterval(fetchData, 3000) as unknown as number;  // 3 seconds - MAXIMUM SPEED!
      }, 100);
      return;
    }
    
    // Check if password is approved
    try {
      const res = await axios.post(`${API_URL}/api/auth/check-password`, { password });
      if (res.data.approved) {
        setPasswordInput('');
        setShowError(false);
        setIsAuthenticated(true);
        localStorage.setItem('ontologic_auth', 'true');
        setTimeout(() => {
          fetchData();
          interval = setInterval(fetchData, 10000) as unknown as number;
        }, 100);
      } else {
        setShowError(true);
        setTimeout(() => setShowError(false), 3000);
      }
    } catch (error) {
      setShowError(true);
      setTimeout(() => setShowError(false), 3000);
    }
  };

  const handleSignup = async (e: Event) => {
    e.preventDefault();
    setSignupError('');
    
    if (!signupPhone() || !signupPassword()) {
      setSignupError('Please fill in all fields');
      return;
    }
    
    if (signupPassword().length < 8) {
      setSignupError('Password must be at least 8 characters');
      return;
    }
    
    try {
      await axios.post(`${API_URL}/api/auth/signup`, {
        phone: signupPhone(),
        password: signupPassword()
      });
      setSignupSuccess(true);
      setSignupPhone('');
      setSignupPassword('');
    } catch (error: any) {
      setSignupError(error.response?.data?.detail || 'Error submitting request');
    }
  };

  const handleLogout = () => {
    setIsAuthenticated(false);
    localStorage.removeItem('ontologic_auth');
    setPasswordInput('');
    // Stop fetching data on logout
    clearInterval(interval);
  };

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
  const [showMLModel, setShowMLModel] = createSignal(false);
  const [modelState, setModelState] = createSignal<any>(null);
  const [showLiveTest, setShowLiveTest] = createSignal(false);
  const [liveTestData, setLiveTestData] = createSignal<any>(null);
  const [showGameDetail, setShowGameDetail] = createSignal(false);
  const [selectedGame, setSelectedGame] = createSignal<Game | null>(null);

  const fetchData = async () => {
    try {
      // Add timestamp to prevent caching
      const timestamp = Date.now();
      const config = {
        headers: {
          'Cache-Control': 'no-cache, no-store, must-revalidate',
          'Pragma': 'no-cache',
          'Expires': '0'
        }
      };
      
      const [gamesRes, oppsRes, riskRes, portfolioRes, modelRes] = await Promise.all([
        axios.get(`${API_URL}/api/live-games?t=${timestamp}`, config).catch(() => ({ data: { games: [] } })),
        axios.get(`${API_URL}/api/opportunities?t=${timestamp}`, config).catch(() => ({ data: { opportunities: [] } })),
        axios.get(`${API_URL}/api/risk-status?t=${timestamp}`, config).catch(() => ({ data: { risk_status: null } })),
        axios.get(`${API_URL}/api/bets/summary`).catch(() => ({ data: null })),
        axios.get(`${API_URL}/api/model/state`).catch(() => ({ data: null }))
      ]);

      setLiveGames(gamesRes.data.games || []);
      setOpportunities(oppsRes.data.all_opportunities || oppsRes.data.opportunities || []);
      setRiskStatus(riskRes.data.risk_status);
      setPortfolioSummary(portfolioRes.data);
      setModelState(modelRes.data);
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

  onMount(() => {
    // Check for existing auth
    const authStatus = localStorage.getItem('ontologic_auth');
    if (authStatus === 'true') {
      setIsAuthenticated(true);
      // Start fetching data if authenticated
      fetchData();
      interval = setInterval(fetchData, 10000) as unknown as number;
    }
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

  const testLiveData = async () => {
    try {
      const timestamp = Date.now();
      const config = {
        headers: {
          'Cache-Control': 'no-cache, no-store, must-revalidate',
          'Pragma': 'no-cache',
          'Expires': '0'
        }
      };
      
      const [nbaRes, oddsRes] = await Promise.all([
        axios.get(`${API_URL}/api/live-games?t=${timestamp}`, config),
        axios.get(`${API_URL}/api/opportunities?t=${timestamp}`, config)
      ]);
      
      setLiveTestData({
        nba: nbaRes.data,
        odds: oddsRes.data,
        timestamp: new Date().toLocaleTimeString()
      });
      setShowLiveTest(true);
    } catch (error) {
      console.error('Error fetching live test data:', error);
      setLiveTestData({
        error: 'Failed to fetch live data',
        timestamp: new Date().toLocaleTimeString()
      });
      setShowLiveTest(true);
    }
  };

  const openGameDetail = (game: Game) => {
    setSelectedGame(game);
    setShowGameDetail(true);
  };

  const profitColor = () => {
    const status = riskStatus();
    if (!status) return '#10b981';
    const profit = -status.daily_loss;
    return profit >= 0 ? '#10b981' : '#ef4444';
  };

  return (
    <>
      {/* Login Screen */}
      <Show when={!isAuthenticated()}>
        <div class="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 flex items-center justify-center">
        {/* Animated Background */}
        <div class="fixed inset-0 overflow-hidden pointer-events-none">
          <div class="absolute top-0 left-1/4 w-96 h-96 bg-purple-500 rounded-full mix-blend-multiply filter blur-3xl opacity-20 animate-blob"></div>
          <div class="absolute top-0 right-1/4 w-96 h-96 bg-yellow-500 rounded-full mix-blend-multiply filter blur-3xl opacity-20 animate-blob animation-delay-2000"></div>
          <div class="absolute bottom-0 left-1/3 w-96 h-96 bg-pink-500 rounded-full mix-blend-multiply filter blur-3xl opacity-20 animate-blob animation-delay-4000"></div>
        </div>

        {/* Login Form */}
        <div class="relative z-10 w-full max-w-md px-4">
          <div class="bg-white/10 backdrop-blur-xl rounded-3xl p-8 border border-white/20 shadow-2xl">
            {/* Logo/Title */}
            <div class="text-center mb-8">
              <div class="text-4xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-pink-400 mb-2">
                ONTOLOGIC XYZ
              </div>
              <div class="text-white/60 text-sm">
                NBA Trading Dashboard
              </div>
            </div>

            {/* Login Form */}
            <form onSubmit={handleLogin} class="space-y-6">
              <div>
                <label for="login-password" class="block text-white/80 text-sm font-medium mb-2">
                  Password
                </label>
                <input
                  id="login-password"
                  name="password"
                  type="password"
                  value={passwordInput()}
                  onInput={(e) => setPasswordInput(e.currentTarget.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') {
                      e.preventDefault();
                      handleLogin(e);
                    }
                  }}
                  placeholder="Enter password"
                  class="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-xl text-white placeholder-white/30 focus:outline-none focus:border-purple-400 transition-colors"
                  autocomplete="current-password"
                />
              </div>

              {/* Error Message */}
              <Show when={showError()}>
                <div class="bg-red-500/10 border border-red-500/20 rounded-xl p-3 text-red-400 text-sm text-center animate-pulse">
                  ❌ Incorrect password
                </div>
              </Show>

              {/* Submit Button */}
              <button
                type="submit"
                class="w-full bg-gradient-to-r from-purple-500 to-pink-500 hover:from-purple-600 hover:to-pink-600 text-white font-semibold py-3 rounded-xl transition-all transform hover:scale-105 shadow-lg"
              >
                🔓 Unlock Dashboard
              </button>
            </form>

            {/* Sign Up Button */}
            <div class="mt-6">
              <button
                onClick={() => setShowSignup(true)}
                class="w-full bg-white/5 hover:bg-white/10 border border-white/20 hover:border-white/30 text-white/80 font-medium py-3 rounded-xl transition-all"
              >
                📝 Request Access
              </button>
            </div>

            {/* Footer */}
            <div class="mt-6 text-center text-white/40 text-xs">
              Protected Access • Mamba Mentality System
            </div>
          </div>
        </div>

        {/* Signup Modal */}
        <Show when={showSignup()}>
          <div class="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm px-4">
            <div class="bg-white/10 backdrop-blur-xl rounded-3xl p-8 border border-white/20 shadow-2xl max-w-md w-full">
              <Show when={!signupSuccess()}>
                <div>
                  <h2 class="text-3xl font-bold text-white mb-2">Request Access</h2>
                  <p class="text-white/60 text-sm mb-6">
                    Submit your request and we'll review it within 24 hours
                  </p>

                  <form onSubmit={handleSignup} class="space-y-4">
                    <div>
                      <label for="signup-phone" class="block text-white/80 text-sm font-medium mb-2">
                        Phone Number
                      </label>
                      <input
                        id="signup-phone"
                        name="phone"
                        type="tel"
                        value={signupPhone()}
                        onInput={(e) => setSignupPhone(e.currentTarget.value)}
                        placeholder="+1 (555) 123-4567"
                        class="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-xl text-white placeholder-white/30 focus:outline-none focus:border-purple-400 transition-colors"
                        required
                      />
                    </div>

                    <div>
                      <label for="signup-password" class="block text-white/80 text-sm font-medium mb-2">
                        Create Password
                      </label>
                      <input
                        id="signup-password"
                        name="new-password"
                        type="password"
                        value={signupPassword()}
                        onInput={(e) => setSignupPassword(e.currentTarget.value)}
                        placeholder="Min 8 characters"
                        class="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-xl text-white placeholder-white/30 focus:outline-none focus:border-purple-400 transition-colors"
                        required
                      />
                      <p class="text-white/40 text-xs mt-1">
                        This will be your unique password after approval
                      </p>
                    </div>

                    <Show when={signupError()}>
                      <div class="bg-red-500/10 border border-red-500/20 rounded-xl p-3 text-red-400 text-sm">
                        {signupError()}
                      </div>
                    </Show>

                    <div class="flex gap-3">
                      <button
                        type="button"
                        onClick={() => {
                          setShowSignup(false);
                          setSignupError('');
                          setSignupPhone('');
                          setSignupPassword('');
                        }}
                        class="flex-1 bg-white/5 hover:bg-white/10 text-white/80 font-medium py-3 rounded-xl transition-all"
                      >
                        Cancel
                      </button>
                      <button
                        type="submit"
                        class="flex-1 bg-gradient-to-r from-purple-500 to-pink-500 hover:from-purple-600 hover:to-pink-600 text-white font-semibold py-3 rounded-xl transition-all"
                      >
                        Submit Request
                      </button>
                    </div>
                  </form>
                </div>
              </Show>

              <Show when={signupSuccess()}>
                <div class="text-center">
                  <div class="text-6xl mb-4">✅</div>
                  <h2 class="text-3xl font-bold text-white mb-2">Request Submitted!</h2>
                  <p class="text-white/70 mb-6">
                    We'll review your request within 24 hours. You'll receive a notification at your phone number once approved.
                  </p>
                  <button
                    onClick={() => {
                      setShowSignup(false);
                      setSignupSuccess(false);
                    }}
                    class="w-full bg-gradient-to-r from-purple-500 to-pink-500 hover:from-purple-600 hover:to-pink-600 text-white font-semibold py-3 rounded-xl transition-all"
                  >
                    Close
                  </button>
                </div>
              </Show>
            </div>
          </div>
        </Show>
      </div>
      </Show>

      {/* Main Dashboard */}
      <Show when={isAuthenticated()}>
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
                  <button 
                    onClick={() => setShowMLModel(!showMLModel())}
                    class="px-4 py-1 bg-gradient-to-r from-blue-500 to-purple-500 text-white rounded-full text-sm font-semibold hover:shadow-lg transition-all"
                  >
                    🧠 Watch Model Think
                  </button>
                  <button 
                    onClick={testLiveData}
                    class="px-4 py-1 bg-gradient-to-r from-green-500 to-emerald-500 text-white rounded-full text-sm font-semibold hover:shadow-lg transition-all"
                  >
                    📡 Test Live Data
                  </button>
                  <button 
                    onClick={handleLogout}
                    class="px-4 py-1 bg-red-500/20 hover:bg-red-500/30 text-red-300 rounded-full text-sm font-semibold transition-all"
                  >
                    🔒 Logout
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

        {/* ML Model Brain Visualization */}
        <Show when={showMLModel()}>
          <section class="mb-8">
            <div class="bg-white/10 backdrop-blur-xl rounded-3xl p-8 border border-white/20 shadow-2xl">
              <div class="flex justify-between items-center mb-6">
                <h2 class="text-3xl font-black text-white flex items-center gap-3">
                  <span class="text-4xl">🧠</span>
                  <span>ML Model Brain - Live Thinking</span>
                </h2>
                <button 
                  onClick={() => setShowMLModel(false)}
                  class="text-white/60 hover:text-white text-2xl"
                >
                  ✕
                </button>
              </div>
              <div class="h-[700px]">
                <MLModelVisualization3D prediction={modelState()} />
              </div>
            </div>
          </section>
        </Show>

        {/* Live Data Test Modal */}
        <Show when={showLiveTest()}>
          <section class="mb-8">
            <div class="bg-white/10 backdrop-blur-xl rounded-3xl p-8 border border-white/20 shadow-2xl">
              <div class="flex justify-between items-center mb-6">
                <h2 class="text-3xl font-black text-white flex items-center gap-3">
                  <span class="text-4xl">📡</span>
                  <span>Live Data Test - NBA API + BetOnline</span>
                </h2>
                <button 
                  onClick={() => setShowLiveTest(false)}
                  class="text-white/60 hover:text-white text-2xl"
                >
                  ✕
                </button>
              </div>
              
              <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* NBA API Data */}
                <div class="bg-black/40 rounded-xl p-6 border border-green-500/30">
                  <h3 class="text-xl font-bold text-green-400 mb-4 flex items-center gap-2">
                    <span>🏀</span>
                    <span>NBA API (Live Scores)</span>
                  </h3>
                  <div class="text-xs text-white/90 font-mono bg-black/50 rounded-lg p-4 overflow-auto max-h-96">
                    <pre class="whitespace-pre-wrap">{JSON.stringify(liveTestData()?.nba || {}, null, 2)}</pre>
                  </div>
                  <div class="mt-4 text-sm text-white/60">
                    ✅ Pulling live scores every 30 seconds
                  </div>
                </div>

                {/* BetOnline Data */}
                <div class="bg-black/40 rounded-xl p-6 border border-yellow-500/30">
                  <h3 class="text-xl font-bold text-yellow-400 mb-4 flex items-center gap-2">
                    <span>💰</span>
                    <span>BetOnline (Live Odds)</span>
                  </h3>
                  <div class="text-xs text-white/90 font-mono bg-black/50 rounded-lg p-4 overflow-auto max-h-96">
                    <pre class="whitespace-pre-wrap">{JSON.stringify(liveTestData()?.odds || {}, null, 2)}</pre>
                  </div>
                  <div class="mt-4 text-sm text-white/60">
                    ✅ Scraping live spreads when Q2 6:00 hits
                  </div>
                </div>
              </div>

              {/* Stats */}
              <div class="mt-6 bg-gradient-to-r from-green-500/20 to-yellow-500/20 rounded-xl p-6 border border-white/10">
                <div class="grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
                  <div>
                    <div class="text-2xl font-black text-white">{liveTestData()?.nba?.count || 0}</div>
                    <div class="text-sm text-white/60">Live Games</div>
                  </div>
                  <div>
                    <div class="text-2xl font-black text-white">{liveTestData()?.odds?.total_count || 0}</div>
                    <div class="text-sm text-white/60">Total Opportunities</div>
                  </div>
                  <div>
                    <div class="text-2xl font-black text-green-400">{liveTestData()?.odds?.betting_count || 0}</div>
                    <div class="text-sm text-white/60">Approved Bets</div>
                  </div>
                  <div>
                    <div class="text-2xl font-black text-gray-400">{liveTestData()?.odds?.context_count || 0}</div>
                    <div class="text-sm text-white/60">Context Only</div>
                  </div>
                </div>
              </div>

              {/* Refresh Button */}
              <div class="mt-6 text-center">
                <button 
                  onClick={testLiveData}
                  class="px-6 py-3 bg-gradient-to-r from-green-500 to-emerald-500 hover:from-green-600 hover:to-emerald-600 text-white font-bold rounded-xl transition-all"
                >
                  🔄 Refresh Live Data
                </button>
                <div class="text-sm text-white/50 mt-2">
                  Last Updated: {liveTestData()?.timestamp || 'Never'}
                </div>
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

        {/* All Opportunities (Betting + Context) */}
        <section class="mb-8">
          <div class="bg-white/10 backdrop-blur-xl rounded-3xl p-8 border border-white/20 shadow-2xl">
            <div class="flex justify-between items-center mb-6">
              <h2 class="text-3xl font-black text-white flex items-center gap-3">
                <span class="text-4xl">🎯</span>
                <span>Q2 6:00 Opportunities</span>
              </h2>
              <div class="flex gap-3">
                <div class="bg-gradient-to-r from-purple-500 to-pink-500 text-white font-black px-4 py-2 rounded-full text-sm">
                  {opportunities().filter((o: any) => o.should_bet).length} TO BET
                </div>
                <div class="bg-gray-500/50 text-white font-semibold px-4 py-2 rounded-full text-sm">
                  {opportunities().filter((o: any) => !o.should_bet).length} CONTEXT
                </div>
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

            {/* Betting Opportunities */}
            <Show when={opportunities().filter((o: any) => o.should_bet).length > 0}>
              <div class="mb-8">
                <h3 class="text-xl font-bold text-white mb-4 flex items-center gap-2">
                  <span>✅</span>
                  <span>Approved for Betting ({opportunities().filter((o: any) => o.should_bet).length})</span>
                </h3>
                <div class="grid gap-6">
                  <For each={opportunities().filter((o: any) => o.should_bet)}>
                    {(opp) => (
                      <OpportunityCard opportunity={opp} onPlaceBet={() => openBetModal(opp)} />
                    )}
                  </For>
                </div>
              </div>
            </Show>

            {/* Context Opportunities */}
            <Show when={opportunities().filter((o: any) => !o.should_bet).length > 0}>
              <div>
                <h3 class="text-xl font-bold text-white/70 mb-4 flex items-center gap-2">
                  <span>ℹ️</span>
                  <span>For Context Only ({opportunities().filter((o: any) => !o.should_bet).length})</span>
                </h3>
                <div class="grid gap-6">
                  <For each={opportunities().filter((o: any) => !o.should_bet)}>
                    {(opp) => (
                      <OpportunityCard opportunity={opp} onPlaceBet={() => openBetModal(opp)} />
                    )}
                  </For>
                </div>
              </div>
            </Show>

            <Show when={opportunities().length > 0}>
              <div class="mt-8 bg-black/30 rounded-xl p-6 border border-white/10">
                <div class="font-bold text-white mb-4">📊 Full Breakdown:</div>
                <div class="grid grid-cols-3 gap-4 text-xs">
                  <div>
                    <div class="text-white/50 mb-1">Lead Held Games:</div>
                    <div class="text-white font-semibold">53.2% • 9.26 MAE • 72% accuracy</div>
                  </div>
                  <div>
                    <div class="text-white/50 mb-1">Close Games:</div>
                    <div class="text-white font-semibold">26% • 9.85 MAE • 65% accuracy</div>
                  </div>
                  <div>
                    <div class="text-white/50 mb-1">High Confidence Zone:</div>
                    <div class="text-white font-semibold">31.7% • 2.57 MAE • 82.5% accuracy</div>
                  </div>
                </div>
              </div>
            </Show>
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
                  <div 
                    onClick={() => openGameDetail(game)}
                    class={`
                    bg-gradient-to-r rounded-2xl p-6 border-2 transition-all duration-300 cursor-pointer
                    ${game.status_text === 'LIVE' 
                      ? 'from-red-500/20 to-orange-500/20 border-red-500/50 shadow-lg shadow-red-500/20' 
                      : 'from-slate-500/20 to-slate-600/20 border-slate-500/30'}
                    ${game.is_q2_6min ? 'ring-4 ring-yellow-500/50 shadow-xl shadow-yellow-500/30' : ''}
                    hover:scale-[1.02] hover:shadow-2xl
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
                        <div class="text-xs text-white/40 mt-2 hover:text-white/70 transition-all">
                          👁️ Click for details
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
              <div class="mt-4 pt-4 border-t border-white/10">
                <a 
                  href="file:///Users/test/Desktop/Tuscan%20Money/Ontologic%20XYZ/ML%20Research/5.%20Live%20System/OntologicXYZ.com"
                  target="_blank"
                  class="inline-flex items-center gap-2 text-purple-400 hover:text-purple-300 transition-colors text-xs font-medium"
                >
                  <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                  </svg>
                  Supporting Files & Documentation
                </a>
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

      {/* Game Detail Modal */}
      <Show when={showGameDetail() && selectedGame()}>
        <GameDetailModal 
          game={selectedGame()!}
          onClose={() => setShowGameDetail(false)}
        />
      </Show>
        </div>
      </Show>
    </>
  );
}

export default App;

