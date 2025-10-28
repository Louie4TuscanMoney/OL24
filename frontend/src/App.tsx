/**
 * Main App Component
 * NBA Real-Time Prediction Dashboard + Analytics Platform
 * 
 * Features:
 * - Live score updates & ML predictions
 * - Complete NBA analytics (stats, injuries, depth charts)
 * - Schedule viewer
 * - Team pages with lineups
 */

import { type Component, createSignal, createEffect, lazy, Suspense, Show } from 'solid-js';
import LoadingScreen from './components/LoadingScreen';
import { wsService } from './services/websocket';

// Lazy load pages for faster initial load
const Dashboard = lazy(() => import('./components/Dashboard'));
const StatsPage = lazy(() => import('./components/StatsPage'));
const SchedulePage = lazy(() => import('./components/SchedulePage'));
const TeamPage = lazy(() => import('./components/TeamPage'));
const TeamsDirectory = lazy(() => import('./components/TeamsDirectory'));
const GameDetailPage = lazy(() => import('./components/GameDetailPage'));
const TradingDesk = lazy(() => import('./components/TradingDesk'));

const App: Component = () => {
  const [currentPage, setCurrentPage] = createSignal<'predictions' | 'stats' | 'schedule' | 'teams' | 'team' | 'game' | 'trading'>('predictions');
  const [selectedTeam, setSelectedTeam] = createSignal('');
  const [selectedGame, setSelectedGame] = createSignal('');
  const [initialLoading, setInitialLoading] = createSignal(true);
  const [connected] = wsService.connected;

  // Hide loading screen after connection OR after 3 seconds max
  createEffect(() => {
    if (connected() || currentPage() !== 'predictions') {
      setTimeout(() => setInitialLoading(false), 500); // Small delay for smooth transition
    }
    
    // FAILSAFE: Always hide loading after 3 seconds, even if not connected
    const timeout = setTimeout(() => {
      setInitialLoading(false);
    }, 3000);
    
    return () => clearTimeout(timeout);
  });

  // Handle URL-based routing
  createEffect(() => {
    const path = window.location.pathname;
    if (path.startsWith('/game/')) {
      const gameId = path.split('/')[2];
      setSelectedGame(gameId);
      setCurrentPage('game');
    } else if (path.startsWith('/team/')) {
      const teamAbbr = path.split('/')[2];
      setSelectedTeam(teamAbbr);
      setCurrentPage('team');
    } else if (path === '/teams') {
      setCurrentPage('teams');
    } else if (path === '/stats') {
      setCurrentPage('stats');
    } else if (path === '/schedule') {
      setCurrentPage('schedule');
    } else if (path === '/trading') {
      setCurrentPage('trading');
    } else {
      setCurrentPage('predictions');
    }
  });

  const navigate = (page: 'predictions' | 'stats' | 'schedule' | 'teams' | 'team' | 'game' | 'trading', id?: string) => {
    if (page === 'game' && id) {
      window.history.pushState({}, '', `/game/${id}`);
      setSelectedGame(id);
      setCurrentPage('game');
    } else if (page === 'team' && id) {
      window.history.pushState({}, '', `/team/${id}`);
      setSelectedTeam(id);
      setCurrentPage('team');
    } else if (page === 'teams') {
      window.history.pushState({}, '', '/teams');
      setCurrentPage('teams');
    } else if (page === 'stats') {
      window.history.pushState({}, '', '/stats');
      setCurrentPage('stats');
    } else if (page === 'schedule') {
      window.history.pushState({}, '', '/schedule');
      setCurrentPage('schedule');
    } else if (page === 'trading') {
      window.history.pushState({}, '', '/trading');
      setCurrentPage('trading');
    } else {
      window.history.pushState({}, '', '/');
      setCurrentPage('predictions');
    }
  };

  return (
    <div>
      {/* Loading Screen (shows on initial load) */}
      <Show when={initialLoading()}>
        <LoadingScreen message="Connecting to NBA Live API..." />
      </Show>

      {/* Main App (shows after loading or immediately for non-predictions pages) */}
      <Show when={!initialLoading()}>
        {/* Modern Navigation Bar */}
        <nav class="bg-black/50 border-b border-gray-800/50 sticky top-0 z-50 backdrop-blur-xl">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div class="flex items-center justify-between h-16">
            {/* Logo */}
            <div class="flex items-center gap-3">
              <div class="text-2xl">🎯</div>
              <h1 class="text-white text-xl font-bold tracking-tight hidden sm:block">Ontologic XYZ</h1>
            </div>
            
            {/* Navigation Tabs */}
            <div class="flex gap-2">
              <button
                onClick={() => navigate('predictions')}
                class={`px-4 py-2 rounded-xl text-sm font-semibold transition-all duration-200 ${
                  currentPage() === 'predictions'
                    ? 'bg-gradient-to-r from-blue-600 to-purple-600 text-white shadow-lg shadow-blue-500/30'
                    : 'bg-gray-800/50 text-gray-400 hover:bg-gray-700/50 hover:text-white'
                }`}
              >
                <span class="hidden sm:inline">Live</span>
                <span class="sm:hidden">🔮</span>
              </button>
              <button
                onClick={() => navigate('stats')}
                class={`px-4 py-2 rounded-xl text-sm font-semibold transition-all duration-200 ${
                  currentPage() === 'stats'
                    ? 'bg-gradient-to-r from-blue-600 to-purple-600 text-white shadow-lg shadow-blue-500/30'
                    : 'bg-gray-800/50 text-gray-400 hover:bg-gray-700/50 hover:text-white'
                }`}
              >
                <span class="hidden sm:inline">Stats</span>
                <span class="sm:hidden">📊</span>
              </button>
              <button
                onClick={() => navigate('schedule')}
                class={`px-4 py-2 rounded-xl text-sm font-semibold transition-all duration-200 ${
                  currentPage() === 'schedule'
                    ? 'bg-gradient-to-r from-blue-600 to-purple-600 text-white shadow-lg shadow-blue-500/30'
                    : 'bg-gray-800/50 text-gray-400 hover:bg-gray-700/50 hover:text-white'
                }`}
              >
                <span class="hidden sm:inline">Schedule</span>
                <span class="sm:hidden">📅</span>
              </button>
              <button
                onClick={() => navigate('teams')}
                class={`px-4 py-2 rounded-xl text-sm font-semibold transition-all duration-200 ${
                  currentPage() === 'teams' || currentPage() === 'team'
                    ? 'bg-gradient-to-r from-blue-600 to-purple-600 text-white shadow-lg shadow-blue-500/30'
                    : 'bg-gray-800/50 text-gray-400 hover:bg-gray-700/50 hover:text-white'
                }`}
              >
                <span class="hidden sm:inline">Teams</span>
                <span class="sm:hidden">🏀</span>
              </button>
              <button
                onClick={() => navigate('trading')}
                class={`px-4 py-2 rounded-xl text-sm font-semibold transition-all duration-200 ${
                  currentPage() === 'trading'
                    ? 'bg-gradient-to-r from-yellow-600 to-orange-600 text-white shadow-lg shadow-yellow-500/30'
                    : 'bg-gray-800/50 text-gray-400 hover:bg-gray-700/50 hover:text-white'
                }`}
              >
                <span class="hidden sm:inline">Trading</span>
                <span class="sm:hidden">💰</span>
              </button>
            </div>
          </div>
        </div>
      </nav>

        {/* Page Content */}
        <Suspense fallback={<div class="flex justify-center items-center min-h-screen"><div class="text-white text-xl">Loading page...</div></div>}>
          {currentPage() === 'predictions' && <Dashboard onGameClick={(id) => navigate('game', id)} onTeamClick={(abbr) => navigate('team', abbr)} />}
          {currentPage() === 'stats' && <StatsPage onTeamClick={(abbr) => navigate('team', abbr)} />}
          {currentPage() === 'schedule' && <SchedulePage />}
          {currentPage() === 'teams' && <TeamsDirectory onTeamClick={(abbr) => navigate('team', abbr)} />}
          {currentPage() === 'team' && selectedTeam() && <TeamPage teamAbbr={selectedTeam()} />}
          {currentPage() === 'game' && selectedGame() && <GameDetailPage gameId={selectedGame()} />}
          {currentPage() === 'trading' && <TradingDesk games={wsService.games()} backendUrl={import.meta.env.VITE_BACKEND_URL || 'https://ol24-production.up.railway.app'} />}
        </Suspense>
      </Show>
    </div>
  );
};

export default App;
