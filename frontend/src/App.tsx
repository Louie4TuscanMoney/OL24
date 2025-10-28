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

import { type Component, createSignal, createEffect, lazy, Suspense } from 'solid-js';

// Lazy load pages for faster initial load
const Dashboard = lazy(() => import('./components/Dashboard'));
const StatsPage = lazy(() => import('./components/StatsPage'));
const SchedulePage = lazy(() => import('./components/SchedulePage'));
const TeamPage = lazy(() => import('./components/TeamPage'));
const TeamsDirectory = lazy(() => import('./components/TeamsDirectory'));

const LoadingSpinner = () => (
  <div class="flex items-center justify-center min-h-screen">
    <div class="text-white text-2xl">Loading...</div>
  </div>
);

const App: Component = () => {
  const [currentPage, setCurrentPage] = createSignal<'predictions' | 'stats' | 'schedule' | 'teams' | 'team'>('predictions');
  const [selectedTeam, setSelectedTeam] = createSignal('');

  // Handle URL-based routing
  createEffect(() => {
    const path = window.location.pathname;
    if (path.startsWith('/team/')) {
      const teamAbbr = path.split('/')[2];
      setSelectedTeam(teamAbbr);
      setCurrentPage('team');
    } else if (path === '/teams') {
      setCurrentPage('teams');
    } else if (path === '/stats') {
      setCurrentPage('stats');
    } else if (path === '/schedule') {
      setCurrentPage('schedule');
    } else {
      setCurrentPage('predictions');
    }
  });

  const navigate = (page: 'predictions' | 'stats' | 'schedule' | 'teams' | 'team', team?: string) => {
    if (page === 'team' && team) {
      window.history.pushState({}, '', `/team/${team}`);
      setSelectedTeam(team);
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
    } else {
      window.history.pushState({}, '', '/');
      setCurrentPage('predictions');
    }
  };

  return (
    <div>
      {/* Navigation Bar */}
      <nav class="bg-gray-900 border-b border-gray-800 sticky top-0 z-50">
        <div class="max-w-7xl mx-auto px-4">
          <div class="flex items-center justify-between h-16">
            <div class="flex items-center gap-2">
              <span class="text-2xl">🎯</span>
              <h1 class="text-white text-xl font-bold">Ontologic XYZ</h1>
            </div>
            <div class="flex gap-2">
              <button
                onClick={() => navigate('predictions')}
                class={`px-4 py-2 rounded-lg font-semibold transition-all ${
                  currentPage() === 'predictions'
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
                }`}
              >
                🔮 Live Predictions
              </button>
              <button
                onClick={() => navigate('stats')}
                class={`px-4 py-2 rounded-lg font-semibold transition-all ${
                  currentPage() === 'stats'
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
                }`}
              >
                📊 Stats & Injuries
              </button>
              <button
                onClick={() => navigate('schedule')}
                class={`px-4 py-2 rounded-lg font-semibold transition-all ${
                  currentPage() === 'schedule'
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
                }`}
              >
                📅 Schedule
              </button>
              <button
                onClick={() => navigate('teams')}
                class={`px-4 py-2 rounded-lg font-semibold transition-all ${
                  currentPage() === 'teams' || currentPage() === 'team'
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
                }`}
              >
                🏀 Teams
              </button>
            </div>
          </div>
        </div>
      </nav>

      {/* Page Content */}
      <Suspense fallback={<LoadingSpinner />}>
        {currentPage() === 'predictions' && <Dashboard />}
        {currentPage() === 'stats' && <StatsPage onTeamClick={(abbr) => navigate('team', abbr)} />}
        {currentPage() === 'schedule' && <SchedulePage />}
        {currentPage() === 'teams' && <TeamsDirectory onTeamClick={(abbr) => navigate('team', abbr)} />}
        {currentPage() === 'team' && selectedTeam() && <TeamPage teamAbbr={selectedTeam()} />}
      </Suspense>
    </div>
  );
};

export default App;
