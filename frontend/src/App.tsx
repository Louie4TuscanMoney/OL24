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

import { type Component, createSignal } from 'solid-js';
import Dashboard from './components/Dashboard';
import StatsPage from './components/StatsPage';
import SchedulePage from './components/SchedulePage';
import TeamPage from './components/TeamPage';

const App: Component = () => {
  const [currentPage, setCurrentPage] = createSignal<'predictions' | 'stats' | 'schedule' | 'team'>('predictions');
  const [selectedTeam, setSelectedTeam] = createSignal('LAL');

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
                onClick={() => setCurrentPage('predictions')}
                class={`px-4 py-2 rounded-lg font-semibold transition-all ${
                  currentPage() === 'predictions'
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
                }`}
              >
                🔮 Live Predictions
              </button>
              <button
                onClick={() => setCurrentPage('stats')}
                class={`px-4 py-2 rounded-lg font-semibold transition-all ${
                  currentPage() === 'stats'
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
                }`}
              >
                📊 Stats & Injuries
              </button>
              <button
                onClick={() => setCurrentPage('schedule')}
                class={`px-4 py-2 rounded-lg font-semibold transition-all ${
                  currentPage() === 'schedule'
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
                }`}
              >
                📅 Schedule
              </button>
              <button
                onClick={() => {
                  setCurrentPage('team');
                  setSelectedTeam('LAL');
                }}
                class={`px-4 py-2 rounded-lg font-semibold transition-all ${
                  currentPage() === 'team'
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
      {currentPage() === 'predictions' && <Dashboard />}
      {currentPage() === 'stats' && <StatsPage />}
      {currentPage() === 'schedule' && <SchedulePage />}
      {currentPage() === 'team' && <TeamPage teamAbbr={selectedTeam()} />}
    </div>
  );
};

export default App;
