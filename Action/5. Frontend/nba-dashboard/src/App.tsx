/**
 * Main App Component
 * NBA Real-Time Prediction Dashboard
 * 
 * Features:
 * - Live score updates (10-second intervals)
 * - ML predictions (Ensemble: 5.39 MAE)
 * - Edge detection (ML vs BetOnline)
 * - Risk management recommendations (5-layer system)
 */

import { type Component } from 'solid-js';
import Dashboard from './components/Dashboard';

const App: Component = () => {
  return <Dashboard />;
};

export default App;
