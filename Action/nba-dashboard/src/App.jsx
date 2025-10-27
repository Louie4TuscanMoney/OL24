import { createSignal, onMount, onCleanup, For, Show } from 'solid-js';
import axios from 'axios';

function App() {
  const [games, setGames] = createSignal([]);
  const [predictions, setPredictions] = createSignal({});
  const [odds, setOdds] = createSignal({});
  const [loading, setLoading] = createSignal(true);
  const [lastUpdate, setLastUpdate] = createSignal(new Date());
  const [activeSection, setActiveSection] = createSignal('overview');

  // Fetch live data
  const fetchData = async () => {
    try {
      // Fetch NBA games
      const gamesRes = await axios.get('/api/nba/games');
      setGames(gamesRes.data.games || []);

      // Fetch BetOnline odds
      const oddsRes = await axios.get('/api/betonline/odds');
      setOdds(oddsRes.data.odds || {});

      // Fetch predictions
      const predsRes = await axios.get('/api/predictions');
      setPredictions(predsRes.data.predictions || {});

      setLastUpdate(new Date());
      setLoading(false);
    } catch (error) {
      console.error('Fetch error:', error);
      setLoading(false);
    }
  };

  // Auto-refresh every 5 seconds
  onMount(() => {
    fetchData();
    const interval = setInterval(fetchData, 5000);
    onCleanup(() => clearInterval(interval));
  });

  return (
    <div class="app">
      {/* Header */}
      <header class="header">
        <div class="header-content">
          <h1>🏀 NBA Prediction System</h1>
          <div class="live-indicator">
            <span class="pulse"></span>
            <span>LIVE - Opening Night</span>
          </div>
        </div>
      </header>

      {/* Navigation */}
      <nav class="nav">
        <button 
          class={activeSection() === 'overview' ? 'nav-btn active' : 'nav-btn'}
          onClick={() => setActiveSection('overview')}
        >
          📊 Overview
        </button>
        <button 
          class={activeSection() === 'nba' ? 'nav-btn active' : 'nav-btn'}
          onClick={() => setActiveSection('nba')}
        >
          🏀 NBA Model
        </button>
        <button 
          class={activeSection() === 'odds' ? 'nav-btn active' : 'nav-btn'}
          onClick={() => setActiveSection('odds')}
        >
          💰 Live Odds
        </button>
        <button 
          class={activeSection() === 'bets' ? 'nav-btn active' : 'nav-btn'}
          onClick={() => setActiveSection('bets')}
        >
          🎯 Bet Tracker
        </button>
      </nav>

      {/* Main Content */}
      <main class="main">
        <Show when={activeSection() === 'overview'}>
          <OverviewSection games={games()} predictions={predictions()} odds={odds()} />
        </Show>

        <Show when={activeSection() === 'nba'}>
          <NBAModelSection games={games()} predictions={predictions()} loading={loading()} />
        </Show>

        <Show when={activeSection() === 'odds'}>
          <OddsSection games={games()} odds={odds()} loading={loading()} />
        </Show>

        <Show when={activeSection() === 'bets'}>
          <BetTrackerSection />
        </Show>
      </main>

      {/* Footer */}
      <footer class="footer">
        Last updated: {lastUpdate().toLocaleTimeString()} • 
        Auto-refresh: 5s • 
        Status: <span class="status-ok">Operational</span>
      </footer>
    </div>
  );
}

function OverviewSection(props) {
  return (
    <div class="section">
      <h2>📊 Live Overview</h2>
      
      <div class="stats-grid">
        <div class="stat-card">
          <div class="stat-label">Active Games</div>
          <div class="stat-value">{props.games.length}</div>
        </div>
        
        <div class="stat-card">
          <div class="stat-label">Predictions Made</div>
          <div class="stat-value">{Object.keys(props.predictions).length}</div>
        </div>
        
        <div class="stat-card">
          <div class="stat-label">Live Odds</div>
          <div class="stat-value">{Object.keys(props.odds).length}</div>
        </div>
        
        <div class="stat-card">
          <div class="stat-label">Bet Opportunities</div>
          <div class="stat-value">0</div>
        </div>
      </div>

      <div class="games-list">
        <h3>Active Games</h3>
        <Show when={props.games.length === 0}>
          <div class="empty-state">
            <p>⏳ Waiting for games to start...</p>
            <p class="small">Games begin 4:00 PM PST (Opening Night!)</p>
          </div>
        </Show>
        
        <For each={props.games}>
          {(game) => <GameCard game={game} />}
        </For>
      </div>
    </div>
  );
}

function NBAModelSection(props) {
  return (
    <div class="section">
      <h2>🏀 NBA Model - Dual Branch Predictions</h2>
      
      <div class="model-info">
        <div class="info-card">
          <h3>Branch A: Halftime</h3>
          <p>Predicts score at halftime (24 min)</p>
          <div class="model-stat">MAE: 6.00 pts</div>
        </div>
        
        <div class="info-card">
          <h3>Branch B: Final</h3>
          <p>Predicts final game score</p>
          <div class="model-stat">MAE: ~10 pts</div>
        </div>
      </div>

      <Show when={props.loading}>
        <p>Loading predictions...</p>
      </Show>

      <div class="predictions-list">
        <For each={props.games}>
          {(game) => (
            <PredictionCard 
              game={game} 
              prediction={props.predictions[game.gameId]} 
            />
          )}
        </For>
      </div>
    </div>
  );
}

function OddsSection(props) {
  return (
    <div class="section">
      <h2>💰 BetOnline Live Odds</h2>
      
      <div class="scraper-status">
        <span class="status-indicator">🟢</span>
        <span>Scraper Active - No Blocking Detected</span>
      </div>

      <Show when={props.loading}>
        <p>Fetching odds...</p>
      </Show>

      <div class="odds-grid">
        <For each={props.games}>
          {(game) => (
            <OddsCard 
              game={game} 
              odds={props.odds[game.gameId]} 
            />
          )}
        </For>
      </div>
    </div>
  );
}

function BetTrackerSection() {
  const [bets, setBets] = createSignal([]);

  return (
    <div class="section">
      <h2>🎯 Bet Tracker</h2>
      
      <div class="bet-summary">
        <div class="summary-item">
          <span>Today's Bets:</span>
          <strong>0</strong>
        </div>
        <div class="summary-item">
          <span>P&L:</span>
          <strong class="pnl-positive">$0</strong>
        </div>
        <div class="summary-item">
          <span>Bankroll:</span>
          <strong>$5,000</strong>
        </div>
      </div>

      <Show when={bets().length === 0}>
        <div class="empty-state">
          <p>No bets placed yet</p>
          <p class="small">Week 1: Learn and validate</p>
        </div>
      </Show>
    </div>
  );
}

function GameCard(props) {
  const game = props.game;
  const awayTeam = game.awayTeam?.teamTricode || 'N/A';
  const homeTeam = game.homeTeam?.teamTricode || 'N/A';
  const awayScore = game.awayTeam?.score || 0;
  const homeScore = game.homeTeam?.score || 0;
  const status = game.gameStatusText || 'Scheduled';

  return (
    <div class="game-card">
      <div class="game-header">
        <span class="teams">{awayTeam} @ {homeTeam}</span>
        <span class="status">{status}</span>
      </div>
      <div class="score">
        <span>{awayTeam} {awayScore}</span>
        <span>-</span>
        <span>{homeTeam} {homeScore}</span>
      </div>
    </div>
  );
}

function PredictionCard(props) {
  const game = props.game;
  const pred = props.prediction;
  
  return (
    <div class="prediction-card">
      <h4>{game.awayTeam?.teamTricode} @ {game.homeTeam?.teamTricode}</h4>
      
      <Show when={pred} fallback={<p class="waiting">Waiting for 18-minute mark...</p>}>
        <div class="pred-values">
          <div class="pred-item">
            <span class="label">Halftime</span>
            <span class="value">{pred.halftime > 0 ? '+' : ''}{pred.halftime?.toFixed(1)}</span>
          </div>
          <div class="pred-item">
            <span class="label">Final</span>
            <span class="value">{pred.final > 0 ? '+' : ''}{pred.final?.toFixed(1)}</span>
          </div>
        </div>
        
        <div class={`confidence confidence-${pred.confidence?.toLowerCase()}`}>
          {pred.confidence} Confidence
        </div>
      </Show>
    </div>
  );
}

function OddsCard(props) {
  const game = props.game;
  const odds = props.odds;
  
  return (
    <div class="odds-card">
      <h4>{game.awayTeam?.teamTricode} @ {game.homeTeam?.teamTricode}</h4>
      
      <Show when={odds} fallback={<p class="waiting">Fetching odds...</p>}>
        <div class="odds-values">
          <div class="odds-row">
            <span class="label">1H Spread</span>
            <span class="value">{game.homeTeam?.teamTricode} {odds.spread_1h || '-'}</span>
          </div>
          <div class="odds-row">
            <span class="label">FG Spread</span>
            <span class="value">{game.homeTeam?.teamTricode} {odds.spread_fg || '-'}</span>
          </div>
          <div class="odds-row">
            <span class="label">Total</span>
            <span class="value">{odds.total || '-'}</span>
          </div>
        </div>
      </Show>
    </div>
  );
}

export default App;

