/**
 * INTERACTIVE TRADING DASHBOARD
 * 
 * Sports betting dashboard with:
 * - Live Mamba predictions
 * - Interactive odds calculator
 * - EV computation
 * - Kelly Criterion betting
 * - Real-time P&L tracking
 */

import { createSignal, createEffect, Show, For } from 'solid-js';

interface BetOpportunity {
  game_id: string;
  home_team: string;
  away_team: string;
  current_score: string;
  mamba_prediction: number;
  mamba_confidence: number;
  home_win_probability: number;
  opportunities: Array<{
    type: string;
    odds: number;
    ev: number;
    recommendation: string;
  }>;
}

export function TradingDashboard() {
  const [opportunities, setOpportunities] = createSignal<BetOpportunity[]>([]);
  const [selectedGame, setSelectedGame] = createSignal<string | null>(null);
  const [bankroll, setBankroll] = createSignal(1000);
  const [customOdds, setCustomOdds] = createSignal(-110);
  const [betAmount, setBetAmount] = createSignal(100);
  const [betAnalysis, setBetAnalysis] = createSignal<any>(null);
  const [performance, setPerformance] = createSignal<any>(null);

  // Fetch live opportunities
  createEffect(() => {
    fetchOpportunities();
    const interval = setInterval(fetchOpportunities, 10000); // Every 10 seconds
    return () => clearInterval(interval);
  });

  // Fetch performance stats
  createEffect(() => {
    fetchPerformance();
  });

  async function fetchOpportunities() {
    try {
      const response = await fetch('/api/trading/live-opportunities');
      const data = await response.json();
      setOpportunities(data.opportunities || []);
    } catch (error) {
      console.error('Error fetching opportunities:', error);
    }
  }

  async function fetchPerformance() {
    try {
      const response = await fetch('/api/trading/performance');
      const data = await response.json();
      setPerformance(data);
    } catch (error) {
      console.error('Error fetching performance:', error);
    }
  }

  async function analyzeBet(gameId: string, betType: string, side: string) {
    try {
      const response = await fetch('/api/trading/analyze-bet', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          game_id: gameId,
          bet_type: betType,
          side: side,
          odds: customOdds(),
          stake: betAmount()
        })
      });
      const data = await response.json();
      setBetAnalysis(data);
    } catch (error) {
      console.error('Error analyzing bet:', error);
    }
  }

  async function placeBet(gameId: string, betType: string, side: string) {
    if (!confirm(`Place $${betAmount()} bet on ${side} ${betType}?`)) return;

    try {
      const response = await fetch('/api/trading/place-bet', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          game_id: gameId,
          bet_type: betType,
          side: side,
          odds: customOdds(),
          stake: betAmount(),
          book: 'DraftKings'
        })
      });
      const data = await response.json();
      alert(data.message);
      fetchPerformance(); // Refresh stats
    } catch (error) {
      console.error('Error placing bet:', error);
      alert('Error placing bet');
    }
  }

  return (
    <div class="trading-dashboard">
      {/* Header */}
      <div class="dashboard-header">
        <h1>🎯 Live Trading Dashboard</h1>
        <div class="bankroll-display">
          <label>Bankroll:</label>
          <input
            type="number"
            value={bankroll()}
            onInput={(e) => setBankroll(parseFloat(e.currentTarget.value))}
            class="bankroll-input"
          />
        </div>
      </div>

      {/* Performance Stats */}
      <Show when={performance()}>
        {(perf) => (
          <div class="performance-panel">
            <div class="stat-card">
              <div class="stat-label">Total Bets</div>
              <div class="stat-value">{perf().total_bets}</div>
            </div>
            <div class="stat-card">
              <div class="stat-label">Win Rate</div>
              <div class="stat-value success">{perf().win_rate}%</div>
            </div>
            <div class="stat-card">
              <div class="stat-label">Total P&L</div>
              <div classList={{
                'stat-value': true,
                'success': perf().total_profit_loss > 0,
                'danger': perf().total_profit_loss < 0
              }}>
                ${perf().total_profit_loss.toFixed(2)}
              </div>
            </div>
            <div class="stat-card">
              <div class="stat-label">ROI</div>
              <div classList={{
                'stat-value': true,
                'success': perf().roi > 0,
                'danger': perf().roi < 0
              }}>
                {perf().roi}%
              </div>
            </div>
          </div>
        )}
      </Show>

      {/* Live Opportunities */}
      <div class="opportunities-section">
        <h2>Live Betting Opportunities</h2>
        
        <Show when={opportunities().length === 0}>
          <div class="empty-state">
            <p>⏳ Waiting for live games with Mamba predictions...</p>
            <p class="empty-subtitle">Predictions trigger at Q2 6:00</p>
          </div>
        </Show>

        <For each={opportunities()}>
          {(opp) => (
            <div class="opportunity-card">
              {/* Game Info */}
              <div class="game-header">
                <div class="matchup">
                  <span class="team">{opp.away_team}</span>
                  <span class="at">@</span>
                  <span class="team">{opp.home_team}</span>
                </div>
                <div class="score">{opp.current_score}</div>
              </div>

              {/* Mamba Prediction */}
              <div class="mamba-info">
                <div class="prediction-value">
                  <span class="label">Mamba Prediction:</span>
                  <span classList={{
                    'value': true,
                    'positive': opp.mamba_prediction > 0,
                    'negative': opp.mamba_prediction < 0
                  }}>
                    {opp.mamba_prediction > 0 ? '+' : ''}{opp.mamba_prediction.toFixed(1)}
                  </span>
                </div>
                <div class="confidence">
                  <span class="label">Confidence:</span>
                  <span class="value">{opp.mamba_confidence}%</span>
                </div>
                <div class="probabilities">
                  <div>Home: {opp.home_win_probability}%</div>
                  <div>Away: {100 - opp.home_win_probability}%</div>
                </div>
              </div>

              {/* Betting Options */}
              <div class="betting-options">
                <h4>Betting Opportunities:</h4>
                <For each={opp.opportunities}>
                  {(bet) => (
                    <div classList={{
                      'bet-option': true,
                      'recommended': bet.recommendation === 'BET',
                      'strong-bet': bet.recommendation === 'BET' && bet.ev > 5
                    }}>
                      <div class="bet-info">
                        <span class="bet-type">{bet.type.replace('_', ' ')}</span>
                        <span class="odds">{bet.odds > 0 ? '+' : ''}{bet.odds}</span>
                      </div>
                      <div class="bet-metrics">
                        <span classList={{
                          'ev': true,
                          'positive': bet.ev > 0,
                          'negative': bet.ev < 0
                        }}>
                          EV: {bet.ev.toFixed(1)}%
                        </span>
                        <span classList={{
                          'recommendation': true,
                          'bet': bet.recommendation === 'BET',
                          'pass': bet.recommendation === 'PASS'
                        }}>
                          {bet.recommendation}
                        </span>
                      </div>
                      <button
                        class="analyze-btn"
                        onClick={() => {
                          setSelectedGame(opp.game_id);
                          analyzeBet(opp.game_id, bet.type, bet.type.includes('home') ? 'home' : 'away');
                        }}
                      >
                        Analyze
                      </button>
                    </div>
                  )}
                </For>
              </div>

              {/* Interactive Calculator (if selected) */}
              <Show when={selectedGame() === opp.game_id}>
                <div class="calculator-panel">
                  <h4>Custom Bet Calculator</h4>
                  
                  <div class="calculator-inputs">
                    <div class="input-group">
                      <label>Odds (American)</label>
                      <input
                        type="number"
                        value={customOdds()}
                        onInput={(e) => setCustomOdds(parseFloat(e.currentTarget.value))}
                        class="odds-input"
                        placeholder="-110"
                      />
                    </div>
                    
                    <div class="input-group">
                      <label>Bet Amount ($)</label>
                      <input
                        type="number"
                        value={betAmount()}
                        onInput={(e) => setBetAmount(parseFloat(e.currentTarget.value))}
                        class="amount-input"
                        placeholder="100"
                      />
                    </div>

                    <button
                      class="calculate-btn"
                      onClick={() => analyzeBet(opp.game_id, 'spread', 'home')}
                    >
                      Calculate EV
                    </button>
                  </div>

                  {/* Analysis Results */}
                  <Show when={betAnalysis()}>
                    {(analysis) => (
                      <div class="analysis-results">
                        <h5>Analysis Results</h5>
                        
                        <div class="result-grid">
                          <div class="result-item">
                            <span class="result-label">Implied Probability</span>
                            <span class="result-value">{analysis().implied_probability}%</span>
                          </div>
                          
                          <div class="result-item">
                            <span class="result-label">Mamba Probability</span>
                            <span class="result-value">{analysis().mamba_probability}%</span>
                          </div>
                          
                          <div class="result-item highlight">
                            <span class="result-label">Expected Value</span>
                            <span classList={{
                              'result-value': true,
                              'positive': analysis().expected_value > 0,
                              'negative': analysis().expected_value < 0
                            }}>
                              ${analysis().expected_value.toFixed(2)}
                            </span>
                          </div>
                          
                          <div class="result-item highlight">
                            <span class="result-label">Expected Profit</span>
                            <span classList={{
                              'result-value': true,
                              'positive': analysis().expected_profit > 0,
                              'negative': analysis().expected_profit < 0
                            }}>
                              ${analysis().expected_profit.toFixed(2)}
                            </span>
                          </div>
                          
                          <div class="result-item">
                            <span class="result-label">Kelly Stake</span>
                            <span class="result-value">${analysis().kelly_stake.toFixed(2)}</span>
                          </div>
                          
                          <div class="result-item">
                            <span class="result-label">Risk Level</span>
                            <span classList={{
                              'result-value': true,
                              'risk-low': analysis().risk_level === 'LOW',
                              'risk-medium': analysis().risk_level === 'MEDIUM',
                              'risk-high': analysis().risk_level === 'HIGH'
                            }}>
                              {analysis().risk_level}
                            </span>
                          </div>
                        </div>

                        <div class="recommendation-box">
                          <strong>Recommendation:</strong> {analysis().recommendation}
                        </div>

                        <button
                          class="place-bet-btn"
                          onClick={() => placeBet(opp.game_id, 'spread', 'home')}
                        >
                          Place Bet - ${betAmount()}
                        </button>
                      </div>
                    )}
                  </Show>
                </div>
              </Show>
            </div>
          )}
        </For>
      </div>

      <style>{`
        .trading-dashboard {
          max-width: 1400px;
          margin: 0 auto;
          padding: 20px;
          background: #0f172a;
          min-height: 100vh;
        }

        .dashboard-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 30px;
          padding: 20px;
          background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 100%);
          border-radius: 12px;
          color: white;
        }

        .dashboard-header h1 {
          margin: 0;
          font-size: 28px;
        }

        .bankroll-display {
          display: flex;
          align-items: center;
          gap: 10px;
        }

        .bankroll-input {
          width: 150px;
          padding: 8px 12px;
          border-radius: 6px;
          border: 2px solid #3b82f6;
          background: rgba(255, 255, 255, 0.1);
          color: white;
          font-size: 18px;
          font-weight: bold;
        }

        .performance-panel {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
          gap: 16px;
          margin-bottom: 30px;
        }

        .stat-card {
          background: rgba(255, 255, 255, 0.05);
          padding: 20px;
          border-radius: 8px;
          text-align: center;
        }

        .stat-label {
          font-size: 14px;
          color: #94a3b8;
          margin-bottom: 8px;
        }

        .stat-value {
          font-size: 32px;
          font-weight: bold;
          color: white;
        }

        .stat-value.success {
          color: #10b981;
        }

        .stat-value.danger {
          color: #ef4444;
        }

        .opportunities-section h2 {
          color: white;
          margin-bottom: 20px;
        }

        .empty-state {
          text-align: center;
          padding: 60px 20px;
          color: #94a3b8;
        }

        .empty-state p {
          font-size: 18px;
          margin: 10px 0;
        }

        .empty-subtitle {
          font-size: 14px !important;
          opacity: 0.7;
        }

        .opportunity-card {
          background: rgba(255, 255, 255, 0.05);
          border-radius: 12px;
          padding: 24px;
          margin-bottom: 20px;
          border: 1px solid rgba(255, 255, 255, 0.1);
        }

        .game-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 20px;
          padding-bottom: 16px;
          border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }

        .matchup {
          font-size: 20px;
          font-weight: bold;
          color: white;
        }

        .matchup .at {
          margin: 0 12px;
          color: #64748b;
        }

        .score {
          font-size: 24px;
          font-weight: bold;
          color: #fbbf24;
        }

        .mamba-info {
          display: grid;
          grid-template-columns: repeat(3, 1fr);
          gap: 16px;
          margin-bottom: 20px;
          padding: 16px;
          background: rgba(59, 130, 246, 0.1);
          border-radius: 8px;
        }

        .mamba-info .label {
          color: #94a3b8;
          font-size: 14px;
          display: block;
          margin-bottom: 4px;
        }

        .mamba-info .value {
          color: white;
          font-size: 20px;
          font-weight: bold;
        }

        .value.positive {
          color: #10b981;
        }

        .value.negative {
          color: #ef4444;
        }

        .betting-options h4 {
          color: white;
          margin-bottom: 12px;
        }

        .bet-option {
          background: rgba(255, 255, 255, 0.03);
          padding: 16px;
          border-radius: 8px;
          margin-bottom: 12px;
          border: 2px solid transparent;
          display: flex;
          justify-content: space-between;
          align-items: center;
        }

        .bet-option.recommended {
          border-color: #3b82f6;
        }

        .bet-option.strong-bet {
          border-color: #10b981;
          background: rgba(16, 185, 129, 0.1);
        }

        .bet-info {
          display: flex;
          gap: 16px;
          align-items: center;
        }

        .bet-type {
          color: white;
          font-weight: bold;
          text-transform: uppercase;
        }

        .odds {
          color: #fbbf24;
          font-weight: bold;
        }

        .bet-metrics {
          display: flex;
          gap: 16px;
          align-items: center;
        }

        .ev {
          font-weight: bold;
          padding: 4px 8px;
          border-radius: 4px;
          background: rgba(0, 0, 0, 0.3);
        }

        .ev.positive {
          color: #10b981;
        }

        .ev.negative {
          color: #ef4444;
        }

        .recommendation {
          padding: 6px 12px;
          border-radius: 6px;
          font-weight: bold;
          font-size: 12px;
        }

        .recommendation.bet {
          background: #10b981;
          color: white;
        }

        .recommendation.pass {
          background: #64748b;
          color: white;
        }

        .analyze-btn {
          padding: 8px 16px;
          background: #3b82f6;
          color: white;
          border: none;
          border-radius: 6px;
          font-weight: bold;
          cursor: pointer;
        }

        .analyze-btn:hover {
          background: #2563eb;
        }

        .calculator-panel {
          margin-top: 20px;
          padding: 20px;
          background: rgba(59, 130, 246, 0.1);
          border-radius: 8px;
          border: 2px solid #3b82f6;
        }

        .calculator-panel h4 {
          color: white;
          margin-bottom: 16px;
        }

        .calculator-inputs {
          display: grid;
          grid-template-columns: 1fr 1fr auto;
          gap: 16px;
          margin-bottom: 20px;
        }

        .input-group label {
          display: block;
          color: #94a3b8;
          font-size: 14px;
          margin-bottom: 8px;
        }

        .odds-input, .amount-input {
          width: 100%;
          padding: 10px;
          border-radius: 6px;
          border: 2px solid #3b82f6;
          background: rgba(0, 0, 0, 0.3);
          color: white;
          font-size: 16px;
        }

        .calculate-btn {
          padding: 10px 24px;
          background: #10b981;
          color: white;
          border: none;
          border-radius: 6px;
          font-weight: bold;
          cursor: pointer;
          align-self: end;
        }

        .calculate-btn:hover {
          background: #059669;
        }

        .analysis-results {
          background: rgba(0, 0, 0, 0.3);
          padding: 20px;
          border-radius: 8px;
        }

        .analysis-results h5 {
          color: white;
          margin-bottom: 16px;
        }

        .result-grid {
          display: grid;
          grid-template-columns: repeat(3, 1fr);
          gap: 16px;
          margin-bottom: 20px;
        }

        .result-item {
          padding: 12px;
          background: rgba(255, 255, 255, 0.05);
          border-radius: 6px;
        }

        .result-item.highlight {
          background: rgba(59, 130, 246, 0.2);
          border: 2px solid #3b82f6;
        }

        .result-label {
          display: block;
          color: #94a3b8;
          font-size: 12px;
          margin-bottom: 8px;
        }

        .result-value {
          display: block;
          color: white;
          font-size: 20px;
          font-weight: bold;
        }

        .result-value.risk-low {
          color: #10b981;
        }

        .result-value.risk-medium {
          color: #fbbf24;
        }

        .result-value.risk-high {
          color: #ef4444;
        }

        .recommendation-box {
          padding: 16px;
          background: rgba(59, 130, 246, 0.2);
          border-radius: 6px;
          color: white;
          margin-bottom: 16px;
        }

        .place-bet-btn {
          width: 100%;
          padding: 16px;
          background: #10b981;
          color: white;
          border: none;
          border-radius: 8px;
          font-size: 18px;
          font-weight: bold;
          cursor: pointer;
        }

        .place-bet-btn:hover {
          background: #059669;
        }
      `}</style>
    </div>
  );
}

