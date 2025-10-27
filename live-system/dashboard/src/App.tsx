import { createSignal, onMount, onCleanup, For, Show } from 'solid-js';
import axios from 'axios';

const API_URL = 'http://localhost:8001';

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
  game_id: string;
  matchup: string;
  current_score: string;
  period: string;
  prediction: number;
  market_spread: number;
  edge: number;
  p_win: number;
  kelly_edge: number;
  bet_recommended: boolean;
  bet_line: string;
  recommended_stake: number;
}

interface RiskStatus {
  current_bankroll: number;
  peak_bankroll: number;
  current_drawdown: number;
  daily_loss: number;
  can_bet: boolean;
  alerts: string[];
}

function App() {
  const [liveGames, setLiveGames] = createSignal<Game[]>([]);
  const [opportunities, setOpportunities] = createSignal<Opportunity[]>([]);
  const [riskStatus, setRiskStatus] = createSignal<RiskStatus | null>(null);
  const [loading, setLoading] = createSignal(true);
  const [lastUpdate, setLastUpdate] = createSignal<string>('');

  // Fetch data
  const fetchData = async () => {
    try {
      // Get live games
      const gamesRes = await axios.get(`${API_URL}/api/live-games`);
      setLiveGames(gamesRes.data.games || []);

      // Get opportunities
      const oppsRes = await axios.get(`${API_URL}/api/opportunities`);
      setOpportunities(oppsRes.data.opportunities || []);

      // Get risk status
      try {
        const riskRes = await axios.get(`${API_URL}/api/risk-status`);
        setRiskStatus(riskRes.data.risk_status);
      } catch (e) {
        console.warn('Risk status unavailable:', e);
      }

      setLastUpdate(new Date().toLocaleTimeString());
      setLoading(false);
    } catch (error) {
      console.error('Error fetching data:', error);
      setLoading(false);
    }
  };

  // Auto-refresh every 10 seconds
  let interval: number;
  onMount(() => {
    fetchData();
    interval = setInterval(fetchData, 10000) as unknown as number;
  });

  onCleanup(() => {
    if (interval) clearInterval(interval);
  });

  return (
    <div style={{
      "min-height": "100vh",
      "background": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
      "padding": "20px",
      "color": "#fff",
      "font-family": "system-ui, -apple-system, sans-serif"
    }}>
      {/* Header */}
      <div style={{
        "background": "rgba(255,255,255,0.1)",
        "backdrop-filter": "blur(10px)",
        "border-radius": "20px",
        "padding": "30px",
        "margin-bottom": "20px",
        "box-shadow": "0 8px 32px rgba(0,0,0,0.1)"
      }}>
        <h1 style={{
          "margin": "0 0 10px 0",
          "font-size": "2.5rem",
          "font-weight": "bold"
        }}>
          🔥 Ontologic XYZ - Live Trading Dashboard
        </h1>
        <p style={{ "margin": "0", "opacity": "0.9" }}>
          NBA Predictions • OntoRisk • Real-time Opportunities
        </p>
        <p style={{ "margin": "10px 0 0 0", "font-size": "0.9rem", "opacity": "0.7" }}>
          Last Update: {lastUpdate()}
        </p>
      </div>

      {/* Risk Status */}
      <Show when={riskStatus()}>
        <div style={{
          "background": riskStatus()!.can_bet 
            ? "rgba(16, 185, 129, 0.2)" 
            : "rgba(239, 68, 68, 0.2)",
          "backdrop-filter": "blur(10px)",
          "border-radius": "20px",
          "padding": "25px",
          "margin-bottom": "20px",
          "border": `2px solid ${riskStatus()!.can_bet ? '#10b981' : '#ef4444'}`
        }}>
          <h2 style={{ "margin": "0 0 15px 0", "font-size": "1.5rem" }}>
            🛡️ Risk Status
          </h2>
          <div style={{ "display": "grid", "grid-template-columns": "repeat(auto-fit, minmax(200px, 1fr))", "gap": "15px" }}>
            <div>
              <div style={{ "font-size": "0.9rem", "opacity": "0.8" }}>Bankroll</div>
              <div style={{ "font-size": "1.8rem", "font-weight": "bold" }}>
                ${riskStatus()!.current_bankroll.toLocaleString()}
              </div>
            </div>
            <div>
              <div style={{ "font-size": "0.9rem", "opacity": "0.8" }}>Drawdown</div>
              <div style={{ "font-size": "1.8rem", "font-weight": "bold" }}>
                {(riskStatus()!.current_drawdown * 100).toFixed(1)}%
              </div>
            </div>
            <div>
              <div style={{ "font-size": "0.9rem", "opacity": "0.8" }}>Daily P&L</div>
              <div style={{ 
                "font-size": "1.8rem", 
                "font-weight": "bold",
                "color": riskStatus()!.daily_loss < 0 ? '#10b981' : '#ef4444'
              }}>
                ${(-riskStatus()!.daily_loss).toLocaleString()}
              </div>
            </div>
            <div>
              <div style={{ "font-size": "0.9rem", "opacity": "0.8" }}>Status</div>
              <div style={{ "font-size": "1.8rem", "font-weight": "bold" }}>
                {riskStatus()!.can_bet ? '✅ ACTIVE' : '🚨 HALTED'}
              </div>
            </div>
          </div>
        </div>
      </Show>

      {/* Betting Opportunities */}
      <div style={{
        "background": "rgba(255,255,255,0.1)",
        "backdrop-filter": "blur(10px)",
        "border-radius": "20px",
        "padding": "25px",
        "margin-bottom": "20px"
      }}>
        <h2 style={{ "margin": "0 0 20px 0", "font-size": "1.5rem" }}>
          🎯 Betting Opportunities ({opportunities().length})
        </h2>
        
        <Show when={loading()}>
          <p>Loading opportunities...</p>
        </Show>

        <Show when={!loading() && opportunities().length === 0}>
          <div style={{
            "padding": "40px",
            "text-align": "center",
            "opacity": "0.7"
          }}>
            <p style={{ "font-size": "1.2rem" }}>No opportunities right now</p>
            <p>Waiting for games with Edge ≥ 5 points and P(Win) ≥ 55%</p>
          </div>
        </Show>

        <div style={{ "display": "grid", "gap": "15px" }}>
          <For each={opportunities()}>
            {(opp) => (
              <div style={{
                "background": "linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(6, 182, 212, 0.2) 100%)",
                "border-radius": "15px",
                "padding": "20px",
                "border": "2px solid rgba(16, 185, 129, 0.5)"
              }}>
                <div style={{ "display": "flex", "justify-content": "space-between", "align-items": "start" }}>
                  <div>
                    <h3 style={{ "margin": "0 0 10px 0", "font-size": "1.3rem" }}>
                      {opp.matchup}
                    </h3>
                    <p style={{ "margin": "0 0 5px 0", "opacity": "0.8" }}>
                      Current: {opp.current_score} ({opp.period})
                    </p>
                  </div>
                  <div style={{ "text-align": "right" }}>
                    <div style={{ "font-size": "2rem", "font-weight": "bold", "color": "#10b981" }}>
                      {opp.edge.toFixed(1)} pts
                    </div>
                    <div style={{ "font-size": "0.9rem", "opacity": "0.8" }}>EDGE</div>
                  </div>
                </div>

                <div style={{ 
                  "display": "grid", 
                  "grid-template-columns": "repeat(auto-fit, minmax(150px, 1fr))", 
                  "gap": "15px",
                  "margin": "20px 0"
                }}>
                  <div>
                    <div style={{ "font-size": "0.85rem", "opacity": "0.7" }}>Our Prediction</div>
                    <div style={{ "font-size": "1.2rem", "font-weight": "bold" }}>
                      {opp.prediction > 0 ? '+' : ''}{opp.prediction.toFixed(1)}
                    </div>
                  </div>
                  <div>
                    <div style={{ "font-size": "0.85rem", "opacity": "0.7" }}>Market Spread</div>
                    <div style={{ "font-size": "1.2rem", "font-weight": "bold" }}>
                      {opp.market_spread > 0 ? '+' : ''}{opp.market_spread.toFixed(1)}
                    </div>
                  </div>
                  <div>
                    <div style={{ "font-size": "0.85rem", "opacity": "0.7" }}>Win Probability</div>
                    <div style={{ "font-size": "1.2rem", "font-weight": "bold" }}>
                      {(opp.p_win * 100).toFixed(1)}%
                    </div>
                  </div>
                  <div>
                    <div style={{ "font-size": "0.85rem", "opacity": "0.7" }}>Kelly Edge</div>
                    <div style={{ "font-size": "1.2rem", "font-weight": "bold" }}>
                      {(opp.kelly_edge * 100).toFixed(1)}%
                    </div>
                  </div>
                </div>

                <div style={{
                  "background": "rgba(0,0,0,0.3)",
                  "border-radius": "10px",
                  "padding": "15px",
                  "margin-bottom": "15px"
                }}>
                  <div style={{ "font-size": "1.1rem", "font-weight": "bold", "margin-bottom": "5px" }}>
                    ✅ Recommended Bet:
                  </div>
                  <div style={{ "font-size": "1.3rem", "color": "#10b981" }}>
                    {opp.bet_line}
                  </div>
                </div>

                <div style={{ "display": "flex", "justify-content": "space-between", "align-items": "center" }}>
                  <div>
                    <div style={{ "font-size": "0.85rem", "opacity": "0.7" }}>Recommended Stake</div>
                    <div style={{ "font-size": "1.5rem", "font-weight": "bold" }}>
                      ${opp.recommended_stake.toLocaleString()}
                    </div>
                  </div>
                  <button 
                    style={{
                      "background": "linear-gradient(135deg, #10b981, #06b6d4)",
                      "border": "none",
                      "border-radius": "10px",
                      "padding": "15px 30px",
                      "color": "#fff",
                      "font-size": "1.1rem",
                      "font-weight": "bold",
                      "cursor": "pointer",
                      "box-shadow": "0 4px 15px rgba(0,0,0,0.2)"
                    }}
                    onClick={() => placeBet(opp)}
                  >
                    PLACE BET
                  </button>
                </div>
              </div>
            )}
          </For>
        </div>
      </div>

      {/* Live Games */}
      <div style={{
        "background": "rgba(255,255,255,0.1)",
        "backdrop-filter": "blur(10px)",
        "border-radius": "20px",
        "padding": "25px"
      }}>
        <h2 style={{ "margin": "0 0 20px 0", "font-size": "1.5rem" }}>
          🏀 Live Games ({liveGames().length})
        </h2>
        
        <Show when={liveGames().length === 0}>
          <div style={{
            "padding": "40px",
            "text-align": "center",
            "opacity": "0.7"
          }}>
            <p style={{ "font-size": "1.2rem" }}>No live games right now</p>
            <p>Dashboard will update when games start</p>
          </div>
        </Show>

        <div style={{ "display": "grid", "gap": "10px" }}>
          <For each={liveGames()}>
            {(game) => (
              <div style={{
                "background": game.status_text === 'LIVE' 
                  ? "rgba(239, 68, 68, 0.2)" 
                  : "rgba(255,255,255,0.1)",
                "border-radius": "10px",
                "padding": "15px",
                "display": "flex",
                "justify-content": "space-between",
                "align-items": "center",
                "border": game.is_q2_6min ? "2px solid #fbbf24" : "none"
              }}>
                <div>
                  <div style={{ "font-weight": "bold", "font-size": "1.1rem" }}>
                    {game.can_predict && '🎯 '}
                    {game.is_q2_6min && '⭐ '}
                    {game.away_team} @ {game.home_team}
                  </div>
                  <div style={{ "opacity": "0.8", "margin-top": "5px" }}>
                    Q{game.period} {game.clock} • {game.status_text}
                  </div>
                </div>
                <div style={{ "text-align": "right" }}>
                  <div style={{ "font-size": "1.5rem", "font-weight": "bold" }}>
                    {game.away_score} - {game.home_score}
                  </div>
                  <div style={{ "opacity": "0.8", "margin-top": "5px" }}>
                    Diff: {game.current_diff > 0 ? '+' : ''}{game.current_diff}
                  </div>
                </div>
              </div>
            )}
          </For>
        </div>
      </div>

      {/* Footer */}
      <div style={{
        "text-align": "center",
        "margin-top": "30px",
        "opacity": "0.6",
        "font-size": "0.9rem"
      }}>
        <p>Ontologic XYZ • Mamba Mentality System • 9.0 MAE</p>
        <p>Powered by ML + OntoRisk • Auto-updates every 10s</p>
      </div>
    </div>
  );

  function placeBet(opp: Opportunity) {
    alert(`Bet logged: ${opp.bet_line} for $${opp.recommended_stake}\n\nIn production, this would place the bet with your sportsbook.`);
    
    // In production: Send to API
    axios.post(`${API_URL}/api/place-bet`, {
      matchup: opp.matchup,
      bet_line: opp.bet_line,
      stake: opp.recommended_stake,
      edge: opp.edge,
      p_win: opp.p_win
    }).then(() => {
      console.log('Bet placed successfully');
    }).catch(err => {
      console.error('Error placing bet:', err);
    });
  }
}

export default App;

