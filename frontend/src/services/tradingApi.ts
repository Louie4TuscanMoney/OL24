/**
 * Trading API Service
 * Handles all API calls to the trading dashboard backend
 */

const API_BASE = 'https://ol24-production.up.railway.app';

export interface BetInput {
  game_id: string;
  bet_type: string;
  side: string;
  odds: number;
  stake: number;
  book?: string;
}

export interface BetAnalysis {
  bet_input: BetInput;
  mamba_prediction: number;
  mamba_confidence: number;
  implied_probability: number;
  mamba_probability: number;
  expected_value: number;
  expected_profit: number;
  kelly_stake: number;
  recommendation: string;
  risk_level: string;
}

export interface BettingOpportunity {
  game_id: string;
  home_team: string;
  away_team: string;
  current_score: string;
  current_margin: number;
  mamba_prediction: number;
  mamba_confidence: number;
  home_win_probability: number;
  away_win_probability: number;
  opportunities: Array<{
    type: string;
    odds: number;
    ev: number;
    recommendation: string;
  }>;
  triggered_at: string;
}

export interface TradingPerformance {
  total_bets: number;
  wins: number;
  losses: number;
  win_rate: number;
  total_profit_loss: number;
  total_staked: number;
  roi: number;
  avg_expected_value: number;
  recent_bets: Array<any>;
}

export const tradingApi = {
  async getLiveOpportunities(): Promise<{ opportunities: BettingOpportunity[]; count: number }> {
    const response = await fetch(`${API_BASE}/api/trading/live-opportunities`);
    return response.json();
  },

  async analyzeBet(bet: BetInput): Promise<BetAnalysis> {
    const response = await fetch(`${API_BASE}/api/trading/analyze-bet`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(bet)
    });
    return response.json();
  },

  async placeBet(bet: BetInput): Promise<{ bet_id: number; status: string; expected_value: number; message: string }> {
    const response = await fetch(`${API_BASE}/api/trading/place-bet`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(bet)
    });
    return response.json();
  },

  async getPerformance(): Promise<TradingPerformance> {
    const response = await fetch(`${API_BASE}/api/trading/performance`);
    return response.json();
  },

  async getMambaPerformance(): Promise<any> {
    const response = await fetch(`${API_BASE}/api/mamba/performance`);
    return response.json();
  },

  createMambaWebSocket(gameId: string): WebSocket {
    return new WebSocket(`wss://ol24-production.up.railway.app/ws/mamba/${gameId}`);
  }
};
