/**
 * TypeScript types for NBA Dashboard
 * Real-time prediction and betting system
 */

// NBA Game data
export interface NBAGame {
  game_id: string;
  home_team: string;
  away_team: string;
  score_home: number;
  score_away: number;
  quarter: number;
  time_remaining: string;
  is_live: boolean;
}

// ML Prediction from Ensemble
export interface Prediction {
  point_forecast: number;
  interval_lower: number;
  interval_upper: number;
  coverage_probability: number;
  timestamp: string;
}

// Score differential pattern (18 minutes)
export interface ScorePattern {
  minute: number;
  differential: number;
}

// Edge detection
export interface Edge {
  has_edge: boolean;
  edge_size: number;
  direction: 'home' | 'away';
  confidence: 'high' | 'medium' | 'low';
  ml_forecast: number;
  market_spread: number;
}

// Betting recommendation from Risk system
export interface BettingRecommendation {
  final_bet: number;
  kelly_bet: number;
  delta_bet: number;
  portfolio_bet: number;
  decision_tree_bet: number;
  safety_mode: 'GREEN' | 'YELLOW' | 'RED';
  reasoning: string[];
  expected_value: number;
}

// WebSocket message types
export type WSMessage =
  | { type: 'score_update'; data: NBAGame }
  | { type: 'pattern_progress'; data: { game_id: string; pattern: ScorePattern[] } }
  | { type: 'ml_prediction'; data: { game_id: string; prediction: Prediction } }
  | { type: 'edge_detected'; data: { game_id: string; edge: Edge } }
  | { type: 'bet_recommendation'; data: { game_id: string; recommendation: BettingRecommendation } };

// Dashboard state
export interface DashboardState {
  games: Map<string, NBAGame>;
  predictions: Map<string, Prediction>;
  patterns: Map<string, ScorePattern[]>;
  edges: Map<string, Edge>;
  recommendations: Map<string, BettingRecommendation>;
  connected: boolean;
  lastUpdate: Date;
}

