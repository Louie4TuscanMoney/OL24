/**
 * TypeScript types for NBA Dashboard
 * Real-time prediction and betting system
 * ENHANCED: All 33 Mamba features included
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

// ============================================================================
// ENHANCED: ALL 33 MAMBA FEATURES
// ============================================================================

export interface MambaFeatures {
  // Pattern Analysis (10 features)
  pattern_analysis: {
    mean_diff: number;
    std_diff: number;
    trend: number;
    volatility: number;
    velocity: number;
    acceleration: number;
    recent_momentum: number;
    lead_changes: number;
    max_swing: number;
    comeback_potential: number;
  };
  
  // Spectral Features (6 features)
  spectral: {
    spectral_energy: number;
    spectral_entropy: number;
    low_freq_power: number;
    mid_freq_power: number;
    high_freq_power: number;
    dominant_freq: number;
  };
  
  // Autocorrelation (3 features)
  autocorrelation: {
    lag1: number;
    lag2: number;
    lag3: number;
  };
  
  // Advanced Stats (8 features)
  advanced_stats: {
    pace_proxy: number;
    efg_proxy: number;
    ts_proxy: number;
    netrtg_proxy: number;
    usg_proxy: number;
    pm_proxy: number;
    pie_proxy: number;
    four_factors: number;
  };
  
  // Team Form (6 features)
  team_form: {
    team_diff_lag1: number;
    team_mean_lag1: number;
    team_diff_rolling3: number;
    team_volatility_rolling3: number;
    team_form_10games: number;
    team_consistency: number;
  };
  
  // Metadata
  extraction_time_ms: number;
  pbp_events_count: number;
  pattern_length: number;
}

// Enhanced Prediction with ALL features
export interface EnhancedPrediction extends Prediction {
  features?: MambaFeatures;
  model_confidence: number;
  feature_importance?: {
    top_5: Array<{
      name: string;
      value: number;
      impact: number;
    }>;
  };
}

// WebSocket message types
export type WSMessage =
  | { type: 'score_update'; data: NBAGame }
  | { type: 'pattern_progress'; data: { game_id: string; pattern: ScorePattern[] } }
  | { type: 'ml_prediction'; data: { game_id: string; prediction: EnhancedPrediction } }
  | { type: 'edge_detected'; data: { game_id: string; edge: Edge } }
  | { type: 'bet_recommendation'; data: { game_id: string; recommendation: BettingRecommendation } };

// Dashboard state
export interface DashboardState {
  games: Map<string, NBAGame>;
  predictions: Map<string, EnhancedPrediction>;
  patterns: Map<string, ScorePattern[]>;
  edges: Map<string, Edge>;
  recommendations: Map<string, BettingRecommendation>;
  connected: boolean;
  lastUpdate: Date;
}
