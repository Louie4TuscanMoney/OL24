/**
 * WebSocket Service - Connect to NBA_API WebSocket (port 8765)
 * Real-time score updates, ML predictions, betting edges
 */

import { createSignal } from 'solid-js';
import type { WSMessage, NBAGame, Prediction, Edge, BettingRecommendation, ScorePattern } from '../types';

// WebSocket URL - Use Railway backend in production
const WS_URL = import.meta.env.VITE_WS_URL || 'wss://ol24-production.up.railway.app/ws';

export class WebSocketService {
  private ws: WebSocket | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 2000;

  // Signals for reactive state (SolidJS magic!)
  public connected = createSignal(false);
  public games = createSignal<Map<string, NBAGame>>(new Map());
  public predictions = createSignal<Map<string, Prediction>>(new Map());
  public patterns = createSignal<Map<string, ScorePattern[]>>(new Map());
  public edges = createSignal<Map<string, Edge>>(new Map());
  public recommendations = createSignal<Map<string, BettingRecommendation>>(new Map());
  public lastUpdate = createSignal<Date>(new Date());

  connect() {
    // Prevent multiple connections
    if (this.ws && (this.ws.readyState === WebSocket.CONNECTING || this.ws.readyState === WebSocket.OPEN)) {
      console.log('⚠️ WebSocket already connected/connecting, skipping');
      return;
    }

    console.log('🔌 Connecting to WebSocket:', WS_URL);

    try {
      this.ws = new WebSocket(WS_URL);

      this.ws.onopen = () => {
        console.log('✅ WebSocket connected');
        console.log('   Connection ID:', Date.now()); // Track which connection this is
        this.connected[1](true);
        this.reconnectAttempts = 0;
      };

      this.ws.onmessage = (event) => {
        try {
          const message: WSMessage = JSON.parse(event.data);
          this.handleMessage(message);
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      };

      this.ws.onerror = (error) => {
        console.error('❌ WebSocket error:', error);
      };

      this.ws.onclose = () => {
        console.log('🔌 WebSocket disconnected');
        this.connected[1](false);
        this.attemptReconnect();
      };

    } catch (error) {
      console.error('Failed to create WebSocket:', error);
      this.attemptReconnect();
    }
  }

  private lastMessageTimestamp: string = '';

  private handleMessage(message: any) {
    const now = new Date();
    this.lastUpdate[1](now);
    
    console.log(`⚡ [${now.toLocaleTimeString()}] Message received from Railway`);

    // Handle Railway backend format (type: "update")
    if (message.type === 'update') {
      // ANTI-CACHING: Ignore duplicate/old messages
      if (message.timestamp && message.timestamp === this.lastMessageTimestamp) {
        console.log('⚠️ DUPLICATE MESSAGE - Ignoring (same timestamp)');
        return;
      }
      this.lastMessageTimestamp = message.timestamp;
      
      console.log('📦 Received update from Railway:');
      console.log('   - Games:', message.live_games?.length || 0);
      console.log('   - Predictions:', message.opportunities?.length || 0);
      console.log('   - Timestamp:', message.timestamp);
      
      // Update all games
      if (message.live_games) {
        const gamesMap = new Map();
        let updatedCount = 0;
        let skippedCount = 0;
        
        message.live_games.forEach((game: any) => {
          // Map backend fields to frontend types
          const mappedGame: NBAGame = {
            game_id: game.game_id,
            home_team: game.home_team,
            away_team: game.away_team,
            score_home: game.home_score,
            score_away: game.away_score,
            quarter: game.period, // Backend sends 'period', frontend expects 'quarter'
            time_remaining: game.clock, // Backend sends 'clock', frontend expects 'time_remaining'
            clock: game.clock, // Keep original for compatibility
            is_live: game.status === 2 // status 2 = live
          };
          
          // Check if this is newer data than what we have
          const currentGames = this.games[0]();
          const existingGame = currentGames.get(game.game_id);
          if (existingGame) {
            // Only update if scores changed or clock changed
            const scoresChanged = existingGame.score_home !== mappedGame.score_home || 
                                 existingGame.score_away !== mappedGame.score_away;
            const clockChanged = existingGame.clock !== mappedGame.clock;
            
            if (!scoresChanged && !clockChanged) {
              skippedCount++;
              console.log(`   ⏭️  No changes for ${game.away_team} @ ${game.home_team}`);
              gamesMap.set(game.game_id, existingGame); // Keep existing
              return;
            }
          }
          
          gamesMap.set(game.game_id, mappedGame);
          updatedCount++;
          
          // Log score updates for debugging latency
          if (game.status === 2) {
            console.log(`   🏀 ${game.away_team} ${game.away_score} @ ${game.home_team} ${game.home_score} | Q${game.period} ${game.clock}`);
          }
        });
        this.games[1](gamesMap);
        console.log(`✅ Updated ${updatedCount} games, skipped ${skippedCount} unchanged`);
      }
      
      // Update predictions from opportunities
      if (message.opportunities) {
        const predsMap = new Map();
        const patternsMap = new Map();
        const edgesMap = new Map();
        const recsMap = new Map();
        
        message.opportunities.forEach((opp: any) => {
          const gameId = opp.game_id;
          
          // Extract prediction
          if (opp.prediction) {
            predsMap.set(gameId, {
              point_forecast: opp.prediction,
              interval_lower: opp.interval_lower || opp.prediction - 5,
              interval_upper: opp.interval_upper || opp.prediction + 5,
              coverage_probability: opp.p_win || 0.95,
              timestamp: opp.timestamp,
              mamba_features: opp.mamba_features,
              features_extracted: opp.features_extracted || false
            });
          }
          
          // Extract pattern
          if (opp.pattern) {
            patternsMap.set(gameId, opp.pattern);
          }
          
          // Extract edge
          if (opp.edge !== undefined) {
            edgesMap.set(gameId, {
              has_edge: Math.abs(opp.edge) > 5,
              edge_size: Math.abs(opp.edge),
              direction: opp.edge > 0 ? 'home' : 'away',
              confidence: Math.abs(opp.edge) <= 5 ? 'high' : Math.abs(opp.edge) <= 12 ? 'medium' : 'low',
              ml_forecast: opp.prediction || 0,
              market_spread: opp.market_spread || 0
            });
          }
          
          // Extract recommendation
          if (opp.final_bet !== undefined) {
            recsMap.set(gameId, {
              final_bet: opp.final_bet,
              kelly_bet: opp.kelly_bet || 0,
              delta_bet: opp.delta_bet || 0,
              portfolio_bet: opp.portfolio_bet || 0,
              decision_tree_bet: opp.decision_tree_bet || 0,
              should_bet: opp.should_bet || false,
              bet_side: opp.bet_side || '',
              confidence: opp.confidence || 0
            });
          }
        });
        
        this.predictions[1](predsMap);
        this.patterns[1](patternsMap);
        this.edges[1](edgesMap);
        this.recommendations[1](recsMap);
      }
      
      return;
    }

    // Legacy format (for backwards compatibility)
    switch (message.type) {
      case 'score_update':
        this.updateGame(message.data);
        break;

      case 'pattern_progress':
        this.updatePattern(message.data.game_id, message.data.pattern);
        break;

      case 'ml_prediction':
        this.updatePrediction(message.data.game_id, message.data.prediction);
        break;

      case 'edge_detected':
        this.updateEdge(message.data.game_id, message.data.edge);
        break;

      case 'bet_recommendation':
        this.updateRecommendation(message.data.game_id, message.data.recommendation);
        break;

      default:
        console.warn('Unknown message type:', message);
    }
  }

  private updateGame(game: NBAGame) {
    const [games, setGames] = this.games;
    const updated = new Map(games());
    updated.set(game.game_id, game);
    setGames(updated);
  }

  private updatePattern(gameId: string, pattern: ScorePattern[]) {
    const [patterns, setPatterns] = this.patterns;
    const updated = new Map(patterns());
    updated.set(gameId, pattern);
    setPatterns(updated);
  }

  private updatePrediction(gameId: string, prediction: Prediction) {
    const [predictions, setPredictions] = this.predictions;
    const updated = new Map(predictions());
    updated.set(gameId, prediction);
    setPredictions(updated);
  }

  private updateEdge(gameId: string, edge: Edge) {
    const [edges, setEdges] = this.edges;
    const updated = new Map(edges());
    updated.set(gameId, edge);
    setEdges(updated);
  }

  private updateRecommendation(gameId: string, recommendation: BettingRecommendation) {
    const [recommendations, setRecommendations] = this.recommendations;
    const updated = new Map(recommendations());
    updated.set(gameId, recommendation);
    setRecommendations(updated);
  }

  private attemptReconnect() {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      console.log(`🔄 Reconnecting... (attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts})`);

      setTimeout(() => {
        this.connect();
      }, this.reconnectDelay * this.reconnectAttempts);
    } else {
      console.error('❌ Max reconnect attempts reached');
    }
  }

  disconnect() {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }

  send(message: any) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(message));
    } else {
      console.warn('WebSocket not connected');
    }
  }
}

// Singleton instance
export const wsService = new WebSocketService();

