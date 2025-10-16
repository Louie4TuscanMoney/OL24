/**
 * WebSocket Service - Connect to NBA_API WebSocket (port 8765)
 * Real-time score updates, ML predictions, betting edges
 */

import { createSignal } from 'solid-js';
import type { WSMessage, NBAGame, Prediction, Edge, BettingRecommendation, ScorePattern } from '../types';

// WebSocket URL (change for production)
const WS_URL = 'ws://localhost:8765';

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
    console.log('🔌 Connecting to WebSocket:', WS_URL);

    try {
      this.ws = new WebSocket(WS_URL);

      this.ws.onopen = () => {
        console.log('✅ WebSocket connected');
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

  private handleMessage(message: WSMessage) {
    this.lastUpdate[1](new Date());

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

