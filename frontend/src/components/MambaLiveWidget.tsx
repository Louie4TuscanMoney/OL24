/**
 * MAMBA LIVE VISUALIZATION COMPONENT
 * 
 * Shows real-time play-by-play pattern under each live game
 * Displays countdown to Q2 6:00 trigger
 * Shows Mamba prediction when it fires
 * 
 * Usage:
 *   <MambaLiveWidget gameId="0022500079" />
 */

import { createSignal, onMount, onCleanup, Show, For } from 'solid-js';
import { Line } from 'solid-chartjs';

interface MambaLiveWidgetProps {
  gameId: string;
}

interface GameState {
  game_id: string;
  total_events: number;
  current_period: number;
  current_clock: string;
  pattern_data: { margin: number; time: number }[];
  mamba_prediction?: {
    prediction: number;
    confidence: number;
    triggered_at: string;
  };
  countdown_to_trigger: {
    message: string;
    seconds?: number;
    status: string;
  };
  recent_events: any[];
}

export function MambaLiveWidget(props: MambaLiveWidgetProps) {
  const [gameState, setGameState] = createSignal<GameState | null>(null);
  const [connected, setConnected] = createSignal(false);
  const [ws, setWs] = createSignal<WebSocket | null>(null);

  // Connect to WebSocket on mount
  onMount(() => {
    connectWebSocket();
  });

  // Cleanup on unmount
  onCleanup(() => {
    const socket = ws();
    if (socket) {
      socket.close();
    }
  });

  function connectWebSocket() {
    const wsUrl = `wss://ol24-production.up.railway.app/ws/mamba/${props.gameId}`;
    const socket = new WebSocket(wsUrl);

    socket.onopen = () => {
      console.log('🟢 Mamba WebSocket connected');
      setConnected(true);
    };

    socket.onmessage = (event) => {
      const message = JSON.parse(event.data);
      
      if (message.type === 'initial_state' || message.type === 'game_update') {
        setGameState(message.data);
      } else if (message.type === 'mamba_trigger') {
        console.log('⚡', message.message);
        // Show animation/notification
      } else if (message.type === 'mamba_prediction') {
        // Update with prediction
        setGameState((prev) => prev ? {
          ...prev,
          mamba_prediction: message.data
        } : null);
      }
    };

    socket.onerror = (error) => {
      console.error('❌ WebSocket error:', error);
      setConnected(false);
    };

    socket.onclose = () => {
      console.log('🔴 WebSocket disconnected');
      setConnected(false);
      
      // Reconnect after 5 seconds
      setTimeout(() => {
        connectWebSocket();
      }, 5000);
    };

    setWs(socket);
  }

  // Chart data for scoring pattern
  const chartData = () => {
    const state = gameState();
    if (!state || !state.pattern_data.length) {
      return {
        labels: [],
        datasets: []
      };
    }

    return {
      labels: state.pattern_data.map(d => `${Math.floor(d.time / 60)}:${(d.time % 60).toString().padStart(2, '0')}`),
      datasets: [
        {
          label: 'Score Margin (Home - Away)',
          data: state.pattern_data.map(d => d.margin),
          borderColor: 'rgb(59, 130, 246)',
          backgroundColor: 'rgba(59, 130, 246, 0.1)',
          tension: 0.4,
          fill: true
        }
      ]
    };
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: false
      },
      title: {
        display: true,
        text: 'Live Scoring Pattern (Last 18 Minutes)',
        color: '#fff'
      }
    },
    scales: {
      y: {
        title: {
          display: true,
          text: 'Score Differential',
          color: '#fff'
        },
        grid: {
          color: 'rgba(255, 255, 255, 0.1)'
        },
        ticks: {
          color: '#fff'
        }
      },
      x: {
        title: {
          display: true,
          text: 'Game Time',
          color: '#fff'
        },
        grid: {
          color: 'rgba(255, 255, 255, 0.1)'
        },
        ticks: {
          color: '#fff',
          maxRotation: 45,
          minRotation: 45
        }
      }
    }
  };

  // Format countdown display
  const countdownDisplay = () => {
    const state = gameState();
    if (!state) return null;

    const countdown = state.countdown_to_trigger;
    if (!countdown) return null;

    if (countdown.status === 'COUNTDOWN' && countdown.seconds) {
      const mins = Math.floor(countdown.seconds / 60);
      const secs = countdown.seconds % 60;
      return (
        <div class="countdown-active">
          <div class="countdown-timer">
            {mins}:{secs.toString().padStart(2, '0')}
          </div>
          <div class="countdown-label">Until Mamba Trigger</div>
        </div>
      );
    } else if (countdown.status === 'TRIGGERED') {
      return (
        <div class="countdown-triggered">
          <div class="pulse-animation">⚡</div>
          <div class="countdown-label">MAMBA TRIGGERING NOW!</div>
        </div>
      );
    } else {
      return (
        <div class="countdown-waiting">
          <div class="countdown-label">{countdown.message}</div>
        </div>
      );
    }
  };

  return (
    <div class="mamba-live-widget">
      {/* Connection Status */}
      <div class="connection-status">
        <Show when={connected()} fallback={
          <span class="status-disconnected">🔴 Connecting...</span>
        }>
          <span class="status-connected">🟢 Live</span>
        </Show>
      </div>

      <Show when={gameState()} fallback={
        <div class="loading">Loading Mamba data...</div>
      }>
        {(state) => (
          <>
            {/* Countdown Section */}
            <div class="mamba-countdown">
              {countdownDisplay()}
            </div>

            {/* Scoring Pattern Chart */}
            <div class="mamba-chart">
              <div style="height: 200px;">
                <Line data={chartData()} options={chartOptions} />
              </div>
              <div class="chart-info">
                <span>Events: {state().total_events}</span>
                <span>Q{state().current_period} {state().current_clock}</span>
              </div>
            </div>

            {/* Mamba Prediction */}
            <Show when={state().mamba_prediction}>
              {(prediction) => (
                <div class="mamba-prediction">
                  <div class="prediction-header">
                    <span class="prediction-icon">🎯</span>
                    <span class="prediction-title">MAMBA PREDICTION</span>
                  </div>
                  <div class="prediction-value">
                    {prediction().prediction > 0 ? '+' : ''}{prediction().prediction.toFixed(1)}
                  </div>
                  <div class="prediction-confidence">
                    Confidence: {prediction().confidence.toFixed(0)}%
                  </div>
                  <div class="prediction-time">
                    Triggered: {new Date(prediction().triggered_at).toLocaleTimeString()}
                  </div>
                </div>
              )}
            </Show>

            {/* Recent Events */}
            <div class="recent-events">
              <div class="events-header">Recent Scoring Events</div>
              <div class="events-list">
                <For each={state().recent_events.slice(0, 5)}>
                  {(event) => (
                    <div class="event-item">
                      <span class="event-time">Q{event.period} {event.clock}</span>
                      <span class="event-score">{event.home_score}-{event.away_score}</span>
                      <span class="event-margin" classList={{
                        'positive': event.margin > 0,
                        'negative': event.margin < 0,
                        'tied': event.margin === 0
                      }}>
                        {event.margin > 0 ? '+' : ''}{event.margin}
                      </span>
                    </div>
                  )}
                </For>
              </div>
            </div>
          </>
        )}
      </Show>

      <style>{`
        .mamba-live-widget {
          background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 100%);
          border-radius: 12px;
          padding: 20px;
          margin: 16px 0;
          color: white;
          box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        }

        .connection-status {
          text-align: right;
          margin-bottom: 10px;
          font-size: 12px;
        }

        .status-connected {
          color: #10b981;
        }

        .status-disconnected {
          color: #ef4444;
        }

        .mamba-countdown {
          background: rgba(0, 0, 0, 0.3);
          border-radius: 8px;
          padding: 16px;
          margin-bottom: 16px;
          text-align: center;
        }

        .countdown-active .countdown-timer {
          font-size: 48px;
          font-weight: bold;
          color: #fbbf24;
          font-family: 'Courier New', monospace;
        }

        .countdown-active .countdown-label {
          margin-top: 8px;
          font-size: 14px;
          color: #d1d5db;
        }

        .countdown-triggered {
          animation: pulse 1s infinite;
        }

        .countdown-triggered .pulse-animation {
          font-size: 64px;
        }

        @keyframes pulse {
          0%, 100% { opacity: 1; transform: scale(1); }
          50% { opacity: 0.7; transform: scale(1.1); }
        }

        .countdown-triggered .countdown-label {
          font-size: 18px;
          font-weight: bold;
          color: #fbbf24;
        }

        .countdown-waiting .countdown-label {
          font-size: 14px;
          color: #9ca3af;
        }

        .mamba-chart {
          background: rgba(0, 0, 0, 0.2);
          border-radius: 8px;
          padding: 16px;
          margin-bottom: 16px;
        }

        .chart-info {
          display: flex;
          justify-content: space-between;
          margin-top: 8px;
          font-size: 12px;
          color: #d1d5db;
        }

        .mamba-prediction {
          background: linear-gradient(135deg, #10b981 0%, #059669 100%);
          border-radius: 8px;
          padding: 20px;
          margin-bottom: 16px;
          text-align: center;
          animation: fadeIn 0.5s ease-in;
        }

        @keyframes fadeIn {
          from { opacity: 0; transform: translateY(-10px); }
          to { opacity: 1; transform: translateY(0); }
        }

        .prediction-header {
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 8px;
          margin-bottom: 12px;
        }

        .prediction-icon {
          font-size: 24px;
        }

        .prediction-title {
          font-size: 16px;
          font-weight: bold;
        }

        .prediction-value {
          font-size: 48px;
          font-weight: bold;
          margin: 8px 0;
        }

        .prediction-confidence {
          font-size: 14px;
          opacity: 0.9;
        }

        .prediction-time {
          font-size: 12px;
          opacity: 0.7;
          margin-top: 4px;
        }

        .recent-events {
          background: rgba(0, 0, 0, 0.2);
          border-radius: 8px;
          padding: 16px;
        }

        .events-header {
          font-size: 14px;
          font-weight: bold;
          margin-bottom: 12px;
          opacity: 0.9;
        }

        .events-list {
          display: flex;
          flex-direction: column;
          gap: 8px;
        }

        .event-item {
          display: flex;
          justify-content: space-between;
          padding: 8px 12px;
          background: rgba(255, 255, 255, 0.05);
          border-radius: 6px;
          font-size: 13px;
        }

        .event-time {
          color: #9ca3af;
          min-width: 80px;
        }

        .event-score {
          font-weight: bold;
          min-width: 60px;
          text-align: center;
        }

        .event-margin {
          min-width: 50px;
          text-align: right;
          font-weight: bold;
        }

        .event-margin.positive {
          color: #10b981;
        }

        .event-margin.negative {
          color: #ef4444;
        }

        .event-margin.tied {
          color: #fbbf24;
        }

        .loading {
          text-align: center;
          padding: 40px;
          color: #9ca3af;
        }
      `}</style>
    </div>
  );
}

