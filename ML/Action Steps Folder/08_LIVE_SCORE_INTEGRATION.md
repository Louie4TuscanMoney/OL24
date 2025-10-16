# Step 8: Live Score Integration (5-Second Scraper)

**Objective:** Integrate forecasting models with live NBA game score scraper

**Duration:** 3-4 hours  
**Prerequisites:** Completed Step 7 (production API), existing 5-second score scraper  
**Output:** Real-time halftime prediction system updating every 5 seconds

---

## System Architecture

```
Live Game (NBA.com / ESPN) 
    ↓ (every 5 seconds)
Your Score Scraper
    ↓
Score Buffer (accumulate minute-by-minute)
    ↓
Trigger at 6:00 2Q
    ↓
Ensemble API (Dejavu + LSTM + Conformal)
    ↓
Halftime Prediction + Interval
    ↓
Dashboard / Betting System / Analytics
```

---

## Action Items

### 8.1 Create Live Score Buffer (1 hour)

**File:** `live/score_buffer.py`

```python
"""
Buffer live scores and convert to minute-by-minute differentials
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import deque
from typing import Optional, List

class LiveScoreBuffer:
    """
    Accumulate live scores and create minute-by-minute differential pattern
    """
    def __init__(self, game_id: str, home_team: str, away_team: str):
        """
        Args:
            game_id: Unique game identifier
            home_team: Home team code
            away_team: Away team code
        """
        self.game_id = game_id
        self.home_team = home_team
        self.away_team = away_team
        
        # Buffer for score updates (5-second granularity)
        self.score_updates = []
        
        # Minute-by-minute aggregated scores
        self.minute_differentials = []
        
        self.game_start_time = None
        self.current_quarter = 1
    
    def add_score_update(
        self,
        timestamp: datetime,
        quarter: int,
        time_remaining: str,  # "10:45"
        score_home: int,
        score_away: int
    ):
        """
        Add a score update from live scraper
        
        Called every 5 seconds by your scraper
        """
        self.score_updates.append({
            'timestamp': timestamp,
            'quarter': quarter,
            'time_remaining': time_remaining,
            'score_home': score_home,
            'score_away': score_away,
            'differential': score_home - score_away
        })
        
        self.current_quarter = quarter
        
        # Update minute-by-minute aggregation
        self._update_minute_aggregates()
    
    def _update_minute_aggregates(self):
        """
        Aggregate 5-second updates to minute-by-minute
        """
        # Group updates by minute
        minute_groups = {}
        
        for update in self.score_updates:
            quarter = update['quarter']
            time_parts = update['time_remaining'].split(':')
            minute_in_quarter = int(time_parts[0]) + 1  # Round up
            
            # Game minute (0-47)
            game_minute = (quarter - 1) * 12 + (12 - minute_in_quarter)
            
            if game_minute not in minute_groups:
                minute_groups[game_minute] = []
            
            minute_groups[game_minute].append(update)
        
        # Take last score in each minute
        self.minute_differentials = []
        for minute in sorted(minute_groups.keys()):
            last_update = minute_groups[minute][-1]
            self.minute_differentials.append({
                'minute': minute,
                'differential': last_update['differential'],
                'quarter': last_update['quarter']
            })
    
    def get_pattern_at_6min_2q(self) -> Optional[np.ndarray]:
        """
        Get 18-minute differential pattern (start to 6:00 2Q)
        
        Returns:
            Pattern array or None if not ready
        """
        # 6:00 2Q = minute 18 (start of minute 19)
        if len(self.minute_differentials) < 18:
            return None  # Not enough data yet
        
        # Extract first 18 minutes
        pattern = [d['differential'] for d in self.minute_differentials[:18]]
        
        return np.array(pattern)
    
    def is_ready_for_prediction(self) -> bool:
        """
        Check if we're at or past 6:00 2Q and ready to predict
        """
        return (self.current_quarter >= 2 and 
                len(self.minute_differentials) >= 18)
    
    def get_current_differential(self) -> Optional[float]:
        """Get most recent differential"""
        if self.score_updates:
            return self.score_updates[-1]['differential']
        return None


if __name__ == "__main__":
    # Test buffer
    buffer = LiveScoreBuffer("TEST_GAME", "LAL", "LAC")
    
    # Simulate 5-second updates for first 20 minutes
    from datetime import datetime
    base_time = datetime.now()
    
    # Simulate game progression
    for second in range(0, 1200, 5):  # 20 minutes = 1200 seconds
        minute = second // 60
        time_in_quarter = 12 - (minute % 12)
        quarter = (minute // 12) + 1
        
        # Simulate score (simple random walk)
        score_home = second // 3 + np.random.randint(-2, 3)
        score_away = second // 3 + np.random.randint(-2, 3)
        
        buffer.add_score_update(
            timestamp=base_time + timedelta(seconds=second),
            quarter=quarter,
            time_remaining=f"{time_in_quarter}:{(60 - second % 60):02d}",
            score_home=score_home,
            score_away=score_away
        )
    
    # Check if ready
    print(f"Ready for prediction: {buffer.is_ready_for_prediction()}")
    print(f"Pattern length: {len(buffer.minute_differentials)}")
    
    if buffer.is_ready_for_prediction():
        pattern = buffer.get_pattern_at_6min_2q()
        print(f"Pattern: {pattern}")
```

---

### 8.2 Create Live Prediction Service (1 hour)

**File:** `live/live_prediction_service.py`

```python
"""
Live prediction service that monitors games and triggers forecasts
"""

import asyncio
from datetime import datetime
import numpy as np
from live.score_buffer import LiveScoreBuffer
from clients.nba_forecast_client import NBAForecastClient
from typing import Dict, Optional

class LivePredictionService:
    """
    Monitor live games and generate real-time halftime predictions
    """
    def __init__(self, api_url='http://localhost:8080'):
        """
        Args:
            api_url: URL of forecasting API
        """
        self.api_client = NBAForecastClient(api_url)
        self.active_games = {}  # {game_id: LiveScoreBuffer}
        self.predictions = {}   # {game_id: prediction_result}
    
    def register_game(self, game_id: str, home_team: str, away_team: str):
        """
        Start monitoring a new game
        """
        self.active_games[game_id] = LiveScoreBuffer(game_id, home_team, away_team)
        print(f"✓ Monitoring game: {game_id} ({away_team} @ {home_team})")
    
    def process_score_update(
        self,
        game_id: str,
        quarter: int,
        time_remaining: str,
        score_home: int,
        score_away: int
    ):
        """
        Process score update from your 5-second scraper
        
        This function is called by your existing scraper every 5 seconds
        """
        if game_id not in self.active_games:
            print(f"Warning: Unknown game {game_id}")
            return
        
        buffer = self.active_games[game_id]
        
        # Add score update
        buffer.add_score_update(
            timestamp=datetime.now(),
            quarter=quarter,
            time_remaining=time_remaining,
            score_home=score_home,
            score_away=score_away
        )
        
        # Check if ready to predict (at 6:00 2Q)
        if buffer.is_ready_for_prediction() and game_id not in self.predictions:
            self._make_prediction(game_id, buffer)
    
    def _make_prediction(self, game_id: str, buffer: LiveScoreBuffer):
        """
        Make halftime prediction when 6:00 2Q reached
        """
        print(f"\n{'='*80}")
        print(f"HALFTIME PREDICTION TRIGGER: {game_id}")
        print(f"{'='*80}")
        
        # Get 18-minute pattern
        pattern = buffer.get_pattern_at_6min_2q()
        
        print(f"Pattern at 6:00 2Q: {pattern}")
        print(f"Current differential: {buffer.get_current_differential():.1f}")
        
        # Call prediction API
        result = self.api_client.predict(
            pattern=pattern.tolist(),
            alpha=0.05,
            return_explanation=True
        )
        
        # Store prediction
        self.predictions[game_id] = {
            'timestamp': datetime.now(),
            'pattern': pattern.tolist(),
            'forecast': result['point_forecast'],
            'interval': (result['interval_lower'], result['interval_upper']),
            'explanation': result['explanation']
        }
        
        # Display prediction
        print(f"\n🏀 HALFTIME PREDICTION:")
        print(f"   Point Forecast: {result['point_forecast']:+.1f} points")
        print(f"   95% Interval:   [{result['interval_lower']:+.1f}, {result['interval_upper']:+.1f}]")
        
        if result['explanation']:
            print(f"\n   Model Breakdown:")
            print(f"     Dejavu:   {result['explanation']['dejavu_prediction']:+.1f}")
            print(f"     LSTM:     {result['explanation']['lstm_prediction']:+.1f}")
            print(f"     Ensemble: {result['explanation']['ensemble_forecast']:+.1f}")
            
            # Show similar games
            if 'similar_games' in result['explanation']:
                print(f"\n   Similar Historical Games:")
                for i, game in enumerate(result['explanation']['similar_games'][:3], 1):
                    print(f"     {i}. {game['game_id']} - diff: {game['halftime_differential']:+.1f}")
        
        print(f"{'='*80}\n")
        
        # Trigger callbacks (webhook, database, dashboard update, etc.)
        self._on_prediction_made(game_id, result)
    
    def _on_prediction_made(self, game_id: str, result: Dict):
        """
        Callback when prediction is made
        Override this to integrate with your systems
        """
        # Examples:
        # - Send webhook notification
        # - Update dashboard
        # - Log to database
        # - Send to betting system
        # - Alert stakeholders
        pass
    
    def get_prediction(self, game_id: str) -> Optional[Dict]:
        """Get prediction for game if available"""
        return self.predictions.get(game_id)


if __name__ == "__main__":
    # Example usage
    service = LivePredictionService(api_url='http://localhost:8080')
    
    # Register active games
    service.register_game("202510150LAL", "LAL", "GSW")
    service.register_game("202510150BOS", "BOS", "MIA")
    
    # Simulate score updates (in production, called by your scraper)
    for minute in range(20):
        service.process_score_update(
            game_id="202510150LAL",
            quarter=1 if minute < 12 else 2,
            time_remaining=f"{12 - (minute % 12)}:00",
            score_home=minute * 2 + np.random.randint(-3, 4),
            score_away=minute * 2 + np.random.randint(-3, 4)
        )
        
        # Print progress
        if minute == 17:
            print(f"\n6:00 2Q reached - prediction will be made")
```

---

### 8.3 Integration with Your Existing 5-Second Scraper (1 hour)

**File:** `integrations/scraper_integration.py`

```python
"""
Integration adapter for your existing 5-second score scraper
"""

from live.live_prediction_service import LivePredictionService
import json

class ScraperIntegration:
    """
    Adapter to connect your existing scraper to prediction service
    """
    def __init__(self, prediction_service: LivePredictionService):
        self.prediction_service = prediction_service
        self.active_games = set()
    
    def on_game_start(self, game_data: Dict):
        """
        Called when your scraper detects a new game starting
        
        Args:
            game_data: {
                'game_id': str,
                'home_team': str,
                'away_team': str,
                'start_time': datetime
            }
        """
        game_id = game_data['game_id']
        
        if game_id not in self.active_games:
            self.prediction_service.register_game(
                game_id,
                game_data['home_team'],
                game_data['away_team']
            )
            self.active_games.add(game_id)
            
            print(f"🏀 New game monitoring: {game_data['away_team']} @ {game_data['home_team']}")
    
    def on_score_update(self, score_data: Dict):
        """
        Called every 5 seconds by your scraper with current score
        
        Args:
            score_data: {
                'game_id': str,
                'quarter': int,
                'time_remaining': str,  # "10:45"
                'score_home': int,
                'score_away': int,
                'timestamp': datetime
            }
        """
        self.prediction_service.process_score_update(
            game_id=score_data['game_id'],
            quarter=score_data['quarter'],
            time_remaining=score_data['time_remaining'],
            score_home=score_data['score_home'],
            score_away=score_data['score_away']
        )
    
    def on_game_end(self, game_id: str):
        """
        Called when game finishes
        """
        if game_id in self.active_games:
            # Get final result
            buffer = self.prediction_service.active_games.get(game_id)
            prediction = self.prediction_service.predictions.get(game_id)
            
            if buffer and prediction:
                # Log for model improvement
                actual_halftime = buffer.minute_differentials[23]['differential']  # Minute 23 = 0:00 2Q
                forecast = prediction['forecast']
                error = abs(actual_halftime - forecast)
                
                print(f"\n📊 Game {game_id} complete:")
                print(f"   Predicted: {forecast:+.1f}")
                print(f"   Actual:    {actual_halftime:+.1f}")
                print(f"   Error:     {error:.1f} points")
                
                # Store for retraining/recalibration
                self._log_result(game_id, prediction, actual_halftime)
            
            self.active_games.remove(game_id)


# Integration point with YOUR existing scraper
def integrate_with_your_scraper(your_scraper_instance):
    """
    Connect your existing 5-second scraper to prediction service
    
    Modify this based on your scraper's architecture
    """
    # Create prediction service
    prediction_service = LivePredictionService(api_url='http://localhost:8080')
    
    # Create adapter
    adapter = ScraperIntegration(prediction_service)
    
    # Register callbacks with your scraper
    # (Adjust these based on your scraper's callback system)
    
    your_scraper_instance.on_game_start_callback = adapter.on_game_start
    your_scraper_instance.on_score_update_callback = adapter.on_score_update
    your_scraper_instance.on_game_end_callback = adapter.on_game_end
    
    print("✓ Prediction service integrated with scraper")
    
    return adapter
```

---

### 8.4 WebSocket API for Real-Time Updates (1 hour)

**File:** `api/websocket_api.py`

```python
"""
WebSocket API for real-time prediction updates
"""

from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
import asyncio
import json
from typing import Dict, Set

app = FastAPI(title="NBA Live Predictions WebSocket")

# Active WebSocket connections
active_connections: Set[WebSocket] = set()

# Prediction cache
live_predictions: Dict[str, Dict] = {}

@app.websocket("/ws/live")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket for real-time prediction updates
    """
    await websocket.accept()
    active_connections.add(websocket)
    
    try:
        while True:
            # Send any new predictions
            if live_predictions:
                await websocket.send_json({
                    'type': 'predictions',
                    'data': live_predictions
                })
            
            await asyncio.sleep(1)  # Update every second
    
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        active_connections.remove(websocket)

@app.post("/update_prediction")
async def update_prediction(game_id: str, prediction: Dict):
    """
    Called by live prediction service when new forecast generated
    """
    live_predictions[game_id] = {
        'game_id': game_id,
        'timestamp': datetime.now().isoformat(),
        **prediction
    }
    
    # Broadcast to all connected clients
    await broadcast_update(game_id, prediction)
    
    return {"status": "updated"}

async def broadcast_update(game_id: str, prediction: Dict):
    """Broadcast to all WebSocket clients"""
    message = {
        'type': 'new_prediction',
        'game_id': game_id,
        'prediction': prediction
    }
    
    disconnected = set()
    for connection in active_connections:
        try:
            await connection.send_json(message)
        except:
            disconnected.add(connection)
    
    # Remove disconnected clients
    active_connections.difference_update(disconnected)

# Simple frontend to test
@app.get("/")
async def get():
    html = """
    <!DOCTYPE html>
    <html>
        <head><title>Live NBA Predictions</title></head>
        <body>
            <h1>🏀 Live NBA Halftime Predictions</h1>
            <div id="predictions"></div>
            <script>
                const ws = new WebSocket("ws://localhost:8080/ws/live");
                
                ws.onmessage = function(event) {
                    const data = JSON.parse(event.data);
                    document.getElementById('predictions').innerHTML = 
                        JSON.stringify(data, null, 2);
                };
            </script>
        </body>
    </html>
    """
    return HTMLResponse(html)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
```

---

### 8.5 Complete Integration Example (30 minutes)

**File:** `examples/live_integration_example.py`

```python
"""
Complete example: Your 5-second scraper → Our prediction system
"""

import time
import numpy as np
from datetime import datetime
from live.score_buffer import LiveScoreBuffer
from clients.nba_forecast_client import NBAForecastClient

def simulate_live_game_with_predictions():
    """
    Simulate a live game with real-time predictions
    """
    print("=" * 80)
    print("LIVE GAME SIMULATION WITH PREDICTIONS")
    print("=" * 80)
    
    # Create score buffer
    buffer = LiveScoreBuffer("SIM_GAME_001", "LAL", "GSW")
    
    # Create API client
    client = NBAForecastClient(base_url='http://localhost:8080')
    
    # Simulate game progression (fast-forward)
    print("\nSimulating live game score updates every 5 seconds...")
    print("(Fast-forwarding for demo - in production, updates come from your scraper)\n")
    
    score_home = 0
    score_away = 0
    
    for second in range(0, 1500, 5):  # 25 minutes (past 6:00 2Q)
        # Calculate game clock
        total_seconds_elapsed = second
        minute = total_seconds_elapsed // 60
        second_in_minute = total_seconds_elapsed % 60
        
        quarter = (minute // 12) + 1
        minute_in_quarter = minute % 12
        time_remaining_quarter = 12 - minute_in_quarter - 1
        second_remaining = 60 - second_in_minute
        
        # Simulate score progression (random walk)
        score_home += np.random.randint(0, 2)
        score_away += np.random.randint(0, 2)
        
        # Add to buffer
        buffer.add_score_update(
            timestamp=datetime.now(),
            quarter=quarter,
            time_remaining=f"{time_remaining_quarter}:{second_remaining:02d}",
            score_home=score_home,
            score_away=score_away
        )
        
        # Check if we've reached 6:00 2Q (minute 18)
        if buffer.is_ready_for_prediction() and minute == 18:
            print(f"\n⏰ 6:00 2Q REACHED!")
            print(f"   Current Score: {score_away}-{score_home}")
            print(f"   Current Differential: {score_home - score_away:+d}")
            
            # Get pattern and predict
            pattern = buffer.get_pattern_at_6min_2q()
            
            print(f"\n🔮 Generating halftime prediction...")
            result = client.predict(pattern=pattern.tolist(), alpha=0.05)
            
            print(f"\n📊 HALFTIME FORECAST:")
            print(f"   Point Prediction: {result['point_forecast']:+.1f} points")
            print(f"   95% Interval:     [{result['interval_lower']:+.1f}, {result['interval_upper']:+.1f}]")
            
            if result.get('explanation'):
                exp = result['explanation']
                print(f"\n   Component Models:")
                print(f"     Dejavu (40%): {exp['dejavu_prediction']:+.1f}")
                print(f"     LSTM (60%):   {exp['lstm_prediction']:+.1f}")
            
            break  # Stop simulation after prediction made
    
    print("\n" + "=" * 80)
    print("SIMULATION COMPLETE")
    print("=" * 80)
    print("\nIn production, this runs continuously for all live games.")
    print("Your 5-second scraper calls process_score_update() with each update.")

if __name__ == "__main__":
    simulate_live_game_with_predictions()
```

---

### 8.6 Integration Checklist

- [ ] ✅ Score buffer accumulating 5-second updates
- [ ] ✅ Minute-by-minute aggregation working
- [ ] ✅ Prediction triggered at 6:00 2Q
- [ ] ✅ API responding within 100ms
- [ ] ✅ Results broadcast via WebSocket
- [ ] ✅ Integration tested with simulated game

**Integration Points with Your Scraper:**
```python
# In your existing 5-second scraper, add:

from integrations.scraper_integration import ScraperIntegration
from live.live_prediction_service import LivePredictionService

# Initialize once
prediction_service = LivePredictionService(api_url='http://localhost:8080')
adapter = ScraperIntegration(prediction_service)

# In your scraper's main loop (every 5 seconds):
def your_scraper_callback(score_data):
    # Your existing code to get scores...
    
    # Add this line to trigger predictions:
    adapter.on_score_update(score_data)
```

---

## Expected Outputs

```
live/
├── score_buffer.py                 ← Accumulates 5-sec updates
├── live_prediction_service.py      ← Monitors games, triggers predictions
└── prediction_log.json             ← Historical predictions

integrations/
└── scraper_integration.py          ← Adapter for your scraper

api/
├── production_api.py                ← REST API
└── websocket_api.py                 ← Real-time updates
```

---

## Monitoring Dashboard

**Real-time display:**
- Current games being monitored
- Predictions made (with timestamps)
- Accuracy tracking (vs. actual halftime scores)
- API response times
- System health

---

## Next Step

Proceed to **Step 9: Production Deployment** for full production infrastructure with monitoring, logging, and auto-scaling.

---

*Action Step 8 of 10 - Live Score Integration*

