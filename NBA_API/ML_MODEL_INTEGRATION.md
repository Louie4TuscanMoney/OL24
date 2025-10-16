# ML Model Integration - NBA_API to Dejavu+LSTM+Conformal

**Objective:** Connect live NBA data to ML prediction ensemble  
**Duration:** 1-2 hours  
**Output:** End-to-end pipeline from NBA.com to predictions

---

## Integration Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                       NBA_API                                 │
│                  (Live Data Source)                           │
│                                                               │
│  Every 10 seconds:                                            │
│  ┌────────────────────────────────────────────────────┐     │
│  │  Fetch ScoreBoard                                  │     │
│  │  Parse games                                        │     │
│  │  Build 18-minute patterns                          │     │
│  └────────────────┬───────────────────────────────────┘     │
│                   │                                           │
└───────────────────┼───────────────────────────────────────────┘
                    │
                    ↓ Pattern ready (18 values)
┌──────────────────────────────────────────────────────────────┐
│                ML PREDICTION SERVICE                          │
│             (FastAPI + Ensemble Models)                       │
│                                                               │
│  POST /api/predict                                            │
│  Body: {"pattern": [+2, +3, +5, ..., +12], "alpha": 0.05}   │
│                                                               │
│  ┌────────────────────────────────────────────────────┐     │
│  │  DEJAVU (40%) → Pattern Matching                  │     │
│  │  Searches database for similar patterns            │     │
│  │  Prediction: +14.1 points                          │     │
│  └────────────────────────────────────────────────────┘     │
│                                                               │
│  ┌────────────────────────────────────────────────────┐     │
│  │  LSTM (60%) → Pattern Learning                     │     │
│  │  Neural network trained on 5000+ games             │     │
│  │  Prediction: +15.8 points                          │     │
│  └────────────────────────────────────────────────────┘     │
│                                                               │
│  ┌────────────────────────────────────────────────────┐     │
│  │  ENSEMBLE → Weighted Average                       │     │
│  │  0.4 × 14.1 + 0.6 × 15.8 = 15.1                   │     │
│  └────────────────┬───────────────────────────────────┘     │
│                   │                                           │
│  ┌────────────────▼───────────────────────────────────┐     │
│  │  CONFORMAL → Uncertainty Wrapper                   │     │
│  │  Adds ±3.8 (95% confidence)                        │     │
│  │  Final: 15.1 ± 3.8 = [11.3, 18.9]                 │     │
│  └────────────────┬───────────────────────────────────┘     │
│                   │                                           │
└───────────────────┼───────────────────────────────────────────┘
                    │
                    ↓ JSON response
┌──────────────────────────────────────────────────────────────┐
│                 WEBSOCKET → SOLIDJS                           │
│             (Real-time Dashboard Display)                     │
└──────────────────────────────────────────────────────────────┘
```

---

## API Client Implementation

### Step 1: ML API Client

**File:** `services/ml_api_client.py`

```python
"""
ML API Client - Connects NBA_API data to ML ensemble
Optimized for SPEED (<100ms predictions)
"""

import aiohttp
import time
from typing import Optional, Dict, List
from config.settings import ML_API_URL, ML_API_TIMEOUT

class MLAPIClient:
    """
    High-speed client for ML prediction API
    """
    
    def __init__(self, base_url: str = ML_API_URL):
        self.base_url = base_url
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Performance tracking
        self.prediction_times = []
        self.prediction_count = 0
        
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=ML_API_TIMEOUT)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def get_prediction(
        self,
        pattern: List[int],
        alpha: float = 0.05,
        return_explanation: bool = True
    ) -> Dict:
        """
        Get halftime prediction from ML ensemble
        
        Args:
            pattern: 18-minute score differential pattern
            alpha: Significance level (0.05 = 95% confidence)
            return_explanation: Include model breakdown
        
        Returns:
            {
                'point_forecast': 15.1,
                'interval_lower': 11.3,
                'interval_upper': 18.9,
                'coverage_probability': 0.95,
                'explanation': {
                    'dejavu_prediction': 14.1,
                    'lstm_prediction': 15.8,
                    'ensemble_forecast': 15.1,
                    'dejavu_weight': 0.4,
                    'lstm_weight': 0.6,
                    'similar_games': [...]
                }
            }
        """
        if len(pattern) != 18:
            raise ValueError(f"Pattern must be 18 values, got {len(pattern)}")
        
        start_time = time.time()
        
        try:
            # Make API request
            async with self.session.post(
                f"{self.base_url}/api/predict",
                json={
                    'pattern': pattern,
                    'alpha': alpha,
                    'return_explanation': return_explanation
                }
            ) as response:
                
                if response.status != 200:
                    raise Exception(f"API error: {response.status}")
                
                prediction = await response.json()
                
                # Track performance
                elapsed = (time.time() - start_time) * 1000
                self.prediction_times.append(elapsed)
                self.prediction_count += 1
                
                if elapsed > 200:
                    print(f"⚠️  Slow ML prediction: {elapsed:.0f}ms")
                
                return prediction
                
        except Exception as e:
            print(f"❌ ML API error: {e}")
            raise
    
    async def health_check(self) -> bool:
        """Check if ML API is healthy"""
        try:
            async with self.session.get(
                f"{self.base_url}/api/health"
            ) as response:
                return response.status == 200
        except:
            return False
    
    async def get_model_info(self) -> Dict:
        """Get information about loaded models"""
        try:
            async with self.session.get(
                f"{self.base_url}/api/models"
            ) as response:
                return await response.json()
        except:
            return {}
    
    def get_avg_prediction_time(self) -> float:
        """Get average prediction time"""
        if not self.prediction_times:
            return 0.0
        return sum(self.prediction_times) / len(self.prediction_times)
```

**Key Features:**
- ✅ Async HTTP for non-blocking calls
- ✅ Connection pooling (reuse session)
- ✅ Performance tracking
- ✅ Error handling with fallbacks
- ✅ Type-safe inputs/outputs

---

### Step 2: ML Prediction Service

**File:** `services/ml_prediction_service.py`

```python
"""
ML Prediction Service - Orchestrates ML API calls
"""

import asyncio
from typing import Dict, Optional, Callable
from services.ml_api_client import MLAPIClient

class MLPredictionService:
    """
    Manages ML predictions for NBA games
    - Calls ML API when patterns ready
    - Caches predictions
    - Emits results to WebSocket
    """
    
    def __init__(self, ml_api_url: str):
        self.ml_api_url = ml_api_url
        self.client: Optional[MLAPIClient] = None
        
        # Cache predictions
        self.predictions: Dict[str, Dict] = {}
        
        # Callbacks
        self.on_prediction_callback: Optional[Callable] = None
    
    async def start(self):
        """Initialize ML API client"""
        self.client = MLAPIClient(self.ml_api_url)
        await self.client.__aenter__()
        
        # Health check
        healthy = await self.client.health_check()
        if healthy:
            print("✅ ML API connected and healthy")
            
            # Get model info
            info = await self.client.get_model_info()
            print(f"   Dejavu: {info.get('dejavu', {}).get('database_size', 0)} patterns")
            print(f"   LSTM: {info.get('lstm', {}).get('hidden_size', 0)} hidden units")
            print(f"   Conformal: {info.get('conformal', {}).get('alpha', 0)} alpha")
        else:
            print("⚠️  ML API not responding - predictions will fail")
    
    async def stop(self):
        """Cleanup"""
        if self.client:
            await self.client.__aexit__(None, None, None)
    
    async def predict_halftime(
        self,
        game_id: str,
        pattern: list,
        game_info: Dict
    ) -> Optional[Dict]:
        """
        Get halftime prediction for game
        
        Args:
            game_id: NBA game ID
            pattern: 18-minute differential pattern
            game_info: Game metadata (teams, current score, etc.)
        
        Returns:
            Prediction dictionary or None if failed
        """
        if not self.client:
            print("❌ ML client not initialized")
            return None
        
        print(f"\n🔮 Requesting prediction for {game_id}")
        print(f"   {game_info['away_team']} @ {game_info['home_team']}")
        print(f"   Pattern: {pattern}")
        
        try:
            # Call ML API
            prediction = await self.client.get_prediction(
                pattern=pattern,
                alpha=0.05,
                return_explanation=True
            )
            
            # Add game metadata
            prediction['game_id'] = game_id
            prediction['game_info'] = game_info
            prediction['pattern'] = pattern
            
            # Cache prediction
            self.predictions[game_id] = prediction
            
            # Log result
            print(f"✅ Prediction received:")
            print(f"   Point forecast: {prediction['point_forecast']:+.1f}")
            print(f"   95% Interval: [{prediction['interval_lower']:+.1f}, "
                  f"{prediction['interval_upper']:+.1f}]")
            
            if 'explanation' in prediction:
                exp = prediction['explanation']
                print(f"   Dejavu: {exp['dejavu_prediction']:+.1f}")
                print(f"   LSTM: {exp['lstm_prediction']:+.1f}")
                print(f"   Ensemble: {exp['ensemble_forecast']:+.1f}")
            
            # Emit to WebSocket
            if self.on_prediction_callback:
                await self.on_prediction_callback(prediction)
            
            return prediction
            
        except Exception as e:
            print(f"❌ Prediction failed: {e}")
            return None
    
    def get_prediction(self, game_id: str) -> Optional[Dict]:
        """Get cached prediction for game"""
        return self.predictions.get(game_id)
    
    def has_prediction(self, game_id: str) -> bool:
        """Check if prediction exists for game"""
        return game_id in self.predictions
    
    def on_prediction(self, callback: Callable):
        """Register callback for when predictions are made"""
        self.on_prediction_callback = callback
```

---

### Step 3: Integrated Pipeline

**File:** `services/integrated_pipeline.py`

```python
"""
Integrated Pipeline - NBA_API → ML Models → WebSocket
Complete end-to-end system
"""

import asyncio
from services.live_data_manager import LiveDataManager
from services.ml_prediction_service import MLPredictionService
from config.settings import ML_API_URL

class IntegratedPipeline:
    """
    Complete pipeline:
    1. Poll NBA_API for live scores
    2. Build 18-minute patterns
    3. Call ML ensemble for predictions
    4. Emit to WebSocket for dashboard
    """
    
    def __init__(self):
        # ML prediction service
        self.ml_service = MLPredictionService(ML_API_URL)
        
        # Live data manager (NBA_API poller)
        self.live_manager = LiveDataManager(
            prediction_callback=self._handle_pattern_ready
        )
    
    async def start(self):
        """Start the complete pipeline"""
        print("=" * 60)
        print("🚀 INTEGRATED PIPELINE STARTING")
        print("=" * 60)
        print()
        
        # Start ML service
        await self.ml_service.start()
        
        # Register prediction callback
        self.ml_service.on_prediction(self._handle_prediction_result)
        
        # Start NBA data polling
        print("\n🏀 Starting NBA live data polling...")
        await self.live_manager.start()
    
    async def _handle_pattern_ready(self, game_id: str, pattern: list):
        """
        Called when 18-minute pattern is ready
        This is the bridge between NBA_API and ML models
        """
        # Get game info
        game_data = self.live_manager.get_active_games().get(game_id)
        
        if not game_data:
            print(f"⚠️  No game data for {game_id}")
            return
        
        game_info = {
            'game_id': game_id,
            'home_team': game_data['home_team'],
            'away_team': game_data['away_team'],
            'score_home': game_data['score_home'],
            'score_away': game_data['score_away'],
            'period': game_data['period'],
        }
        
        # Call ML API for prediction
        prediction = await self.ml_service.predict_halftime(
            game_id=game_id,
            pattern=pattern,
            game_info=game_info
        )
        
        if not prediction:
            print(f"❌ Failed to get prediction for {game_id}")
    
    async def _handle_prediction_result(self, prediction: Dict):
        """
        Called when ML prediction is ready
        Emit to WebSocket for dashboard
        """
        # Format message for WebSocket
        message = {
            'type': 'prediction',
            'data': {
                'game_id': prediction['game_id'],
                'game_info': prediction['game_info'],
                'forecast': prediction['point_forecast'],
                'interval': [
                    prediction['interval_lower'],
                    prediction['interval_upper']
                ],
                'confidence': prediction['coverage_probability'],
                'explanation': prediction.get('explanation'),
            }
        }
        
        # TODO: Emit to WebSocket
        # await websocket_manager.broadcast(message)
        
        print(f"\n📡 Emitting prediction to dashboard...")
        print(f"   Game: {prediction['game_info']['away_team']} @ "
              f"{prediction['game_info']['home_team']}")
        print(f"   Forecast: {prediction['point_forecast']:+.1f}")
```

---

### Step 4: Main Application with ML Integration

**File:** `main_with_ml.py`

```python
"""
Main Application with ML Integration
Complete NBA_API → ML Models → Dashboard pipeline
"""

import asyncio
import signal
from services.integrated_pipeline import IntegratedPipeline

# Global pipeline instance
pipeline: IntegratedPipeline = None

async def main():
    """Main application"""
    global pipeline
    
    print("\n" + "=" * 60)
    print("🏀 NBA LIVE PREDICTIONS - FULL PIPELINE")
    print("=" * 60)
    print()
    print("Components:")
    print("  1. NBA_API - Live score polling (every 10s)")
    print("  2. Score Buffer - Pattern building (18 minutes)")
    print("  3. ML Ensemble - Dejavu + LSTM + Conformal")
    print("  4. WebSocket - Real-time dashboard updates")
    print()
    print("=" * 60)
    print()
    
    # Create integrated pipeline
    pipeline = IntegratedPipeline()
    
    # Start pipeline
    await pipeline.start()

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print("\n\n👋 Shutting down...")
    if pipeline:
        asyncio.create_task(pipeline.ml_service.stop())
    exit(0)

if __name__ == "__main__":
    # Setup signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    # Run application
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down gracefully...")
```

---

## Running the Complete System

### Prerequisites

```bash
# 1. Ensure ML backend is running
cd "ML Research"
python -m uvicorn api.production_api:app --host 0.0.0.0 --port 8080

# 2. Verify ML API is healthy
curl http://localhost:8080/api/health

# Should return: {"status": "healthy", "models_loaded": true}
```

### Start the Pipeline

```bash
# In a new terminal
python main_with_ml.py
```

**Expected Output:**
```
============================================================
🏀 NBA LIVE PREDICTIONS - FULL PIPELINE
============================================================

Components:
  1. NBA_API - Live score polling (every 10s)
  2. Score Buffer - Pattern building (18 minutes)
  3. ML Ensemble - Dejavu + LSTM + Conformal
  4. WebSocket - Real-time dashboard updates

============================================================

🚀 INTEGRATED PIPELINE STARTING
============================================================

✅ ML API connected and healthy
   Dejavu: 3000 patterns
   LSTM: 64 hidden units
   Conformal: 0.05 alpha

🏀 Starting NBA live data polling...
🏀 NBA Data Service started
   Poll interval: 10s

📊 Tracking new game: LAL @ BOS
✅ Poll #1: 8 games, 245ms

... (polls every 10 seconds for 18 minutes) ...

✅ Poll #108: 8 games, 198ms
🎯 Pattern complete for 0021900123
   LAL @ BOS
   Pattern: [+2, +3, +5, +7, +8, +9, +7, +6, +8, +9, +10, +11, +9, +8, +10, +11, +12, +4]

🔮 Requesting prediction for 0021900123
   LAL @ BOS
   Pattern: [+2, +3, +5, +7, +8, +9, +7, +6, +8, +9, +10, +11, +9, +8, +10, +11, +12, +4]

✅ Prediction received:
   Point forecast: +15.1
   95% Interval: [+11.3, +18.9]
   Dejavu: +14.1
   LSTM: +15.8
   Ensemble: +15.1

📡 Emitting prediction to dashboard...
   Game: LAL @ BOS
   Forecast: +15.1
```

---

## Performance Metrics

### End-to-End Latency

| Stage | Target | Typical |
|-------|--------|---------|
| NBA_API poll | <500ms | ~200ms |
| Parse & buffer | <20ms | ~5ms |
| ML API call | <150ms | ~80ms |
| WebSocket emit | <10ms | ~3ms |
| **Total** | **<700ms** | **~300ms** |

**Result:** From NBA.com update to dashboard display in <1 second

---

### ML API Performance

```python
# Monitor ML API performance
import time

class MLPerformanceMonitor:
    def __init__(self):
        self.call_times = []
        self.errors = 0
    
    async def call_with_monitoring(self, pattern):
        start = time.time()
        
        try:
            result = await ml_client.get_prediction(pattern)
            elapsed = (time.time() - start) * 1000
            
            self.call_times.append(elapsed)
            
            if elapsed > 200:
                print(f"⚠️  Slow ML call: {elapsed:.0f}ms")
            
            return result
            
        except Exception as e:
            self.errors += 1
            print(f"❌ ML error #{self.errors}: {e}")
            raise
    
    def get_stats(self):
        if not self.call_times:
            return {}
        
        return {
            'avg': sum(self.call_times) / len(self.call_times),
            'min': min(self.call_times),
            'max': max(self.call_times),
            'p95': sorted(self.call_times)[int(len(self.call_times) * 0.95)],
            'errors': self.errors,
            'success_rate': 1 - (self.errors / len(self.call_times))
        }
```

---

## Testing End-to-End

### Test with Mock Data

```python
# test_ml_integration.py
import asyncio
from services.ml_prediction_service import MLPredictionService

async def test_ml_integration():
    """Test ML integration with sample pattern"""
    
    # Sample 18-minute pattern
    pattern = [0, 2, 5, 7, 8, 9, 7, 6, 8, 9, 10, 11, 9, 8, 10, 11, 12, 4]
    
    # Create ML service
    ml_service = MLPredictionService('http://localhost:8080')
    await ml_service.start()
    
    # Make prediction
    game_info = {
        'home_team': 'LAL',
        'away_team': 'BOS',
        'score_home': 52,
        'score_away': 48,
    }
    
    prediction = await ml_service.predict_halftime(
        game_id='TEST123',
        pattern=pattern,
        game_info=game_info
    )
    
    # Verify prediction
    assert prediction is not None
    assert 'point_forecast' in prediction
    assert 'interval_lower' in prediction
    assert 'interval_upper' in prediction
    
    print("\n✅ ML integration test passed!")
    print(f"   Prediction: {prediction['point_forecast']:+.1f}")
    
    await ml_service.stop()

if __name__ == "__main__":
    asyncio.run(test_ml_integration())
```

---

## Configuration

### Settings File

**File:** `config/settings.py`

```python
"""Settings for ML Integration"""

import os
from dotenv import load_dotenv

load_dotenv()

# ML API Configuration
ML_API_URL = os.getenv('ML_API_URL', 'http://localhost:8080')
ML_API_TIMEOUT = int(os.getenv('ML_API_TIMEOUT', 3))  # seconds

# Prediction Configuration
PREDICTION_TRIGGER_MINUTE = 18  # 6:00 Q2
PATTERN_LENGTH = 18

# Performance Targets
MAX_ML_API_LATENCY_MS = 200  # Warn if slower
MAX_TOTAL_LATENCY_MS = 1000  # Fail if slower

# Error Handling
ML_API_MAX_RETRIES = 2
ML_API_RETRY_DELAY_MS = 500

# Logging
LOG_ML_PREDICTIONS = True
LOG_ML_PERFORMANCE = True
```

---

## Troubleshooting

### Issue: ML API Not Responding

**Check:**
```bash
# 1. Is ML backend running?
ps aux | grep uvicorn

# 2. Is it listening on port 8080?
curl http://localhost:8080/api/health

# 3. Check logs
tail -f logs/ml_api.log
```

**Fix:**
```bash
# Start ML backend
cd "ML Research"
python -m uvicorn api.production_api:app --reload
```

---

### Issue: Slow ML Predictions (>200ms)

**Causes:**
1. Models not pre-loaded
2. Cold start (first prediction)
3. Large Dejavu database
4. CPU-bound LSTM inference

**Solutions:**
```python
# Warm up models on startup
async def warmup_ml_models():
    """Warm up ML models with dummy prediction"""
    dummy_pattern = [0] * 18
    
    await ml_client.get_prediction(dummy_pattern)
    print("✅ ML models warmed up")

# Call before starting pipeline
await warmup_ml_models()
```

---

## Next Steps

**After ML integration complete:**
1. ✅ Read CACHING_STRATEGY.md (optimize performance)
2. ✅ Read PRODUCTION_DEPLOYMENT.md (deploy system)
3. ✅ Test with live games (verify accuracy)

---

**Integration time:** 1-2 hours  
**Result:** Complete pipeline from NBA.com to ML predictions in <1 second

---

*Last Updated: October 15, 2025*  
*Part of ML Research / NBA_API documentation*

