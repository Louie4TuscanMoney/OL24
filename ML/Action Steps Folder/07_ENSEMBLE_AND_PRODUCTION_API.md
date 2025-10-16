# Step 7: Ensemble Models & Production API

**Objective:** Combine Dejavu + LSTM + Conformal into unified production system

**Duration:** 4-6 hours  
**Prerequisites:** Completed Steps 4-6 (all three models)  
**Output:** Production-ready API with ensemble forecasting and uncertainty

---

## Action Items

### 7.1 Create Ensemble Forecaster (1 hour)

**File:** `models/ensemble_forecaster.py`

```python
"""
Ensemble forecaster combining Dejavu and LSTM
"""

import numpy as np
import torch
import joblib
from models.dejavu_forecaster import DejavuNBAForecaster
from models.lstm_forecaster import LSTMForecaster
from models.conformal_wrapper import AdaptiveConformalNBA

class EnsembleNBAForecaster:
    """
    Ensemble of Dejavu + LSTM with Conformal wrapper
    """
    def __init__(
        self,
        dejavu_weight=0.4,
        lstm_weight=0.6
    ):
        """
        Args:
            dejavu_weight: Weight for Dejavu predictions
            lstm_weight: Weight for LSTM predictions
        """
        assert abs(dejavu_weight + lstm_weight - 1.0) < 1e-6
        
        self.dejavu_weight = dejavu_weight
        self.lstm_weight = lstm_weight
        
        # Models (loaded separately)
        self.dejavu_model = None
        self.lstm_model = None
        self.lstm_normalization = None
        self.conformal_model = None
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def load_models(
        self,
        dejavu_path='models/dejavu_forecaster.pkl',
        lstm_path='models/lstm_best.pth',
        lstm_norm_path='models/lstm_normalization.pkl',
        conformal_path='models/conformal_predictor.pkl'
    ):
        """Load all models"""
        print("Loading ensemble models...")
        
        # Load Dejavu
        self.dejavu_model = DejavuNBAForecaster.load(dejavu_path)
        print(f"  ✓ Dejavu loaded ({len(self.dejavu_model.database)} patterns)")
        
        # Load LSTM
        self.lstm_model = LSTMForecaster(hidden_size=64, num_layers=2)
        self.lstm_model.load_state_dict(torch.load(lstm_path))
        self.lstm_model = self.lstm_model.to(self.device)
        self.lstm_model.eval()
        self.lstm_normalization = joblib.load(lstm_norm_path)
        print(f"  ✓ LSTM loaded")
        
        # Load Conformal
        self.conformal_model = AdaptiveConformalNBA.load(conformal_path)
        print(f"  ✓ Conformal loaded (quantile: {self.conformal_model.quantile:.2f})")
    
    def predict(self, query_pattern):
        """
        Ensemble prediction with uncertainty
        
        Args:
            query_pattern: 18-minute differential pattern
        
        Returns:
            forecast: Ensemble point prediction
            interval: (lower, upper) conformal interval
            explanation: Dict with component predictions and neighbors
        """
        # Dejavu prediction
        dejavu_pred, neighbors = self.dejavu_model.predict(query_pattern)
        
        # LSTM prediction
        # Normalize pattern
        pattern_norm = (query_pattern - self.lstm_normalization['pattern_mean']) / \
                       self.lstm_normalization['pattern_std']
        
        # Predict
        with torch.no_grad():
            pattern_tensor = torch.FloatTensor(pattern_norm).unsqueeze(0).unsqueeze(-1)
            pattern_tensor = pattern_tensor.to(self.device)
            lstm_output = self.lstm_model(pattern_tensor)
            lstm_pred_norm = lstm_output[0, -1].cpu().numpy()  # Halftime prediction
        
        # Denormalize LSTM prediction
        lstm_pred = lstm_pred_norm * self.lstm_normalization['outcome_std'][-1] + \
                    self.lstm_normalization['outcome_mean'][-1]
        
        # Ensemble
        ensemble_forecast = (self.dejavu_weight * dejavu_pred + 
                           self.lstm_weight * lstm_pred)
        
        # Conformal interval (use ensemble as point forecast)
        lower = ensemble_forecast - self.conformal_model.quantile
        upper = ensemble_forecast + self.conformal_model.quantile
        
        explanation = {
            'dejavu_prediction': float(dejavu_pred),
            'lstm_prediction': float(lstm_pred),
            'ensemble_forecast': float(ensemble_forecast),
            'dejavu_weight': self.dejavu_weight,
            'lstm_weight': self.lstm_weight,
            'similar_games': neighbors[:5] if neighbors else []
        }
        
        return ensemble_forecast, (lower, upper), explanation
    
    def save(self, filepath):
        """Save ensemble configuration"""
        config = {
            'dejavu_weight': self.dejavu_weight,
            'lstm_weight': self.lstm_weight
        }
        joblib.dump(config, filepath)


if __name__ == "__main__":
    # Create and load ensemble
    ensemble = EnsembleNBAForecaster(dejavu_weight=0.4, lstm_weight=0.6)
    ensemble.load_models()
    
    # Test prediction
    test_df = pd.read_parquet('data/splits/test.parquet')
    test_pattern = test_df['pattern'].iloc[0]
    
    forecast, interval, explanation = ensemble.predict(test_pattern)
    
    print(f"\nEnsemble Prediction:")
    print(f"  Dejavu component:  {explanation['dejavu_prediction']:.2f}")
    print(f"  LSTM component:    {explanation['lstm_prediction']:.2f}")
    print(f"  Ensemble forecast: {explanation['ensemble_forecast']:.2f}")
    print(f"  95% Interval:      [{interval[0]:.2f}, {interval[1]:.2f}]")
    
    # Save ensemble
    ensemble.save('models/ensemble_config.pkl')
```

---

### 7.2 Production API with All Models (1 hour)

**File:** `api/production_api.py`

```python
"""
Production API: Dejavu + LSTM Ensemble + Conformal
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import numpy as np
from models.ensemble_forecaster import EnsembleNBAForecaster

app = FastAPI(
    title="NBA Halftime Prediction API",
    version="1.0.0",
    description="Ensemble forecasting with uncertainty quantification"
)

# Global ensemble model
ensemble = None

class PredictionRequest(BaseModel):
    pattern: List[float]  # 18-minute differential pattern
    alpha: float = 0.05   # 95% coverage
    return_explanation: bool = True

class PredictionResponse(BaseModel):
    point_forecast: float
    interval_lower: float
    interval_upper: float
    coverage_probability: float
    explanation: Optional[Dict] = None

@app.on_event("startup")
async def load_models():
    """Load all models at startup"""
    global ensemble
    
    print("Starting NBA Halftime Prediction API...")
    ensemble = EnsembleNBAForecaster(dejavu_weight=0.4, lstm_weight=0.6)
    ensemble.load_models()
    print("✓ All models loaded and ready")

@app.post("/predict", response_model=PredictionResponse)
async def predict_halftime_ensemble(request: PredictionRequest):
    """
    Predict NBA halftime differential with uncertainty
    
    Input: 18-minute score differential pattern (start to 6:00 2Q)
    Output: Halftime prediction with 95% interval + explanation
    """
    if ensemble is None:
        raise HTTPException(status_code=500, detail="Models not loaded")
    
    # Validate input
    if len(request.pattern) != 18:
        raise HTTPException(
            status_code=400,
            detail=f"Pattern must be 18 minutes, got {len(request.pattern)}"
        )
    
    # Convert to numpy
    query_pattern = np.array(request.pattern)
    
    # Predict
    forecast, interval, explanation = ensemble.predict(query_pattern)
    lower, upper = interval
    
    # Build response
    response = PredictionResponse(
        point_forecast=float(forecast),
        interval_lower=float(lower),
        interval_upper=float(upper),
        coverage_probability=1 - request.alpha,
        explanation=explanation if request.return_explanation else None
    )
    
    return response

@app.get("/health")
async def health_check():
    """Health check"""
    return {
        "status": "healthy",
        "models_loaded": ensemble is not None,
        "dejavu_database_size": len(ensemble.dejavu_model.database) if ensemble else 0
    }

@app.get("/models")
async def model_info():
    """Model information"""
    if ensemble is None:
        raise HTTPException(status_code=500, detail="Models not loaded")
    
    return {
        "ensemble": {
            "dejavu_weight": ensemble.dejavu_weight,
            "lstm_weight": ensemble.lstm_weight
        },
        "dejavu": {
            "database_size": len(ensemble.dejavu_model.database),
            "K": ensemble.dejavu_model.K,
            "similarity_method": ensemble.dejavu_model.similarity_method
        },
        "conformal": {
            "alpha": ensemble.conformal_model.alpha,
            "quantile": float(ensemble.conformal_model.quantile),
            "calibration_samples": len(ensemble.conformal_model.scores)
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080, workers=4)
```

---

### 7.3 Create Client Library (30 minutes)

**File:** `clients/nba_forecast_client.py`

```python
"""
Python client for NBA Halftime Prediction API
"""

import requests
from typing import List, Tuple, Dict

class NBAForecastClient:
    """
    Client for NBA halftime prediction API
    """
    def __init__(self, base_url='http://localhost:8080'):
        self.base_url = base_url
    
    def predict(
        self,
        pattern: List[float],
        alpha: float = 0.05,
        return_explanation: bool = True
    ) -> Dict:
        """
        Predict halftime differential
        
        Args:
            pattern: 18-minute score differential pattern
            alpha: Significance level (0.05 = 95% coverage)
            return_explanation: Include model breakdown
        
        Returns:
            Full prediction response
        """
        response = requests.post(
            f"{self.base_url}/predict",
            json={
                'pattern': pattern,
                'alpha': alpha,
                'return_explanation': return_explanation
            }
        )
        
        if response.status_code != 200:
            raise Exception(f"API error: {response.status_code} - {response.text}")
        
        return response.json()
    
    def health_check(self) -> Dict:
        """Check API health"""
        response = requests.get(f"{self.base_url}/health")
        return response.json()


if __name__ == "__main__":
    # Example usage
    client = NBAForecastClient()
    
    # Check health
    health = client.health_check()
    print(f"API Status: {health['status']}")
    
    # Example prediction
    example_pattern = [0, 2, 5, 8, 10, 12, 13, 15, 14, 15, 16, 18, 17, 15, 14, 13, 12, 15]
    
    result = client.predict(example_pattern, alpha=0.05)
    
    print(f"\nPrediction for pattern: {example_pattern}")
    print(f"  Point forecast: {result['point_forecast']:.1f} points")
    print(f"  95% Interval: [{result['interval_lower']:.1f}, {result['interval_upper']:.1f}]")
    
    if result['explanation']:
        print(f"\n  Model Breakdown:")
        print(f"    Dejavu: {result['explanation']['dejavu_prediction']:.1f}")
        print(f"    LSTM:   {result['explanation']['lstm_prediction']:.1f}")
        print(f"    Ensemble: {result['explanation']['ensemble_forecast']:.1f}")
```

---

### 7.4 Docker Deployment (1 hour)

**File:** `Dockerfile`

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8080/health || exit 1

# Run API
CMD ["python", "-m", "uvicorn", "api.production_api:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "4"]
```

**File:** `docker-compose.yml`

```yaml
version: '3.8'

services:
  nba-api:
    build: .
    ports:
      - "8080:8080"
    volumes:
      - ./models:/app/models:ro
      - ./data:/app/data:ro
    environment:
      - PYTHONUNBUFFERED=1
    restart: unless-stopped
    
  redis-cache:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    restart: unless-stopped

volumes:
  redis-data:
```

**Build and run:**
```bash
docker-compose up -d
```

---

### 7.5 Validation Checklist

- [ ] ✅ Ensemble forecaster created
- [ ] ✅ Production API running
- [ ] ✅ Docker container working
- [ ] ✅ Health check endpoint responding
- [ ] ✅ Client library functional
- [ ] ✅ Response times <100ms

**Test deployment:**
```bash
# Test health
curl http://localhost:8080/health

# Test prediction
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"pattern": [0,2,5,8,10,12,13,15,14,15,16,18,17,15,14,13,12,15]}'
```

---

## Expected Performance

**Ensemble vs Individual Models:**

| Model | MAE | Coverage | Interpretability | Speed |
|-------|-----|----------|------------------|-------|
| Dejavu | ~6 pts | N/A | ★★★★★ | <5ms |
| LSTM | ~4 pts | N/A | ★☆☆☆☆ | <10ms |
| **Ensemble (Dejavu + LSTM)** | **~3.5 pts** | **N/A** | **★★★★☆** | **<15ms** |
| **+ Conformal Wrapper** | **—** | **✅ 95%** | **—** | **<1ms** |

**Note:** Conformal Prediction is not a forecasting model but a **wrapper** that provides uncertainty quantification with theoretical coverage guarantees. It takes the ensemble's point forecast and wraps it with calibrated prediction intervals.

**Benefits of Full System:**
- Better accuracy than individual models (Ensemble)
- Maintains interpretability (Dejavu neighbors)
- **Statistical coverage guarantees (Conformal wrapper)**
- Adaptive to distribution shifts (weighted quantiles)
- Production-ready performance

---

## Next Step

Proceed to **Step 8: Live Score Integration** to connect with your 5-second live scoring system.

---

*Action Step 7 of 10 - Ensemble & Production API*

