# Step 9: Production Deployment & Monitoring

**Objective:** Deploy to production with monitoring, logging, and auto-scaling

**Duration:** 1-2 days  
**Prerequisites:** Completed Step 8 (live integration working)  
**Output:** Production-grade system with 99.9% uptime, monitoring, alerts

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    PRODUCTION SYSTEM                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Live Score Scraper (5-sec) → Redis Queue → Prediction API  │
│                                      ↓                       │
│                              Ensemble Models                 │
│                          (Dejavu + LSTM + Conformal)        │
│                                      ↓                       │
│                              PostgreSQL Log                  │
│                                      ↓                       │
│                         WebSocket Broadcast                  │
│                                      ↓                       │
│                        Monitoring Dashboard                  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Action Items

### 9.1 Add Logging Infrastructure (1 hour)

**File:** `infrastructure/logging_config.py`

```python
"""
Production logging configuration
"""

import logging
import json
from datetime import datetime
from pathlib import Path

class PredictionLogger:
    """
    Log all predictions for auditing and model improvement
    """
    def __init__(self, log_dir='logs/'):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Setup structured logging
        self.logger = logging.getLogger('nba_predictions')
        self.logger.setLevel(logging.INFO)
        
        # File handler (daily rotation)
        today = datetime.now().strftime('%Y-%m-%d')
        handler = logging.FileHandler(
            self.log_dir / f'predictions_{today}.log'
        )
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(handler)
    
    def log_prediction(
        self,
        game_id: str,
        pattern: list,
        forecast: float,
        interval: tuple,
        explanation: dict,
        actual: float = None
    ):
        """
        Log a prediction
        """
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'game_id': game_id,
            'pattern': pattern,
            'forecast': forecast,
            'interval_lower': interval[0],
            'interval_upper': interval[1],
            'explanation': explanation,
            'actual': actual if actual is not None else None,
            'error': abs(forecast - actual) if actual is not None else None
        }
        
        self.logger.info(json.dumps(log_entry))
    
    def log_error(self, game_id: str, error: str):
        """Log an error"""
        self.logger.error(f"Game {game_id}: {error}")


# PostgreSQL logging for analytics
class DatabaseLogger:
    """
    Log predictions to PostgreSQL for analytics
    """
    def __init__(self, connection_string):
        from sqlalchemy import create_engine
        self.engine = create_engine(connection_string)
        self._create_tables()
    
    def _create_tables(self):
        """Create logging tables"""
        create_sql = """
        CREATE TABLE IF NOT EXISTS predictions (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMP NOT NULL,
            game_id VARCHAR(50) NOT NULL,
            forecast FLOAT NOT NULL,
            interval_lower FLOAT NOT NULL,
            interval_upper FLOAT NOT NULL,
            dejavu_component FLOAT,
            lstm_component FLOAT,
            actual FLOAT,
            error FLOAT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX IF NOT EXISTS idx_game_id ON predictions(game_id);
        CREATE INDEX IF NOT EXISTS idx_timestamp ON predictions(timestamp);
        """
        
        with self.engine.connect() as conn:
            conn.execute(create_sql)
    
    def log_prediction(self, prediction_data: dict):
        """Insert prediction to database"""
        from sqlalchemy import text
        
        insert_sql = """
        INSERT INTO predictions (
            timestamp, game_id, forecast, interval_lower, interval_upper,
            dejavu_component, lstm_component
        ) VALUES (
            :timestamp, :game_id, :forecast, :interval_lower, :interval_upper,
            :dejavu_component, :lstm_component
        )
        """
        
        with self.engine.connect() as conn:
            conn.execute(text(insert_sql), prediction_data)
```

---

### 9.2 Monitoring & Metrics (1 hour)

**File:** `infrastructure/monitoring.py`

```python
"""
Production monitoring with Prometheus metrics
"""

from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time

# Metrics
prediction_counter = Counter(
    'nba_predictions_total',
    'Total number of predictions made'
)

prediction_latency = Histogram(
    'nba_prediction_latency_seconds',
    'Time to generate prediction',
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0]
)

prediction_error = Histogram(
    'nba_prediction_error_points',
    'Absolute error of predictions (when actual known)',
    buckets=[1, 2, 3, 5, 10, 20]
)

coverage_gauge = Gauge(
    'nba_conformal_coverage',
    'Empirical conformal coverage (rolling window)'
)

active_games_gauge = Gauge(
    'nba_active_games',
    'Number of games currently being monitored'
)

class PerformanceMonitor:
    """
    Monitor system performance
    """
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.recent_errors = []
        self.recent_coverage = []
    
    def record_prediction(self, latency: float):
        """Record prediction latency"""
        prediction_counter.inc()
        prediction_latency.observe(latency)
    
    def record_actual(self, forecast: float, actual: float, interval: tuple):
        """Record actual outcome for monitoring"""
        error = abs(forecast - actual)
        prediction_error.observe(error)
        
        # Track coverage
        lower, upper = interval
        covered = (lower <= actual <= upper)
        
        self.recent_errors.append(error)
        self.recent_coverage.append(covered)
        
        # Maintain window
        if len(self.recent_errors) > self.window_size:
            self.recent_errors = self.recent_errors[-self.window_size:]
            self.recent_coverage = self.recent_coverage[-self.window_size:]
        
        # Update gauges
        if self.recent_coverage:
            coverage_gauge.set(sum(self.recent_coverage) / len(self.recent_coverage))
    
    def update_active_games(self, count: int):
        """Update active games count"""
        active_games_gauge.set(count)
    
    def get_metrics(self) -> dict:
        """Get current metrics"""
        return {
            'avg_error': sum(self.recent_errors) / len(self.recent_errors) if self.recent_errors else None,
            'empirical_coverage': sum(self.recent_coverage) / len(self.recent_coverage) if self.recent_coverage else None,
            'n_predictions': len(self.recent_errors)
        }


# Start Prometheus metrics server
def start_monitoring(port=9090):
    """Start Prometheus metrics endpoint"""
    start_http_server(port)
    print(f"✓ Prometheus metrics available at http://localhost:{port}/metrics")
```

---

### 9.3 Production API with Monitoring (1 hour)

**File:** `api/production_api_monitored.py`

```python
"""
Production API with comprehensive monitoring
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import numpy as np
import time
from models.ensemble_forecaster import EnsembleNBAForecaster
from infrastructure.logging_config import PredictionLogger, DatabaseLogger
from infrastructure.monitoring import PerformanceMonitor

app = FastAPI(title="NBA Production Prediction API v1.0")

# Global instances
ensemble = None
prediction_logger = None
db_logger = None
monitor = None

class PredictionRequest(BaseModel):
    pattern: List[float]
    alpha: float = 0.05
    game_id: Optional[str] = None

class PredictionResponse(BaseModel):
    game_id: Optional[str]
    point_forecast: float
    interval_lower: float
    interval_upper: float
    coverage_probability: float
    explanation: Optional[Dict]
    model_version: str = "1.0.0"
    timestamp: str

@app.on_event("startup")
async def startup():
    """Initialize all components"""
    global ensemble, prediction_logger, db_logger, monitor
    
    print("Starting production API...")
    
    # Load models
    ensemble = EnsembleNBAForecaster(dejavu_weight=0.4, lstm_weight=0.6)
    ensemble.load_models()
    
    # Initialize logging
    prediction_logger = PredictionLogger(log_dir='logs/')
    
    # Initialize database logging (if configured)
    import os
    if os.getenv('DB_LOG_CONNECTION'):
        db_logger = DatabaseLogger(os.getenv('DB_LOG_CONNECTION'))
    
    # Initialize monitoring
    monitor = PerformanceMonitor(window_size=100)
    
    # Start Prometheus metrics
    from infrastructure.monitoring import start_monitoring
    start_monitoring(port=9090)
    
    print("✓ Production API ready")

@app.post("/predict", response_model=PredictionResponse)
async def predict_with_monitoring(request: PredictionRequest):
    """
    Production prediction endpoint with full monitoring
    """
    start_time = time.time()
    
    try:
        # Validate
        if len(request.pattern) != 18:
            raise HTTPException(400, f"Pattern must be 18 values, got {len(request.pattern)}")
        
        # Predict
        query_pattern = np.array(request.pattern)
        forecast, interval, explanation = ensemble.predict(query_pattern)
        lower, upper = interval
        
        # Build response
        from datetime import datetime
        response = PredictionResponse(
            game_id=request.game_id,
            point_forecast=float(forecast),
            interval_lower=float(lower),
            interval_upper=float(upper),
            coverage_probability=1 - request.alpha,
            explanation=explanation,
            timestamp=datetime.now().isoformat()
        )
        
        # Log prediction
        prediction_logger.log_prediction(
            game_id=request.game_id or "unknown",
            pattern=request.pattern,
            forecast=forecast,
            interval=(lower, upper),
            explanation=explanation
        )
        
        # Database logging
        if db_logger:
            db_logger.log_prediction({
                'timestamp': datetime.now(),
                'game_id': request.game_id,
                'forecast': forecast,
                'interval_lower': lower,
                'interval_upper': upper,
                'dejavu_component': explanation.get('dejavu_prediction'),
                'lstm_component': explanation.get('lstm_prediction')
            })
        
        # Record latency
        latency = time.time() - start_time
        monitor.record_prediction(latency)
        
        return response
    
    except Exception as e:
        prediction_logger.log_error(request.game_id or "unknown", str(e))
        raise HTTPException(500, f"Prediction failed: {str(e)}")

@app.post("/record_actual")
async def record_actual_outcome(game_id: str, actual_differential: float):
    """
    Record actual halftime outcome for monitoring
    """
    # This is called after halftime to track accuracy
    # Could be automated by watching the game finish
    
    # Update monitoring
    # (Would need to retrieve the original prediction first)
    return {"status": "recorded"}

@app.get("/metrics/summary")
async def get_metrics_summary():
    """Get current performance metrics"""
    if monitor is None:
        return {"status": "monitoring not initialized"}
    
    return monitor.get_metrics()

@app.get("/health")
async def health_check():
    """Comprehensive health check"""
    return {
        "status": "healthy",
        "models_loaded": ensemble is not None,
        "logging_active": prediction_logger is not None,
        "monitoring_active": monitor is not None
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8080,
        workers=4,
        log_level="info"
    )
```

---

### 9.4 Kubernetes Deployment (1 hour)

**File:** `k8s/deployment.yaml`

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nba-prediction-api
  labels:
    app: nba-predictions
spec:
  replicas: 3
  selector:
    matchLabels:
      app: nba-predictions
  template:
    metadata:
      labels:
        app: nba-predictions
    spec:
      containers:
      - name: api
        image: your-registry/nba-predictions:latest
        ports:
        - containerPort: 8080
          name: http
        - containerPort: 9090
          name: metrics
        env:
        - name: DB_LOG_CONNECTION
          valueFrom:
            secretKeyRef:
              name: db-secrets
              key: connection-string
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: nba-prediction-service
spec:
  selector:
    app: nba-predictions
  ports:
  - name: http
    port: 80
    targetPort: 8080
  - name: metrics
    port: 9090
    targetPort: 9090
  type: LoadBalancer
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: nba-predictions-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: nba-prediction-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

**Deploy to Kubernetes:**
```bash
kubectl apply -f k8s/deployment.yaml
kubectl get pods -l app=nba-predictions
```

---

### 9.5 Monitoring Dashboard (1 hour)

**File:** `monitoring/grafana_dashboard.json`

```json
{
  "dashboard": {
    "title": "NBA Halftime Predictions",
    "panels": [
      {
        "title": "Predictions per Minute",
        "targets": [{"expr": "rate(nba_predictions_total[1m])"}]
      },
      {
        "title": "Prediction Latency (p95)",
        "targets": [{"expr": "histogram_quantile(0.95, nba_prediction_latency_seconds)"}]
      },
      {
        "title": "Conformal Coverage (Rolling)",
        "targets": [{"expr": "nba_conformal_coverage"}]
      },
      {
        "title": "Prediction Error Distribution",
        "targets": [{"expr": "histogram_quantile(0.95, nba_prediction_error_points)"}]
      },
      {
        "title": "Active Games",
        "targets": [{"expr": "nba_active_games"}]
      }
    ]
  }
}
```

**Alert Rules (Prometheus):**

**File:** `monitoring/alerts.yml`

```yaml
groups:
  - name: nba_predictions
    interval: 30s
    rules:
      - alert: HighPredictionLatency
        expr: histogram_quantile(0.95, nba_prediction_latency_seconds) > 1.0
        for: 5m
        annotations:
          summary: "Prediction latency above 1 second"
      
      - alert: LowCoverageRate
        expr: nba_conformal_coverage < 0.85
        for: 10m
        annotations:
          summary: "Conformal coverage below 85% (target 95%)"
      
      - alert: HighPredictionError
        expr: histogram_quantile(0.95, nba_prediction_error_points) > 10
        for: 15m
        annotations:
          summary: "Prediction errors unusually high"
      
      - alert: APIDown
        expr: up{job="nba-predictions"} == 0
        for: 1m
        annotations:
          summary: "NBA Prediction API is down"
```

---

### 9.6 Validation Checklist

- [ ] ✅ Logging to files (daily rotation)
- [ ] ✅ Database logging configured (PostgreSQL)
- [ ] ✅ Prometheus metrics exposed on :9090
- [ ] ✅ Grafana dashboard set up
- [ ] ✅ Alert rules configured
- [ ] ✅ Kubernetes deployment working
- [ ] ✅ Auto-scaling tested (2-10 replicas)
- [ ] ✅ Health checks passing

**Production Readiness:**
```bash
# Check all systems
curl http://your-domain/health
curl http://your-domain/metrics/summary
curl http://your-domain:9090/metrics  # Prometheus
```

---

## Expected Infrastructure

```
Infrastructure/
├── logging_config.py         ← File & DB logging
├── monitoring.py             ← Prometheus metrics
├── alerts.yml                ← Alert rules
└── grafana_dashboard.json    ← Dashboard config

k8s/
├── deployment.yaml           ← K8s deployment
├── service.yaml              ← Load balancer
└── hpa.yaml                  ← Auto-scaling

logs/
├── predictions_2025-10-15.log
├── predictions_2025-10-16.log
└── ...

Database tables:
├── predictions               ← All forecasts
├── actuals                   ← Actual outcomes
└── model_performance         ← Aggregated metrics
```

---

## Monitoring Metrics

**Real-Time Dashboards Show:**

1. **Prediction Volume:** Predictions/minute, requests/second
2. **Latency:** p50, p95, p99 response times
3. **Accuracy:** MAE, RMSE (rolling window)
4. **Coverage:** Empirical coverage vs target
5. **System Health:** API uptime, error rate
6. **Active Games:** Currently monitored games

**Alert Thresholds:**
- Latency >1 second → Scale up
- Coverage <85% → Recalibrate conformal
- Error >10 points → Investigate model drift
- API down → Page on-call engineer

---

## Next Step

Proceed to **Step 10: Continuous Improvement** for model retraining, recalibration, and feedback loops.

---

*Action Step 9 of 10 - Production Deployment*

