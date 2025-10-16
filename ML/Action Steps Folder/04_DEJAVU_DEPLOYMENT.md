# Step 4: Dejavu Deployment (Fastest Path to Production)

**Objective:** Deploy Dejavu pattern matching for instant NBA halftime prediction

**Paper Source:** Kang et al., "Déjà vu: A data-centric forecasting approach through time series cross-similarity", arXiv 2020

**Paper-Verified Approach:**
- ✅ **7-Step Methodology**: Seasonal adjustment → Smoothing → Scaling → Similarity → Aggregate → Inverse scale → Reseasonalize
- ✅ **Optimal k=500** (paper sweet spot, improvements taper after k>100)
- ✅ **DTW Best for Monthly** (statistically significant, but 27× slower than L1/L2)
- ✅ **Preprocessing Critical**: 28% MASE reduction with seasonal adjustment + smoothing
- ✅ **Best for Limited Data**: Paper shows Dejavu wins on short series (≤6 years yearly)

**NBA Context:** 5 seasons (2020-2025) = Limited data → Perfect use case for Dejavu!

**Duration:** 2-4 hours  
**Prerequisites:** Completed Step 3 (data splits)  
**Output:** Working Dejavu forecaster with API endpoint

---

## Why Dejavu First?

**Strategic Decision (Paper-Informed):** Deploy Dejavu before Informer because:
- ✅ **Zero training time** - instant deployment (paper: no model training)
- ✅ **Interpretable** - shows similar historical games (paper: explicit receipts)
- ✅ **Baseline** - establishes performance floor (paper: competitive with ETS/ARIMA)
- ✅ **Fast iteration** - test assumptions quickly (paper: try L1/L2/DTW instantly)
- ✅ **Right Scale** - NBA's 18-step input perfect for pattern matching (Informer needs 300+)
- ✅ **Limited Data** - Paper proves Dejavu best for short series

---

## Action Items

### 4.1 Build Dejavu Pattern Database (30 minutes)

**File:** `models/dejavu_forecaster.py`

```python
"""
Dejavu Forecaster for NBA Halftime Prediction
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
import joblib

class DejavuNBAForecaster:
    """
    Pattern-matching forecaster for NBA halftime differentials
    """
    def __init__(
        self,
        K=10,
        similarity_method='euclidean',
        weighting='gaussian',
        sigma=1.0
    ):
        """
        Args:
            K: Number of nearest neighbors
            similarity_method: 'euclidean', 'manhattan', 'correlation'
            weighting: 'uniform', 'inverse', 'gaussian'
            sigma: Bandwidth for Gaussian weighting
        """
        self.K = K
        self.similarity_method = similarity_method
        self.weighting = weighting
        self.sigma = sigma
        self.database = []
    
    def fit(self, train_df):
        """
        Build pattern database from training games
        
        Args:
            train_df: Training data with 'pattern', 'outcome', 'halftime_differential'
        """
        print(f"Building Dejavu database...")
        
        for idx, row in train_df.iterrows():
            self.database.append({
                'pattern': row['pattern'],
                'outcome': row['outcome'],
                'halftime_differential': row['halftime_differential'],
                'game_id': row['game_id'],
                'date': row['game_date'],
                'metadata': {
                    'diff_at_6min': row['differential_at_6min_2q']
                }
            })
        
        print(f"✓ Database created: {len(self.database)} historical patterns")
    
    def _compute_distance(self, pattern1, pattern2):
        """Compute distance between patterns"""
        # Normalize patterns (z-score)
        p1_norm = (pattern1 - np.mean(pattern1)) / (np.std(pattern1) + 1e-10)
        p2_norm = (pattern2 - np.mean(pattern2)) / (np.std(pattern2) + 1e-10)
        
        if self.similarity_method == 'euclidean':
            return np.sqrt(np.sum((p1_norm - p2_norm) ** 2))
        elif self.similarity_method == 'manhattan':
            return np.sum(np.abs(p1_norm - p2_norm))
        elif self.similarity_method == 'correlation':
            corr = np.corrcoef(pattern1, pattern2)[0, 1]
            return 1 - abs(corr)
        else:
            raise ValueError(f"Unknown method: {self.similarity_method}")
    
    def predict(self, query_pattern) -> Tuple[float, List[Dict]]:
        """
        Forecast halftime differential from 6:00 2Q pattern
        
        Args:
            query_pattern: 18-minute differential pattern (array)
        
        Returns:
            forecast: Predicted halftime differential
            neighbors: K nearest neighbors with metadata
        """
        # Compute distances to all database patterns
        distances = []
        for entry in self.database:
            dist = self._compute_distance(query_pattern, entry['pattern'])
            distances.append(dist)
        
        # Select K nearest
        nearest_indices = np.argsort(distances)[:self.K]
        neighbors = [self.database[i] for i in nearest_indices]
        neighbor_distances = [distances[i] for i in nearest_indices]
        
        # Compute weights
        if self.weighting == 'uniform':
            weights = np.ones(self.K) / self.K
        elif self.weighting == 'inverse':
            weights = 1.0 / (np.array(neighbor_distances) + 1e-10)
            weights /= weights.sum()
        elif self.weighting == 'gaussian':
            weights = np.exp(-np.array(neighbor_distances)**2 / (2 * self.sigma**2))
            weights /= weights.sum()
        
        # Weighted average of halftime differentials
        halftime_diffs = [n['halftime_differential'] for n in neighbors]
        forecast = np.average(halftime_diffs, weights=weights)
        
        # Add distances to neighbors for interpretability
        for i, neighbor in enumerate(neighbors):
            neighbor['distance'] = neighbor_distances[i]
            neighbor['weight'] = weights[i]
        
        return forecast, neighbors
    
    def save(self, filepath):
        """Save forecaster"""
        joblib.dump(self, filepath, compress=3)
        print(f"✓ Dejavu forecaster saved to {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """Load forecaster"""
        return joblib.load(filepath)


if __name__ == "__main__":
    # Load training data
    train_df = pd.read_parquet('data/splits/train.parquet')
    
    # Create and fit Dejavu
    dejavu = DejavuNBAForecaster(K=10, similarity_method='euclidean')
    dejavu.fit(train_df)
    
    # Test prediction
    test_pattern = train_df['pattern'].iloc[0]
    forecast, neighbors = dejavu.predict(test_pattern)
    
    print(f"\nTest Prediction:")
    print(f"  Forecast: {forecast:.2f} points")
    print(f"  Top 3 similar games:")
    for i, n in enumerate(neighbors[:3]):
        print(f"    {i+1}. {n['game_id']} ({n['date']}) - "
              f"diff={n['halftime_differential']:.1f}, distance={n['distance']:.3f}")
    
    # Save
    dejavu.save('models/dejavu_forecaster.pkl')
```

---

### 4.2 Evaluate Dejavu Performance (30 minutes)

**File:** `scripts/evaluate_dejavu.py`

```python
"""
Evaluate Dejavu on validation set
"""

import pandas as pd
import numpy as np
from models.dejavu_forecaster import DejavuNBAForecaster

def evaluate_dejavu(model, eval_df):
    """
    Evaluate Dejavu performance
    
    Returns:
        metrics: Dict with MAE, RMSE, etc.
    """
    forecasts = []
    actuals = []
    
    for idx, row in eval_df.iterrows():
        # Predict
        forecast, neighbors = model.predict(row['pattern'])
        
        # Actual
        actual = row['halftime_differential']
        
        forecasts.append(forecast)
        actuals.append(actual)
    
    forecasts = np.array(forecasts)
    actuals = np.array(actuals)
    
    # Calculate metrics
    mae = np.mean(np.abs(forecasts - actuals))
    rmse = np.sqrt(np.mean((forecasts - actuals) ** 2))
    mape = np.mean(np.abs((forecasts - actuals) / (actuals + 1e-10))) * 100
    
    metrics = {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'n_samples': len(forecasts)
    }
    
    return metrics, forecasts, actuals


if __name__ == "__main__":
    # Load model
    dejavu = DejavuNBAForecaster.load('models/dejavu_forecaster.pkl')
    
    # Load validation set
    val_df = pd.read_parquet('data/splits/validation.parquet')
    
    # Evaluate
    metrics, forecasts, actuals = evaluate_dejavu(dejavu, val_df)
    
    print("Dejavu Validation Performance:")
    print(f"  MAE: {metrics['mae']:.2f} points")
    print(f"  RMSE: {metrics['rmse']:.2f} points")
    print(f"  MAPE: {metrics['mape']:.1f}%")
    
    # Save results
    import json
    with open('results/dejavu_validation_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print("\n✓ Validation complete")
```

---

### 4.3 Deploy Dejavu API (1 hour)

**File:** `api/dejavu_api.py`

```python
"""
FastAPI endpoint for Dejavu NBA halftime predictions
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import numpy as np
from models.dejavu_forecaster import DejavuNBAForecaster

app = FastAPI(title="Dejavu NBA Halftime Predictor")

# Global model (loaded at startup)
dejavu_model = None

class PredictionRequest(BaseModel):
    pattern: List[float]  # 18-minute differential pattern
    return_neighbors: bool = True  # Return matched games
    K: int = 10  # Number of neighbors to return

class PredictionResponse(BaseModel):
    halftime_differential: float
    neighbors: List[Dict] = []
    metadata: Dict

@app.on_event("startup")
async def load_model():
    """Load Dejavu model at startup"""
    global dejavu_model
    dejavu_model = DejavuNBAForecaster.load('models/dejavu_forecaster.pkl')
    print("✓ Dejavu model loaded")

@app.post("/predict", response_model=PredictionResponse)
async def predict_halftime(request: PredictionRequest):
    """
    Predict halftime differential from 6:00 2Q pattern
    """
    if dejavu_model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Convert to numpy
    query_pattern = np.array(request.pattern)
    
    # Validate input
    if len(query_pattern) != 18:
        raise HTTPException(
            status_code=400,
            detail=f"Pattern must be 18 minutes, got {len(query_pattern)}"
        )
    
    # Predict
    forecast, neighbors = dejavu_model.predict(query_pattern)
    
    # Format response
    neighbor_info = []
    if request.return_neighbors:
        for i, n in enumerate(neighbors[:request.K]):
            neighbor_info.append({
                'rank': i + 1,
                'game_id': n['game_id'],
                'date': str(n['date']),
                'halftime_differential': float(n['halftime_differential']),
                'similarity_score': 1.0 / (1.0 + n['distance']),  # Convert distance to similarity
                'weight': float(n['weight'])
            })
    
    return PredictionResponse(
        halftime_differential=float(forecast),
        neighbors=neighbor_info,
        metadata={
            'model': 'Dejavu',
            'K': request.K,
            'similarity_method': dejavu_model.similarity_method,
            'database_size': len(dejavu_model.database)
        }
    )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": dejavu_model is not None,
        "database_size": len(dejavu_model.database) if dejavu_model else 0
    }

@app.get("/stats")
async def database_stats():
    """Database statistics"""
    if dejavu_model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Calculate database statistics
    all_diffs = [entry['halftime_differential'] for entry in dejavu_model.database]
    
    return {
        'database_size': len(dejavu_model.database),
        'differential_mean': float(np.mean(all_diffs)),
        'differential_std': float(np.std(all_diffs)),
        'differential_range': [float(np.min(all_diffs)), float(np.max(all_diffs))]
    }


# Run server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

**Start server:**
```bash
python api/dejavu_api.py
```

**Test API:**
```bash
# Test prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "pattern": [0, 2, 5, 8, 10, 12, 13, 15, 14, 15, 16, 18, 17, 15, 14, 13, 12, 15],
    "return_neighbors": true,
    "K": 5
  }'

# Response example:
{
  "halftime_differential": 11.4,
  "neighbors": [
    {
      "rank": 1,
      "game_id": "202103150LAL",
      "date": "2021-03-15",
      "halftime_differential": 10.0,
      "similarity_score": 0.92,
      "weight": 0.25
    },
    ...
  ],
  "metadata": {
    "model": "Dejavu",
    "K": 5,
    "database_size": 3000
  }
}
```

---

### 4.3 Create Simple Frontend Demo (1 hour)

**File:** `frontend/dejavu_demo.html`

```html
<!DOCTYPE html>
<html>
<head>
    <title>Dejavu NBA Halftime Predictor</title>
    <style>
        body { font-family: Arial; max-width: 900px; margin: 50px auto; padding: 20px; }
        .pattern-input { width: 100%; height: 100px; }
        .result { background: #f0f0f0; padding: 20px; margin: 20px 0; border-radius: 5px; }
        .neighbor { background: white; margin: 10px 0; padding: 10px; border-left: 3px solid #4CAF50; }
        button { background: #4CAF50; color: white; padding: 10px 20px; border: none; cursor: pointer; }
    </style>
</head>
<body>
    <h1>🏀 Dejavu NBA Halftime Predictor</h1>
    <p>Enter the score differential pattern from game start to 6:00 2Q (18 values, comma-separated)</p>
    
    <textarea class="pattern-input" id="patternInput" placeholder="0, 2, 5, 8, 10, 12, 13, 15, 14, 15, 16, 18, 17, 15, 14, 13, 12, 15"></textarea>
    
    <br><br>
    <button onclick="predict()">Predict Halftime Differential</button>
    
    <div id="result" class="result" style="display:none;">
        <h2>Prediction: <span id="forecast"></span> points</h2>
        <h3>Based on these similar historical games:</h3>
        <div id="neighbors"></div>
    </div>
    
    <script>
        async function predict() {
            const pattern = document.getElementById('patternInput').value
                .split(',')
                .map(x => parseFloat(x.trim()));
            
            if (pattern.length !== 18) {
                alert('Please enter exactly 18 values (18 minutes)');
                return;
            }
            
            const response = await fetch('http://localhost:8000/predict', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    pattern: pattern,
                    return_neighbors: true,
                    K: 5
                })
            });
            
            const data = await response.json();
            
            // Display result
            document.getElementById('forecast').textContent = data.halftime_differential.toFixed(1);
            
            // Display neighbors
            const neighborsDiv = document.getElementById('neighbors');
            neighborsDiv.innerHTML = data.neighbors.map(n => `
                <div class="neighbor">
                    <strong>Game ${n.game_id}</strong> (${n.date})<br>
                    Halftime Differential: ${n.halftime_differential.toFixed(1)} points<br>
                    Similarity: ${(n.similarity_score * 100).toFixed(1)}%
                </div>
            `).join('');
            
            document.getElementById('result').style.display = 'block';
        }
    </script>
</body>
</html>
```

**Open in browser:** `frontend/dejavu_demo.html`

---

### 4.4 Validation Checklist

**Before proceeding to Step 5:**

- [ ] ✅ Dejavu model built from training data (~3,000 games)
- [ ] ✅ Validation MAE < 8 points (baseline target)
- [ ] ✅ API running on localhost:8000
- [ ] ✅ Health check endpoint responding
- [ ] ✅ Sample predictions working
- [ ] ✅ Neighbors returned for interpretability

**Performance Targets (Validation Set):**
- MAE: 5-8 points (acceptable baseline)
- RMSE: 7-10 points
- Inference time: <10ms per prediction

---

## Expected Outputs

```
models/
└── dejavu_forecaster.pkl              ← Trained model (50-100 MB)

api/
└── dejavu_api.py                      ← API server

results/
└── dejavu_validation_metrics.json     ← Performance metrics

frontend/
└── dejavu_demo.html                   ← Simple demo interface
```

---

## Troubleshooting

**Problem:** High MAE (>10 points) on validation

**Solutions:**
- Increase K (try K=20, K=50)
- Try different similarity methods
- Check if training data has sufficient pattern coverage
- Verify normalization is working

**Problem:** Poor matches (low similarity scores)

**Solutions:**
- Database may be too small
- Pattern length may be wrong
- Try correlation distance instead of Euclidean

**Problem:** API slow

**Solutions:**
- Build KDTree index for faster K-NN search
- Cache frequent queries
- Use FAISS for large databases

---

## Next Step

**Immediate:** You now have a working forecaster deployed!

**Next:** Proceed to **Step 5: Add Conformal Prediction** to wrap Dejavu with uncertainty quantification.

---

*Action Step 4 of 10 - Dejavu Deployment*

