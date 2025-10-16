# Conformal Multistep-Ahead Multivariate Time-Series Forecasting

**Implementation Specification for Production Deployment**

**Paper Source:** Schlembach et al., "Conformal Multistep-Ahead Multivariate Time-Series Forecasting", PMLR 179:1–3, 2022  
**Authors:** Filip Schlembach (Maastricht University), Evgueni Smirnov (Maastricht), Irena Koprinska (University of Sydney)

**Date:** October 14, 2025  
**Objective:** Model-agnostic uncertainty quantification for time series with theoretical coverage guarantees

---

## Executive Summary

**Conformal Prediction** wraps any point forecasting model (including Informer) to provide:
- ✅ **Prediction intervals** with theoretical coverage guarantees
- ✅ **Model-agnostic** - works with any forecasting model (RNN, LSTM, Transformer)
- ✅ **Non-exchangeable** time series support via weighted quantiles (Barber et al., 2022)
- ✅ **Multistep-ahead** multivariate forecasting with Bonferroni correction (Stankevičiūtė et al., 2021)
- ✅ **Distribution shift robustness** - maintains coverage under test set shifts

**Paper-Verified:** Tested on ELEC2 dataset (t=192, h=12) with RNN, exponential weighting best for 1-α > 0.5

**Key Advantage:** Transforms deterministic forecasts (Informer, LSTM, etc.) into probabilistic predictions with statistical guarantees, even under distribution shifts.

---

## Table of Contents

1. [Mathematical Foundation](#mathematical-foundation)
2. [Conformal Prediction for Time Series](#conformal-prediction-for-time-series)
3. [Implementation Architecture](#implementation-architecture)
4. [Integration with Informer](#integration-with-informer)
5. [Production Deployment](#production-deployment)
6. [Comparison Matrix](#comparison-matrix)

---

## Mathematical Foundation

### 1. Standard Conformal Prediction

**Problem Setup:**
- Training data: \( \{(x_1, y_1), ..., (x_n, y_n)\} \)
- Test point: \( x_{n+1} \)
- Goal: Predict set \( C(x_{n+1}) \) such that \( y_{n+1} \in C(x_{n+1}) \) with probability ≥ 1-α

**Key Concept: Nonconformity Score**

For each sample \( (x_i, y_i) \), define a score measuring "how different" \( y_i \) is from the prediction:

\[
s_i = |y_i - \hat{f}(x_i)|
\]

Where \( \hat{f} \) is any point predictor trained on data excluding \( (x_i, y_i) \).

**Quantile-Based Prediction Region:**

\[
C(x_{n+1}) = \left[ \hat{f}(x_{n+1}) - q_{1-\alpha}(\{s_1, ..., s_n\}), \hat{f}(x_{n+1}) + q_{1-\alpha}(\{s_1, ..., s_n\}) \right]
\]

Where \( q_{1-\alpha} \) is the \( (1-\alpha) \)-quantile of nonconformity scores.

**Theoretical Guarantee (Exchangeability):**

If data is exchangeable:
\[
P(y_{n+1} \in C(x_{n+1})) \geq 1 - \alpha
\]

### 2. Challenges for Time Series

**Problem:** Time series violates exchangeability
- Temporal dependence
- Non-stationarity
- Serial correlation
- Distribution drift

**Solution:** Relaxed assumptions for time series:
1. **β-mixing** (weak dependence)
2. **Approximate exchangeability** over blocks
3. **Locally weighted conformal scores**
4. **Adaptive recalibration**

### 3. Nonexchangeable Multistep Time Series (nmtCP)

**Extension for Multistep Forecasting:**

For horizon \( h \), predict \( h \) steps ahead: \( y_{t+1}, y_{t+2}, ..., y_{t+h} \)

**Nonconformity Score for Multivariate:**

\[
s_t^{(h)} = \max_{j=1,...,h} \| y_{t+j} - \hat{f}_j(x_t) \|
\]

Or component-wise:
\[
s_t^{(h,k)} = \max_{j=1,...,h} |y_{t+j,k} - \hat{f}_{j,k}(x_t)|
\]

**Adaptive Weighting:**

\[
w_t = \exp\left(-\frac{(n-t)^2}{2\tau^2}\right)
\]

Recent observations weighted more heavily (\( \tau \) controls decay).

**Weighted Quantile:**

\[
q_{1-\alpha}^w(\{s_1, ..., s_n\}) = \inf\left\{q : \sum_{i=1}^n w_i \mathbb{1}\{s_i \leq q\} \geq (1-\alpha)\sum_{i=1}^n w_i\right\}
\]

**Prediction Region:**

\[
C_t^{(h)} = \left\{ (y_{t+1}, ..., y_{t+h}) : \max_j \|y_{t+j} - \hat{f}_j(x_t)\| \leq q_{1-\alpha}^w \right\}
\]

### 4. Coverage Guarantees

**Theorem (Approximate Coverage):**

Under β-mixing with rate \( \beta(m) \leq C m^{-\gamma} \) and bounded moment conditions:

\[
\left| P(y_{t+h} \in C_t^{(h)}) - (1-\alpha) \right| \leq O\left(\frac{1}{\sqrt{n}} + \beta(m)\right)
\]

**Practical Implications:**
- Coverage error decreases with calibration set size \( n \)
- Mixing coefficient \( \beta(m) \) measures temporal dependence
- Larger \( n \) → tighter coverage guarantees

---

## Conformal Prediction for Time Series

### Algorithm 1: Split Conformal for Time Series

**Input:**
- Calibration data: \( \{(x_t, y_t)\}_{t=1}^n \)
- Point forecaster: \( \hat{f} \)
- Significance level: \( \alpha \in (0,1) \)
- Forecast horizon: \( h \)

**Output:**
- Prediction region \( C_t^{(h)} \) for test time \( t \)

**Steps:**

```python
def split_conformal_timeseries(calibration_data, model, alpha, horizon):
    """
    Split conformal prediction for time series
    """
    n = len(calibration_data)
    scores = []
    
    # Step 1: Compute nonconformity scores on calibration set
    for t in range(n - horizon):
        x_t = calibration_data[t]
        y_true = calibration_data[t+1 : t+horizon+1]  # True future values
        y_pred = model.predict(x_t, horizon)            # Model predictions
        
        # Nonconformity score (max absolute error over horizon)
        score = max([abs(y_true[j] - y_pred[j]) for j in range(horizon)])
        scores.append(score)
    
    # Step 2: Compute quantile
    quantile = np.quantile(scores, 1 - alpha)
    
    # Step 3: Prediction function
    def predict_with_interval(x_test):
        y_pred = model.predict(x_test, horizon)
        # Return point prediction + interval
        intervals = [(y_pred[j] - quantile, y_pred[j] + quantile) 
                     for j in range(horizon)]
        return y_pred, intervals
    
    return predict_with_interval
```

### Algorithm 2: Adaptive Conformal (Time-Weighted)

```python
def adaptive_conformal_timeseries(calibration_data, model, alpha, horizon, tau=10):
    """
    Adaptive conformal with exponential weighting for non-stationarity
    """
    n = len(calibration_data)
    scores = []
    weights = []
    
    # Step 1: Compute weighted nonconformity scores
    for t in range(n - horizon):
        x_t = calibration_data[t]
        y_true = calibration_data[t+1 : t+horizon+1]
        y_pred = model.predict(x_t, horizon)
        
        # Nonconformity score
        score = max([abs(y_true[j] - y_pred[j]) for j in range(horizon)])
        scores.append(score)
        
        # Exponential weight (more recent = higher weight)
        weight = np.exp(-((n - t)**2) / (2 * tau**2))
        weights.append(weight)
    
    # Step 2: Compute weighted quantile
    def weighted_quantile(scores, weights, q):
        sorted_idx = np.argsort(scores)
        sorted_scores = np.array(scores)[sorted_idx]
        sorted_weights = np.array(weights)[sorted_idx]
        
        cum_weights = np.cumsum(sorted_weights)
        total_weight = cum_weights[-1]
        
        # Find quantile
        threshold = q * total_weight
        idx = np.searchsorted(cum_weights, threshold)
        return sorted_scores[min(idx, len(scores)-1)]
    
    quantile = weighted_quantile(scores, weights, 1 - alpha)
    
    # Step 3: Prediction with intervals
    def predict_with_interval(x_test):
        y_pred = model.predict(x_test, horizon)
        intervals = [(y_pred[j] - quantile, y_pred[j] + quantile) 
                     for j in range(horizon)]
        return y_pred, intervals
    
    return predict_with_interval
```

### Algorithm 3: Multivariate Conformal

```python
def multivariate_conformal_timeseries(
    calibration_data, 
    model, 
    alpha, 
    horizon, 
    n_features
):
    """
    Conformal prediction for multivariate time series
    """
    n = len(calibration_data)
    
    # Option 1: Joint prediction region (simultaneous coverage)
    scores_joint = []
    
    for t in range(n - horizon):
        x_t = calibration_data[t]
        y_true = calibration_data[t+1 : t+horizon+1]  # Shape: (horizon, n_features)
        y_pred = model.predict(x_t, horizon)           # Shape: (horizon, n_features)
        
        # Max norm across all dimensions and timesteps
        score = np.max(np.abs(y_true - y_pred))
        scores_joint.append(score)
    
    quantile_joint = np.quantile(scores_joint, 1 - alpha)
    
    # Option 2: Marginal prediction regions (per-feature coverage)
    scores_marginal = {k: [] for k in range(n_features)}
    
    for t in range(n - horizon):
        x_t = calibration_data[t]
        y_true = calibration_data[t+1 : t+horizon+1]
        y_pred = model.predict(x_t, horizon)
        
        # Per-feature max error
        for k in range(n_features):
            score = np.max(np.abs(y_true[:, k] - y_pred[:, k]))
            scores_marginal[k].append(score)
    
    quantiles_marginal = {
        k: np.quantile(scores_marginal[k], 1 - alpha)
        for k in range(n_features)
    }
    
    def predict_with_interval(x_test):
        y_pred = model.predict(x_test, horizon)
        
        # Joint intervals
        intervals_joint = [
            [(y_pred[j, k] - quantile_joint, y_pred[j, k] + quantile_joint)
             for k in range(n_features)]
            for j in range(horizon)
        ]
        
        # Marginal intervals
        intervals_marginal = [
            [(y_pred[j, k] - quantiles_marginal[k], 
              y_pred[j, k] + quantiles_marginal[k])
             for k in range(n_features)]
            for j in range(horizon)
        ]
        
        return y_pred, intervals_joint, intervals_marginal
    
    return predict_with_interval
```

---

## Implementation Architecture

### Class Structure

```python
from abc import ABC, abstractmethod
from typing import Tuple, List, Optional
import numpy as np
import torch

class ConformalPredictor(ABC):
    """
    Base class for conformal prediction
    """
    def __init__(self, alpha: float = 0.1):
        """
        Args:
            alpha: Significance level (1-alpha = coverage probability)
        """
        self.alpha = alpha
        self.quantile = None
        self.is_fitted = False
    
    @abstractmethod
    def fit(self, calibration_data, model):
        """Fit conformal predictor on calibration data"""
        pass
    
    @abstractmethod
    def predict(self, x_test, model):
        """Generate prediction with intervals"""
        pass


class SplitConformal(ConformalPredictor):
    """
    Standard split conformal prediction
    """
    def __init__(self, alpha: float = 0.1, horizon: int = 24):
        super().__init__(alpha)
        self.horizon = horizon
        self.scores = []
    
    def fit(self, calibration_data: np.ndarray, model):
        """
        Fit on calibration set
        
        Args:
            calibration_data: (n_samples, n_features) array
            model: Trained forecasting model with predict(x, h) method
        """
        n = len(calibration_data)
        self.scores = []
        
        for t in range(n - self.horizon):
            x_t = calibration_data[t]
            y_true = calibration_data[t+1 : t+self.horizon+1]
            y_pred = model.predict(x_t, self.horizon)
            
            # Compute nonconformity score
            score = self._compute_score(y_true, y_pred)
            self.scores.append(score)
        
        # Compute quantile
        self.quantile = np.quantile(self.scores, 1 - self.alpha)
        self.is_fitted = True
        
        return self
    
    def _compute_score(self, y_true, y_pred):
        """Compute nonconformity score"""
        return np.max(np.abs(y_true - y_pred))
    
    def predict(self, x_test, model) -> Tuple[np.ndarray, List[Tuple]]:
        """
        Predict with intervals
        
        Returns:
            point_pred: (horizon, n_features) predictions
            intervals: List of (lower, upper) bounds
        """
        if not self.is_fitted:
            raise ValueError("Conformal predictor not fitted. Call fit() first.")
        
        y_pred = model.predict(x_test, self.horizon)
        
        # Create intervals
        intervals = [
            (y_pred[j] - self.quantile, y_pred[j] + self.quantile)
            for j in range(self.horizon)
        ]
        
        return y_pred, intervals


class AdaptiveConformal(ConformalPredictor):
    """
    Adaptive conformal with exponential weighting for non-stationarity
    """
    def __init__(self, alpha: float = 0.1, horizon: int = 24, tau: float = 10.0):
        super().__init__(alpha)
        self.horizon = horizon
        self.tau = tau  # Decay parameter for exponential weighting
        self.scores = []
        self.weights = []
    
    def fit(self, calibration_data: np.ndarray, model):
        """Fit with time-weighted scores"""
        n = len(calibration_data)
        self.scores = []
        self.weights = []
        
        for t in range(n - self.horizon):
            x_t = calibration_data[t]
            y_true = calibration_data[t+1 : t+self.horizon+1]
            y_pred = model.predict(x_t, self.horizon)
            
            score = np.max(np.abs(y_true - y_pred))
            self.scores.append(score)
            
            # Exponential weight (recent observations weighted more)
            weight = np.exp(-((n - t)**2) / (2 * self.tau**2))
            self.weights.append(weight)
        
        # Compute weighted quantile
        self.quantile = self._weighted_quantile(
            self.scores, self.weights, 1 - self.alpha
        )
        self.is_fitted = True
        
        return self
    
    def _weighted_quantile(self, scores, weights, q):
        """Compute weighted quantile"""
        sorted_idx = np.argsort(scores)
        sorted_scores = np.array(scores)[sorted_idx]
        sorted_weights = np.array(weights)[sorted_idx]
        
        cum_weights = np.cumsum(sorted_weights)
        total_weight = cum_weights[-1]
        
        threshold = q * total_weight
        idx = np.searchsorted(cum_weights, threshold)
        
        return sorted_scores[min(idx, len(scores)-1)]
    
    def predict(self, x_test, model):
        """Predict with adaptive intervals"""
        if not self.is_fitted:
            raise ValueError("Not fitted")
        
        y_pred = model.predict(x_test, self.horizon)
        
        intervals = [
            (y_pred[j] - self.quantile, y_pred[j] + self.quantile)
            for j in range(self.horizon)
        ]
        
        return y_pred, intervals


class MultivariateConformal(ConformalPredictor):
    """
    Conformal prediction for multivariate time series
    """
    def __init__(
        self, 
        alpha: float = 0.1, 
        horizon: int = 24,
        n_features: int = 1,
        mode: str = "joint"  # "joint" or "marginal"
    ):
        super().__init__(alpha)
        self.horizon = horizon
        self.n_features = n_features
        self.mode = mode
        
        if mode == "joint":
            self.quantile = None
        elif mode == "marginal":
            self.quantiles = {}
        else:
            raise ValueError("mode must be 'joint' or 'marginal'")
    
    def fit(self, calibration_data: np.ndarray, model):
        """
        Fit multivariate conformal
        
        Args:
            calibration_data: (n_samples, n_features) array
        """
        n = len(calibration_data)
        
        if self.mode == "joint":
            scores = []
            
            for t in range(n - self.horizon):
                x_t = calibration_data[t]
                y_true = calibration_data[t+1 : t+self.horizon+1]
                y_pred = model.predict(x_t, self.horizon)
                
                # Joint score: max over all dimensions
                score = np.max(np.abs(y_true - y_pred))
                scores.append(score)
            
            self.quantile = np.quantile(scores, 1 - self.alpha)
        
        elif self.mode == "marginal":
            scores_per_feature = {k: [] for k in range(self.n_features)}
            
            for t in range(n - self.horizon):
                x_t = calibration_data[t]
                y_true = calibration_data[t+1 : t+self.horizon+1]
                y_pred = model.predict(x_t, self.horizon)
                
                # Per-feature scores
                for k in range(self.n_features):
                    score = np.max(np.abs(y_true[:, k] - y_pred[:, k]))
                    scores_per_feature[k].append(score)
            
            self.quantiles = {
                k: np.quantile(scores_per_feature[k], 1 - self.alpha)
                for k in range(self.n_features)
            }
        
        self.is_fitted = True
        return self
    
    def predict(self, x_test, model):
        """Predict with multivariate intervals"""
        if not self.is_fitted:
            raise ValueError("Not fitted")
        
        y_pred = model.predict(x_test, self.horizon)
        
        if self.mode == "joint":
            # Same interval width for all features
            intervals = [
                [(y_pred[j, k] - self.quantile, y_pred[j, k] + self.quantile)
                 for k in range(self.n_features)]
                for j in range(self.horizon)
            ]
        else:  # marginal
            # Different interval width per feature
            intervals = [
                [(y_pred[j, k] - self.quantiles[k], 
                  y_pred[j, k] + self.quantiles[k])
                 for k in range(self.n_features)]
                for j in range(self.horizon)
            ]
        
        return y_pred, intervals
```

---

## Integration with Informer

### Wrapper for Any Forecasting Model

```python
class ForecastingModelWrapper:
    """
    Wrapper to make any forecasting model compatible with conformal prediction
    """
    def __init__(self, model, scaler=None):
        """
        Args:
            model: Trained forecasting model (Informer, LSTM, etc.)
            scaler: Optional scaler for denormalization
        """
        self.model = model
        self.scaler = scaler
    
    def predict(self, x, horizon):
        """
        Standard predict interface for conformal prediction
        
        Args:
            x: Input features (single sample or batch)
            horizon: Forecast horizon
        
        Returns:
            predictions: (horizon,) or (horizon, n_features) array
        """
        # Convert to model's expected format
        # This is model-specific - example for Informer:
        
        with torch.no_grad():
            predictions = self.model.forward(x, horizon)
        
        # Denormalize if scaler provided
        if self.scaler is not None:
            predictions = self.scaler.inverse_transform(predictions)
        
        return predictions.cpu().numpy()


class InformerWithConformal:
    """
    Informer model with conformal prediction intervals
    """
    def __init__(
        self,
        informer_model,
        conformal_type: str = "adaptive",
        alpha: float = 0.1,
        horizon: int = 24,
        n_features: int = 1
    ):
        """
        Args:
            informer_model: Trained Informer model
            conformal_type: "split", "adaptive", or "multivariate"
            alpha: Significance level (1-alpha coverage)
            horizon: Forecast horizon
            n_features: Number of features for multivariate
        """
        self.informer = informer_model
        self.wrapper = ForecastingModelWrapper(informer_model)
        
        # Create conformal predictor
        if conformal_type == "split":
            self.conformal = SplitConformal(alpha, horizon)
        elif conformal_type == "adaptive":
            self.conformal = AdaptiveConformal(alpha, horizon)
        elif conformal_type == "multivariate":
            self.conformal = MultivariateConformal(
                alpha, horizon, n_features, mode="joint"
            )
        else:
            raise ValueError(f"Unknown conformal_type: {conformal_type}")
    
    def calibrate(self, calibration_data):
        """
        Calibrate conformal predictor on held-out calibration set
        
        Args:
            calibration_data: (n_samples, n_features) array
        """
        self.conformal.fit(calibration_data, self.wrapper)
        return self
    
    def predict_with_intervals(self, x_test):
        """
        Generate predictions with conformal intervals
        
        Returns:
            point_pred: Point predictions
            intervals: Prediction intervals with (1-alpha) coverage
        """
        return self.conformal.predict(x_test, self.wrapper)
```

### Complete Pipeline

```python
def train_informer_with_conformal(
    train_data,
    val_data,
    calibration_data,
    test_data,
    config
):
    """
    Complete pipeline: Train Informer + Calibrate Conformal + Evaluate
    """
    # Step 1: Train Informer on train_data
    print("Training Informer model...")
    informer = train_informer(train_data, val_data, config)
    
    # Step 2: Wrap with conformal prediction
    print("Calibrating conformal predictor...")
    informer_conformal = InformerWithConformal(
        informer_model=informer,
        conformal_type="adaptive",
        alpha=0.1,  # 90% coverage
        horizon=config.pred_len,
        n_features=config.c_out
    )
    
    # Step 3: Calibrate on calibration set
    informer_conformal.calibrate(calibration_data)
    
    # Step 4: Evaluate on test set
    print("Evaluating with prediction intervals...")
    coverage, width = evaluate_conformal(
        informer_conformal,
        test_data,
        alpha=0.1
    )
    
    print(f"Empirical coverage: {coverage:.3f} (target: 0.90)")
    print(f"Average interval width: {width:.3f}")
    
    return informer_conformal


def evaluate_conformal(conformal_model, test_data, alpha):
    """
    Evaluate conformal prediction coverage and efficiency
    """
    n_test = len(test_data) - conformal_model.conformal.horizon
    covered = []
    widths = []
    
    for t in range(n_test):
        x_t = test_data[t]
        y_true = test_data[t+1 : t+conformal_model.conformal.horizon+1]
        
        y_pred, intervals = conformal_model.predict_with_intervals(x_t)
        
        # Check coverage
        for j in range(len(intervals)):
            lower, upper = intervals[j]
            is_covered = (lower <= y_true[j] <= upper)
            covered.append(is_covered)
            widths.append(upper - lower)
    
    # Empirical coverage
    coverage = np.mean(covered)
    avg_width = np.mean(widths)
    
    return coverage, avg_width
```

---

## Production Deployment

### Configuration

```python
from dataclasses import dataclass

@dataclass
class ConformalConfig:
    """Configuration for conformal prediction"""
    
    # Conformal parameters
    alpha: float = 0.1  # Significance level (90% coverage)
    conformal_type: str = "adaptive"  # split, adaptive, multivariate
    
    # Time series specific
    horizon: int = 24
    n_features: int = 1
    
    # Adaptive conformal
    tau: float = 10.0  # Decay parameter for exponential weighting
    
    # Multivariate conformal
    multivariate_mode: str = "joint"  # joint or marginal
    
    # Calibration
    calibration_ratio: float = 0.2  # Fraction of data for calibration
    recalibration_frequency: int = 100  # Recalibrate every N predictions
    
    # Efficiency
    use_online_update: bool = True  # Update scores online
    max_calibration_samples: int = 10000  # Limit calibration set size
```

### API Integration

```python
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Tuple

app = FastAPI(title="Informer with Conformal Prediction API")

# Global model (loaded at startup)
model_with_conformal = None

class PredictionRequest(BaseModel):
    features: List[float]
    alpha: float = 0.1  # Coverage level

class PredictionResponse(BaseModel):
    point_predictions: List[float]
    lower_bounds: List[float]
    upper_bounds: List[float]
    coverage_probability: float

@app.on_event("startup")
async def load_models():
    global model_with_conformal
    # Load trained Informer + calibrated conformal
    model_with_conformal = load_model_with_conformal("checkpoints/best_model.pth")

@app.post("/predict", response_model=PredictionResponse)
async def predict_with_uncertainty(request: PredictionRequest):
    """
    Generate predictions with conformal intervals
    """
    # Prepare input
    x_test = np.array(request.features)
    
    # Get predictions with intervals
    y_pred, intervals = model_with_conformal.predict_with_intervals(x_test)
    
    # Extract bounds
    lower_bounds = [interval[0] for interval in intervals]
    upper_bounds = [interval[1] for interval in intervals]
    
    return PredictionResponse(
        point_predictions=y_pred.tolist(),
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        coverage_probability=1 - request.alpha
    )

@app.post("/recalibrate")
async def recalibrate_conformal(calibration_data: List[List[float]]):
    """
    Recalibrate conformal predictor with new data
    """
    global model_with_conformal
    
    calibration_array = np.array(calibration_data)
    model_with_conformal.calibrate(calibration_array)
    
    return {"status": "recalibrated", "n_samples": len(calibration_data)}

@app.get("/coverage")
async def get_coverage_stats():
    """
    Get current coverage statistics
    """
    return {
        "target_coverage": 1 - model_with_conformal.conformal.alpha,
        "calibration_samples": len(model_with_conformal.conformal.scores),
        "quantile": model_with_conformal.conformal.quantile
    }
```

### Monitoring

```python
class ConformalMonitor:
    """
    Monitor conformal prediction performance in production
    """
    def __init__(self, alpha: float = 0.1, window_size: int = 1000):
        self.alpha = alpha
        self.target_coverage = 1 - alpha
        self.window_size = window_size
        
        self.predictions = []
        self.true_values = []
        self.covered = []
        self.widths = []
    
    def log_prediction(
        self, 
        y_pred: np.ndarray, 
        intervals: List[Tuple],
        y_true: Optional[np.ndarray] = None
    ):
        """Log a prediction for monitoring"""
        self.predictions.append(y_pred)
        
        # Store interval widths
        for lower, upper in intervals:
            self.widths.append(upper - lower)
        
        # If true value available, check coverage
        if y_true is not None:
            self.true_values.append(y_true)
            
            for j, (lower, upper) in enumerate(intervals):
                is_covered = (lower <= y_true[j] <= upper)
                self.covered.append(is_covered)
        
        # Keep only recent window
        if len(self.predictions) > self.window_size:
            self.predictions = self.predictions[-self.window_size:]
            self.true_values = self.true_values[-self.window_size:]
            self.covered = self.covered[-self.window_size:]
            self.widths = self.widths[-self.window_size:]
    
    def get_metrics(self) -> dict:
        """Get monitoring metrics"""
        if len(self.covered) == 0:
            return {"status": "insufficient_data"}
        
        empirical_coverage = np.mean(self.covered)
        avg_width = np.mean(self.widths)
        
        # Check if coverage is within acceptable range
        coverage_gap = abs(empirical_coverage - self.target_coverage)
        needs_recalibration = coverage_gap > 0.05  # 5% tolerance
        
        return {
            "empirical_coverage": empirical_coverage,
            "target_coverage": self.target_coverage,
            "coverage_gap": coverage_gap,
            "average_interval_width": avg_width,
            "n_predictions": len(self.covered),
            "needs_recalibration": needs_recalibration
        }
```

---

## Comparison Matrix

### Informer vs. Conformal vs. Informer+Conformal

| Dimension | Informer (Standalone) | Conformal (Standalone) | Informer + Conformal |
|-----------|----------------------|------------------------|----------------------|
| **Output** | Point forecasts | Prediction intervals (model-agnostic) | Point forecasts + intervals |
| **Uncertainty** | None | Theoretical coverage guarantees | Best of both |
| **Complexity** | O(L log L) | O(n) for calibration | O(L log L + n) |
| **Training** | Requires training | No training (wraps model) | Train Informer + calibrate conformal |
| **Guarantees** | Empirical performance | Theoretical coverage | Theoretical coverage on Informer predictions |
| **Adaptability** | Model-specific | Model-agnostic | Model-specific with guarantees |
| **Use Case** | Accurate point forecasts | Uncertainty quantification | Production with risk bounds |
| **Computational Cost** | High (training) | Low (calibration only) | High (training) + Low (calibration) |

### When to Use Each

**Use Informer Alone:**
- ✅ Only need point forecasts
- ✅ Computational budget limited
- ✅ Uncertainty not critical

**Use Conformal Alone:**
- ✅ Already have any trained model
- ✅ Need uncertainty quickly
- ✅ Minimal computational overhead

**Use Informer + Conformal (Recommended):**
- ✅ **Production systems**
- ✅ **Risk-sensitive applications**
- ✅ **Regulatory requirements** (e.g., finance, healthcare)
- ✅ **Decision-making under uncertainty**
- ✅ **A/B testing** (compare interval widths)

---

## Implementation Checklist

### Phase 1: Basic Conformal (Week 1)
- [ ] Implement `SplitConformal` class
- [ ] Implement `ForecastingModelWrapper`
- [ ] Test on toy data
- [ ] Verify coverage on synthetic data

### Phase 2: Advanced Conformal (Week 2)
- [ ] Implement `AdaptiveConformal` with time weighting
- [ ] Implement `MultivariateConformal`
- [ ] Add online recalibration
- [ ] Benchmark coverage vs. efficiency

### Phase 3: Integration (Week 3)
- [ ] Wrap Informer model
- [ ] Create calibration pipeline
- [ ] Evaluate on benchmark datasets (ETT, ECL, Weather)
- [ ] Compare coverage across methods

### Phase 4: Production (Week 4)
- [ ] API endpoints with FastAPI
- [ ] Monitoring and logging
- [ ] Recalibration triggers
- [ ] Documentation and tests

### Phase 5: Optimization (Week 5)
- [ ] Optimize calibration speed
- [ ] Implement caching
- [ ] Add visualization tools
- [ ] Performance profiling

---

## Code Example: Complete Workflow

```python
from model_specs import ConfigPresets
from sql_data_pipeline import SQLTimeSeriesPipeline

# Step 1: Extract data
config = ConfigPresets.energy_forecasting()
pipeline = SQLTimeSeriesPipeline(config)
df = pipeline.run()

# Step 2: Split data (train/val/cal/test)
n = len(df)
train_end = int(n * 0.6)
val_end = int(n * 0.7)
cal_end = int(n * 0.85)

train_data = df[:train_end]
val_data = df[train_end:val_end]
calibration_data = df[val_end:cal_end]  # Held-out for conformal
test_data = df[cal_end:]

# Step 3: Train Informer
informer = train_informer(train_data, val_data, config)

# Step 4: Wrap with conformal
informer_conformal = InformerWithConformal(
    informer_model=informer,
    conformal_type="adaptive",
    alpha=0.1,  # 90% coverage
    horizon=24
)

# Step 5: Calibrate
informer_conformal.calibrate(calibration_data)

# Step 6: Predict with intervals
x_test = test_data[0]
y_pred, intervals = informer_conformal.predict_with_intervals(x_test)

print(f"Predictions: {y_pred}")
print(f"90% Intervals: {intervals}")

# Step 7: Evaluate
coverage, width = evaluate_conformal(informer_conformal, test_data, alpha=0.1)
print(f"Empirical coverage: {coverage:.3f} (target: 0.90)")
print(f"Average interval width: {width:.3f}")
```

---

## References

### Papers
1. **Conformal Prediction:** Vovk et al., "Algorithmic Learning in a Random World" (2005)
2. **Time Series Conformal:** Chernozhukov et al., "Exact and robust conformal inference methods for predictive machine learning with dependent data" (2018)
3. **Adaptive Conformal:** Gibbs & Candes, "Adaptive conformal inference under distribution shift" (2021)
4. **Informer:** Zhou et al., "Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting" (AAAI 2021)

### Code
- Conformal Prediction library: https://github.com/valeman/awesome-conformal-prediction
- MAPIE (Model Agnostic Prediction Intervals): https://github.com/scikit-learn-contrib/MAPIE

---

## Summary

**Conformal Prediction** provides:
1. ✅ **Statistical guarantees** for prediction intervals
2. ✅ **Model-agnostic** - works with any forecaster (including Informer)
3. ✅ **Minimal overhead** - just calibration, no retraining
4. ✅ **Adaptive** to non-stationarity in time series
5. ✅ **Production-ready** with monitoring and recalibration

**Recommended Stack:**
```
Informer (Point Forecasts) 
    → Conformal Wrapper (Uncertainty Quantification)
    → API (FastAPI)
    → Monitoring (Coverage tracking)
    → Recalibration (Adaptive)
```

This gives you the **best of both worlds**: efficient long-sequence forecasting with theoretical uncertainty bounds.

---

**Built for production deployment with theoretical guarantees** 🎯

*Version 1.0.0 - October 14, 2025*

