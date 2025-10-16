# Data Engineering for Conformal Prediction

**Optimized Data Pipeline for Uncertainty Quantification**

**Date:** October 14, 2025  
**Objective:** Production-ready data preparation for conformal prediction with time series forecasting

---

## Executive Summary

Data engineering for conformal prediction requires careful handling of:
1. **Calibration Set Isolation** - Never use calibration data for training
2. **Temporal Ordering** - Maintain strict chronological splits
3. **Score Computation** - Efficient nonconformity score calculation
4. **Adaptive Weighting** - Handle non-stationarity
5. **Multi-target Support** - Multivariate time series handling

**Critical Principle:** Calibration set must be completely independent from training to ensure valid coverage guarantees.

---

## Table of Contents

1. [Data Splitting Strategy](#data-splitting-strategy)
2. [Calibration Data Preparation](#calibration-data-preparation)
3. [Score Computation Pipeline](#score-computation-pipeline)
4. [Adaptive Weighting](#adaptive-weighting)
5. [Multivariate Handling](#multivariate-handling)
6. [Online Recalibration](#online-recalibration)
7. [Implementation Code](#implementation-code)

---

## Data Splitting Strategy

### Four-Way Split for Conformal Prediction

```python
class ConformalDataSplitter:
    """
    Proper data splitting for conformal prediction
    """
    def __init__(
        self,
        train_ratio: float = 0.60,
        val_ratio: float = 0.10,
        calibration_ratio: float = 0.15,
        test_ratio: float = 0.15
    ):
        """
        Args:
            train_ratio: For training base forecasting model
            val_ratio: For validating/tuning base model
            calibration_ratio: For fitting conformal predictor (CRITICAL)
            test_ratio: For final evaluation
        """
        assert abs(train_ratio + val_ratio + calibration_ratio + test_ratio - 1.0) < 1e-6
        
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.calibration_ratio = calibration_ratio
        self.test_ratio = test_ratio
    
    def split(self, df: pd.DataFrame, time_col: str = 'timestamp'):
        """
        Chronological split (NEVER shuffle!)
        
        Returns:
            train_df, val_df, calibration_df, test_df
        """
        # Sort by time
        df = df.sort_values(time_col).reset_index(drop=True)
        
        n = len(df)
        
        # Calculate split points
        train_end = int(n * self.train_ratio)
        val_end = int(n * (self.train_ratio + self.val_ratio))
        cal_end = int(n * (self.train_ratio + self.val_ratio + self.calibration_ratio))
        
        # Split
        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()
        calibration_df = df.iloc[val_end:cal_end].copy()
        test_df = df.iloc[cal_end:].copy()
        
        # Validate splits
        assert len(calibration_df) > 0, "Calibration set is empty!"
        assert calibration_df[time_col].min() > val_df[time_col].max(), "Temporal leak!"
        assert test_df[time_col].min() > calibration_df[time_col].max(), "Temporal leak!"
        
        print(f"Data Split:")
        print(f"  Train:       {len(train_df):,} samples ({self.train_ratio:.1%})")
        print(f"  Validation:  {len(val_df):,} samples ({self.val_ratio:.1%})")
        print(f"  Calibration: {len(calibration_df):,} samples ({self.calibration_ratio:.1%}) ← CRITICAL")
        print(f"  Test:        {len(test_df):,} samples ({self.test_ratio:.1%})")
        
        return train_df, val_df, calibration_df, test_df
```

### Critical Validation

```python
def validate_data_split(train_df, val_df, cal_df, test_df, time_col='timestamp'):
    """
    Ensure no data leakage between splits
    """
    checks = []
    
    # Check 1: No temporal overlap
    train_max = train_df[time_col].max()
    val_min = val_df[time_col].min()
    val_max = val_df[time_col].max()
    cal_min = cal_df[time_col].min()
    cal_max = cal_df[time_col].max()
    test_min = test_df[time_col].min()
    
    checks.append(("Train < Val", train_max < val_min))
    checks.append(("Val < Calibration", val_max < cal_min))
    checks.append(("Calibration < Test", cal_max < test_min))
    
    # Check 2: Sufficient calibration samples
    min_calibration = 50  # Minimum for stable quantile
    checks.append(("Calibration size", len(cal_df) >= min_calibration))
    
    # Check 3: No duplicate timestamps
    checks.append(("No duplicate timestamps", 
                   train_df[time_col].is_unique and 
                   val_df[time_col].is_unique and
                   cal_df[time_col].is_unique and
                   test_df[time_col].is_unique))
    
    # Report
    print("\nData Split Validation:")
    for check_name, passed in checks:
        status = "✓" if passed else "✗"
        print(f"  {status} {check_name}")
    
    all_passed = all(passed for _, passed in checks)
    
    if not all_passed:
        raise ValueError("Data split validation failed!")
    
    return all_passed
```

---

## Calibration Data Preparation

### Calibration Set Requirements

```python
class CalibrationSetBuilder:
    """
    Prepare calibration set for conformal prediction
    """
    def __init__(
        self,
        horizon: int = 24,
        min_samples: int = 100,
        max_samples: int = 5000
    ):
        """
        Args:
            horizon: Forecast horizon (must match model)
            min_samples: Minimum calibration samples
            max_samples: Maximum (for computational efficiency)
        """
        self.horizon = horizon
        self.min_samples = min_samples
        self.max_samples = max_samples
    
    def prepare(
        self,
        df: pd.DataFrame,
        model,
        target_col: str,
        feature_cols: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare calibration data: inputs and multi-step outputs
        
        Returns:
            X_cal: (n_samples, n_features) calibration inputs
            Y_cal: (n_samples, horizon, n_targets) calibration targets
        """
        n = len(df)
        n_valid = n - self.horizon
        
        if n_valid < self.min_samples:
            raise ValueError(
                f"Insufficient calibration data: {n_valid} < {self.min_samples}"
            )
        
        # Limit to max_samples for efficiency
        if n_valid > self.max_samples:
            # Take most recent samples
            start_idx = n_valid - self.max_samples
        else:
            start_idx = 0
        
        X_cal = []
        Y_cal = []
        
        for t in range(start_idx, n_valid):
            # Input at time t
            x_t = df[feature_cols].iloc[t].values
            
            # Target: next h steps
            y_t = df[target_col].iloc[t+1 : t+self.horizon+1].values
            
            if len(y_t) == self.horizon:  # Ensure full horizon available
                X_cal.append(x_t)
                Y_cal.append(y_t)
        
        X_cal = np.array(X_cal)
        Y_cal = np.array(Y_cal)
        
        print(f"Calibration Set Prepared:")
        print(f"  Samples: {len(X_cal):,}")
        print(f"  Horizon: {self.horizon}")
        print(f"  Features: {X_cal.shape[1]}")
        
        return X_cal, Y_cal
```

### Data Quality for Calibration

```python
def validate_calibration_data(X_cal, Y_cal):
    """
    Validate calibration data quality
    """
    issues = []
    
    # Check for NaN
    if np.isnan(X_cal).any():
        n_nan = np.isnan(X_cal).sum()
        issues.append(f"X_cal contains {n_nan} NaN values")
    
    if np.isnan(Y_cal).any():
        n_nan = np.isnan(Y_cal).sum()
        issues.append(f"Y_cal contains {n_nan} NaN values")
    
    # Check for infinite values
    if np.isinf(X_cal).any():
        issues.append("X_cal contains infinite values")
    
    if np.isinf(Y_cal).any():
        issues.append("Y_cal contains infinite values")
    
    # Check variance
    if np.var(Y_cal) < 1e-10:
        issues.append("Y_cal has near-zero variance (constant?)")
    
    # Check sample size
    if len(X_cal) < 50:
        issues.append(f"Very small calibration set: {len(X_cal)} samples")
    
    if issues:
        print("⚠️  Calibration Data Issues:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("✓ Calibration data validation passed")
        return True
```

---

## Score Computation Pipeline

### Nonconformity Score Calculator

```python
class NonconformityScoreCalculator:
    """
    Compute nonconformity scores for conformal prediction
    """
    def __init__(self, score_type: str = "max_abs"):
        """
        Args:
            score_type: 
                - "max_abs": max absolute error over horizon
                - "mean_abs": mean absolute error
                - "weighted": time-weighted error
                - "quantile": quantile-based asymmetric score
        """
        self.score_type = score_type
    
    def compute_scores(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> np.ndarray:
        """
        Compute nonconformity scores
        
        Args:
            y_true: (n_samples, horizon) or (n_samples, horizon, n_features)
            y_pred: Same shape as y_true
        
        Returns:
            scores: (n_samples,) array of nonconformity scores
        """
        if y_true.shape != y_pred.shape:
            raise ValueError(f"Shape mismatch: {y_true.shape} vs {y_pred.shape}")
        
        errors = np.abs(y_true - y_pred)
        
        if self.score_type == "max_abs":
            # Max absolute error over all dimensions
            scores = np.max(errors, axis=tuple(range(1, errors.ndim)))
        
        elif self.score_type == "mean_abs":
            # Mean absolute error
            scores = np.mean(errors, axis=tuple(range(1, errors.ndim)))
        
        elif self.score_type == "weighted":
            # Time-weighted (later steps weighted more)
            horizon = errors.shape[1]
            weights = np.arange(1, horizon + 1) / horizon
            weights = weights.reshape(1, -1)
            if errors.ndim == 3:
                weights = weights.reshape(1, -1, 1)
            weighted_errors = errors * weights
            scores = np.max(weighted_errors, axis=tuple(range(1, errors.ndim)))
        
        elif self.score_type == "quantile":
            # Quantile loss (asymmetric)
            # For now, use absolute error (can extend)
            scores = np.max(errors, axis=tuple(range(1, errors.ndim)))
        
        else:
            raise ValueError(f"Unknown score_type: {self.score_type}")
        
        return scores
    
    def compute_scores_efficient(
        self,
        model,
        X_cal: np.ndarray,
        Y_cal: np.ndarray,
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Efficiently compute scores in batches
        """
        n_samples = len(X_cal)
        scores = []
        
        for i in range(0, n_samples, batch_size):
            batch_X = X_cal[i : i+batch_size]
            batch_Y_true = Y_cal[i : i+batch_size]
            
            # Get predictions
            batch_Y_pred = model.predict(batch_X)
            
            # Compute scores for batch
            batch_scores = self.compute_scores(batch_Y_true, batch_Y_pred)
            scores.extend(batch_scores)
        
        return np.array(scores)
```

### Quantile Computation

```python
class QuantileComputer:
    """
    Compute conformal quantiles with various methods
    """
    def __init__(self, alpha: float = 0.1):
        """
        Args:
            alpha: Significance level (1-alpha = target coverage)
        """
        self.alpha = alpha
    
    def compute_quantile(
        self,
        scores: np.ndarray,
        weights: Optional[np.ndarray] = None
    ) -> float:
        """
        Compute (1-alpha) quantile of scores
        
        Args:
            scores: Nonconformity scores
            weights: Optional weights for adaptive conformal
        
        Returns:
            quantile: Threshold for prediction intervals
        """
        if weights is None:
            # Standard quantile
            # Use ceiling for finite-sample guarantee
            n = len(scores)
            k = int(np.ceil((1 - self.alpha) * (n + 1)))
            k = min(k, n)  # Ensure within bounds
            
            sorted_scores = np.sort(scores)
            quantile = sorted_scores[k-1] if k > 0 else sorted_scores[0]
        
        else:
            # Weighted quantile
            quantile = self._weighted_quantile(scores, weights, 1 - self.alpha)
        
        return quantile
    
    def _weighted_quantile(
        self,
        scores: np.ndarray,
        weights: np.ndarray,
        q: float
    ) -> float:
        """Compute weighted quantile efficiently"""
        # Sort scores and weights together
        sorted_idx = np.argsort(scores)
        sorted_scores = scores[sorted_idx]
        sorted_weights = weights[sorted_idx]
        
        # Cumulative weights
        cum_weights = np.cumsum(sorted_weights)
        total_weight = cum_weights[-1]
        
        # Find quantile
        threshold = q * total_weight
        idx = np.searchsorted(cum_weights, threshold)
        
        return sorted_scores[min(idx, len(scores)-1)]
```

---

## Adaptive Weighting

### Time-Weighted Scores for Non-Stationarity

```python
class AdaptiveWeightCalculator:
    """
    Compute adaptive weights for non-stationary time series
    """
    def __init__(self, decay_type: str = "exponential", tau: float = None):
        """
        Args:
            decay_type: "exponential", "linear", or "polynomial"
            tau: Decay parameter (if None, set to n/5)
        """
        self.decay_type = decay_type
        self.tau = tau
    
    def compute_weights(
        self,
        n: int,
        current_time: Optional[int] = None
    ) -> np.ndarray:
        """
        Compute weights for n calibration samples
        
        Args:
            n: Number of calibration samples
            current_time: Current time index (defaults to n)
        
        Returns:
            weights: (n,) array with recent samples weighted more
        """
        if current_time is None:
            current_time = n
        
        if self.tau is None:
            tau = n / 5  # Default: moderate decay
        else:
            tau = self.tau
        
        # Time indices (0 to n-1)
        t = np.arange(n)
        
        if self.decay_type == "exponential":
            # w_t = exp(-λ(T-t)) where λ = 1/τ
            weights = np.exp(-(current_time - t) / tau)
        
        elif self.decay_type == "gaussian":
            # w_t = exp(-(T-t)²/(2τ²))
            weights = np.exp(-((current_time - t) ** 2) / (2 * tau ** 2))
        
        elif self.decay_type == "linear":
            # w_t = max(0, 1 - (T-t)/τ)
            weights = np.maximum(0, 1 - (current_time - t) / tau)
        
        elif self.decay_type == "polynomial":
            # w_t = (1 + (T-t))^(-γ)
            gamma = 1.0
            weights = (1 + (current_time - t)) ** (-gamma)
        
        else:
            raise ValueError(f"Unknown decay_type: {self.decay_type}")
        
        # Normalize weights
        weights = weights / weights.sum()
        
        return weights
```

### Adaptive Calibration

```python
class AdaptiveCalibrator:
    """
    Adaptive conformal calibration for non-stationary data
    """
    def __init__(
        self,
        alpha: float = 0.1,
        tau: float = None,
        recalibration_freq: int = 100
    ):
        """
        Args:
            alpha: Significance level
            tau: Decay parameter for weighting
            recalibration_freq: Recalibrate every N predictions
        """
        self.alpha = alpha
        self.tau = tau
        self.recalibration_freq = recalibration_freq
        
        self.weight_calculator = AdaptiveWeightCalculator(
            decay_type="exponential",
            tau=tau
        )
        self.quantile_computer = QuantileComputer(alpha)
        
        # State
        self.scores = []
        self.weights = None
        self.quantile = None
        self.n_predictions = 0
    
    def fit(self, scores: np.ndarray):
        """Initial calibration"""
        self.scores = list(scores)
        self.weights = self.weight_calculator.compute_weights(len(scores))
        self.quantile = self.quantile_computer.compute_quantile(
            np.array(self.scores),
            self.weights
        )
        self.n_predictions = 0
        
        print(f"Initial calibration: {len(self.scores)} scores, quantile={self.quantile:.4f}")
    
    def update(self, new_score: float):
        """Add new score and update if needed"""
        self.scores.append(new_score)
        self.n_predictions += 1
        
        # Recalibrate periodically
        if self.n_predictions % self.recalibration_freq == 0:
            self.recalibrate()
    
    def recalibrate(self):
        """Recompute quantile with updated weights"""
        n = len(self.scores)
        self.weights = self.weight_calculator.compute_weights(n)
        self.quantile = self.quantile_computer.compute_quantile(
            np.array(self.scores),
            self.weights
        )
        
        print(f"Recalibrated: {n} scores, new quantile={self.quantile:.4f}")
```

---

## Multivariate Handling

### Joint vs. Marginal Strategies

```python
class MultivariateConformalPreparer:
    """
    Prepare data for multivariate conformal prediction
    """
    def __init__(
        self,
        mode: str = "joint",  # "joint" or "marginal"
        alpha: float = 0.1,
        n_features: int = 1
    ):
        """
        Args:
            mode: Joint (single quantile) or marginal (per-feature quantiles)
            alpha: Significance level
            n_features: Number of output features
        """
        self.mode = mode
        self.alpha = alpha
        self.n_features = n_features
        
        if mode == "marginal":
            # Bonferroni correction for simultaneous coverage
            self.alpha_corrected = alpha / n_features
        else:
            self.alpha_corrected = alpha
    
    def compute_multivariate_scores(
        self,
        Y_true: np.ndarray,
        Y_pred: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Compute scores for multivariate data
        
        Args:
            Y_true: (n_samples, horizon, n_features)
            Y_pred: (n_samples, horizon, n_features)
        
        Returns:
            scores: Dict with 'joint' and/or per-feature scores
        """
        scores = {}
        
        if self.mode == "joint":
            # Single score: max over all dimensions
            errors = np.abs(Y_true - Y_pred)
            scores['joint'] = np.max(errors, axis=(1, 2))
        
        elif self.mode == "marginal":
            # Per-feature scores
            for k in range(self.n_features):
                errors_k = np.abs(Y_true[:, :, k] - Y_pred[:, :, k])
                scores[f'feature_{k}'] = np.max(errors_k, axis=1)
        
        return scores
    
    def compute_quantiles(
        self,
        scores: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """
        Compute quantiles for multivariate scores
        """
        quantiles = {}
        
        if self.mode == "joint":
            quantile_computer = QuantileComputer(self.alpha)
            quantiles['joint'] = quantile_computer.compute_quantile(scores['joint'])
        
        elif self.mode == "marginal":
            quantile_computer = QuantileComputer(self.alpha_corrected)
            for k in range(self.n_features):
                quantiles[f'feature_{k}'] = quantile_computer.compute_quantile(
                    scores[f'feature_{k}']
                )
        
        return quantiles
```

---

## Online Recalibration

### Rolling Window Recalibration

```python
class OnlineRecalibrator:
    """
    Online recalibration for production deployment
    """
    def __init__(
        self,
        alpha: float = 0.1,
        window_size: int = 1000,
        min_samples: int = 50,
        recalibration_trigger: str = "periodic"  # "periodic" or "coverage_based"
    ):
        """
        Args:
            alpha: Significance level
            window_size: Maximum calibration samples to keep
            min_samples: Minimum before first calibration
            recalibration_trigger: When to recalibrate
        """
        self.alpha = alpha
        self.window_size = window_size
        self.min_samples = min_samples
        self.recalibration_trigger = recalibration_trigger
        
        # State
        self.scores = []
        self.covered = []  # Track coverage
        self.quantile = None
        self.n_updates = 0
    
    def add_observation(
        self,
        score: float,
        was_covered: Optional[bool] = None
    ):
        """
        Add new observation
        
        Args:
            score: Nonconformity score for this observation
            was_covered: Whether true value was in interval
        """
        self.scores.append(score)
        
        if was_covered is not None:
            self.covered.append(was_covered)
        
        # Maintain rolling window
        if len(self.scores) > self.window_size:
            self.scores = self.scores[-self.window_size:]
            if self.covered:
                self.covered = self.covered[-self.window_size:]
        
        self.n_updates += 1
    
    def should_recalibrate(self) -> bool:
        """Check if recalibration is needed"""
        if len(self.scores) < self.min_samples:
            return False
        
        if self.recalibration_trigger == "periodic":
            # Recalibrate every 100 updates
            return self.n_updates % 100 == 0
        
        elif self.recalibration_trigger == "coverage_based":
            # Recalibrate if coverage drifts
            if len(self.covered) >= 50:
                recent_coverage = np.mean(self.covered[-50:])
                target_coverage = 1 - self.alpha
                
                # If coverage drops below target - 5%, recalibrate
                return recent_coverage < (target_coverage - 0.05)
        
        return False
    
    def recalibrate(self) -> float:
        """
        Recompute quantile
        
        Returns:
            New quantile value
        """
        if len(self.scores) < self.min_samples:
            raise ValueError(f"Insufficient samples: {len(self.scores)} < {self.min_samples}")
        
        quantile_computer = QuantileComputer(self.alpha)
        self.quantile = quantile_computer.compute_quantile(np.array(self.scores))
        
        # Reset update counter
        self.n_updates = 0
        
        return self.quantile
    
    def get_coverage_stats(self) -> Dict[str, float]:
        """Get current coverage statistics"""
        if not self.covered:
            return {"status": "no_coverage_data"}
        
        empirical_coverage = np.mean(self.covered)
        target_coverage = 1 - self.alpha
        coverage_gap = abs(empirical_coverage - target_coverage)
        
        return {
            "empirical_coverage": empirical_coverage,
            "target_coverage": target_coverage,
            "coverage_gap": coverage_gap,
            "n_observations": len(self.covered),
            "needs_recalibration": coverage_gap > 0.05
        }
```

---

## Implementation Code

### Complete Pipeline

```python
class ConformalDataPipeline:
    """
    Complete data pipeline for conformal prediction
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Initialize components
        self.splitter = ConformalDataSplitter(
            train_ratio=config.get('train_ratio', 0.6),
            val_ratio=config.get('val_ratio', 0.1),
            calibration_ratio=config.get('calibration_ratio', 0.15),
            test_ratio=config.get('test_ratio', 0.15)
        )
        
        self.score_calculator = NonconformityScoreCalculator(
            score_type=config.get('score_type', 'max_abs')
        )
        
        self.quantile_computer = QuantileComputer(
            alpha=config.get('alpha', 0.1)
        )
        
        if config.get('adaptive', False):
            self.calibrator = AdaptiveCalibrator(
                alpha=config.get('alpha', 0.1),
                tau=config.get('tau', None),
                recalibration_freq=config.get('recalibration_freq', 100)
            )
        else:
            self.calibrator = None
    
    def run(
        self,
        df: pd.DataFrame,
        model,
        target_col: str,
        feature_cols: List[str],
        time_col: str = 'timestamp'
    ) -> Dict[str, Any]:
        """
        Run complete conformal data pipeline
        
        Returns:
            Dictionary with all prepared data and calibration info
        """
        print("=" * 80)
        print("CONFORMAL DATA PIPELINE")
        print("=" * 80)
        
        # Step 1: Split data
        print("\n[1/5] Splitting data...")
        train_df, val_df, cal_df, test_df = self.splitter.split(df, time_col)
        
        # Step 2: Validate split
        print("\n[2/5] Validating split...")
        validate_data_split(train_df, val_df, cal_df, test_df, time_col)
        
        # Step 3: Prepare calibration set
        print("\n[3/5] Preparing calibration set...")
        cal_builder = CalibrationSetBuilder(
            horizon=self.config['horizon'],
            min_samples=self.config.get('min_cal_samples', 100)
        )
        X_cal, Y_cal = cal_builder.prepare(
            cal_df, model, target_col, feature_cols
        )
        
        # Step 4: Compute scores
        print("\n[4/5] Computing nonconformity scores...")
        scores = self.score_calculator.compute_scores_efficient(
            model, X_cal, Y_cal
        )
        
        print(f"  Scores computed: {len(scores)}")
        print(f"  Score range: [{scores.min():.4f}, {scores.max():.4f}]")
        print(f"  Score mean: {scores.mean():.4f}")
        
        # Step 5: Compute quantile
        print("\n[5/5] Computing conformal quantile...")
        if self.calibrator is not None:
            self.calibrator.fit(scores)
            quantile = self.calibrator.quantile
        else:
            quantile = self.quantile_computer.compute_quantile(scores)
        
        print(f"  Quantile (α={self.config['alpha']}): {quantile:.4f}")
        print(f"  Target coverage: {1 - self.config['alpha']:.1%}")
        
        print("\n" + "=" * 80)
        print("PIPELINE COMPLETE")
        print("=" * 80)
        
        return {
            'train_df': train_df,
            'val_df': val_df,
            'calibration_df': cal_df,
            'test_df': test_df,
            'X_cal': X_cal,
            'Y_cal': Y_cal,
            'scores': scores,
            'quantile': quantile,
            'calibrator': self.calibrator
        }
```

### Usage Example

```python
# Configuration
config = {
    'alpha': 0.1,  # 90% coverage
    'horizon': 24,
    'score_type': 'max_abs',
    'adaptive': True,
    'tau': None,  # Will be set to n/5
    'recalibration_freq': 100,
    'train_ratio': 0.6,
    'val_ratio': 0.1,
    'calibration_ratio': 0.15,
    'test_ratio': 0.15
}

# Run pipeline
pipeline = ConformalDataPipeline(config)

result = pipeline.run(
    df=your_dataframe,
    model=trained_forecasting_model,
    target_col='target',
    feature_cols=['feature1', 'feature2', 'feature3'],
    time_col='timestamp'
)

# Access results
quantile = result['quantile']
calibrator = result['calibrator']
test_df = result['test_df']
```

---

## Data Engineering Checklist

### Pre-Processing
- [ ] Load time series data
- [ ] Validate timestamps (no duplicates, monotonic)
- [ ] Handle missing values
- [ ] Remove outliers (carefully!)
- [ ] Normalize/scale features

### Data Splitting
- [ ] Sort data chronologically
- [ ] Four-way split (train/val/cal/test)
- [ ] Validate no temporal overlap
- [ ] Ensure sufficient calibration samples (≥100)
- [ ] Save split indices for reproducibility

### Calibration Preparation
- [ ] Extract multi-step targets from calibration set
- [ ] Validate calibration data quality
- [ ] Check for NaN/inf values
- [ ] Verify variance is non-zero
- [ ] Match model input/output formats

### Score Computation
- [ ] Choose appropriate score function
- [ ] Compute scores efficiently (batched)
- [ ] Validate score distribution
- [ ] Check for outlier scores
- [ ] Store scores for recalibration

### Quantile Computation
- [ ] Select significance level α
- [ ] Compute quantile (standard or weighted)
- [ ] Validate quantile is reasonable
- [ ] Store quantile for predictions
- [ ] Document quantile value

### Online Operations
- [ ] Implement score logging
- [ ] Set up recalibration triggers
- [ ] Monitor coverage over time
- [ ] Update quantile periodically
- [ ] Track calibration quality metrics

---

## Best Practices

1. **Never Mix Calibration with Training**
   - Calibration set must be independent
   - Use strict chronological split
   - No overlap between sets

2. **Maintain Temporal Order**
   - Never shuffle time series data
   - Respect causality
   - Validate split ordering

3. **Sufficient Calibration Samples**
   - Minimum: 50-100 samples
   - Recommended: 500-1000 samples
   - More samples → stable quantile

4. **Adaptive Weighting for Drift**
   - Use exponential decay for non-stationarity
   - Tune τ parameter via validation
   - Monitor coverage over time

5. **Recalibrate Periodically**
   - Fixed schedule or coverage-based
   - Keep rolling window of scores
   - Update every 100-1000 predictions

6. **Validate Everything**
   - Check data quality at each step
   - Monitor coverage empirically
   - Alert on coverage drift

---

## Troubleshooting

**Problem: Coverage below target**
- Solution: Check for data leakage, increase calibration set, recalibrate

**Problem: Very wide intervals**
- Solution: Check model quality, try different score function, increase training data

**Problem: Unstable quantile**
- Solution: Increase calibration samples, use adaptive weighting, smooth scores

**Problem: Coverage drifts over time**
- Solution: Enable adaptive weighting, increase recalibration frequency, check for distribution shift

---

## Summary

**Critical Requirements:**
1. ✅ Independent calibration set (15% of data)
2. ✅ Strict chronological splits (no shuffling)
3. ✅ Sufficient samples (≥100 for calibration)
4. ✅ Proper score computation (efficient, batched)
5. ✅ Periodic recalibration (every 100-1000 predictions)

**Data Flow:**
```
Raw Data → Split (train/val/cal/test) → 
→ Train Model → Compute Cal Scores → 
→ Compute Quantile → Deploy with Intervals → 
→ Monitor Coverage → Recalibrate
```

**Key Metrics:**
- Empirical coverage (target: 1-α)
- Interval width (minimize while maintaining coverage)
- Recalibration frequency (balance stability vs. adaptation)

---

**Built for production conformal prediction** 🎯

*Version 1.0.0 - October 14, 2025*

