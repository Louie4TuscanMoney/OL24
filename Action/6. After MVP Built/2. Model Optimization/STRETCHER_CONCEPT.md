# Stretcher - Model Optimization Framework

**Purpose:** Deep introspection and optimization of ML architecture  
**Concept:** "Stretch out" the model to see every component, then optimize  
**Timeline:** Post-MVP (Month 2 after live data collection)  
**Priority:** OPTIMIZATION (data-driven, not premature)

---

## 🎯 What is "Stretcher"?

**Stretcher = Stretching out the black box to see inside**

```
Current MVP Model (Black Box):
┌─────────────────────────────────┐
│    Input: 18-minute pattern      │
│            ↓                     │
│         [ MODEL ]                │
│            ↓                     │
│    Output: +15.1 [+11.3, +18.9] │
└─────────────────────────────────┘
MAE: 5.39 (Good enough for MVP!)


Stretcher Model (Transparent):
┌─────────────────────────────────────────────────────────────┐
│  Input: 18-minute pattern                                    │
│      ↓                                                       │
│  [Feature Engineering Layer]                                │
│    • Raw differentials                                       │
│    • Moving averages (3, 5, 7 minutes)                      │
│    • Momentum indicators                                     │
│    • Volatility measures                                     │
│    • Streak detection                                        │
│      ↓                                                       │
│  [Dejavu Branch]              [LSTM Branch]                 │
│    K=500 neighbors              64 hidden units             │
│    Euclidean distance           2 layers                     │
│    Median aggregation           Dropout 0.1                  │
│    Weight: 40%                  Weight: 60%                  │
│      ↓                              ↓                        │
│  [Ensemble Layer]                                           │
│    Weighted combination                                      │
│    Uncertainty quantification                                │
│      ↓                                                       │
│  [Conformal Wrapper]                                        │
│    Calibration set adjustment                                │
│    95% coverage guarantee                                    │
│      ↓                                                       │
│  Output: +15.1 [+11.3, +18.9]                               │
│                                                              │
│  EVERY COMPONENT VISIBLE AND OPTIMIZABLE                    │
└─────────────────────────────────────────────────────────────┘
```

**Goal:** Understand WHAT to optimize based on REAL performance data

---

## Why "Stretcher"?

**The name comes from:**
- Stretching a rubber band to see the internal fibers
- Pulling apart layers to inspect each one
- "Stretching out" the model architecture

**Philosophy:** Can't optimize what you can't see

---

## The Jungle (Custom Neural Architecture)

**"Jungle" = Dense network of custom optimizations**

**Concept:**
```
Traditional LSTM (What We Have):
  Input → LSTM(64) → LSTM(64) → Output

Jungle (What We Could Build):
  Input → Feature Extraction → 
          ├─ Attention Layer
          ├─ Temporal Conv1D
          ├─ Bi-LSTM
          ├─ GRU cells
          ├─ Skip connections
          └─ Multi-head attention
              ↓
          Fusion Layer
              ↓
          Output

Like a jungle: Dense, interconnected, optimized for specific task
```

---

## What to Optimize (Data-Driven)

**After 2-4 weeks of live betting, we'll have data on:**

### 1. Model Performance by Game Type
```python
# Analyze live performance
close_games_mae = 6.2  # Model struggles with close games
blowout_games_mae = 3.8  # Model excels with blowouts

# Optimization opportunity:
# Train separate model for close games (Q2 score diff < 5)
# Or add "game closeness" as input feature
```

### 2. Feature Importance
```python
# Which features drive predictions?
feature_importance = {
    'minute_17_diff': 0.25,  # Most important
    'minute_16_diff': 0.18,
    'minute_15_diff': 0.12,
    'minute_14_diff': 0.08,
    ...
    'minute_1_diff': 0.01    # Least important
}

# Optimization opportunity:
# Weight recent minutes higher
# Or use attention mechanism to learn importance
```

### 3. Ensemble Weights
```python
# Current: 40% Dejavu, 60% LSTM
# After live data:
dejavu_accuracy_by_scenario = {
    'home_team_leading': 5.2 MAE,
    'away_team_leading': 7.1 MAE,
    'close_game': 6.8 MAE
}

lstm_accuracy_by_scenario = {
    'home_team_leading': 4.9 MAE,
    'away_team_leading': 5.3 MAE,
    'close_game': 5.1 MAE
}

# Optimization opportunity:
# Dynamic ensemble weights based on game state
# Home leading → 30% Dejavu, 70% LSTM
# Away leading → 45% Dejavu, 55% LSTM
```

### 4. Conformal Calibration
```python
# Current: 94.6% coverage (target 95%)
# After live data:
actual_coverage_by_confidence = {
    'tight_intervals': 0.92,  # Underconfident
    'medium_intervals': 0.95,  # Perfect
    'wide_intervals': 0.98   # Overconfident
}

# Optimization opportunity:
# Adjust quantile based on interval width
# Tighten intervals for high-confidence predictions
```

---

## Stretcher Architecture

### Module 1: Model Introspector

**File:** `model_introspector.py`

```python
"""
Model Introspector
Analyze every component of ensemble model
"""

import numpy as np
from typing import Dict, List

class ModelIntrospector:
    """
    Stretch out the model to inspect every layer
    """
    
    def __init__(self, dejavu_model, lstm_model, conformal_wrapper):
        self.dejavu = dejavu_model
        self.lstm = lstm_model
        self.conformal = conformal_wrapper
    
    def analyze_feature_importance(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Which input features matter most?
        
        Method: Permutation importance
        """
        baseline_mae = self.calculate_mae(X, y)
        importances = {}
        
        for i in range(X.shape[1]):
            # Shuffle feature i
            X_shuffled = X.copy()
            np.random.shuffle(X_shuffled[:, i])
            
            # Calculate MAE with shuffled feature
            shuffled_mae = self.calculate_mae(X_shuffled, y)
            
            # Importance = increase in error
            importances[f'minute_{i+1}'] = shuffled_mae - baseline_mae
        
        return importances
    
    def analyze_dejavu_neighbors(self, pattern: np.ndarray) -> Dict:
        """
        Which historical patterns does Dejavu use?
        """
        distances = self.dejavu.calculate_distances(pattern)
        top_k_indices = np.argsort(distances)[:500]
        
        return {
            'top_neighbors': top_k_indices,
            'distance_distribution': distances[top_k_indices],
            'neighbor_outcomes': self.dejavu.historical_outcomes[top_k_indices]
        }
    
    def analyze_lstm_attention(self, pattern: np.ndarray) -> Dict:
        """
        Which timesteps does LSTM focus on?
        
        Method: Gradient-based attention
        """
        # Calculate gradients w.r.t. each input timestep
        # Higher gradient = more important
        
        return {
            'attention_weights': [...],  # 18 values (one per minute)
            'critical_minutes': [16, 17, 18],  # Most important
            'ignored_minutes': [1, 2, 3]  # Least important
        }
    
    def analyze_ensemble_contribution(self, X: np.ndarray) -> Dict:
        """
        How much does each model contribute to final prediction?
        """
        dejavu_preds = self.dejavu.predict(X)
        lstm_preds = self.lstm.predict(X)
        ensemble_preds = 0.4 * dejavu_preds + 0.6 * lstm_preds
        
        # Calculate correlation
        dejavu_contrib = np.corrcoef(dejavu_preds, ensemble_preds)[0, 1]
        lstm_contrib = np.corrcoef(lstm_preds, ensemble_preds)[0, 1]
        
        return {
            'dejavu_influence': dejavu_contrib,
            'lstm_influence': lstm_contrib,
            'dejavu_unique_value': dejavu_contrib - lstm_contrib,
            'optimal_weights_estimate': self.estimate_optimal_weights(X)
        }
    
    def analyze_conformal_efficiency(self) -> Dict:
        """
        Is conformal prediction well-calibrated?
        """
        return {
            'target_coverage': 0.95,
            'actual_coverage': 0.946,
            'interval_widths': [...],
            'calibration_by_confidence': {...},
            'opportunities_for_tightening': [...]
        }
```

---

### Module 2: Custom Architecture Builder

**File:** `jungle_architect.py`

```python
"""
Jungle Architect
Build custom neural architecture based on introspection results
"""

import torch
import torch.nn as nn

class AttentionLSTM(nn.Module):
    """
    LSTM with attention mechanism
    Learns which timesteps matter most
    """
    
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super().__init__()
        
        # Bi-directional LSTM (forward + backward)
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1
        )
        
        # Attention layer
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        # Output layer
        self.fc = nn.Linear(hidden_size * 2, 7)  # 7-step forecast
    
    def forward(self, x):
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Calculate attention weights
        attention_weights = torch.softmax(
            self.attention(lstm_out).squeeze(-1),
            dim=1
        )
        
        # Weighted sum of LSTM outputs
        context = torch.sum(
            lstm_out * attention_weights.unsqueeze(-1),
            dim=1
        )
        
        # Final prediction
        output = self.fc(context)
        
        return output, attention_weights


class TemporalConvLSTM(nn.Module):
    """
    Combines temporal convolutions with LSTM
    Captures both local patterns and long-range dependencies
    """
    
    def __init__(self):
        super().__init__()
        
        # Temporal convolutions (capture local patterns)
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        
        # LSTM (capture long-range dependencies)
        self.lstm = nn.LSTM(64, 128, 2, batch_first=True)
        
        # Attention
        self.attention = nn.MultiheadAttention(128, num_heads=4)
        
        # Output
        self.fc = nn.Linear(128, 7)
    
    def forward(self, x):
        # Conv layers
        x = x.transpose(1, 2)  # (batch, channels, seq)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.transpose(1, 2)  # (batch, seq, channels)
        
        # LSTM
        lstm_out, _ = self.lstm(x)
        
        # Self-attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Output (use last timestep)
        output = self.fc(attn_out[:, -1, :])
        
        return output


class MultiScaleEnsemble(nn.Module):
    """
    Process pattern at multiple timescales
    Combine for robust prediction
    """
    
    def __init__(self):
        super().__init__()
        
        # Short-term (last 6 minutes)
        self.short_term = nn.LSTM(1, 32, 1, batch_first=True)
        
        # Medium-term (last 12 minutes)
        self.medium_term = nn.LSTM(1, 32, 1, batch_first=True)
        
        # Long-term (all 18 minutes)
        self.long_term = nn.LSTM(1, 32, 1, batch_first=True)
        
        # Fusion
        self.fusion = nn.Linear(32 * 3, 7)
    
    def forward(self, x):
        # Process at different scales
        _, (short, _) = self.short_term(x[:, -6:, :])
        _, (medium, _) = self.medium_term(x[:, -12:, :])
        _, (long, _) = self.long_term(x)
        
        # Concatenate
        combined = torch.cat([short[-1], medium[-1], long[-1]], dim=1)
        
        # Fuse and predict
        output = self.fusion(combined)
        
        return output
```

---

### Module 3: Performance Analyzer

**File:** `performance_analyzer.py`

```python
"""
Analyze live performance to guide optimization
"""

import pandas as pd
import numpy as np
from typing import Dict, List

class PerformanceAnalyzer:
    """
    Analyze where model performs well/poorly
    Guide Stretcher optimizations
    """
    
    def __init__(self):
        self.live_predictions = []
        self.actual_outcomes = []
        self.game_contexts = []
    
    def record_live_prediction(
        self,
        prediction: float,
        interval: tuple,
        actual: float,
        game_context: Dict
    ):
        """Record each live prediction for analysis"""
        self.live_predictions.append({
            'prediction': prediction,
            'interval_lower': interval[0],
            'interval_upper': interval[1],
            'actual': actual,
            'error': abs(prediction - actual),
            **game_context
        })
    
    def analyze_error_patterns(self) -> Dict:
        """
        Where does model make mistakes?
        
        Returns insights like:
        - Struggles with close games (diff < 5)
        - Overconfident in blowouts
        - Underestimates comebacks
        - etc.
        """
        df = pd.DataFrame(self.live_predictions)
        
        # Analyze by game state
        close_games = df[abs(df['current_diff']) < 5]
        blowouts = df[abs(df['current_diff']) > 15]
        
        return {
            'overall_mae': df['error'].mean(),
            'close_game_mae': close_games['error'].mean(),
            'blowout_mae': blowouts['error'].mean(),
            'improvement_opportunities': self.identify_opportunities(df)
        }
    
    def identify_opportunities(self, df: pd.DataFrame) -> List[str]:
        """
        Specific opportunities for improvement
        """
        opportunities = []
        
        # Check if certain scenarios have high error
        if df[df['quarter'] == 2]['error'].mean() > df['error'].mean() + 1:
            opportunities.append('Q2 predictions less accurate - needs optimization')
        
        if df[df['home_team_leading']]['error'].mean() > 6.0:
            opportunities.append('Home team leading scenarios need work')
        
        if df['interval_upper'] - df['interval_lower'] > 15:
            opportunities.append('Intervals too wide - overconfident')
        
        return opportunities
    
    def recommend_architecture_changes(self) -> Dict:
        """
        Based on live data, what should we change?
        """
        analysis = self.analyze_error_patterns()
        
        recommendations = []
        
        # If close games are problematic
        if analysis['close_game_mae'] > analysis['overall_mae'] + 1:
            recommendations.append({
                'component': 'Feature Engineering',
                'change': 'Add volatility features for close games',
                'expected_improvement': '0.5-1.0 MAE reduction'
            })
        
        # If attention could help
        recommendations.append({
            'component': 'LSTM',
            'change': 'Replace with AttentionLSTM',
            'expected_improvement': '0.3-0.7 MAE reduction'
        })
        
        # If ensemble weights are suboptimal
        if self.ensemble_weight_analysis()['optimal'] != [0.4, 0.6]:
            recommendations.append({
                'component': 'Ensemble',
                'change': 'Dynamic weights based on game state',
                'expected_improvement': '0.2-0.5 MAE reduction'
            })
        
        return {
            'total_opportunities': len(recommendations),
            'recommendations': recommendations,
            'estimated_total_improvement': '1.0-2.2 MAE reduction',
            'target_mae': '3.2-4.4 (from current 5.39)'
        }
```

---

## Optimization Strategies

### Strategy 1: Feature Engineering+
**What:** Add advanced features beyond raw differentials

**Ideas:**
```python
# Momentum features
momentum_3min = pattern[-3:].mean() - pattern[-6:-3].mean()
momentum_6min = pattern[-6:].mean() - pattern[-12:-6].mean()

# Volatility features
volatility = np.std(pattern[-6:])
trend_strength = abs(pattern[-1] - pattern[0]) / 18

# Streak features
current_streak = count_consecutive_sign_changes(pattern)
max_lead = np.max(pattern)
max_deficit = np.min(pattern)

# Contextual features
is_close_game = abs(pattern[-1]) < 5
is_blowout = abs(pattern[-1]) > 15
lead_changes = count_lead_changes(pattern)
```

**Expected:** 0.3-0.5 MAE improvement

---

### Strategy 2: Architecture Search
**What:** Find optimal neural architecture

**Approaches:**
```python
# A) Add attention to LSTM
AttentionLSTM(
    hidden_size=128,  # Increase capacity
    num_heads=4,      # Multi-head attention
    dropout=0.2       # Prevent overfitting
)

# B) Temporal convolutions
TemporalConvNet(
    num_channels=[32, 64, 128],
    kernel_size=3,
    dropout=0.1
)

# C) Transformer encoder
TransformerEncoder(
    d_model=128,
    nhead=8,
    num_layers=3
)

# D) Hybrid (Conv + LSTM + Attention)
HybridArchitecture()  # Best of all worlds
```

**Expected:** 0.5-1.0 MAE improvement

---

### Strategy 3: Ensemble Optimization
**What:** Optimize how we combine models

**Ideas:**
```python
# Dynamic weights based on game state
def get_ensemble_weights(game_state):
    if game_state['is_close']:
        return [0.30, 0.70]  # Trust LSTM more in close games
    elif game_state['is_blowout']:
        return [0.50, 0.50]  # Equal weight in blowouts
    elif game_state['high_volatility']:
        return [0.45, 0.55]  # Slightly favor LSTM
    else:
        return [0.40, 0.60]  # Default

# Confidence-weighted ensemble
def confidence_weighted_ensemble(dejavu_pred, lstm_pred, dejavu_conf, lstm_conf):
    total_conf = dejavu_conf + lstm_conf
    w_dejavu = dejavu_conf / total_conf
    w_lstm = lstm_conf / total_conf
    
    return w_dejavu * dejavu_pred + w_lstm * lstm_pred

# Stacked ensemble (meta-learner)
meta_model = LightGBM(
    inputs=[dejavu_pred, lstm_pred, game_features],
    output=optimal_prediction
)
```

**Expected:** 0.2-0.5 MAE improvement

---

### Strategy 4: Conformal Optimization
**What:** Tighten confidence intervals without losing coverage

**Ideas:**
```python
# Adaptive quantiles based on confidence
def adaptive_quantile(base_quantile, confidence_score):
    if confidence_score > 0.9:
        # High confidence → tighter interval
        return base_quantile * 0.90
    elif confidence_score < 0.6:
        # Low confidence → wider interval
        return base_quantile * 1.10
    else:
        return base_quantile

# Conditional conformal prediction
# Different quantiles for different game states

# Localized conformal prediction
# Use only similar historical games for calibration
```

**Expected:** 10-20% narrower intervals with same coverage

---

## The Jungle Neural Architecture (Concept)

```python
class JungleNet(nn.Module):
    """
    Custom neural architecture optimized for NBA halftime prediction
    
    Like a jungle: Dense, interconnected, specialized
    """
    
    def __init__(self):
        super().__init__()
        
        # Multi-scale temporal convolutions
        self.conv_short = nn.Conv1d(1, 32, kernel_size=3)  # Local patterns
        self.conv_medium = nn.Conv1d(1, 32, kernel_size=5)  # Medium patterns
        self.conv_long = nn.Conv1d(1, 32, kernel_size=7)   # Long patterns
        
        # Bi-directional LSTM
        self.bilstm = nn.LSTM(96, 128, 2, bidirectional=True, batch_first=True)
        
        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(256, num_heads=8)
        
        # Skip connections (ResNet-style)
        self.skip_fc = nn.Linear(18, 256)
        
        # Dense fusion
        self.fusion = nn.Sequential(
            nn.Linear(256 + 256, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 7)  # 7-step forecast
        )
    
    def forward(self, x):
        # Multi-scale convolutions
        x_t = x.transpose(1, 2)
        conv_short = self.conv_short(x_t)
        conv_medium = self.conv_medium(x_t)
        conv_long = self.conv_long(x_t)
        
        # Concatenate
        conv_out = torch.cat([conv_short, conv_medium, conv_long], dim=1)
        conv_out = conv_out.transpose(1, 2)
        
        # Bi-LSTM
        lstm_out, _ = self.bilstm(conv_out)
        
        # Attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Skip connection from input
        skip = self.skip_fc(x.squeeze(-1))
        
        # Fuse everything
        combined = torch.cat([attn_out[:, -1, :], skip], dim=1)
        
        # Final prediction
        output = self.fusion(combined)
        
        return output

# Train this AFTER we have live data showing where to optimize!
```

---

## Implementation Timeline

**Month 1: MVP Running**
- Collect live predictions
- Record actual outcomes
- Track performance by scenario
- Build database of edge cases

**Month 2: Analysis Phase**
- Run ModelIntrospector
- Analyze error patterns
- Identify optimization opportunities
- Design Jungle architecture

**Month 3: Optimization Phase**
- Implement JungleNet
- Train on historical + live data
- A/B test vs current model
- Deploy if better

**Month 4: Iteration**
- Fine-tune based on results
- Add attention mechanisms
- Optimize ensemble weights
- Tighten conformal intervals

---

## Expected Results

**Current MVP:**
- MAE: 5.39
- Coverage: 94.6%
- Good enough to start

**After Stretcher Optimization:**
- Target MAE: 3.5-4.5 (20-35% improvement)
- Target Coverage: 95% (tighter intervals)
- Better performance in close games
- Adaptive ensemble weights

**Result:** More accurate predictions = larger edges = more profit

---

## Why This Approach is Smart

### Avoid Premature Optimization:
- Current 5.39 MAE is GOOD
- Don't know which optimizations matter yet
- Need real data to guide decisions

### Data-Driven Optimization:
- Collect 2-4 weeks of live data
- See where model actually struggles
- Optimize those specific areas
- Measure improvement

### Risk Management:
- Keep current model as backup
- A/B test new architecture
- Only deploy if proven better
- Can always roll back

---

**✅ Stretcher: Architecture defined, ready to build with live data**

*Model introspection framework  
Custom neural architectures (Jungle)  
Performance-guided optimization  
Build timeline: Month 2-3 of NBA season*

