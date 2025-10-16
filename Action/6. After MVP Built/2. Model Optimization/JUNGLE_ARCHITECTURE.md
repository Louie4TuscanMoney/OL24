# Jungle Architecture - Custom Neural Network for NBA

**Concept:** Dense, interconnected neural architecture optimized for NBA halftime prediction  
**Name Origin:** Like a jungle - dense, complex, specialized for environment  
**Status:** Conceptual framework, build after live data collection

---

## 🌴 What is "The Jungle"?

**Traditional LSTM (What We Have):**
```
Input (18 minutes)
    ↓
LSTM Layer 1 (64 units)
    ↓
LSTM Layer 2 (64 units)
    ↓
Dense Layer
    ↓
Output (7-step forecast)

Simple, linear, works well (5.39 MAE)
```

**Jungle Architecture (What We Could Build):**
```
Input (18 minutes)
    ↓
┌──────────────── JUNGLE ─────────────────┐
│                                          │
│  [Temporal Conv 3×3]                    │
│      ↓          ↘                        │
│  [Conv 5×5]     [Skip Connection]       │
│      ↓            ↓                      │
│  [Bi-LSTM] ───→ [Concat]                │
│      ↓            ↓                      │
│  [Attention] ─→ [Fusion]                │
│      ↓            ↓                      │
│  [GRU Cell]   [Residual]                │
│      ↓          ↓                        │
│  [Dense 256] → [Combine] → Output       │
│                                          │
│  Dense like jungle: Many paths to output│
│  Each path captures different patterns  │
└──────────────────────────────────────────┘

Complex, interconnected, potentially better
(But needs validation with real data!)
```

---

## Why "Jungle"?

**Metaphor:**
- **Forest:** Organized, planted rows (traditional architectures)
- **Jungle:** Dense, organic growth, specialized niches (our custom architecture)

**Characteristics:**
- Multiple parallel pathways
- Skip connections everywhere
- Attention mechanisms
- Multi-scale processing
- Adaptive components

**Goal:** Capture complex NBA game dynamics that simple LSTM might miss

---

## Mathematical Foundation

### Component 1: Multi-Scale Temporal Convolutions

**Purpose:** Capture patterns at different timescales

```python
"""
Short-term patterns (last 3 minutes)
Medium-term patterns (last 9 minutes)
Long-term patterns (all 18 minutes)
"""

class MultiScaleConv(nn.Module):
    def __init__(self):
        super().__init__()
        # Different kernel sizes = different timescales
        self.conv_3 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.conv_5 = nn.Conv1d(1, 32, kernel_size=5, padding=2)
        self.conv_7 = nn.Conv1d(1, 32, kernel_size=7, padding=3)
    
    def forward(self, x):
        # Process at 3 scales simultaneously
        short = self.conv_3(x)
        medium = self.conv_5(x)
        long = self.conv_7(x)
        
        # Concatenate → 96 channels total
        return torch.cat([short, medium, long], dim=1)

# Captures both:
# - Rapid momentum shifts (short-term)
# - Sustained trends (long-term)
```

---

### Component 2: Bi-Directional LSTM with Attention

**Purpose:** Understand patterns both forward and backward in time

```python
class AttentiveBiLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        # Bi-directional: Reads pattern forward AND backward
        self.bilstm = nn.LSTM(
            input_size=96,
            hidden_size=128,
            num_layers=2,
            bidirectional=True,  # Forward + backward
            batch_first=True
        )
        
        # Attention: Learn which timesteps matter most
        self.attention = nn.Sequential(
            nn.Linear(256, 128),  # 256 = 128×2 (bidirectional)
            nn.Tanh(),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        # BiLSTM processes sequence
        lstm_out, _ = self.bilstm(x)  # (batch, 18, 256)
        
        # Calculate attention scores
        scores = self.attention(lstm_out)  # (batch, 18, 1)
        weights = torch.softmax(scores, dim=1)
        
        # Weighted sum (emphasizes important timesteps)
        context = torch.sum(lstm_out * weights, dim=1)
        
        return context, weights

# Learns: Which minutes (1-18) are most predictive
# Example: Minutes 16-18 might get 60% of attention
```

---

### Component 3: Residual Connections

**Purpose:** Allow information to skip layers (like ResNet)

```python
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(dim, dim)
        )
    
    def forward(self, x):
        # x + F(x) = Residual connection
        return x + self.layers(x)

# Allows gradient flow
# Prevents vanishing gradients
# Enables deeper networks
```

---

### Component 4: Multi-Head Attention

**Purpose:** Capture different types of patterns simultaneously

```python
class MultiHeadProcessor(nn.Module):
    def __init__(self):
        super().__init__()
        # 8 attention heads = 8 different "views" of data
        self.attention = nn.MultiheadAttention(
            embed_dim=256,
            num_heads=8,
            batch_first=True
        )
    
    def forward(self, x):
        # Each head learns different pattern type:
        # Head 1: Momentum patterns
        # Head 2: Volatility patterns
        # Head 3: Streak patterns
        # Head 4: Lead change patterns
        # etc.
        
        attn_out, attn_weights = self.attention(x, x, x)
        
        return attn_out, attn_weights

# More flexible than single LSTM
# Learns what matters through data
```

---

## Complete Jungle Architecture

```python
class JungleNet(nn.Module):
    """
    The Jungle: Dense custom architecture for NBA prediction
    
    Components:
    1. Multi-scale temporal convolutions
    2. Bi-directional LSTM with attention
    3. Multi-head self-attention
    4. Residual connections
    5. Dynamic ensemble fusion
    6. Uncertainty quantification
    
    Expected: 3.5-4.5 MAE (vs current 5.39)
    """
    
    def __init__(self):
        super().__init__()
        
        # === FEATURE EXTRACTION (Multi-Scale) ===
        self.multi_scale_conv = MultiScaleConv()  # 96 channels
        
        # === TEMPORAL PROCESSING (Bi-LSTM) ===
        self.bilstm = nn.LSTM(96, 128, 2, bidirectional=True, batch_first=True)
        
        # === ATTENTION MECHANISM ===
        self.attention = nn.MultiheadAttention(256, num_heads=8, batch_first=True)
        
        # === SKIP CONNECTION (Direct from input) ===
        self.skip = nn.Linear(18, 256)
        
        # === RESIDUAL BLOCKS ===
        self.residual1 = ResidualBlock(256)
        self.residual2 = ResidualBlock(256)
        
        # === FUSION LAYER ===
        self.fusion = nn.Sequential(
            nn.Linear(256 + 256, 256),  # Context + skip
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 7)  # 7-step forecast
        )
        
        # === UNCERTAINTY ESTIMATOR ===
        self.uncertainty = nn.Linear(256, 7)  # Predict interval width
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Multi-scale convolutions
        conv_out = self.multi_scale_conv(x)
        
        # Bi-LSTM
        lstm_out, _ = self.bilstm(conv_out)
        
        # Self-attention
        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Take last timestep
        context = attn_out[:, -1, :]
        
        # Skip connection (ResNet-style)
        skip = self.skip(x.squeeze(-1))
        
        # Residual processing
        context = self.residual1(context)
        context = self.residual2(context)
        
        # Combine context + skip
        combined = torch.cat([context, skip], dim=1)
        
        # Final prediction
        point_forecast = self.fusion(combined)
        
        # Uncertainty estimate
        interval_width = torch.exp(self.uncertainty(context))  # Always positive
        
        return {
            'point_forecast': point_forecast,
            'interval_width': interval_width,
            'attention_weights': attn_weights  # For interpretation
        }


# Training
model = JungleNet()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

# Custom loss: MAE + Interval calibration
def jungle_loss(pred, target, interval_width):
    # Point forecast loss
    mae_loss = torch.abs(pred - target).mean()
    
    # Interval calibration loss (want tight but accurate)
    interval_loss = interval_width.mean()  # Penalize wide intervals
    
    # Coverage loss (ensure 95% of targets within intervals)
    lower = pred - 1.96 * interval_width
    upper = pred + 1.96 * interval_width
    coverage = ((target >= lower) & (target <= upper)).float().mean()
    coverage_loss = torch.abs(coverage - 0.95) * 10  # Want exactly 95%
    
    total_loss = mae_loss + 0.1 * interval_loss + coverage_loss
    
    return total_loss
```

---

## Optimization Workflow

### Step 1: Collect Live Data (Weeks 1-4)
```python
# After each game
live_data = {
    'pattern': [0, -2, +1, ..., +4],  # 18-minute pattern
    'prediction': +15.1,
    'actual_halftime': +13.5,
    'error': 1.6,
    'game_context': {
        'home_team': 'LAL',
        'away_team': 'BOS',
        'was_close': False,
        'quarter_2_momentum': 'home'
    }
}

# Store in database
store_prediction(live_data)
```

### Step 2: Analyze Performance (Week 5)
```python
# Run introspection
analyzer = PerformanceAnalyzer()
results = analyzer.analyze_error_patterns()

# Example findings:
{
    'overall_mae': 5.39,
    'close_game_mae': 6.8,  # WORSE in close games
    'blowout_mae': 3.9,     # BETTER in blowouts
    'opportunities': [
        'Optimize for close games',
        'Add momentum features',
        'Improve Q2 performance'
    ]
}
```

### Step 3: Design Jungle (Week 6)
```python
# Based on analysis, design custom architecture
jungle = JungleNet()

# Add components that address weaknesses:
# - Attention for close games
# - Momentum features
# - Multi-scale processing
```

### Step 4: Train Jungle (Week 7)
```python
# Train on historical + live data
train_jungle(
    historical_data=6600_games,
    live_data=50_games,  # New live data is gold!
    epochs=100,
    validation=live_validation_set
)
```

### Step 5: A/B Test (Week 8)
```python
# Test both models on new live games
for game in next_10_games:
    mvp_pred = mvp_model.predict(game.pattern)
    jungle_pred = jungle_model.predict(game.pattern)
    
    # Track which is more accurate
    mvp_errors.append(abs(mvp_pred - game.actual))
    jungle_errors.append(abs(jungle_pred - game.actual))

# After 10 games:
mvp_mae = mean(mvp_errors)      # 5.39
jungle_mae = mean(jungle_errors) # 4.1 (if successful)

# If Jungle is better → deploy
# If MVP is better → keep current
```

---

## Advanced Features (Jungle)

### Feature 1: Game State Encoder
```python
class GameStateEncoder(nn.Module):
    """
    Encode game context beyond just score differential
    """
    
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(10, 32),  # 10 context features
            nn.ReLU(),
            nn.Linear(32, 16)
        )
    
    def forward(self, context):
        # Context features:
        # - Is close game? (|diff| < 5)
        # - Home team momentum
        # - Lead changes count
        # - Volatility
        # - Time remaining in Q2
        # - Home/away team strength
        # - Recent form
        # - Rest days
        # - Back-to-back?
        # - Rivalry game?
        
        encoded = self.encoder(context)
        return encoded
```

### Feature 2: Adaptive Ensemble
```python
class AdaptiveEnsemble(nn.Module):
    """
    Learn optimal ensemble weights based on game state
    Instead of fixed 40/60, adapt dynamically
    """
    
    def __init__(self):
        super().__init__()
        # Neural network learns weights
        self.weight_predictor = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softmax(dim=1)  # Weights sum to 1
        )
    
    def forward(self, dejavu_pred, lstm_pred, game_features):
        # Predict optimal weights for THIS specific game
        weights = self.weight_predictor(game_features)
        
        # Weighted combination
        ensemble = weights[:, 0] * dejavu_pred + weights[:, 1] * lstm_pred
        
        return ensemble, weights

# Example: Close game → [0.35, 0.65] (favor LSTM)
#          Blowout → [0.48, 0.52] (more balanced)
```

### Feature 3: Uncertainty Network
```python
class UncertaintyNetwork(nn.Module):
    """
    Predict confidence interval width based on game characteristics
    Better than fixed quantile from conformal prediction
    """
    
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus()  # Ensure positive
        )
    
    def forward(self, features):
        # Predict interval width
        width = self.network(features)
        
        return width

# Learn: Tight intervals for high-confidence
#        Wide intervals for uncertain games
```

---

## Expected Improvements

**Current MVP:**
```
MAE: 5.39
Coverage: 94.6%
Interval width: ±13.04
```

**After Jungle Optimization (Estimated):**
```
MAE: 3.5-4.5 (18-35% improvement)
Coverage: 95.0% (target achieved)
Interval width: ±9-11 (20-30% tighter)

Result: More accurate + more confident
```

**Value:**
- Smaller MAE → Detect more edges
- Tighter intervals → Higher conviction bets
- Better calibration → More reliable

**Risk:**
- Overfitting to historical data
- More complex = harder to debug
- Requires more training data

**Mitigation:**
- Use live data for validation
- A/B test before deploying
- Keep MVP as backup

---

## Implementation Plan

### Phase 1: Data Collection (Weeks 1-4)
```python
# Collect comprehensive live data
for game in live_games:
    # Record everything
    record({
        'pattern': game.pattern_18min,
        'prediction': mvp_model.predict(game.pattern),
        'actual': game.halftime_differential,
        'error': abs(prediction - actual),
        'game_context': {
            'teams': [...],
            'score_state': [...],
            'momentum': [...],
            'all_contextual_features': [...]
        }
    })

# After 4 weeks: ~50-80 games of live data
# This is GOLD for optimization
```

### Phase 2: Analysis (Week 5)
```python
# Run Stretcher introspection
stretcher = ModelIntrospector(mvp_model)
analysis = stretcher.full_analysis()

# Outputs:
{
    'error_patterns': [...],
    'feature_importance': [...],
    'ensemble_weight_analysis': [...],
    'improvement_opportunities': [
        'Close games: +1.4 MAE vs baseline',
        'Q2 start predictions: Less accurate',
        'Home team bias: Slight underestimation'
    ],
    'recommended_changes': [...]
}
```

### Phase 3: Design Jungle (Week 6)
```python
# Design custom architecture targeting weaknesses
jungle = JungleNet(
    # Add components that address identified issues
    attention_for_close_games=True,
    momentum_features=True,
    adaptive_ensemble_weights=True
)
```

### Phase 4: Train (Week 7)
```python
# Train Jungle on all data
jungle.train(
    historical=6600_games,
    live=80_games,  # Weight live data higher
    epochs=200,
    early_stopping=True
)
```

### Phase 5: Validate (Week 8)
```python
# A/B test for 2 weeks
for game in next_20_games:
    mvp_mae = test(mvp_model, game)
    jungle_mae = test(jungle, game)
    
    track_both()

# Deploy winner
if jungle_mae < mvp_mae - 0.5:
    deploy(jungle)
    print("Jungle is better! MAE improved by", mvp_mae - jungle_mae)
else:
    keep(mvp_model)
    print("MVP still optimal")
```

---

## Why Wait?

**Don't optimize prematurely:**

1. **Need Data:** Can't optimize without knowing what's broken
2. **MVP Works:** 5.39 MAE is good enough to start
3. **Risk:** Premature optimization often makes things worse
4. **Time:** 4-6 weeks to do properly

**Better Approach:**
1. Deploy MVP
2. Collect real performance data
3. Let data show us what to optimize
4. Build Jungle targeting specific weaknesses
5. A/B test before deploying

**Data-driven optimization > Guessing**

---

**✅ Jungle/Stretcher: Framework ready, build with live data**

*Custom neural architecture  
Multi-scale, attention, residual  
Data-driven optimization  
Target: 3.5-4.5 MAE*

