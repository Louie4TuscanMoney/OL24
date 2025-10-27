# 🐍 MAMBA MENTALITY SYSTEM - USAGE GUIDE

Complete guide to using the Mamba system in your projects.

---

## 📦 INSTALLATION

### Step 1: Copy Files

Copy the `mambaofficial` folder to your project:

```bash
cp -r mambaofficial /path/to/your/project/
```

### Step 2: Install Dependencies

```bash
pip install numpy pandas scikit-learn scipy pickle5
```

### Step 3: Verify Installation

```python
import pickle
import os

# Check if model exists
model_path = 'mambaofficial/models/MAMBA_MENTALITY_SYSTEM.pkl'
if os.path.exists(model_path):
    print("✅ Mamba model found!")
    print(f"   Size: {os.path.getsize(model_path) / 1024 / 1024:.1f} MB")
else:
    print("❌ Mamba model not found!")
```

---

## 🚀 BASIC USAGE

### Load the Model

```python
import pickle
import numpy as np

# Load Mamba model
print("📂 Loading Mamba model...")
with open('mambaofficial/models/MAMBA_MENTALITY_SYSTEM.pkl', 'rb') as f:
    mamba = pickle.load(f)

print("✅ Mamba model loaded!")
print(f"   Training games: {mamba['metadata']['train_games']:,}")
print(f"   Test games: {mamba['metadata']['test_games']:,}")
print(f"   Features: {mamba['metadata']['feature_count']}")
```

### Load Training Data

```python
# Load full training data (6,912 games)
print("📂 Loading training data...")
with open('mambaofficial/training_data/ENHANCED_PATTERNS_FULL.pkl', 'rb') as f:
    training_data = pickle.load(f)

print(f"✅ Training data loaded: {len(training_data):,} games")

# Inspect first game
first_game = training_data[0]
print(f"\n📊 Example game:")
print(f"   Game ID: {first_game['game_id']}")
print(f"   Pattern: {first_game['pattern'][:5]}... (first 5 minutes)")
print(f"   Halftime: {first_game['diff_at_halftime']:+d}")
print(f"   Final: {first_game['diff_at_final']:+d}")
```

---

## 🎯 MAKING PREDICTIONS

### Extract Features from Live Game

```python
def extract_features_from_live_game(game_data):
    """
    Extract 67 features from live game data
    
    Args:
        game_data: Dict with game info (scores, time, etc.)
    
    Returns:
        numpy array with 67 features
    """
    features = []
    
    # 1. Pattern features (18) - minute-by-minute differentials
    pattern = game_data.get('pattern', [0] * 18)  # 18-minute pattern
    features.extend(pattern)
    
    # 2. Statistical features
    features.append(np.mean(pattern))      # mean_diff
    features.append(np.std(pattern))       # std_diff
    features.append(np.polyfit(range(len(pattern)), pattern, 1)[0])  # trend
    features.append(np.std(np.diff(pattern)) if len(pattern) > 1 else 0)  # volatility
    
    # 3. Spectral features (6)
    fft = np.fft.fft(pattern)
    features.append(np.sum(np.abs(fft) ** 2))  # spectral_energy
    features.append(0.07)  # low_freq_power (placeholder)
    features.append(0.05)  # mid_freq_power (placeholder)
    features.append(0.16)  # high_freq_power (placeholder)
    features.append(0.11)  # dominant_freq (placeholder)
    features.append(1.25)  # spectral_entropy (placeholder)
    
    # 4. Momentum features (6)
    velocity = np.diff(pattern) if len(pattern) > 1 else [0]
    features.append(np.mean(velocity))  # velocity
    features.append(np.mean(np.diff(velocity)) if len(velocity) > 1 else 0)  # acceleration
    features.append(np.mean(velocity[-3:]) if len(velocity) >= 3 else np.mean(velocity))  # recent_momentum
    features.append(len([i for i in range(1, len(pattern)) if (pattern[i] > 0) != (pattern[i-1] > 0)]))  # lead_changes
    features.append(np.max(pattern) - np.min(pattern))  # max_swing
    features.append(abs(pattern[-1]) / max(1, sum(game_data.get('scores', [100, 100]))))  # comeback_potential
    
    # 5. Autocorrelation features (3)
    if len(pattern) > 5:
        features.append(np.corrcoef(pattern[:-1], pattern[1:])[0, 1])  # autocorr_lag1
        features.append(np.corrcoef(pattern[:-3], pattern[3:])[0, 1])  # autocorr_lag3
        features.append(np.corrcoef(pattern[:-5], pattern[5:])[0, 1])  # autocorr_lag5
    else:
        features.extend([0.0, 0.0, 0.0])
    
    # 6. Advanced features (remaining to reach 67 total)
    # Add team form, home advantage, etc.
    features.extend([0.5] * (67 - len(features)))  # Fill remaining with defaults
    
    return np.array(features, dtype=np.float32)


# Example usage
game_data = {
    'pattern': [-1, 0, -3, -3, -5, -6, -4, -6, -2, -7, -6, -9, -5, -1, -2, -7, -1, -7],
    'scores': [52, 48],  # Home, Away
    'quarter': 2,
    'time': '6:00'
}

features = extract_features_from_live_game(game_data)
print(f"✅ Extracted {len(features)} features")
```

### Make Prediction

```python
def make_prediction(mamba, features):
    """
    Make prediction using Mamba model
    
    Args:
        mamba: Loaded Mamba model
        features: numpy array with 67 features
    
    Returns:
        prediction (float): Predicted score differential
    """
    # Scale features if scaler exists
    if 'scaler' in mamba and mamba['scaler'] is not None:
        X = mamba['scaler'].transform(features.reshape(1, -1))
    else:
        X = features.reshape(1, -1)
    
    # Get model
    if 'model' in mamba:
        model = mamba['model']
    elif 'branch_b_final' in mamba and 'model' in mamba['branch_b_final']:
        model = mamba['branch_b_final']['model']
    else:
        raise ValueError("Model not found in Mamba structure")
    
    # Make prediction
    prediction = model.predict(X)[0]
    
    return prediction


# Example usage
prediction = make_prediction(mamba, features)
print(f"🎯 Prediction: {prediction:+.1f} points")
print(f"   Interpretation: Home team will win by {prediction:.1f} points")
```

---

## 💰 BETTING INTEGRATION

### Calculate Edge

```python
def calculate_edge(prediction, market_spread):
    """
    Calculate edge (difference between prediction and market)
    
    Args:
        prediction: ML model prediction
        market_spread: Current market spread (negative = home favored)
    
    Returns:
        edge (float): Absolute difference
    """
    edge = abs(prediction - market_spread)
    return edge


# Example
prediction = 2.5  # Home wins by 2.5
market_spread = -3.5  # Home favored by 3.5

edge = calculate_edge(prediction, market_spread)
print(f"📊 Edge: {edge:.1f} points")

if edge >= 5.0:
    print("✅ BETTING OPPORTUNITY! (edge ≥ 5.0)")
else:
    print("❌ No betting opportunity (edge < 5.0)")
```

### Determine Bet Side

```python
def determine_bet_side(prediction, market_spread):
    """
    Determine which side to bet
    
    Args:
        prediction: ML model prediction
        market_spread: Current market spread
    
    Returns:
        side (str): "HOME" or "AWAY"
        line (str): Formatted bet line
    """
    if prediction > market_spread:
        # Prediction is better than spread - bet HOME
        side = "HOME"
        line = f"Home {market_spread:+.1f}"
    else:
        # Prediction is worse than spread - bet AWAY
        side = "AWAY"
        line = f"Away {-market_spread:+.1f}"
    
    return side, line


# Example
side, line = determine_bet_side(prediction, market_spread)
print(f"🎲 BET: {side} - {line}")
```

---

## 📊 PERFORMANCE TRACKING

### Track Predictions

```python
import json
from datetime import datetime

class PredictionTracker:
    def __init__(self):
        self.predictions = []
    
    def add_prediction(self, game_id, prediction, market_spread, actual_result=None):
        """Add a prediction to track"""
        self.predictions.append({
            'game_id': game_id,
            'prediction': prediction,
            'market_spread': market_spread,
            'edge': abs(prediction - market_spread),
            'actual_result': actual_result,
            'timestamp': datetime.now().isoformat()
        })
    
    def get_performance(self):
        """Calculate performance metrics"""
        completed = [p for p in self.predictions if p['actual_result'] is not None]
        
        if not completed:
            return {'status': 'No completed predictions'}
        
        # Calculate metrics
        total = len(completed)
        correct = sum(1 for p in completed if abs(p['prediction'] - p['actual_result']) <= 9.029)
        accuracy = correct / total * 100
        
        return {
            'total_predictions': total,
            'correct_predictions': correct,
            'accuracy': f"{accuracy:.1f}%",
            'avg_edge': np.mean([p['edge'] for p in completed])
        }
    
    def save_to_file(self, filename='predictions.json'):
        """Save predictions to file"""
        with open(filename, 'w') as f:
            json.dump(self.predictions, f, indent=2)


# Example usage
tracker = PredictionTracker()

# Add prediction
tracker.add_prediction(
    game_id='0022500123',
    prediction=2.5,
    market_spread=-3.5,
    actual_result=4.0  # Home won by 4
)

# Get performance
performance = tracker.get_performance()
print(f"📊 Performance: {performance}")
```

---

## 🔧 ADVANCED USAGE

### Retrain Model with New Data

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

def retrain_mamba(training_data, test_size=0.2):
    """
    Retrain Mamba model with new data
    
    Args:
        training_data: List of game dicts with features
        test_size: Fraction of data for testing
    
    Returns:
        trained_model: Dict with model, scaler, and metadata
    """
    # Prepare features and targets
    X = []
    y = []
    
    for game in training_data:
        # Extract features (67 total)
        features = [game.get('mean_diff', 0), game.get('std_diff', 0)]  # etc...
        X.append(features)
        y.append(game.get('diff_at_final', 0))
    
    X = np.array(X)
    y = np.array(y)
    
    # Split train/test
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    train_score = model.score(X_train_scaled, y_train)
    test_score = model.score(X_test_scaled, y_test)
    
    print(f"✅ Model retrained!")
    print(f"   Train R²: {train_score:.3f}")
    print(f"   Test R²: {test_score:.3f}")
    
    return {
        'model': model,
        'scaler': scaler,
        'metadata': {
            'train_games': len(X_train),
            'test_games': len(X_test),
            'train_r2': train_score,
            'test_r2': test_score
        }
    }
```

---

## 🎓 BEST PRACTICES

### 1. Always Validate Features
```python
def validate_features(features):
    """Ensure features are valid"""
    if len(features) != 67:
        raise ValueError(f"Expected 67 features, got {len(features)}")
    
    if np.any(np.isnan(features)):
        raise ValueError("Features contain NaN values")
    
    if np.any(np.isinf(features)):
        raise ValueError("Features contain infinite values")
    
    return True
```

### 2. Use Feature Scaling
```python
# Always scale features if model was trained with scaler
if 'scaler' in mamba and mamba['scaler'] is not None:
    features_scaled = mamba['scaler'].transform(features.reshape(1, -1))
else:
    features_scaled = features.reshape(1, -1)
```

### 3. Handle Errors Gracefully
```python
try:
    prediction = make_prediction(mamba, features)
    print(f"✅ Prediction: {prediction:+.1f}")
except Exception as e:
    print(f"❌ Prediction error: {e}")
    prediction = None
```

### 4. Track All Predictions
```python
# Always track predictions for performance analysis
tracker.add_prediction(
    game_id=game_id,
    prediction=prediction,
    market_spread=market_spread,
    actual_result=None  # Update later
)
```

---

## 🚨 COMMON ISSUES

### Issue 1: Model Not Loading
```python
# Solution: Check file path and permissions
import os
model_path = 'mambaofficial/models/MAMBA_MENTALITY_SYSTEM.pkl'
if not os.path.exists(model_path):
    print(f"❌ Model not found at: {model_path}")
else:
    print(f"✅ Model found: {os.path.getsize(model_path) / 1024 / 1024:.1f} MB")
```

### Issue 2: Wrong Number of Features
```python
# Solution: Ensure you're extracting exactly 67 features
features = extract_features_from_live_game(game_data)
assert len(features) == 67, f"Expected 67 features, got {len(features)}"
```

### Issue 3: Predictions Out of Range
```python
# Solution: Check if features are reasonable
if abs(prediction) > 50:
    print(f"⚠️ Warning: Prediction seems unrealistic: {prediction:+.1f}")
```

---

## 📞 SUPPORT

For issues or questions:
1. Check this guide
2. Review the README.md
3. Check the documentation folder
4. Contact support

---

**🐍 Built with Mamba Mentality**

