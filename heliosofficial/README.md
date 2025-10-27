# 🌟 HELIOS OFFICIAL - INTELLIGENT FEATURE ENGINEERING SYSTEM

**The Feature Extraction Powerhouse Behind Mamba Mentality**

---

## 🎯 **WHAT IS HELIOS?**

**Helios** is the **intelligent feature engineering system** that powers the Mamba Mentality ML model. It extracts **67 high-dimensional features** from live NBA game data, transforming raw play-by-play data into predictive signals that capture momentum, patterns, and winning probabilities.

### **Why "Helios"?**
Named after the Greek god of the sun, Helios illuminates hidden patterns in NBA data, extracting features that would be invisible to traditional statistical analysis.

---

## 🔥 **KEY CAPABILITIES**

### **1. 67-Feature Extraction Pipeline**
Helios extracts **67 specialized features** from live game data:

- **18 Pattern Features** - Historical 18-minute differential patterns
- **4 Statistical Features** - Mean, std, min, max of patterns
- **5 Spectral Features** - FFT analysis, dominant frequency, power spectrum
- **10 Momentum Features** - Acceleration, jerk, EMA, RSI, directional movement
- **10 Autocorrelation Features** - Lag-1 to lag-10 autocorrelations
- **20 Advanced NBA Stats** - PER, TS%, USG%, ORtg, DRtg, pace, etc.

### **2. Auto Feature Mining**
- **LASSO-based feature selection** to identify the most predictive features
- **720+ candidate features** reduced to **elite 67**
- **Automated feature importance ranking**

### **3. Real-Time Extraction**
- Extracts features from **live game state** in <100ms
- **Streaming data processing** for Q2, Q3, Q4 predictions
- **Seamless integration** with NBA API and ESPN data sources

### **4. Play-by-Play Analysis**
- **Possession-level tracking** (offensive efficiency, defensive stops)
- **Shot stream analysis** (shot quality, shot clock pressure)
- **Momentum analysis** (swing detection, run analysis)
- **Signal transforms** (Fourier, wavelet, spectral analysis)

---

## 📂 **PACKAGE STRUCTURE**

```
heliosofficial/
├── core/                          # Core feature extraction engines
│   ├── helios_fix_and_retrain.py        # Helios retraining pipeline
│   ├── helios_proper_extraction.py      # 67-feature extraction
│   ├── phase_3_extract_720_features.py  # Full feature mining (720+)
│   ├── phase_4_lasso_mine_elite.py      # LASSO feature selection
│   ├── phase_5_train_final_model.py     # Final model training
│   └── helios_proof_of_concept.py       # POC validation
│
├── src/                           # Modular source code
│   ├── collectors/
│   │   └── pbp_collector.py              # Play-by-play data collection
│   ├── feature_engineering/
│   │   └── feature_builder.py            # Feature construction logic
│   └── transformers/
│       ├── momentum_analyzer.py          # Momentum feature extraction
│       ├── possession_stream.py          # Possession-level analysis
│       ├── shot_stream.py                # Shot quality analysis
│       └── signal_transforms.py          # Spectral/FFT transforms
│
├── documentation/                 # Complete documentation
│   ├── HELIOS_COMPLETE_DOCUMENTATION.md # Full system docs
│   ├── HELIOS_FINAL_RESULTS.md          # Results and performance
│   ├── PROJECT_HELIOS_MASTER.md         # Master project overview
│   ├── HOTSPOT_TRANSITION_PLAN.md       # Evolution roadmap
│   └── POC_CRITICAL_ANALYSIS.md         # POC analysis
│
├── examples/                      # Usage examples
│   ├── auto_helios_feature_mining.py    # Auto feature mining demo
│   ├── 18_feature_breakdown.py          # 18-pattern feature demo
│   └── 76_feature_breakdown.py          # 76-feature demo
│
└── README.md                      # This file
```

---

## 🚀 **QUICK START**

### **1. Extract 67 Features from Live Game**

```python
from heliosofficial.core.helios_proper_extraction import extract_67_features

# Live game data
game_data = {
    'period': 2,
    'clock': '6:00',
    'home_score': 55,
    'away_score': 48,
    'current_diff': 7
}

# Extract features
features = extract_67_features(game_data)
print(f"Extracted {len(features)} features: {features[:5]}...")
```

### **2. Run Auto Feature Mining**

```python
from heliosofficial.examples.auto_helios_feature_mining import run_auto_mining

# Mine elite features from 720+ candidates
elite_features = run_auto_mining(
    training_data='path/to/games.parquet',
    n_features=67,
    method='lasso'
)
```

### **3. Analyze Play-by-Play Data**

```python
from heliosofficial.src.collectors.pbp_collector import collect_pbp_data
from heliosofficial.src.transformers.momentum_analyzer import analyze_momentum

# Collect play-by-play
pbp_data = collect_pbp_data(game_id='0022500037')

# Analyze momentum
momentum_features = analyze_momentum(pbp_data)
print(f"Momentum score: {momentum_features['momentum_score']}")
```

---

## 🧠 **THE 67 FEATURES EXPLAINED**

### **Category 1: Pattern Features (18)**
Historical 18-minute differential patterns extracted from similar game states:
- `pattern_0` to `pattern_17`: Minute-by-minute score differentials

### **Category 2: Statistical Features (4)**
- `mean_diff`: Average differential over 18 minutes
- `std_diff`: Volatility of differential
- `min_diff`: Minimum differential (comeback potential)
- `max_diff`: Maximum differential (blowout potential)

### **Category 3: Spectral Features (5)**
FFT-based frequency analysis:
- `fft_dominant_freq`: Dominant scoring frequency
- `fft_power_spectrum`: Total spectral power
- `fft_harmonic_ratio`: Harmonic vs. noise ratio
- `fft_trend_component`: Long-term trend
- `fft_cycle_strength`: Cyclic pattern strength

### **Category 4: Momentum Features (10)**
Real-time momentum indicators:
- `momentum_velocity`: Rate of score change
- `momentum_acceleration`: Acceleration of scoring
- `momentum_jerk`: Jerk (change in acceleration)
- `ema_5min`: 5-minute exponential moving average
- `ema_10min`: 10-minute exponential moving average
- `rsi_14`: 14-period relative strength index
- `adx`: Average directional index
- `plus_di`: Positive directional indicator
- `minus_di`: Negative directional indicator
- `momentum_direction`: Current momentum direction

### **Category 5: Autocorrelation Features (10)**
Lag-based autocorrelations (scoring persistence):
- `autocorr_lag1` to `autocorr_lag10`: Lag-1 through lag-10

### **Category 6: Advanced NBA Stats (20)**
Real-time NBA analytics:
- `home_per`, `away_per`: Player efficiency rating
- `home_ts_pct`, `away_ts_pct`: True shooting percentage
- `home_usg_pct`, `away_usg_pct`: Usage percentage
- `home_ortg`, `away_ortg`: Offensive rating
- `home_drtg`, `away_drtg`: Defensive rating
- `home_ast_ratio`, `away_ast_ratio`: Assist ratio
- `home_tov_ratio`, `away_tov_ratio`: Turnover ratio
- `home_reb_pct`, `away_reb_pct`: Rebound percentage
- `pace`: Game pace (possessions per 48 minutes)
- `home_efg`, `away_efg`: Effective field goal percentage
- `home_ftr`, `away_ftr`: Free throw rate
- `home_3par`, `away_3par`: 3-point attempt rate

---

## 🎯 **HOW HELIOS POWERS MAMBA**

```
┌─────────────────────────────────────────────────────────────┐
│                    HELIOS WORKFLOW                          │
└─────────────────────────────────────────────────────────────┘

1. LIVE GAME DATA (ESPN/NBA API)
   ↓
2. HELIOS FEATURE EXTRACTION
   ├─ Pattern Features (18)
   ├─ Statistical Features (4)
   ├─ Spectral Features (5)
   ├─ Momentum Features (10)
   ├─ Autocorrelation Features (10)
   └─ Advanced NBA Stats (20)
   ↓
3. 67-DIMENSIONAL FEATURE VECTOR
   ↓
4. MAMBA MENTALITY MODEL
   ↓
5. FINAL PREDICTION (Score Differential at 18:00 Q4)
```

---

## 📊 **PERFORMANCE METRICS**

### **Feature Extraction Speed**
- **67 features extracted in <100ms**
- **Real-time processing** for live games
- **Scalable** to multiple games simultaneously

### **Predictive Power**
- **Mamba MAE: 6.8 points** (trained on 5,529 games)
- **Test performance: 9.3 MAE** (1,383 games)
- **67 features identified** through LASSO from 720+ candidates

### **Feature Importance (Top 10)**
1. `pattern_17` (final minute differential) - **0.42**
2. `home_ortg` (offensive rating) - **0.38**
3. `momentum_acceleration` - **0.35**
4. `fft_dominant_freq` - **0.33**
5. `ema_10min` - **0.31**
6. `home_ts_pct` (true shooting) - **0.29**
7. `autocorr_lag1` - **0.27**
8. `std_diff` - **0.25**
9. `pace` - **0.23**
10. `home_per` - **0.21**

---

## 🔧 **TRAINING & RETRAINING**

### **Retrain Helios Model**

```bash
cd heliosofficial/core
python3 helios_fix_and_retrain.py
```

### **Extract 720+ Features & Mine Elite 67**

```bash
# Step 1: Extract all 720+ candidate features
python3 phase_3_extract_720_features.py

# Step 2: Mine elite 67 features using LASSO
python3 phase_4_lasso_mine_elite.py

# Step 3: Train final model
python3 phase_5_train_final_model.py
```

---

## 🌐 **INTEGRATION WITH OTHER SYSTEMS**

### **1. Mamba Mentality Model**
Helios provides the **67 features** that Mamba uses to predict final score differentials.

```python
from mambaofficial.models import load_mamba_model
from heliosofficial.core.helios_proper_extraction import extract_67_features

# Extract features
features = extract_67_features(live_game_data)

# Make prediction
mamba_model = load_mamba_model()
prediction = mamba_model.predict([features])
print(f"Predicted final differential: {prediction[0]:.1f}")
```

### **2. OntoRisk System**
Helios features can be used by OntoRisk for **probability calibration** and **Kelly bet sizing**.

```python
from ontoriskofficial.components.ontorisk_phase1_probability_calibration import ProbabilityCalibrator

# Calibrate prediction
calibrator = ProbabilityCalibrator()
prob_result = calibrator.calculate_probability(
    prediction=prediction[0],
    spread_line=-6.0,
    home_team="Lakers",
    away_team="Warriors"
)
```

### **3. Live Trading Engine**
Helios is integrated into the **live trading engine** for real-time feature extraction.

```python
from livesystemofficial.core.live_trading_engine import LiveTradingEngine

engine = LiveTradingEngine()
opportunities = engine.scan_live_opportunities()  # Uses Helios internally
```

---

## 🧪 **TESTING & VALIDATION**

### **Run Proof of Concept**

```bash
python3 core/helios_proof_of_concept.py
```

### **Validate Feature Extraction**

```python
from heliosofficial.examples.auto_helios_feature_mining import validate_features

# Test on historical games
validation_results = validate_features(test_games='test_data.parquet')
print(f"Validation MAE: {validation_results['mae']:.2f}")
print(f"Feature importance: {validation_results['importance']}")
```

---

## 📈 **FUTURE ENHANCEMENTS**

### **Phase 1: Real-Time Play-by-Play** ✅ COMPLETE
- Implemented PBP collectors
- Possession stream analysis
- Shot quality tracking
- Momentum detection

### **Phase 2: 720+ Feature Mining** ✅ COMPLETE
- LASSO-based feature selection
- Elite 67 features identified
- Feature importance ranking

### **Phase 3: Advanced Signal Processing** 🚧 IN PROGRESS
- Wavelet transforms
- Chaos theory indicators
- Non-linear pattern detection
- Multi-scale analysis

### **Phase 4: Deep Learning Integration** 📋 PLANNED
- LSTM-based momentum prediction
- Transformer models for play-by-play
- Attention mechanisms for key plays
- Reinforcement learning for feature selection

---

## 🛠️ **DEPENDENCIES**

```python
numpy >= 1.21.0
pandas >= 1.3.0
scikit-learn >= 0.24.0
scipy >= 1.7.0
xgboost >= 1.4.0  # For Mamba model
nba_api >= 1.1.9  # For live data
requests >= 2.26.0
```

---

## 📚 **DOCUMENTATION**

- **[HELIOS_COMPLETE_DOCUMENTATION.md](documentation/HELIOS_COMPLETE_DOCUMENTATION.md)** - Full system documentation
- **[HELIOS_FINAL_RESULTS.md](documentation/HELIOS_FINAL_RESULTS.md)** - Performance results
- **[PROJECT_HELIOS_MASTER.md](documentation/PROJECT_HELIOS_MASTER.md)** - Project overview
- **[POC_CRITICAL_ANALYSIS.md](documentation/POC_CRITICAL_ANALYSIS.md)** - POC analysis

---

## 🤝 **CONTRIBUTING**

Helios is a critical component of the Ontologic XYZ trading system. To contribute:

1. Test new features on historical data first
2. Ensure real-time extraction remains <100ms
3. Validate predictive power (MAE improvement)
4. Document all new features in this README

---

## 📄 **LICENSE**

Proprietary - Ontologic XYZ © 2025

---

## 🎯 **QUICK REFERENCE**

```python
# Extract 67 features
from heliosofficial.core.helios_proper_extraction import extract_67_features
features = extract_67_features(game_data)

# Run auto feature mining
from heliosofficial.examples.auto_helios_feature_mining import run_auto_mining
elite_features = run_auto_mining(training_data, n_features=67)

# Analyze momentum
from heliosofficial.src.transformers.momentum_analyzer import analyze_momentum
momentum = analyze_momentum(pbp_data)

# Collect play-by-play
from heliosofficial.src.collectors.pbp_collector import collect_pbp_data
pbp = collect_pbp_data(game_id)
```

---

**🌟 Helios illuminates the path to profitable NBA betting through intelligent feature engineering! 🌟**

