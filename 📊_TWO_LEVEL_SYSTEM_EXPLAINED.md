# 📊 TWO-LEVEL ML SYSTEM EXPLAINED

**Win Probability (Continuous) + Mamba Triggers (Dual)**

---

## 🎯 HOW IT WORKS

### **Level 1: Win Probability (Every Minute)**
```
Frequency: Every 60 seconds
Start: After 6+ minutes of data
Duration: Entire game
Method: Quick sigmoid conversion
Purpose: Live tracking, visualization
```

**What It Does:**
- Updates **every minute** after sufficient data
- Uses same 33 Mamba features
- Simple: margin → win probability (sigmoid)
- Stores in `win_probability_timeline` table

**Display:**
- Live probability gauge
- Continuous chart
- Real-time confidence

---

### **Level 2: Mamba Official Predictions (Trigger Points)**

```
Frequency: Twice per game
Trigger 1: Q1 11:00 (FIRST!)
Trigger 2: Q2 6:00 (traditional)
Method: Full Mamba model analysis
Purpose: Trading decisions, official forecasts
```

**What It Does:**
- Triggers at **specific moments**
- Full 33 feature extraction
- Complete statistical analysis
- Official trading signals
- Stores in `mamba_game_cache`

**Display:**
- Golden prediction box
- Trading opportunity cards
- Performance tracking

---

## 📊 DATA FLOW

```
Game Start (Q1 0:00)
↓
Collect score every 60s
↓
After 6+ minutes: Start win prob updates
├─ Every 60s: Update win probability
├─ Display: Live probability gauge
└─ Store: win_probability_timeline
↓
Q1 11:00 → ⚡ FIRST MAMBA TRIGGER
├─ Extract 33 features (from first 10 min)
├─ Make official prediction
├─ Display: Golden Q1 prediction box
├─ Create: Trading opportunity
└─ Store: mamba_game_cache (triggered_type='Q1_11:00')
↓
Continue win prob updates every 60s
↓
Q2 6:00 → ⚡ SECOND MAMBA TRIGGER
├─ Extract 33 features (from first 18 min)
├─ Make official prediction
├─ Display: Golden Q2 prediction box
├─ Create: Trading opportunity
└─ Store: mamba_game_cache (triggered_type='Q2_6:00')
↓
Continue win prob updates until game ends
↓
Game End → Track performance
```

---

## 🎨 VISUAL COMPARISON

### **Win Probability (Continuous)**
```
Q1:    ████░░░░░  45% (smooth, updates every min)
Q2:    ██████░░░  60% (trending upward)
Q3:    ████████░  80% (confident)
Q4:    █████████  90% (almost certain)
```

### **Mamba Triggers (Dual)**
```
Q1 11:00:  🏆 OFFICIAL PREDICTION: +5.2 points
           Confidence: 75%
           EV: +12%

Q2 6:00:   🏆 OFFICIAL PREDICTION: +4.8 points
           Confidence: 80%
           EV: +15%
```

---

## 💡 WHY TWO LEVELS?

### **Win Probability (Continuous)**
**Purpose:**
- ✅ Real-time visualization
- ✅ Smooth probability tracking
- ✅ Live confidence indicators
- ✅ Pattern progression

**When Used:**
- Watching game flow
- Tracking momentum
- Real-time dashboard
- Live pattern chart

### **Mamba Triggers (Official)**
**Purpose:**
- ✅ Betting decisions
- ✅ Trading signals
- ✅ Performance tracking
- ✅ Historical records

**When Used:**
- Making bets
- Calculating EV
- Trading dashboard
- Performance analysis

---

## 🎯 SUMMARY

**You have BOTH:**

1. **📊 Continuous Win Probabilities**
   - Every minute
   - Real-time tracking
   - Visualization
   - Live confidence

2. **🏆 Official Mamba Predictions**
   - Q1 11:00 + Q2 6:00
   - Full feature analysis
   - Trading signals
   - Performance tracking

**They work together:**
- Win prob = smooth, continuous tracking
- Mamba = official, actionable predictions

**Best of both worlds!** 🚀
