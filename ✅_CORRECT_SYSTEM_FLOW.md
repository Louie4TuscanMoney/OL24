# ✅ CORRECT SYSTEM FLOW

**Win Probability (Continuous) + Mamba Trigger (Single)**

---

## 🎯 HOW IT WORKS

### **Win Probability (Continuous Every Minute)**

```
Frequency: Every 60 seconds
Start: After 6+ minutes of data
Duration: Entire game (until final buzzer)
Method: Quick 33-feature extraction + sigmoid conversion
Purpose: Real-time visualization, live tracking
```

**Flow:**
1. Collect score every 60s
2. After 6+ minutes → Extract 33 features
3. Convert margin → win probability
4. Update database
5. Broadcast via WebSocket
6. Repeat every minute

---

### **Mamba Model (Single Trigger)**

```
Frequency: ONCE per game
Trigger: Q2 6:00 (only)
Method: Full Mamba model analysis with intervals
Purpose: Official trading prediction, performance tracking
```

**Flow:**
1. Game reaches Q2 6:00
2. Check if 18+ minutes of data
3. Extract 33 features from all data
4. Full statistical analysis
5. Make official prediction
6. Store in mamba_game_cache
7. Create trading opportunity
8. Track for performance

---

## 📊 GAME TIMELINE EXAMPLE

```
Q1 0:00  → Game starts
           Cron collects: Score every 60s
           
Q1 6:00  → ✅ Win prob starts updating (6+ min data)
           📊 Every minute: Update probability
           📈 Display: Live probability gauge
           
Q2 0:00  → Win prob continues updating
           
Q2 6:00  → 🏆 MAMBA TRIGGERS!
           • Extract 33 features (from 18 min)
           • Full analysis with intervals
           • Official prediction: +5.2 points
           • Display: Golden prediction box
           • Trading opportunity created
           
Q2 5:59  → Win prob continues
Q2 5:58  → Win prob continues
... (every minute)
           
Q4 12:00 → Win prob updates (final seconds)
           
Final    → Track performance
           • Compare Mamba prediction vs actual
           • Update accuracy metrics
```

---

## 🔥 KEY DIFFERENCES

| Feature | Win Probability | Mamba Trigger |
|---------|----------------|---------------|
| **Frequency** | Every 60 seconds | Once per game |
| **When** | After 6+ minutes | Q2 6:00 only |
| **Features** | 33 (same) | 33 (same) |
| **Analysis** | Quick sigmoid | Full model + intervals |
| **Purpose** | Visualization | Trading decision |
| **Display** | Live gauge/chart | Golden prediction box |
| **Storage** | win_probability_timeline | mamba_game_cache |
| **Tracking** | Continuous timeline | Single prediction |

---

## ✅ SUMMARY

**You have TWO complementary systems:**

1. **📊 Win Probability (Continuous)**
   - Updates every minute
   - Shows live game flow
   - Real-time visualization
   - Tracks momentum

2. **🏆 Mamba Model (Single Trigger)**
   - Triggers at Q2 6:00
   - Official trading prediction
   - Full statistical analysis
   - Performance tracking

**They work together perfectly:**
- Win prob = Smooth tracking throughout
- Mamba = Official decision point
