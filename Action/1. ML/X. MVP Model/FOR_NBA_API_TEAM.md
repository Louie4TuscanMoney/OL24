# For NBA_API Integration Team

**ML Model Status:** ✅ READY (5.39 MAE, 94.6% coverage)

---

## What You Need to Provide

The ML model needs this input from NBA_API:

### Input Format: 18-Number Array

**Example:**
```python
[0, +2, +5, +8, +10, +12, +15, +17, +18, +19, +20, +21, +22, +20, +19, +18, +17, +16]
```

**What it represents:**
- 18 score differentials (home_score - away_score)
- One per minute from game start to 6:00 left in Q2
- Minute 0 always starts at 0 (game start)
- Minute 17 is the last value (at 6:00 Q2)

**When to call the model:**
- At exactly 6:00 remaining in Q2
- That's when you have all 18 minutes of data
- Model will predict halftime differential (0:00 Q2)

---

## What the Model Returns

```json
{
  "point_forecast": +18.5,
  "interval_lower": +5.5,
  "interval_upper": +31.5,
  "coverage_probability": 0.95,
  "components": {
    "dejavu_prediction": +20.0,
    "lstm_prediction": +17.5
  }
}
```

**Interpretation:**
- **point_forecast:** Expected halftime differential (+18.5 = home team leads by 18.5)
- **interval:** 95% confident true value is between +5.5 and +31.5
- **coverage:** 95 out of 100 times, actual will be in this range

---

## Your Implementation Task

### 1. Score Buffer (Simple)

```python
class ScoreBuffer:
    def __init__(self):
        self.differentials = []
        self.start_time = None
    
    def update(self, home_score, away_score, game_time_seconds):
        """
        Called every time score changes
        
        Args:
            home_score: Current home team score
            away_score: Current away team score
            game_time_seconds: Seconds elapsed in game
        """
        diff = home_score - away_score
        minute = game_time_seconds // 60  # Convert to minute
        
        # Store differential for this minute
        if minute < 18:
            while len(self.differentials) <= minute:
                self.differentials.append(0)
            self.differentials[minute] = diff
    
    def get_pattern(self):
        """
        Returns: 18-element array when ready
        """
        if len(self.differentials) >= 18:
            return np.array(self.differentials[:18])
        return None  # Not ready yet
```

### 2. Integration Point

```python
# In your game monitoring loop
def monitor_game(game_id):
    buffer = ScoreBuffer()
    
    while game_in_progress:
        # Get current score from NBA_API
        score_data = nba_api.get_live_score(game_id)
        
        buffer.update(
            score_data['home_score'],
            score_data['away_score'],
            score_data['game_time_seconds']
        )
        
        # Check if we're at 6:00 Q2 (1080 seconds)
        if score_data['game_time_seconds'] >= 1080:
            pattern = buffer.get_pattern()
            
            if pattern is not None:
                # CALL ML MODEL
                prediction = ml_model.predict(pattern)
                
                # Send to betting system
                send_to_betting(prediction)
                
                break  # Only predict once per game
        
        time.sleep(10)  # Check every 10 seconds
```

---

## Critical Timing

**Game Timeline:**
```
0:00 Q1  → Minute 0   (game start)
6:00 Q1  → Minute 6
0:00 Q1  → Minute 12  (end Q1)
6:00 Q2  → Minute 18  ← MAKE PREDICTION HERE
0:00 Q2  → Minute 24  ← What we're predicting (halftime)
```

**Your task:**
1. Start buffering at game start
2. Update buffer every score change
3. At 1080 seconds (6:00 Q2), you have 18 minutes
4. Call ML model
5. Get prediction for minute 24 (halftime)

---

## Data Format Details

### What Each Minute Represents

```python
pattern[0]  = Differential at minute 0  (0:00 Q1 start) = 0
pattern[1]  = Differential at minute 1  (11:00 Q1)
pattern[2]  = Differential at minute 2  (10:00 Q1)
...
pattern[11] = Differential at minute 11 (1:00 Q1)
pattern[12] = Differential at minute 12 (0:00 Q1 = end Q1)
pattern[13] = Differential at minute 13 (11:00 Q2)
...
pattern[17] = Differential at minute 17 (7:00 Q2)
```

**At minute 18 (6:00 Q2):** Make prediction for minute 24 (0:00 Q2 = halftime)

---

## Example NBA_API Code

```python
from nba_api.live.nba.endpoints import scoreboard
from nba_api.live.nba.endpoints import boxscore
import time

def extract_pattern_from_live_game(game_id):
    """
    Extract 18-minute pattern from live NBA game
    """
    pattern = []
    
    while len(pattern) < 18:
        # Get current box score
        box = boxscore.BoxScore(game_id=game_id)
        data = box.get_dict()
        
        # Extract scores
        home_score = data['game']['homeTeam']['score']
        away_score = data['game']['awayTeam']['score']
        
        # Game time
        period = data['game']['period']
        clock = data['game']['gameClock']  # e.g., "06:23"
        
        # Calculate minute
        minute = calculate_minute(period, clock)
        
        if minute <= 17:
            diff = home_score - away_score
            pattern.append(diff)
        
        time.sleep(30)  # Check every 30 seconds
    
    return np.array(pattern)
```

---

## What Happens After Prediction

### Flow:
```
NBA_API (live scores)
    ↓
18-minute pattern
    ↓
ML Model (this MVP)
    ↓
Prediction: +15.0 ± 13.0
    ↓
BetOnline (scrape odds)
    ↓
Edge detection (ML vs market)
    ↓
Risk Management (Kelly, Portfolio, etc.)
    ↓
Place bet
```

**Your responsibility:** Steps 1-3 (get pattern, call model)  
**Next teams:** Steps 4-7 (betting logic)

---

## Testing Your Integration

### Test Pattern (Known Result)

```python
# Test game: POR @ UTA (December 26, 2019)
test_pattern = np.array([0, 2, 3, 8, 7, 7, 9, 9, 10, 7, 9, 12, 15, 13, 9, 8, 13, 12])

# Expected output:
# Ensemble: +12.2 points
# Interval: [-0.8, +25.2]
# Actual: +10.0 (so it's covered!)

prediction = ml_model.predict(test_pattern)
assert -1 < prediction['point_forecast'] < 26, "Prediction out of expected range"
```

---

## Performance Guarantees

**What we guarantee:**
- ✅ Predictions in <100ms
- ✅ 95% of actuals fall within intervals
- ✅ Average error ~5.4 points

**What we DON'T guarantee:**
- ❌ Perfect predictions (impossible)
- ❌ Narrow intervals (uncertainty is real)
- ❌ Performance on 2025 data (may need retraining)

---

## Files You Need

**Minimum for deployment:**
1. `Models/dejavu_k500.pkl` (55 MB)
2. `Models/lstm_best.pth` (200 KB)
3. `Models/lstm_normalization.pkl` (5 KB) ← CRITICAL!
4. `Models/conformal_predictor.pkl` (10 KB)
5. `Code/*.py` (all 4 Python files)

**Total:** ~56 MB (easily deployable)

---

## Contact Points

**ML Model Issues:**
- See `MVP_COMPLETE_SPECIFICATIONS.md`
- Check `Results/` folder for evaluation metrics

**Integration Questions:**
- Input format: 18-number array
- Output format: JSON with forecast + interval
- Timing: Call at 6:00 Q2

---

**Ready to integrate with NBA_API! Model is trained, tested, and documented.**

*Next: Build NBA_API live score buffering (Folder 2)*

