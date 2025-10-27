# 🖨️ PRINT THIS - Saturday Test Protocol

**COPY/PASTE COMMANDS - NO THINKING REQUIRED**

---

## ☕ MORNING (9:00 AM)

### Open Terminal and run:
```bash
cd "/Users/test/Desktop/Tuscan Money/Ontologic XYZ/ML Research/Action"

# Test system
python3 -c "from game_engine import GameEngine; import numpy as np; e=GameEngine(); r=e.predict(np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17])); print(f'✅ Ready! Test: {r[\"halftime\"]:+.1f}')"
```

✅ See "✅ Ready!"? → Continue  
❌ See error? → Fix it first

---

## 🏀 WHEN GAME HITS 18 MIN (6:00 2Q)

### **DO THIS IN ORDER:**

### 1️⃣ GET GAME ID (30 seconds)
```bash
python3 -c "from nba_api.live.nba.endpoints import scoreboard; g=scoreboard.ScoreBoard().get_dict()['scoreboard']['games'][0]; print(f'{g[\"gameId\"]} - {g[\"awayTeam\"][\"teamTricode\"]} @ {g[\"homeTeam\"][\"teamTricode\"]}')"
```
**Write here:** Game ID = ___________________

---

### 2️⃣ EXTRACT PATTERN (2 minutes)
```bash
# Copy this ENTIRE block, replace XXXXX with Game ID from above
python3 << 'END'
from nba_api.stats.endpoints import playbyplayv2
import pandas as pd

game_id = '0012500XXXXX'  # ← PUT GAME ID HERE

pbp = playbyplayv2.PlayByPlayV2(game_id=game_id)
plays = pbp.get_data_frames()[0]

diffs = [0]
for idx, play in plays.iterrows():
    period = play['PERIOD']
    if period > 2: break
    
    time_str = play['PCTIMESTRING']
    margin = play['SCOREMARGIN']
    
    if pd.notna(time_str) and pd.notna(margin):
        parts = time_str.split(':')
        mins_left = int(parts[0])
        secs_left = int(parts[1])
        
        if period == 1:
            elapsed = 12 - mins_left - secs_left/60.0
        else:
            elapsed = 12 + 12 - mins_left - secs_left/60.0
        
        if elapsed > 18: break
        
        diff = 0 if margin == 'TIE' else int(margin)
        min_idx = int(elapsed)
        
        while len(diffs) <= min_idx:
            diffs.append(diffs[-1])
        diffs[min_idx] = diff

while len(diffs) < 18:
    diffs.append(diffs[-1])

pattern = diffs[:18]
print(f'COPY THIS LINE:')
print(f'pattern = {pattern}')
END
```
**Copy the pattern line!**

---

### 3️⃣ MAKE PREDICTION (1 minute)
```bash
# Paste pattern from above, then run
python3 << 'END'
from game_engine import GameEngine
import numpy as np

pattern = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]  # ← PASTE REAL PATTERN HERE

engine = GameEngine()
r = engine.predict(np.array(pattern), return_details=True)

print('='*60)
print('PREDICTIONS:')
print(f'  Halftime: {r["halftime"]:+.1f}')
print(f'  Final: {r["final"]:+.1f}')
print(f'  Confidence: {r["confidence"]}')
print('='*60)
END
```

**Write here:**  
Halftime pred: ___________  
Final pred: ___________  
Time: ___________

---

### 4️⃣ CHECK HALFTIME (~12 min later)
```bash
# When game reaches halftime
python3 -c "from nba_api.live.nba.endpoints import boxscore; g=boxscore.BoxScore('0012500XXXXX').get_dict()['game']; print(f'Halftime: {g[\"homeTeam\"][\"score\"]} - {g[\"awayTeam\"][\"score\"]} = {g[\"homeTeam\"][\"score\"] - g[\"awayTeam\"][\"score\"]:+d}')"
```

**Calculate error:**  
Predicted: _______  
Actual: _______  
Error: _______ (= |predicted - actual|)

---

### 5️⃣ CHECK FINAL (~90 min later)
```bash
# When game ends
python3 -c "from nba_api.live.nba.endpoints import boxscore; g=boxscore.BoxScore('0012500XXXXX').get_dict()['game']; print(f'Final: {g[\"homeTeam\"][\"score\"]} - {g[\"awayTeam\"][\"score\"]} = {g[\"homeTeam\"][\"score\"]} - {g[\"awayTeam\"][\"score\"]:+d}')"
```

**Calculate error:**  
Predicted: _______  
Actual: _______  
Error: _______

---

## 📝 SIMPLE LOG (Fill This Out)

```
GAME 1: _____ @ _____ 
  18-min pattern extracted: YES / NO
  Halftime pred: _____ actual: _____ error: _____
  Final pred: _____ actual: _____ error: _____
  
GAME 2: _____ @ _____
  18-min pattern extracted: YES / NO  
  Halftime pred: _____ actual: _____ error: _____
  Final pred: _____ actual: _____ error: _____

GAME 3: _____ @ _____
  18-min pattern extracted: YES / NO
  Halftime pred: _____ actual: _____ error: _____
  Final pred: _____ actual: _____ error: _____

GAME 4: _____ @ _____
  18-min pattern extracted: YES / NO
  Halftime pred: _____ actual: _____ error: _____
  Final pred: _____ actual: _____ error: _____

GAME 5: _____ @ _____
  18-min pattern extracted: YES / NO
  Halftime pred: _____ actual: _____ error: _____
  Final pred: _____ actual: _____ error: _____

END OF DAY:
  Games tested: _____
  Halftime MAE: _____ (average all halftime errors)
  Final MAE: _____ (average all final errors)
  
  Major issues found:
  1. _________________________________
  2. _________________________________
  3. _________________________________
  
  Monday decision: LAUNCH / PAPER / DELAY
```

---

## 🎯 END OF DAY (11 PM)

### Calculate Final MAE:
```bash
python3 -c "
import numpy as np

# Put your errors here
halftime_errors = [2.5, 3.0, 4.5, 6.0, 2.0]  # FILL IN REAL VALUES
final_errors = [6.0, 8.0, 5.0, 12.0, 7.0]    # FILL IN REAL VALUES

h_mae = np.mean(halftime_errors)
f_mae = np.mean(final_errors)

print(f'FINAL RESULTS:')
print(f'  Games: {len(halftime_errors)}')
print(f'  Halftime MAE: {h_mae:.2f}')
print(f'  Final MAE: {f_mae:.2f}')

if h_mae < 8 and f_mae < 12:
    print(f'  ✅ EXCELLENT - Launch Monday!')
elif h_mae < 10 and f_mae < 15:
    print(f'  ✅ GOOD - Launch conservatively')
else:
    print(f'  ⚠️  HIGH - Consider paper trade')
"
```

---

## 🚨 IF SOMETHING BREAKS

### Pattern Extraction Fails:
```bash
# Manually create pattern from scores
# If you can't automate, do this:
# Look at score every minute from 0-18
# Write down: [0, -2, -1, 1, 3, 5, etc.]
# This is OK for testing!
```

### Model Crashes:
```bash
# Check pattern
python3 -c "
pattern = [0,1,2]  # Your pattern
print(f'Length: {len(pattern)} (need 18)')
print(f'Type: {type(pattern)}')
print(f'Values: {pattern}')
"
# Must be 18 values!
```

### NBA API Fails:
```bash
# Use NBA.com manually
# Note scores at halftime and final
# Still validates your predictions!
```

---

## ✅ TONIGHT'S PREP CHECKLIST

Before bed, do these:
- [ ] Read full Saturday guide once
- [ ] Understand the 5-step flow (get ID → extract → predict → check HT → check final)
- [ ] Set alarms (9 AM, 12:30 PM, 3:30 PM, 6:30 PM)
- [ ] Charge MacBook fully
- [ ] Know where to copy/paste commands
- [ ] Print this sheet
- [ ] **Get good sleep!** 😴

---

## 🎯 ONE SENTENCE SUMMARY

**Tomorrow: Copy/paste these commands for every game, write down the errors, calculate MAE at 11 PM, decide if you're launching Monday.**

**That's it!** 🎯

---

**Sleep now. Execute tomorrow. You got this!** 💪

