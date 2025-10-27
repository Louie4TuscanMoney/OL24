# ✅ QUICK START - Saturday Morning

**Print this and follow step by step!**

---

## ☕ STEP 1: Setup (9:00 AM)

```bash
# Terminal 1
cd "/Users/test/Desktop/Tuscan Money/Ontologic XYZ/ML Research/Action"

# Check everything works
python3 -c "
from game_engine import GameEngine
import numpy as np
engine = GameEngine()
result = engine.predict(np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]))
print('✅ System ready!')
print(f'Test prediction: {result[\"halftime\"]:+.1f}')
"
```

**✅ If you see prediction → Ready!**  
**❌ If error → Fix before continuing**

---

## 🏀 STEP 2: When Game Hits 18 Minutes

### **A. Get Game ID:**
```bash
python3 -c "from nba_api.live.nba.endpoints import scoreboard; g = scoreboard.ScoreBoard().get_dict()['scoreboard']['games'][0]; print(f'ID: {g[\"gameId\"]} - {g[\"awayTeam\"][\"teamTricode\"]} @ {g[\"homeTeam\"][\"teamTricode\"]}')"
```
**Write down:** Game ID = ________________

### **B. Extract Pattern:**
```bash
# Replace XXXXX with game ID from above
python3 << 'PATTERN_SCRIPT'
from nba_api.stats.endpoints import playbyplayv2
import pandas as pd
import numpy as np

game_id = '0012500XXXXX'  # REPLACE THIS

pbp = playbyplayv2.PlayByPlayV2(game_id=game_id)
plays_df = pbp.get_data_frames()[0]

differentials = [0]

for idx, play in plays_df.iterrows():
    period = play['PERIOD']
    if period > 2: break
    
    pctimestring = play['PCTIMESTRING']
    score_margin = play['SCOREMARGIN']
    
    if pd.notna(pctimestring) and pd.notna(score_margin):
        parts = pctimestring.split(':')
        mins_remaining = int(parts[0])
        secs_remaining = int(parts[1])
        
        if period == 1:
            elapsed = 12 - mins_remaining - (secs_remaining / 60.0)
        else:
            elapsed = 12 + (12 - mins_remaining - (secs_remaining / 60.0))
        
        if elapsed > 18: break
        
        diff = 0 if score_margin == 'TIE' else int(score_margin)
        minute = int(elapsed)
        
        while len(differentials) <= minute:
            differentials.append(differentials[-1])
        differentials[minute] = diff

while len(differentials) < 18:
    differentials.append(differentials[-1])

pattern = differentials[:18]
print(f'pattern = {pattern}')
PATTERN_SCRIPT
```
**Copy the pattern output!**

### **C. Make Prediction:**
```bash
python3 << 'PREDICT_SCRIPT'
from game_engine import GameEngine
import numpy as np

# PASTE pattern from above here:
pattern = np.array([0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18])

engine = GameEngine()
result = engine.predict(pattern, return_details=True)

print('PREDICTIONS:')
print(f'  Halftime: {result["halftime"]:+.1f}')
print(f'  Final: {result["final"]:+.1f}')
print(f'  Confidence: {result["confidence"]}')
PREDICT_SCRIPT
```

**Write down predictions:**  
Halftime: ______  
Final: ______

### **D. Compare at Halftime & Final:**
```bash
# At halftime
python3 -c "
from nba_api.live.nba.endpoints import boxscore
box = boxscore.BoxScore('0012500XXXXX')  # Your game ID
g = box.get_dict()['game']
print(f'Halftime: {g[\"awayTeam\"][\"score\"]} - {g[\"homeTeam\"][\"score\"]}')
print(f'Diff: {g[\"homeTeam\"][\"score\"] - g[\"awayTeam\"][\"score\"]:+d}')
"

# Calculate error
predicted = ___  # Your prediction
actual = ___     # From above
error = abs(predicted - actual)
print(f'Error: {error:.1f}')
```

---

## 📝 SIMPLE LOG SHEET

**Print this and fill manually:**

```
SATURDAY TEST LOG
=================

Game 1: _____ @ _____ (_____ PM)
  Pattern extracted: YES / NO / PARTIAL
  Halftime pred: _____ Actual: _____ Error: _____
  Final pred: _____ Actual: _____ Error: _____
  Issues: _________________________________

Game 2: _____ @ _____ (_____ PM)
  Pattern extracted: YES / NO / PARTIAL
  Halftime pred: _____ Actual: _____ Error: _____
  Final pred: _____ Actual: _____ Error: _____
  Issues: _________________________________

Game 3: _____ @ _____ (_____ PM)
  Pattern extracted: YES / NO / PARTIAL
  Halftime pred: _____ Actual: _____ Error: _____
  Final pred: _____ Actual: _____ Error: _____
  Issues: _________________________________

(Continue for all games...)

END OF DAY:
Total games: _____
Games successful: _____
Halftime MAE: _____
Final MAE: _____

Monday Decision: LAUNCH / PAPER TRADE / DELAY
```

---

## 🎯 ONE-PAGE CHEAT SHEET

**PRINT THIS:**

```
═══════════════════════════════════════
   SATURDAY TEST QUICK REFERENCE
═══════════════════════════════════════

1. Get Game ID:
   python3 -c "from nba_api.live.nba.endpoints import scoreboard; print(scoreboard.ScoreBoard().get_dict()['scoreboard']['games'][0]['gameId'])"

2. Extract Pattern (at 18 min):
   Use full script from Step 2B above
   Copy pattern output

3. Predict:
   python3 game_engine.py
   (Then enter pattern when prompted)

4. Check Halftime:
   Use NBA.com or script from Step D

5. Check Final:
   Use NBA.com or script from Step D

6. Log Everything:
   Write in log sheet

REPEAT FOR EVERY GAME!

═══════════════════════════════════════
```

---

## 🎊 FINAL PREP TONIGHT

### **Do Before Bed:**
- [x] Read this document fully
- [ ] Print game checklist (10 copies)
- [ ] Print quick reference
- [ ] Set alarms (9 AM, 12:30, 3:30, 6:30 PM)
- [ ] Prepare log sheet
- [ ] Charge MacBook
- [ ] Clear desk space
- [ ] Get good sleep! 😴

### **Tomorrow Morning (9 AM):**
- [ ] Coffee
- [ ] Read quick start
- [ ] Set up terminals
- [ ] Run system check
- [ ] **Wait for games and TEST!**

---

**This is your playbook for tomorrow. Follow it and you'll know EXACTLY where you stand!** 🎯

**Sleep now. Execute tomorrow. Launch Monday!** 💪🏀🚀

