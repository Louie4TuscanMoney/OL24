# 🔴 WEEKEND LIVE TESTING PLAN

**Saturday + Sunday = Test on EVERY preseason game**  
**Goal: PROVE system works before Monday launch**  
**Strategy: Real games, real patterns, real predictions, real validation**

---

## 🎯 THE PLAN

### **Saturday, October 19:**
✅ Test on EVERY preseason game  
✅ Extract real patterns at 18-minute mark  
✅ Make real predictions  
✅ Calculate REAL 2025 MAE  
✅ Find what breaks  
✅ Fix it immediately  

### **Sunday, October 20:**
✅ Test fixes on any remaining games  
✅ Validate everything works  
✅ Make final go/no-go decision  
✅ Prepare for Monday  

---

## ⏰ SATURDAY SCHEDULE (All Day Testing)

### **Morning (9:00 AM) - Setup:**
```bash
cd "/Users/test/Desktop/Tuscan Money/Ontologic XYZ/ML Research/Action"

# Open 3 terminal windows:
# Window 1: Launch system
# Window 2: Logs
# Window 3: Manual commands

# Start the system
python3 launch_monday.py
```

### **Check Schedule:**
```bash
# See when games start
python3 -c "
from nba_api.live.nba.endpoints import scoreboard
games = scoreboard.ScoreBoard().get_dict()['scoreboard']['games']
print(f'{len(games)} games today')
for g in games:
    away = g['awayTeam']['teamTricode']
    home = g['homeTeam']['teamTricode']
    status = g['gameStatusText']
    print(f'{away} @ {home} - {status}')
"
```

### **For EACH Game (Repeat All Day):**

**When game starts:**
```
Track game clock
Wait for 18-minute mark (6:00 2Q)
```

**At 18 minutes:**
```bash
# 1. Extract pattern manually (first time)
python3 -c "
from nba_api.stats.endpoints import playbyplayv2
game_id = '0012500XXX'  # Get from live API

# Get play-by-play
pbp = playbyplayv2.PlayByPlayV2(game_id=game_id)
plays = pbp.get_data_frames()[0]

# Extract differentials at each minute
# (Use parsing code from tonight)
pattern = extract_18min_pattern(plays)
print(f'Pattern: {pattern}')
"

# 2. Make prediction
python3 -c "
from game_engine import GameEngine
import numpy as np

engine = GameEngine()
pattern = np.array([...])  # From above

result = engine.predict(pattern, return_details=True)

print(f'PREDICTIONS:')
print(f'  Halftime: {result[\"halftime\"]:+.1f}')
print(f'  Final: {result[\"final\"]:+.1f}')
print(f'  Confidence: {result[\"confidence\"]}')
print(f'  Should bet: {result[\"should_bet\"]}')
"

# 3. Record prediction
# Note: Current score, prediction, time
```

**At halftime (~24 minutes):**
```
Get actual halftime score
Compare to Branch A prediction
Calculate error
Log result
```

**At game end (~48 minutes):**
```
Get actual final score
Compare to Branch B prediction
Calculate error
Update feedback loop
Log final result
```

### **Running Tally:**
```python
# Keep updating this after each game
games_tested = 0
halftime_errors = []
final_errors = []

# After each game:
games_tested += 1
halftime_mae = np.mean(halftime_errors)
final_mae = np.mean(final_errors)

print(f'After {games_tested} games:')
print(f'  Halftime MAE: {halftime_mae:.2f}')
print(f'  Final MAE: {final_mae:.2f}')
```

---

## 📊 DATA TO COLLECT (Each Game)

```python
game_result = {
    # Game info
    'game_id': '0012500XXX',
    'teams': 'LAL @ CHI',
    'date': 'Oct 19, 2025',
    'time': '7:00 PM ET',
    
    # At 18 minutes
    '18min_score': [55, 48],  # LAL 55, CHI 48
    '18min_diff': +7,
    'pattern': [0, -2, 1, 3, 5, 6, 7],  # Full 18 values
    
    # Predictions
    'pred_halftime': +8.5,
    'pred_final': +12.0,
    'confidence': 'HIGH',
    'avg_neighbor_dist': 2.4,
    'should_bet': True,
    
    # Actual outcomes
    'actual_halftime_score': [62, 56],
    'actual_halftime_diff': +6,
    'actual_final_score': [104, 98],
    'actual_final_diff': +6,
    
    # Errors
    'error_halftime': 2.5,
    'error_final': 6.0,
    
    # Betting
    'odds_1h': -110,
    'odds_fg': -110,
    'bet_1h': True,
    'bet_1h_size': 400,
    'bet_fg': True,
    'bet_fg_size': 300,
    
    # Issues encountered
    'issues': ['Pattern extraction took 30 sec', 'Had to manually format'],
    
    # What worked / didn't work
    'what_worked': ['NBA API fast', 'Prediction accurate'],
    'what_failed': ['Auto pattern extract broke', 'Dashboard not updating']
}
```

---

## ✅ SUCCESS CRITERIA (By Sunday Night)

### **Minimum (Must Have):**
- [ ] Tested on at least 5 games
- [ ] Got real MAE (halftime + final)
- [ ] Full pipeline worked at least once
- [ ] Know what breaks and how to fix
- [ ] Documented manual workarounds

### **Ideal (Want to Have):**
- [ ] Tested on 10+ games  
- [ ] MAE < 10 on both branches
- [ ] Full automation works
- [ ] No manual intervention needed
- [ ] System stable

### **Go/No-Go Decision:**
```
If 5/5 minimum criteria met:
  → LAUNCH Monday (possibly manual mode)

If ideal criteria met:
  → LAUNCH Monday (full automation)

If <3 minimum criteria:
  → Paper trade Week 1, real betting Week 2
```

---

## 🔧 EXPECTED ISSUES & FIXES

### **Issue #1: Pattern Extraction Fails**
**Symptom:** Can't convert play-by-play to 18-value pattern  
**Fix:** Manual extraction first game, then automate  
**Time:** 30-60 minutes  
**Backup:** Use simplified pattern (recent 5 min average)

### **Issue #2: Odds Parsing Fails**
**Symptom:** Scraper gets HTML but can't extract spread  
**Fix:** Debug selectors, update parsing logic  
**Time:** 30 minutes  
**Backup:** Manual odds entry from BetOnline website

### **Issue #3: System Too Slow**
**Symptom:** Takes >10 seconds to process  
**Fix:** Optimize hot paths, add caching  
**Time:** 20 minutes  
**Backup:** Reduce polling frequency to 20 seconds

### **Issue #4: Dashboard Not Updating**
**Symptom:** Dashboard shows stale data  
**Fix:** Add backend API or WebSocket  
**Time:** 1 hour  
**Backup:** Refresh browser manually

### **Issue #5: Model Predictions Seem Wrong**
**Symptom:** Predictions way off actual scores  
**Fix:** Verify pattern format, check normalization  
**Time:** 30 minutes  
**Backup:** Use conservative betting only

---

## 📋 SATURDAY TEST CHECKLIST (Per Game)

### **Pre-Game (When Game Detected):**
- [ ] System detects game via NBA API
- [ ] Displays game on dashboard/terminal
- [ ] Game ID captured
- [ ] Teams identified

### **At 18 Minutes (6:00 2Q):**
- [ ] System knows it's 18 minutes (how?)
- [ ] Fetches play-by-play data
- [ ] Extracts 18-minute pattern
- [ ] Pattern is correct format (18 values)
- [ ] Feeds to model
- [ ] Model returns predictions
- [ ] Both branches work (halftime + final)
- [ ] Confidence calculated
- [ ] Filters applied
- [ ] Risk calculation works
- [ ] Bet size calculated
- [ ] Displayed clearly

### **At Halftime (~24 minutes):**
- [ ] Get actual halftime score
- [ ] Compare to Branch A prediction
- [ ] Calculate error
- [ ] Log result
- [ ] Update running MAE

### **At Game End (~48 minutes):**
- [ ] Get actual final score
- [ ] Compare to Branch B prediction
- [ ] Calculate error
- [ ] Update feedback loop
- [ ] Log final result
- [ ] System learns from outcome

### **Post-Game:**
- [ ] Review what worked
- [ ] Document what broke
- [ ] Fix critical issues before next game
- [ ] Update system for next test

---

## 🎯 REALISTIC EXPECTATIONS

### **First Game Saturday:**
```
Expected: 5-10 things will break
Reality: Manual intervention needed
Time: 30-60 min to troubleshoot
Outcome: Learn what needs fixing
```

### **Second Game Saturday:**
```
Expected: 3-5 things still broken
Reality: Fixes applied, fewer issues
Time: 15-30 min intervention
Outcome: System getting better
```

### **Third+ Games Saturday:**
```
Expected: 1-2 minor issues
Reality: Mostly automated
Time: <10 min intervention
Outcome: Confidence building!
```

### **By Sunday:**
```
Expected: System mostly works
Reality: Know exact readiness level
Outcome: Informed go/no-go decision
```

---

## 💡 MANUAL MODE (If Automation Fails)

### **Fallback: Manual Prediction System**
```
1. Watch game on TV/stream
2. Note score at 18 minutes
3. Manually enter pattern into model
4. Get predictions
5. Look up odds on BetOnline
6. Calculate bet size
7. Place bet manually
8. Log in spreadsheet

This is TOTALLY OK for Week 1!
Automation is a bonus, not required.
```

---

## 🔥 THE CRITICAL QUESTION

**Tomorrow at 1 PM when first game starts:**

"Can I extract the 18-minute pattern and get a prediction?"

**IF YES:**
- ✅ System fundamentally works
- ✅ Launch Monday
- ✅ Fix polish issues Week 1

**IF NO:**
- ⚠️ Need to debug extraction
- ⚠️ May need manual mode
- ⚠️ Paper trade if can't fix

---

## 🚀 BOTTOM LINE

### **Tonight Built:** Infrastructure (85% done)
### **Tomorrow Tests:** Integration (the real validation)
### **Sunday Prepares:** Final readiness
### **Monday Launches:** For real!

**Saturday is THE day that matters.**

**Not tomorrow. TOMORROW.** 🎯

---

## 💪 YOU'RE ACTUALLY ON TRACK

**Original 6-day plan:**
- Day 1: Component testing ✅ (You did tonight)
- Day 2: Bug fixes ✅ (You did tonight)
- Day 3: Live testing ← TOMORROW (Saturday)
- Day 4: Calibration ← SUNDAY
- Day 5: Final prep ← SUNDAY  
- Day 6: Launch ← MONDAY

**You compressed Days 1-2 into tonight.**  
**You're doing Day 3 tomorrow.**  
**You're ON SCHEDULE!**

---

**Sleep. Test tomorrow. Launch Monday.** 😴🧪🚀

**Real readiness: 65% → Will be 90%+ after Saturday!** 💪

