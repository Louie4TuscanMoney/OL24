# 📅 SATURDAY LIVE TEST SCHEDULE - October 19, 2025

**Purpose:** Final validation of complete system before Monday launch  
**Duration:** 4-6 hours (as games become available)  
**Goal:** Prove system works end-to-end on real 2025 NBA data

---

## 🎯 TEST OBJECTIVES

### Primary Goals:
1. ✅ **Validate full pipeline** (NBA API → Model → Risk → Output)
2. ✅ **Calculate real 2025 MAE** (expect 6-8 points vs 5.39 training)
3. ✅ **Confirm no scraper blocking** (test with real traffic)
4. ✅ **Verify performance** (< 6 seconds end-to-end)
5. ✅ **Test error handling** (what breaks, how to fix)

### Secondary Goals:
- Document any issues for quick fixes
- Build confidence in system reliability
- Practice operating procedures
- Calibrate expectations for Monday

---

## ⏰ TEST SCHEDULE

### Morning Setup (9:00 AM - 10:00 AM)
**Duration:** 1 hour  
**Location:** This MacBook

```bash
cd "/Users/test/Desktop/Tuscan Money/Ontologic XYZ/ML Research/Action"

# 1. Verify all components (10 min)
python3 -c "
print('Quick System Check:')
print('1. NBA API:', end=' ')
from nba_api.live.nba.endpoints import scoreboard
print('✅')

print('2. ML Model:', end=' ')
import sys; sys.path.insert(0, '1. ML/1. Dejavu')
from dejavu_model import DejavuForecaster
print('✅')

print('3. Playwright:', end=' ')
import playwright
print('✅')

print('\\nAll systems GO! ✅')
"

# 2. Check today's NBA schedule
python3 -c "
from nba_api.live.nba.endpoints import scoreboard
games = scoreboard.ScoreBoard().get_dict()['scoreboard']['games']
print(f'Games today: {len(games)}')
for g in games:
    away = g['awayTeam']['teamTricode']
    home = g['homeTeam']['teamTricode']
    status = g['gameStatusText']
    print(f'  {away} @ {home} - {status}')
"

# 3. Prepare test script
chmod +x 🔴_LIVE_TEST_SATURDAY.py
```

---

### Test Window 1: Early Games (1:00 PM - 4:00 PM ET)
**Expected:** 2-3 preseason games

#### 1:00 PM - Game Start
- Start monitoring script
- Wait for games to begin
- System on standby

#### ~1:15 PM - 18-Minute Mark (First Game)
```bash
# Run live test
python3 🔴_LIVE_TEST_SATURDAY.py
```

**What to Watch:**
- ✅ NBA API fetches game data
- ✅ Model makes prediction
- ✅ Scraper gets odds (no blocking!)
- ✅ System completes in < 6 seconds
- ⚠️ Note any errors or warnings

#### ~1:30 PM - Halftime (First Game)
- **CRITICAL:** Compare prediction to actual halftime score
- Calculate error: `|predicted - actual|`
- Record result for MAE calculation

#### ~2:00 PM - Game Ends
- Record final score
- Document prediction accuracy
- Note any issues

**Repeat for additional games in this window**

---

### Test Window 2: Afternoon Games (4:00 PM - 7:00 PM ET)
**Expected:** 2-3 preseason games

Same process as Window 1:
- Monitor game starts
- Make predictions at 18-minute mark
- Validate at halftime
- Record results

**Running Calculations:**
- Update MAE after each game
- Track: `MAE = average of all errors`
- Target: MAE < 8 points

---

### Test Window 3: Evening Games (7:00 PM - 10:00 PM ET)
**Expected:** 2-3 preseason games

**Focus:** Consistency & Reliability
- System should be stable by now
- Any issues should be documented
- Procedures should be smooth

---

### Evening Summary (10:00 PM - 11:00 PM)
**Duration:** 1 hour

```bash
# Create summary report
python3 -c "
import json
from datetime import datetime

summary = {
    'date': 'Oct 19, 2025',
    'games_tested': 8,  # Update with actual
    'predictions_made': 8,
    'mae': 6.5,  # Update with actual
    'system_uptime': '95%',
    'issues_found': [],
    'ready_for_launch': True
}

print('='*80)
print('SATURDAY TEST SUMMARY')
print('='*80)
for key, val in summary.items():
    print(f'{key}: {val}')
"
```

---

## 📊 DATA TO COLLECT

### For Each Game:
```python
game_result = {
    'game_id': '...',
    'teams': 'LAL @ GSW',
    'prediction_time': '1:15 PM',
    'pattern': [0, -2, -1, ...],  # 18-minute differentials
    'predicted_halftime': +8.5,
    'actual_halftime': +12,
    'error': 3.5,
    'bet_size': 450,
    'scraper_time_ms': 1850,
    'model_time_ms': 85,
    'total_time_ms': 2100,
    'issues': []
}
```

### Aggregate Metrics:
- **MAE:** Average of all errors
- **Success Rate:** % of predictions made successfully
- **Scraper Success:** % of times odds fetched
- **Performance:** Average end-to-end time

---

## ✅ SUCCESS CRITERIA

### Must Have (Go/No-Go for Monday):
- [ ] Made at least 3 predictions successfully
- [ ] MAE < 10 points (ideally 6-8)
- [ ] No scraper blocking (or have manual backup)
- [ ] End-to-end time < 10 seconds (target 6 sec)
- [ ] System stable (no crashes)

### Nice to Have:
- [ ] MAE < 8 points
- [ ] End-to-end time < 6 seconds
- [ ] Scraper 100% successful
- [ ] Zero manual interventions needed

---

## 🚨 ISSUE TRACKING

### If Scraper Gets Blocked:
**Action:**
1. Note time and frequency
2. Add delays (10-15 seconds)
3. Switch to manual odds entry
4. Document for Monday backup plan

### If MAE > 10:
**Action:**
1. Check if specific game types causing issues
2. Review similar historical games
3. Consider model recalibration
4. May need to be more conservative Week 1

### If System Too Slow (>10 sec):
**Action:**
1. Profile which component is slow
2. Check internet connection
3. Reduce scraper wait times
4. Accept slightly slower for Week 1

### If Model Crashes:
**Action:**
1. Check error logs
2. Test model loading separately
3. Verify data format
4. Have backup: Use simpler prediction method

---

## 📝 TESTING CHECKLIST

### Pre-Test (Morning):
- [ ] All dependencies installed
- [ ] ML model loads successfully
- [ ] NBA API connects
- [ ] Scraper initializes
- [ ] Test script ready
- [ ] Laptop charged
- [ ] Internet stable

### During Test (Each Game):
- [ ] Game detected by NBA API
- [ ] Data extracted correctly
- [ ] Prediction made
- [ ] Odds fetched (or manual entry)
- [ ] Bet size calculated
- [ ] Results logged
- [ ] Halftime compared
- [ ] Error recorded

### Post-Test (Evening):
- [ ] All results compiled
- [ ] MAE calculated
- [ ] Issues documented
- [ ] Fixes identified
- [ ] Summary report created
- [ ] Monday readiness assessed

---

## 🎯 EXPECTED OUTCOMES

### Best Case (90%):
- ✅ 8+ games tested
- ✅ MAE: 6.0-7.0 points
- ✅ No scraper issues
- ✅ System fast and stable
- ✅ **Ready for Monday launch with confidence!**

### Realistic Case (70%):
- ✅ 5-6 games tested
- ✅ MAE: 7.0-8.5 points
- ⚠️ Minor scraper delays
- ✅ System stable with small fixes
- ✅ **Ready for Monday, some caution**

### Worst Case (30%):
- ⚠️ 2-3 games tested
- ⚠️ MAE: 9-10 points
- ❌ Scraper blocked (use manual)
- ⚠️ Several bugs found
- ⚠️ **Can still launch but in paper-trade mode**

---

## 🔄 ITERATION PLAN

### After Game 1:
If issues found → Fix immediately before Game 2

### After Game 3:
Calculate interim MAE → Adjust strategy if needed

### After Game 5:
Final calibration → Make go/no-go decision

### After All Games:
Create final report → Prepare for Monday

---

## 📞 QUICK REFERENCE COMMANDS

### Check for Live Games:
```bash
python3 -c "from nba_api.live.nba.endpoints import scoreboard; print(len(scoreboard.ScoreBoard().get_dict()['scoreboard']['games']), 'games')"
```

### Run Live Test:
```bash
python3 🔴_LIVE_TEST_SATURDAY.py
```

### Check Model Loading:
```bash
python3 🧪_TEST_REAL_PREDICTION.py
```

### Test Scraper:
```bash
cd "3. Bet Online/1. Scrape" && python3 test_scraper.py
```

---

## 🎊 POST-TEST DELIVERABLES

### 1. Performance Report
- Games tested
- MAE achieved
- System reliability
- Issues encountered

### 2. Accuracy Validation
- Compare predicted vs actual
- Calculate confidence intervals
- Identify patterns in errors

### 3. Issue Log
- What broke
- How we fixed it
- Preventive measures

### 4. Monday Readiness Assessment
- GO / NO-GO decision
- Backup plans documented
- Confidence level
- Final checklist

---

## 🚀 FINAL ASSESSMENT

### Saturday Night (11:00 PM):
Ask yourself:

1. **Can I make predictions?** YES/NO
2. **Do predictions seem reasonable?** YES/NO
3. **Can I get odds?** YES/NO
4. **Can I calculate bet sizes?** YES/NO
5. **Did system crash?** NO/YES
6. **Am I comfortable launching Monday?** YES/NO

**If 5/6 are YES:** ✅ **LAUNCH MONDAY**  
**If 4/6 are YES:** ⚠️ **LAUNCH IN PAPER-TRADE MODE**  
**If <4 are YES:** ❌ **DELAY TO WEEK 2**

---

## 💪 MINDSET FOR TOMORROW

**Remember:**
- Week 1 is for learning, not perfection
- Any live test data is valuable
- Issues are expected and OK
- Fast iteration > careful planning
- Monday launch is about starting, not finishing

**You Got This!** 🏀🚀

---

**Created:** October 18, 2025, 8:30 PM  
**Test Date:** Saturday, October 19, 2025  
**Test Window:** 9 AM - 11 PM (as games occur)  
**Expected Duration:** 6-8 hours  
**Launch:** Monday, October 21, 2025

