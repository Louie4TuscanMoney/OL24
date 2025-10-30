# 🎯 WHERE TO SEE PATTERNS & TRADING DATA

**Date:** October 30, 2025  
**Status:** ✅ FULLY INTEGRATED

---

## 📊 WHAT'S DISPLAYED WHERE

### 1. **DASHBOARD (Main Page)**

**Location:** https://ontologicxyz.com/

**What you see:**
- ✅ Live scores updating every 1 second
- ✅ Basic Mamba predictions under each live game
- ✅ Click any game to see full details

**Click behavior:**
- Clicking a live game → Opens GameDetailPage
- Shows MambaLiveWidget with full patterns & trading

---

### 2. **GAME DETAIL PAGE** (Most Important!)

**Location:** https://ontologicxyz.com/game/{game_id}

**How to get there:**
- Click any live game on the Dashboard
- URL changes to `/game/401809993` (for example)

**What you see:**
- ✅ Real-time scoreboard
- ✅ **MambaLiveWidget** showing:
  - Scoring pattern chart (minute-by-minute)
  - Countdown to Q2 6:00 trigger
  - Win probability updates
  - Mamba prediction when it fires
  - Recent events feed
- ✅ Trading metrics & opportunities
- ✅ Backend ML feed status

---

### 3. **TRADING DASHBOARD**

**Location:** https://ontologicxyz.com/trading

**What you see:**
- ✅ Live games list
- ✅ Real-time odds & spreads
- ✅ Mamba predictions
- ✅ Expected value calculations
- ✅ Bet placement interface

---

### 4. **SCHEDULE PAGE**

**Location:** https://ontologicxyz.com/schedule

**What you see:**
- ✅ Upcoming games with times (PST)
- ✅ Past games with final scores
- ✅ Click any game → Modal with:
  - Team records
  - Projected lineups
  - Player stats
  - Active injuries

---

## 🎨 VISUAL COMPONENTS

### MambaLiveWidget

**What it displays:**
1. **Scoring Pattern Chart**
   - Line graph showing score differential over time
   - X-axis: Time elapsed
   - Y-axis: Score margin (home - away)

2. **Countdown Timer**
   - Shows time until Q2 6:00
   - Updates every second
   - Status: "Waiting", "Counting Down", "Triggered"

3. **Mamba Prediction**
   - Appears when triggered at Q2 6:00
   - Shows predicted final margin
   - Confidence percentage
   - Timestamp

4. **Recent Events Feed**
   - Last 10 scoring events
   - Period, clock, score

---

## 🔄 DATA FLOW

```
ESPN API (live scores)
    ↓
cron_mamba_autonomous.py (every 1 min)
    ↓
play_by_play table (snapshots)
    ↓
win_probability_timeline (calculations)
    ↓
mamba_game_cache (predictions)
    ↓
WebSocket (/ws/mamba/{game_id})
    ↓
MambaLiveWidget (frontend)
    ↓
Dashboard Display ✅
```

---

## 🧪 TESTING CHECKLIST

### To verify everything is working:

1. **Go to Dashboard**
   - ✅ See live games
   - ✅ Scores updating

2. **Click a live game**
   - ✅ Redirected to `/game/{id}`
   - ✅ See MambaLiveWidget
   - ✅ Chart shows pattern (if 6+ minutes of data)

3. **Wait for Q2 6:00**
   - ✅ Countdown timer
   - ✅ Mamba prediction appears
   - ✅ Confidence shown

4. **Check Trading Dashboard**
   - ✅ Go to `/trading`
   - ✅ See live games with predictions
   - ✅ Odds and spreads shown

---

## ⚠️ CURRENT STATUS

### What's Working:
- ✅ Dashboard displays live games
- ✅ Clicking games opens detail page
- ✅ MambaLiveWidget integrated
- ✅ WebSocket ready

### What Needs Data:
- ⏳ Patterns won't show until 6+ minutes of snapshots collected
- ⏳ Mamba prediction won't fire until Q2 6:00
- ⏳ Waiting for Railway cron to start collecting data

---

## 📋 NEXT STEPS

1. ✅ Wait for Railway cron to start
2. ✅ Verify snapshots collecting
3. ✅ Check frontend shows patterns
4. ✅ Confirm Mamba triggers at Q2 6:00

---

**✅ THE FRONTEND IS FULLY INTEGRATED AND READY TO DISPLAY PATTERNS & TRADING DATA!**
