# 🚀 PREPARING FOR 5:00 PM MAMBA TEST

**Time:** ~5:00 PM (in ~8 minutes)  
**Expected:** Q2 6:00 trigger for ORL @ CHA

---

## ✅ DEPLOYED FEATURES

### 1. Background Snapshot Collection
- ✅ Runs every 60 seconds automatically
- ✅ No manual cron needed
- ✅ Collects snapshots for ALL live games

### 2. Mamba Visualization
- ✅ MambaLiveWidget (patterns)
- ✅ LiveSnapshotsWidget (score progression)
- ✅ WinProbabilityWidget (probability timeline)
- ✅ MambaFeaturesWidget (33 features)
- ✅ All widgets auto-refresh every 30 seconds

### 3. ESPN Stability
- ✅ 5 retries with exponential backoff
- ✅ 5-second cache prevents rate limiting
- ✅ 5-minute fallback during outages
- ✅ Debug logging for quarter transitions
- ✅ Error isolation per game

### 4. Multiple Games Support
- ✅ Each game processed independently
- ✅ One game failure doesn't affect others
- ✅ Complete error logging

---

## 🎯 WHAT TO EXPECT AT 5:00 PM

When ORL @ CHA reaches Q2 6:00:

1. **Cron runs** (every 60 seconds)
2. **Detects Q2 6:00** trigger condition
3. **Extracts 33 features** from snapshots
4. **Stores prediction** in mamba_game_cache
5. **Frontend widgets** automatically show:
   - Mamba prediction with confidence
   - 33 features breakdown
   - Win probability timeline
   - Scoring patterns

---

## 🧪 TO VERIFY IT'S WORKING

### Check Railway Logs
Look for:
```
🤖 MAMBA AUTONOMOUS CRON
📡 Found 1 live games
   🏀 ORL @ CHA - Q2 6:00
      ⚡ MAMBA Q2 6:00 TRIGGER DETECTED!
      📊 Extracting features...
      ✅ MAMBA PREDICTION (Q2 6:00): +X.X
```

### Check Frontend
Go to: https://ontologicxyz.com/game/401809993
Should see:
- ✅ MambaLiveWidget with prediction
- ✅ LiveSnapshotsWidget with chart
- ✅ WinProbabilityWidget with timeline
- ✅ MambaFeaturesWidget with 33 features

---

## 🔧 IF IT DOESN'T WORK

### Debug Steps:

1. **Check if background thread running**
   - Look for "Background Mamba collector starting" in logs
   - Should see cron runs every 60 seconds

2. **Check snapshots**
   ```bash
   python3 📊_MAMBA_STATUS_REPORT.py
   ```
   Should show multiple snapshots

3. **Check ESPN is returning game**
   ```bash
   python3 🏀_CHECK_LIVE_GAMES.py
   ```
   Should show ORL @ CHA live

4. **Check Mamba cache**
   ```bash
   curl https://ol24-production.up.railway.app/api/ml/prediction/401809993 | jq
   ```
   Should return prediction after trigger

---

**✅ SYSTEM IS READY FOR 5:00 PM MAMBA TEST!**
