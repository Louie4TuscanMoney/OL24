# 🎯 MAMBA LIVE SYSTEM - COMPLETE DEPLOYMENT

**Complete system for autonomous Mamba predictions with live GUI and performance tracking**

---

## 📦 WHAT WAS CREATED

### Backend Files:
1. **`play_by_play_schema.sql`** - Database schema
   - `play_by_play` table (stores minute-by-minute scoring)
   - `mamba_game_cache` table (stores predictions + results)
   - Includes 2H tracking columns

2. **`cron_mamba_autonomous.py`** - Autonomous prediction system
   - Runs every 30 seconds on Railway
   - Processes ALL live games simultaneously
   - Triggers Mamba at Q2 6:00 for each game
   - Tracks final results + 2H results

3. **`mamba_live_websocket.py`** - Real-time WebSocket streaming
   - Streams play-by-play to frontend
   - Shows countdown to Q2 6:00
   - Broadcasts Mamba prediction

### Frontend Files:
4. **`frontend_mamba_live_component.tsx`** - Live GUI component
   - Real-time scoring pattern visualization
   - Countdown timer to Mamba trigger
   - Prediction display
   - Recent events feed

---

## 🎯 HOW IT WORKS

### For EVERY Live Game:

```
Q1 0:00  → Cron fetches play-by-play every 30 seconds
         → Stores: home_score, away_score, score_margin, time
         → ALL games processed simultaneously

Q1 6:00  → 6 minutes of data collected
         → Frontend shows live pattern building

Q2 0:00  → 12 minutes of data (half game)
         → Countdown timer shows "Triggers in 6:00"

Q2 6:00  → ⚡ AUTOMATIC MAMBA TRIGGER!
         → Extracts 33 features from last 18 minutes
         → Runs ML prediction
         → Stores in mamba_game_cache
         → WebSocket broadcasts to all connected clients

Q4 END   → Game finishes
         → Cron updates final scores
         → Calculates 2H scores
         → Tracks Mamba accuracy
         → Stores performance metrics
```

---

## 🚀 DEPLOYMENT STEPS

### Step 1: Deploy Database Schema

```bash
cd "/Users/test/Desktop/Tuscan Money/Ontologic XYZ/ML Research/live-system"

export DATABASE_URL="postgresql://postgres:lubdUxBtqzIWOynpPZiEXqFWPJvBlkVb@yamabiko.proxy.rlwy.net:37192/railway"

# Deploy schema
psql "$DATABASE_URL" -f play_by_play_schema.sql
```

**Verify:**
```sql
SELECT table_name FROM information_schema.tables 
WHERE table_name IN ('play_by_play', 'mamba_game_cache');
```

Should return:
- ✅ play_by_play
- ✅ mamba_game_cache

---

### Step 2: Add WebSocket to Backend

Add to `trading_dashboard_api.py`:

```python
from mamba_live_websocket import mamba_websocket_handler

@app.websocket("/ws/mamba/{game_id}")
async def websocket_mamba_endpoint(websocket: WebSocket, game_id: str):
    """Real-time Mamba updates for a specific game"""
    await mamba_websocket_handler(websocket, game_id)


@app.get("/api/mamba/performance")
async def get_mamba_performance():
    """Get Mamba model performance stats"""
    conn = get_db_connection()
    if not conn:
        return {"error": "Database not configured"}
    
    try:
        cursor = conn.cursor()
        
        # Get overall stats
        cursor.execute("""
            SELECT 
                COUNT(*) as total_predictions,
                SUM(CASE WHEN mamba_correct THEN 1 ELSE 0 END) as correct,
                AVG(mamba_error) as avg_error,
                AVG(confidence) as avg_confidence
            FROM mamba_game_cache
            WHERE final_home_score IS NOT NULL
        """)
        
        stats = cursor.fetchone()
        
        # Get recent predictions
        cursor.execute("""
            SELECT 
                game_id,
                prediction,
                actual_margin,
                mamba_error,
                mamba_correct,
                h2_margin,
                triggered_at
            FROM mamba_game_cache
            WHERE final_home_score IS NOT NULL
            ORDER BY triggered_at DESC
            LIMIT 10
        """)
        
        recent = []
        for row in cursor.fetchall():
            recent.append({
                'game_id': row[0],
                'prediction': float(row[1]),
                'actual': row[2],
                'error': float(row[3]),
                'correct': row[4],
                'h2_margin': row[5],
                'date': row[6].isoformat()
            })
        
        conn.close()
        
        accuracy = (stats[1] / stats[0] * 100) if stats[0] > 0 else 0
        
        return {
            'total_predictions': stats[0],
            'correct': stats[1],
            'accuracy': round(accuracy, 1),
            'avg_error': round(float(stats[2]), 2) if stats[2] else 0,
            'avg_confidence': round(float(stats[3]), 1) if stats[3] else 0,
            'recent_predictions': recent
        }
        
    except Exception as e:
        if conn:
            conn.close()
        return {"error": str(e)}
```

---

### Step 3: Add Frontend Component

In your game detail page (e.g., `/game/[id].jsx`):

```jsx
import { MambaLiveWidget } from './components/MambaLiveWidget';

export default function GamePage(props) {
  const game_id = props.params.id;
  
  return (
    <div>
      {/* Game header, scores, etc. */}
      
      {/* MAMBA LIVE WIDGET */}
      <MambaLiveWidget gameId={game_id} />
      
      {/* Rest of game details */}
    </div>
  );
}
```

---

### Step 4: Set Up Railway Cron

**Option A: Railway Dashboard**
1. Go to Railway dashboard → Your project
2. Settings → Cron
3. Add new cron:
   - **Schedule:** `*/30 * * * * *` (every 30 seconds)
   - **Command:** `python cron_mamba_autonomous.py`

**Option B: railway.json**

Create/update `railway.json`:

```json
{
  "build": {
    "builder": "NIXPACKS"
  },
  "deploy": {
    "startCommand": "uvicorn trading_dashboard_api:app --host 0.0.0.0 --port $PORT",
    "healthcheckPath": "/",
    "restartPolicyType": "ON_FAILURE"
  },
  "cron": [
    {
      "schedule": "*/30 * * * * *",
      "command": "python cron_mamba_autonomous.py",
      "description": "Mamba autonomous predictions for ALL live games"
    }
  ]
}
```

---

### Step 5: Deploy to Railway

```bash
cd live-system

# Add all files
git add play_by_play_schema.sql
git add cron_mamba_autonomous.py
git add mamba_live_websocket.py
git add trading_dashboard_api.py
git add railway.json

# Commit
git commit -m "Add Mamba autonomous system with live GUI and performance tracking"

# Push
git push origin main
```

Wait 1-2 minutes for Railway to deploy.

---

## ✅ VERIFICATION

### 1. Check Database

```sql
-- Check tables exist
SELECT table_name FROM information_schema.tables 
WHERE table_name IN ('play_by_play', 'mamba_game_cache');

-- Check during live game
SELECT COUNT(*) FROM play_by_play WHERE game_id = '0022500079';

-- Check Mamba predictions
SELECT * FROM mamba_game_cache ORDER BY triggered_at DESC LIMIT 5;
```

### 2. Check WebSocket

```javascript
// In browser console:
const ws = new WebSocket('wss://ol24-production.up.railway.app/ws/mamba/0022500079');
ws.onmessage = (e) => console.log(JSON.parse(e.data));
```

Should see:
- `type: 'initial_state'` on connect
- `type: 'game_update'` every 5 seconds
- `type: 'mamba_trigger'` at Q2 6:00
- `type: 'mamba_prediction'` after trigger

### 3. Check API Endpoints

```bash
# Get Mamba performance stats
curl "https://ol24-production.up.railway.app/api/mamba/performance"

# Should return:
# {
#   "total_predictions": 15,
#   "correct": 12,
#   "accuracy": 80.0,
#   "avg_error": 3.2,
#   "recent_predictions": [...]
# }
```

### 4. Check Cron Logs

```bash
railway logs --service=ol24 | grep "MAMBA AUTONOMOUS"
```

Should see entries every 30 seconds during game time.

---

## 🎨 FRONTEND FEATURES

### What Users See:

1. **Real-Time Scoring Pattern Chart**
   - Line graph showing score differential
   - Updates every 30 seconds
   - Shows last 18 minutes (Mamba window)

2. **Countdown Timer**
   - Shows time until Q2 6:00
   - Changes color as it approaches
   - Big "⚡ TRIGGERING NOW!" at 6:00

3. **Mamba Prediction Display**
   - Shows prediction value (+/- points)
   - Confidence percentage
   - Time triggered

4. **Recent Events Feed**
   - Last 5 scoring events
   - Shows time, score, margin
   - Color-coded (green/red for lead changes)

5. **Connection Status**
   - 🟢 Live (connected)
   - 🔴 Connecting... (reconnecting)

---

## 📊 PERFORMANCE TRACKING

### Automatic Tracking:

For EVERY game with a Mamba prediction, the system stores:

1. **Prediction:**
   - Value (spread prediction)
   - Confidence
   - 33 features used

2. **Game State at Q2 6:00:**
   - Home score
   - Away score
   - Current margin

3. **Final Result:**
   - Final home score
   - Final away score
   - Actual margin
   - Was Mamba correct? (within 5 points)
   - Prediction error

4. **2H Result:**
   - 2H home score
   - 2H away score
   - 2H margin
   - 2H prediction error

### Access Performance Stats:

```bash
GET /api/mamba/performance
```

Returns:
```json
{
  "total_predictions": 50,
  "correct": 42,
  "accuracy": 84.0,
  "avg_error": 3.8,
  "avg_confidence": 78.5,
  "recent_predictions": [
    {
      "game_id": "0022500123",
      "prediction": +5.2,
      "actual": +7,
      "error": 1.8,
      "correct": true,
      "h2_margin": +3,
      "date": "2025-10-29T19:30:00"
    },
    ...
  ]
}
```

---

## 🎯 USER EXPERIENCE

### During Live Game:

1. **Q1 Start:**
   - Widget appears under live game
   - Shows "Building pattern data..."
   - Chart starts populating

2. **Q1 6:00:**
   - 6 minutes of data visible
   - Countdown shows "Triggers in 18:00"

3. **Q2 3:00:**
   - Chart shows full 15-minute pattern
   - Countdown: "Triggers in 3:00"
   - Timer turns yellow/orange

4. **Q2 6:00:**
   - ⚡ BIG FLASH ANIMATION
   - "MAMBA TRIGGERING NOW!"
   - Progress indicator shows feature extraction

5. **Q2 5:30:**
   - Prediction appears with slide-in animation
   - Shows: "+5.2 points (78% confidence)"
   - Stays visible rest of game

6. **Game Ends:**
   - Shows actual result vs prediction
   - Displays accuracy: "Mamba was 1.8 points off!"
   - Updates performance stats

---

## 🔧 TROUBLESHOOTING

### No play-by-play data:
```sql
-- Check if cron is running
SELECT MAX(created_at) FROM play_by_play;

-- Should be < 1 minute ago during live games
```

### WebSocket not connecting:
- Check Railway logs: `railway logs`
- Verify endpoint: `/ws/mamba/{game_id}`
- Check firewall/SSL

### Mamba not triggering:
```sql
-- Check if prediction exists
SELECT * FROM mamba_game_cache WHERE game_id = '0022500079';

-- Check play-by-play count
SELECT COUNT(*) FROM play_by_play 
WHERE game_id = '0022500079' 
AND time_elapsed_seconds <= 1080;

-- Need at least 10 events
```

---

## ✅ SUMMARY

**What You Get:**

✅ Autonomous Mamba predictions for EVERY live game  
✅ Real-time GUI showing pattern development  
✅ Countdown timer to trigger (transparent to user)  
✅ Live prediction display  
✅ Automatic result tracking  
✅ 2H result tracking  
✅ Performance metrics dashboard  
✅ Complete historical record  

**No manual intervention needed - it all runs automatically!**

**Railway cron processes ALL games, stores ALL results, tracks ALL performance!** 🎯

