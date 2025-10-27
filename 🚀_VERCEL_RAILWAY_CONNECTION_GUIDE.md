# 🚀 VERCEL + RAILWAY CONNECTION GUIDE

**Status:** ✅ Railway backend is LIVE and working!  
**Backend URL:** `https://ol24-production.up.railway.app`  
**Frontend:** Needs to be connected to Railway

---

## **✅ WHAT'S WORKING RIGHT NOW:**

### Railway Backend (https://ol24-production.up.railway.app):
- ✅ **System Online** - No more crashes!
- ✅ **NBA API Connected** - Fetching live games
- ✅ **Mamba Model Loaded** - Ready for predictions
- ✅ **WebSocket Ready** - `/ws` endpoint available
- ✅ **All Endpoints Working:**
  - `GET /` - Health check
  - `GET /api/live-games` - Live NBA games
  - `GET /api/opportunities` - Betting opportunities (Mamba + OntoRisk)
  - `WS /ws` - Real-time WebSocket updates

---

## **🔧 CONNECT VERCEL FRONTEND TO RAILWAY:**

### **Option 1: Via Vercel Dashboard (EASIEST)**

1. **Go to Vercel Dashboard:**
   - Visit: https://vercel.com/dashboard
   - Select your project (the one at `ontologicxyz.com`)

2. **Add Environment Variable:**
   - Click **"Settings"** → **"Environment Variables"**
   - Add new variable:
     - **Name:** `VITE_API_URL`
     - **Value:** `https://ol24-production.up.railway.app`
     - **Environments:** ✅ Production, ✅ Preview, ✅ Development

3. **Redeploy:**
   - Go to **"Deployments"** tab
   - Click **"Redeploy"** on the latest deployment
   - OR: Push a new commit to trigger auto-deploy

---

### **Option 2: Via Environment File (For Local Testing)**

Create `.env` file in `live-system/dashboard_pro/`:

```bash
VITE_API_URL=https://ol24-production.up.railway.app
```

Then test locally:
```bash
cd "live-system/dashboard_pro"
npm install
npm run dev
```

Open http://localhost:5173 and verify:
- ✅ Live games appear
- ✅ Console shows: "🔌 Connecting to WebSocket: wss://ol24-production.up.railway.app/ws"
- ✅ Console shows: "✅ WebSocket connected!"

---

## **🧪 VERIFY THE CONNECTION:**

### **1. Test Railway Backend Directly:**

```bash
# Health check
curl "https://ol24-production.up.railway.app/"

# Live games
curl "https://ol24-production.up.railway.app/api/live-games"

# Opportunities
curl "https://ol24-production.up.railway.app/api/opportunities"
```

### **2. Test WebSocket (Browser Console):**

Once Vercel is deployed, open https://ontologicxyz.com and check browser console:

**Expected logs:**
```
🔌 Connecting to WebSocket: wss://ol24-production.up.railway.app/ws
✅ WebSocket connected!
📦 Received WebSocket message: update
```

### **3. Test Frontend Display:**

After logging in with password `Rwwc2018!!`:
- ✅ **Live Games Section:** Shows scheduled/live NBA games
- ✅ **Opportunities Section:** Shows Mamba predictions when Q2 6:00 hits
- ✅ **Risk Status:** Shows bankroll, drawdown, alerts
- ✅ **Real-time Updates:** Dashboard updates every 10 seconds via WebSocket

---

## **🎮 WHEN GAMES START (Q2 6:00 MARK):**

### **Automatic Flow:**

1. **Railway Backend (Every 10 seconds):**
   - ✅ Fetches live NBA games via ESPN API
   - ✅ Checks if Q2 has 6:00 remaining
   - ✅ Downloads 18-minute play-by-play data
   - ✅ Extracts 33 real features
   - ✅ Runs Mamba ML model
   - ✅ Calculates OntoRisk (Kelly sizing, probability calibration)
   - ✅ Pushes complete package to frontend via WebSocket

2. **Vercel Frontend (Real-time):**
   - ✅ Receives WebSocket update
   - ✅ Displays Mamba prediction
   - ✅ Shows optimal bet size
   - ✅ Shows BetOnline odds (when available)
   - ✅ Shows confidence intervals
   - ✅ Alerts user if good betting opportunity

---

## **📊 CURRENT SYSTEM ARCHITECTURE:**

```
┌─────────────────────────────────────────────┐
│  USER'S BROWSER (ontologicxyz.com)         │
│  ↓ Vercel Frontend (SolidJS)               │
└─────────────────────────────────────────────┘
                    ↕ WebSocket
┌─────────────────────────────────────────────┐
│  RAILWAY BACKEND (24/7 Autonomous)         │
│  ↓ FastAPI + Uvicorn                       │
│  ↓ Live Trading Engine                     │
│  ↓ Mamba ML Model (322MB, auto-downloaded) │
│  ↓ OntoRisk (Kelly + Calibration)          │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│  DATA SOURCES                               │
│  • ESPN API (live scores)                   │
│  • NBA API (play-by-play)                   │
│  • BetOnline (manual entry for now)         │
└─────────────────────────────────────────────┘
```

---

## **🚨 TROUBLESHOOTING:**

### **Frontend not showing data:**
1. Check Vercel environment variable: `VITE_API_URL=https://ol24-production.up.railway.app`
2. Check browser console for WebSocket errors
3. Check Railway logs for backend errors
4. Verify CORS is enabled (already configured in `trading_dashboard_api.py`)

### **WebSocket not connecting:**
1. Vercel may have WebSocket timeout limits (max 5 minutes)
2. Backend sends updates every 10 seconds to keep connection alive
3. Frontend auto-reconnects if disconnected

### **Mamba predictions not appearing:**
1. Wait for Q2 6:00 mark in a live game
2. Check Railway logs: Should see "🐍 Making Mamba prediction..."
3. Verify 18-minute play-by-play data was fetched
4. Check `/api/opportunities` endpoint directly

---

## **🎯 NEXT STEPS:**

### **NOW (Immediate):**
1. ✅ Set `VITE_API_URL` in Vercel dashboard
2. ✅ Redeploy Vercel frontend
3. ✅ Test at https://ontologicxyz.com
4. ✅ Verify WebSocket connection in browser console

### **WHEN GAMES START:**
1. ✅ Verify live games appear on dashboard
2. ✅ Wait for Q2 6:00 mark
3. ✅ Check for Mamba prediction to appear
4. ✅ Verify OntoRisk analysis displays

### **FUTURE IMPROVEMENTS:**
1. ⏳ Automate BetOnline odds scraping (currently manual entry)
2. ⏳ Add trade history persistence (Postgres)
3. ⏳ Add real-time notifications (Discord/SMS)
4. ⏳ Add performance tracking dashboard
5. ⏳ Add A/B testing for different MAE thresholds

---

## **🎉 YOU'RE READY TO GO LIVE!**

Once you:
1. Set the Vercel environment variable
2. Redeploy the frontend
3. Test the connection

**Your system will be FULLY AUTONOMOUS:**
- ✅ Railway backend runs 24/7 (your computer can be OFF)
- ✅ Fetches live NBA data every 10 seconds
- ✅ Makes Mamba predictions automatically at Q2 6:00
- ✅ Sends real-time updates to all connected users
- ✅ Your friends can access https://ontologicxyz.com anytime!

---

**Questions? Issues? Let me know!** 🚀

