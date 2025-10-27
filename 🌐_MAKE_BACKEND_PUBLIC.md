# 🌐 MAKE BACKEND PUBLIC FOR VERCEL

**Your friends need to access your backend from the internet!**

Right now: Backend runs on `localhost:8001` (only you can access)  
Need: Public URL so Vercel dashboard can connect

---

## 🚀 OPTION 1: NGROK (5 MINUTES - INSTANT!)

**Fastest way to make your local backend public**

### Step 1: Install ngrok
```bash
brew install ngrok
```

### Step 2: Run ngrok
```bash
ngrok http 8001
```

### Step 3: You'll get a public URL
```
Forwarding: https://abc123.ngrok.io -> http://localhost:8001
```

### Step 4: Update Vercel Environment Variable
1. Go to your Vercel project settings
2. Environment Variables
3. Add:
   - Name: `VITE_API_URL`
   - Value: `https://abc123.ngrok.io` (your ngrok URL)
4. Redeploy

### ✅ DONE!
Your friends can now access live data on Vercel!

**Pros:**
- ✅ 5 minutes setup
- ✅ Free tier available
- ✅ Works immediately

**Cons:**
- ❌ URL changes every time you restart ngrok (free tier)
- ❌ Need to keep your computer running
- ❌ Need to update Vercel env var each restart

---

## 🏗️ OPTION 2: RAILWAY (15 MINUTES - PERMANENT!)

**Deploy backend to cloud - permanent solution**

### Step 1: Create railway.json
Already done! File exists in your repo.

### Step 2: Push backend to GitHub
```bash
# Create new repo for backend
cd "5. Live System"
git init
git remote add origin https://github.com/Louie4TuscanMoney/OL24-Backend.git
git add trading_dashboard_api.py autonomous_trading_daemon.py nba_live_scores.py betonline_live_lines.py
git commit -m "Backend deployment"
git push origin main
```

### Step 3: Deploy to Railway
1. Go to: https://railway.app
2. "New Project" → "Deploy from GitHub repo"
3. Select: OL24-Backend
4. Railway auto-detects Python
5. Add environment variables (if needed)
6. Deploy!

### Step 4: Get Public URL
Railway gives you: `https://ol24-backend.up.railway.app`

### Step 5: Update Vercel
Add `VITE_API_URL` = Railway URL

### ✅ DONE!
Permanent, scalable, always-on backend!

**Pros:**
- ✅ Permanent URL
- ✅ Always running
- ✅ No need to keep computer on
- ✅ Auto-scales

**Cons:**
- ❌ Takes 15 minutes
- ❌ May need to pay ($5/mo after free tier)

---

## 🎯 RECOMMENDATION FOR RIGHT NOW (GAME 2 IN 45 MIN)

**Use ngrok for TONIGHT:**

```bash
# Install ngrok
brew install ngrok

# Run ngrok (keep this terminal open!)
ngrok http 8001

# Copy the https URL
# Update Vercel env var: VITE_API_URL = your ngrok URL
# Redeploy Vercel

# Share Vercel URL with friends!
```

**Then after tonight, deploy to Railway for permanent solution.**

---

## 📋 WHAT NEEDS TO BE PUBLIC

Your backend provides these endpoints:
- `/api/live-games` - Live NBA scores
- `/api/betonline/live/{game_id}` - BetOnline odds
- `/api/opportunities` - ML predictions
- `/api/auth/*` - User authentication
- `/api/bets/*` - Bet tracking

All of these need to be accessible from the internet for Vercel to work.

---

## 🔥 QUICK START (5 MINUTES!)

```bash
# Terminal 1: Keep backend running
# (already running)

# Terminal 2: Run ngrok
brew install ngrok
ngrok http 8001

# Terminal 3: Update and redeploy Vercel
# Go to vercel.com → your project → Settings → Environment Variables
# Add: VITE_API_URL = https://YOUR_NGROK_URL.ngrok.io
# Redeploy

# Share with friends!
```

---

## ⚠️ IMPORTANT: CORS IS ALREADY CONFIGURED!

Your backend already has:
```python
allow_origins=["*"]  # Accepts requests from anywhere
```

So no backend changes needed! Just need to make it publicly accessible.

---

## 💡 ALTERNATIVE: LOCAL ONLY (NO SETUP)

If you don't want to deal with this now:

**Option A: Share your screen during games**
- Friends watch on Zoom/Discord
- You run dashboard locally
- No deployment needed

**Option B: Friends run locally**
- Share GitHub repo
- They install and run on their machines
- Everyone has their own copy

**Option C: Deploy after Game 2**
- Watch Game 2 yourself tonight
- Deploy to Railway tomorrow
- Share with friends for future games

---

**🎯 YOUR CALL! WHAT DO YOU WANT TO DO?**

1. **Ngrok now** (5 min, works for tonight)
2. **Railway now** (15 min, permanent)
3. **Skip for now** (deploy after Game 2)

