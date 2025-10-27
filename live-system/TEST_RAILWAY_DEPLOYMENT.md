# 🧪 TEST YOUR RAILWAY DEPLOYMENT

**Quick tests to verify your backend is live!**

---

## **🎯 GET YOUR RAILWAY URL:**

1. Railway dashboard → Your service
2. Go to **"Settings"** tab
3. Under **"Domains"**, click **"Generate Domain"**
4. Copy your URL (e.g., `https://live-system-production-abc123.up.railway.app`)

---

## **✅ TEST 1: SYSTEM STATUS**

```bash
# Replace with your Railway URL
RAILWAY_URL="https://your-app.up.railway.app"

# Test root endpoint
curl $RAILWAY_URL/
```

**Expected response:**
```json
{
  "status": "online",
  "system": "Ontologic XYZ Trading Dashboard",
  "version": "1.0.0",
  "ontorisk_enabled": true,
  "timestamp": "2025-10-27T..."
}
```

---

## **✅ TEST 2: LIVE NBA GAMES**

```bash
curl $RAILWAY_URL/api/live-games
```

**Expected response:**
```json
{
  "games": [
    {
      "game_id": "0022500045",
      "home_team": "LAL",
      "away_team": "GSW",
      "home_score": 105,
      "away_score": 98,
      ...
    }
  ],
  "count": 11,
  "timestamp": "..."
}
```

---

## **✅ TEST 3: BETONLINE ODDS**

```bash
curl $RAILWAY_URL/api/betonline-odds
```

**Expected response:**
```json
{
  "odds": [
    {
      "game_id": "0022500045",
      "spread": -6.0,
      "total": 215.5,
      "home_ml": -250,
      "away_ml": +210,
      ...
    }
  ]
}
```

---

## **⚠️ TEST 4: MAMBA PREDICTIONS (WILL FAIL - NO MODEL YET)**

```bash
curl $RAILWAY_URL/api/opportunities
```

**Expected response:**
```json
{
  "error": "Model not loaded",
  "opportunities": []
}
```

**This is OK!** We haven't uploaded the Mamba model yet.

---

## **🎯 VERIFY DEPLOYMENT SUCCESS:**

If Tests 1-3 pass, your backend is **LIVE AND WORKING!** ✅

The only thing missing is the Mamba model for predictions.

---

## **🔥 NEXT STEPS:**

### **Option A: Test Without Model (Good for now!)**

Your backend is live and serving:
- ✅ Live NBA games
- ✅ BetOnline odds
- ✅ System health

You can connect your Vercel frontend and see live games!

### **Option B: Upload Mamba Model (For predictions)**

To get predictions working, you need to upload the 322MB model:

**Quick Method (Dropbox):**
1. Upload `mambaofficial/models/MAMBA_MENTALITY_SYSTEM.pkl` to Dropbox
2. Get share link (change `www.dropbox.com` to `dl.dropboxusercontent.com` and remove `?dl=0`)
3. Add to Railway env vars:
   ```
   MODEL_URL=https://dl.dropboxusercontent.com/s/abc123/MAMBA_MENTALITY_SYSTEM.pkl
   ```
4. Redeploy

**Better Method (Railway Volume):**
1. Install Railway CLI: `npm install -g @railway/cli`
2. Login: `railway login`
3. Link project: `railway link`
4. Upload model:
   ```bash
   railway run cp mambaofficial/models/MAMBA_MENTALITY_SYSTEM.pkl /app/models/
   ```

---

## **🚀 CONNECT VERCEL FRONTEND:**

Once your Railway backend is live, connect your Vercel frontend:

```bash
cd "/Users/test/Desktop/Tuscan Money/Ontologic XYZ/ML Research/live-system/dashboard_pro"

# Update Vercel env var
vercel env rm VITE_API_BASE_URL production
vercel env add VITE_API_BASE_URL production
# Enter: https://your-app.up.railway.app

# Redeploy
vercel --prod
```

Now your friends can access the full dashboard at your Vercel URL!

---

## **📊 MONITORING:**

### **View Logs:**
Railway dashboard → Deployments → View Logs

### **Check Metrics:**
Railway dashboard → Metrics
- CPU usage
- Memory usage
- Request count

### **Restart:**
Railway dashboard → Settings → Restart

---

## **🎉 YOU'RE LIVE!**

Your backend is now deployed on Railway and accessible 24/7!

**Next:** Test the endpoints above and connect your Vercel frontend! 🚀

