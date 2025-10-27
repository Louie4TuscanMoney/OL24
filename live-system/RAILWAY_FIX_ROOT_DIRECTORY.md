# 🔧 RAILWAY FIX: SET ROOT DIRECTORY

**Your Railway deployment failed because it's looking at the entire OL24 repo instead of just `5. Live System/`**

---

## **🚨 THE PROBLEM:**

Railway tried to deploy:
```
OL24/
├── 5. Live System/          ← Backend is HERE
├── dashboard_pro/           ← Frontend confusing Railway
├── mambaofficial/
├── ontoriskofficial/
└── ... (everything else)
```

Railway doesn't know what to build! It's seeing a Python backend, a Node frontend, and a bunch of other stuff.

---

## **✅ THE FIX: TELL RAILWAY TO ONLY LOOK AT `5. Live System/`**

### **STEP 1: Go to Railway Dashboard**

Go to: https://railway.app/dashboard

Find your project (the failed deployment)

---

### **STEP 2: Click on Your Service**

You should see:
- Your Python service (probably failed)
- Postgres database
- Redis cache

Click on the **Python service**.

---

### **STEP 3: Go to Settings**

Click the **"Settings"** tab at the top.

---

### **STEP 4: Find "Root Directory"**

Scroll down to **"Service Settings"** section.

You'll see:
- **Build Command** (optional)
- **Start Command** (should be set from Procfile)
- **Root Directory** ← THIS IS WHAT WE NEED!

---

### **STEP 5: Set Root Directory**

In the **"Root Directory"** field, enter:
```
5. Live System
```

**Note:** Railway uses the path relative to repo root, so it's `5. Live System`, NOT `/5. Live System/`

Click **"Update"** or **"Save"**.

---

### **STEP 6: Redeploy**

After updating the root directory:

1. Go to the **"Deployments"** tab
2. Click **"Redeploy"** on the failed deployment
   
   OR
   
   Just push a new commit to GitHub:
   ```bash
   cd "/Users/test/Desktop/Tuscan Money/Ontologic XYZ/ML Research/5. Live System"
   git commit --allow-empty -m "🚂 Trigger Railway redeploy"
   git push origin main
   ```

---

### **STEP 7: Watch the Build Logs**

Go to **"Deployments"** → Click the new deployment → **"View Logs"**

You should see:
```
✅ Detected Python app
✅ Installing dependencies from requirements.txt
✅ Starting with: uvicorn trading_dashboard_api:app --host 0.0.0.0 --port $PORT
✅ Health check passed
✅ Deployment successful!
```

---

## **🎯 EXPECTED BUILD OUTPUT:**

```bash
# Railway will now ONLY see:
5. Live System/
├── trading_dashboard_api.py     ← Main app
├── nba_live_scores.py
├── live_trading_engine.py
├── betonline_live_lines.py
├── requirements.txt              ← Dependencies
├── Procfile                      ← Start command
├── railway.json                  ← Railway config
└── ...

# It will:
1. Detect Python app ✅
2. Run: pip install -r requirements.txt ✅
3. Start: uvicorn trading_dashboard_api:app ✅
4. Health check: GET / ✅
5. Deploy! ✅
```

---

## **🚨 TROUBLESHOOTING:**

### **Problem: "Root Directory not found"**

**Fix:** Make sure you typed `5. Live System` EXACTLY (with the space and period!)

### **Problem: "requirements.txt not found"**

**Fix:** 
1. Check that `5. Live System/requirements.txt` exists in your GitHub repo
2. Go to: https://github.com/Louie4TuscanMoney/OL24/blob/main/5.%20Live%20System/requirements.txt
3. If missing, push it:
   ```bash
   cd "/Users/test/Desktop/Tuscan Money/Ontologic XYZ/ML Research/5. Live System"
   git add requirements.txt Procfile railway.json
   git commit -m "Add Railway files"
   git push origin main
   ```

### **Problem: "Port not listening"**

**Fix:** Make sure your `trading_dashboard_api.py` uses Railway's `$PORT` env var:
```python
import os
port = int(os.getenv("PORT", 8000))
uvicorn.run(app, host="0.0.0.0", port=port)
```

### **Problem: Build succeeds but health check fails**

**Fix:** Make sure your root endpoint (`/`) returns a response:
```python
@app.get("/")
async def root():
    return {"status": "online", "system": "Ontologic XYZ"}
```

---

## **🎯 VERIFICATION:**

Once deployed successfully, test your Railway URL:

```bash
# Get your Railway URL from dashboard (e.g., https://your-app.up.railway.app)
RAILWAY_URL="https://your-app.up.railway.app"

# Test root endpoint
curl $RAILWAY_URL/

# Test live games
curl $RAILWAY_URL/api/live-games

# Test opportunities
curl $RAILWAY_URL/api/opportunities
```

You should get real data back! ✅

---

## **📸 SCREENSHOT GUIDE:**

If you're still stuck, here's what to look for:

### **Settings Tab:**
```
┌─────────────────────────────────────┐
│  Settings                           │
├─────────────────────────────────────┤
│                                     │
│  Service Settings                   │
│  ├─ Build Command    [optional]    │
│  ├─ Start Command    [from Procfile]│
│  └─ Root Directory   [5. Live System]│ ← SET THIS!
│                                     │
│  Environment                        │
│  ├─ PYTHONUNBUFFERED  1            │
│  ├─ DATABASE_URL      ${{...}}     │
│  └─ ...                            │
│                                     │
└─────────────────────────────────────┘
```

---

## **🔥 ALTERNATIVE: SEPARATE BACKEND REPO**

If you want a cleaner setup, create a dedicated backend repo:

```bash
# Option: Create new repo for just the backend
cd "/Users/test/Desktop/Tuscan Money/Ontologic XYZ/ML Research"
mkdir Mamba-Backend
cp -r "5. Live System/"* Mamba-Backend/
cd Mamba-Backend
git init
git add .
git commit -m "🚂 Initial backend for Railway"
# Create new repo on GitHub: Louie4TuscanMoney/Mamba-Backend
git remote add origin https://github.com/Louie4TuscanMoney/Mamba-Backend.git
git push -u origin main
```

Then deploy `Mamba-Backend` on Railway (no root directory needed!)

**But the root directory fix is easier!** ✅

---

## **🎯 RECAP:**

1. ✅ Go to Railway dashboard
2. ✅ Click your service → Settings
3. ✅ Set Root Directory: `5. Live System`
4. ✅ Redeploy
5. ✅ Test endpoints
6. ✅ Connect Vercel

**That's it! Railway will now only build your backend!** 🚂🚀

