# 🚂 RAILWAY DEPLOYMENT - QUICK START

**Your backend is ready to deploy! Let's get it live in 15 minutes.**

---

## **✅ PREPARATION COMPLETE:**

- ✅ `requirements.txt` updated with all dependencies
- ✅ `Procfile` configured for Railway
- ✅ `railway.json` created with deploy settings
- ✅ `.gitignore` set up
- ✅ **Pushed to GitHub:** https://github.com/Louie4TuscanMoney/OL24

---

## **🚀 STEP-BY-STEP DEPLOYMENT:**

### **STEP 1: Sign Up for Railway** (2 minutes)

1. Go to https://railway.app
2. Click **"Start a New Project"**
3. Click **"Login with GitHub"**
4. Authorize Railway to access your GitHub

---

### **STEP 2: Create Your Project** (3 minutes)

1. Click **"New Project"**
2. Select **"Deploy from GitHub repo"**
3. Search for and select: **`Louie4TuscanMoney/OL24`**
4. Railway will detect Python and start deploying!

**Important:** Railway will deploy the root directory by default, but your backend is in `5. Live System/`. We'll fix this in Step 5.

---

### **STEP 3: Add PostgreSQL Database** (1 minute)

1. In your Railway project, click **"+ New"**
2. Select **"Database"** → **"Add PostgreSQL"**
3. Railway automatically provisions a Postgres instance
4. Connection string is auto-added to env vars as `DATABASE_URL`

---

### **STEP 4: Add Redis Cache** (1 minute)

1. Click **"+ New"** again
2. Select **"Database"** → **"Add Redis"**
3. Railway automatically provisions a Redis instance
4. Connection string is auto-added to env vars as `REDIS_URL`

---

### **STEP 5: Configure Your Service** (5 minutes)

#### **5.1: Set Root Directory**

Your backend code is in `5. Live System/`, not root:

1. Click on your **service** (Python app)
2. Go to **Settings** tab
3. Find **"Root Directory"**
4. Set to: `5. Live System`
5. Click **"Save"**

#### **5.2: Add Environment Variables**

Go to **Variables** tab and add:

```bash
# Python
PYTHONUNBUFFERED=1

# Database (auto-populated by Railway)
DATABASE_URL=${{Postgres.DATABASE_URL}}
REDIS_URL=${{Redis.REDIS_URL}}

# Model Path (we'll upload model separately)
MODEL_PATH=/app/models/MAMBA_MENTALITY_SYSTEM.pkl

# CORS (allow your domains)
ALLOWED_ORIGINS=https://ontologicxyz.com,https://*.vercel.app,https://*.railway.app

# Port (Railway auto-sets this)
PORT=${{PORT}}

# Risk Management
STARTING_BANKROLL=1000
MAX_POSITION_SIZE=0.05
```

Click **"Add Variable"** for each one.

#### **5.3: Redeploy**

After setting root directory and env vars:
1. Go to **Deployments** tab
2. Click **"Deploy"** or push a new commit to GitHub

---

### **STEP 6: Upload Mamba Model** (CRITICAL!)

Your Mamba model (322MB) is too big for GitHub. You have **3 options:**

#### **Option A: Railway Volume (Recommended)**

1. In Railway dashboard → Your Service → **Settings** → **Volumes**
2. Click **"New Volume"**
3. Mount path: `/app/models`
4. Click **"Add Volume"**

Then upload via Railway CLI:
```bash
# Install Railway CLI
npm install -g @railway/cli

# Login
railway login

# Link to your project
cd "/Users/test/Desktop/Tuscan Money/Ontologic XYZ/ML Research/5. Live System"
railway link

# Upload model
railway run cp ../mambaofficial/models/MAMBA_MENTALITY_SYSTEM.pkl /app/models/
```

#### **Option B: Cloud Storage (Alternative)**

Upload to Dropbox/Google Drive and download on startup:

1. Upload `MAMBA_MENTALITY_SYSTEM.pkl` to Dropbox
2. Get shareable link (e.g., `https://www.dropbox.com/s/abc123/model.pkl?dl=1`)
3. Add env var in Railway:
   ```
   MODEL_URL=https://www.dropbox.com/s/abc123/model.pkl?dl=1
   ```
4. Backend will download on startup (see Step 7)

#### **Option C: Smaller Model (Quick Test)**

For testing, use a smaller dummy model:
```bash
# Create a tiny test model
cd "/Users/test/Desktop/Tuscan Money/Ontologic XYZ/ML Research/5. Live System"
python3 -c "import pickle; pickle.dump({'model': 'test'}, open('test_model.pkl', 'wb'))"
git add test_model.pkl
git commit -m "Add test model"
git push
```

Then update `MODEL_PATH` to `test_model.pkl`.

---

### **STEP 7: Update Backend for Model Download (If using Option B)**

If you chose Option B (cloud storage), update `trading_dashboard_api.py`:

```python
import os
import requests
from pathlib import Path

# Add this at startup
@app.on_event("startup")
async def download_model():
    MODEL_URL = os.getenv("MODEL_URL")
    MODEL_PATH = os.getenv("MODEL_PATH", "/tmp/MAMBA_MENTALITY_SYSTEM.pkl")
    
    if MODEL_URL and not os.path.exists(MODEL_PATH):
        print(f"📥 Downloading Mamba model from {MODEL_URL}...")
        
        # Create directory
        Path(MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
        
        # Download
        response = requests.get(MODEL_URL, stream=True)
        with open(MODEL_PATH, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"✅ Model downloaded to {MODEL_PATH}")
```

---

### **STEP 8: Verify Deployment** (2 minutes)

#### **8.1: Check Deployment Status**

1. Go to **Deployments** tab
2. Wait for status to show **"SUCCESS" ✅**
3. If it fails, check **Logs** for errors

#### **8.2: Get Your Railway URL**

1. Go to **Settings** tab
2. Under **Domains**, click **"Generate Domain"**
3. You'll get something like: `https://your-app.up.railway.app`

#### **8.3: Test Your API**

```bash
# Test health endpoint
curl https://your-app.up.railway.app/

# Test live games
curl https://your-app.up.railway.app/api/live-games

# Test opportunities
curl https://your-app.up.railway.app/api/opportunities
```

---

### **STEP 9: Connect Vercel Frontend** (2 minutes)

Update your Vercel environment variable:

```bash
cd "/Users/test/Desktop/Tuscan Money/Ontologic XYZ/ML Research/5. Live System/dashboard_pro"

# Remove old ngrok URL
vercel env rm VITE_API_BASE_URL production

# Add Railway URL
vercel env add VITE_API_BASE_URL production
# Enter: https://your-app.up.railway.app

# Redeploy
vercel --prod
```

---

### **STEP 10: Add Custom Domain** (OPTIONAL)

#### **For Backend API:**

1. Railway dashboard → Your Service → **Settings** → **Domains**
2. Click **"Custom Domain"**
3. Enter: `api.ontologicxyz.com`
4. Add DNS record (Railway will show you):
   - Type: `CNAME`
   - Name: `api`
   - Value: `your-app.up.railway.app`

#### **For Frontend:**

Already configured on Vercel for `ontologicxyz.com`!

---

## **🔥 DEPLOYMENT CHECKLIST:**

```
☐ Step 1: Sign up for Railway with GitHub
☐ Step 2: Create project from Louie4TuscanMoney/OL24
☐ Step 3: Add PostgreSQL database
☐ Step 4: Add Redis cache
☐ Step 5: Configure service (root directory, env vars)
☐ Step 6: Upload Mamba model (volume/cloud/test)
☐ Step 7: Update backend for model download (if needed)
☐ Step 8: Verify deployment (test endpoints)
☐ Step 9: Connect Vercel frontend
☐ Step 10: Add custom domain (optional)
```

---

## **💰 COST BREAKDOWN:**

### **Railway Free Tier:**
- **$5/month FREE credit** (no credit card required!)
- Python backend: ~$0.50/month
- Postgres: ~$1/month
- Redis: ~$0.50/month
- **Total: ~$2/month** (covered by free tier!)

### **After Free Tier:**
- $5/month gets you ~500 hours of runtime
- Perfect for hobby projects and testing
- Upgrade to Pro ($20/month) when profitable!

---

## **📊 MONITORING & LOGS:**

### **View Logs:**
Railway dashboard → Your Service → **Deployments** → Click deployment → **View Logs**

### **Restart Service:**
Railway dashboard → Your Service → **Settings** → **Restart**

### **Scale Resources:**
Railway dashboard → Your Service → **Settings** → **Resources**
- Default: 512MB RAM, 1 vCPU
- Can increase as needed

---

## **🎯 TROUBLESHOOTING:**

### **Problem: Deployment fails with "requirements.txt not found"**

**Solution:** Make sure **Root Directory** is set to `5. Live System`

### **Problem: "Module not found" error**

**Solution:** Check `requirements.txt` includes all dependencies

### **Problem: Model file not found**

**Solution:** Make sure you uploaded the model (Step 6) and `MODEL_PATH` env var is correct

### **Problem: Database connection error**

**Solution:** Make sure Postgres is added and `DATABASE_URL` env var is set

---

## **🚀 AUTO-DEPLOY (BONUS!):**

Railway automatically deploys when you push to GitHub!

```bash
cd "/Users/test/Desktop/Tuscan Money/Ontologic XYZ/ML Research/5. Live System"

# Make changes
git add .
git commit -m "Update backend"
git push origin main

# Railway auto-detects and deploys! 🎉
```

---

## **🎉 YOU'RE READY!**

Follow these steps and you'll have a production-ready backend in 15 minutes!

**Questions? Issues? Check the logs first, then troubleshoot!**

---

## **🔗 USEFUL LINKS:**

- **Railway Dashboard:** https://railway.app/dashboard
- **Railway Docs:** https://docs.railway.app
- **Your GitHub Repo:** https://github.com/Louie4TuscanMoney/OL24
- **Railway CLI Docs:** https://docs.railway.app/develop/cli

---

**READY TO DEPLOY? GO TO https://railway.app AND START!** 🚂🚀

