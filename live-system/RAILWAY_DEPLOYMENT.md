# 🚂 RAILWAY DEPLOYMENT - PRODUCTION HOSTING

**Deploy your Mamba system to production-ready infrastructure**

---

## **🎯 WHAT RAILWAY GIVES YOU:**

- ✅ **Persistent hosting** (24/7 uptime)
- ✅ **Postgres database** (store predictions, users, bets)
- ✅ **Redis caching** (rate limiting, fast lookups)
- ✅ **Custom domain** (ontologicxyz.com)
- ✅ **Automatic deploys** (push to GitHub → auto-deploy)
- ✅ **Environment variables** (secrets, API keys)
- ✅ **$5/month FREE credit** (enough for hobby projects)

---

## **📦 ARCHITECTURE:**

```
┌─────────────────────────────────────────────────────────┐
│                    Railway Project                       │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────────────┐    ┌──────────────────┐          │
│  │   Python Backend │───▶│   Postgres DB    │          │
│  │   (FastAPI)      │    │   (predictions)  │          │
│  └──────────────────┘    └──────────────────┘          │
│           │                                              │
│           │               ┌──────────────────┐          │
│           └──────────────▶│   Redis Cache    │          │
│                           │   (rate limits)  │          │
│                           └──────────────────┘          │
│                                                          │
└─────────────────────────────────────────────────────────┘
                            │
                            │ HTTPS
                            ▼
                   https://your-app.up.railway.app
                            │
                            │
                            ▼
                   ┌──────────────────┐
                   │  Vercel Frontend │
                   │   (SolidJS)      │
                   └──────────────────┘
                            │
                            ▼
                   https://ontologicxyz.com
```

---

## **🚀 STEP 1: PREPARE YOUR BACKEND**

### **1.1: Create `requirements.txt`**

Already exists, but let's verify:
```bash
cd "/Users/test/Desktop/Tuscan Money/Ontologic XYZ/ML Research/5. Live System"
cat requirements.txt
```

Should include:
```txt
fastapi==0.104.1
uvicorn[standard]==0.24.0
nba_api==1.4.1
numpy==1.26.2
pandas==2.1.3
scipy==1.11.4
xgboost==2.0.3
lightgbm==4.1.0
scikit-learn==1.3.2
requests==2.31.0
aiohttp==3.9.1
psycopg2-binary==2.9.9
redis==5.0.1
python-dotenv==1.0.0
```

### **1.2: Create `Procfile`**

Already exists, but let's verify:
```bash
cat Procfile
```

Should contain:
```
web: uvicorn trading_dashboard_api:app --host 0.0.0.0 --port $PORT
```

### **1.3: Create `railway.json`**

```bash
cat > railway.json << 'EOF'
{
  "$schema": "https://railway.app/railway.schema.json",
  "build": {
    "builder": "NIXPACKS"
  },
  "deploy": {
    "startCommand": "uvicorn trading_dashboard_api:app --host 0.0.0.0 --port $PORT",
    "healthcheckPath": "/api/health",
    "healthcheckTimeout": 100,
    "restartPolicyType": "ON_FAILURE",
    "restartPolicyMaxRetries": 10
  }
}
EOF
```

---

## **🚀 STEP 2: SIGN UP FOR RAILWAY**

1. Go to https://railway.app
2. Click "Start a New Project"
3. Sign in with GitHub
4. Authorize Railway

---

## **🚀 STEP 3: CREATE A NEW PROJECT**

### **3.1: Create Project**
1. Click "New Project"
2. Select "Deploy from GitHub repo"
3. Select `Louie4TuscanMoney/OL24`
4. Railway will auto-detect Python and deploy!

### **3.2: Add Postgres**
1. Click "New" in your project
2. Select "Database" → "Add PostgreSQL"
3. Railway auto-provisions a Postgres instance
4. Connection string is auto-added to env vars

### **3.3: Add Redis**
1. Click "New" in your project
2. Select "Database" → "Add Redis"
3. Railway auto-provisions a Redis instance
4. Connection string is auto-added to env vars

---

## **🚀 STEP 4: CONFIGURE ENVIRONMENT VARIABLES**

In Railway dashboard → Your Service → Variables:

```bash
# Python
PYTHONUNBUFFERED=1

# Database (auto-populated by Railway)
DATABASE_URL=${{Postgres.DATABASE_URL}}
REDIS_URL=${{Redis.REDIS_URL}}

# NBA API
NBA_API_KEY=your_nba_api_key_if_needed

# Model Path (Railway absolute path)
MODEL_PATH=/app/mambaofficial/models/MAMBA_MENTALITY_SYSTEM.pkl

# CORS (allow your Vercel domain)
ALLOWED_ORIGINS=https://ontologicxyz.com,https://*.vercel.app

# BetOnline
BETONLINE_SCRAPER_ENABLED=false

# Risk Management
STARTING_BANKROLL=1000
MAX_POSITION_SIZE=0.05
```

---

## **🚀 STEP 5: UPLOAD MAMBA MODEL TO RAILWAY**

**Problem:** Your Mamba model is 322MB, too big for GitHub.

**Solution:** Use Railway volumes or environment secrets.

### **Option A: Railway Volume (Recommended)**

1. In Railway dashboard → Your Service → Settings → Volumes
2. Click "New Volume"
3. Mount path: `/app/models`
4. Upload `MAMBA_MENTALITY_SYSTEM.pkl` via Railway CLI

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login
railway login

# Link to your project
railway link

# Upload model
railway run scp mambaofficial/models/MAMBA_MENTALITY_SYSTEM.pkl /app/models/
```

5. Update `MODEL_PATH` env var:
```
MODEL_PATH=/app/models/MAMBA_MENTALITY_SYSTEM.pkl
```

### **Option B: Download on Startup (Alternative)**

Store model on Dropbox/Google Drive and download on startup:

```python
# In trading_dashboard_api.py startup
import requests
import os

MODEL_URL = os.getenv("MODEL_URL")  # Dropbox/Drive link
MODEL_PATH = "/tmp/MAMBA_MENTALITY_SYSTEM.pkl"

if not os.path.exists(MODEL_PATH):
    print("📥 Downloading Mamba model...")
    response = requests.get(MODEL_URL, stream=True)
    with open(MODEL_PATH, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print("✅ Model downloaded")
```

---

## **🚀 STEP 6: UPDATE BACKEND FOR POSTGRES**

Replace SQLite with Postgres:

```python
# In trading_dashboard_api.py
import os
import psycopg2
from urllib.parse import urlparse

DATABASE_URL = os.getenv("DATABASE_URL")

def get_db_connection():
    """Get Postgres connection"""
    if DATABASE_URL:
        # Railway Postgres
        return psycopg2.connect(DATABASE_URL)
    else:
        # Local SQLite fallback
        import sqlite3
        return sqlite3.connect("user_database.db")

# Initialize tables on startup
@app.on_event("startup")
async def startup():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Create tables
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_requests (
            id SERIAL PRIMARY KEY,
            email VARCHAR(255) UNIQUE NOT NULL,
            phone VARCHAR(50),
            full_name VARCHAR(255),
            status VARCHAR(50) DEFAULT 'pending',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS mamba_predictions (
            id SERIAL PRIMARY KEY,
            game_id VARCHAR(50) NOT NULL,
            prediction FLOAT NOT NULL,
            features JSONB,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    conn.commit()
    conn.close()
    
    print("✅ Database initialized")
```

---

## **🚀 STEP 7: DEPLOY!**

### **Option A: Auto-Deploy (Recommended)**

Railway auto-deploys when you push to GitHub:

```bash
cd "/Users/test/Desktop/Tuscan Money/Ontologic XYZ/ML Research/5. Live System"

git add .
git commit -m "🚂 Railway deployment ready"
git push origin main
```

Railway automatically:
1. Detects the push
2. Builds the Docker image
3. Deploys to production
4. Runs health checks
5. Goes live!

### **Option B: Manual Deploy**

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login
railway login

# Link to project
railway link

# Deploy
railway up
```

---

## **🚀 STEP 8: VERIFY DEPLOYMENT**

### **Check 1: Railway Dashboard**
- Go to your Railway project
- Check "Deployments" tab
- Status should be "Active" ✅

### **Check 2: Test API**
```bash
# Get your Railway URL (e.g., https://your-app.up.railway.app)
curl https://your-app.up.railway.app/api/health
```

Should return:
```json
{"status":"healthy","timestamp":"2025-10-27T..."}
```

### **Check 3: Test Live Games**
```bash
curl https://your-app.up.railway.app/api/live-games
```

Should return live NBA games!

### **Check 4: Test Mamba Prediction**
```bash
curl https://your-app.up.railway.app/api/opportunities
```

Should return Mamba predictions!

---

## **🚀 STEP 9: CONNECT VERCEL TO RAILWAY**

Update Vercel environment variable:

```bash
cd "/Users/test/Desktop/Tuscan Money/Ontologic XYZ/ML Research/5. Live System/dashboard_pro"

# Update production env
vercel env rm VITE_API_BASE_URL production
vercel env add VITE_API_BASE_URL production
# Enter: https://your-app.up.railway.app

# Redeploy
vercel --prod
```

Now your Vercel frontend connects to Railway backend!

---

## **🚀 STEP 10: CUSTOM DOMAIN (OPTIONAL)**

### **For Backend:**
1. Railway dashboard → Your Service → Settings → Domains
2. Click "Generate Domain" (free Railway subdomain)
3. Or add custom domain: `api.ontologicxyz.com`

### **For Frontend:**
1. Vercel dashboard → Your Project → Settings → Domains
2. Add custom domain: `ontologicxyz.com`
3. Update DNS:
   - Type: CNAME
   - Name: @
   - Value: cname.vercel-dns.com

---

## **📊 COST BREAKDOWN:**

### **Railway:**
- $5/month FREE credit (hobby tier)
- Python backend: ~$0.50/month
- Postgres: ~$1/month
- Redis: ~$0.50/month
- **Total: $2/month (covered by free credit!)**

### **Vercel:**
- Frontend: FREE (hobby tier)

### **Domain:**
- ontologicxyz.com: ~$12/year

**TOTAL: $12/year for production hosting!** 🎉

---

## **🔥 MONITORING & MAINTENANCE:**

### **View Logs:**
Railway dashboard → Your Service → Deployments → View Logs

### **Restart Service:**
Railway dashboard → Your Service → Settings → Restart

### **Scale Up:**
Railway dashboard → Your Service → Settings → Resources
- Increase RAM (default: 512MB)
- Increase CPU (default: 1 vCPU)

### **Add Alerts:**
Railway dashboard → Your Service → Settings → Notifications
- Deploy failures
- Health check failures
- High CPU/RAM usage

---

## **🚀 QUICK REFERENCE:**

### **Railway URLs:**
- Dashboard: https://railway.app/dashboard
- Docs: https://docs.railway.app

### **Deploy Commands:**
```bash
# Auto-deploy (push to GitHub)
git push origin main

# Manual deploy (Railway CLI)
railway up

# View logs
railway logs

# Open dashboard
railway open
```

---

## **🎯 CHECKLIST:**

```
☐ Step 1: Verify requirements.txt, Procfile, railway.json
☐ Step 2: Sign up for Railway
☐ Step 3: Create project, add Postgres, add Redis
☐ Step 4: Configure environment variables
☐ Step 5: Upload Mamba model (volume or download)
☐ Step 6: Update backend for Postgres
☐ Step 7: Deploy (auto or manual)
☐ Step 8: Verify deployment (health, live-games, opportunities)
☐ Step 9: Connect Vercel to Railway
☐ Step 10: Add custom domain (optional)
☐ Step 11: Monitor logs and celebrate! 🎉
```

---

**READY TO DEPLOY?** Let me know when you're ready to start and I'll walk you through it! 🚂🚀

