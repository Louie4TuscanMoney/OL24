# 📖 MANUAL DEPLOYMENT GUIDE - STEP BY STEP

**Issue:** Vercel needs SSH key / login first  
**Solution:** Do it manually in steps

---

## 🚀 STEP-BY-STEP DEPLOYMENT

### **STEP 1: Start Autonomous System (Local)**

```bash
cd "/Users/test/Desktop/Tuscan Money/Ontologic XYZ/ML Research"
bash 🚀_START_AUTONOMOUS_SYSTEM.sh
```

**Wait for:**
```
✅ Daemon started (PID: XXXXX)
✅ API started (PID: XXXXX)
```

**Leave this running!** (Runs 24/7 on your computer)

---

### **STEP 2: Copy Dashboard to Website Repo**

```bash
cd "/Users/test/Desktop/Tuscan Money/Ontologic XYZ/ML Research"
bash 📦_DEPLOY_TO_WEBSITE.sh
```

**This copies files to:**
```
/Users/test/Desktop/Tuscan Money/Websites/OntologicXYZ.com/NBADashboard/
```

---

### **STEP 3: Install Dashboard Dependencies**

```bash
cd "/Users/test/Desktop/Tuscan Money/Websites/OntologicXYZ.com/NBADashboard"
npm install
```

**This installs:** SolidJS, Vite, TailwindCSS, Three.js, etc.

---

### **STEP 4: Test Locally (Optional)**

```bash
npm run dev
```

**Access:** `http://localhost:3000`

**Check:**
- Dashboard loads ✅
- Shows "ONTOLOGIC XYZ" header
- Risk management section
- Betting opportunities section
- Live games section

**Press Ctrl+C to stop** when satisfied

---

### **STEP 5: Login to Vercel**

```bash
npm install -g vercel
vercel login
```

**Follow prompts:**
1. Choose login method (email, GitHub, GitLab, Bitbucket)
2. Complete authentication
3. Wait for "✅ Success! Authentication complete"

---

### **STEP 6: Deploy to Vercel**

```bash
vercel deploy --prod
```

**Vercel will ask:**

1. **Set up and deploy?** → `Y` (yes)
2. **Which scope?** → Choose your account
3. **Link to existing project?** → `N` (no, create new)
4. **Project name?** → `nba-dashboard` or keep default
5. **Directory?** → `.` (current, press Enter)
6. **Build settings?** → Use defaults (press Enter)

**Wait for deployment...**

**When complete:**
```
✅ Production: https://nba-dashboard-xxxxx.vercel.app
```

---

### **STEP 7: Configure Custom Domain (Optional)**

**In Vercel Dashboard:**
1. Go to project settings
2. Domains → Add domain
3. Enter: `ontologicxyz.com/NBADashboard` or subdomain
4. Follow DNS instructions

**Or use Vercel URL directly** (works immediately)

---

### **STEP 8: Update API URL in Dashboard**

**If backend is on your computer:**

In Vercel dashboard:
1. Project → Settings → Environment Variables
2. Add: `VITE_API_URL` = `http://your-public-ip:8001`

**OR deploy backend to cloud:**

**Option A: Heroku**
```bash
cd "/Users/test/Desktop/Tuscan Money/Ontologic XYZ/ML Research/5. Live System"
heroku create ontologic-xyz-api
git init
git add .
git commit -m "Initial commit"
git push heroku main
```

**Option B: Railway**
```bash
railway init
railway up
```

Then update `VITE_API_URL` to Railway/Heroku URL

---

## 🎯 SIMPLIFIED: JUST GET DASHBOARD LIVE

### **Quick Deploy (Dashboard Only):**

```bash
# 1. Go to website directory
cd "/Users/test/Desktop/Tuscan Money/Websites/OntologicXYZ.com/NBADashboard"

# 2. Copy files (if not done)
cd "/Users/test/Desktop/Tuscan Money/Ontologic XYZ/ML Research"
bash 📦_DEPLOY_TO_WEBSITE.sh
cd "/Users/test/Desktop/Tuscan Money/Websites/OntologicXYZ.com/NBADashboard"

# 3. Install
npm install

# 4. Login to Vercel (first time only)
vercel login

# 5. Deploy
vercel deploy --prod
```

**That's it!** Dashboard will be live (backend can wait for Week 2)

---

## 🔥 ALTERNATIVE: GitHub → Vercel (Easier)

### **Option 1: Push to GitHub**

```bash
cd "/Users/test/Desktop/Tuscan Money/Websites/OntologicXYZ.com/NBADashboard"

# Initialize git (if not already)
git init
git add .
git commit -m "NBA Trading Dashboard"

# Push to GitHub
git remote add origin https://github.com/yourusername/ontologic-xyz.git
git push -u origin main
```

### **Option 2: Connect to Vercel**

1. Go to https://vercel.com
2. Click "Add New Project"
3. Import from GitHub
4. Select your repo
5. Click "Deploy"

**Vercel auto-deploys!** No SSH keys needed.

---

## 💡 RECOMMENDED PATH (Easiest)

### **For Now (Tonight):**

**Just start the autonomous system locally:**
```bash
cd "/Users/test/Desktop/Tuscan Money/Ontologic XYZ/ML Research"
bash 🚀_START_AUTONOMOUS_SYSTEM.sh
```

**Test dashboard locally:**
```bash
cd "5. Live System/dashboard_pro"
npm install
npm run dev
```

**Access:** `http://localhost:3000`

---

### **For Week 2 (When NBA Season Starts):**

**Then deploy to Vercel:**
1. Push to GitHub
2. Connect Vercel to GitHub
3. Auto-deploys on every push

**This is easier than CLI deployment** (no SSH issues)

---

## 🎯 CURRENT STATUS

**Working Now:**
- ✅ Autonomous system (can run on your computer)
- ✅ Dashboard (can run locally)
- ✅ All features work
- ✅ Bet tracking works
- ✅ 3D court works

**For Production:**
- ⚠️ Need Vercel login/GitHub setup
- ⚠️ Can do Week 2 when NBA starts
- ⚠️ Or push to GitHub and use Vercel GUI

---

## 💡 EASIEST SOLUTION

**For Tonight:**
```bash
# Start system
cd "/Users/test/Desktop/Tuscan Money/Ontologic XYZ/ML Research"
bash 🚀_START_AUTONOMOUS_SYSTEM.sh

# Test dashboard
cd "5. Live System/dashboard_pro"
npm install
npm run dev

# Access: http://localhost:3000
```

**For Week 2:**
- Push to GitHub
- Deploy via Vercel GUI (no CLI needed)
- Avoids SSH/authentication issues

---

**SYSTEM IS READY - Just use locally for now, deploy to website Week 2!** ✅

