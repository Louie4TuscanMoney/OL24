# 🚀 DEPLOY FRONTEND TO VERCEL

Your **backend is LIVE** at: https://ol24-production.up.railway.app/

Now let's deploy the **frontend dashboard** to Vercel!

---

## **📦 OPTION 1: DEPLOY VIA VERCEL CLI (FASTEST)**

### Step 1: Install Vercel CLI
```bash
npm install -g vercel
```

### Step 2: Login to Vercel
```bash
vercel login
```

### Step 3: Deploy from this directory
```bash
cd "/Users/test/Desktop/Tuscan Money/Ontologic XYZ/ML Research/5. Live System/dashboard_pro"
vercel --prod
```

### Step 4: Follow the prompts
- **Set up and deploy?** → Yes
- **Which scope?** → Your Vercel account
- **Link to existing project?** → No (first time) or Yes (if exists)
- **Project name?** → `ontologic-xyz-dashboard` (or your choice)
- **Directory?** → `./` (current directory)
- **Override settings?** → No

### Step 5: Done!
Vercel will give you a URL like:
```
https://ontologic-xyz-dashboard.vercel.app
```

---

## **📦 OPTION 2: DEPLOY VIA VERCEL DASHBOARD (EASIEST)**

### Step 1: Go to Vercel
1. Visit https://vercel.com/new
2. Click **"Import Project"**

### Step 2: Connect GitHub
1. Click **"Import Git Repository"**
2. Select your **OL24** repository
3. Click **"Import"**

### Step 3: Configure Project
1. **Framework Preset:** Vite
2. **Root Directory:** `5. Live System/dashboard_pro`
3. **Build Command:** `npm run build`
4. **Output Directory:** `dist`
5. **Install Command:** `npm install`

### Step 4: Add Environment Variable
Click **"Environment Variables"** and add:
```
VITE_API_URL = https://ol24-production.up.railway.app
```

### Step 5: Deploy!
Click **"Deploy"** and wait ~2 minutes.

---

## **🎯 AFTER DEPLOYMENT:**

### 1. Get Your Frontend URL
Vercel will give you a URL like:
```
https://ontologic-xyz-dashboard.vercel.app
```

### 2. Update CORS on Railway Backend
The backend needs to allow requests from your Vercel frontend.

**Option A: Allow All Origins (Quick)**
Add this environment variable in Railway:
```
CORS_ORIGINS=*
```

**Option B: Specific Origin (Secure)**
Add this environment variable in Railway:
```
CORS_ORIGINS=https://your-frontend.vercel.app
```

### 3. Test Your Dashboard!
Open your Vercel URL in a browser and you should see:
- ✅ Live NBA games
- ✅ BetOnline odds
- ✅ Mamba predictions
- ✅ OntoRisk analysis
- ✅ Real-time updates

---

## **🔥 CUSTOM DOMAIN (OPTIONAL)**

### Deploy to ontologicxyz.com:

1. **In Vercel Dashboard:**
   - Go to your project → Settings → Domains
   - Add `ontologicxyz.com`
   - Follow DNS instructions

2. **In Your Domain Registrar:**
   - Add CNAME record: `ontologicxyz.com` → `cname.vercel-dns.com`
   - Or A record to Vercel's IP

3. **Wait for DNS propagation** (~5-10 minutes)

---

## **✅ VERIFICATION CHECKLIST:**

- [ ] Backend is live: https://ol24-production.up.railway.app/
- [ ] Frontend is deployed to Vercel
- [ ] Environment variable `VITE_API_URL` is set
- [ ] CORS is configured on Railway
- [ ] Dashboard loads in browser
- [ ] Live games are displayed
- [ ] BetOnline odds are showing
- [ ] Predictions are working

---

## **🚨 TROUBLESHOOTING:**

### Frontend shows "Network Error"
→ Check CORS settings on Railway backend

### "System not initialized"
→ Backend is starting up, wait 30 seconds and refresh

### No games showing
→ No live NBA games right now (check ESPN.com)

### Odds not loading
→ BetOnline scraper might be blocked, using fallback

---

## **📞 NEED HELP?**

Just tell me:
1. Which deployment method you're using (CLI or Dashboard)
2. Any errors you see
3. Your Vercel deployment URL

Let's get this live! 🚀

