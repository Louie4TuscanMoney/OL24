# 🌐 NGROK SETUP - EXPOSE YOUR BACKEND TO THE INTERNET

**Get your Mamba system live in 5 minutes for testing!**

---

## **🎯 WHAT NGROK DOES:**

Ngrok creates a **secure tunnel** from the internet to your local backend:

```
Internet → https://your-app.ngrok.io → Your Mac (localhost:8000)
```

This means:
- ✅ Your backend runs on your Mac (with the Mamba model)
- ✅ Friends can access it from anywhere
- ✅ Vercel frontend can connect to it
- ✅ **FREE** for testing

---

## **📦 STEP 1: INSTALL NGROK**

### **Option A: Homebrew (Recommended)**
```bash
brew install ngrok/ngrok/ngrok
```

### **Option B: Direct Download**
1. Go to https://ngrok.com/download
2. Download for macOS
3. Unzip and move to `/usr/local/bin`

---

## **🔑 STEP 2: GET YOUR AUTHTOKEN**

1. Sign up at https://dashboard.ngrok.com/signup (FREE)
2. Go to https://dashboard.ngrok.com/get-started/your-authtoken
3. Copy your authtoken

Then configure it:
```bash
ngrok config add-authtoken YOUR_AUTHTOKEN_HERE
```

---

## **🚀 STEP 3: START YOUR BACKEND**

```bash
cd "/Users/test/Desktop/Tuscan Money/Ontologic XYZ/ML Research/5. Live System"

# Start the trading API
python3 trading_dashboard_api.py
```

You should see:
```
✅ Mamba model loaded successfully
INFO:     Uvicorn running on http://127.0.0.1:8000
```

**Keep this terminal running!**

---

## **🌐 STEP 4: START NGROK (NEW TERMINAL)**

Open a **NEW** terminal:

```bash
cd "/Users/test/Desktop/Tuscan Money/Ontologic XYZ/ML Research/5. Live System"

# Expose port 8000 to the internet
ngrok http 8000
```

You'll see:
```
Session Status                online
Account                       Your Name (Plan: Free)
Forwarding                    https://abc123.ngrok.io -> http://localhost:8000
```

**COPY THAT URL!** (e.g., `https://abc123.ngrok.io`)

---

## **🧪 STEP 5: TEST IT**

### **Test 1: From your Mac**
```bash
curl https://abc123.ngrok.io/api/health
```

Should return:
```json
{"status":"healthy","timestamp":"2025-10-27T..."}
```

### **Test 2: From your phone**
Open Safari on your phone and go to:
```
https://abc123.ngrok.io/api/live-games
```

You should see live NBA games!

### **Test 3: From Vercel**
Go to your Vercel dashboard:
1. Project Settings → Environment Variables
2. Add: `VITE_API_BASE_URL = https://abc123.ngrok.io`
3. Redeploy

Now your Vercel frontend will connect to your local backend!

---

## **📊 STEP 6: MONITOR TRAFFIC**

Ngrok has a built-in web interface:
```
http://localhost:4040
```

Open this in your browser to see:
- All API requests
- Response times
- Request/response bodies
- Errors

**This is GOLD for debugging!**

---

## **⚠️ NGROK LIMITATIONS (Why we need Railway):**

### **Free Tier:**
- ✅ 1 online ngrok process
- ✅ 4 tunnels/ngrok process
- ✅ 40 connections/minute
- ❌ Random URL every restart (changes to `https://xyz789.ngrok.io`)
- ❌ No custom domains
- ❌ Session expires after 2 hours (need to restart)

### **For Production:**
You'll want Railway for:
- ✅ Persistent URL (never changes)
- ✅ No session limits
- ✅ Custom domain (ontologicxyz.com)
- ✅ Automatic restarts
- ✅ Postgres database
- ✅ Redis caching

---

## **🔥 QUICK COMMANDS:**

### **Start Everything:**
```bash
# Terminal 1: Backend
cd "/Users/test/Desktop/Tuscan Money/Ontologic XYZ/ML Research/5. Live System"
python3 trading_dashboard_api.py

# Terminal 2: Ngrok
ngrok http 8000

# Terminal 3: Frontend (local testing)
cd "/Users/test/Desktop/Tuscan Money/Ontologic XYZ/ML Research/5. Live System/dashboard_pro"
npm run dev
```

### **Stop Everything:**
```bash
# Press Ctrl+C in each terminal
```

---

## **🎯 NEXT: UPDATE VERCEL**

Once ngrok is running:

1. Copy your ngrok URL (e.g., `https://abc123.ngrok.io`)

2. Update Vercel:
```bash
cd "/Users/test/Desktop/Tuscan Money/Ontologic XYZ/ML Research/5. Live System/dashboard_pro"

# Set environment variable
vercel env add VITE_API_BASE_URL production
# Paste: https://abc123.ngrok.io

# Redeploy
vercel --prod
```

3. Test your Vercel deployment:
```
https://your-app.vercel.app
```

It should now connect to your local backend via ngrok!

---

## **💡 PRO TIPS:**

### **Tip 1: Keep Ngrok Running**
Ngrok sessions expire. To keep it alive:
```bash
# Add this to your ngrok config
ngrok http 8000 --log=stdout > ngrok.log 2>&1 &
```

### **Tip 2: Use Ngrok Dashboard**
Monitor at: http://localhost:4040

### **Tip 3: Update Vercel Env When Ngrok Restarts**
Every time you restart ngrok, the URL changes. You'll need to:
1. Copy new ngrok URL
2. Update Vercel env: `vercel env add VITE_API_BASE_URL production`
3. Redeploy: `vercel --prod`

**This is why Railway is better for production!**

---

## **🚀 WHEN TO MOVE TO RAILWAY:**

Move to Railway when:
- ✅ You've tested with friends (ngrok works!)
- ✅ You're ready for 24/7 uptime
- ✅ You want a custom domain
- ✅ You want to add Postgres/Redis
- ✅ You're making money and want reliability

**For now, ngrok is PERFECT for testing!**

---

**READY TO START?** Let me know when ngrok is running and I'll help you deploy to Railway next! 🚀

