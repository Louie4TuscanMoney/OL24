# 🌐 DEPLOY FOR FRIENDS - COMPLETE GUIDE

**Get your friends online in 10 minutes! Game 2 starts in 35 minutes!**

---

## ✅ WHAT'S READY

- ✅ Backend running locally (port 8001)
- ✅ Dashboard code on GitHub (OL24 repo)
- ✅ ngrok installed
- ✅ CORS configured
- ✅ User sign-ups working (3 pending requests!)
- ✅ Terser included (Vercel will build!)

---

## 🚀 3-STEP DEPLOYMENT (10 MINUTES)

### **STEP 1: START NGROK (2 MIN)**

Open a **NEW terminal** and run:

```bash
ngrok http 8001
```

You'll see:

```
ngrok                                                                   
                                                                        
Session Status: online                                                
Account: Your Account (Plan: Free)                                    
Forwarding: https://1a2b-3c4d-5e6f.ngrok-free.app -> http://localhost:8001
```

**✅ COPY THE HTTPS URL!**

Example: `https://1a2b-3c4d-5e6f.ngrok-free.app`

**⚠️ LEAVE THIS TERMINAL OPEN!** (Closing it stops the tunnel)

---

### **STEP 2: DEPLOY TO VERCEL (5 MIN)**

1. **Go to:** https://vercel.com/new

2. **Click:** "Import Git Repository"

3. **Find:** Louie4TuscanMoney/OL24

4. **Click:** "Import"

5. **Add Environment Variable:**
   - Click "Environment Variables"
   - Variable: `VITE_API_URL`
   - Value: `[YOUR NGROK URL]` (e.g., `https://1a2b-3c4d-5e6f.ngrok-free.app`)
   - Apply to: Production, Preview, Development

6. **Verify Settings:**
   - Framework: Vite ✅ (auto-detected)
   - Build Command: `npm run build` ✅
   - Output Directory: `dist` ✅

7. **Click:** "Deploy"

8. **Wait:** ~2 minutes for build

---

### **STEP 3: SHARE WITH FRIENDS (1 MIN)**

After deploy, you'll get a URL like:

```
https://ol24-louie4tuscanmoney.vercel.app
```

**Share this with friends:**

```
🏀 OntologicXYZ NBA Trading Dashboard
URL: https://ol24-louie4tuscanmoney.vercel.app
Password: rwwc2018

Features:
- Live NBA scores (3-second updates)
- BetOnline odds (dynamic, real-time)
- ML predictions (Q2 6:00 windows)
- 3D visualizations
- Bet tracking

Game 2 tonight: LAL vs GSW at 7:00 PM PT
First prediction: ~7:51 PM (Q2 6:00)
```

---

## 💡 WHAT YOUR FRIENDS WILL SEE

### **Before Game 2 Starts:**
- HALFTIME status for HOU vs OKC
- Scores: HOU 57 - OKC 51
- BetOnline: LOCKED (during halftime)

### **During Game 2 (LAL vs GSW):**
- Live scores updating every 3 seconds
- BetOnline odds (spread, total, moneyline)
- Game state (period, clock, leading team)

### **At Q2 6:00 (~7:51 PM):**
- 🎯 ML Prediction fires!
- Opportunity card appears
- Shows: Prediction, edge, bet recommendation
- Or: Context info with skip reason

### **They Can:**
- Click any game for detailed view (2-second updates!)
- See implied probabilities
- Track bets (if they make any)
- Watch 3D ML brain visualization

---

## ⚠️ IMPORTANT NOTES

### **Keep Your Computer On:**
- ngrok tunnels your LOCAL backend
- If you shut down computer, friends lose connection
- Backend must keep running

### **Keep 3 Terminals Open:**
1. **Backend:** `bash 🚀_START_AUTONOMOUS_SYSTEM.sh` (already running)
2. **Ngrok:** `ngrok http 8001` (NEW - must stay open!)
3. **Local Dashboard:** `npm run dev` (for your own use, optional)

### **Ngrok URL Changes:**
- Free tier: URL changes every restart
- If you close ngrok and restart, you'll get a NEW URL
- You'd need to update Vercel env var again
- Paid tier ($8/mo): Permanent URL

---

## 🔧 IF NGROK URL CHANGES

If you restart ngrok and get a new URL:

1. Go to: https://vercel.com/dashboard
2. Select: OL24 project
3. Settings → Environment Variables
4. Edit `VITE_API_URL` → Enter new ngrok URL
5. Deployments → Redeploy latest

---

## 👥 APPROVING FRIENDS

You have 1 real user waiting! (8472579747)

### **To approve them:**

```bash
curl -X POST http://localhost:8001/api/auth/approve-request \
  -H "Content-Type: application/json" \
  -d '{"request_id": 1, "admin_password": "rwwc2018"}'
```

### **Or run the approval script:**

```bash
bash 👥_APPROVE_USERS.sh
```

Then they can log in with the password they chose!

---

## 🎯 COMPLETE CHECKLIST

- [ ] New terminal open
- [ ] Run: `ngrok http 8001`
- [ ] Copy ngrok HTTPS URL
- [ ] Go to vercel.com/new
- [ ] Import OL24
- [ ] Add env var: VITE_API_URL = ngrok URL
- [ ] Deploy
- [ ] Copy Vercel URL
- [ ] Share with friends
- [ ] Approve user requests
- [ ] Test by visiting Vercel URL yourself

---

## 💰 COSTS

**Today (Free!):**
- ngrok free tier ✅
- Vercel free tier ✅
- Total: $0

**For permanent setup (later):**
- ngrok paid: $8/mo (permanent URL)
- OR Railway backend: $5/mo
- Vercel: Free forever (hobbyist)

---

## 🏀 TIMELINE

**Now - 6:30 PM:**
1. Start ngrok (2 min)
2. Deploy Vercel (5 min)
3. Share with friends (1 min)
4. Approve user (1 min)

**7:00 PM:** Game 2 starts (LAL vs GSW)

**7:51 PM:** Q2 6:00 - First live prediction!
- Your friends see it LIVE on Vercel! 🎉
- You see it on localhost:3002
- Everyone celebrates together! 🏀🔥

---

## 🚀 GO! OPEN NEW TERMINAL AND RUN:

```bash
ngrok http 8001
```

**Then paste the HTTPS URL here and I'll help with Vercel!**

