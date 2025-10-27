# 🔥 Ontologic XYZ - Live Trading Dashboard

**SolidJS + Vite + Vercel**  
**Real-time NBA betting with ML predictions and risk management**

---

## 🚀 Quick Start

### **Local Development:**

```bash
# Install dependencies
npm install

# Start backend API (in another terminal)
cd ../
python trading_dashboard_api.py

# Start dashboard
npm run dev
```

Dashboard will be at: `http://localhost:3000`  
API will be at: `http://localhost:8001`

---

## 📦 Deploy to Vercel

### **Option 1: Vercel CLI**

```bash
npm install -g vercel
vercel login
vercel deploy
```

### **Option 2: GitHub Integration**

1. Push to GitHub
2. Connect repo to Vercel
3. Vercel auto-deploys

---

## 🎯 Features

### **Live Games View** 🏀
- Real-time NBA scores
- Game status (Live, Final, Scheduled)
- Quarter and clock
- ⭐ Highlights Q2 6:00 (prediction point)
- 🎯 Shows predictable games

### **Betting Opportunities** 💰
- ML predictions (9.0 MAE)
- Market spreads (from BetOnline)
- Edge calculation
- Win probability (OntoRisk)
- Kelly-optimal stake sizing
- One-click bet placement

### **Risk Management** 🛡️
- Current bankroll
- Peak bankroll
- Drawdown %
- Daily P&L
- Can bet status
- Risk alerts

### **Real-time Updates** ⚡
- Auto-refreshes every 10s
- WebSocket support (future)
- Live odds tracking
- Instant opportunity alerts

---

## 🏗️ Architecture

```
Dashboard (SolidJS)
    ↓
API (FastAPI)
    ↓
Live Trading Engine
    ├── NBA API (scores)
    ├── BetOnline (lines)
    ├── ML Models (predictions)
    └── OntoRisk (risk mgmt)
```

---

## 🎨 UI/UX

- **Modern gradient design**
- **Card-based layout**
- **Responsive (mobile-ready)**
- **Real-time indicators**
- **One-click betting**
- **Risk status dashboard**

---

## 🔧 Configuration

**Edit API_URL in App.tsx:**

```typescript
const API_URL = 'http://localhost:8001';  // Local
// const API_URL = 'https://your-api.vercel.app';  // Production
```

**Backend runs on port 8001** (separate from OntoRisk API on 8000)

---

## 📊 What You See

1. **Risk Status Card** (top)
   - Bankroll: $10,000
   - Drawdown: 0%
   - Daily P&L: +$0
   - Status: ACTIVE ✅

2. **Betting Opportunities**
   - Game matchup
   - Current score
   - Our prediction
   - Market spread
   - Edge (points)
   - P(Win) %
   - Recommended stake
   - **PLACE BET button**

3. **Live Games**
   - All NBA games today
   - Live scores
   - Quarter/clock
   - Status indicators

---

## 🚀 Deployment

**Backend:**
- Needs to run on server (not Vercel functions)
- Heroku, Railway, Render, or DigitalOcean
- Keep running 24/7

**Frontend:**
- Vercel (easy deploy)
- Updates automatically
- Global CDN
- Free for hobby projects

---

## ⚠️ Current Status

**Working:**
- ✅ Dashboard UI
- ✅ API integration
- ✅ Risk display
- ✅ Opportunity cards
- ✅ Live updates

**Using Synthetic Data:**
- ⚠️ NBA API (no live games currently)
- ⚠️ BetOnline (synthetic lines)

**Production Ready Week 2:**
- Real NBA games
- Real BetOnline lines
- Full integration

---

## 💡 Usage

**Auto-updates every 10 seconds**  
**When NBA games are live:**
1. Dashboard shows live scores
2. At Q2 6:00, makes prediction
3. Compares to BetOnline spread
4. If edge ≥ 5 pts: Shows bet opportunity
5. Click "PLACE BET" to log bet

**Risk management enforced:**
- Won't show bets if daily limit hit
- Won't show bets if drawdown > 30%
- Kelly sizing prevents over-betting

---

**BUILT FOR ONTOLOGIC XYZ** 🔥  
**Mamba Mentality System**  
**9.0 MAE • OntoRisk • Production Ready**

