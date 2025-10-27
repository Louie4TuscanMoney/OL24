# 🚀 LIVE SYSTEM OFFICIAL - AUTONOMOUS NBA TRADING ENGINE

**The Heart of the Real-Time Prediction & Betting System**

---

## 🎯 **WHAT IS LIVE SYSTEM?**

**Live System** is the **autonomous trading engine** that powers real-time NBA betting predictions. It integrates:

- **Live game data** (ESPN API, NBA API, CDN)
- **BetOnline odds** (scraped dynamically)
- **Mamba Mentality ML predictions** (67 features via Helios)
- **OntoRisk probability calibration** (Kelly bet sizing)
- **Real-time dashboard** (SolidJS/Vite, deployed on Vercel at **ontologicxyz.com**)

This is the **operational core** - the system your friends use to access live predictions and betting opportunities.

---

## 🔥 **KEY CAPABILITIES**

### **1. Live Trading Engine**
- **Scans opportunities every 3 seconds** for all live NBA games
- **Makes predictions** at Q2 6:00 and continuously through Q3, Q4
- **Extracts 67 Mamba features** in real-time
- **Calibrates probabilities** using OntoRisk
- **Calculates optimal bet sizes** using Kelly Criterion

### **2. FastAPI Backend**
- **REST API** for dashboard communication
- **CORS-enabled** for Vercel frontend
- **Endpoints:**
  - `/api/live-games` - Live game scores and status
  - `/api/betonline-odds` - Real-time BetOnline odds
  - `/api/opportunities` - ML-powered betting opportunities
  - `/api/mamba-performance` - Mamba prediction tracking
  - `/api/mamba-scores` - Stored Mamba scores after Q2 6:00
  - `/api/user/signup` - User authentication
  - `/api/admin/pending-requests` - Admin approval system

### **3. SolidJS Dashboard**
- **Real-time updates** (polling every 3 seconds)
- **3D basketball court visualization**
- **ML model visualization** (feature importance, confidence)
- **Bet tracking modal** (track your bets, ROI, performance)
- **Game detail modal** (deep dive into specific games)
- **Responsive design** (mobile-friendly)
- **Deployed on Vercel** at **ontologicxyz.com**

### **4. User Authentication**
- **SQLite-based** user management
- **Approval workflow** (users request access, admin approves)
- **Secure API** (authentication tokens)

### **5. BetOnline Scraper**
- **Crawlee-based** web scraper (Playwright + Stealth)
- **Fallback chain:** Crawlee → API → HTML → Synthetic
- **Real-time odds** for spread, total, moneylines
- **Implied probabilities** (no-vig, vig percentage)

---

## 📂 **PACKAGE STRUCTURE**

```
livesystemofficial/
├── core/                          # Core trading engine
│   └── live_trading_engine.py           # Main prediction engine
│
├── api/                           # FastAPI backend
│   └── trading_dashboard_api.py         # REST API endpoints
│
├── authentication/                # User management
│   └── user_auth_manager.py             # SQLite auth system
│
├── utilities/                     # Helper modules
│   ├── implied_probability_calculator.py  # Odds → Probability
│   └── crawlee_betonline_scraper.py       # BetOnline scraper
│
├── dashboard/                     # SolidJS frontend
│   ├── package.json                      # npm dependencies
│   ├── vite.config.ts                    # Vite config
│   ├── vercel.json                       # Vercel deployment
│   ├── index.html                        # Entry point
│   └── src/
│       ├── App.tsx                       # Main app
│       ├── AppFinal.tsx                  # Production app
│       ├── index.tsx                     # Entry point
│       └── components/
│           ├── BasketballCourt3D.tsx     # 3D court viz
│           ├── BetTrackingModal.tsx      # Bet tracking
│           ├── GameDetailModal.tsx       # Game details
│           ├── MLModelVisualization3D.tsx # ML viz
│           └── OpportunityCard.tsx       # Opportunity cards
│
├── config/                        # Configuration & scripts
│   ├── requirements.txt                  # Python dependencies
│   ├── Procfile                          # Deployment config
│   ├── start_autonomous_system.sh        # Start script
│   ├── stop_autonomous_system.sh         # Stop script
│   └── approve_users.sh                  # User approval script
│
├── documentation/                 # Deployment guides
│   ├── DEPLOY_FOR_FRIENDS_NOW.md        # Friend deployment
│   ├── MAKE_BACKEND_PUBLIC.md           # Public backend guide
│   └── GIT_VERCEL_GUIDE.md              # Git/Vercel setup
│
└── README.md                      # This file
```

---

## 🚀 **QUICK START**

### **1. Start the Backend (Local)**

```bash
cd livesystemofficial/api
python3 trading_dashboard_api.py
```

Backend will run on `http://localhost:8001`.

### **2. Start the Dashboard (Local)**

```bash
cd livesystemofficial/dashboard
npm install
npm run dev
```

Dashboard will run on `http://localhost:3000`.

### **3. Deploy to Vercel**

```bash
# From dashboard directory
cd livesystemofficial/dashboard
vercel --prod
```

### **4. Make Backend Public (ngrok)**

```bash
ngrok config add-authtoken YOUR_NGROK_TOKEN
ngrok http 8001
```

Then set the ngrok URL as `VITE_API_BASE_URL` in Vercel environment variables.

---

## 🎯 **HOW IT WORKS**

```
┌─────────────────────────────────────────────────────────────┐
│                    LIVE SYSTEM WORKFLOW                     │
└─────────────────────────────────────────────────────────────┘

1. LIVE DATA FETCHING (ESPN/NBA API)
   ├─ espnapiofficial (nba_live_scores.py)
   ├─ Multi-source fallback (ESPN → NBA API → CDN)
   └─ Poll every 3 seconds
   ↓
2. BETONLINE ODDS SCRAPING
   ├─ betonlineofficial (crawlee_betonline_scraper.py)
   ├─ Fallback chain: Crawlee → API → HTML → Synthetic
   └─ Real-time spread, total, moneylines
   ↓
3. FEATURE EXTRACTION (Q2 6:00+)
   ├─ heliosofficial (67 features)
   ├─ 18 pattern + 4 stat + 5 spectral + 10 momentum + 10 autocorr + 20 NBA
   └─ <100ms extraction time
   ↓
4. MAMBA PREDICTION
   ├─ mambaofficial (MAMBA_MENTALITY_SYSTEM.pkl)
   ├─ XGBoost model (trained on 5,529 games)
   └─ Predicts final score differential
   ↓
5. ONTORISK CALIBRATION
   ├─ ontoriskofficial (ProbabilityCalibrator, RiskManager)
   ├─ Converts prediction → calibrated probability
   ├─ Compares to implied odds from BetOnline
   └─ Calculates Kelly edge
   ↓
6. OPTIMAL BET SIZING (Kelly Criterion)
   ├─ Kelly edge = Mamba probability - Market probability
   ├─ Stake = (edge × bankroll) / odds
   └─ Risk limits: Max 5% per bet, max 20% total exposure
   ↓
7. DASHBOARD DISPLAY (ontologicxyz.com)
   ├─ Live scores, odds, predictions
   ├─ 3D visualizations
   ├─ Bet tracking
   └─ Real-time updates (3s polling)
```

---

## 🧠 **CORE COMPONENTS**

### **1. Live Trading Engine** (`core/live_trading_engine.py`)

The **brain** of the system:

```python
from livesystemofficial.core.live_trading_engine import LiveTradingEngine

engine = LiveTradingEngine()

# Scan for opportunities
opportunities = engine.scan_live_opportunities()
for opp in opportunities:
    print(f"Game: {opp['game_id']}")
    print(f"Mamba prediction: {opp['mamba_prediction']:.1f}")
    print(f"Recommended stake: ${opp['stake']:.2f}")
    print(f"Kelly edge: {opp['kelly_edge']:.2%}")
```

**Key methods:**
- `scan_live_opportunities()` - Main loop (runs every 3s)
- `make_live_prediction()` - Extract features → Mamba → OntoRisk
- `extract_features_from_live_game()` - 67 features via Helios
- `get_mamba_performance()` - Track prediction accuracy
- `get_stored_mamba_scores()` - Retrieve stored scores

### **2. Trading Dashboard API** (`api/trading_dashboard_api.py`)

The **communication layer** between backend and frontend:

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS for Vercel
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production: ["https://ontologicxyz.com"]
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/opportunities")
async def get_opportunities():
    return trading_engine.scan_live_opportunities()
```

**Endpoints:**
- `GET /api/live-games` - All live games
- `GET /api/betonline-odds?game_id={id}` - Odds for specific game
- `GET /api/opportunities` - ML opportunities
- `GET /api/mamba-performance` - Mamba stats
- `GET /api/mamba-scores` - Stored scores
- `POST /api/user/signup` - User signup
- `GET /api/admin/pending-requests` - Admin panel

### **3. User Auth Manager** (`authentication/user_auth_manager.py`)

Manages user access:

```python
from livesystemofficial.authentication.user_auth_manager import UserAuthManager

auth = UserAuthManager()

# Request access
auth.request_access(
    username="john_doe",
    email="john@example.com",
    reason="Want to try the system!"
)

# Approve user (admin)
auth.approve_request(request_id=1, admin_notes="Approved by Louie")
```

**Database:** `users.db` (SQLite)
- `user_requests` table (pending approvals)
- `approved_users` table (active users)

### **4. Implied Probability Calculator** (`utilities/implied_probability_calculator.py`)

Converts American odds to probabilities:

```python
from livesystemofficial.utilities.implied_probability_calculator import ImpliedProbabilityCalculator

calc = ImpliedProbabilityCalculator()

# American odds → implied probability
prob = calc.american_to_probability(-110)  # 52.38%

# Calculate no-vig probabilities
result = calc.calculate_market_probabilities(
    home_odds=-110,
    away_odds=-110
)
print(f"Home no-vig prob: {result['home_no_vig_prob']:.2%}")
print(f"Vig: {result['vig_percentage']:.2%}")
```

### **5. Crawlee BetOnline Scraper** (`utilities/crawlee_betonline_scraper.py`)

Real-time odds scraper:

```python
from livesystemofficial.utilities.crawlee_betonline_scraper import get_crawlee_betonline_odds
import asyncio

# Scrape live odds
odds = asyncio.run(get_crawlee_betonline_odds())
for game in odds:
    print(f"{game['away_team']} @ {game['home_team']}")
    print(f"Spread: {game['spread_home']}")
    print(f"Total: {game['total_over']}")
```

**Features:**
- Playwright-based browser automation
- Stealth mode (evade bot detection)
- Fallback chain (3 methods + synthetic)
- Real-time updates (5s polling)

---

## 🎨 **DASHBOARD (SolidJS/Vite)**

### **Frontend Tech Stack**
- **SolidJS** - Reactive UI framework
- **Vite** - Build tool (fast HMR)
- **TypeScript** - Type safety
- **Three.js** - 3D visualizations
- **Tailwind CSS** - Styling
- **Deployed on Vercel** at **ontologicxyz.com**

### **Key Components**

#### **App.tsx** - Main application
```tsx
import { createSignal, onMount } from 'solid-js';

const [games, setGames] = createSignal([]);
const [opportunities, setOpportunities] = createSignal([]);

// Poll every 3 seconds
setInterval(() => {
  fetchLiveGames();
  fetchOpportunities();
}, 3000);
```

#### **BasketballCourt3D.tsx** - 3D court visualization
```tsx
import * as THREE from 'three';

// Render 3D court with live game state
function BasketballCourt3D(props) {
  // Three.js scene, camera, renderer
  // Animate based on live scores
}
```

#### **BetTrackingModal.tsx** - Bet tracking
```tsx
function BetTrackingModal() {
  const [bets, setBets] = createSignal([]);
  
  // Track bets, calculate ROI
  const totalProfit = () => bets().reduce((sum, bet) => sum + bet.profit, 0);
  const roi = () => (totalProfit() / totalStaked()) * 100;
}
```

---

## 📊 **SYSTEM PERFORMANCE**

### **Latency Analysis**
- **ESPN API delay:** 10-15 seconds (inherent)
- **Backend polling:** 3 seconds
- **Dashboard polling:** 3 seconds
- **Game detail polling:** 2 seconds
- **BetOnline scraping:** 5 seconds
- **Total average latency:** **13.2 seconds** (faster than DraftKings, FanDuel!)

### **Prediction Accuracy**
- **Mamba MAE (training):** 6.8 points
- **Mamba MAE (test):** 9.3 points
- **Training data:** 5,529 games
- **Test data:** 1,383 games

### **Feature Extraction Speed**
- **67 features extracted:** <100ms
- **Helios pipeline:** Real-time (live games)

### **OntoRisk Calibration**
- **Probability calibration:** Sigmoid-based
- **Kelly edge calculation:** Real-time
- **Risk limits:** Max 5% per bet, max 20% total exposure

---

## 🔧 **DEPLOYMENT**

### **Local Development**

```bash
# Backend
cd livesystemofficial/api
python3 trading_dashboard_api.py

# Frontend
cd livesystemofficial/dashboard
npm install
npm run dev
```

### **Production (Vercel + ngrok)**

```bash
# 1. Deploy frontend to Vercel
cd livesystemofficial/dashboard
vercel --prod

# 2. Make backend public with ngrok
ngrok config add-authtoken YOUR_TOKEN
ngrok http 8001

# 3. Set Vercel environment variable
# Go to Vercel Dashboard → Settings → Environment Variables
# Add: VITE_API_BASE_URL = https://your-ngrok-url.ngrok.io
```

### **Autonomous Mode**

```bash
# Start autonomous system (backend only)
cd livesystemofficial/config
bash start_autonomous_system.sh

# Stop autonomous system
bash stop_autonomous_system.sh
```

---

## 🤝 **USER MANAGEMENT**

### **Request Access**

Users visit **ontologicxyz.com** and request access:

```bash
# Backend receives request
POST /api/user/signup
{
  "username": "john_doe",
  "email": "john@example.com",
  "reason": "Want to try the system!"
}
```

### **Approve Users (Admin)**

```bash
# View pending requests
python3 -c "from user_auth_manager import UserAuthManager; auth = UserAuthManager(); print(auth.get_pending_requests())"

# Approve user
bash approve_users.sh
# Follow prompts to approve specific users
```

---

## 🧪 **TESTING**

### **Test Backend API**

```bash
curl http://localhost:8001/api/live-games
curl http://localhost:8001/api/opportunities
curl http://localhost:8001/api/mamba-performance
```

### **Test BetOnline Scraper**

```python
from utilities.crawlee_betonline_scraper import get_crawlee_betonline_odds
import asyncio

odds = asyncio.run(get_crawlee_betonline_odds())
print(f"Found {len(odds)} games with odds")
```

### **Test Mamba Prediction**

```python
from core.live_trading_engine import LiveTradingEngine

engine = LiveTradingEngine()
opportunities = engine.scan_live_opportunities()
print(f"Found {len(opportunities)} opportunities")
```

---

## 📈 **FUTURE ENHANCEMENTS**

### **Phase 1: Real-Time Push (WebSockets)** 📋 PLANNED
- Replace polling with WebSocket push updates
- Sub-second latency
- Lower server load

### **Phase 2: Cloud Deployment** 📋 PLANNED
- Deploy backend to Railway/Render (persistent hosting)
- Postgres database (replace SQLite)
- Redis caching (rate limiting, session management)

### **Phase 3: Advanced Risk Management** 🚧 IN PROGRESS
- Portfolio optimization (max Sharpe ratio)
- Delta tracking (line movement)
- Live adjustments (hedge positions)

### **Phase 4: Mobile App** 📋 PLANNED
- React Native app
- Push notifications for opportunities
- Offline mode (cached predictions)

---

## 🛠️ **DEPENDENCIES**

### **Backend (Python)**
```
fastapi >= 0.68.0
uvicorn >= 0.15.0
requests >= 2.26.0
nba_api >= 1.1.9
numpy >= 1.21.0
pandas >= 1.3.0
scikit-learn >= 0.24.0
xgboost >= 1.4.0
playwright >= 1.20.0
asyncio
```

### **Frontend (npm)**
```json
{
  "solid-js": "^1.8.0",
  "vite": "^5.0.0",
  "typescript": "^5.0.0",
  "three": "^0.160.0",
  "tailwindcss": "^3.4.0"
}
```

---

## 📚 **DOCUMENTATION**

- **[DEPLOY_FOR_FRIENDS_NOW.md](documentation/DEPLOY_FOR_FRIENDS_NOW.md)** - Friend deployment guide
- **[MAKE_BACKEND_PUBLIC.md](documentation/MAKE_BACKEND_PUBLIC.md)** - Public backend guide
- **[GIT_VERCEL_GUIDE.md](documentation/GIT_VERCEL_GUIDE.md)** - Git/Vercel setup

---

## 🎯 **QUICK REFERENCE**

```python
# Start backend
from api.trading_dashboard_api import app
import uvicorn
uvicorn.run(app, host="0.0.0.0", port=8001)

# Scan opportunities
from core.live_trading_engine import LiveTradingEngine
engine = LiveTradingEngine()
opportunities = engine.scan_live_opportunities()

# Get Mamba performance
performance = engine.get_mamba_performance()
print(f"Total predictions: {performance['total_predictions']}")
print(f"Avg accuracy: {performance['avg_accuracy']:.2%}")

# Approve user
from authentication.user_auth_manager import UserAuthManager
auth = UserAuthManager()
auth.approve_request(request_id=1)
```

---

**🚀 Live System is the operational heart of your autonomous NBA trading empire! 🚀**

**Currently deployed at: https://ontologicxyz.com**

