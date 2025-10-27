# 🏀 NBA Prediction Dashboard - SolidJS + Vite

**Modern web app for live NBA predictions and betting**

Built with:
- ⚡ Vite (blazing fast)
- 🎨 SolidJS (reactive UI)
- 🐍 Python backend (ML models)
- 🚀 Vercel-ready deployment

---

## 🚀 Quick Start

### **Install:**
```bash
cd nba-dashboard
npm install
```

### **Run Development:**
```bash
# Terminal 1: Start backend
npm run server

# Terminal 2: Start frontend  
npm run dev

# Open: http://localhost:3000
```

### **Deploy to Vercel:**
```bash
npm run build
vercel deploy
```

---

## 📂 Project Structure

```
nba-dashboard/
├── src/
│   ├── App.jsx          # Main app component
│   ├── index.jsx        # Entry point
│   └── styles.css       # Global styles
├── server/
│   └── index.js         # Backend API
├── package.json         # Dependencies
├── vite.config.js       # Vite config
└── index.html           # HTML template
```

---

## 🎯 Features

### **Navigation:**
- 📊 Overview - Live stats and games
- 🏀 NBA Model - Dual branch predictions
- 💰 Live Odds - BetOnline scraper
- 🎯 Bet Tracker - P&L and history

### **Real-time Updates:**
- Auto-refresh every 5 seconds
- Live game scores from NBA API
- Live odds from BetOnline
- Predictions at 18-minute mark

### **Dual Branch Predictions:**
- Branch A: Halftime (MAE ~6 pts)
- Branch B: Final (MAE ~10 pts)
- Confidence scoring
- Quality filtering

---

## 🔌 API Endpoints

```
GET  /api/nba/games        - Live NBA games
GET  /api/betonline/odds   - Current betting odds
GET  /api/predictions      - Model predictions
POST /api/predict/:gameId  - Trigger prediction
GET  /api/health           - Server status
```

---

## 🎨 UI Sections

### **Overview:**
- Live game count
- Predictions made today
- Active odds
- Bet opportunities

### **NBA Model:**
- All games listed
- Dual predictions (halftime + final)
- Confidence levels
- Similar games shown

### **Live Odds:**
- Current spreads (1H + FG)
- Totals
- Real-time from BetOnline
- Scraper status

### **Bet Tracker:**
- Bets placed
- P&L tracking
- Bankroll
- Win rate

---

## ⚙️ Configuration

Edit `server/index.js` to configure:
- Port (default: 5000)
- Refresh rate (default: 5s)
- Python paths
- API endpoints

---

## 🚀 Monday Launch

```bash
# Start both servers
npm run server  # Terminal 1
npm run dev     # Terminal 2

# Open in browser
open http://localhost:3000

# Click "NBA Model" to see predictions!
```

---

## 🎯 Opening Night Ready!

Built: October 18, 2025  
Launch: Monday, October 21, 2025, 4 PM PST  
Status: READY 🟢

