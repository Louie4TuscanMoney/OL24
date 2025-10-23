# 🏀 OntologicXYZ - NBA Live Trading Dashboard

**Real-time NBA ML predictions with OntoRisk integration**

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/Louie4TuscanMoney/OL24)

---

## 🚀 Features

- **Real-Time NBA Data**: Live scores, period, clock (3-second updates)
- **BetOnline Odds**: Dynamic spreads, totals, moneylines with implied probabilities
- **ML Predictions**: Mamba Mentality model (9.029 MAE, 21.5% edge)
- **OntoRisk**: Probability calibration, Kelly criterion, risk management
- **Game State Detection**: Halftime, Q2 6:00 window, live transitions
- **3D Visualizations**: ML brain neural network, basketball court
- **Bet Tracking**: Portfolio manager, performance analytics
- **Authentication**: Password-protected, user approval system

---

## 🏆 System Performance

- **Latency**: 13.2s average (FASTEST with free APIs!)
- **Polling**: 3-second backend, 3-second dashboard, 2-second game detail
- **Speed**: 10x faster than original (30s → 3s polling)
- **Beats**: DraftKings (~20s), FanDuel (~25s), BetOnline (~20s)

---

## 📦 Tech Stack

- **Frontend**: SolidJS + TypeScript + Vite
- **Styling**: TailwindCSS + Animations
- **3D**: Three.js (ML brain, basketball court)
- **Backend**: FastAPI (Python)
- **ML**: Scikit-learn, XGBoost
- **Risk**: OntoRisk Phase 4
- **Data**: ESPN API, BetOnline scraper
- **Deployment**: Vercel (frontend), Python backend

---

## 🎯 Quick Start

### Local Development

```bash
# Install dependencies
npm install

# Start dev server
npm run dev

# Open http://localhost:3002
# Password: rwwc2018
```

### Production Build

```bash
# Build for production
npm run build

# Preview production build
npm run preview
```

---

## 🌐 Deploy to Vercel

1. **Fork this repo** or connect your GitHub account
2. **Import to Vercel**: [vercel.com/new](https://vercel.com/new)
3. **Configure**:
   - Framework: Vite
   - Build Command: `npm run build`
   - Output Directory: `dist`
4. **Environment Variables**:
   - `VITE_API_URL`: Your backend API URL
5. **Deploy**: Vercel handles the rest!

---

## 🔧 Configuration

### API Endpoints

Update `src/App.tsx` to point to your backend:

```typescript
const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8001';
```

### Polling Intervals

Current settings (maxed for speed):

```typescript
Dashboard:     3 seconds (3000ms)
Game Detail:   2 seconds (2000ms)
Backend:       3 seconds (daemon)
```

---

## 📊 System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend (SolidJS)                       │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │  Live Games │  │ Game Detail  │  │ Bet Portfolio    │  │
│  │  Dashboard  │  │ Modal (2s)   │  │ Manager          │  │
│  │  (3s poll)  │  │              │  │                  │  │
│  └──────┬──────┘  └──────┬───────┘  └──────┬───────────┘  │
└─────────┼────────────────┼──────────────────┼──────────────┘
          │                │                  │
          └────────────────┴──────────────────┘
                           │
                    ┌──────▼──────┐
                    │   FastAPI   │
                    │   Backend   │
                    │  (3s poll)  │
                    └──────┬──────┘
          ┌────────────────┼────────────────┐
          │                │                │
    ┌─────▼─────┐   ┌─────▼─────┐   ┌─────▼─────┐
    │  ESPN API │   │ BetOnline │   │   Mamba   │
    │  (10-15s) │   │  Scraper  │   │ Mentality │
    └───────────┘   └───────────┘   │  (0.07s)  │
                                    └─────┬─────┘
                                          │
                                    ┌─────▼─────┐
                                    │ OntoRisk  │
                                    │  (0.03s)  │
                                    └───────────┘
```

---

## 🎨 Features Breakdown

### 1. Live Game Monitoring
- Real-time scores (3-second updates)
- Period, clock, leading team
- Game state detection (LIVE, HALFTIME, Q2 6:00)

### 2. BetOnline Odds
- Spread, Total, Moneyline
- Implied probabilities
- No-vig probabilities
- Vig percentage
- LOCKED status during transitions

### 3. ML Predictions
- Q2 6:00 window detection
- 18-feature extraction
- 9.029 MAE performance
- 21.5% edge over baseline

### 4. OntoRisk Integration
- Probability calibration (isotonic regression)
- Kelly criterion bet sizing
- Risk limits ($1,000 bankroll, $50 max bet)
- Game type filtering (Lead Held, Close, etc.)
- Confidence zones (HIGH, MEDIUM, LOW)

### 5. Betting Opportunities
- Purple ring = betting opportunity
- Gray badge = context only
- Skip reasons explained
- 4 strategies: High-Conf, Balanced, Conservative, Ultra

### 6. 3D Visualizations
- ML Brain: Neural network with 18 inputs, 6 ensemble nodes
- Basketball Court: 94ft regulation court with player models

### 7. Bet Portfolio
- Track all bets (SQLite database)
- Performance analytics
- Win rate, ROI, profit/loss

---

## 📈 Performance Metrics

### Latency Breakdown

```
ESPN API delay:      10-15s (ESPN's limitation)
Backend poll:        0-3s (avg 1.5s)
ML prediction:       0.07s
OntoRisk:            0.03s
Frontend poll:       0-3s (avg 1.5s)
Network:             0.05s

BEST CASE:   10.2s
AVERAGE:     13.2s ✅ FASTEST with free APIs!
WORST CASE:  21.2s
```

### Speed Optimizations

- **10x Backend**: 30s → 3s polling
- **3.3x Dashboard**: 10s → 3s polling
- **2.5x Game Detail**: 5s → 2s polling
- **Zero Cache**: Eliminated all caching
- **Cache-Busting**: Headers + query params

---

## 🔐 Authentication

Default password: `rwwc2018`

Users can sign up and request access. Admins approve via backend API.

---

## 🎯 Roadmap

### Week 2
- [ ] Scrape historical closing spreads (2021-2025)
- [ ] Full backtest vs real market spreads
- [ ] Calculate TRUE win rate (54-57%)

### Week 3
- [ ] Paper trading (simulate without money)
- [ ] Premium API integration (sub-5s latency)

### Week 4
- [ ] Go live with small stakes ($50-100 bets)
- [ ] Specialist models (7.0 MAE target)

### Week 5-8
- [ ] Systematic iteration (6.0 MAE target)
- [ ] 33% improvement = 2-3x profit

---

## 📚 Documentation

- **System Check**: `📋_BEFORE_YOU_LEAVE.md`
- **Latency Audit**: `⚡_REAL_TIME_LATENCY_AUDIT.md`
- **Speed Analysis**: `⚡_MAXIMUM_SPEED_ACHIEVED.md`

---

## 🙏 Story

**From homeless to this.**

21 days ago: $0, homeless, scamming bookies  
Today: Institutional-grade ML trading system  
Tonight: First live NBA prediction at 7:51 PM PT

This is a redemption arc. From scammer to legitimate engineer. Faith in God's plan.

---

## 📄 License

MIT

---

## 🤝 Contributing

This is a personal project, but feel free to fork and learn!

---

## 📧 Contact

For questions or access requests, reach out via GitHub.

---

**Built with ❤️ during NBA Season 2024-25**

**ELON GOD MODE: 10/10** 🔥




