# Frontend Deployment Guide

## 🎯 Your New Components

### 1. Trading Dashboard (Full Page)
- **Location:** `/trading` route
- **Component:** `src/components/TradingPage.tsx`
- **Features:**
  - Interactive EV calculator
  - Kelly Criterion bet sizing
  - Custom odds input
  - Live opportunities from all games
  - Performance tracking

### 2. Mamba Live Widget (On Game Pages)
- **Location:** Integrated into `GameDetailPage.tsx`
- **Component:** `src/components/MambaLiveWidget.tsx`
- **Features:**
  - Real-time pattern visualization
  - WebSocket streaming
  - Q2 6:00 prediction display
  - Live confidence updates

## 🚀 Deploy to Production

### Option 1: Vercel (Recommended)
```bash
# Install Vercel CLI
npm i -g vercel

# Deploy
vercel --prod
```

### Option 2: Netlify
```bash
# Install Netlify CLI
npm i -g netlify-cli

# Build
npm run build

# Deploy
netlify deploy --prod --dir=dist
```

### Option 3: Railway (Same as Backend)
```bash
# Add to Railway project
railway add
railway up
```

## 📡 API Configuration

The frontend is already configured to use:
- **Production API:** `https://ol24-production.up.railway.app`
- **WebSocket:** `wss://ol24-production.up.railway.app/ws/mamba/{gameId}`

All API calls are routed through `src/services/tradingApi.ts`.

## 🧪 Test Locally

```bash
# Development server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

## 🎨 Customization

### Change API URL
Edit `.env`:
```env
VITE_BACKEND_URL=https://your-api.com
```

### Modify Colors
Components use Tailwind classes. Example:
```tsx
// Change from blue to green
class="bg-blue-600" → class="bg-green-600"
```

## 📊 What Each Route Shows

| Route | Purpose | Live Data |
|-------|---------|-----------|
| `/` | Live predictions | WebSocket games |
| `/trading` | Interactive trading | Mamba predictions + EV |
| `/game/{id}` | Game details + Mamba | Pattern visualization |
| `/stats` | Player/team stats | PostgreSQL data |
| `/schedule` | Upcoming games | Schedule + injuries |
| `/teams` | Team directory | Rosters + depth charts |

## ✅ Verify Deployment

After deploying, test these URLs:
1. `https://your-site.com/` - Should show live games
2. `https://your-site.com/trading` - Should show trading dashboard
3. `https://your-site.com/game/0042400101` - Should show Mamba widget

## 🔥 Next Game

When the next NBA game starts:
1. Go to `/trading` to see live opportunities
2. Click any game to see Mamba patterns
3. At Q2 6:00, the prediction will trigger automatically
4. Enter custom odds to calculate EV and Kelly stake

**Everything is connected and ready!** 🚀
