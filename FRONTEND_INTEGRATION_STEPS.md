# 🎨 FRONTEND INTEGRATION - QUICK START

## Add Components to Your Frontend

### 1. Copy Component Files

```bash
# Copy to your frontend/src/components/
cp frontend_trading_dashboard.tsx your-frontend/src/components/TradingDashboard.tsx
cp frontend_mamba_live_component.tsx your-frontend/src/components/MambaLiveWidget.tsx
```

### 2. Install Dependencies

```bash
cd your-frontend
npm install solid-chartjs chart.js
```

### 3. Add to Your App

#### Option A: Full Trading Dashboard Page

```tsx
// pages/trading.tsx or routes/trading.tsx
import { TradingDashboard } from '@/components/TradingDashboard';

export default function TradingPage() {
  return <TradingDashboard />;
}
```

#### Option B: Add to Game Pages

```tsx
// pages/game/[id].tsx
import { MambaLiveWidget } from '@/components/MambaLiveWidget';

export default function GamePage({ params }) {
  return (
    <div>
      <h1>Game Details</h1>
      
      {/* Your existing game info */}
      
      {/* Add Mamba Live Widget */}
      <MambaLiveWidget gameId={params.id} />
      
      {/* Rest of game details */}
    </div>
  );
}
```

### 4. Update API Base URL

In both components, update the API URL:

```tsx
// Change from:
fetch('/api/trading/live-opportunities')

// To:
fetch('https://ol24-production.up.railway.app/api/trading/live-opportunities')
```

### 5. WebSocket URL

Update WebSocket connection:

```tsx
// Change from:
const wsUrl = `wss://ol24-production.up.railway.app/ws/mamba/${props.gameId}`;

// Already correct! ✅
```

## That's It! 🎉

Your trading dashboard is now integrated and will show:
- ✅ Live Mamba predictions
- ✅ Interactive EV calculator
- ✅ Real-time pattern visualization
- ✅ Bet tracking and P&L
- ✅ Performance metrics
