#!/bin/bash

echo "========================================================================"
echo "🚀 DEPLOYING COMPLETE MAMBA TRADING SYSTEM"
echo "========================================================================"
echo ""

cd "/Users/test/Desktop/Tuscan Money/Ontologic XYZ/ML Research/live-system"

export DATABASE_URL="postgresql://postgres:lubdUxBtqzIWOynpPZiEXqFWPJvBlkVb@yamabiko.proxy.rlwy.net:37192/railway"

echo "Step 1: Deploy Database Schemas"
echo "======================================================================"
echo ""

# Deploy play-by-play schema
echo "Deploying play-by-play schema..."
psql "$DATABASE_URL" -f play_by_play_schema.sql

echo ""
echo "Deploying tracked_bets schema..."
psql "$DATABASE_URL" -f tracked_bets_schema.sql

echo ""
echo "Step 2: Verify Database"
echo "======================================================================"
echo ""

python3 << 'EOF'
import os, psycopg2

conn = psycopg2.connect(os.getenv('DATABASE_URL'))
cur = conn.cursor()

# Check all tables
cur.execute("""
    SELECT table_name 
    FROM information_schema.tables 
    WHERE table_schema = 'public' 
    AND table_name IN ('play_by_play', 'mamba_game_cache', 'tracked_bets')
    ORDER BY table_name
""")

tables = {row[0] for row in cur.fetchall()}

print("✅ Database Verification:")
print(f"   • play_by_play: {'✅' if 'play_by_play' in tables else '❌'}")
print(f"   • mamba_game_cache: {'✅' if 'mamba_game_cache' in tables else '❌'}")
print(f"   • tracked_bets: {'✅' if 'tracked_bets' in tables else '❌'}")

cur.close()
conn.close()
EOF

echo ""
echo "Step 3: Add Trading Dashboard to API"
echo "======================================================================"
echo ""

# Add import to trading_dashboard_api.py
echo "Adding trading dashboard routes..."

python3 << 'EOF'
import os

api_file = 'trading_dashboard_api.py'

# Check if already added
with open(api_file, 'r') as f:
    content = f.read()
    
if 'from trading_dashboard_live import router as trading_router' not in content:
    print("Adding trading dashboard import...")
    
    # Find where to add import (after other imports)
    lines = content.split('\n')
    import_index = 0
    for i, line in enumerate(lines):
        if line.startswith('from') or line.startswith('import'):
            import_index = i
    
    # Add import
    lines.insert(import_index + 1, 'from trading_dashboard_live import router as trading_router')
    
    # Find where to add router (after app = FastAPI())
    for i, line in enumerate(lines):
        if 'app = FastAPI()' in line:
            lines.insert(i + 1, 'app.include_router(trading_router)')
            break
    
    # Write back
    with open(api_file, 'w') as f:
        f.write('\n'.join(lines))
    
    print("✅ Trading dashboard routes added")
else:
    print("✅ Trading dashboard already integrated")
EOF

# Add WebSocket handler
python3 << 'EOF'
import os

api_file = 'trading_dashboard_api.py'

with open(api_file, 'r') as f:
    content = f.read()

if 'ws/mamba' not in content:
    print("Adding Mamba WebSocket endpoint...")
    
    # Add at end of file
    websocket_code = '''

# Mamba Live WebSocket
from mamba_live_websocket import mamba_websocket_handler

@app.websocket("/ws/mamba/{game_id}")
async def websocket_mamba_endpoint(websocket: WebSocket, game_id: str):
    """Real-time Mamba updates for a specific game"""
    await mamba_websocket_handler(websocket, game_id)
'''
    
    with open(api_file, 'a') as f:
        f.write(websocket_code)
    
    print("✅ WebSocket endpoint added")
else:
    print("✅ WebSocket already integrated")
EOF

echo ""
echo "Step 4: Test Cron Script"
echo "======================================================================"
echo ""

echo "Running cron test..."
python3 cron_mamba_autonomous.py

echo ""
echo "Step 5: Git Commit"
echo "======================================================================"
echo ""

git add .
git status

echo ""
read -p "Commit and deploy? (y/n): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    git commit -m "Deploy complete Mamba trading system: live predictions + trading dashboard + performance tracking"
    
    echo ""
    echo "Step 6: Push to Railway"
    echo "======================================================================"
    echo ""
    
    git push origin main
    
    echo ""
    echo "======================================================================"
    echo "✅ DEPLOYMENT COMPLETE!"
    echo "======================================================================"
    echo ""
    echo "📝 Next Steps:"
    echo ""
    echo "1. Wait 1-2 minutes for Railway to deploy"
    echo ""
    echo "2. Verify API endpoints:"
    echo "   curl https://ol24-production.up.railway.app/api/trading/live-opportunities"
    echo "   curl https://ol24-production.up.railway.app/api/trading/performance"
    echo ""
    echo "3. Test WebSocket:"
    echo "   wss://ol24-production.up.railway.app/ws/mamba/{game_id}"
    echo ""
    echo "4. Add cron schedule in Railway Dashboard:"
    echo "   Schedule: */30 * * * * *"
    echo "   Command: python cron_mamba_autonomous.py"
    echo ""
    echo "5. Add Trading Dashboard to frontend:"
    echo "   <TradingDashboard />"
    echo ""
    echo "======================================================================"
    echo ""
    echo "🎯 SYSTEM FEATURES:"
    echo "======================================================================"
    echo ""
    echo "✅ Live play-by-play data collection"
    echo "✅ Automatic Mamba predictions at Q2 6:00 for ALL games"
    echo "✅ Real-time WebSocket streaming"
    echo "✅ Interactive trading dashboard"
    echo "✅ EV calculator with custom odds"
    echo "✅ Kelly Criterion bet sizing"
    echo "✅ Automatic bet tracking"
    echo "✅ Performance metrics & P&L"
    echo "✅ 2H result tracking"
    echo "✅ Complete historical record"
    echo ""
    echo "======================================================================"
else
    echo "Deployment cancelled"
fi

