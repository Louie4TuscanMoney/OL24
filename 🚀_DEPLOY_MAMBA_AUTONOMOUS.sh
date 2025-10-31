#!/bin/bash

echo "========================================================================"
echo "🤖 DEPLOYING MAMBA AUTONOMOUS SYSTEM"
echo "========================================================================"
echo ""

cd "/Users/test/Desktop/Tuscan Money/Ontologic XYZ/ML Research/live-system"

export DATABASE_URL="postgresql://postgres:lubdUxBtqzIWOynpPZiEXqFWPJvBlkVb@yamabiko.proxy.rlwy.net:37192/railway"

echo "Step 1: Deploy Play-by-Play Schema"
echo "======================================================================"
echo ""

psql "$DATABASE_URL" -f play_by_play_schema.sql

echo ""
echo "Step 2: Verify Schema"
echo "======================================================================"
echo ""

python3 << 'EOF'
import os, psycopg2

conn = psycopg2.connect(os.getenv('DATABASE_URL'))
cur = conn.cursor()

# Check tables
cur.execute("""
    SELECT table_name 
    FROM information_schema.tables 
    WHERE table_schema = 'public' 
    AND table_name IN ('play_by_play', 'mamba_game_cache')
    ORDER BY table_name
""")

tables = [row[0] for row in cur.fetchall()]

print("✅ Schema Verification:")
print(f"   • play_by_play: {'✅' if 'play_by_play' in tables else '❌'}")
print(f"   • mamba_game_cache: {'✅' if 'mamba_game_cache' in tables else '❌'}")

cur.close()
conn.close()
EOF

echo ""
echo "Step 3: Test Cron Script Locally"
echo "======================================================================"
echo ""

python3 cron_mamba_autonomous.py

echo ""
echo "Step 4: Add to Git"
echo "======================================================================"
echo ""

git add play_by_play_schema.sql cron_mamba_autonomous.py
git status

echo ""
echo "Step 5: Commit"
echo "======================================================================"
echo ""

git commit -m "Add Mamba autonomous system with play-by-play tracking"

echo ""
echo "Step 6: Push to Railway"
echo "======================================================================"
echo ""

git push origin main

echo ""
echo "========================================================================"
echo "✅ DEPLOYMENT COMPLETE!"
echo "========================================================================"
echo ""
echo "📝 Next Steps:"
echo ""
echo "1. Add Cron Schedule to Railway Dashboard:"
echo "   • Go to: railway.app/project/ol24/settings"
echo "   • Add Cron Job:"
echo "     - Schedule: */30 * * * * * (every 30 seconds)"
echo "     - Command: python cron_mamba_autonomous.py"
echo ""
echo "2. Verify During Live Game:"
echo "   • Wait for Q2 6:00 during any live NBA game"
echo "   • Check logs: railway logs"
echo "   • Check database: SELECT * FROM mamba_game_cache"
echo ""
echo "3. Test API Endpoint:"
echo "   • GET /api/mamba/{game_id}"
echo "   • Should return prediction after Q2 6:00"
echo ""
echo "========================================================================"

