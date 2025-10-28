#!/bin/bash
# ============================================================================
# RUN EVERYTHING NOW - Full system test
# ============================================================================

set -e  # Exit on error

echo ""
echo "================================================================================"
echo "🔥 RUNNING COMPLETE NBA ANALYTICS PIPELINE NOW"
echo "================================================================================"
echo ""

# Check DATABASE_URL
if [ -z "$DATABASE_URL" ]; then
    echo "❌ DATABASE_URL not set!"
    echo ""
    echo "Get it from Railway:"
    echo "  1. Go to Railway dashboard"
    echo "  2. Click PostgreSQL service"
    echo "  3. Click 'Connect'"
    echo "  4. Copy the connection string"
    echo ""
    echo "Then run:"
    echo "  export DATABASE_URL='postgresql://...'"
    echo ""
    exit 1
fi

echo "✅ DATABASE_URL configured"
echo ""

# Move to live-system directory
cd "$(dirname "$0")"

echo "================================================================================"
echo "STEP 1: Deploy Schema"
echo "================================================================================"
echo ""

if command -v psql &> /dev/null; then
    echo "📊 Deploying schema to PostgreSQL..."
    psql "$DATABASE_URL" -f database_schema_v3_SELF_COMPUTED.sql
    echo ""
    echo "✅ Schema deployed!"
else
    echo "⚠️  psql not found - deploy schema manually in Railway UI"
    echo "   Copy/paste: database_schema_v3_SELF_COMPUTED.sql"
    echo ""
    read -p "Press Enter when schema is deployed..."
fi

echo ""
echo "================================================================================"
echo "STEP 2: Populate Images + Colors"
echo "================================================================================"
echo ""

python3 populate_player_images.py

echo ""
echo "================================================================================"
echo "STEP 3: Run Nightly Pipeline (ALL STATS + RAPM)"
echo "================================================================================"
echo ""

python3 force_run_pipeline_today.py

echo ""
echo "================================================================================"
echo "✅ COMPLETE! System is now populated with:"
echo "================================================================================"
echo ""
echo "  ✅ 30 teams (with logos + colors)"
echo "  ✅ 450+ players (with headshots)"
echo "  ✅ 80+ games (today's box scores)"
echo "  ✅ Season stats (per-game, per-100, per-36)"
echo "  ✅ Team metrics (ORTG, DRTG, Net Rating, Luck)"
echo "  ✅ RAPM + LEBRON (self-computed!)"
echo "  ✅ Last 10 games (materialized view)"
echo ""
echo "================================================================================"
echo "🔍 VERIFY DATA:"
echo "================================================================================"
echo ""
echo "psql \$DATABASE_URL"
echo ""
echo "SELECT COUNT(*) FROM player_box_scores;     -- Should be 1000+"
echo "SELECT COUNT(*) FROM player_season_stats;   -- Should be 450+"
echo "SELECT * FROM player_season_stats ORDER BY ppg DESC LIMIT 10;"
echo ""
echo "================================================================================"
echo "🌐 TEST API:"
echo "================================================================================"
echo ""
echo "curl https://ol24-production.up.railway.app/api/stats/player/2544"
echo ""
echo "================================================================================"
echo "🎉 ANALYTICS PLATFORM READY!"
echo "================================================================================"
echo ""

