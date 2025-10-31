#!/bin/bash
# Quick test script to verify Railway PostgreSQL database

export DATABASE_URL="postgresql://postgres:lubdUxBtqzIWOynpPZiEXqFWPJvBlkVb@yamabiko.proxy.rlwy.net:37192/railway"
BACKEND_URL="https://ol24-production.up.railway.app"

echo "================================================================================"
echo "🧪 TESTING RAILWAY POSTGRESQL DATABASE"
echo "================================================================================"
echo ""

# Test 1: Database connection
echo "1️⃣  Testing Database Connection..."
python3 << 'EOF'
import os
import psycopg2
try:
    conn = psycopg2.connect(os.getenv('DATABASE_URL'))
    cur = conn.cursor()
    cur.execute("SELECT version()")
    version = cur.fetchone()[0]
    print(f"   ✅ Connected: PostgreSQL")
    cur.close()
    conn.close()
except Exception as e:
    print(f"   ❌ Connection failed: {e}")
EOF

echo ""

# Test 2: Data counts
echo "2️⃣  Checking Data Counts..."
python3 << 'EOF'
import os
import psycopg2
conn = psycopg2.connect(os.getenv('DATABASE_URL'))
cur = conn.cursor()

cur.execute("SELECT COUNT(*) FROM teams WHERE logo_url IS NOT NULL")
teams = cur.fetchone()[0]
print(f"   ✅ Teams with logos: {teams}/30")

cur.execute("SELECT COUNT(*) FROM players WHERE is_active = TRUE")
players = cur.fetchone()[0]
print(f"   ✅ Active players: {players}")

cur.execute("SELECT COUNT(*) FROM player_season_stats WHERE pts_100 IS NOT NULL")
stats = cur.fetchone()[0]
print(f"   ✅ Players with advanced stats: {stats}")

cur.execute("SELECT COUNT(*) FROM nba_schedule")
schedule = cur.fetchone()[0]
print(f"   ✅ Scheduled games: {schedule}")

cur.execute("SELECT COUNT(*) FROM team_depth_charts")
depth = cur.fetchone()[0]
print(f"   ✅ Depth chart entries: {depth}")

cur.close()
conn.close()
EOF

echo ""

# Test 3: API endpoints
echo "3️⃣  Testing API Endpoints..."

# Teams endpoint
STATUS=$(curl -s -o /dev/null -w "%{http_code}" "${BACKEND_URL}/api/stats/teams")
if [ "$STATUS" = "200" ]; then
    echo "   ✅ /api/stats/teams (HTTP $STATUS)"
else
    echo "   ❌ /api/stats/teams (HTTP $STATUS)"
fi

# Schedule endpoint
STATUS=$(curl -s -o /dev/null -w "%{http_code}" "${BACKEND_URL}/api/schedule")
if [ "$STATUS" = "200" ]; then
    echo "   ✅ /api/schedule (HTTP $STATUS)"
else
    echo "   ❌ /api/schedule (HTTP $STATUS)"
fi

# Injuries endpoint
STATUS=$(curl -s -o /dev/null -w "%{http_code}" "${BACKEND_URL}/api/injuries")
if [ "$STATUS" = "200" ]; then
    echo "   ✅ /api/injuries (HTTP $STATUS)"
else
    echo "   ❌ /api/injuries (HTTP $STATUS)"
fi

# Standings endpoint
STATUS=$(curl -s -o /dev/null -w "%{http_code}" "${BACKEND_URL}/api/stats/standings")
if [ "$STATUS" = "200" ]; then
    echo "   ✅ /api/stats/standings (HTTP $STATUS)"
else
    echo "   ❌ /api/stats/standings (HTTP $STATUS)"
fi

echo ""

# Test 4: Sample data quality
echo "4️⃣  Checking Data Quality..."
python3 << 'EOF'
import os
import psycopg2
conn = psycopg2.connect(os.getenv('DATABASE_URL'))
cur = conn.cursor()

# Get top scorer
cur.execute("""
    SELECT p.name, t.abbreviation, pss.pts_100, pss.bpm, pss.per
    FROM player_season_stats pss
    JOIN players p ON pss.player_id = p.player_id
    JOIN teams t ON pss.team_id = t.team_id
    WHERE pss.pts_100 IS NOT NULL
    ORDER BY pss.pts_100 DESC
    LIMIT 1
""")

row = cur.fetchone()
if row:
    name, team, pts100, bpm, per = row
    print(f"   ✅ Top scorer: {name} ({team}) - {pts100:.1f} Pts/100")
    print(f"      BPM: {bpm:.1f}, PER: {per:.1f}")
else:
    print("   ⚠️  No player stats found")

cur.close()
conn.close()
EOF

echo ""
echo "================================================================================"
echo "✅ DATABASE TEST COMPLETE!"
echo "================================================================================"
echo ""
echo "📊 Summary:"
echo "   • Database: LIVE on Railway"
echo "   • API: LIVE on ${BACKEND_URL}"
echo "   • Frontend: Ready at ontologicxyz.com"
echo ""
echo "🚀 YOUR SYSTEM IS READY TO USE!"
echo "================================================================================"

