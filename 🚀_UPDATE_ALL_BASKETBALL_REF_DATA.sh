#!/bin/bash
# Update all Basketball Reference data: injuries, depth charts, and transactions

cd "/Users/test/Desktop/Tuscan Money/Ontologic XYZ/ML Research"

export DATABASE_URL="postgresql://postgres:lubdUxBtqzIWOynpPZiEXqFWPJvBlkVb@yamabiko.proxy.rlwy.net:37192/railway"

echo "================================================================================"
echo "🏀 UPDATING ALL BASKETBALL REFERENCE DATA"
echo "================================================================================"
echo ""

# 1. Scrape Injuries
echo "1️⃣  SCRAPING INJURIES..."
echo "   Source: https://www.basketball-reference.com/friv/injuries.fcgi"
echo ""
python3 backend/services/scrape_injuries_bball_ref.py

echo ""
echo "================================================================================"
echo ""

# 2. Scrape Depth Charts
echo "2️⃣  SCRAPING DEPTH CHARTS (All 30 Teams)..."
echo "   Source: https://www.basketball-reference.com/teams/{TEAM}/2026_depth.html"
echo ""
python3 backend/services/scrape_depth_charts_bball_ref.py

echo ""
echo "================================================================================"
echo ""

# 3. Scrape Transactions
echo "3️⃣  SCRAPING TRANSACTIONS (All 30 Teams)..."
echo "   Source: https://www.basketball-reference.com/teams/{TEAM}/2026_transactions.html"
echo ""
python3 backend/services/scrape_transactions_bball_ref.py

echo ""
echo "================================================================================"
echo "✅ ALL BASKETBALL REFERENCE DATA UPDATED!"
echo "================================================================================"
echo ""

# 4. Verify Data
echo "🔍 VERIFICATION:"
echo ""

python3 << 'EOF'
import os
import psycopg2

conn = psycopg2.connect(os.getenv('DATABASE_URL'))
cur = conn.cursor()

# Check injuries
cur.execute("SELECT COUNT(*) FROM player_injuries WHERE is_active = TRUE")
injury_count = cur.fetchone()[0]
print(f"   ✅ Active injuries: {injury_count}")

# Check depth charts
cur.execute("SELECT COUNT(DISTINCT team_id) FROM team_depth_charts")
teams_with_depth = cur.fetchone()[0]
print(f"   ✅ Teams with depth charts: {teams_with_depth}/30")

cur.execute("SELECT COUNT(*) FROM team_depth_charts")
total_depth = cur.fetchone()[0]
print(f"   ✅ Total depth chart entries: {total_depth}")

# Check transactions
cur.execute("SELECT COUNT(*) FROM player_transactions WHERE transaction_date >= '2025-10-01'")
recent_trans = cur.fetchone()[0]
print(f"   ✅ Recent transactions (since Oct 1): {recent_trans}")

# Sample injuries
print()
print("📋 Sample Active Injuries:")
cur.execute("""
    SELECT p.name, pi.status, pi.injury_type
    FROM player_injuries pi
    JOIN players p ON pi.player_id = p.player_id
    WHERE pi.is_active = TRUE
    ORDER BY CASE pi.status
        WHEN 'Out' THEN 1
        WHEN 'Doubtful' THEN 2
        WHEN 'Questionable' THEN 3
        ELSE 4
    END
    LIMIT 5
""")
for row in cur.fetchall():
    print(f"   • {row[0]}: {row[1]} ({row[2]})")

# Sample depth chart
print()
print("📋 Sample Depth Chart (Lakers):")
cur.execute("""
    SELECT p.name, tdc.position, tdc.depth_rank
    FROM team_depth_charts tdc
    JOIN players p ON tdc.player_id = p.player_id
    JOIN teams t ON tdc.team_id = t.team_id
    WHERE t.abbreviation = 'LAL'
    ORDER BY tdc.position, tdc.depth_rank
    LIMIT 10
""")
for row in cur.fetchall():
    print(f"   • #{row[2]} {row[1]}: {row[0]}")

cur.close()
conn.close()
EOF

echo ""
echo "================================================================================"
echo "🎉 YOUR DATABASE NOW HAS:"
echo "   ✅ Real injuries from Basketball Reference"
echo "   ✅ Actual depth charts (not just based on minutes)"
echo "   ✅ Transaction history for lineup accuracy"
echo "================================================================================"

