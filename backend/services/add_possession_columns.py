"""
Add possession tracking columns to existing schema
(Migration script - safe to run multiple times)
"""

import os
import psycopg2

DATABASE_URL = os.getenv('DATABASE_URL')
if not DATABASE_URL:
    print("❌ DATABASE_URL not set")
    exit(1)

print("="*80)
print("🔧 ADDING POSSESSION TRACKING COLUMNS")
print("="*80)
print()

conn = psycopg2.connect(DATABASE_URL)
cur = conn.cursor()

print("✅ Connected to PostgreSQL")
print()

# Add columns to player_box_scores
print("1️⃣  Adding per-100 columns to player_box_scores...")
try:
    cur.execute("""
        ALTER TABLE player_box_scores 
        ADD COLUMN IF NOT EXISTS pts_100 DECIMAL(7,2),
        ADD COLUMN IF NOT EXISTS reb_100 DECIMAL(7,2),
        ADD COLUMN IF NOT EXISTS ast_100 DECIMAL(7,2),
        ADD COLUMN IF NOT EXISTS stl_100 DECIMAL(7,2),
        ADD COLUMN IF NOT EXISTS blk_100 DECIMAL(7,2),
        ADD COLUMN IF NOT EXISTS tov_100 DECIMAL(7,2);
    """)
    conn.commit()
    print("   ✅ Added per-100 columns")
except Exception as e:
    print(f"   ⚠️  {e}")
    conn.rollback()

# Add column to player_season_stats
print("2️⃣  Adding total_team_possessions to player_season_stats...")
try:
    cur.execute("""
        ALTER TABLE player_season_stats
        ADD COLUMN IF NOT EXISTS total_team_possessions INT;
    """)
    conn.commit()
    print("   ✅ Added total_team_possessions")
except Exception as e:
    print(f"   ⚠️  {e}")
    conn.rollback()

# Verify
print()
print("🔍 Verification:")
cur.execute("""
    SELECT column_name 
    FROM information_schema.columns 
    WHERE table_name = 'player_box_scores' AND column_name LIKE '%_100'
    ORDER BY column_name
""")
per_100_cols = [row[0] for row in cur.fetchall()]
print(f"   player_box_scores per-100 columns: {per_100_cols}")

cur.execute("""
    SELECT column_name 
    FROM information_schema.columns 
    WHERE table_name = 'player_season_stats' AND column_name = 'total_team_possessions'
""")
if cur.fetchone():
    print(f"   player_season_stats.total_team_possessions: ✅ EXISTS")
else:
    print(f"   player_season_stats.total_team_possessions: ❌ MISSING")

conn.close()

print()
print("="*80)
print("✅ MIGRATION COMPLETE")
print("="*80)
print()
print("Next: python3 backend/services/populate_with_real_possessions.py")

