"""
Add advanced stat columns for Basketball Reference data
(PER, USG%, WS, OBPM, DBPM, VORP)
"""

import os
import psycopg2

DATABASE_URL = os.getenv('DATABASE_URL')
if not DATABASE_URL:
    print("❌ DATABASE_URL not set")
    exit(1)

print("="*80)
print("🔧 ADDING ADVANCED STAT COLUMNS")
print("="*80)
print()

conn = psycopg2.connect(DATABASE_URL)
cur = conn.cursor()

print("✅ Connected to PostgreSQL")
print()

# Add columns
print("📊 Adding advanced stat columns to player_season_stats...")
try:
    cur.execute("""
        ALTER TABLE player_season_stats 
        ADD COLUMN IF NOT EXISTS obpm DECIMAL(7,3),
        ADD COLUMN IF NOT EXISTS dbpm DECIMAL(7,3),
        ADD COLUMN IF NOT EXISTS vorp DECIMAL(7,3),
        ADD COLUMN IF NOT EXISTS per DECIMAL(7,2),
        ADD COLUMN IF NOT EXISTS usage_pct DECIMAL(5,3),
        ADD COLUMN IF NOT EXISTS win_shares DECIMAL(7,2),
        ADD COLUMN IF NOT EXISTS win_shares_48 DECIMAL(7,3);
    """)
    conn.commit()
    print("   ✅ Added: obpm, dbpm, vorp, per, usage_pct, win_shares, win_shares_48")
except Exception as e:
    print(f"   ⚠️  {e}")
    conn.rollback()

# Verify
print()
print("🔍 Verification:")
cur.execute("""
    SELECT column_name 
    FROM information_schema.columns 
    WHERE table_name = 'player_season_stats' 
    AND column_name IN ('obpm', 'dbpm', 'vorp', 'per', 'usage_pct', 'win_shares', 'win_shares_48')
    ORDER BY column_name
""")

added_cols = [row[0] for row in cur.fetchall()]
print(f"   Columns added: {added_cols}")
print(f"   Count: {len(added_cols)}/7")

conn.close()

print()
print("="*80)
print("✅ MIGRATION COMPLETE")
print("="*80)
print()
print("Next: python3 backend/services/scrape_basketball_reference_all.py")

