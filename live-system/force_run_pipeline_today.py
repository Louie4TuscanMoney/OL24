"""
FORCE RUN PIPELINE TODAY - Don't wait for 3:30 AM!
Populates database immediately for testing
"""

import os
import sys
from datetime import datetime

# Set environment variable for local testing
if not os.environ.get('DATABASE_URL'):
    print("⚠️  DATABASE_URL not set")
    print("Set it with: export DATABASE_URL='postgresql://...'")
    sys.exit(1)

print("\n" + "="*80)
print("🔥 FORCE RUNNING NIGHTLY PIPELINE NOW")
print("="*80)
print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Mode: IMMEDIATE (not waiting for 3:30 AM)")
print("="*80 + "\n")

# Import and run pipeline
try:
    from nba_nightly_pipeline import NBANightlyPipeline
    
    pipeline = NBANightlyPipeline()
    
    # RUN NOW (don't wait for schedule)
    pipeline.run()
    
    # Verify data IMMEDIATELY before closing
    import psycopg2
    cursor = pipeline.cursor
    cursor.execute("SELECT COUNT(*) FROM player_box_scores")
    box_count = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM players")
    player_count = cursor.fetchone()[0]
    
    pipeline.close()
    
    print("\n" + "="*80)
    print("✅ FORCE RUN COMPLETED!")
    print("="*80)
    print(f"\n📊 Immediate verification (before close):")
    print(f"   Box scores: {box_count}")
    print(f"   Players: {player_count}")
    print("\n🎯 Check database:")
    print("   psql $DATABASE_URL")
    print("   SELECT COUNT(*) FROM player_box_scores;")
    print("   SELECT COUNT(*) FROM player_season_stats;")
    print("   SELECT * FROM player_season_stats ORDER BY ppg DESC LIMIT 10;")
    print("\n")
    
except Exception as e:
    print(f"\n❌ FAILED: {e}")
    import traceback
    traceback.print_exc()

