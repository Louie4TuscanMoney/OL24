"""
POPULATE HISTORICAL DATA - Test the pipeline with real games
Fetches games from the 2024-25 season opener (Oct 22-29, 2024)
"""

import os
import sys

# Set the date range for historical data
os.environ['FORCE_DATE_FROM'] = '10/22/2024'
os.environ['FORCE_DATE_TO'] = '10/29/2024'

print(f"""
================================================================================
📊 POPULATING HISTORICAL DATA
================================================================================
Date Range: {os.environ['FORCE_DATE_FROM']} → {os.environ['FORCE_DATE_TO']}
Season: 2024-25 Opening Week
================================================================================
""")

# Import and run the pipeline
from nba_nightly_pipeline import NBANightlyPipeline

try:
    pipeline = NBANightlyPipeline()
    pipeline.run()
    
    print("""
================================================================================
✅ HISTORICAL DATA POPULATED!
================================================================================

🎯 Check your data:
   SELECT COUNT(*) FROM player_box_scores;
   SELECT COUNT(*) FROM player_season_stats;
   SELECT * FROM player_season_stats ORDER BY ppg DESC LIMIT 10;

""")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

