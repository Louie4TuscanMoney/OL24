"""
SCRAPE FULL SEASON OPENER - Oct 21-28, 2025 (All 53 Games)
Then run aggregation pipeline to populate all tables
"""

import os
import sys
from datetime import datetime, timedelta

# Set environment
if not os.environ.get('DATABASE_URL'):
    print("⚠️  DATABASE_URL not set")
    sys.exit(1)

print("\n" + "="*80)
print("🏀 SCRAPING FULL 2025-26 SEASON OPENER")
print("="*80)
print("Date Range: Oct 21-28, 2025")
print("Expected: ~53 games")
print("="*80 + "\n")

from basketball_reference_scraper import BasketballReferenceScraper

scraper = BasketballReferenceScraper()

# Scrape each day from Oct 21-28
start_date = datetime(2025, 10, 21)
end_date = datetime(2025, 10, 28)
total_games = 0

current_date = start_date
while current_date <= end_date:
    print(f"\n{'='*80}")
    print(f"📅 {current_date.strftime('%A, %B %d, %Y')}")
    print("="*80)
    
    games_count = scraper.scrape_daily_games(current_date)
    total_games += games_count
    
    print(f"✅ Scraped {games_count} games")
    print(f"📊 Total so far: {total_games} games")
    
    current_date += timedelta(days=1)

scraper.close()

print("\n" + "="*80)
print(f"✅ SCRAPED {total_games} GAMES FROM SEASON OPENER!")
print("="*80)

# Now run aggregation pipeline
print("\n" + "="*80)
print("📊 RUNNING AGGREGATION PIPELINE")
print("="*80 + "\n")

from nba_nightly_pipeline import NBANightlyPipeline

pipeline = NBANightlyPipeline()

# Override season to 2025-26
pipeline.current_season = '2025-26'

# Run only the aggregation steps (skip game scraping)
print("🧮 Computing season aggregates...")
pipeline.compute_season_aggregates()
print("✅ Season stats updated\n")

print("📊 Computing team metrics...")
pipeline.compute_team_metrics()
print("✅ Team stats updated\n")

# Don't run RAPM yet - need more games
# print("🧠 Computing RAPM + LEBRON...")
# pipeline.compute_rapm_and_lebron()
# print("✅ RAPM updated\n")

pipeline.close()

print("\n" + "="*80)
print("🎉 ALL TABLES POPULATED!")
print("="*80)
print("""
Check your database:
  SELECT COUNT(*) FROM player_box_scores;
  SELECT COUNT(*) FROM player_season_stats;
  SELECT * FROM player_season_stats ORDER BY ppg DESC LIMIT 10;
  SELECT COUNT(*) FROM team_season_stats;
""")

