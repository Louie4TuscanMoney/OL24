"""
NIGHTLY BASKETBALL REFERENCE SCRAPER
Runs at 3:30 AM UTC on Railway
Scrapes yesterday's games and updates all stats
"""

import os
import sys
from datetime import datetime, timedelta
import psycopg2

DATABASE_URL = os.environ.get('DATABASE_URL')

def main():
    """
    Main nightly job:
    1. Scrape yesterday's games from Basketball Reference
    2. Aggregate stats
    3. Update rolling windows
    4. Compute advanced metrics
    """
    print("\n" + "="*80)
    print(f"🌙 NIGHTLY JOB: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("="*80 + "\n")
    
    from basketball_reference_scraper import BasketballReferenceScraper
    from nba_nightly_pipeline import NBANightlyPipeline
    
    # STEP 1: Scrape yesterday's games
    print("📥 STEP 1: Scraping yesterday's games from Basketball Reference...")
    scraper = BasketballReferenceScraper()
    
    yesterday = datetime.now() - timedelta(days=1)
    games_count = scraper.scrape_daily_games(yesterday)
    
    scraper.close()
    
    print(f"✅ Scraped {games_count} games\n")
    
    if games_count == 0:
        print("ℹ️  No games yesterday - season may not be active")
        return
    
    # STEP 2: Run aggregation pipeline
    print("="*80)
    print("📊 STEP 2: Aggregating stats and updating tables")
    print("="*80 + "\n")
    
    pipeline = NBANightlyPipeline()
    pipeline.current_season = '2025-26'
    
    # Only run aggregation steps (no scraping)
    print("🧮 Computing season aggregates...")
    pipeline.compute_season_aggregates()
    print("✅ Season stats updated\n")
    
    print("📊 Computing team metrics...")
    pipeline.compute_team_metrics()
    print("✅ Team stats updated\n")
    
    print("🔄 Refreshing last 10 games view...")
    pipeline.prune_and_refresh_last10()
    print("✅ Last 10 refreshed\n")
    
    # Only compute RAPM once per week (Sunday)
    if datetime.now().weekday() == 6:  # Sunday
        print("🧠 Computing RAPM + LEBRON (weekly)...")
        pipeline.compute_rapm_and_lebron()
        print("✅ RAPM updated\n")
    
    pipeline.close()
    
    # STEP 3: Create snapshot
    print("="*80)
    print("📸 STEP 3: Creating daily snapshot")
    print("="*80 + "\n")
    
    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()
    
    today = datetime.now().date()
    
    cursor.execute("SELECT COUNT(*) FROM player_box_scores WHERE game_date = %s", (yesterday.date(),))
    box_scores_today = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM player_season_stats WHERE season_id = '2025-26'")
    total_players = cursor.fetchone()[0]
    
    cursor.execute("""
        INSERT INTO daily_snapshots (
            snapshot_date, games_processed, players_updated, status, finished_at, duration_seconds
        ) VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (snapshot_date) DO UPDATE SET
            games_processed = EXCLUDED.games_processed,
            players_updated = EXCLUDED.players_updated,
            status = EXCLUDED.status
    """, (today, games_count, total_players, 'success', datetime.now(), 0))
    
    conn.commit()
    conn.close()
    
    print(f"✅ Snapshot created")
    print(f"   Games: {games_count}")
    print(f"   Players updated: {total_players}\n")
    
    print("="*80)
    print("✅ NIGHTLY JOB COMPLETE!")
    print("="*80 + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ NIGHTLY JOB FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

