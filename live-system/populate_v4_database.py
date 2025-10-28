#!/usr/bin/env python3
"""
Quick V4 Database Population
Populates the newly migrated V4 schema with NBA data
"""

import os
import psycopg2
from datetime import datetime, date, timedelta

DATABASE_URL = os.environ.get('DATABASE_URL')

def populate_database():
    print("=" * 80)
    print("ONTOLOGIC XYZ - V4 DATABASE POPULATION")
    print("=" * 80)
    print()
    
    if not DATABASE_URL:
        print("❌ DATABASE_URL not set!")
        return False
    
    try:
        conn = psycopg2.connect(DATABASE_URL)
        conn.autocommit = False
        cur = conn.cursor()
        
        print("1️⃣  Inserting 30 NBA Teams...")
        
        teams_data = [
            ('1610612737', 'ATL', 'Atlanta Hawks', 'East', 'Southeast'),
            ('1610612738', 'BOS', 'Boston Celtics', 'East', 'Atlantic'),
            ('1610612751', 'BKN', 'Brooklyn Nets', 'East', 'Atlantic'),
            ('1610612766', 'CHA', 'Charlotte Hornets', 'East', 'Southeast'),
            ('1610612741', 'CHI', 'Chicago Bulls', 'East', 'Central'),
            ('1610612739', 'CLE', 'Cleveland Cavaliers', 'East', 'Central'),
            ('1610612742', 'DAL', 'Dallas Mavericks', 'West', 'Southwest'),
            ('1610612743', 'DEN', 'Denver Nuggets', 'West', 'Northwest'),
            ('1610612765', 'DET', 'Detroit Pistons', 'East', 'Central'),
            ('1610612744', 'GSW', 'Golden State Warriors', 'West', 'Pacific'),
            ('1610612745', 'HOU', 'Houston Rockets', 'West', 'Southwest'),
            ('1610612754', 'IND', 'Indiana Pacers', 'East', 'Central'),
            ('1610612746', 'LAC', 'LA Clippers', 'West', 'Pacific'),
            ('1610612747', 'LAL', 'Los Angeles Lakers', 'West', 'Pacific'),
            ('1610612763', 'MEM', 'Memphis Grizzlies', 'West', 'Southwest'),
            ('1610612748', 'MIA', 'Miami Heat', 'East', 'Southeast'),
            ('1610612749', 'MIL', 'Milwaukee Bucks', 'East', 'Central'),
            ('1610612750', 'MIN', 'Minnesota Timberwolves', 'West', 'Northwest'),
            ('1610612740', 'NOP', 'New Orleans Pelicans', 'West', 'Southwest'),
            ('1610612752', 'NYK', 'New York Knicks', 'East', 'Atlantic'),
            ('1610612760', 'OKC', 'Oklahoma City Thunder', 'West', 'Northwest'),
            ('1610612753', 'ORL', 'Orlando Magic', 'East', 'Southeast'),
            ('1610612755', 'PHI', 'Philadelphia 76ers', 'East', 'Atlantic'),
            ('1610612756', 'PHX', 'Phoenix Suns', 'West', 'Pacific'),
            ('1610612757', 'POR', 'Portland Trail Blazers', 'West', 'Northwest'),
            ('1610612758', 'SAC', 'Sacramento Kings', 'West', 'Pacific'),
            ('1610612759', 'SAS', 'San Antonio Spurs', 'West', 'Southwest'),
            ('1610612761', 'TOR', 'Toronto Raptors', 'East', 'Atlantic'),
            ('1610612762', 'UTA', 'Utah Jazz', 'West', 'Northwest'),
            ('1610612764', 'WAS', 'Washington Wizards', 'East', 'Southeast')
        ]
        
        for team in teams_data:
            cur.execute("""
                INSERT INTO teams (team_id, abbreviation, full_name, conference, division)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (team_id) DO NOTHING
            """, team)
        
        conn.commit()
        print(f"   ✅ {len(teams_data)} teams inserted")
        print()
        
        print("2️⃣  Running Basketball Reference Scraper...")
        print("   This will populate player box scores for recent games...")
        print()
        
        # Import and run the scraper
        try:
            from basketball_reference_scraper import BasketballReferenceScraper
            scraper = BasketballReferenceScraper()
            
            # Scrape last 7 days of games
            end_date = date.today()
            start_date = end_date - timedelta(days=7)
            
            print(f"   Scraping games from {start_date} to {end_date}...")
            scraper.scrape_date_range(start_date, end_date)
            print("   ✅ Box scores populated")
            
        except Exception as e:
            print(f"   ⚠️  Scraper error: {e}")
            print("   → Run manually: python3 basketball_reference_scraper.py")
        
        print()
        print("3️⃣  Running Comprehensive NBA System...")
        print("   This will populate injuries, depth charts, schedule...")
        print()
        
        try:
            from comprehensive_nba_system import ComprehensiveNBASystem
            system = ComprehensiveNBASystem()
            
            # Scrape injuries
            print("   → Scraping injuries...")
            system.scrape_injuries()
            print("   ✅ Injuries populated")
            
            # Fetch schedule
            print("   → Fetching NBA schedule...")
            system.fetch_nba_schedule()
            print("   ✅ Schedule populated")
            
            # Build depth charts
            print("   → Building depth charts...")
            system.build_depth_charts()
            print("   ✅ Depth charts populated")
            
        except Exception as e:
            print(f"   ⚠️  System error: {e}")
            print("   → Run manually: python3 comprehensive_nba_system.py")
        
        print()
        print("=" * 80)
        print("✅ DATABASE POPULATION COMPLETE!")
        print("=" * 80)
        print()
        
        # Verify data
        cur.execute("SELECT COUNT(*) FROM teams")
        teams_count = cur.fetchone()[0]
        print(f"✓ Teams: {teams_count}")
        
        cur.execute("SELECT COUNT(*) FROM players")
        players_count = cur.fetchone()[0]
        print(f"✓ Players: {players_count}")
        
        cur.execute("SELECT COUNT(*) FROM player_box_scores")
        box_scores_count = cur.fetchone()[0]
        print(f"✓ Box Scores: {box_scores_count}")
        
        cur.execute("SELECT COUNT(*) FROM player_injuries")
        injuries_count = cur.fetchone()[0]
        print(f"✓ Injuries: {injuries_count}")
        
        cur.execute("SELECT COUNT(*) FROM nba_schedule")
        schedule_count = cur.fetchone()[0]
        print(f"✓ Schedule: {schedule_count}")
        
        print()
        print("🎯 Next: Refresh ontologicxyz.com to see data!")
        print()
        
        cur.close()
        conn.close()
        return True
        
    except Exception as e:
        print()
        print("=" * 80)
        print("❌ POPULATION FAILED")
        print("=" * 80)
        print(f"Error: {e}")
        print()
        import traceback
        traceback.print_exc()
        if conn:
            conn.rollback()
            conn.close()
        return False


if __name__ == "__main__":
    success = populate_database()
    exit(0 if success else 1)

