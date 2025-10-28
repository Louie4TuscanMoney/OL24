#!/usr/bin/env python3
"""
Quick V4 Database Population
Just inserts teams - other data populated by existing scripts
"""

import os
import psycopg2

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
        conn.autocommit = True  # No transactions - each insert independent
        cur = conn.cursor()
        
        print("1️⃣  Inserting 30 NBA Teams...")
        
        teams_data = [
            ('1610612737', 'ATL', 'Atlanta Hawks', 'East', 'Southeast', 'Atlanta', 'GA', 'State Farm Arena'),
            ('1610612738', 'BOS', 'Boston Celtics', 'East', 'Atlantic', 'Boston', 'MA', 'TD Garden'),
            ('1610612751', 'BKN', 'Brooklyn Nets', 'East', 'Atlantic', 'Brooklyn', 'NY', 'Barclays Center'),
            ('1610612766', 'CHA', 'Charlotte Hornets', 'East', 'Southeast', 'Charlotte', 'NC', 'Spectrum Center'),
            ('1610612741', 'CHI', 'Chicago Bulls', 'East', 'Central', 'Chicago', 'IL', 'United Center'),
            ('1610612739', 'CLE', 'Cleveland Cavaliers', 'East', 'Central', 'Cleveland', 'OH', 'Rocket Mortgage FieldHouse'),
            ('1610612742', 'DAL', 'Dallas Mavericks', 'West', 'Southwest', 'Dallas', 'TX', 'American Airlines Center'),
            ('1610612743', 'DEN', 'Denver Nuggets', 'West', 'Northwest', 'Denver', 'CO', 'Ball Arena'),
            ('1610612765', 'DET', 'Detroit Pistons', 'East', 'Central', 'Detroit', 'MI', 'Little Caesars Arena'),
            ('1610612744', 'GSW', 'Golden State Warriors', 'West', 'Pacific', 'San Francisco', 'CA', 'Chase Center'),
            ('1610612745', 'HOU', 'Houston Rockets', 'West', 'Southwest', 'Houston', 'TX', 'Toyota Center'),
            ('1610612754', 'IND', 'Indiana Pacers', 'East', 'Central', 'Indianapolis', 'IN', 'Gainbridge Fieldhouse'),
            ('1610612746', 'LAC', 'LA Clippers', 'West', 'Pacific', 'Los Angeles', 'CA', 'Crypto.com Arena'),
            ('1610612747', 'LAL', 'Los Angeles Lakers', 'West', 'Pacific', 'Los Angeles', 'CA', 'Crypto.com Arena'),
            ('1610612763', 'MEM', 'Memphis Grizzlies', 'West', 'Southwest', 'Memphis', 'TN', 'FedExForum'),
            ('1610612748', 'MIA', 'Miami Heat', 'East', 'Southeast', 'Miami', 'FL', 'Kaseya Center'),
            ('1610612749', 'MIL', 'Milwaukee Bucks', 'East', 'Central', 'Milwaukee', 'WI', 'Fiserv Forum'),
            ('1610612750', 'MIN', 'Minnesota Timberwolves', 'West', 'Northwest', 'Minneapolis', 'MN', 'Target Center'),
            ('1610612740', 'NOP', 'New Orleans Pelicans', 'West', 'Southwest', 'New Orleans', 'LA', 'Smoothie King Center'),
            ('1610612752', 'NYK', 'New York Knicks', 'East', 'Atlantic', 'New York', 'NY', 'Madison Square Garden'),
            ('1610612760', 'OKC', 'Oklahoma City Thunder', 'West', 'Northwest', 'Oklahoma City', 'OK', 'Paycom Center'),
            ('1610612753', 'ORL', 'Orlando Magic', 'East', 'Southeast', 'Orlando', 'FL', 'Amway Center'),
            ('1610612755', 'PHI', 'Philadelphia 76ers', 'East', 'Atlantic', 'Philadelphia', 'PA', 'Wells Fargo Center'),
            ('1610612756', 'PHX', 'Phoenix Suns', 'West', 'Pacific', 'Phoenix', 'AZ', 'Footprint Center'),
            ('1610612757', 'POR', 'Portland Trail Blazers', 'West', 'Northwest', 'Portland', 'OR', 'Moda Center'),
            ('1610612758', 'SAC', 'Sacramento Kings', 'West', 'Pacific', 'Sacramento', 'CA', 'Golden 1 Center'),
            ('1610612759', 'SAS', 'San Antonio Spurs', 'West', 'Southwest', 'San Antonio', 'TX', 'AT&T Center'),
            ('1610612761', 'TOR', 'Toronto Raptors', 'East', 'Atlantic', 'Toronto', 'ON', 'Scotiabank Arena'),
            ('1610612762', 'UTA', 'Utah Jazz', 'West', 'Northwest', 'Salt Lake City', 'UT', 'Delta Center'),
            ('1610612764', 'WAS', 'Washington Wizards', 'East', 'Southeast', 'Washington', 'DC', 'Capital One Arena')
        ]
        
        teams_inserted = 0
        for team in teams_data:
            try:
                cur.execute("""
                    INSERT INTO teams (team_id, abbreviation, full_name, conference, division, city, state, arena)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (team_id) DO NOTHING
                """, team)
                teams_inserted += 1
            except Exception as e:
                print(f"   ⚠️  Team {team[2]} failed: {e}")
        
        print(f"   ✅ {teams_inserted}/{len(teams_data)} teams inserted")
        print()
        
        print("=" * 80)
        print("✅ TEAMS POPULATED!")
        print("=" * 80)
        print()
        
        # Verify teams
        cur.execute("SELECT COUNT(*) FROM teams")
        teams_count = cur.fetchone()[0]
        print(f"✓ Teams in database: {teams_count}")
        print()
        
        print("Next steps:")
        print("  1. Populate player data:")
        print("     python3 basketball_reference_scraper.py")
        print()
        print("  2. Populate injuries & schedule:")
        print("     python3 comprehensive_nba_system.py")
        print()
        print("  3. Check frontend:")
        print("     https://ontologicxyz.com/stats")
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
            conn.close()
        return False


if __name__ == "__main__":
    success = populate_database()
    exit(0 if success else 1)
