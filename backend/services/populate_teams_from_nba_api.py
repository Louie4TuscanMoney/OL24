"""
POPULATE TEAMS FROM NBA_API
Fetches all team data from nba_api and stores in PostgreSQL

Run this to populate:
- Teams table (all 30 teams)
- Players table (all active players)
- Team rosters (current season)
- Depth charts (projected starters)

This ensures frontend has data to display!
"""

import os
import sys
from datetime import datetime


def populate_all_teams():
    """Populate all teams and rosters from nba_api → PostgreSQL"""
    
    print("="*80)
    print("📊 POPULATING TEAMS FROM NBA_API → POSTGRESQL")
    print("="*80)
    print()
    
    # Import dependencies
    try:
        import psycopg2
        from nba_api.stats.static import teams, players
        from nba_api.stats.endpoints import commonteamroster, leaguegamefinder
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Install: pip install nba-api psycopg2-binary")
        return
    
    # Get database connection
    DATABASE_URL = os.getenv('DATABASE_URL')
    if not DATABASE_URL:
        print("❌ DATABASE_URL not set")
        return
    
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        print("✅ Connected to PostgreSQL")
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return
    
    # STEP 1: Populate all 30 NBA teams
    print("\n📋 STEP 1: Populating teams...")
    
    all_teams = teams.get_teams()
    teams_inserted = 0
    
    for team in all_teams:
        try:
            cur.execute("""
                INSERT INTO teams (team_id, abbreviation, nickname, full_name, city, state, year_founded)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (team_id) DO UPDATE SET
                    abbreviation = EXCLUDED.abbreviation,
                    nickname = EXCLUDED.nickname,
                    full_name = EXCLUDED.full_name,
                    city = EXCLUDED.city
            """, (
                team['id'],
                team['abbreviation'],
                team['nickname'],
                team['full_name'],
                team.get('city', team['full_name'].split()[-1]),  # Extract city from name if not present
                team.get('state', ''),
                team.get('year_founded', 1946)
            ))
            teams_inserted += 1
        except Exception as e:
            print(f"   ⚠️  Error inserting {team['abbreviation']}: {e}")
    
    conn.commit()
    print(f"✅ Inserted/updated {teams_inserted} teams")
    
    # STEP 2: Populate players and rosters for each team
    print("\n👥 STEP 2: Populating rosters for all teams...")
    
    players_inserted = 0
    roster_entries = 0
    
    for team in all_teams:
        team_id = team['id']
        team_abbr = team['abbreviation']
        
        print(f"\n   📊 {team_abbr}...")
        
        try:
            # Fetch roster
            roster = commonteamroster.CommonTeamRoster(team_id=team_id)
            roster_df = roster.get_data_frames()[0]
            
            if roster_df.empty:
                print(f"      ⚠️  No roster data")
                continue
            
            # Insert each player
            for _, player_row in roster_df.iterrows():
                player_id = player_row['PLAYER_ID']
                player_name = player_row['PLAYER']
                
                try:
                    # Insert player
                    cur.execute("""
                        INSERT INTO players (player_id, player_name, team_id, position, jersey_number)
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (player_id) DO UPDATE SET
                            player_name = EXCLUDED.player_name,
                            team_id = EXCLUDED.team_id,
                            position = EXCLUDED.position,
                            jersey_number = EXCLUDED.jersey_number
                    """, (
                        player_id,
                        player_name,
                        team_id,
                        player_row.get('POSITION', 'N/A'),
                        player_row.get('NUM', '')
                    ))
                    players_inserted += 1
                    
                    # Insert depth chart entry (assume first 5 are starters)
                    is_starter = roster_entries % len(roster_df) < 5
                    
                    cur.execute("""
                        INSERT INTO team_depth_charts (team_id, player_id, position, depth_order, is_projected_starter)
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (team_id, player_id) DO UPDATE SET
                            position = EXCLUDED.position,
                            depth_order = EXCLUDED.depth_order,
                            is_projected_starter = EXCLUDED.is_projected_starter
                    """, (
                        team_id,
                        player_id,
                        player_row.get('POSITION', 'N/A'),
                        roster_entries % len(roster_df),
                        is_starter
                    ))
                    roster_entries += 1
                    
                except Exception as e:
                    print(f"      ⚠️  Error with player {player_name}: {e}")
                    continue
            
            conn.commit()
            print(f"      ✅ {len(roster_df)} players")
            
        except Exception as e:
            print(f"      ❌ Roster fetch failed: {e}")
            conn.rollback()
            continue
    
    print(f"\n✅ Total: {players_inserted} players, {roster_entries} roster entries")
    
    # STEP 3: Verify data
    print("\n🔍 STEP 3: Verifying data...")
    
    cur.execute("SELECT COUNT(*) FROM teams")
    team_count = cur.fetchone()[0]
    
    cur.execute("SELECT COUNT(*) FROM players")
    player_count = cur.fetchone()[0]
    
    cur.execute("SELECT COUNT(*) FROM team_depth_charts")
    depth_count = cur.fetchone()[0]
    
    print(f"   Teams in DB: {team_count}")
    print(f"   Players in DB: {player_count}")
    print(f"   Depth chart entries: {depth_count}")
    
    conn.close()
    
    print("\n" + "="*80)
    print("✅ POPULATION COMPLETE!")
    print("="*80)
    print()
    print("Next steps:")
    print("1. Check API endpoint: curl https://ol24-production.up.railway.app/api/team/LAL/depth-chart")
    print("2. View frontend: ontologicxyz.com/team/LAL")
    print()


if __name__ == "__main__":
    populate_all_teams()

