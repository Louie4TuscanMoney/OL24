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
        from nba_api.stats.endpoints import (
            commonteamroster, 
            leaguegamefinder,
            teamplayerdashboard,  # For MPG stats
            leaguedashplayerstats  # For league-wide stats
        )
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
    
    # STEP 3: Populate player stats (MPG, PPG, etc.)
    print("\n📊 STEP 3: Fetching player stats and MPG...")
    
    try:
        # Get current season stats for all players
        season_stats = leaguedashplayerstats.LeagueDashPlayerStats(
            season='2025-26',
            season_type_all_star='Regular Season'
        )
        stats_df = season_stats.get_data_frames()[0]
        
        stats_updated = 0
        for _, stat_row in stats_df.iterrows():
            player_id = stat_row['PLAYER_ID']
            
            try:
                # Update player season stats
                cur.execute("""
                    INSERT INTO player_season_stats (
                        player_id, season_id, team_id, games_played,
                        minutes_played, ppg, rpg, apg, mpg
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (player_id, season_id) DO UPDATE SET
                        games_played = EXCLUDED.games_played,
                        minutes_played = EXCLUDED.minutes_played,
                        ppg = EXCLUDED.ppg,
                        rpg = EXCLUDED.rpg,
                        apg = EXCLUDED.apg,
                        mpg = EXCLUDED.mpg
                """, (
                    player_id,
                    '2025-26',
                    stat_row['TEAM_ID'],
                    stat_row.get('GP', 0),
                    stat_row.get('MIN', 0),
                    stat_row.get('PTS', 0) / max(stat_row.get('GP', 1), 1),  # PPG
                    stat_row.get('REB', 0) / max(stat_row.get('GP', 1), 1),  # RPG
                    stat_row.get('AST', 0) / max(stat_row.get('GP', 1), 1),  # APG
                    stat_row.get('MIN', 0) / max(stat_row.get('GP', 1), 1)   # MPG
                ))
                stats_updated += 1
            except Exception as e:
                print(f"   ⚠️  Error updating stats for player {player_id}: {e}")
        
        conn.commit()
        print(f"✅ Updated stats for {stats_updated} players")
        
    except Exception as e:
        print(f"⚠️  Could not fetch player stats: {e}")
        print("   Stats will be empty - populate later")
    
    # STEP 4: Fetch injuries from ESPN
    print("\n🏥 STEP 4: Fetching injury data from ESPN...")
    
    injuries_added = 0
    
    # ESPN has team injury endpoints!
    # Format: sports.core.api.espn.com/v2/sports/basketball/leagues/nba/teams/{team_id}/injuries
    
    import requests
    
    for team in all_teams:
        team_id = team['id']
        team_abbr = team['abbreviation']
        
        try:
            # ESPN team ID mapping (NBA team IDs work with ESPN)
            espn_url = f"https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/teams/{team_id}/injuries"
            
            response = requests.get(
                espn_url,
                headers={'User-Agent': 'Mozilla/5.0'},
                timeout=3
            )
            
            if response.status_code == 200:
                injury_data = response.json()
                
                # ESPN returns injuries in 'items' array
                if 'items' in injury_data:
                    for injury in injury_data['items']:
                        try:
                            # Extract injury details
                            athlete = injury.get('athlete', {})
                            athlete_id = athlete.get('id')
                            
                            if not athlete_id:
                                continue
                            
                            injury_status = injury.get('status', 'Unknown')
                            injury_type = injury.get('type', {}).get('name', 'Unknown')
                            injury_date = injury.get('date', datetime.now().isoformat())
                            
                            # Insert injury
                            cur.execute("""
                                INSERT INTO player_injuries (
                                    player_id, team_id, injury_status, injury_type, injury_date
                                ) VALUES (%s, %s, %s, %s, %s)
                                ON CONFLICT (player_id, injury_date) DO UPDATE SET
                                    injury_status = EXCLUDED.injury_status,
                                    injury_type = EXCLUDED.injury_type
                            """, (
                                athlete_id,
                                team_id,
                                injury_status,
                                injury_type,
                                injury_date
                            ))
                            injuries_added += 1
                            
                        except Exception as e:
                            print(f"      ⚠️  Error processing injury: {e}")
                    
                    if injury_data['items']:
                        print(f"   {team_abbr}: {len(injury_data['items'])} injuries")
                
        except Exception as e:
            print(f"   ⚠️  Could not fetch injuries for {team_abbr}: {e}")
        
        # Small delay to avoid rate limiting
        time.sleep(0.2)
    
    conn.commit()
    print(f"✅ Added {injuries_added} injury records")
    
    # STEP 5: Verify data
    print("\n🔍 STEP 5: Verifying data...")
    
    cur.execute("SELECT COUNT(*) FROM teams")
    team_count = cur.fetchone()[0]
    
    cur.execute("SELECT COUNT(*) FROM players")
    player_count = cur.fetchone()[0]
    
    cur.execute("SELECT COUNT(*) FROM team_depth_charts")
    depth_count = cur.fetchone()[0]
    
    cur.execute("SELECT COUNT(*) FROM player_season_stats WHERE season_id = '2025-26'")
    stats_count = cur.fetchone()[0]
    
    cur.execute("SELECT COUNT(*) FROM player_injuries")
    injury_count = cur.fetchone()[0]
    
    print(f"   Teams: {team_count}")
    print(f"   Players: {player_count}")
    print(f"   Depth chart entries: {depth_count}")
    print(f"   Player stats (2025-26): {stats_count}")
    print(f"   Current injuries: {injury_count}")
    
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

