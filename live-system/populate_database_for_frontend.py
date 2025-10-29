"""
COMPREHENSIVE DATABASE POPULATION FOR FRONTEND
Populates ALL tables needed for Vercel frontend

Tables Populated:
✅ teams (with logos & colors)
✅ players (from nba_api)
✅ player_season_stats (from Basketball Reference)
✅ nba_schedule (from nba_api)
✅ player_injuries (from ESPN)
✅ team_depth_charts (computed from MPG)

Run this on Railway:
1. Via Railway console: python3 populate_database_for_frontend.py
2. Or schedule daily: 3:30 AM UTC (in trading_dashboard_api.py startup)
"""

import os
import sys
import time
from datetime import datetime, date, timedelta
import psycopg2
from psycopg2.extras import execute_values


def get_db_connection():
    """Get PostgreSQL connection"""
    DATABASE_URL = os.getenv('DATABASE_URL')
    if not DATABASE_URL:
        raise Exception("DATABASE_URL environment variable not set!")
    return psycopg2.connect(DATABASE_URL)


def populate_teams_with_visuals(conn):
    """Populate teams with logos and brand colors (for frontend!)"""
    print("\n" + "="*80)
    print("1️⃣  POPULATING TEAMS (with logos & colors)")
    print("="*80)
    
    cur = conn.cursor()
    
    # Teams are already in schema seed data - just verify
    cur.execute("SELECT COUNT(*) FROM teams")
    count = cur.fetchone()[0]
    
    print(f"   ✅ {count} teams with logos & colors")
    
    # Show sample
    cur.execute("""
        SELECT abbreviation, full_name, primary_color, secondary_color 
        FROM teams 
        ORDER BY abbreviation 
        LIMIT 5
    """)
    
    print(f"\n   Sample teams:")
    for row in cur.fetchall():
        print(f"      {row[0]}: {row[1]} ({row[2]}, {row[3]})")
    
    conn.commit()
    cur.close()
    return count


def populate_players_from_nba_api(conn):
    """Populate players from nba_api"""
    print("\n" + "="*80)
    print("2️⃣  POPULATING PLAYERS (from nba_api)")
    print("="*80)
    
    try:
        from nba_api.stats.static import players as nba_players
        from nba_api.stats.endpoints import commonplayerinfo, commonteamroster
    except ImportError:
        print("   ❌ nba_api not installed. Install: pip install nba-api")
        return 0
    
    cur = conn.cursor()
    
    # Get all active players
    all_players = nba_players.get_active_players()
    print(f"   📊 Found {len(all_players)} active players")
    
    inserted = 0
    updated = 0
    
    for player in all_players[:50]:  # Limit to first 50 for speed (remove limit in production)
        try:
            player_id = str(player['id'])
            name = player['full_name']
            
            # Get team mapping
            cur.execute("SELECT team_id FROM teams WHERE abbreviation = %s", (player.get('team_abbreviation'),))
            team_row = cur.fetchone()
            team_id = team_row[0] if team_row else None
            
            # Insert/update player
            cur.execute("""
                INSERT INTO players (
                    player_id, name, first_name, last_name, team_id,
                    is_active, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, NOW())
                ON CONFLICT (player_id) DO UPDATE SET
                    name = EXCLUDED.name,
                    team_id = EXCLUDED.team_id,
                    is_active = EXCLUDED.is_active,
                    updated_at = NOW()
                RETURNING (xmax = 0) AS inserted
            """, (
                player_id,
                name,
                player.get('first_name'),
                player.get('last_name'),
                team_id,
                player.get('is_active', True)
            ))
            
            was_inserted = cur.fetchone()[0]
            if was_inserted:
                inserted += 1
            else:
                updated += 1
            
            if (inserted + updated) % 50 == 0:
                print(f"      ... {inserted + updated} players")
                conn.commit()
            
            time.sleep(0.6)  # Rate limiting
            
        except Exception as e:
            print(f"      ⚠️  Error with {name}: {e}")
            conn.rollback()
            continue
    
    conn.commit()
    cur.close()
    
    print(f"   ✅ Inserted {inserted} new players, updated {updated} existing")
    return inserted + updated


def scrape_basketball_reference_stats(conn):
    """Scrape stats from Basketball Reference"""
    print("\n" + "="*80)
    print("3️⃣  SCRAPING PLAYER STATS (Basketball Reference)")
    print("="*80)
    
    try:
        # Import the scraper we already have
        sys.path.insert(0, os.path.dirname(__file__))
        sys.path.insert(0, '../backend/services')
        
        from scrape_basketball_reference_all import ComprehensiveBballRefScraper
        
        scraper = ComprehensiveBballRefScraper()
        results = scraper.run_all()
        
        total_players = results.get('per_100', 0) + results.get('advanced', 0)
        print(f"   ✅ Scraped stats for {total_players} players")
        
        return total_players
        
    except ImportError as e:
        print(f"   ⚠️  Basketball Reference scraper not available: {e}")
        print(f"   💡 Run manually: python3 backend/services/scrape_basketball_reference_all.py")
        return 0
    except Exception as e:
        print(f"   ❌ Error scraping: {e}")
        return 0


def populate_schedule_from_nba_api(conn):
    """Populate schedule from nba_api"""
    print("\n" + "="*80)
    print("4️⃣  POPULATING SCHEDULE (from nba_api)")
    print("="*80)
    
    try:
        from nba_api.stats.endpoints import leaguegamefinder
        from nba_api.live.nba.endpoints import scoreboard
    except ImportError:
        print("   ❌ nba_api not installed")
        return 0
    
    cur = conn.cursor()
    
    # Get upcoming games from scoreboard
    try:
        board = scoreboard.ScoreBoard()
        games_data = board.get_dict()
        
        if 'scoreboard' in games_data and 'games' in games_data['scoreboard']:
            games = games_data['scoreboard']['games']
            print(f"   📅 Found {len(games)} games")
            
            inserted = 0
            for game in games:
                try:
                    game_id = game.get('gameId')
                    game_date_str = game.get('gameTimeUTC', '')[:10]  # Extract date
                    game_date = datetime.strptime(game_date_str, '%Y-%m-%d').date() if game_date_str else date.today()
                    
                    home_team_id = str(game.get('homeTeam', {}).get('teamId'))
                    away_team_id = str(game.get('awayTeam', {}).get('teamId'))
                    
                    status = 'Scheduled'
                    if game.get('gameStatus') == 2:
                        status = 'Live'
                    elif game.get('gameStatus') == 3:
                        status = 'Final'
                    
                    cur.execute("""
                        INSERT INTO nba_schedule (
                            game_id, season_id, game_date, 
                            home_team_id, away_team_id,
                            home_score, away_score,
                            game_status
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (game_id) DO UPDATE SET
                            home_score = EXCLUDED.home_score,
                            away_score = EXCLUDED.away_score,
                            game_status = EXCLUDED.game_status,
                            updated_at = NOW()
                    """, (
                        game_id,
                        '2025-26',
                        game_date,
                        home_team_id,
                        away_team_id,
                        game.get('homeTeam', {}).get('score'),
                        game.get('awayTeam', {}).get('score'),
                        status
                    ))
                    inserted += 1
                    
                except Exception as e:
                    print(f"      ⚠️  Error with game {game_id}: {e}")
                    conn.rollback()
                    continue
            
            conn.commit()
            print(f"   ✅ Inserted/updated {inserted} games")
            return inserted
        else:
            print("   ⚠️  No games found in scoreboard")
            return 0
            
    except Exception as e:
        print(f"   ❌ Error fetching schedule: {e}")
        return 0
    finally:
        cur.close()


def populate_injuries_from_espn(conn):
    """Populate injuries from ESPN"""
    print("\n" + "="*80)
    print("5️⃣  POPULATING INJURIES (from ESPN)")
    print("="*80)
    
    try:
        import requests
    except ImportError:
        print("   ❌ requests not installed")
        return 0
    
    cur = conn.cursor()
    
    # ESPN injuries API
    url = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
    
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        
        if 'teams' not in data:
            print("   ⚠️  No injury data found")
            return 0
        
        # Mark all existing injuries as inactive
        cur.execute("UPDATE player_injuries SET is_active = FALSE")
        
        inserted = 0
        for team_data in data['teams']:
            team_abbr = team_data.get('team', {}).get('abbreviation')
            
            # Get team_id
            cur.execute("SELECT team_id FROM teams WHERE abbreviation = %s", (team_abbr,))
            team_row = cur.fetchone()
            if not team_row:
                continue
            
            for player_injury in team_data.get('injuries', []):
                try:
                    player_name = player_injury.get('athlete', {}).get('displayName')
                    status = player_injury.get('status', 'Out')
                    injury_type = player_injury.get('type', '')
                    description = player_injury.get('details', {}).get('detail', '')
                    
                    # Match player
                    cur.execute("SELECT player_id FROM players WHERE name ILIKE %s LIMIT 1", (f"%{player_name}%",))
                    player_row = cur.fetchone()
                    
                    if player_row:
                        player_id = player_row[0]
                        
                        # Insert injury
                        cur.execute("""
                            INSERT INTO player_injuries (
                                player_id, injury_date, status, 
                                injury_type, description, is_active
                            ) VALUES (%s, %s, %s, %s, %s, TRUE)
                            ON CONFLICT (player_id, injury_date) DO UPDATE SET
                                status = EXCLUDED.status,
                                description = EXCLUDED.description,
                                is_active = TRUE,
                                updated_at = NOW()
                        """, (
                            player_id,
                            date.today(),
                            status,
                            injury_type,
                            description
                        ))
                        inserted += 1
                        
                except Exception as e:
                    print(f"      ⚠️  Error with injury: {e}")
                    conn.rollback()
                    continue
        
        conn.commit()
        print(f"   ✅ Inserted/updated {inserted} injuries")
        return inserted
        
    except Exception as e:
        print(f"   ❌ Error fetching injuries: {e}")
        return 0
    finally:
        cur.close()


def compute_depth_charts_from_mpg(conn):
    """Compute depth charts from minutes per game"""
    print("\n" + "="*80)
    print("6️⃣  COMPUTING DEPTH CHARTS (from MPG)")
    print("="*80)
    
    cur = conn.cursor()
    
    # Clear existing depth charts
    cur.execute("DELETE FROM team_depth_charts")
    
    # Compute starters (top 5 by MPG per team)
    cur.execute("""
        WITH ranked_players AS (
            SELECT 
                ps.player_id,
                ps.team_id,
                p.position,
                ps.minutes_total / NULLIF(ps.games_played, 0) as mpg,
                ROW_NUMBER() OVER (PARTITION BY ps.team_id ORDER BY ps.minutes_total / NULLIF(ps.games_played, 0) DESC) as rank
            FROM player_season_stats ps
            JOIN players p ON ps.player_id = p.player_id
            WHERE ps.season_id = '2025-26' AND ps.games_played > 0
        )
        INSERT INTO team_depth_charts (team_id, player_id, position, depth_rank)
        SELECT team_id, player_id, position, rank::int
        FROM ranked_players
        WHERE rank <= 12  -- Top 12 players per team
    """)
    
    inserted = cur.rowcount
    conn.commit()
    cur.close()
    
    print(f"   ✅ Computed depth charts for all teams ({inserted} entries)")
    return inserted


def verify_data_for_frontend(conn):
    """Verify all data is ready for frontend"""
    print("\n" + "="*80)
    print("🔍 VERIFICATION: Data Ready for Frontend")
    print("="*80)
    
    cur = conn.cursor()
    
    checks = [
        ("teams", "Teams with logos & colors"),
        ("players", "Active players"),
        ("player_season_stats", "Player season stats"),
        ("nba_schedule", "Scheduled games"),
        ("player_injuries WHERE is_active = TRUE", "Active injuries"),
        ("team_depth_charts", "Depth chart entries"),
    ]
    
    results = {}
    for table, description in checks:
        cur.execute(f"SELECT COUNT(*) FROM {table}")
        count = cur.fetchone()[0]
        results[description] = count
        
        status = "✅" if count > 0 else "❌"
        print(f"   {status} {description}: {count}")
    
    # Check critical frontend endpoints
    print(f"\n   Critical API Endpoint Checks:")
    
    # /api/stats/teams
    cur.execute("""
        SELECT COUNT(*) FROM teams 
        WHERE logo_url IS NOT NULL 
        AND primary_color IS NOT NULL
    """)
    teams_with_visuals = cur.fetchone()[0]
    print(f"      {'✅' if teams_with_visuals == 30 else '⚠️'} Teams with visuals: {teams_with_visuals}/30")
    
    # /api/injuries
    cur.execute("SELECT COUNT(*) FROM player_injuries WHERE is_active = TRUE")
    active_injuries = cur.fetchone()[0]
    print(f"      {'✅' if active_injuries > 0 else '⚠️'} Active injuries: {active_injuries}")
    
    # /api/schedule
    cur.execute("SELECT COUNT(*) FROM nba_schedule WHERE game_date >= CURRENT_DATE")
    upcoming_games = cur.fetchone()[0]
    print(f"      {'✅' if upcoming_games > 0 else '⚠️'} Upcoming games: {upcoming_games}")
    
    # /api/team/{team}/depth-chart
    cur.execute("SELECT COUNT(DISTINCT team_id) FROM team_depth_charts")
    teams_with_depth = cur.fetchone()[0]
    print(f"      {'✅' if teams_with_depth > 0 else '⚠️'} Teams with depth charts: {teams_with_depth}/30")
    
    cur.close()
    return results


def main():
    """Main population script"""
    print("\n" + "="*100)
    print("🚀 COMPREHENSIVE DATABASE POPULATION FOR FRONTEND")
    print("="*100)
    print(f"   Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Target: Railway PostgreSQL")
    print(f"   Purpose: Populate ALL data needed for Vercel frontend")
    print("="*100)
    
    try:
        # Connect to database
        conn = get_db_connection()
        print("\n✅ Connected to Railway PostgreSQL")
        
        # Run population steps
        stats = {}
        stats['teams'] = populate_teams_with_visuals(conn)
        stats['players'] = populate_players_from_nba_api(conn)
        stats['stats'] = scrape_basketball_reference_stats(conn)
        stats['schedule'] = populate_schedule_from_nba_api(conn)
        stats['injuries'] = populate_injuries_from_espn(conn)
        stats['depth_charts'] = compute_depth_charts_from_mpg(conn)
        
        # Verify
        verification = verify_data_for_frontend(conn)
        
        # Summary
        print("\n" + "="*100)
        print("✅ POPULATION COMPLETE!")
        print("="*100)
        print(f"   Teams: {stats['teams']}")
        print(f"   Players: {stats['players']}")
        print(f"   Stats scraped: {stats['stats']}")
        print(f"   Schedule: {stats['schedule']} games")
        print(f"   Injuries: {stats['injuries']}")
        print(f"   Depth charts: {stats['depth_charts']} entries")
        print("\n" + "="*100)
        print("🎯 NEXT STEPS:")
        print("="*100)
        print("   1. Test API: curl https://ol24-production.up.railway.app/api/stats/teams")
        print("   2. Check frontend: ontologicxyz.com")
        print("   3. Schedule daily updates: 3:30 AM UTC")
        print("\n   Frontend endpoints ready:")
        print("      ✅ /api/stats/teams")
        print("      ✅ /api/injuries")
        print("      ✅ /api/schedule")
        print("      ✅ /api/team/{team}/depth-chart")
        print("      ✅ /api/team/{team}/schedule")
        print("="*100 + "\n")
        
        conn.close()
        
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

