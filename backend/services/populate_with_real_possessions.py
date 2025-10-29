"""
POPULATE NBA STATS WITH REAL POSSESSIONS
Uses boxscoreadvancedv2 to get NBA's official possession counts

This is the OPTIMAL approach:
- Uses NBA's own possession calculation (100% accurate)
- Faster than parsing play-by-play
- Works for all completed games
- Enables accurate per-100 stats
"""

import os
import time
from datetime import datetime, timedelta
import psycopg2
from nba_api.stats.static import teams
from nba_api.stats.endpoints import (
    leaguegamefinder,
    boxscoretraditionalv2,
    boxscoreadvancedv2
)


def populate_with_real_possessions(season='2025-26', days_back=7):
    """
    Populate player_box_scores with REAL possession data
    
    For each game:
    1. Fetch box score (pts, reb, ast, etc.)
    2. Fetch advanced box score (POSS - real possessions!)
    3. Calculate accurate per-100 stats
    4. Store in player_box_scores table
    """
    
    print("="*80)
    print(f"🏀 POPULATING STATS WITH REAL POSSESSIONS")
    print(f"   Season: {season}")
    print(f"   Days back: {days_back}")
    print("="*80)
    print()
    
    # Database connection
    DATABASE_URL = os.getenv('DATABASE_URL')
    if not DATABASE_URL:
        print("❌ DATABASE_URL not set")
        return
    
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    print("✅ Connected to PostgreSQL")
    print()
    
    # Get recent games
    try:
        cutoff_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        
        games_finder = leaguegamefinder.LeagueGameFinder(
            season_nullable=season,
            season_type_nullable='Regular Season',
            date_from_nullable=cutoff_date
        )
        games_df = games_finder.get_data_frames()[0]
        
        # Get unique games
        unique_games = games_df['GAME_ID'].unique()
        
        print(f"📅 Found {len(unique_games)} games since {cutoff_date}")
        print()
        
    except Exception as e:
        print(f"❌ Error fetching games: {e}")
        conn.close()
        return
    
    # Process each game
    games_processed = 0
    players_inserted = 0
    
    for i, game_id in enumerate(unique_games, 1):
        print(f"Game {i}/{len(unique_games)}: {game_id}")
        
        try:
            # 1. Get traditional box score
            print(f"   📊 Fetching box score...")
            traditional = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=game_id)
            player_stats = traditional.get_data_frames()[0]
            
            # 2. Get advanced box score (HAS REAL POSSESSIONS!)
            print(f"   🎯 Fetching advanced stats (possessions)...")
            advanced = boxscoreadvancedv2.BoxScoreAdvancedV2(game_id=game_id)
            team_advanced = advanced.get_data_frames()[0]
            
            # Build team_id → possessions map
            team_possessions = {}
            for _, team_row in team_advanced.iterrows():
                team_possessions[team_row['TEAM_ID']] = team_row['POSS']
            
            print(f"   ✅ Possessions: {team_possessions}")
            
            # 3. Insert player stats with REAL per-100
            for _, player_row in player_stats.iterrows():
                player_id = player_row['PLAYER_ID']
                team_id = player_row['TEAM_ID']
                
                # Get real possessions for this team
                team_poss = team_possessions.get(team_id, 0)
                if team_poss == 0:
                    print(f"      ⚠️  No possessions for team {team_id}, skipping")
                    continue
                
                # Calculate REAL per-100 stats
                pts = player_row.get('PTS', 0)
                reb = player_row.get('REB', 0)
                ast = player_row.get('AST', 0)
                stl = player_row.get('STL', 0)
                blk = player_row.get('BLK', 0)
                tov = player_row.get('TO', 0)
                
                pts_100 = (pts * 100) / team_poss
                reb_100 = (reb * 100) / team_poss
                ast_100 = (ast * 100) / team_poss
                stl_100 = (stl * 100) / team_poss
                blk_100 = (blk * 100) / team_poss
                tov_100 = (tov * 100) / team_poss
                
                # Parse minutes (format: "MM:SS")
                minutes_str = player_row.get('MIN', '0:00')
                if isinstance(minutes_str, str) and ':' in minutes_str:
                    try:
                        mins, secs = minutes_str.split(':')
                        minutes_decimal = int(mins) + int(secs) / 60
                        seconds_played = int(mins) * 60 + int(secs)
                    except:
                        minutes_decimal = 0
                        seconds_played = 0
                else:
                    minutes_decimal = float(minutes_str) if minutes_str else 0
                    seconds_played = int(minutes_decimal * 60)
                
                # Insert into player_box_scores
                try:
                    cur.execute("""
                        INSERT INTO player_box_scores (
                            player_id, game_id, season_id, team_id,
                            minutes, seconds_played,
                            pts, reb, ast, stl, blk, tov,
                            fgm, fga, fg3m, fg3a, ftm, fta,
                            team_possessions,
                            pts_100, reb_100, ast_100, stl_100, blk_100, tov_100
                        ) VALUES (
                            %s, %s, %s, %s,
                            %s, %s,
                            %s, %s, %s, %s, %s, %s,
                            %s, %s, %s, %s, %s, %s,
                            %s,
                            %s, %s, %s, %s, %s, %s
                        )
                        ON CONFLICT (player_id, game_id) DO UPDATE SET
                            pts_100 = EXCLUDED.pts_100,
                            reb_100 = EXCLUDED.reb_100,
                            team_possessions = EXCLUDED.team_possessions
                    """, (
                        player_id, game_id, season, str(team_id),
                        minutes_decimal, seconds_played,
                        pts, reb, ast, stl, blk, tov,
                        player_row.get('FGM', 0), player_row.get('FGA', 0),
                        player_row.get('FG3M', 0), player_row.get('FG3A', 0),
                        player_row.get('FTM', 0), player_row.get('FTA', 0),
                        int(team_poss),
                        round(pts_100, 2), round(reb_100, 2), round(ast_100, 2),
                        round(stl_100, 2), round(blk_100, 2), round(tov_100, 2)
                    ))
                    
                    players_inserted += 1
                    
                except Exception as e:
                    print(f"      ⚠️  Error inserting player {player_id}: {e}")
                    continue
            
            conn.commit()
            games_processed += 1
            print(f"   ✅ Inserted {len(player_stats)} player performances")
            print()
            
            # Rate limit
            if i % 5 == 0:
                print(f"   ⏸️  Rate limit pause...")
                time.sleep(2)
            
        except Exception as e:
            print(f"   ❌ Error processing game: {e}")
            conn.rollback()
            continue
    
    # Final stats
    print()
    print("="*80)
    print("✅ POPULATION COMPLETE")
    print("="*80)
    print(f"   Games processed: {games_processed}")
    print(f"   Player performances: {players_inserted}")
    print()
    
    # Now aggregate to player_season_stats
    print("📊 Aggregating to player_season_stats...")
    try:
        cur.execute("""
            INSERT INTO player_season_stats (
                player_id, season_id, team_id,
                games_played, minutes_total,
                pts_total, reb_total, ast_total, stl_total, blk_total, tov_total,
                fgm_total, fga_total, fg3m_total, fg3a_total, ftm_total, fta_total,
                total_team_possessions,
                pts_100, reb_100, ast_100, stl_100, blk_100, tov_100
            )
            SELECT 
                player_id, 
                season_id,
                team_id,
                COUNT(*) as games_played,
                SUM(minutes) as minutes_total,
                SUM(pts) as pts_total,
                SUM(reb) as reb_total,
                SUM(ast) as ast_total,
                SUM(stl) as stl_total,
                SUM(blk) as blk_total,
                SUM(tov) as tov_total,
                SUM(fgm) as fgm_total,
                SUM(fga) as fga_total,
                SUM(fg3m) as fg3m_total,
                SUM(fg3a) as fg3a_total,
                SUM(ftm) as ftm_total,
                SUM(fta) as fta_total,
                SUM(team_possessions) as total_team_possessions,
                AVG(pts_100) as pts_100,
                AVG(reb_100) as reb_100,
                AVG(ast_100) as ast_100,
                AVG(stl_100) as stl_100,
                AVG(blk_100) as blk_100,
                AVG(tov_100) as tov_100
            FROM player_box_scores
            WHERE season_id = %s
            GROUP BY player_id, season_id, team_id
            ON CONFLICT (player_id, season_id) DO UPDATE SET
                games_played = EXCLUDED.games_played,
                minutes_total = EXCLUDED.minutes_total,
                pts_total = EXCLUDED.pts_total,
                reb_total = EXCLUDED.reb_total,
                ast_total = EXCLUDED.ast_total,
                total_team_possessions = EXCLUDED.total_team_possessions,
                pts_100 = EXCLUDED.pts_100,
                reb_100 = EXCLUDED.reb_100,
                ast_100 = EXCLUDED.ast_100
        """, (season,))
        
        conn.commit()
        
        cur.execute("SELECT COUNT(*) FROM player_season_stats WHERE season_id = %s", (season,))
        player_count = cur.fetchone()[0]
        
        print(f"✅ Aggregated stats for {player_count} players")
        
    except Exception as e:
        print(f"❌ Error aggregating stats: {e}")
    
    conn.close()
    
    print()
    print("="*80)
    print("🎯 NEXT STEPS:")
    print("="*80)
    print(f"   1. Verify: SELECT COUNT(*) FROM player_box_scores WHERE season_id = '{season}'")
    print(f"   2. Verify: SELECT COUNT(*) FROM player_season_stats WHERE season_id = '{season}'")
    print(f"   3. Test API: curl https://ol24-production.up.railway.app/api/stats/teams")
    print("="*80)


if __name__ == "__main__":
    import sys
    
    season = sys.argv[1] if len(sys.argv) > 1 else '2025-26'
    days_back = int(sys.argv[2]) if len(sys.argv) > 2 else 7
    
    populate_with_real_possessions(season, days_back)

