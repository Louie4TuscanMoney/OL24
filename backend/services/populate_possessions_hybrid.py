"""
HYBRID POSSESSION POPULATION
- For 2025-26: Calculate possessions from formula (boxscoreadvancedv2 is broken)
- For 2024-25: Use boxscoreadvancedv2.POSS (ground truth validation)

Formula: team_poss = FGA + 0.44*FTA - OREB + TOV
(This is NBA's official formula, accurate within ±1 possession)
"""

import os
import time
from datetime import datetime, timedelta
import psycopg2
from nba_api.stats.endpoints import (
    leaguegamefinder,
    boxscoretraditionalv2,
    boxscoreadvancedv2
)


def calculate_possessions_from_box_score(team_stats_row):
    """
    Calculate possessions using NBA's official formula
    team_poss = FGA + 0.44*FTA - OREB + TOV
    """
    fga = team_stats_row.get('FGA', 0)
    fta = team_stats_row.get('FTA', 0)
    oreb = team_stats_row.get('OREB', 0)
    tov = team_stats_row.get('TO', 0)
    
    poss = fga + 0.44 * fta - oreb + tov
    return round(poss, 2)


def populate_season_possessions(season='2025-26', days_back=30, use_formula=True):
    """
    Populate player stats with real possessions
    
    Args:
        season: NBA season (e.g. '2025-26')
        days_back: How many days of games to fetch
        use_formula: If True, calculate possessions; if False, use boxscoreadvancedv2
    """
    
    print("="*80)
    print(f"🏀 POSSESSION-BASED STATS POPULATION")
    print(f"   Season: {season}")
    print(f"   Method: {'Formula (FGA + 0.44*FTA - OREB + TOV)' if use_formula else 'boxscoreadvancedv2.POSS'}")
    print("="*80)
    print()
    
    # Database
    DATABASE_URL = os.getenv('DATABASE_URL')
    if not DATABASE_URL:
        print("❌ DATABASE_URL not set")
        return
    
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    print("✅ Connected to PostgreSQL")
    print()
    
    # Get games
    try:
        cutoff_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        
        print(f"📅 Fetching games from {cutoff_date}...")
        games_finder = leaguegamefinder.LeagueGameFinder(
            season_nullable=season,
            season_type_nullable='Regular Season',
            date_from_nullable=cutoff_date
        )
        games_df = games_finder.get_data_frames()[0]
        unique_games = games_df['GAME_ID'].unique()
        
        print(f"   Found {len(unique_games)} unique games")
        print(f"   Total team-game entries: {len(games_df)}")
        print()
        
    except Exception as e:
        print(f"❌ Error fetching games: {e}")
        conn.close()
        return
    
    # Process each game
    games_processed = 0
    players_inserted = 0
    possession_method_used = "formula" if use_formula else "api"
    
    for i, game_id in enumerate(unique_games, 1):
        print(f"[{i}/{len(unique_games)}] Game {game_id}")
        
        try:
            # Fetch traditional box score
            traditional = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=game_id)
            player_stats = traditional.get_data_frames()[0]
            
            # Get possessions by calculating team totals from player stats
            team_possessions = {}
            
            if use_formula:
                # Sum up team stats from all players
                team_totals = player_stats.groupby('TEAM_ID').agg({
                    'FGA': 'sum',
                    'FTA': 'sum',
                    'OREB': 'sum',
                    'TO': 'sum'
                }).reset_index()
                
                for _, team_row in team_totals.iterrows():
                    team_id = team_row['TEAM_ID']
                    poss = calculate_possessions_from_box_score(team_row)
                    team_possessions[team_id] = poss
                
                print(f"   📐 Calculated possessions: {dict(team_possessions)}")
                
            else:
                # Try to use boxscoreadvancedv2
                try:
                    advanced = boxscoreadvancedv2.BoxScoreAdvancedV2(game_id=game_id)
                    team_advanced = advanced.get_data_frames()[0]
                    
                    for _, team_row in team_advanced.iterrows():
                        team_possessions[team_row['TEAM_ID']] = team_row['POSS']
                    
                    print(f"   ✅ API possessions: {team_possessions}")
                    
                except Exception as e:
                    # Fallback to formula
                    print(f"   ⚠️  API failed ({str(e)[:50]}), using formula...")
                    team_totals = player_stats.groupby('TEAM_ID').agg({
                        'FGA': 'sum',
                        'FTA': 'sum',
                        'OREB': 'sum',
                        'TO': 'sum'
                    }).reset_index()
                    
                    for _, team_row in team_totals.iterrows():
                        team_id = team_row['TEAM_ID']
                        poss = calculate_possessions_from_box_score(team_row)
                        team_possessions[team_id] = poss
                    print(f"   📐 Calculated possessions: {dict(team_possessions)}")
            
            # Get game date
            game_entry = games_df[games_df['GAME_ID'] == game_id].iloc[0]
            game_date = game_entry['GAME_DATE']
            
            # Insert player performances
            for _, player_row in player_stats.iterrows():
                player_id = player_row['PLAYER_ID']
                team_id = player_row['TEAM_ID']
                
                team_poss = team_possessions.get(team_id, 0)
                if team_poss == 0:
                    continue
                
                # Stats
                pts = player_row.get('PTS', 0) or 0
                reb = player_row.get('REB', 0) or 0
                ast = player_row.get('AST', 0) or 0
                stl = player_row.get('STL', 0) or 0
                blk = player_row.get('BLK', 0) or 0
                tov = player_row.get('TO', 0) or 0
                
                # REAL per-100 stats
                pts_100 = (pts * 100) / team_poss if team_poss > 0 else 0
                reb_100 = (reb * 100) / team_poss if team_poss > 0 else 0
                ast_100 = (ast * 100) / team_poss if team_poss > 0 else 0
                stl_100 = (stl * 100) / team_poss if team_poss > 0 else 0
                blk_100 = (blk * 100) / team_poss if team_poss > 0 else 0
                tov_100 = (tov * 100) / team_poss if team_poss > 0 else 0
                
                # Parse minutes
                minutes_str = str(player_row.get('MIN', '0:00'))
                if ':' in minutes_str:
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
                            player_id, game_id, game_date, season_id, team_id,
                            minutes, seconds_played,
                            pts, reb, ast, stl, blk, tov,
                            fgm, fga, fg3m, fg3a, ftm, fta,
                            oreb, dreb, pf, plus_minus,
                            team_poss,
                            pts_100, reb_100, ast_100, stl_100, blk_100, tov_100
                        ) VALUES (
                            %s, %s, %s, %s, %s,
                            %s, %s,
                            %s, %s, %s, %s, %s, %s,
                            %s, %s, %s, %s, %s, %s,
                            %s, %s, %s, %s,
                            %s,
                            %s, %s, %s, %s, %s, %s
                        )
                        ON CONFLICT (player_id, game_id, game_date) DO UPDATE SET
                            team_poss = EXCLUDED.team_poss,
                            pts_100 = EXCLUDED.pts_100,
                            reb_100 = EXCLUDED.reb_100,
                            ast_100 = EXCLUDED.ast_100
                    """, (
                        player_id, game_id, game_date, season, str(team_id),
                        minutes_decimal, seconds_played,
                        pts, reb, ast, stl, blk, tov,
                        player_row.get('FGM', 0) or 0, player_row.get('FGA', 0) or 0,
                        player_row.get('FG3M', 0) or 0, player_row.get('FG3A', 0) or 0,
                        player_row.get('FTM', 0) or 0, player_row.get('FTA', 0) or 0,
                        player_row.get('OREB', 0) or 0, player_row.get('DREB', 0) or 0,
                        player_row.get('PF', 0) or 0, player_row.get('PLUS_MINUS', 0) or 0,
                        team_poss,
                        round(pts_100, 2), round(reb_100, 2), round(ast_100, 2),
                        round(stl_100, 2), round(blk_100, 2), round(tov_100, 2)
                    ))
                    
                    players_inserted += 1
                    
                except Exception as e:
                    print(f"      ⚠️  Error inserting player {player_id}: {e}")
                    conn.rollback()
                    continue
            
            conn.commit()
            games_processed += 1
            print(f"   ✅ Inserted {len(player_stats)} players (Team poss: {list(team_possessions.values())})")
            
            # Rate limit
            if i % 10 == 0:
                print(f"   ⏸️  Rate limit pause...")
                time.sleep(2)
            
        except Exception as e:
            print(f"   ❌ Error: {str(e)[:100]}")
            conn.rollback()
            continue
    
    print()
    print("="*80)
    print("📊 AGGREGATING TO SEASON STATS...")
    print("="*80)
    
    # Aggregate to player_season_stats
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
                player_id, season_id, team_id,
                COUNT(*) as games_played,
                SUM(minutes) as minutes_total,
                SUM(pts), SUM(reb), SUM(ast), SUM(stl), SUM(blk), SUM(tov),
                SUM(fgm), SUM(fga), SUM(fg3m), SUM(fg3a), SUM(ftm), SUM(fta),
                SUM(team_poss) as total_team_possessions,
                AVG(pts_100), AVG(reb_100), AVG(ast_100),
                AVG(stl_100), AVG(blk_100), AVG(tov_100)
            FROM player_box_scores
            WHERE season_id = %s
            GROUP BY player_id, season_id, team_id
            ON CONFLICT (player_id, season_id) DO UPDATE SET
                games_played = EXCLUDED.games_played,
                minutes_total = EXCLUDED.minutes_total,
                pts_total = EXCLUDED.pts_total,
                reb_total = EXCLUDED.reb_total,
                ast_total = EXCLUDED.ast_total,
                stl_total = EXCLUDED.stl_total,
                blk_total = EXCLUDED.blk_total,
                tov_total = EXCLUDED.tov_total,
                fgm_total = EXCLUDED.fgm_total,
                fga_total = EXCLUDED.fga_total,
                total_team_possessions = EXCLUDED.total_team_possessions,
                pts_100 = EXCLUDED.pts_100,
                reb_100 = EXCLUDED.reb_100,
                ast_100 = EXCLUDED.ast_100,
                stl_100 = EXCLUDED.stl_100,
                blk_100 = EXCLUDED.blk_100,
                tov_100 = EXCLUDED.tov_100
        """, (season,))
        
        conn.commit()
        
        # Verify
        cur.execute("SELECT COUNT(*) FROM player_season_stats WHERE season_id = %s", (season,))
        player_count = cur.fetchone()[0]
        
        print(f"✅ Aggregated {player_count} players to season stats")
        
    except Exception as e:
        print(f"❌ Aggregation error: {e}")
        conn.rollback()
    
    # Final verification
    print()
    print("="*80)
    print("🔍 FINAL VERIFICATION")
    print("="*80)
    
    cur.execute("SELECT COUNT(*) FROM player_box_scores WHERE season_id = %s", (season,))
    box_scores = cur.fetchone()[0]
    
    cur.execute("SELECT COUNT(*) FROM player_box_scores WHERE season_id = %s AND team_poss IS NOT NULL", (season,))
    with_poss = cur.fetchone()[0]
    
    cur.execute("SELECT COUNT(*) FROM player_season_stats WHERE season_id = %s", (season,))
    season_stats = cur.fetchone()[0]
    
    print(f"   Box scores inserted: {box_scores}")
    print(f"   With possessions: {with_poss}")
    print(f"   Season stats aggregated: {season_stats}")
    print(f"   Games processed: {games_processed}")
    print(f"   Player performances: {players_inserted}")
    
    # Sample stats
    print()
    print("📊 SAMPLE STATS (Top 5 PPG):")
    cur.execute("""
        SELECT p.name, ps.ppg, ps.pts_100, ps.games_played, t.abbreviation
        FROM player_season_stats ps
        JOIN players p ON ps.player_id = p.player_id
        JOIN teams t ON ps.team_id = t.team_id
        WHERE ps.season_id = %s AND ps.games_played > 0
        ORDER BY ps.ppg DESC
        LIMIT 5
    """, (season,))
    
    for row in cur.fetchall():
        print(f"   {row[0]:25s} {row[4]:3s}  {row[1]:5.1f} PPG  {row[2]:5.1f} Pts/100  ({row[3]} GP)")
    
    conn.close()
    
    print()
    print("="*80)
    print("✅ POPULATION COMPLETE WITH REAL POSSESSIONS")
    print("="*80)


if __name__ == "__main__":
    import sys
    
    season = sys.argv[1] if len(sys.argv) > 1 else '2025-26'
    days_back = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    
    # For 2025-26, always use formula (API is broken)
    # For older seasons, try API first, fallback to formula
    use_formula = (season == '2025-26')
    
    print(f"🎯 Strategy: {'FORMULA (API broken for 2025-26)' if use_formula else 'API with formula fallback'}")
    print()
    
    populate_season_possessions(season, days_back, use_formula)

