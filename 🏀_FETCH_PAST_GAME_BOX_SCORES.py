#!/usr/bin/env python3
"""
Fetch and store box scores for all past (completed) games

This will:
1. Find all completed games (status='Final')
2. Fetch box scores using nba_api
3. Store in player_box_scores table
4. Update game records
"""

import os
import psycopg2
from nba_api.stats.endpoints import boxscoretraditionalv2
from datetime import datetime
import time

DATABASE_URL = os.getenv('DATABASE_URL')

def fetch_box_scores_for_past_games():
    """Fetch box scores for all completed games"""
    print()
    print("="*80)
    print("🏀 FETCHING BOX SCORES FOR PAST GAMES")
    print("="*80)
    print()
    
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    
    try:
        # Get all completed games without box scores
        cur.execute("""
            SELECT game_id, home_team_id, away_team_id, game_date
            FROM nba_schedule
            WHERE game_status = 'Final'
            AND game_id IS NOT NULL
            ORDER BY game_date DESC
            LIMIT 20
        """)
        
        completed_games = cur.fetchall()
        
        if not completed_games:
            print("   ℹ️  No completed games found")
            return 0
        
        print(f"   Found {len(completed_games)} completed games")
        print()
        
        total_players_added = 0
        games_processed = 0
        
        for game_id, home_id, away_id, game_date in completed_games:
            try:
                time.sleep(0.6)  # Rate limiting
                
                print(f"   Processing game {game_id}...")
                
                # Fetch box score from nba_api
                boxscore = boxscoretraditionalv2.BoxScoreTraditionalV2(
                    game_id=game_id
                )
                
                # Get player stats
                player_stats_df = boxscore.get_data_frames()[0]
                
                if player_stats_df.empty:
                    print(f"      ⚠️  No box score data available")
                    continue
                
                # Insert each player's stats
                players_added = 0
                for _, player in player_stats_df.iterrows():
                    player_id = str(player['PLAYER_ID'])
                    team_id = str(player['TEAM_ID'])
                    
                    # Insert into player_box_scores
                    cur.execute("""
                        INSERT INTO player_box_scores (
                            player_id, team_id, game_id, game_date,
                            minutes, pts, reb, ast, stl, blk,
                            fgm, fga, fg3m, fg3a, ftm, fta,
                            turnovers, fouls, plus_minus
                        ) VALUES (
                            %s, %s, %s, %s,
                            %s, %s, %s, %s, %s, %s,
                            %s, %s, %s, %s, %s, %s,
                            %s, %s, %s
                        )
                        ON CONFLICT (player_id, game_id) DO UPDATE SET
                            pts = EXCLUDED.pts,
                            reb = EXCLUDED.reb,
                            ast = EXCLUDED.ast
                    """, (
                        player_id, team_id, game_id, game_date,
                        player.get('MIN', 0),
                        player.get('PTS', 0),
                        player.get('REB', 0),
                        player.get('AST', 0),
                        player.get('STL', 0),
                        player.get('BLK', 0),
                        player.get('FGM', 0),
                        player.get('FGA', 0),
                        player.get('FG3M', 0),
                        player.get('FG3A', 0),
                        player.get('FTM', 0),
                        player.get('FTA', 0),
                        player.get('TO', 0),
                        player.get('PF', 0),
                        player.get('PLUS_MINUS', 0)
                    ))
                    players_added += 1
                
                conn.commit()
                
                print(f"      ✅ Added {players_added} player box scores")
                total_players_added += players_added
                games_processed += 1
                
            except Exception as e:
                print(f"      ❌ Error: {str(e)[:50]}")
                continue
        
        print()
        print(f"   ✅ Processed {games_processed} games")
        print(f"   ✅ Added {total_players_added} player box scores")
        
        return total_players_added
        
    except Exception as e:
        print(f"   ❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        conn.rollback()
        return 0
    finally:
        cur.close()
        conn.close()


def create_box_score_api_endpoint_guide():
    """Create guide for adding box score endpoint to API"""
    print()
    print("="*80)
    print("📝 API ENDPOINT GUIDE")
    print("="*80)
    print()
    print("Add this endpoint to trading_dashboard_api.py:")
    print()
    print('''
@app.get("/api/game/{game_id}/boxscore")
async def get_game_boxscore(game_id: str):
    """Get box score for a specific game"""
    conn = get_db_connection()
    if not conn:
        return {"error": "Database not configured"}
    
    try:
        cursor = conn.cursor()
        
        # Get player box scores
        cursor.execute("""
            SELECT 
                p.name,
                p.position,
                t.abbreviation as team_abbr,
                pbs.minutes,
                pbs.pts,
                pbs.reb,
                pbs.ast,
                pbs.stl,
                pbs.blk,
                pbs.fgm,
                pbs.fga,
                pbs.fg3m,
                pbs.fg3a,
                pbs.ftm,
                pbs.fta,
                pbs.turnovers,
                pbs.fouls,
                pbs.plus_minus
            FROM player_box_scores pbs
            JOIN players p ON pbs.player_id = p.player_id
            JOIN teams t ON pbs.team_id = t.team_id
            WHERE pbs.game_id = %s
            ORDER BY pbs.pts DESC
        """, (game_id,))
        
        box_scores = []
        for row in cursor.fetchall():
            box_scores.append({
                "player_name": row[0],
                "position": row[1],
                "team": row[2],
                "minutes": row[3],
                "points": row[4],
                "rebounds": row[5],
                "assists": row[6],
                "steals": row[7],
                "blocks": row[8],
                "fg": f"{row[9]}/{row[10]}",
                "fg3": f"{row[11]}/{row[12]}",
                "ft": f"{row[13]}/{row[14]}",
                "turnovers": row[15],
                "fouls": row[16],
                "plus_minus": row[17]
            })
        
        conn.close()
        return {"game_id": game_id, "box_scores": box_scores, "count": len(box_scores)}
    except Exception as e:
        if conn:
            conn.close()
        return {"error": str(e)}
''')
    print()
    print("Then test with:")
    print('  curl "https://ol24-production.up.railway.app/api/game/{game_id}/boxscore"')
    print()


def main():
    print()
    print("="*80)
    print("🚀 FETCH PAST GAME BOX SCORES")
    print("="*80)
    print()
    
    if not DATABASE_URL:
        print("❌ DATABASE_URL not set!")
        print("   Run: export DATABASE_URL='...'")
        return False
    
    # Fetch box scores
    total = fetch_box_scores_for_past_games()
    
    if total > 0:
        create_box_score_api_endpoint_guide()
    
    print()
    print("="*80)
    print("✅ COMPLETE!")
    print("="*80)
    print(f"   Box scores added: {total}")
    print()
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

