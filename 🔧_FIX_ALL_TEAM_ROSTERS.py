#!/usr/bin/env python3
"""
Fix all team rosters by fetching from nba_api

This ensures every player is on the correct team.
"""

import os
import psycopg2
from nba_api.stats.endpoints import commonteamroster
import time

DATABASE_URL = os.getenv('DATABASE_URL')

def update_all_team_rosters():
    """Update rosters for all 30 teams from nba_api"""
    print()
    print("="*80)
    print("🏀 UPDATING ALL TEAM ROSTERS FROM NBA_API")
    print("="*80)
    print()
    
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    
    try:
        # Get all teams
        cur.execute("SELECT team_id, abbreviation FROM teams ORDER BY abbreviation")
        teams = cur.fetchall()
        
        print(f"   Updating rosters for {len(teams)} teams...")
        print()
        
        total_updated = 0
        
        for team_id, abbr in teams:
            try:
                time.sleep(0.6)  # Rate limiting
                
                # Get roster from nba_api
                roster = commonteamroster.CommonTeamRoster(
                    team_id=team_id,
                    season='2025-26'
                )
                df = roster.get_data_frames()[0]
                
                if df.empty:
                    print(f"   ⚠️  {abbr}: No roster found")
                    continue
                
                # Update each player
                updated_count = 0
                for _, player in df.iterrows():
                    player_id = str(player['PLAYER_ID'])
                    player_name = player['PLAYER']
                    jersey = player.get('NUM', '')
                    position = player.get('POSITION', '')
                    
                    # Check if player exists
                    cur.execute("""
                        SELECT player_id FROM players WHERE player_id = %s
                    """, (player_id,))
                    
                    # Clean position (must be valid or NULL)
                    clean_pos = position if position and position.strip() else None
                    clean_jersey = jersey if jersey else None
                    
                    if cur.fetchone():
                        # Update existing player
                        cur.execute("""
                            UPDATE players
                            SET team_id = %s,
                                jersey_number = COALESCE(%s, jersey_number),
                                position = COALESCE(%s, position)
                            WHERE player_id = %s
                        """, (team_id, clean_jersey, clean_pos, player_id))
                        updated_count += 1
                    else:
                        # Insert new player (skip if position would violate constraint)
                        if clean_pos:  # Only insert if we have a position
                            cur.execute("""
                                INSERT INTO players (player_id, name, team_id, jersey_number, position)
                                VALUES (%s, %s, %s, %s, %s)
                                ON CONFLICT (player_id) DO UPDATE SET
                                    team_id = EXCLUDED.team_id,
                                    jersey_number = COALESCE(EXCLUDED.jersey_number, players.jersey_number)
                            """, (player_id, player_name, team_id, clean_jersey, clean_pos))
                            updated_count += 1
                
                conn.commit()
                
                print(f"   ✅ {abbr}: {updated_count} players")
                total_updated += updated_count
                
            except Exception as e:
                print(f"   ❌ {abbr}: {str(e)[:50]}")
                continue
        
        print()
        print(f"   ✅ Total players updated: {total_updated}")
        
        return total_updated
        
    except Exception as e:
        print(f"   ❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        conn.rollback()
        return 0
    finally:
        cur.close()
        conn.close()


def rebuild_depth_charts_correct():
    """Rebuild depth charts with correct positions based on real data"""
    print()
    print("="*80)
    print("🏀 REBUILDING DEPTH CHARTS (MPG-BASED)")
    print("="*80)
    
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    
    try:
        # Clear existing
        cur.execute("DELETE FROM team_depth_charts")
        
        # Get all teams
        cur.execute("SELECT team_id, abbreviation FROM teams")
        teams = cur.fetchall()
        
        print(f"   Processing {len(teams)} teams...")
        print()
        
        for team_id, abbr in teams:
            # Get top 5 players by MPG
            cur.execute("""
                SELECT 
                    p.player_id, 
                    p.name,
                    p.position,
                    COALESCE(pss.minutes_total / NULLIF(pss.games_played, 0), 0) as mpg
                FROM players p
                LEFT JOIN player_season_stats pss 
                    ON p.player_id = pss.player_id AND pss.season_id = '2025-26'
                WHERE p.team_id = %s
                ORDER BY mpg DESC
                LIMIT 5
            """, (team_id,))
            
            starters = cur.fetchall()
            
            if len(starters) < 5:
                print(f"   ⚠️  {abbr}: Only {len(starters)} players found")
            
            # Insert into depth chart
            for rank, (player_id, player_name, position, mpg) in enumerate(starters, 1):
                # Use position if available, otherwise default to F
                pos = position if position and position != '' else 'F'
                
                cur.execute("""
                    INSERT INTO team_depth_charts (team_id, player_id, position, depth_rank)
                    VALUES (%s, %s, %s, %s)
                """, (team_id, player_id, pos, rank))
                
                if rank == 1:
                    print(f"   {abbr}: #{rank} {player_name} ({pos}) - {mpg:.1f} MPG")
        
        conn.commit()
        
        print()
        print("   ✅ Depth charts rebuilt!")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        conn.rollback()
        return False
    finally:
        cur.close()
        conn.close()


def main():
    print()
    print("="*80)
    print("🚀 FIX ALL TEAM ROSTERS AND DEPTH CHARTS")
    print("="*80)
    print()
    
    if not DATABASE_URL:
        print("❌ DATABASE_URL not set!")
        return False
    
    # Step 1: Update all rosters
    updated = update_all_team_rosters()
    
    # Step 2: Rebuild depth charts
    success = rebuild_depth_charts_correct()
    
    print()
    print("="*80)
    print("✅ COMPLETE!")
    print("="*80)
    print(f"   Players updated: {updated}")
    print(f"   Depth charts: {'✅' if success else '❌'}")
    print()
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

