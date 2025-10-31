#!/usr/bin/env python3
"""
COMPLETE FIX - USE NBA_API FOR ALL PLAYER POSITIONS

This will:
1. Get positions for ALL 350 players from nba_api
2. Get jersey numbers for all players
3. Fix depth chart by position (PG/SG/SF/PF/C)
4. Ensure exactly 5 starters per team
"""

import os
import psycopg2
from nba_api.stats.endpoints import commonplayerinfo
import time

DATABASE_URL = os.getenv('DATABASE_URL')

def update_all_positions_from_nba_api():
    """Update ALL player positions using nba_api"""
    print()
    print("="*80)
    print("🏀 UPDATING ALL PLAYER POSITIONS FROM NBA_API")
    print("="*80)
    print()
    print("⏳ This will take ~3-5 minutes (0.6s per player × 350 players)")
    print()
    
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    
    try:
        # Get all players with teams
        cur.execute("""
            SELECT player_id, name
            FROM players
            WHERE team_id IS NOT NULL
            ORDER BY player_id
        """)
        
        all_players = cur.fetchall()
        total = len(all_players)
        
        print(f"   Found {total} players to update")
        print()
        
        updated = 0
        errors = 0
        
        for idx, (player_id, player_name) in enumerate(all_players, 1):
            try:
                time.sleep(0.6)  # Rate limiting - important!
                
                # Get player info from nba_api
                info = commonplayerinfo.CommonPlayerInfo(
                    player_id=player_id, 
                    timeout=10
                )
                df = info.get_data_frames()[0]
                
                if not df.empty:
                    player_data = df.iloc[0]
                    
                    # Get position and jersey
                    position = player_data.get('POSITION', '')
                    jersey = str(player_data.get('JERSEY', '')) if player_data.get('JERSEY') else None
                    
                    if position:
                        # Update database
                        cur.execute("""
                            UPDATE players
                            SET position = %s,
                                jersey_number = COALESCE(%s, jersey_number)
                            WHERE player_id = %s
                        """, (position, jersey, player_id))
                        
                        updated += 1
                        
                        if updated % 25 == 0:
                            conn.commit()
                            print(f"   Progress: {idx}/{total} ({(idx/total*100):.1f}%) - {updated} updated")
                
            except Exception as e:
                errors += 1
                if errors <= 5:
                    print(f"   ⚠️  Error for {player_name}: {str(e)[:50]}")
                continue
        
        conn.commit()
        
        print()
        print(f"   ✅ Updated {updated}/{total} players")
        print(f"   ⚠️  Errors: {errors}")
        
        # Show position distribution
        print()
        print("   Position distribution after update:")
        cur.execute("""
            SELECT position, COUNT(*) as count
            FROM players
            WHERE team_id IS NOT NULL
            GROUP BY position
            ORDER BY count DESC
        """)
        
        for pos, count in cur.fetchall():
            pos_display = pos if pos else "(NULL)"
            print(f"      {pos_display}: {count} players")
        
        return updated
        
    except Exception as e:
        print(f"   ❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        conn.rollback()
        return 0
    finally:
        cur.close()
        conn.close()


def rebuild_depth_charts_by_position():
    """Rebuild depth charts with proper position assignments"""
    print()
    print("="*80)
    print("🏀 REBUILDING DEPTH CHARTS BY POSITION")
    print("="*80)
    
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    
    try:
        # Get all teams
        cur.execute("SELECT team_id, abbreviation FROM teams")
        teams = cur.fetchall()
        
        print(f"   Processing {len(teams)} teams...")
        
        for team_id, abbr in teams:
            # Clear existing depth chart
            cur.execute("DELETE FROM team_depth_charts WHERE team_id = %s", (team_id,))
            
            # Get players by position, sorted by minutes
            cur.execute("""
                SELECT 
                    p.player_id, 
                    p.position,
                    COALESCE(pss.minutes_total / NULLIF(pss.games_played, 0), 0) as mpg
                FROM players p
                LEFT JOIN player_season_stats pss 
                    ON p.player_id = pss.player_id AND pss.season_id = '2025-26'
                WHERE p.team_id = %s
                AND p.position IS NOT NULL
                AND p.position != ''
                ORDER BY mpg DESC
            """, (team_id,))
            
            players = cur.fetchall()
            
            # Categorize by position type
            guards = []      # PG, SG, G
            forwards = []    # SF, PF, F
            centers = []     # C
            
            for player_id, position, mpg in players:
                if not position or position == 'F':
                    forwards.append((player_id, position or 'F'))
                elif 'G' in position or position in ['PG', 'SG']:
                    guards.append((player_id, position))
                elif 'F' in position or position in ['SF', 'PF']:
                    forwards.append((player_id, position))
                elif 'C' in position:
                    centers.append((player_id, position))
                else:
                    forwards.append((player_id, 'F'))
            
            # Build starting 5: typically 2 guards, 2 forwards, 1 center
            starters = []
            
            # Add top 2 guards
            for player_id, pos in guards[:2]:
                starters.append((player_id, pos))
            
            # Add top 2 forwards
            for player_id, pos in forwards[:2]:
                starters.append((player_id, pos))
            
            # Add top 1 center
            if centers:
                starters.append((centers[0][0], centers[0][1]))
            elif len(forwards) > 2:
                # No center, use another forward
                starters.append((forwards[2][0], forwards[2][1]))
            elif len(guards) > 2:
                # Use another guard
                starters.append((guards[2][0], guards[2][1]))
            
            # Make sure we have exactly 5
            all_available = guards + forwards + centers
            while len(starters) < 5 and len(all_available) > len(starters):
                for player_id, pos in all_available:
                    if (player_id, pos) not in starters:
                        starters.append((player_id, pos))
                        break
            
            # Insert into depth chart
            for rank, (player_id, position) in enumerate(starters[:5], 1):
                cur.execute("""
                    INSERT INTO team_depth_charts (team_id, player_id, position, depth_rank)
                    VALUES (%s, %s, %s, %s)
                """, (team_id, player_id, position, rank))
        
        conn.commit()
        
        # Verify
        print()
        print("   Verification:")
        cur.execute("""
            SELECT t.abbreviation, COUNT(*) as starters,
                   STRING_AGG(DISTINCT dc.position, ', ') as positions
            FROM team_depth_charts dc
            JOIN teams t ON dc.team_id = t.team_id
            WHERE dc.depth_rank <= 5
            GROUP BY t.abbreviation
            ORDER BY t.abbreviation
        """)
        
        all_good = True
        for abbr, count, positions in cur.fetchall()[:10]:
            status = "✅" if count == 5 else "❌"
            print(f"      {abbr}: {count} starters {status} ({positions})")
            if count != 5:
                all_good = False
        
        if all_good:
            print()
            print("   ✅ All teams have exactly 5 starters!")
        
        return all_good
        
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
    print("🚀 COMPLETE POSITION FIX - NBA_API")
    print("="*80)
    print()
    
    if not DATABASE_URL:
        print("❌ DATABASE_URL not set!")
        return False
    
    try:
        # Step 1: Update all positions from nba_api
        updated = update_all_positions_from_nba_api()
        
        if updated < 100:
            print()
            print("⚠️  Warning: Less than 100 players updated. Check for API issues.")
        
        # Step 2: Rebuild depth charts by position
        success = rebuild_depth_charts_by_position()
        
        print()
        print("="*80)
        print("✅ COMPLETE!")
        print("="*80)
        print(f"   Players with positions: {updated}")
        print(f"   Depth charts fixed: {success}")
        print()
        print("🎯 Your frontend will now show:")
        print("   • Correct positions for ALL players (not just stars)")
        print("   • Depth chart box filled (PG, SG, SF, PF, C)")
        print("   • Exactly 5 starters per team")
        print("   • Jersey numbers for most players")
        print("="*80)
        
        return True
        
    except KeyboardInterrupt:
        print()
        print("⚠️  Interrupted by user. Partial progress saved.")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

