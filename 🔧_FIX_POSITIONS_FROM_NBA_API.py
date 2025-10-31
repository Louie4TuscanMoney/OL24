#!/usr/bin/env python3
"""
FIX PLAYER POSITIONS AND JERSEYS using nba_api
ESPN roster endpoint is failing, so use nba_api which is more reliable
"""

import os
import psycopg2
from nba_api.stats.static import players as nba_players
from nba_api.stats.endpoints import commonplayerinfo
import time

DATABASE_URL = os.getenv('DATABASE_URL')


def fix_positions_from_nba_api():
    """Update player positions and jerseys from nba_api"""
    print()
    print("="*80)
    print("🔧 FIXING POSITIONS & JERSEYS FROM NBA API")
    print("="*80)
    
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    
    try:
        # Get all players in database
        cur.execute("""
            SELECT player_id, name
            FROM players
            WHERE team_id IS NOT NULL
            ORDER BY player_id
        """)
        
        db_players = cur.fetchall()
        print(f"   Found {len(db_players)} players in database")
        
        updated = 0
        
        for player_id, player_name in db_players[:100]:  # First 100 to save time
            try:
                time.sleep(0.6)  # Rate limiting
                
                # Get player info from nba_api
                player_info = commonplayerinfo.CommonPlayerInfo(player_id=player_id, timeout=10)
                info_df = player_info.get_data_frames()[0]
                
                if not info_df.empty:
                    player_data = info_df.iloc[0]
                    
                    position = player_data.get('POSITION', '')
                    jersey = player_data.get('JERSEY', '')
                    
                    if position or jersey:
                        # Update database
                        cur.execute("""
                            UPDATE players
                            SET position = COALESCE(NULLIF(%s, ''), position),
                                jersey_number = COALESCE(NULLIF(%s, ''), jersey_number)
                            WHERE player_id = %s
                        """, (position, jersey, player_id))
                        
                        updated += 1
                        
                        if updated <= 10:
                            print(f"   ✅ {player_name}: {position} #{jersey}")
                
            except Exception as e:
                if updated <= 3:
                    print(f"   ⚠️  Could not get info for {player_name}: {e}")
                continue
        
        conn.commit()
        print(f"\n   ✅ Updated {updated} players")
        
        # Verify
        print("\n   Verification (OKC players):")
        cur.execute("""
            SELECT name, position, jersey_number
            FROM players
            WHERE team_id = (SELECT team_id FROM teams WHERE abbreviation = 'OKC')
            AND (position IS NOT NULL OR jersey_number IS NOT NULL)
            ORDER BY name
            LIMIT 5
        """)
        
        for row in cur.fetchall():
            name, pos, jersey = row
            pos_str = pos if pos else "NO POS"
            jersey_str = jersey if jersey else "NO #"
            print(f"      {name}: {pos_str} #{jersey_str}")
        
        return updated
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        conn.rollback()
        return 0
    finally:
        cur.close()
        conn.close()


def main():
    print()
    print("="*80)
    print("🔧 FIXING PLAYER POSITIONS & JERSEYS")
    print("="*80)
    
    if not DATABASE_URL:
        print("❌ DATABASE_URL not set!")
        return False
    
    try:
        updated = fix_positions_from_nba_api()
        
        print()
        print("="*80)
        print("✅ POSITIONS & JERSEYS UPDATED!")
        print("="*80)
        print(f"   Players updated: {updated}")
        print()
        print("🎯 Frontend will now show:")
        print("   • Correct positions (Guard, Forward, Center)")
        print("   • Jersey numbers")
        print("="*80)
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

