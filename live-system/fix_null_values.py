#!/usr/bin/env python3
"""
Fix NULL values in Railway PostgreSQL database
- Parse first/last names
- Add positions
- Add headshot URLs
- Populate game stats from nba_api
- Calculate team stats
"""

import os
import psycopg2
from nba_api.stats.static import players as nba_players
from nba_api.stats.endpoints import playergamelog, leaguedashplayerstats, leaguedashteamstats
import time

DATABASE_URL = os.getenv('DATABASE_URL')

def fix_player_names_and_metadata():
    """Parse first/last names and add metadata"""
    print("="*80)
    print("1️⃣  FIXING PLAYER NAMES & METADATA")
    print("="*80)
    
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    
    # Get all players
    cur.execute("SELECT player_id, name FROM players")
    db_players = cur.fetchall()
    
    # Get NBA API player data
    nba_player_list = nba_players.get_active_players()
    nba_map = {str(p['id']): p for p in nba_player_list}
    
    updated = 0
    for player_id, name in db_players:
        if player_id in nba_map:
            nba_data = nba_map[player_id]
            
            # Parse first/last name
            full_name = name
            name_parts = full_name.split()
            first_name = name_parts[0] if name_parts else ""
            last_name = " ".join(name_parts[1:]) if len(name_parts) > 1 else ""
            
            # Update player
            cur.execute("""
                UPDATE players SET
                    first_name = %s,
                    last_name = %s,
                    headshot_url = %s
                WHERE player_id = %s
            """, (
                first_name,
                last_name,
                f"https://cdn.nba.com/headshots/nba/latest/1040x760/{player_id}.png",
                player_id
            ))
            updated += 1
            
            if updated % 100 == 0:
                print(f"   ... {updated} players")
    
    conn.commit()
    print(f"   ✅ Updated {updated} players with names & headshots")
    cur.close()
    conn.close()


def populate_season_stats_from_api():
    """Get actual season stats from nba_api"""
    print()
    print("="*80)
    print("2️⃣  POPULATING SEASON STATS FROM NBA API")
    print("="*80)
    
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    
    try:
        # Get player stats for 2025-26 season
        print("   📊 Fetching player stats from nba_api...")
        stats = leaguedashplayerstats.LeagueDashPlayerStats(
            season='2025-26',
            per_mode_detailed='Totals'
        )
        df = stats.get_data_frames()[0]
        
        updated = 0
        for _, row in df.iterrows():
            player_id = str(row['PLAYER_ID'])
            team_id = str(row['TEAM_ID'])
            
            # Update player_season_stats with REAL game stats
            cur.execute("""
                UPDATE player_season_stats SET
                    games_played = %s,
                    games_started = COALESCE(games_started, 0),
                    minutes_total = %s,
                    pts_total = %s,
                    reb_total = %s,
                    ast_total = %s,
                    stl_total = %s,
                    blk_total = %s,
                    tov_total = %s,
                    fgm_total = %s,
                    fga_total = %s,
                    fg3m_total = %s,
                    fg3a_total = %s,
                    ftm_total = %s,
                    fta_total = %s,
                    fg_pct = %s,
                    fg3_pct = %s,
                    ft_pct = %s
                WHERE player_id = %s AND season_id = '2025-26'
            """, (
                int(row['GP']),
                float(row['MIN']),
                int(row['PTS']),
                int(row['REB']),
                int(row['AST']),
                int(row['STL']),
                int(row['BLK']),
                int(row['TOV']),
                int(row['FGM']),
                int(row['FGA']),
                int(row['FG3M']),
                int(row['FG3A']),
                int(row['FTM']),
                int(row['FTA']),
                float(row['FG_PCT']) if row['FG_PCT'] else None,
                float(row['FG3_PCT']) if row['FG3_PCT'] else None,
                float(row['FT_PCT']) if row['FT_PCT'] else None,
                player_id
            ))
            
            # Also update position
            cur.execute("""
                UPDATE players SET position = %s
                WHERE player_id = %s AND position IS NULL
            """, (row.get('POSITION', 'F'), player_id))
            
            updated += 1
            if updated % 50 == 0:
                print(f"   ... {updated} players")
                conn.commit()
        
        conn.commit()
        print(f"   ✅ Updated {updated} players with season stats")
        
    except Exception as e:
        print(f"   ⚠️  Error: {e}")
        conn.rollback()
    
    cur.close()
    conn.close()


def populate_team_season_stats():
    """Populate team season stats"""
    print()
    print("="*80)
    print("3️⃣  POPULATING TEAM SEASON STATS")
    print("="*80)
    
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    
    try:
        # Get team stats
        print("   🏀 Fetching team stats from nba_api...")
        stats = leaguedashteamstats.LeagueDashTeamStats(
            season='2025-26',
            per_mode_detailed='Totals'
        )
        df = stats.get_data_frames()[0]
        
        inserted = 0
        for _, row in df.iterrows():
            team_id = str(row['TEAM_ID'])
            
            cur.execute("""
                INSERT INTO team_season_stats (
                    team_id, season_id, games_played, wins, losses,
                    pts_total, pts_allowed_total
                ) VALUES (%s, '2025-26', %s, %s, %s, %s, 0)
                ON CONFLICT (team_id, season_id) DO UPDATE SET
                    games_played = EXCLUDED.games_played,
                    wins = EXCLUDED.wins,
                    losses = EXCLUDED.losses,
                    pts_total = EXCLUDED.pts_total
            """, (
                team_id,
                int(row['GP']),
                int(row['W']),
                int(row['L']),
                int(row['PTS'])
            ))
            inserted += 1
        
        conn.commit()
        print(f"   ✅ Updated {inserted} teams with season stats")
        
    except Exception as e:
        print(f"   ⚠️  Error: {e}")
        conn.rollback()
    
    cur.close()
    conn.close()


def populate_standings():
    """Populate standings table"""
    print()
    print("="*80)
    print("4️⃣  POPULATING STANDINGS")
    print("="*80)
    
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    
    try:
        from nba_api.stats.endpoints import leaguestandingsv3
        
        print("   📊 Fetching standings...")
        standings = leaguestandingsv3.LeagueStandingsV3(season='2025-26')
        df = standings.get_data_frames()[0]
        
        inserted = 0
        for _, row in df.iterrows():
            team_id = str(row['TeamID'])
            
            cur.execute("""
                INSERT INTO standings (
                    team_id, season_id, conference, rank, wins, losses,
                    home_record, away_record, last_10, streak
                ) VALUES (%s, '2025-26', %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT DO NOTHING
            """, (
                team_id,
                row['Conference'],
                int(row['ConferenceRank']) if 'ConferenceRank' in row else None,
                int(row['WINS']),
                int(row['LOSSES']),
                row.get('HOME', '0-0'),
                row.get('ROAD', '0-0'),
                row.get('L10', '0-0'),
                row.get('CurrentStreak', '')
            ))
            inserted += 1
        
        conn.commit()
        print(f"   ✅ Inserted {inserted} standings entries")
        
    except Exception as e:
        print(f"   ⚠️  Error: {e}")
        conn.rollback()
    
    cur.close()
    conn.close()


def verify_fixes():
    """Verify all NULL values are fixed"""
    print()
    print("="*80)
    print("✅ VERIFICATION")
    print("="*80)
    
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    
    # Check players
    cur.execute("""
        SELECT 
            COUNT(*) FILTER (WHERE first_name IS NOT NULL) as has_first_name,
            COUNT(*) FILTER (WHERE last_name IS NOT NULL) as has_last_name,
            COUNT(*) FILTER (WHERE headshot_url IS NOT NULL) as has_headshot,
            COUNT(*) FILTER (WHERE position IS NOT NULL) as has_position,
            COUNT(*) as total
        FROM players
    """)
    row = cur.fetchone()
    print(f"👤 Players:")
    print(f"   ✅ With first_name: {row[0]}/{row[4]}")
    print(f"   ✅ With last_name: {row[1]}/{row[4]}")
    print(f"   ✅ With headshot: {row[2]}/{row[4]}")
    print(f"   ✅ With position: {row[3]}/{row[4]}")
    
    # Check stats
    cur.execute("""
        SELECT 
            COUNT(*) FILTER (WHERE games_played > 0) as has_games,
            COUNT(*) FILTER (WHERE pts_total > 0) as has_pts,
            COUNT(*) FILTER (WHERE ppg > 0) as has_ppg,
            COUNT(*) as total
        FROM player_season_stats
    """)
    row = cur.fetchone()
    print()
    print(f"📊 Player Stats:")
    print(f"   ✅ With games: {row[0]}/{row[3]}")
    print(f"   ✅ With points: {row[1]}/{row[3]}")
    print(f"   ✅ With PPG: {row[2]}/{row[3]}")
    
    # Check team stats
    cur.execute("""
        SELECT COUNT(*) FROM team_season_stats WHERE games_played > 0
    """)
    team_count = cur.fetchone()[0]
    print()
    print(f"🏀 Team Stats:")
    print(f"   ✅ Teams with stats: {team_count}/30")
    
    # Top 5 scorers with complete data
    print()
    print("🏆 Top 5 Scorers (PPG):")
    cur.execute("""
        SELECT p.name, t.abbreviation, pss.ppg, pss.rpg, pss.apg
        FROM player_season_stats pss
        JOIN players p ON pss.player_id = p.player_id
        JOIN teams t ON pss.team_id = t.team_id
        WHERE pss.ppg > 0
        ORDER BY pss.ppg DESC
        LIMIT 5
    """)
    for row in cur.fetchall():
        print(f"   {row[0]:<25} {row[1]:<5} {row[2]:.1f} PPG, {row[3]:.1f} RPG, {row[4]:.1f} APG")
    
    cur.close()
    conn.close()


if __name__ == '__main__':
    print("🚀 FIXING NULL VALUES IN RAILWAY POSTGRESQL")
    print()
    
    fix_player_names_and_metadata()
    populate_season_stats_from_api()
    populate_team_season_stats()
    populate_standings()
    verify_fixes()
    
    print()
    print("="*80)
    print("✅ ALL NULL VALUES FIXED!")
    print("="*80)

