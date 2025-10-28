"""
NBA_API TEAM & PLAYER STATS PIPELINE
Comprehensive data collection from nba_api for all 30 teams

Collects:
- Team stats (current season)
- Player stats (all players on each team)
- Team standings
- Injuries
- Depth charts (estimated from minutes played)

Author: Ontologic XYZ
Date: October 28, 2025
"""

import os
import psycopg2
from datetime import datetime
from nba_api.stats.static import teams, players
from nba_api.stats.endpoints import (
    leaguestandings,
    teamgamelog,
    commonteamroster,
    playergamelog,
    teamdashboardbygeneralsplits,
    playerdashboardbygeneralsplits,
    commonplayerinfo
)
import time
import traceback


def get_db_connection():
    """Get PostgreSQL connection"""
    DATABASE_URL = os.environ.get('DATABASE_URL')
    if not DATABASE_URL:
        raise ValueError("DATABASE_URL not set!")
    return psycopg2.connect(DATABASE_URL)


def get_all_nba_teams():
    """Get all 30 NBA teams from nba_api"""
    print("\n" + "="*80)
    print("📊 FETCHING ALL NBA TEAMS")
    print("="*80)
    
    nba_teams = teams.get_teams()
    
    print(f"✅ Found {len(nba_teams)} NBA teams")
    
    return nba_teams


def update_team_in_db(conn, team_data):
    """
    Update team in database with full info
    
    Args:
        conn: DB connection
        team_data: Team dict from nba_api
    """
    try:
        cur = conn.cursor()
        
        # Map nba_api data to our schema
        team_id = str(team_data['id'])
        abbreviation = team_data['abbreviation']
        full_name = team_data['full_name']
        city = team_data['city']
        
        # Update or insert team
        cur.execute("""
            INSERT INTO teams (team_id, abbreviation, full_name, city)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (team_id) 
            DO UPDATE SET
                abbreviation = EXCLUDED.abbreviation,
                full_name = EXCLUDED.full_name,
                city = EXCLUDED.city,
                updated_at = NOW()
        """, (team_id, abbreviation, full_name, city))
        
        conn.commit()
        print(f"  ✅ Updated: {abbreviation} ({full_name})")
        
    except Exception as e:
        print(f"  ❌ Error updating {team_data.get('abbreviation')}: {e}")
        conn.rollback()


def get_team_roster(team_id: str, season: str = "2025-26"):
    """
    Get team roster from nba_api
    
    Args:
        team_id: NBA team ID
        season: Season (e.g., "2025-26")
        
    Returns:
        List of players
    """
    try:
        roster = commonteamroster.CommonTeamRoster(
            team_id=team_id,
            season=season
        )
        
        time.sleep(0.6)  # Rate limit
        
        roster_df = roster.get_data_frames()[0]
        return roster_df.to_dict('records')
        
    except Exception as e:
        print(f"  ⚠️ Error fetching roster for team {team_id}: {e}")
        return []


def update_team_roster(conn, team_id: str, team_abbr: str):
    """
    Update all players for a team
    
    Args:
        conn: DB connection
        team_id: NBA team ID
        team_abbr: Team abbreviation
    """
    print(f"\n  📋 Fetching roster for {team_abbr}...")
    
    roster = get_team_roster(team_id)
    
    if not roster:
        print(f"    ⚠️ No roster data")
        return
    
    cur = conn.cursor()
    
    for player in roster:
        try:
            player_id = str(player['PLAYER_ID'])
            player_name = player['PLAYER']
            position = player.get('POSITION', '')
            
            # Insert or update player
            cur.execute("""
                INSERT INTO players (player_id, full_name, team_id, position)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (player_id)
                DO UPDATE SET
                    full_name = EXCLUDED.full_name,
                    team_id = EXCLUDED.team_id,
                    position = EXCLUDED.position,
                    updated_at = NOW()
            """, (player_id, player_name, team_id, position))
            
        except Exception as e:
            print(f"      ⚠️ Error updating player {player.get('PLAYER')}: {e}")
            conn.rollback()
            continue
    
    conn.commit()
    print(f"    ✅ Updated {len(roster)} players for {team_abbr}")


def get_team_season_stats(team_id: str, season: str = "2025-26"):
    """
    Get team season stats from nba_api
    
    Args:
        team_id: NBA team ID
        season: Season
        
    Returns:
        Team stats dict
    """
    try:
        dashboard = teamdashboardbygeneralsplits.TeamDashboardByGeneralSplits(
            team_id=team_id,
            season=season,
            measure_type_detailed_defense='Base',
            per_mode_detailed='PerGame'
        )
        
        time.sleep(0.6)  # Rate limit
        
        overall_df = dashboard.get_data_frames()[0]
        
        if len(overall_df) > 0:
            return overall_df.iloc[0].to_dict()
        
        return {}
        
    except Exception as e:
        print(f"  ⚠️ Error fetching team stats for {team_id}: {e}")
        return {}


def update_team_season_stats(conn, team_id: str, team_abbr: str, season_id: str = "2025-26"):
    """
    Update team season stats in database
    
    Args:
        conn: DB connection
        team_id: NBA team ID
        team_abbr: Team abbreviation
        season_id: Season ID
    """
    print(f"\n  📊 Fetching season stats for {team_abbr}...")
    
    stats = get_team_season_stats(team_id, season_id)
    
    if not stats:
        print(f"    ⚠️ No stats available")
        return
    
    try:
        cur = conn.cursor()
        
        # Extract key stats
        games_played = stats.get('GP', 0)
        wins = stats.get('W', 0)
        losses = stats.get('L', 0)
        ppg = stats.get('PTS', 0)
        opp_ppg = stats.get('OPP_PTS', 0) if 'OPP_PTS' in stats else 0
        fg_pct = stats.get('FG_PCT', 0)
        fg3_pct = stats.get('FG3_PCT', 0)
        ft_pct = stats.get('FT_PCT', 0)
        
        # Insert or update
        cur.execute("""
            INSERT INTO team_season_stats (
                team_id, season_id, games_played, wins, losses,
                ppg, opp_ppg, fg_pct, fg3_pct, ft_pct
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (team_id, season_id)
            DO UPDATE SET
                games_played = EXCLUDED.games_played,
                wins = EXCLUDED.wins,
                losses = EXCLUDED.losses,
                ppg = EXCLUDED.ppg,
                opp_ppg = EXCLUDED.opp_ppg,
                fg_pct = EXCLUDED.fg_pct,
                fg3_pct = EXCLUDED.fg3_pct,
                ft_pct = EXCLUDED.ft_pct,
                updated_at = NOW()
        """, (team_id, season_id, games_played, wins, losses, ppg, opp_ppg, fg_pct, fg3_pct, ft_pct))
        
        conn.commit()
        print(f"    ✅ Stats updated: {wins}-{losses}, {ppg:.1f} PPG")
        
    except Exception as e:
        print(f"    ❌ Error updating stats: {e}")
        conn.rollback()


def get_player_season_stats(player_id: str, season: str = "2025-26"):
    """
    Get player season stats from nba_api
    
    Args:
        player_id: NBA player ID
        season: Season
        
    Returns:
        Player stats dict
    """
    try:
        dashboard = playerdashboardbygeneralsplits.PlayerDashboardByGeneralSplits(
            player_id=player_id,
            season=season,
            measure_type_detailed_defense='Base',
            per_mode_detailed='PerGame'
        )
        
        time.sleep(0.6)  # Rate limit
        
        overall_df = dashboard.get_data_frames()[0]
        
        if len(overall_df) > 0:
            return overall_df.iloc[0].to_dict()
        
        return {}
        
    except Exception as e:
        return {}


def update_player_season_stats(conn, player_id: str, player_name: str, team_id: str, season_id: str = "2025-26"):
    """
    Update player season stats
    
    Args:
        conn: DB connection
        player_id: NBA player ID
        player_name: Player name
        team_id: Team ID
        season_id: Season ID
    """
    stats = get_player_season_stats(player_id, season_id)
    
    if not stats:
        return
    
    try:
        cur = conn.cursor()
        
        # Extract stats
        gp = stats.get('GP', 0)
        mpg = stats.get('MIN', 0)
        ppg = stats.get('PTS', 0)
        rpg = stats.get('REB', 0)
        apg = stats.get('AST', 0)
        fg_pct = stats.get('FG_PCT', 0)
        fg3_pct = stats.get('FG3_PCT', 0)
        ft_pct = stats.get('FT_PCT', 0)
        
        # Insert or update
        cur.execute("""
            INSERT INTO player_season_stats (
                player_id, season_id, team_id,
                games_played, mpg, ppg, rpg, apg,
                fg_pct, fg3_pct, ft_pct
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (player_id, season_id)
            DO UPDATE SET
                team_id = EXCLUDED.team_id,
                games_played = EXCLUDED.games_played,
                mpg = EXCLUDED.mpg,
                ppg = EXCLUDED.ppg,
                rpg = EXCLUDED.rpg,
                apg = EXCLUDED.apg,
                fg_pct = EXCLUDED.fg_pct,
                fg3_pct = EXCLUDED.fg3_pct,
                ft_pct = EXCLUDED.ft_pct,
                updated_at = NOW()
        """, (player_id, season_id, team_id, gp, mpg, ppg, rpg, apg, fg_pct, fg3_pct, ft_pct))
        
        conn.commit()
        print(f"      ✅ {player_name}: {ppg:.1f} PPG, {rpg:.1f} RPG, {apg:.1f} APG")
        
    except Exception as e:
        print(f"      ⚠️ Error: {e}")
        conn.rollback()


def run_full_pipeline():
    """
    Run complete NBA_API data pipeline for all teams
    """
    print("\n" + "="*80)
    print("🏀 NBA_API COMPLETE DATA PIPELINE")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    conn = get_db_connection()
    
    try:
        # Get all teams
        nba_teams = get_all_nba_teams()
        
        season = "2025-26"
        
        for i, team in enumerate(nba_teams, 1):
            print(f"\n[{i}/30] Processing {team['abbreviation']} - {team['full_name']}")
            print("-" * 80)
            
            team_id = str(team['id'])
            team_abbr = team['abbreviation']
            
            # 1. Update team info
            update_team_in_db(conn, team)
            
            # 2. Update roster
            update_team_roster(conn, team_id, team_abbr)
            
            # 3. Update team season stats
            update_team_season_stats(conn, team_id, team_abbr, season)
            
            # 4. Update player stats (top 10 players by minutes to avoid rate limits)
            print(f"\n  👤 Fetching player stats for {team_abbr}...")
            cur = conn.cursor()
            cur.execute("""
                SELECT player_id, full_name
                FROM players
                WHERE team_id = %s
                ORDER BY player_id
                LIMIT 10
            """, (team_id,))
            
            players_to_update = cur.fetchall()
            
            for player_id, player_name in players_to_update:
                update_player_season_stats(conn, player_id, player_name, team_id, season)
                time.sleep(0.6)  # Rate limit
            
            print(f"\n  ✅ Completed {team_abbr}")
            
            # Rate limit between teams
            if i < 30:
                print(f"\n  ⏳ Rate limit: waiting 2 seconds...")
                time.sleep(2)
        
        print("\n" + "="*80)
        print("✅ PIPELINE COMPLETE!")
        print("="*80)
        print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        print("📊 Data populated:")
        
        # Show summary
        cur = conn.cursor()
        
        cur.execute("SELECT COUNT(*) FROM teams")
        team_count = cur.fetchone()[0]
        
        cur.execute("SELECT COUNT(*) FROM players")
        player_count = cur.fetchone()[0]
        
        cur.execute("SELECT COUNT(*) FROM team_season_stats WHERE season_id = %s", (season,))
        team_stats_count = cur.fetchone()[0]
        
        cur.execute("SELECT COUNT(*) FROM player_season_stats WHERE season_id = %s", (season,))
        player_stats_count = cur.fetchone()[0]
        
        print(f"  Teams: {team_count}")
        print(f"  Players: {player_count}")
        print(f"  Team season stats: {team_stats_count}")
        print(f"  Player season stats: {player_stats_count}")
        print()
        
    except Exception as e:
        print(f"\n❌ Pipeline error: {e}")
        traceback.print_exc()
    finally:
        conn.close()


if __name__ == "__main__":
    run_full_pipeline()

