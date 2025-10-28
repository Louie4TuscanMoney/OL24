"""
POPULATE PLAYER IMAGES - Uses NBA CDN
Adds headshots and team logos to database
"""

import os
import psycopg2

DATABASE_URL = os.environ.get('DATABASE_URL')

# NBA CDN URL patterns
def get_player_headshot_url(player_id):
    """NBA official player headshot"""
    return f"https://cdn.nba.com/headshots/nba/latest/1040x760/{player_id}.png"

def get_team_logo_url(team_id):
    """NBA official team logo"""
    return f"https://cdn.nba.com/logos/nba/{team_id}/primary/L/logo.svg"

# Team colors (official NBA brand colors)
TEAM_COLORS = {
    'ATL': {'primary': '#E03A3E', 'secondary': '#C1D32F'},
    'BOS': {'primary': '#007A33', 'secondary': '#BA9653'},
    'BKN': {'primary': '#000000', 'secondary': '#FFFFFF'},
    'CHA': {'primary': '#1D1160', 'secondary': '#00788C'},
    'CHI': {'primary': '#CE1141', 'secondary': '#000000'},
    'CLE': {'primary': '#860038', 'secondary': '#FDBB30'},
    'DAL': {'primary': '#00538C', 'secondary': '#002B5E'},
    'DEN': {'primary': '#0E2240', 'secondary': '#FEC524'},
    'DET': {'primary': '#C8102E', 'secondary': '#1D42BA'},
    'GSW': {'primary': '#1D428A', 'secondary': '#FFC72C'},
    'HOU': {'primary': '#CE1141', 'secondary': '#000000'},
    'IND': {'primary': '#002D62', 'secondary': '#FDBB30'},
    'LAC': {'primary': '#C8102E', 'secondary': '#1D428A'},
    'LAL': {'primary': '#552583', 'secondary': '#FDB927'},
    'MEM': {'primary': '#5D76A9', 'secondary': '#12173F'},
    'MIA': {'primary': '#98002E', 'secondary': '#F9A01B'},
    'MIL': {'primary': '#00471B', 'secondary': '#EEE1C6'},
    'MIN': {'primary': '#0C2340', 'secondary': '#236192'},
    'NOP': {'primary': '#0C2340', 'secondary': '#C8102E'},
    'NYK': {'primary': '#006BB6', 'secondary': '#F58426'},
    'OKC': {'primary': '#007AC1', 'secondary': '#EF3B24'},
    'ORL': {'primary': '#0077C0', 'secondary': '#C4CED4'},
    'PHI': {'primary': '#006BB6', 'secondary': '#ED174C'},
    'PHX': {'primary': '#1D1160', 'secondary': '#E56020'},
    'POR': {'primary': '#E03A3E', 'secondary': '#000000'},
    'SAC': {'primary': '#5A2D81', 'secondary': '#63727A'},
    'SAS': {'primary': '#C4CED4', 'secondary': '#000000'},
    'TOR': {'primary': '#CE1141', 'secondary': '#000000'},
    'UTA': {'primary': '#002B5C', 'secondary': '#F9A01B'},
    'WAS': {'primary': '#002B5C', 'secondary': '#E31837'}
}


def populate_images():
    """
    Add image URLs and colors to database
    """
    if not DATABASE_URL:
        print("❌ DATABASE_URL not set!")
        return
    
    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()
    
    print("\n" + "="*80)
    print("🖼️  POPULATING IMAGES + COLORS")
    print("="*80 + "\n")
    
    # 1. Update team logos + colors
    print("📊 Updating team visuals...")
    cursor.execute("SELECT team_id, abbreviation FROM teams")
    teams = cursor.fetchall()
    
    for team_id, abbr in teams:
        logo_url = get_team_logo_url(team_id)
        colors = TEAM_COLORS.get(abbr, {'primary': '#000000', 'secondary': '#FFFFFF'})
        
        cursor.execute("""
            UPDATE teams
            SET logo_url = %s,
                primary_color = %s,
                secondary_color = %s,
                updated_at = NOW()
            WHERE team_id = %s
        """, (logo_url, colors['primary'], colors['secondary'], team_id))
    
    print(f"   ✅ {len(teams)} team logos + colors\n")
    
    # 2. Update player headshots
    print("📊 Updating player images...")
    cursor.execute("SELECT player_id FROM players WHERE is_active = TRUE")
    players = cursor.fetchall()
    
    for (player_id,) in players:
        headshot_url = get_player_headshot_url(player_id)
        
        cursor.execute("""
            UPDATE players
            SET headshot_url = %s,
                updated_at = NOW()
            WHERE player_id = %s
        """, (headshot_url, player_id))
    
    print(f"   ✅ {len(players)} player headshots\n")
    
    conn.commit()
    conn.close()
    
    print("="*80)
    print("✅ IMAGES POPULATED!")
    print("="*80 + "\n")


if __name__ == "__main__":
    populate_images()

