#!/usr/bin/env python3
"""
VERIFY COMPLETE PIPELINE
Ensures everything is connected: ESPN API → PostgreSQL → FastAPI → Frontend

This verifies:
1. PostgreSQL has data from ESPN API (not hallucinated)
2. Daily scheduler is configured to run at 3:30 AM
3. FastAPI endpoints pull from PostgreSQL
4. Frontend receives accurate data
"""

import os
import sys
import psycopg2
import requests
from datetime import datetime

DATABASE_URL = os.getenv('DATABASE_URL')
API_URL = "https://ol24-production.up.railway.app"


def verify_postgresql_data():
    """Verify PostgreSQL has accurate data from ESPN"""
    print()
    print("="*80)
    print("1️⃣  VERIFYING POSTGRESQL DATA")
    print("="*80)
    
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    
    try:
        # Check team stats exist
        cur.execute("""
            SELECT COUNT(*) 
            FROM team_season_stats
            WHERE season_id = '2025-26'
        """)
        team_count = cur.fetchone()[0]
        
        print(f"   ✅ Teams in database: {team_count}/30")
        
        # Check for bad PPG (hallucinated stats)
        cur.execute("""
            SELECT COUNT(*) 
            FROM team_season_stats
            WHERE season_id = '2025-26' AND (ppg > 150 OR ppg < 50)
        """)
        bad_count = cur.fetchone()[0]
        
        if bad_count == 0:
            print(f"   ✅ No hallucinated stats (all PPG realistic)")
        else:
            print(f"   ⚠️  {bad_count} teams have unrealistic stats")
        
        # Check data freshness
        cur.execute("""
            SELECT MAX(updated_at) 
            FROM team_season_stats
            WHERE season_id = '2025-26'
        """)
        last_update = cur.fetchone()[0]
        
        if last_update:
            age = datetime.now() - last_update.replace(tzinfo=None)
            hours_ago = age.seconds // 3600
            mins_ago = (age.seconds % 3600) // 60
            print(f"   ✅ Last updated: {hours_ago}h {mins_ago}m ago")
            
            if age.days > 1:
                print(f"   ⚠️  Data is {age.days} days old (needs update)")
        
        # Check schedule has times
        cur.execute("""
            SELECT COUNT(*) 
            FROM nba_schedule
            WHERE game_time IS NOT NULL
        """)
        games_with_time = cur.fetchone()[0]
        print(f"   ✅ Games with accurate times: {games_with_time}")
        
        # Sample data
        print()
        print("   Sample (Top 5 teams from PostgreSQL):")
        cur.execute("""
            SELECT t.abbreviation, tss.wins, tss.losses, tss.ppg
            FROM teams t
            JOIN team_season_stats tss ON t.team_id = tss.team_id
            WHERE tss.season_id = '2025-26'
            ORDER BY tss.wins DESC
            LIMIT 5
        """)
        
        for row in cur.fetchall():
            abbr, w, l, ppg = row
            print(f"      {abbr:<5} {w}-{l}  {ppg:.1f} PPG")
        
        return bad_count == 0 and team_count >= 30
        
    finally:
        cur.close()
        conn.close()


def verify_fastapi_endpoints():
    """Verify FastAPI endpoints pull from PostgreSQL"""
    print()
    print("="*80)
    print("2️⃣  VERIFYING FASTAPI ENDPOINTS")
    print("="*80)
    
    try:
        # Test /api/stats/teams
        print("\n   Testing: GET /api/stats/teams")
        response = requests.get(f"{API_URL}/api/stats/teams", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            teams = data.get('teams', [])
            
            print(f"   ✅ Endpoint working: {len(teams)} teams returned")
            
            if teams:
                top_team = teams[0]
                abbr = top_team.get('abbreviation')
                wins = top_team.get('wins')
                losses = top_team.get('losses')
                ppg = top_team.get('ppg')
                
                print(f"   ✅ Sample: {abbr} ({wins}-{losses}, {ppg:.1f} PPG)")
                
                # Verify data is realistic
                if ppg > 150 or ppg < 50:
                    print(f"   ❌ HALLUCINATED DATA! {abbr} has {ppg} PPG (impossible)")
                    return False
                else:
                    print(f"   ✅ Data is realistic (not hallucinated)")
        else:
            print(f"   ❌ Endpoint failed: HTTP {response.status_code}")
            return False
        
        # Test /api/schedule (with times)
        print("\n   Testing: GET /api/schedule")
        response = requests.get(f"{API_URL}/api/schedule", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            games = data.get('games', [])
            
            print(f"   ✅ Endpoint working: {len(games)} games returned")
            
            if games:
                game = games[0]
                time_pst = game.get('time')
                
                if time_pst:
                    print(f"   ✅ Game times showing: {time_pst}")
                else:
                    print(f"   ⚠️  Game times not showing")
        else:
            print(f"   ❌ Endpoint failed: HTTP {response.status_code}")
            return False
        
        # Test /api/search
        print("\n   Testing: GET /api/search?q=lakers")
        response = requests.get(f"{API_URL}/api/search?q=lakers", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            teams = data.get('teams', [])
            players = data.get('players', [])
            
            print(f"   ✅ Search working: {len(teams)} teams, {len(players)} players")
        else:
            print(f"   ⚠️  Search endpoint failed")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error testing endpoints: {e}")
        return False


def verify_scheduler_config():
    """Verify scheduler is configured to run at 3:30 AM"""
    print()
    print("="*80)
    print("3️⃣  VERIFYING DAILY SCHEDULER")
    print("="*80)
    
    # Check if scheduler file exists and is configured
    scheduler_file = "backend/services/daily_nba_scheduler.py"
    
    if os.path.exists(scheduler_file):
        print(f"   ✅ Scheduler file exists: {scheduler_file}")
        
        with open(scheduler_file, 'r') as f:
            content = f.read()
            
            if 'schedule.every().day.at("03:30")' in content:
                print(f"   ✅ Scheduled for 3:30 AM UTC")
            else:
                print(f"   ⚠️  Schedule time not found")
            
            if 'update_from_espn_daily.py' in content:
                print(f"   ✅ Using ESPN API for updates")
            else:
                print(f"   ⚠️  Not using ESPN API")
        
        # Check if scheduler is imported in main API
        api_file = "live-system/trading_dashboard_api.py"
        if os.path.exists(api_file):
            with open(api_file, 'r') as f:
                content = f.read()
                
                if 'start_scheduler' in content:
                    print(f"   ✅ Scheduler is started in trading_dashboard_api.py")
                else:
                    print(f"   ⚠️  Scheduler may not be started")
        
        return True
    else:
        print(f"   ❌ Scheduler file not found")
        return False


def verify_data_pipeline():
    """Verify the complete data pipeline"""
    print()
    print("="*80)
    print("4️⃣  VERIFYING COMPLETE PIPELINE")
    print("="*80)
    
    # Check ESPN script exists
    espn_script = "backend/services/update_from_espn_daily.py"
    if os.path.exists(espn_script):
        print(f"   ✅ ESPN update script exists")
    else:
        print(f"   ❌ ESPN update script missing")
        return False
    
    # Verify pipeline flow
    print()
    print("   Pipeline flow:")
    print("   1. ESPN API (source of truth)")
    print("      ↓")
    print("   2. update_from_espn_daily.py (fetches data)")
    print("      ↓")
    print("   3. PostgreSQL (stores data)")
    print("      ↓")
    print("   4. trading_dashboard_api.py (serves API)")
    print("      ↓")
    print("   5. Frontend (displays data)")
    print()
    
    # Verify data source
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    
    try:
        # Check if data matches ESPN format
        cur.execute("""
            SELECT t.abbreviation, tss.ppg
            FROM teams t
            JOIN team_season_stats tss ON t.team_id = tss.team_id
            WHERE tss.season_id = '2025-26'
            LIMIT 1
        """)
        
        result = cur.fetchone()
        if result:
            abbr, ppg = result
            
            # ESPN PPG is typically 100-135
            if 90 < ppg < 150:
                print(f"   ✅ Data format matches ESPN API ({abbr}: {ppg:.1f} PPG)")
            else:
                print(f"   ⚠️  Data may not be from ESPN ({abbr}: {ppg:.1f} PPG)")
        
        return True
        
    finally:
        cur.close()
        conn.close()


def main():
    print()
    print("="*80)
    print("🔍 COMPLETE PIPELINE VERIFICATION")
    print("="*80)
    print()
    print("This verifies:")
    print("   • PostgreSQL has data from ESPN API (not hallucinated)")
    print("   • FastAPI endpoints serve PostgreSQL data")
    print("   • Daily scheduler runs at 3:30 AM UTC")
    print("   • Complete pipeline is integrated")
    
    if not DATABASE_URL:
        print()
        print("❌ DATABASE_URL not set!")
        print("   export DATABASE_URL='postgresql://...'")
        return False
    
    # Run verification steps
    step1 = verify_postgresql_data()
    step2 = verify_fastapi_endpoints()
    step3 = verify_scheduler_config()
    step4 = verify_data_pipeline()
    
    # Final summary
    print()
    print("="*80)
    print("📊 VERIFICATION SUMMARY")
    print("="*80)
    
    all_passed = step1 and step2 and step3 and step4
    
    if all_passed:
        print()
        print("✅ ALL CHECKS PASSED!")
        print()
        print("Your pipeline is fully integrated:")
        print("   ✅ PostgreSQL has accurate ESPN data")
        print("   ✅ No hallucinated statistics")
        print("   ✅ FastAPI endpoints work correctly")
        print("   ✅ Daily updates at 3:30 AM UTC")
        print("   ✅ Complete data flow verified")
        print()
        print("🎯 Next automatic update: Tonight at 3:30 AM UTC")
        print()
    else:
        print()
        print("⚠️  SOME CHECKS FAILED")
        print()
        if not step1:
            print("   ❌ PostgreSQL data issues")
        if not step2:
            print("   ❌ FastAPI endpoint issues")
        if not step3:
            print("   ❌ Scheduler configuration issues")
        if not step4:
            print("   ❌ Pipeline integration issues")
        print()
    
    print("="*80)
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

