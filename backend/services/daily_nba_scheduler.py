"""
DAILY NBA DATA SCHEDULER
Runs update_from_espn_daily.py every day at 3:30 AM UTC

This runs as a background thread in the main FastAPI app.
Uses ESPN's hidden API for accurate stats (more reliable than nba_api).

Pipeline: ESPN API → PostgreSQL → FastAPI → Frontend
"""

import schedule
import time
import threading
import subprocess
import os
from datetime import datetime


def run_daily_nba_update():
    """
    Run the NBA data population scripts
    
    Uses Basketball Reference (reliable, comprehensive) instead of nba_api
    (nba_api is broken for 2025-26 season)
    """
    print("\n" + "="*80)
    print(f"🕒 DAILY NBA UPDATE STARTED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("="*80)
    
    try:
        # Get DATABASE_URL from environment
        database_url = os.getenv('DATABASE_URL')
        if not database_url:
            print("❌ DATABASE_URL not set - skipping update")
            return
        
        # Main update: ESPN Comprehensive Pipeline (ALL data!)
        print("\n🏀 Running ESPN Comprehensive Pipeline...")
        result = subprocess.run(
            ['python3', 'backend/services/espn_comprehensive_pipeline.py'],
            env={**os.environ, 'DATABASE_URL': database_url},
            capture_output=True,
            text=True,
            timeout=600
        )
        
        print(result.stdout)
        
        if result.returncode == 0:
            print("   ✅ ESPN API update successful")
        else:
            print(f"   ❌ ESPN API update failed")
            print(result.stderr)
        
        print("\n" + "="*80)
        print("✅ DAILY UPDATE COMPLETE")
        print("   Pipeline: ESPN API → PostgreSQL → FastAPI → Frontend")
        print("="*80)
            
    except Exception as e:
        print(f"❌ Error running daily NBA update: {e}")


def start_scheduler():
    """Start the daily scheduler in a background thread"""
    # Schedule the job for 3:30 AM UTC every day
    schedule.every().day.at("03:30").do(run_daily_nba_update)
    
    print("\n" + "="*80)
    print("⏰ NBA DATA SCHEDULER STARTED")
    print("   Daily updates at 3:30 AM UTC")
    print("="*80 + "\n")
    
    # Run in background thread
    def run_scheduler():
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    
    scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
    scheduler_thread.start()


if __name__ == "__main__":
    # Test run
    print("Testing daily NBA update...")
    run_daily_nba_update()

