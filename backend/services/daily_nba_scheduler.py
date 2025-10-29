"""
DAILY NBA DATA SCHEDULER
Runs populate_comprehensive_nba_data.py every day at 3:30 AM UTC

This runs as a background thread in the main FastAPI app.
"""

import schedule
import time
import threading
import subprocess
import os
from datetime import datetime


def run_daily_nba_update():
    """Run the NBA data population script"""
    print("\n" + "="*80)
    print(f"🕒 DAILY NBA UPDATE STARTED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("="*80)
    
    try:
        # Get DATABASE_URL from environment
        database_url = os.getenv('DATABASE_URL')
        if not database_url:
            print("❌ DATABASE_URL not set - skipping update")
            return
        
        # Run the populate script
        result = subprocess.run(
            ['python3', 'backend/services/populate_comprehensive_nba_data.py'],
            env={**os.environ, 'DATABASE_URL': database_url},
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        if result.returncode == 0:
            print("✅ Daily NBA update completed successfully")
        else:
            print(f"❌ Daily NBA update failed with code {result.returncode}")
            
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

