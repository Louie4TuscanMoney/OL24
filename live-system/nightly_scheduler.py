"""
NIGHTLY SCHEDULER - Runs at 3:30 AM UTC Every Day
Continuously running worker that executes the Basketball Reference scraper
at the scheduled time
"""

import time
from datetime import datetime, timedelta
import subprocess
import sys

def get_seconds_until_next_run():
    """
    Calculate seconds until next 3:30 AM UTC
    """
    now = datetime.utcnow()
    
    # Target time: 3:30 AM UTC
    target_hour = 3
    target_minute = 30
    
    # Calculate next run time
    next_run = now.replace(hour=target_hour, minute=target_minute, second=0, microsecond=0)
    
    # If we've already passed 3:30 AM today, schedule for tomorrow
    if now >= next_run:
        next_run += timedelta(days=1)
    
    seconds_until = (next_run - now).total_seconds()
    
    print(f"⏰ Current time: {now.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"⏰ Next run: {next_run.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"⏰ Sleeping for {seconds_until/3600:.1f} hours ({int(seconds_until)} seconds)")
    
    return seconds_until, next_run


def run_nightly_job():
    """
    Execute the nightly Basketball Reference scraper
    """
    print("\n" + "="*80)
    print("🌙 EXECUTING NIGHTLY JOB")
    print("="*80 + "\n")
    
    try:
        # Run the nightly scraper
        result = subprocess.run(
            [sys.executable, 'nightly_basketball_reference.py'],
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )
        
        print(result.stdout)
        
        if result.returncode != 0:
            print(f"❌ Job failed with return code {result.returncode}")
            print(result.stderr)
        else:
            print("✅ Job completed successfully")
            
    except subprocess.TimeoutExpired:
        print("❌ Job timed out after 10 minutes")
    except Exception as e:
        print(f"❌ Job failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """
    Main scheduler loop - runs forever
    """
    print("\n" + "="*80)
    print("🕐 NBA NIGHTLY SCHEDULER STARTED")
    print("="*80)
    print("Schedule: Every day at 3:30 AM UTC")
    print("Task: Scrape yesterday's games from Basketball Reference")
    print("="*80 + "\n")
    
    while True:
        try:
            # Calculate sleep time
            seconds_until, next_run = get_seconds_until_next_run()
            
            # Sleep until scheduled time
            time.sleep(seconds_until)
            
            # Run the job
            run_nightly_job()
            
            # Sleep 1 minute to avoid re-running immediately
            print("\n⏸️  Sleeping 1 minute to avoid duplicate runs...")
            time.sleep(60)
            
        except KeyboardInterrupt:
            print("\n\n🛑 Scheduler stopped by user")
            break
        except Exception as e:
            print(f"\n❌ Scheduler error: {e}")
            import traceback
            traceback.print_exc()
            
            # Sleep 1 hour on error to avoid rapid retries
            print("\n⏸️  Sleeping 1 hour after error...")
            time.sleep(3600)


if __name__ == "__main__":
    main()

