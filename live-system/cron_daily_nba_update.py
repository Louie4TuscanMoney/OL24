#!/usr/bin/env python3
"""
DAILY NBA DATA UPDATE
Runs ESPN comprehensive pipeline to update all NBA data
Called by Railway cron daily at 3:30 AM UTC
"""

import os
import sys
import subprocess

def main():
    """Run ESPN comprehensive pipeline for daily updates"""
    print("\n" + "="*80)
    print("🌙 DAILY NBA DATA UPDATE - 3:30 AM UTC")
    print("="*80)
    print()
    
    # Make sure DATABASE_URL is set
    database_url = os.getenv('DATABASE_URL')
    if not database_url:
        print("❌ DATABASE_URL not set!")
        return 1
    
    print(f"✅ Database connected: {database_url.split('@')[1] if '@' in database_url else 'Unknown'}")
    print()
    
    # Run ESPN comprehensive pipeline
    # Path for Railway deployment (script is in live-system root)
    script_path = "espn_comprehensive_pipeline.py"
    
    if not os.path.exists(script_path):
        print(f"❌ Could not find ESPN pipeline script")
        print(f"   Tried: espn_comprehensive_pipeline.py")
        return 1
    
    print(f"📜 Running: {script_path}")
    print()
    
    try:
        # Run the ESPN pipeline
        result = subprocess.run(
            [sys.executable, script_path],
            env=os.environ,
            capture_output=False,  # Show output in Railway logs
            text=True,
            timeout=600  # 10 minute timeout
        )
        
        if result.returncode == 0:
            print()
            print("="*80)
            print("✅ DAILY UPDATE COMPLETE")
            print("="*80)
            return 0
        else:
            print()
            print("="*80)
            print(f"❌ UPDATE FAILED (exit code: {result.returncode})")
            print("="*80)
            return 1
            
    except subprocess.TimeoutExpired:
        print()
        print("="*80)
        print("❌ UPDATE TIMEOUT (>10 minutes)")
        print("="*80)
        return 1
    except Exception as e:
        print()
        print("="*80)
        print(f"❌ ERROR: {e}")
        print("="*80)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

