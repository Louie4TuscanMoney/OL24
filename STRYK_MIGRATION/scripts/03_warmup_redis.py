"""
STRYK MIGRATION STEP 3: WARMUP REDIS CACHE
===========================================

Warms Redis cache with active games and features

Usage:
    export REDIS_URL="redis://..."
    python 03_warmup_redis.py
"""

import redis
import json
import os
import sys
from pathlib import Path

# =====================================================
# CONFIGURATION
# =====================================================

INPUT_FILE = Path("../export/migration_data.json")
REDIS_URL = os.getenv("REDIS_URL")

if not REDIS_URL:
    print("❌ ERROR: REDIS_URL environment variable not set!")
    print("   Set it with: export REDIS_URL='redis://...'")
    sys.exit(1)

# =====================================================
# WARMUP FUNCTIONS
# =====================================================

def warmup_active_games(r: redis.Redis, games: list):
    """Warm active games set"""
    print(f"\n📦 Warming active games...")
    
    active_game_ids = []
    for g in games:
        if g.get("status") == 2 and g.get("period", 0) >= 2:
            game_id = g.get("game_id") or g.get("external_id")
            if game_id:
                active_game_ids.append(game_id)
    
    if active_game_ids:
        r.sadd("live:games:active", *active_game_ids)
        print(f"✅ Warmed {len(active_game_ids)} active games")
    else:
        print(f"⚠️  No active games to warm")

def warmup_features(r: redis.Redis, opportunities: list):
    """Warm feature cache"""
    print(f"\n📦 Warming features cache...")
    
    warmed = 0
    for opp in opportunities:
        game_id = opp.get("game_id") or opp.get("external_id")
        features = opp.get("features", [])
        
        if game_id and features:
            key = f"features:nba:{game_id}"
            r.set(key, json.dumps(features), ex=86400)  # 24 hour expiry
            warmed += 1
    
    print(f"✅ Warmed {warmed} feature caches")

def warmup_predictions(r: redis.Redis, opportunities: list):
    """Warm prediction cache"""
    print(f"\n📦 Warming predictions cache...")
    
    warmed = 0
    for opp in opportunities:
        game_id = opp.get("game_id") or opp.get("external_id")
        
        if game_id:
            key = f"prediction:nba:{game_id}"
            r.set(key, json.dumps(opp), ex=3600)  # 1 hour expiry
            warmed += 1
    
    print(f"✅ Warmed {warmed} prediction caches")

def warmup_mamba_scores(r: redis.Redis, mamba_scores: list):
    """Warm Mamba scores cache"""
    print(f"\n📦 Warming Mamba scores cache...")
    
    warmed = 0
    for score in mamba_scores:
        game_id = score.get("game_id") or score.get("external_id")
        
        if game_id:
            key = f"mamba_score:nba:{game_id}"
            r.set(key, json.dumps(score), ex=86400)  # 24 hour expiry
            warmed += 1
    
    print(f"✅ Warmed {warmed} Mamba score caches")

# =====================================================
# MAIN WARMUP
# =====================================================

def run_warmup():
    """Run complete Redis warmup"""
    print("=" * 60)
    print("🚀 STRYK MIGRATION: WARMING REDIS CACHE")
    print("=" * 60)
    
    # Load exported data
    print(f"\n📂 Loading {INPUT_FILE}...")
    if not INPUT_FILE.exists():
        print(f"❌ ERROR: {INPUT_FILE} not found!")
        print("   Run 01_export_old_data.py first")
        sys.exit(1)
    
    with open(INPUT_FILE) as f:
        export_data = json.load(f)
    
    data = export_data.get("data", {})
    print(f"✅ Loaded export from {export_data.get('export_timestamp')}")
    
    # Connect to Redis
    print(f"\n🔌 Connecting to Redis...")
    r = redis.from_url(REDIS_URL)
    
    try:
        r.ping()
        print("✅ Connected to Redis")
    except Exception as e:
        print(f"❌ ERROR: Could not connect to Redis: {e}")
        sys.exit(1)
    
    # Warm caches
    games = data.get("live_games", [])
    if isinstance(games, dict):
        games = [games]
    
    opportunities = data.get("opportunities", [])
    if isinstance(opportunities, dict):
        opportunities = [opportunities]
    
    mamba_scores = data.get("mamba_scores", [])
    if isinstance(mamba_scores, dict):
        mamba_scores = [mamba_scores]
    
    warmup_active_games(r, games)
    warmup_features(r, opportunities)
    warmup_predictions(r, opportunities)
    warmup_mamba_scores(r, mamba_scores)
    
    # Summary
    print("\n" + "=" * 60)
    print("✅ REDIS WARMUP COMPLETE")
    print("=" * 60)
    
    # Show cache stats
    print("\n📊 Cache Stats:")
    print(f"   Active games: {r.scard('live:games:active')}")
    print(f"   Total keys: {r.dbsize()}")
    print(f"   Memory used: {r.info('memory')['used_memory_human']}")
    
    print("\n🎯 Next step: Update backend to use Postgres + Redis")
    print("=" * 60)

# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":
    try:
        run_warmup()
    except KeyboardInterrupt:
        print("\n\n⚠️  Warmup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

