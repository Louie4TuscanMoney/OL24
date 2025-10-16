"""
Test NBA_API Installation and Connectivity
Run this first to verify nba_api is working
"""

print("="*80)
print("NBA_API INSTALLATION TEST")
print("="*80)

# Test 1: Import
print("\n[1/4] Testing import...")
try:
    from nba_api.live.nba.endpoints import scoreboard
    from nba_api.stats.static import teams
    print("✅ NBA_API imported successfully")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("\nRun: pip install nba-api")
    exit(1)

# Test 2: Get teams (static data)
print("\n[2/4] Testing static data...")
try:
    nba_teams = teams.get_teams()
    print(f"✅ Fetched {len(nba_teams)} NBA teams")
    print(f"   Sample: {nba_teams[0]['full_name']}")
except Exception as e:
    print(f"❌ Failed: {e}")
    exit(1)

# Test 3: Get live scoreboard
print("\n[3/4] Testing live data...")
try:
    import time
    start = time.time()
    
    board = scoreboard.ScoreBoard()
    games = board.games.get_dict()
    
    elapsed = (time.time() - start) * 1000
    
    print(f"✅ Live scoreboard fetched")
    print(f"   Response time: {elapsed:.0f}ms")
    print(f"   Games found: {len(games)}")
    
    if len(games) > 0:
        print(f"\n   Sample game:")
        game = games[0]
        print(f"   {game['awayTeam']['teamTricode']} @ {game['homeTeam']['teamTricode']}")
        print(f"   Score: {game['awayTeam']['score']} - {game['homeTeam']['score']}")
        print(f"   Period: {game['period']}")
        print(f"   Status: {game['gameStatusText']}")
    else:
        print("   (No games currently live)")
        
except Exception as e:
    print(f"❌ Failed: {e}")
    exit(1)

# Test 4: JSON parsing speed
print("\n[4/4] Testing JSON parsing...")
try:
    import json
    try:
        import orjson
        print("✅ orjson available (20-30% faster)")
    except ImportError:
        print("⚠️  orjson not installed (slower JSON parsing)")
        print("   Optional: pip install orjson")
except Exception as e:
    print(f"⚠️  Warning: {e}")

print("\n" + "="*80)
print("✅ NBA_API INSTALLATION TEST PASSED")
print("="*80)
print("\nReady to build live score buffer!")

