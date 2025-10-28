"""
TEST BOX SCORE API - Check the response structure
"""

import time
from nba_api.stats.endpoints import boxscoretraditionalv2, boxscoreadvancedv2

game_id = '0022400100'  # ATL vs OKC on Oct 27

print(f"Testing game: {game_id}\n")

print("="*80)
print("📊 TRADITIONAL BOX SCORE")
print("="*80)

try:
    time.sleep(0.6)
    trad = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=game_id)
    
    # Check what methods/attributes are available
    print("\n🔍 Available methods:")
    methods = [m for m in dir(trad) if not m.startswith('_')]
    for m in methods[:10]:
        print(f"   {m}")
    
    # Try different ways to get data
    print("\n🔍 Trying get_data_frames()...")
    try:
        dfs = trad.get_data_frames()
        print(f"   ✅ Got {len(dfs)} dataframes")
        for i, df in enumerate(dfs):
            print(f"   DataFrame {i}: {len(df)} rows, columns: {list(df.columns)[:5]}...")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print("\n🔍 Trying get_dict()...")
    try:
        data_dict = trad.get_dict()
        print(f"   ✅ Got dict with keys: {list(data_dict.keys())}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print("\n🔍 Trying get_json()...")
    try:
        json_str = trad.get_json()
        print(f"   ✅ Got JSON (length: {len(json_str)})")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        
except Exception as e:
    print(f"❌ Failed to create BoxScoreTraditionalV2: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("📊 ADVANCED BOX SCORE")
print("="*80)

try:
    time.sleep(0.6)
    adv = boxscoreadvancedv2.BoxScoreAdvancedV2(game_id=game_id)
    
    print("\n🔍 Trying get_data_frames()...")
    try:
        dfs = adv.get_data_frames()
        print(f"   ✅ Got {len(dfs)} dataframes")
        for i, df in enumerate(dfs):
            print(f"   DataFrame {i}: {len(df)} rows")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        
except Exception as e:
    print(f"❌ Failed to create BoxScoreAdvancedV2: {e}")
    import traceback
    traceback.print_exc()

