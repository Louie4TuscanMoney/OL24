#!/usr/bin/env python3
"""
LIVE API SPEED TEST
Compare all NBA data sources and show which is fastest and freshest
Run this while games are live!
"""

import requests
import time
from datetime import datetime

def test_all_sources():
    """Test all NBA API sources and compare speed/freshness"""
    
    print("="*80)
    print("🏀 LIVE NBA API SPEED TEST")
    print("="*80)
    print(f"Timestamp: {datetime.now().strftime('%H:%M:%S')}")
    print()
    
    sources = []
    
    # SOURCE 1: nba_api library
    print("1️⃣ Testing nba_api library...")
    try:
        from nba_api.live.nba.endpoints import scoreboard
        
        start = time.time()
        board = scoreboard.ScoreBoard()
        data = board.get_dict()
        elapsed = time.time() - start
        
        if data and 'scoreboard' in data and 'games' in data['scoreboard']:
            games = data['scoreboard']['games']
            print(f"   ✅ SUCCESS in {elapsed:.3f}s")
            print(f"   Games: {len(games)}")
            
            for game in games:
                if game.get('gameStatus') == 2:  # Live
                    home = game['homeTeam']
                    away = game['awayTeam']
                    period = game.get('period', 0)
                    clock = game.get('gameClock', '')
                    total = home.get('score', 0) + away.get('score', 0)
                    
                    print(f"   🔴 {away['teamTricode']} @ {home['teamTricode']}: {away.get('score', 0)}-{home.get('score', 0)} | Q{period} {clock}")
                    
                    sources.append({
                        'name': 'nba_api',
                        'speed': elapsed,
                        'game_id': game['gameId'],
                        'total_score': total,
                        'away_score': away.get('score', 0),
                        'home_score': home.get('score', 0),
                        'period': period,
                        'clock': clock
                    })
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
    
    print()
    
    # SOURCE 2: ESPN CDN (xhr=1 - LIVE UPDATES!)
    print("2️⃣ Testing ESPN CDN (xhr=1 - LIVE UPDATES!)...")
    try:
        start = time.time()
        response = requests.get(
            "https://cdn.espn.com/core/nba/scoreboard?xhr=1&limit=50",
            headers={'User-Agent': 'Mozilla/5.0'},
            timeout=3
        )
        elapsed = time.time() - start
        
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ SUCCESS in {elapsed:.3f}s")
            print(f"   Response keys: {list(data.keys())}")
            
            # Try to find games in response
            if 'events' in data:
                print(f"   Games found in 'events': {len(data['events'])}")
                for event in data['events'][:3]:  # Show first 3
                    print(f"   Event: {event.get('name', 'Unknown')}")
            elif 'content' in data:
                print(f"   Response has 'content' key")
                content = data['content']
                print(f"   Content keys: {list(content.keys())}")
            
            # Save full response for inspection
            with open('espn_cdn_response.json', 'w') as f:
                import json
                json.dump(data, f, indent=2)
            print(f"   💾 Full response saved to: espn_cdn_response.json")
                
        else:
            print(f"   ❌ HTTP {response.status_code}")
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
    
    print()
    
    # SOURCE 3: ESPN API (standard)
    print("3️⃣ Testing ESPN API (standard)...")
    try:
        start = time.time()
        response = requests.get(
            "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard",
            headers={'User-Agent': 'Mozilla/5.0'},
            timeout=3
        )
        elapsed = time.time() - start
        
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ SUCCESS in {elapsed:.3f}s")
            
            if 'events' in data:
                games = data['events']
                print(f"   Games: {len(games)}")
                
                for event in games:
                    if event['status']['type']['id'] == '2':  # Live
                        status = event['status']
                        comp = event['competitions'][0]
                        competitors = comp['competitors']
                        
                        home = next(c for c in competitors if c['homeAway'] == 'home')
                        away = next(c for c in competitors if c['homeAway'] == 'away')
                        
                        period = status.get('period', 0)
                        clock = status.get('displayClock', '')
                        total = int(home.get('score', 0)) + int(away.get('score', 0))
                        
                        print(f"   🔴 {away['team']['abbreviation']} @ {home['team']['abbreviation']}: {away.get('score', 0)}-{home.get('score', 0)} | Q{period} {clock}")
                        
                        sources.append({
                            'name': 'espn_api',
                            'speed': elapsed,
                            'game_id': event['id'],
                            'total_score': total,
                            'away_score': int(away.get('score', 0)),
                            'home_score': int(home.get('score', 0)),
                            'period': period,
                            'clock': clock
                        })
        else:
            print(f"   ❌ HTTP {response.status_code}")
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
    
    print()
    print("="*80)
    print("📊 COMPARISON RESULTS")
    print("="*80)
    
    if sources:
        # Compare same game across sources
        print("\n🔍 Comparing data freshness (higher score = more recent):\n")
        
        for source in sources:
            print(f"{source['name']:15} | Speed: {source['speed']:.3f}s | Total: {source['total_score']:3} | {source['away_score']}-{source['home_score']} | Q{source['period']} {source['clock']}")
        
        # Find fastest and freshest
        fastest = min(sources, key=lambda x: x['speed'])
        freshest = max(sources, key=lambda x: x['total_score'])
        
        print()
        print(f"⚡ FASTEST: {fastest['name']} ({fastest['speed']:.3f}s)")
        print(f"🎯 FRESHEST: {freshest['name']} (total score: {freshest['total_score']})")
        
        if fastest['name'] == freshest['name']:
            print(f"\n✅ {fastest['name']} is BOTH fastest AND freshest! Use this!")
        else:
            print(f"\n⚠️  {fastest['name']} is faster but {freshest['name']} has newer data")
            print(f"   Recommendation: Use multi-source strategy (current implementation)")
    else:
        print("\n⚠️  No live games found or all sources failed")
    
    print()
    print("="*80)

if __name__ == "__main__":
    test_all_sources()

