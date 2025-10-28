"""
Test BetOnline APIs with your authenticated cookies
"""

import requests
import json

# Your BetOnline cookies (from browser export)
COOKIES = {
    '__cf_bm': 'q5r2uoTKMsuWEJTw0UpNJ6AdULikm4N7MTcYXkj_7XQ-1761689867-1.0.1.1-exLBCGMVj.1TaCTeYcOklv_gH0wFEuEpHoBfJn1dEl6Zdi1STWPre1Pp9WsuEIrWlmah6_Yp7DoAYNMy5R2YlKB5I7J.z3s0X3mNw6o6z3k',
    '__cfruid': 'c25a9dbb9311d80895f5bb12ced46406e3263ae0-1761608752',
    'CT.CONTENT.NA.STATUS': '3',
    'key': 'eyJhbGciOiJSUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICJTQS1hVUczc05wcnZkSkVRSlZtTW1OVWxSLVNJOHBWZHNtSHV4enp6OUNRIn0.eyJleHAiOjE3NjE2OTA4NjIsImlhdCI6MTc2MTY5MDU2MiwiYXV0aF90aW1lIjoxNzYxNjA4NzU4LCJqdGkiOiI0NzZlODVhMi1hZTY3LTRhZjctYTgxOC04YTFmNzU5Njg3NWIiLCJpc3MiOiJodHRwczovL2FwaS5iZXRvbmxpbmUuYWcvYXBpL2F1dGgvcmVhbG1zL2JldG9ubGluZSIsImF1ZCI6ImFjY291bnQiLCJzdWIiOiI4NGUyZWVlOC02MTU5LTRhYmMtYTAxZC04MjM3Yjk1NDYwZjYiLCJ0eXAiOiJCZWFyZXIiLCJhenAiOiJiZXRvbmxpbmUtd2ViIiwic2Vzc2lvbl9zdGF0ZSI6ImVhMmUzYWYyLTgxMTItNGRlNS04N2E1LTgyN2Q5MTJlYWNjZCIsInJlYWxtX2FjY2VzcyI6eyJyb2xlcyI6WyJvZmZsaW5lX2FjY2VzcyIsInVtYV9hdXRob3JpemF0aW9uIiwiZGVmYXVsdC1yb2xlcy1iZXRvbmxpbmUiXX0sInJlc291cmNlX2FjY2VzcyI6eyJhY2NvdW50Ijp7InJvbGVzIjpbIm1hbmFnZS1hY2NvdW50IiwibWFuYWdlLWFjY291bnQtbGlua3MiLCJ2aWV3LXByb2ZpbGUiXX19LCJzY29wZSI6Im9wZW5pZCB0ZXN0LWFjY291bnQgc2VjdXJpdHktZmxhZ3MgZW1haWwgcHJvZmlsZSIsInNpZCI6ImVhMmUzYWYyLTgxMTItNGRlNS04N2E1LTgyN2Q5MTJlYWNjZCIsImJpcnRoZGF0ZSI6IjIwMDQtMDEtMTUiLCJlbWFpbF92ZXJpZmllZCI6ZmFsc2UsIm5hbWUiOiJMb3VpZSBXZWluaGF1cyIsInRydXN0ZWREZXZpY2VFbmFibGVkIjp0cnVlLCJwcmVmZXJyZWRfdXNlcm5hbWUiOiJiNTk2NzA0NCIsInRlc3RfYWNjb3VudCI6ZmFsc2UsImdpdmVuX25hbWUiOiJMb3VpZSIsImZhbWlseV9uYW1lIjoiV2VpbmhhdXMiLCJ0ZmEiOmZhbHNlLCJlbWFpbCI6ImxvdWlleGZpbmFuY2VAZ21haWwuY29tIn0.A_VWrjknOmWZXex3LIc2zdzgUx1pW1xBiQi8Yp5JECvsMlHW6ai0bZ6rj4cr3yu225xFjVW6t6fjqaAuDl1eyg6zEmKqq9s-kdGsCYi_XBj0_ii6o-OqMe4IH7PlxXge8vwP0KrPRswn9PGP6mKPkBJnaqBdlYLqPhljrvPQtY4WNmgMOFT3f774t1GWXpqdItpOmR5W4PkyPqLoGwUiA5Bi9iU7naQo9is8sLyhlCNj1y8QBw9TQk8LxO5MtYTViFXlQAGtYwk5f9OcRvj2AigviMJXPNnNlu0CWsSXkhMZw4t40ib3nsR9M-PiRdkotAvz5lVXQ1Sp6ecMAPT8-A',
    'kauth': 'eyJhbGciOiJSUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICJTQS1hVUczc05wcnZkSkVRSlZtTW1OVWxSLVNJOHBWZHNtSHV4enp6OUNRIn0.eyJleHAiOjE3NjE2OTA4NjIsImlhdCI6MTc2MTY5MDU2MiwiYXV0aF90aW1lIjoxNzYxNjA4NzU4LCJqdGkiOiI0NzZlODVhMi1hZTY3LTRhZjctYTgxOC04YTFmNzU5Njg3NWIiLCJpc3MiOiJodHRwczovL2FwaS5iZXRvbmxpbmUuYWcvYXBpL2F1dGgvcmVhbG1zL2JldG9ubGluZSIsImF1ZCI6ImFjY291bnQiLCJzdWIiOiI4NGUyZWVlOC02MTU5LTRhYmMtYTAxZC04MjM3Yjk1NDYwZjYiLCJ0eXAiOiJCZWFyZXIiLCJhenAiOiJiZXRvbmxpbmUtd2ViIiwic2Vzc2lvbl9zdGF0ZSI6ImVhMmUzYWYyLTgxMTItNGRlNS04N2E1LTgyN2Q5MTJlYWNjZCIsInJlYWxtX2FjY2VzcyI6eyJyb2xlcyI6WyJvZmZsaW5lX2FjY2VzcyIsInVtYV9hdXRob3JpemF0aW9uIiwiZGVmYXVsdC1yb2xlcy1iZXRvbmxpbmUiXX0sInJlc291cmNlX2FjY2VzcyI6eyJhY2NvdW50Ijp7InJvbGVzIjpbIm1hbmFnZS1hY2NvdW50IiwibWFuYWdlLWFjY291bnQtbGlua3MiLCJ2aWV3LXByb2ZpbGUiXX19LCJzY29wZSI6Im9wZW5pZCB0ZXN0LWFjY291bnQgc2VjdXJpdHktZmxhZ3MgZW1haWwgcHJvZmlsZSIsInNpZCI6ImVhMmUzYWYyLTgxMTItNGRlNS04N2E1LTgyN2Q5MTJlYWNjZCIsImJpcnRoZGF0ZSI6IjIwMDQtMDEtMTUiLCJlbWFpbF92ZXJpZmllZCI6ZmFsc2UsIm5hbWUiOiJMb3VpZSBXZWluaGF1cyIsInRydXN0ZWREZXZpY2VFbmFibGVkIjp0cnVlLCJwcmVmZXJyZWRfdXNlcm5hbWUiOiJiNTk2NzA0NCIsInRlc3RfYWNjb3VudCI6ZmFsc2UsImdpdmVuX25hbWUiOiJMb3VpZSIsImZhbWlseV9uYW1lIjoiV2VpbmhhdXMiLCJ0ZmEiOmZhbHNlLCJlbWFpbCI6ImxvdWlleGZpbmFuY2VAZ21haWwuY29tIn0.A_VWrjknOmWZXex3LIc2zdzgUx1pW1xBiQi8Yp5JECvsMlHW6ai0bZ6rj4cr3yu225xFjVW6t6fjqaAuDl1eyg6zEmKqq9s-kdGsCYi_XBj0_ii6o-OqMe4IH7PlxXge8vwP0KrPRswn9PGP6mKPkBJnaqBdlYLqPhljrvPQtY4WNmgMOFT3f774t1GWXpqdItpOmR5W4PkyPqLoGwUiA5Bi9iU7naQo9is8sLyhlCNj1y8QBw9TQk8LxO5MtYTViFXlQAGtYwk5f9OcRvj2AigviMJXPNnNlu0CWsSXkhMZw4t40ib3nsR9M-PiRdkotAvz5lVXQ1Sp6ecMAPT8-A',
    'CT.CONTENT.NA.STATUS': '3',
}

headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
    'Accept': 'application/json',
    'Referer': 'https://www.betonline.ag/sportsbook/basketball/nba',
}

print('='*80)
print('🧪 BETONLINE AUTHENTICATED API TEST')
print('='*80)

endpoints = [
    ('Feed API', 'https://www.betonline.ag/services/feeds/sportsbookv2/betml/event/live/2'),
    ('Offering API', 'https://api-offering.betonline.ag/api/offerings/sport/2/live'),
    ('External API', 'https://api-offering-ext.betonline.ag/api/events/basketball/live'),
]

for name, url in endpoints:
    print(f'\n🔍 {name}')
    print(f'   {url}')
    try:
        r = requests.get(url, headers=headers, cookies=COOKIES, timeout=15)
        print(f'   Status: {r.status_code}')
        
        if r.status_code == 200:
            try:
                data = r.json()
                print(f'   ✅ JSON RESPONSE!')
                print(f'   Type: {type(data).__name__}')
                
                if isinstance(data, dict):
                    print(f'   Keys: {list(data.keys())[:15]}')
                    
                    # Look for events/games
                    for key in ['events', 'games', 'data', 'items', 'offerings']:
                        if key in data:
                            val = data[key]
                            if isinstance(val, list):
                                print(f'   ✅✅✅ FOUND {key}: {len(val)} items')
                                if len(val) > 0 and isinstance(val[0], dict):
                                    print(f'   First item keys: {list(val[0].keys())[:15]}')
                                    print(f'   Sample data: {json.dumps(val[0], indent=2)[:500]}...')
                            else:
                                print(f'   Found {key}: {type(val).__name__}')
                
                elif isinstance(data, list):
                    print(f'   ✅✅✅ LIST with {len(data)} items')
                    if len(data) > 0 and isinstance(data[0], dict):
                        print(f'   First item: {json.dumps(data[0], indent=2)[:500]}...')
                
                # Save to file
                filename = f'/tmp/betonline_{name.replace(" ", "_").lower()}.json'
                with open(filename, 'w') as f:
                    json.dump(data, f, indent=2)
                print(f'   📄 Saved to {filename}')
                
            except json.JSONDecodeError:
                print(f'   ❌ Not JSON')
                print(f'   Response preview: {r.text[:200]}')
        else:
            print(f'   ❌ HTTP {r.status_code}')
            if r.status_code == 403:
                print(f'   (Still blocked - may need more cookies or CloudFlare bypass)')
    
    except Exception as e:
        print(f'   ❌ Error: {e}')

print('\n' + '='*80)
print('✅ Test complete!')
print('='*80)

