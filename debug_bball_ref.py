"""
Debug Basketball Reference HTML structure
"""

import cloudscraper
from bs4 import BeautifulSoup, Comment
import time

scraper = cloudscraper.create_scraper(
    browser={'browser': 'chrome', 'platform': 'darwin', 'desktop': True}
)

url = "https://www.basketball-reference.com/leagues/NBA_2026_per_poss.html"

print(f"Fetching: {url}")
print()

time.sleep(3)
response = scraper.get(url, timeout=20)

print(f"Status: {response.status_code}")
print(f"Content length: {len(response.content)}")
print()

soup = BeautifulSoup(response.content, 'html.parser')

# Check for all tables
print("All tables on page:")
tables = soup.find_all('table')
for i, table in enumerate(tables):
    table_id = table.get('id', 'NO ID')
    print(f"  {i+1}. Table ID: {table_id}")

print()

# Check HTML comments
print("Tables in HTML comments:")
comments = soup.find_all(string=lambda text: isinstance(text, Comment))
print(f"Found {len(comments)} comments")

for i, comment in enumerate(comments[:5]):
    if '<table' in comment:
        comment_soup = BeautifulSoup(comment, 'html.parser')
        tables_in_comment = comment_soup.find_all('table')
        for table in tables_in_comment:
            table_id = table.get('id', 'NO ID')
            print(f"  Comment {i}: Table ID: {table_id}")

print()

# Try to find specific table ID
print("Looking for 'per_poss_stats' table...")
table = soup.find('table', {'id': 'per_poss_stats'})
if table:
    print("✅ Found directly!")
else:
    print("❌ Not found directly, checking comments...")
    
    for comment in comments:
        if 'per_poss_stats' in comment:
            print("✅ Found in comment!")
            comment_soup = BeautifulSoup(comment, 'html.parser')
            table = comment_soup.find('table', {'id': 'per_poss_stats'})
            if table:
                print("✅ Successfully parsed from comment!")
                
                # Show first few rows
                tbody = table.find('tbody')
                if tbody:
                    rows = tbody.find_all('tr')[:5]
                    print(f"\nFirst 5 players:")
                    for row in rows:
                        player_cell = row.find('td', {'data-stat': 'player'})
                        pts_cell = row.find('td', {'data-stat': 'pts_per_poss'})
                        
                        if player_cell and pts_cell:
                            print(f"  {player_cell.text.strip():30s} {pts_cell.text.strip()}")
                
                break

