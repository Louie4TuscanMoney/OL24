"""
Debug Basketball Reference HTML structure
"""

import cloudscraper
import pandas as pd
from io import StringIO
import time

scraper = cloudscraper.create_scraper(
    browser={'browser': 'chrome', 'platform': 'darwin', 'desktop': True}
)

url = "https://www.basketball-reference.com/leagues/NBA_2026_per_poss.html"

print(f"Fetching: {url}")
print()

time.sleep(3)
response = scraper.get(url, timeout=30)
response.raise_for_status()

print(f"Status: {response.status_code}")
print(f"Content length: {len(response.content)}")
print()

# Parse with pandas
dfs = pd.read_html(StringIO(response.text))

print(f"Found {len(dfs)} tables")
print()

for i, df in enumerate(dfs):
    print(f"Table {i+1}:")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)[:10]}")
    
    if len(df) > 0:
        print(f"  First few rows:")
        for j in range(min(3, len(df))):
            row = df.iloc[j]
            player = str(row.get('Player', row.iloc[1] if len(row) > 1 else '')).strip()
            pts = row.get('PTS', row.get(('Unnamed: 29_level_0', 'PTS'), 'N/A'))
            print(f"    {j+1}. {player}: {pts}")
    print()

# Show the main table structure
if len(dfs) > 0:
    main_df = dfs[0]
    print("Main table structure:")
    print(f"Columns: {list(main_df.columns)}")
    print()
    
    # Show sample data
    print("Sample data (first 5 rows):")
    for i in range(min(5, len(main_df))):
        row = main_df.iloc[i]
        player = str(row.get('Player', row.iloc[1] if len(row) > 1 else '')).strip()
        team = str(row.get('Tm', row.get(('Unnamed: 2_level_0', 'Tm'), 'N/A'))).strip()
        pts = row.get('PTS', row.get(('Unnamed: 29_level_0', 'PTS'), 'N/A'))
        print(f"  {i+1}. {player} ({team}): {pts} PTS")

