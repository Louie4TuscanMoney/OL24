# Step 1: Data Collection Setup

**Objective:** Set up infrastructure to collect historical NBA data from basketball-reference.com (2020-2025)

**Model Context (Paper-Verified):**
- 🔮 **Informer** (Zhou et al., AAAI 2021): Designed for 336-1440 inputs → Use for season-long analysis
- 📊 **Conformal** (Schlembach et al., PMLR 2022): Tested on t=192, h=12 → NBA t=18, h=6 similar scale
- 🎯 **Dejavu** (Kang et al., arXiv 2020): M1/M3 competitions → Optimal for limited data (short series)

**Duration:** 4-6 hours  
**Prerequisites:** Python 3.8+, internet connection  
**Output:** Working data collection pipeline with 5,400+ games

---

## Action Items

### 1.1 Environment Setup (30 minutes)

```bash
# Create project directory
mkdir -p nba-forecasting
cd nba-forecasting

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install requests beautifulsoup4 pandas numpy lxml html5lib pyarrow joblib
```

**Verify installation:**
```bash
python -c "import requests, bs4, pandas, numpy; print('✓ All packages installed')"
```

---

### 1.2 Create Basketball-Reference Scraper (1 hour)

**File:** `scrapers/basketball_reference.py`

```python
"""
Basketball-Reference Scraper
Respectful data collection with rate limiting and caching
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
from pathlib import Path
import pickle
from typing import List, Optional

class BasketballReferenceScraper:
    """
    Scrape NBA data from basketball-reference.com
    """
    def __init__(self, cache_dir='./cache/', rate_limit=(2.0, 4.0)):
        """
        Args:
            cache_dir: Directory for caching scraped data
            rate_limit: (min, max) seconds between requests
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.rate_limit = rate_limit
        
        self.headers = {
            'User-Agent': 'NBA Research Project (contact@your-email.com)'
        }
    
    def _get_with_cache(self, url: str, cache_key: str):
        """Get URL with caching"""
        cache_path = self.cache_dir / f"{cache_key}.pkl"
        
        # Check cache
        if cache_path.exists():
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        
        # Scrape with rate limiting
        time.sleep(random.uniform(*self.rate_limit))
        response = requests.get(url, headers=self.headers)
        
        if response.status_code != 200:
            raise Exception(f"Failed to fetch {url}: {response.status_code}")
        
        # Cache response
        with open(cache_path, 'wb') as f:
            pickle.dump(response.content, f)
        
        return response.content
    
    def scrape_schedule(self, year: int) -> pd.DataFrame:
        """
        Scrape game schedule for a season
        
        Args:
            year: Season end year (2021 = 2020-21 season)
        
        Returns:
            DataFrame with game_id, date, home_team, away_team
        """
        url = f"https://www.basketball-reference.com/leagues/NBA_{year}_games.html"
        cache_key = f"schedule_{year}"
        
        content = self._get_with_cache(url, cache_key)
        soup = BeautifulSoup(content, 'html.parser')
        
        # Find schedule tables (multiple months)
        games = []
        
        for table in soup.find_all('table', id=lambda x: x and 'schedule' in x):
            rows = table.find_all('tr')[1:]  # Skip header
            
            for row in rows:
                cols = row.find_all('td')
                if len(cols) >= 6:
                    date = cols[0].text.strip()
                    away_team = cols[2].find('a')['href'].split('/')[2] if cols[2].find('a') else None
                    home_team = cols[4].find('a')['href'].split('/')[2] if cols[4].find('a') else None
                    
                    # Construct game_id (format: YYYYMMDD0{HOME})
                    if date and home_team:
                        date_obj = pd.to_datetime(date)
                        game_id = f"{date_obj.strftime('%Y%m%d')}0{home_team}"
                        
                        games.append({
                            'game_id': game_id,
                            'date': date,
                            'home_team': home_team,
                            'away_team': away_team,
                            'season': f"{year-1}-{str(year)[2:]}"
                        })
        
        return pd.DataFrame(games)
    
    def scrape_play_by_play(self, game_id: str) -> pd.DataFrame:
        """
        Scrape play-by-play data for a single game
        
        Args:
            game_id: Game identifier (e.g., "202010220LAL")
        
        Returns:
            DataFrame with play-by-play events
        """
        url = f"https://www.basketball-reference.com/boxscores/pbp/{game_id}.html"
        cache_key = f"pbp_{game_id}"
        
        content = self._get_with_cache(url, cache_key)
        soup = BeautifulSoup(content, 'lxml')
        
        # Find play-by-play table
        pbp_table = soup.find('table', {'id': 'pbp'})
        
        if not pbp_table:
            raise Exception(f"No play-by-play table found for {game_id}")
        
        # Parse plays
        plays = []
        current_quarter = 1
        
        for row in pbp_table.find_all('tr'):
            # Check for quarter headers
            if 'thead' in row.get('class', []):
                quarter_text = row.text
                if '1st Q' in quarter_text:
                    current_quarter = 1
                elif '2nd Q' in quarter_text:
                    current_quarter = 2
                elif '3rd Q' in quarter_text:
                    current_quarter = 3
                elif '4th Q' in quarter_text:
                    current_quarter = 4
                continue
            
            cols = row.find_all('td')
            if len(cols) >= 6:
                time_text = cols[0].text.strip()
                score_text = cols[2].text.strip() if len(cols[2].text.strip()) > 0 else None
                
                # Parse score (format: "45-50")
                if score_text and '-' in score_text:
                    try:
                        score_away, score_home = map(int, score_text.split('-'))
                        
                        plays.append({
                            'time': time_text,
                            'quarter': current_quarter,
                            'score_away': score_away,
                            'score_home': score_home,
                            'differential': score_home - score_away,
                            'play_away': cols[1].text.strip(),
                            'play_home': cols[5].text.strip()
                        })
                    except:
                        continue
        
        return pd.DataFrame(plays)


# Usage example
if __name__ == "__main__":
    scraper = BasketballReferenceScraper()
    
    # Test: Scrape one game
    test_game_id = "202010220LAL"  # Lakers season opener 2020
    pbp = scraper.scrape_play_by_play(test_game_id)
    print(f"Scraped {len(pbp)} plays")
    print(pbp.head())
```

---

### 1.3 Collect All Game IDs (30 minutes)

**File:** `scripts/collect_game_ids.py`

```python
"""
Collect all NBA game IDs from 2020-2025
"""

from scrapers.basketball_reference import BasketballReferenceScraper
import pandas as pd

def collect_all_game_ids(start_year=2021, end_year=2025):
    """
    Collect game IDs for all seasons
    """
    scraper = BasketballReferenceScraper()
    all_games = []
    
    for year in range(start_year, end_year + 1):
        print(f"Collecting {year-1}-{year} season...")
        schedule_df = scraper.scrape_schedule(year)
        all_games.append(schedule_df)
        print(f"  Found {len(schedule_df)} games")
    
    # Combine all seasons
    combined_df = pd.concat(all_games, ignore_index=True)
    
    # Save game IDs
    combined_df.to_csv('data/game_ids_2020_2025.csv', index=False)
    print(f"\n✓ Total: {len(combined_df)} games")
    print(f"✓ Saved to: data/game_ids_2020_2025.csv")
    
    return combined_df

if __name__ == "__main__":
    game_ids_df = collect_all_game_ids(start_year=2021, end_year=2025)
    
    # Summary by season
    print("\nGames by season:")
    print(game_ids_df.groupby('season').size())
```

**Expected Output:**
```
Season 2020-21: ~1,080 games
Season 2021-22: ~1,230 games
Season 2022-23: ~1,230 games
Season 2023-24: ~1,230 games
Season 2024-25: ~600 games (partial)
Total: ~5,400 games
```

---

### 1.4 Bulk Scrape Play-by-Play (3-10 hours)

**File:** `scripts/bulk_scrape_pbp.py`

```python
"""
Bulk scrape play-by-play data for all games
WARNING: This takes 3-10 hours depending on number of games
"""

from scrapers.basketball_reference import BasketballReferenceScraper
import pandas as pd
from tqdm import tqdm

def bulk_scrape_pbp(game_ids_csv='data/game_ids_2020_2025.csv', output_dir='data/raw_pbp/'):
    """
    Scrape play-by-play for all games
    """
    # Load game IDs
    game_ids_df = pd.read_csv(game_ids_csv)
    game_ids = game_ids_df['game_id'].tolist()
    
    print(f"Starting bulk scrape of {len(game_ids)} games")
    print(f"Estimated time: {len(game_ids) * 2.5 / 3600:.1f} hours")
    print("This will be respectful to basketball-reference (2-4 sec/request)")
    
    scraper = BasketballReferenceScraper()
    
    # Create output directory
    from pathlib import Path
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    successful = 0
    failed = []
    
    # Scrape with progress bar
    for game_id in tqdm(game_ids, desc="Scraping games"):
        try:
            pbp_df = scraper.scrape_play_by_play(game_id)
            
            # Save individual game
            pbp_df.to_parquet(f"{output_dir}/{game_id}.parquet")
            successful += 1
            
        except Exception as e:
            failed.append((game_id, str(e)))
            continue
    
    # Save failed list for retry
    if failed:
        failed_df = pd.DataFrame(failed, columns=['game_id', 'error'])
        failed_df.to_csv('data/failed_games.csv', index=False)
    
    print(f"\n✓ Successful: {successful}/{len(game_ids)} games")
    print(f"✗ Failed: {len(failed)} games")
    print(f"✓ Saved to: {output_dir}")

if __name__ == "__main__":
    import sys
    
    # Confirmation before long-running scrape
    print("This will scrape 5,400+ games from basketball-reference.com")
    print("Estimated time: 3-10 hours")
    response = input("Continue? (yes/no): ")
    
    if response.lower() == 'yes':
        bulk_scrape_pbp()
    else:
        print("Cancelled. Run again when ready.")
```

---

### 1.5 Validation Checklist

**Before proceeding to Step 2:**

- [ ] ✅ Scrapers/basketball_reference.py created and tested
- [ ] ✅ Game IDs collected (~5,400 games from 2020-2025)
- [ ] ✅ Bulk scrape completed or in progress
- [ ] ✅ Cache directory has ~5,000+ .pkl files
- [ ] ✅ Raw play-by-play Parquet files saved
- [ ] ✅ Failed games list checked (should be <5%)

**Verify Data Quality:**

```python
# Quick validation script
import pandas as pd
from pathlib import Path

pbp_dir = Path('data/raw_pbp/')
pbp_files = list(pbp_dir.glob('*.parquet'))

print(f"Collected {len(pbp_files)} games")

# Sample check
sample = pd.read_parquet(pbp_files[0])
print(f"\nSample game structure:")
print(sample.head())
print(f"Columns: {list(sample.columns)}")
print(f"Quarters: {sample['quarter'].unique()}")
```

---

## Troubleshooting

**Problem:** Rate limited or blocked by basketball-reference

**Solution:** 
- Increase delay between requests (3-5 seconds)
- Run during off-peak hours (late night US time)
- Check robots.txt compliance
- Consider breaking into smaller batches

**Problem:** Missing play-by-play for some games

**Solution:**
- Check game_ids_2020_2025.csv for correct IDs
- Verify URL construction
- Some very old games may not have play-by-play
- Retry failed games list

**Problem:** Parsing errors on specific games

**Solution:**
- Basketball-reference HTML structure occasionally varies
- Add error handling for different table formats
- Log problematic games for manual inspection

---

## Expected Output

```
nba-forecasting/
├── cache/                    ← ~5,400 cached responses
│   ├── pbp_202010220LAL.pkl
│   ├── pbp_202010230LAC.pkl
│   └── ...
├── data/
│   ├── game_ids_2020_2025.csv      ← 5,400 game IDs
│   ├── failed_games.csv             ← Failed scrapes (if any)
│   └── raw_pbp/                     ← Raw play-by-play
│       ├── 202010220LAL.parquet
│       ├── 202010230LAC.parquet
│       └── ... (5,000+ files)
└── scrapers/
    └── basketball_reference.py      ← Scraper class
```

---

## Next Step

Once data collection is complete, proceed to **Step 2: Data Processing** to convert event-based play-by-play into minute-by-minute time series.

**Time savings:** With cache, re-running analysis takes seconds instead of hours!

---

*Action Step 1 of 10 - Data Collection Setup*

