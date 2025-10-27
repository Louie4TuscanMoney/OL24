# 🚀 Better Buzz Toolkit - Quick Start Guide
## How to Use in Any Project (5 Minutes to Setup)

---

## METHOD 1: Copy-Paste (Simplest)

**Step 1: Copy toolkit to your project**

```bash
# From any new project directory
cp -r "/Users/test/Desktop/Tuscan Money/Ontologic XYZ/ML Research/Action/BetterBuzz_Toolkit" .
```

**Step 2: Import and use**

```python
# In your Python script
import sys
sys.path.insert(0, 'BetterBuzz_Toolkit')

from stealth_api_client import StealthAPIClient

# Use it
client = StealthAPIClient()
response = client.get('https://api.example.com/data')
```

**Done. Your API calls are now Better Buzz optimized.** ✅

---

## METHOD 2: Direct Integration (More Control)

**Just copy the code you need:**

### **For API Calls:**

```python
import requests
import random
import time

# Better Buzz optimized session
session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'en-US,en;q=0.9',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'keep-alive'
})

# Make requests with randomized delay
def make_request(url):
    response = session.get(url, timeout=30)
    time.sleep(random.uniform(0.4, 0.8))  # Anti-pattern detection
    return response

# Use it
response = make_request('https://api.example.com/data')
```

### **For Checkpointing:**

```python
import pickle
from pathlib import Path

# Setup
checkpoint_file = Path('my_checkpoint.pkl')
processed_items = []

# Load checkpoint if exists
if checkpoint_file.exists():
    with open(checkpoint_file, 'rb') as f:
        data = pickle.load(f)
        processed_items = data['items']

# Process loop
for i, item in enumerate(items_to_process):
    # Your processing here
    result = process_item(item)
    processed_items.append(result)
    
    # Save every 50 items
    if i % 50 == 0:
        with open(checkpoint_file, 'wb') as f:
            pickle.dump({'items': processed_items}, f)

# Final save
with open('final_output.pkl', 'wb') as f:
    pickle.dump(processed_items, f)
```

### **For Auto-Restart:**

```bash
# Run any Python script with watchdog
cd BetterBuzz_Toolkit
bash smart_watchdog.sh ../your_script.py

# Watchdog monitors and auto-restarts
# Check status: cat watchdog_*.log
```

---

## 📋 COMMON USE CASES

### **USE CASE 1: Scraping a Different API**

```python
# Example: Scraping weather data
from BetterBuzz_Toolkit.stealth_api_client import StealthAPIClient

client = StealthAPIClient()

cities = ['San Diego', 'Los Angeles', 'San Francisco']
weather_data = []

for city in cities:
    response = client.get(f'https://api.weather.com/forecast/{city}')
    
    if response:
        weather_data.append(response.json())

client.print_stats()
```

### **USE CASE 2: Downloading Many Files**

```python
from BetterBuzz_Toolkit.stealth_api_client import StealthAPIClient
import pickle

client = StealthAPIClient()
checkpoint = Path('download_checkpoint.pkl')

# Resume capability
downloaded = []
if checkpoint.exists():
    with open(checkpoint, 'rb') as f:
        downloaded = pickle.load(f)['files']

urls_to_download = ['url1', 'url2', ..., 'url1000']
remaining = [u for u in urls_to_download if u not in downloaded]

for i, url in enumerate(remaining):
    response = client.get(url)
    
    if response:
        # Save file
        filename = url.split('/')[-1]
        with open(f'downloads/{filename}', 'wb') as f:
            f.write(response.content)
        
        downloaded.append(url)
    
    # Checkpoint every 50
    if i % 50 == 0:
        with open(checkpoint, 'wb') as f:
            pickle.dump({'files': downloaded}, f)

print(f"Downloaded {len(downloaded)} files")
client.print_stats()
```

### **USE CASE 3: Web Scraping with BeautifulSoup**

```python
from BetterBuzz_Toolkit.stealth_api_client import StealthAPIClient
from bs4 import BeautifulSoup

client = StealthAPIClient()

urls = ['https://example.com/page1', 'https://example.com/page2', ...]
scraped_data = []

for url in urls:
    response = client.get(url)
    
    if response:
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract data
        title = soup.find('h1').text
        content = soup.find('div', class_='content').text
        
        scraped_data.append({
            'url': url,
            'title': title,
            'content': content
        })

print(f"Scraped {len(scraped_data)} pages")
```

### **USE CASE 4: Crypto/Stock Data Collection**

```python
from BetterBuzz_Toolkit.stealth_api_client import StealthAPIClient
import pandas as pd

client = StealthAPIClient()

# Example: Fetch stock prices
symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']
stock_data = []

for symbol in symbols:
    response = client.get(
        f'https://api.stockdata.com/v1/price',
        params={'symbol': symbol}
    )
    
    if response:
        stock_data.append(response.json())

# Convert to DataFrame
df = pd.DataFrame(stock_data)
df.to_csv('stock_data.csv', index=False)

client.print_stats()
```

---

## 🎯 REAL EXAMPLE (What You Just Did)

**Your NBA extraction (simplified):**

```python
from BetterBuzz_Toolkit.stealth_api_client import StealthAPIClient
from nba_api.stats.endpoints import playbyplayv2
import pickle
from pathlib import Path

# Setup
client = StealthAPIClient()

# Monkey-patch nba_api to use stealth client
import nba_api.stats.endpoints.playbyplayv2 as pbp_module
pbp_module.requests = client.session

# Checkpoint
checkpoint = Path('nba_checkpoint.pkl')
processed = []

if checkpoint.exists():
    with open(checkpoint, 'rb') as f:
        processed = pickle.load(f)['games']

# Process games
game_ids = ['0022101217', '0022101222', ...]  # 7,000 game IDs
remaining = [gid for gid in game_ids if gid not in processed]

for i, game_id in enumerate(remaining):
    # Fetch (automatically uses stealth mode)
    pbp = playbyplayv2.PlayByPlayV2(game_id=game_id)
    plays = pbp.get_data_frames()[0]
    
    # Extract your data
    pattern = extract_pattern(plays)
    processed.append(pattern)
    
    # Checkpoint every 50
    if i % 50 == 0:
        with open(checkpoint, 'wb') as f:
            pickle.dump({'games': processed}, f)

# Save
with open('nba_patterns.pkl', 'wb') as f:
    pickle.dump(processed, f)

print(f"✅ Processed {len(processed)} games")
client.print_stats()
```

**Run with watchdog:**
```bash
cd BetterBuzz_Toolkit
bash smart_watchdog.sh ../nba_scraper.py
```

---

## 🔧 CUSTOMIZATION

### **Change Delays:**

```python
# In better_buzz_config.py

TIMING = {
    'delay_min': 0.2,  # Faster (more aggressive)
    'delay_max': 0.5,
    # OR
    'delay_min': 0.8,  # Slower (safer)
    'delay_max': 1.5,
}
```

### **Different Headers (For Non-NBA APIs):**

```python
# For general APIs
STEALTH_HEADERS = {
    'User-Agent': 'Mozilla/5.0 ...',
    'Accept': 'application/json',
    # Remove NBA-specific headers:
    # 'Referer': 'https://www.nba.com/',
}
```

### **Adjust Cache Duration:**

```python
CACHE_CONFIG = {
    'max_age_days': 1,  # Cache only 1 day (for fresh data)
    # OR
    'max_age_days': 30,  # Cache 30 days (for static data)
}
```

---

## 📚 TEMPLATE FOR NEW PROJECT

**Copy this template for any new API project:**

```python
#!/usr/bin/env python3
"""
My New API Project - Better Buzz Optimized
"""

import sys
sys.path.insert(0, 'BetterBuzz_Toolkit')

from stealth_api_client import StealthAPIClient
import pickle
from pathlib import Path

# Configuration
API_ENDPOINT = 'https://api.example.com'
CHECKPOINT_FILE = Path('my_project_checkpoint.pkl')

# Initialize
client = StealthAPIClient()
processed = []

# Load checkpoint
if CHECKPOINT_FILE.exists():
    with open(CHECKPOINT_FILE, 'rb') as f:
        processed = pickle.load(f)['data']
    print(f"✅ Resuming from {len(processed)} items")

# Get items to process
items_to_fetch = get_my_items()  # Your list
remaining = [i for i in items_to_fetch if i not in [p['id'] for p in processed]]

print(f"Processing {len(remaining)} items...")

# Process loop
try:
    for i, item_id in enumerate(remaining):
        # Fetch data
        response = client.get(f'{API_ENDPOINT}/items/{item_id}')
        
        if response:
            data = response.json()
            processed.append(data)
        
        # Progress report
        if i % 10 == 0:
            print(f"Progress: {i}/{len(remaining)} ({i/len(remaining)*100:.1f}%)")
        
        # Checkpoint
        if i % 50 == 0:
            with open(CHECKPOINT_FILE, 'wb') as f:
                pickle.dump({'data': processed}, f)

except KeyboardInterrupt:
    print("\n⏸️  Paused - saving checkpoint...")
    with open(CHECKPOINT_FILE, 'wb') as f:
        pickle.dump({'data': processed}, f)
    print("✅ Checkpoint saved")
    sys.exit(0)

# Save final
output_file = Path('final_output.pkl')
with open(output_file, 'wb') as f:
    pickle.dump(processed, f)

print(f"\n✅ Complete: {len(processed)} items")
client.print_stats()
```

**Run with:**
```bash
cd BetterBuzz_Toolkit
bash smart_watchdog.sh ../my_new_project.py
```

**Done. Works for ANY API project at Better Buzz.** ✅

---

## 💡 SUMMARY

**To use in any project:**

1. **Copy toolkit** → `cp -r BetterBuzz_Toolkit /new/project/`
2. **Import client** → `from BetterBuzz_Toolkit.stealth_api_client import StealthAPIClient`
3. **Use it** → `client = StealthAPIClient(); response = client.get(url)`
4. **Add checkpointing** → Save every 50 items
5. **Run with watchdog** → `bash smart_watchdog.sh script.py`

**That's it. 5 steps. Works for everything.** 🎯
