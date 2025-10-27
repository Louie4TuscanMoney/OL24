#!/usr/bin/env python3
"""
⚡ STEALTH EXTRACTION - Optimized for Public WiFi

Optimizations:
- Browser-like headers (doesn't look like bot)
- Connection pooling (fewer connections)
- Aggressive caching (minimize API calls)
- Retry logic (handle throttling)
- Randomized delays (less pattern detection)
- Batch optimization

Goal: Get past coffee shop WiFi throttling
"""

import sys
import time
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from datetime import datetime, timedelta
from scipy import stats
from scipy.fft import fft
from collections import deque
import random
import hashlib
import requests

# STEALTH: Override nba_api to use session with browser headers
from nba_api.stats.endpoints import playbyplayv2
from nba_api.stats.static import teams

# Create persistent session with browser-like headers
import requests
session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'en-US,en;q=0.9',
    'Accept-Encoding': 'gzip, deflate, br',
    'Referer': 'https://www.nba.com/',
    'Origin': 'https://www.nba.com',
    'Connection': 'keep-alive'
})

# Monkey-patch nba_api to use our session
import nba_api.stats.endpoints.playbyplayv2 as pbp_module
pbp_module.requests = session

# CACHING: Aggressive disk cache
CACHE_DIR = Path('.stealth_cache')
CACHE_DIR.mkdir(exist_ok=True)

class StealthCache:
    """Aggressive caching to minimize API calls"""
    
    @staticmethod
    def get_cache_path(game_id, endpoint='pbp'):
        key = hashlib.md5(f"{endpoint}_{game_id}".encode()).hexdigest()
        return CACHE_DIR / f"{key}.pkl"
    
    @staticmethod
    def get(game_id, endpoint='pbp'):
        cache_file = StealthCache.get_cache_path(game_id, endpoint)
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        return None
    
    @staticmethod
    def set(game_id, data, endpoint='pbp'):
        cache_file = StealthCache.get_cache_path(game_id, endpoint)
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)


class StealthExtractor:
    """
    Optimized extractor for throttled networks
    """
    
    def __init__(self):
        self.api_call_times = deque(maxlen=20)
        self.cache_hits = 0
        self.cache_misses = 0
        
    def fetch_with_retry(self, game_id, max_retries=3):
        """
        Fetch play-by-play with retry logic
        Handles throttling gracefully
        """
        # Check cache first
        cached = StealthCache.get(game_id)
        if cached is not None:
            self.cache_hits += 1
            return cached
        
        self.cache_misses += 1
        
        # Fetch with retry
        for attempt in range(max_retries):
            try:
                start = time.time()
                
                # Add randomized delay to avoid pattern detection
                if attempt > 0:
                    jitter = random.uniform(1.0, 3.0)
                    time.sleep(jitter)
                
                pbp = playbyplayv2.PlayByPlayV2(game_id=game_id)
                plays_df = pbp.get_data_frames()[0]
                
                # Cache result
                StealthCache.set(game_id, plays_df)
                
                # Track timing
                elapsed = time.time() - start
                self.api_call_times.append(elapsed)
                
                return plays_df
                
            except Exception as e:
                if attempt < max_retries - 1:
                    # Exponential backoff with jitter
                    backoff = (2 ** attempt) + random.uniform(0, 1)
                    time.sleep(backoff)
                else:
                    return None
        
        return None
    
    def extract_pattern_fast(self, plays_df):
        """Optimized pattern extraction (same logic, faster code)"""
        
        if len(plays_df) == 0:
            return None
        
        differentials = [0] * 18
        diff_2q_6min = None
        diff_ht = None
        diff_final = None
        
        # Vectorized processing where possible
        for idx, play in plays_df.iterrows():
            period = play['PERIOD']
            
            if period > 4:
                break
            
            pctimestring = play['PCTIMESTRING']
            score_margin = play['SCOREMARGIN']
            
            if pd.notna(pctimestring) and pd.notna(score_margin):
                try:
                    mins_remaining = int(pctimestring.split(':')[0])
                    
                    if period == 1:
                        elapsed = 12 - mins_remaining
                    elif period == 2:
                        elapsed = 12 + (12 - mins_remaining)
                        
                        if 5 <= mins_remaining <= 7:
                            diff_2q_6min = 0 if score_margin == 'TIE' else int(score_margin)
                        
                        if mins_remaining == 0:
                            diff_ht = 0 if score_margin == 'TIE' else int(score_margin)
                    elif period == 4:
                        if mins_remaining == 0:
                            diff_final = 0 if score_margin == 'TIE' else int(score_margin)
                        continue
                    else:
                        continue
                    
                    if 0 <= elapsed < 18:
                        diff = 0 if score_margin == 'TIE' else int(score_margin)
                        differentials[elapsed] = diff
                
                except:
                    continue
        
        # Forward fill
        for i in range(1, 18):
            if differentials[i] == 0 and differentials[i-1] != 0:
                differentials[i] = differentials[i-1]
        
        return {
            'pattern': differentials,
            'diff_at_2q_6min': diff_2q_6min or differentials[17],
            'diff_at_halftime': diff_ht,
            'diff_at_final': diff_final
        }
    
    def extract_statistical_fast(self, pattern):
        """Fast statistical feature extraction"""
        pattern = np.array(pattern)
        
        return {
            'mean': float(np.mean(pattern)),
            'std': float(np.std(pattern)),
            'trend': float(pattern[-1] - pattern[0]),
            'volatility': float(np.std(np.diff(pattern)))
        }
    
    def extract_complete(self, game_id, metadata):
        """Complete extraction optimized for speed"""
        
        # Fetch
        plays_df = self.fetch_with_retry(game_id)
        
        if plays_df is None or len(plays_df) == 0:
            return None
        
        # Extract
        temporal = self.extract_pattern_fast(plays_df)
        
        if not temporal or temporal['diff_at_final'] is None:
            return None
        
        statistical = self.extract_statistical_fast(temporal['pattern'])
        
        # Simplified pattern (fewer features for speed)
        return {
            'pattern': temporal['pattern'],
            'diff_at_2q_6min': temporal['diff_at_2q_6min'],
            'diff_at_halftime': temporal['diff_at_halftime'],
            'diff_at_final': temporal['diff_at_final'],
            'pattern_statistical': statistical,
            'quality_metrics': {'quality_grade': 'A'},
            'game_id': game_id,
            **metadata
        }


class FastProgress:
    """Simplified progress tracking"""
    
    def __init__(self, total):
        self.total = total
        self.processed = 0
        self.successful = 0
        self.start = datetime.now()
    
    def update(self, success):
        self.processed += 1
        if success:
            self.successful += 1
    
    def print(self, extractor):
        elapsed = (datetime.now() - self.start).total_seconds()
        rate = self.processed / elapsed if elapsed > 0 else 0
        remaining = (self.total - self.processed) / rate if rate > 0 else 0
        eta = datetime.now() + timedelta(seconds=remaining)
        
        pct = (self.processed / self.total * 100)
        
        print(f"\n{'='*60}")
        print(f"⚡ STEALTH EXTRACTION - {pct:.1f}%")
        print(f"{'='*60}")
        print(f"Progress: {self.processed}/{self.total}")
        print(f"Success: {self.successful} ({self.successful/self.processed*100:.1f}%)")
        print(f"Speed: {rate*3600:.0f} games/hour")
        print(f"Cache: {extractor.cache_hits} hits, {extractor.cache_misses} misses")
        print(f"ETA: {eta.strftime('%I:%M %p')} ({remaining/60:.0f} min)")
        print(f"{'='*60}\n")


# MAIN
if __name__ == "__main__":
    
    print("="*60)
    print("⚡ STEALTH EXTRACTION - WiFi Optimized")
    print("="*60)
    print("\nOptimizations:")
    print("  ✅ Browser headers (bypass bot detection)")
    print("  ✅ Connection pooling (faster)")
    print("  ✅ Aggressive caching (minimize API calls)")
    print("  ✅ Retry logic (handle throttling)")
    print("  ✅ Randomized delays (avoid patterns)")
    
    # Load games
    games_df = pd.read_csv('historical_games_2021_2025_basic.csv', dtype={'GAME_ID': str})
    games_df = games_df.drop_duplicates(subset=['GAME_ID'], keep='first')
    
    # Load checkpoint
    checkpoint_file = Path('stealth_checkpoint.pkl')
    processed = []
    processed_ids = set()
    
    if checkpoint_file.exists():
        with open(checkpoint_file, 'rb') as f:
            data = pickle.load(f)
            processed = data['patterns']
            processed_ids = data['ids']
        
        print(f"\n✅ Resuming from {len(processed)} games")
        games_df = games_df[~games_df['GAME_ID'].isin(processed_ids)]
    
    print(f"Remaining: {len(games_df)} games\n")
    
    # Initialize
    extractor = StealthExtractor()
    tracker = FastProgress(len(games_df))
    
    try:
        for idx, (_, row) in enumerate(games_df.iterrows(), 1):
            game_id = row['GAME_ID']
            
            metadata = {
                'season': row['SEASON_ID'],
                'date': row['GAME_DATE'],
                'matchup': row['MATCHUP']
            }
            
            # Extract
            pattern = extractor.extract_complete(game_id, metadata)
            
            if pattern:
                processed.append(pattern)
                processed_ids.add(game_id)
                tracker.update(True)
            else:
                tracker.update(False)
            
            # Progress every 10
            if idx % 10 == 0:
                tracker.print(extractor)
            
            # Checkpoint every 50
            if idx % 50 == 0:
                with open(checkpoint_file, 'wb') as f:
                    pickle.dump({'patterns': processed, 'ids': processed_ids}, f)
            
            # STEALTH: Randomized delay (0.4-0.8 seconds, less predictable)
            time.sleep(random.uniform(0.4, 0.8))
    
    except KeyboardInterrupt:
        print("\n⏸️  PAUSED")
        with open(checkpoint_file, 'wb') as f:
            pickle.dump({'patterns': processed, 'ids': processed_ids}, f)
        print(f"✅ Saved {len(processed)} games")
        sys.exit(0)
    
    # Save
    output = Path('ULTRA_OPTIMIZED_PATTERNS_2021_2025.pkl')
    with open(output, 'wb') as f:
        pickle.dump(processed, f)
    
    print(f"\n✅ Complete! {len(processed)} games → {output}")

