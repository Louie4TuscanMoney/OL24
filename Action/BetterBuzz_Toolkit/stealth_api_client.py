"""
Stealth API Client for Better Buzz Network

Reusable class for any API work at Better Buzz
Handles:
- DPI bypass (browser headers)
- Connection pooling
- Retry logic
- Caching
- Progress tracking
"""

import requests
import time
import random
import pickle
import hashlib
from pathlib import Path
from datetime import datetime, timedelta

from better_buzz_config import STEALTH_HEADERS, TIMING, CACHE_CONFIG


class StealthAPIClient:
    """
    Network-optimized API client for Better Buzz WiFi
    
    Usage:
        client = StealthAPIClient()
        data = client.get('https://api.example.com/data')
    """
    
    def __init__(self, cache_dir=None):
        """Initialize stealth client"""
        
        # Create session with stealth headers
        self.session = requests.Session()
        self.session.headers.update(STEALTH_HEADERS)
        
        # Cache setup
        self.cache_enabled = CACHE_CONFIG['enabled']
        self.cache_dir = Path(cache_dir or CACHE_CONFIG['directory'])
        self.cache_dir.mkdir(exist_ok=True)
        
        # Stats
        self.cache_hits = 0
        self.cache_misses = 0
        self.api_calls = 0
        self.failures = 0
    
    def _get_cache_key(self, url, params=None):
        """Generate cache key from URL and params"""
        key_str = f"{url}_{str(params)}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _get_from_cache(self, cache_key):
        """Retrieve from cache if exists and not expired"""
        if not self.cache_enabled:
            return None
        
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        if cache_file.exists():
            # Check age
            age_days = (datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)).days
            
            if age_days <= CACHE_CONFIG['max_age_days']:
                with open(cache_file, 'rb') as f:
                    self.cache_hits += 1
                    return pickle.load(f)
        
        return None
    
    def _save_to_cache(self, cache_key, data):
        """Save to cache"""
        if not self.cache_enabled:
            return
        
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)
    
    def get(self, url, params=None, max_retries=None):
        """
        GET request with retry logic and caching
        
        Args:
            url: API endpoint
            params: Query parameters
            max_retries: Override default retry count
            
        Returns:
            Response object or None if failed
        """
        if max_retries is None:
            max_retries = TIMING['max_retries']
        
        # Check cache
        cache_key = self._get_cache_key(url, params)
        cached = self._get_from_cache(cache_key)
        
        if cached is not None:
            return cached
        
        self.cache_misses += 1
        
        # Make request with retry
        for attempt in range(max_retries):
            try:
                self.api_calls += 1
                
                # Randomized delay (avoid pattern detection)
                if attempt > 0:
                    backoff = (TIMING['retry_base'] ** attempt) + random.uniform(0, 1)
                    time.sleep(backoff)
                
                # Make request
                response = self.session.get(url, params=params, timeout=30)
                response.raise_for_status()
                
                # Cache successful response
                self._save_to_cache(cache_key, response)
                
                # Delay before next request
                time.sleep(random.uniform(TIMING['delay_min'], TIMING['delay_max']))
                
                return response
                
            except Exception as e:
                self.failures += 1
                
                if attempt == max_retries - 1:
                    # Final attempt failed
                    return None
        
        return None
    
    def post(self, url, data=None, json=None, max_retries=None):
        """POST request with retry logic (no caching)"""
        if max_retries is None:
            max_retries = TIMING['max_retries']
        
        for attempt in range(max_retries):
            try:
                self.api_calls += 1
                
                if attempt > 0:
                    backoff = (TIMING['retry_base'] ** attempt) + random.uniform(0, 1)
                    time.sleep(backoff)
                
                response = self.session.post(url, data=data, json=json, timeout=30)
                response.raise_for_status()
                
                time.sleep(random.uniform(TIMING['delay_min'], TIMING['delay_max']))
                
                return response
                
            except Exception as e:
                self.failures += 1
                
                if attempt == max_retries - 1:
                    return None
        
        return None
    
    def get_stats(self):
        """Get client statistics"""
        return {
            'api_calls': self.api_calls,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'failures': self.failures,
            'cache_hit_rate': self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0
        }
    
    def print_stats(self):
        """Print statistics"""
        stats = self.get_stats()
        print(f"\n📊 API Client Stats:")
        print(f"   Total calls: {stats['api_calls']}")
        print(f"   Cache hits: {stats['cache_hits']}")
        print(f"   Cache misses: {stats['cache_misses']}")
        print(f"   Failures: {stats['failures']}")
        print(f"   Hit rate: {stats['cache_hit_rate']*100:.1f}%")


# Example usage
if __name__ == "__main__":
    client = StealthAPIClient()
    
    # Example API call
    response = client.get('https://httpbin.org/get')
    
    if response:
        print("✅ Request successful")
        client.print_stats()
    else:
        print("❌ Request failed")

