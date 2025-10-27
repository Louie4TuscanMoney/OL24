"""
Better Buzz Network Configuration
Optimized settings for coffee shop WiFi API work
"""

# Network-optimized headers (bypass DPI)
STEALTH_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'en-US,en;q=0.9',
    'Accept-Encoding': 'gzip, deflate, br',
    'Referer': 'https://www.nba.com/',
    'Origin': 'https://www.nba.com',
    'Connection': 'keep-alive',
    'Cache-Control': 'no-cache',
    'Pragma': 'no-cache'
}

# Timing configuration (based on observed performance)
TIMING = {
    'delay_min': 0.4,  # Minimum delay between requests (seconds)
    'delay_max': 0.8,  # Maximum delay (randomized)
    'retry_base': 2.0,  # Exponential backoff base
    'max_retries': 3,
    'checkpoint_interval': 50,  # Save every N items
    'progress_report_interval': 10  # Report every N items
}

# Optimal time windows (based on Better Buzz observations)
OPTIMAL_HOURS = {
    'weekday_morning': {
        'hours': '6:00-9:00 AM',
        'expected_speed': 7000,  # items/hour
        'quality': 'EXCELLENT'
    },
    'weekday_late_morning': {
        'hours': '9:00-11:00 AM',
        'expected_speed': 4000,
        'quality': 'GOOD'
    },
    'lunch_rush': {
        'hours': '11:00 AM-2:00 PM',
        'expected_speed': 1500,
        'quality': 'POOR'
    },
    'afternoon': {
        'hours': '2:00-5:00 PM',
        'expected_speed': 2500,
        'quality': 'MEDIUM'
    },
    'weekend': {
        'hours': 'All day',
        'expected_speed': 1500,
        'quality': 'POOR'
    }
}

# Cache configuration
CACHE_CONFIG = {
    'enabled': True,
    'directory': '.better_buzz_cache',
    'max_age_days': 7,  # Cache expires after 7 days
    'compress': True
}

# Watchdog configuration
WATCHDOG_CONFIG = {
    'check_interval': 60,  # Check every 60 seconds
    'stuck_threshold': 60,  # Restart if no progress for 60 sec
    'max_consecutive_failures': 3,  # Give up after 3 restart failures
    'log_file': 'better_buzz_watchdog.log'
}

