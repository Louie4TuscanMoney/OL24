"""
CONFIGURATION MANAGEMENT
Centralized config for all environment variables and settings
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class APIConfig:
    """API endpoint configuration"""
    ESPN_SCOREBOARD: str = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
    ESPN_TIMEOUT: int = 3
    ESPN_RETRY_ATTEMPTS: int = 3
    
    # Fallback APIs
    NBA_API_ENABLED: bool = True
    
    # Request settings
    REQUEST_TIMEOUT: int = 3
    MAX_RETRIES: int = 3


@dataclass
class DatabaseConfig:
    """Database configuration"""
    DATABASE_URL: Optional[str] = os.getenv('DATABASE_URL')
    CONNECTION_TIMEOUT: int = 3
    KEEPALIVES: bool = True
    KEEPALIVES_IDLE: int = 30
    KEEPALIVES_INTERVAL: int = 10


@dataclass
class MLConfig:
    """ML Model configuration"""
    MODEL_PATH: str = "/tmp/MAMBA_MENTALITY_SYSTEM.pkl"
    MODEL_GOOGLE_DRIVE_ID: str = os.getenv('MODEL_GOOGLE_DRIVE_ID', '1J4vvFK_cMRi7MzlT_6sGWDxPCR8QHQOS')
    MODEL_EXPECTED_SIZE_MB: float = 300.0
    
    # Feature extraction
    FEATURE_COUNT: int = 33
    PBP_WINDOW_MINUTES: int = 18
    
    # Prediction settings
    Q2_6MIN_WINDOW_START: str = "6:00"
    Q2_6MIN_WINDOW_END: str = "5:00"


@dataclass
class WebSocketConfig:
    """WebSocket configuration"""
    UPDATE_INTERVAL_SECONDS: float = 1.0
    MAX_CONNECTIONS: int = 100
    PING_INTERVAL: int = 30


@dataclass
class AppConfig:
    """Main application configuration"""
    # Sub-configs
    api: APIConfig = APIConfig()
    database: DatabaseConfig = DatabaseConfig()
    ml: MLConfig = MLConfig()
    websocket: WebSocketConfig = WebSocketConfig()
    
    # General settings
    ENV: str = os.getenv('ENV', 'production')
    DEBUG: bool = os.getenv('DEBUG', 'false').lower() == 'true'
    LOG_LEVEL: str = os.getenv('LOG_LEVEL', 'INFO')
    
    # Feature flags
    ENABLE_ONTORISK: bool = os.getenv('ENABLE_ONTORISK', 'false').lower() == 'true'
    ENABLE_BETONLINE_SCRAPING: bool = False  # Disabled - manual entry only
    
    # Performance
    CACHE_TTL_SECONDS: float = 1.0
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        if not self.database.DATABASE_URL:
            print("⚠️  DATABASE_URL not set - database features disabled")
        
        if not os.path.exists(self.ml.MODEL_PATH) and not self.ml.MODEL_GOOGLE_DRIVE_ID:
            print("⚠️  MODEL_PATH not found and no GOOGLE_DRIVE_ID - ML predictions disabled")


# Global config instance
config = AppConfig()


# ============================================================================
# CONFIGURATION HELPERS
# ============================================================================

def get_espn_headers() -> dict:
    """Get headers for ESPN API requests (with user-agent rotation)"""
    import random
    
    user_agents = [
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    ]
    
    return {
        'User-Agent': random.choice(user_agents),
        'Accept': 'application/json',
        'Accept-Language': 'en-US,en;q=0.9',
        'Referer': 'https://www.espn.com/'
    }


def validate_config() -> bool:
    """Validate that all critical config is set"""
    errors = []
    
    if not config.api.ESPN_SCOREBOARD:
        errors.append("ESPN_SCOREBOARD URL not configured")
    
    if config.ENV == 'production' and not config.database.DATABASE_URL:
        errors.append("DATABASE_URL required in production")
    
    if errors:
        print("❌ Configuration errors:")
        for error in errors:
            print(f"   - {error}")
        return False
    
    print("✅ Configuration valid")
    return True


if __name__ == "__main__":
    # Test configuration
    print("Testing configuration...")
    print(f"ENV: {config.ENV}")
    print(f"ESPN Scoreboard: {config.api.ESPN_SCOREBOARD}")
    print(f"Database: {'Connected' if config.database.DATABASE_URL else 'Not configured'}")
    print(f"ML Model Path: {config.ml.MODEL_PATH}")
    print(f"WebSocket Interval: {config.websocket.UPDATE_INTERVAL_SECONDS}s")
    
    validate_config()

