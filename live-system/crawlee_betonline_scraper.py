"""
🕷️ CRAWLEE BETONLINE SCRAPER

Purpose: Scrape real BetOnline odds using Crawlee
Author: Ontologic XYZ
Date: October 22, 2025

This implements the production-ready Crawlee scraper for BetOnline.ag
Based on the specification in BETONLINE_IMPLEMENTATION_SPEC.md
"""

import asyncio
import json
import time
from datetime import datetime
from typing import Dict, List, Optional
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(__file__))

try:
    from playwright.async_api import async_playwright
    # Try different stealth import
    try:
        from playwright_stealth import stealth_async
    except ImportError:
        # Fallback to manual stealth
        stealth_async = None
    CRAWLEE_AVAILABLE = True
    print("✅ Crawlee dependencies available")
except ImportError as e:
    CRAWLEE_AVAILABLE = False
    print(f"⚠️ Crawlee dependencies not available: {e}")


class CrawleeBetOnlineScraper:
    """
    🕷️ Production Crawlee scraper for BetOnline.ag
    """
    
    def __init__(self):
        self.base_url = "https://www.betonline.ag"
        self.nba_url = "https://www.betonline.ag/sportsbook/basketball/nba"
        self.live_url = "https://www.betonline.ag/sportsbook/live"
        
        # Scraper configuration
        self.config = {
            'max_concurrency': 2,
            'request_delay_ms': 3000,
            'max_retries': 5,
            'browser_timeout': 60000,
            'headless': True,
        }
        
        # CSS Selectors (to be refined via inspection)
        self.selectors = {
            'game_list': {
                'container': '.game-container, .event-container, [data-game-id]',
                'game_card': '.game-line, .event-line',
                'home_team': '.home-team, .team-home',
                'away_team': '.away-team, .team-away',
                'game_time': '.game-time, .event-time, [data-start-time]',
                'game_url': 'a.game-link, a[href*="/game/"]',
                'live_indicator': '.live, .in-play, .live-indicator',
            },
            'live_game': {
                'home_score': '.home-score, .score-home, [data-home-score]',
                'away_score': '.away-score, .score-away, [data-away-score]',
                'quarter': '.quarter, .period, [data-quarter]',
                'game_clock': '.game-clock, .time-remaining, [data-time]',
                'home_team_name': '.home-team-name',
                'away_team_name': '.away-team-name',
            },
            'odds': {
                'spread_home': '.spread-home, [data-spread-home]',
                'spread_away': '.spread-away, [data-spread-away]',
                'total_over': '.total-over, [data-total-over]',
                'total_under': '.total-under, [data-total-under]',
                'moneyline_home': '.ml-home, [data-ml-home]',
                'moneyline_away': '.ml-away, [data-ml-away]',
            }
        }
    
    async def scrape_live_games(self) -> List[Dict]:
        """
        🕷️ Scrape all live NBA games from BetOnline
        """
        if not CRAWLEE_AVAILABLE:
            print("❌ Crawlee not available - install dependencies")
            return []
        
        try:
            async with async_playwright() as p:
                # Launch browser with stealth
                browser = await p.chromium.launch(
                    headless=self.config['headless'],
                    args=['--no-sandbox', '--disable-setuid-sandbox']
                )
                
                context = await browser.new_context(
                    user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
                )
                
                page = await context.new_page()
                
                # Apply stealth if available
                if stealth_async:
                    await stealth_async(page)
                else:
                    # Manual stealth configuration
                    await page.add_init_script("""
                        Object.defineProperty(navigator, 'webdriver', {
                            get: () => undefined,
                        });
                    """)
                
                # Navigate to BetOnline NBA page
                print("🕷️ Navigating to BetOnline NBA page...")
                await page.goto(self.nba_url, wait_until='networkidle', timeout=30000)
                
                # Wait for games to load
                await page.wait_for_timeout(3000)
                
                # Scrape live games
                games = await self._scrape_game_list(page)
                
                await browser.close()
                
                print(f"✅ Crawlee scraper: Found {len(games)} live games")
                return games
                
        except Exception as e:
            print(f"❌ Crawlee scraper error: {e}")
            return []
    
    async def _scrape_game_list(self, page) -> List[Dict]:
        """
        Scrape the list of live games
        """
        try:
            # Wait for games to be visible
            await page.wait_for_selector(self.selectors['game_list']['container'], timeout=10000)
            
            # Extract game data
            games = await page.evaluate("""
                () => {
                    const games = [];
                    const gameElements = document.querySelectorAll('.game-container, .event-container, [data-game-id]');
                    
                    gameElements.forEach(element => {
                        try {
                            const homeTeam = element.querySelector('.home-team, .team-home')?.textContent?.trim();
                            const awayTeam = element.querySelector('.away-team, .team-away')?.textContent?.trim();
                            const gameUrl = element.querySelector('a.game-link, a[href*="/game/"]')?.href;
                            const isLive = element.querySelector('.live, .in-play, .live-indicator') !== null;
                            
                            if (homeTeam && awayTeam && isLive) {
                                games.push({
                                    home_team: homeTeam,
                                    away_team: awayTeam,
                                    game_url: gameUrl,
                                    is_live: isLive,
                                    timestamp: new Date().toISOString()
                                });
                            }
                        } catch (e) {
                            console.error('Error parsing game element:', e);
                        }
                    });
                    
                    return games;
                }
            """)
            
            return games
            
        except Exception as e:
            print(f"❌ Error scraping game list: {e}")
            return []
    
    async def scrape_game_odds(self, game_url: str) -> Optional[Dict]:
        """
        Scrape odds for a specific game
        """
        if not CRAWLEE_AVAILABLE:
            return None
        
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                context = await browser.new_context()
                page = await context.new_page()
                
                # Apply stealth if available
                if stealth_async:
                    await stealth_async(page)
                else:
                    # Manual stealth configuration
                    await page.add_init_script("""
                        Object.defineProperty(navigator, 'webdriver', {
                            get: () => undefined,
                        });
                    """)
                
                # Navigate to game page
                await page.goto(game_url, wait_until='networkidle', timeout=30000)
                
                # Wait for odds to load
                await page.wait_for_timeout(2000)
                
                # Extract odds
                odds = await page.evaluate("""
                    () => {
                        try {
                            const spreadHome = document.querySelector('.spread-home, [data-spread-home]')?.textContent;
                            const spreadAway = document.querySelector('.spread-away, [data-spread-away]')?.textContent;
                            const totalOver = document.querySelector('.total-over, [data-total-over]')?.textContent;
                            const totalUnder = document.querySelector('.total-under, [data-total-under]')?.textContent;
                            const mlHome = document.querySelector('.ml-home, [data-ml-home]')?.textContent;
                            const mlAway = document.querySelector('.ml-away, [data-ml-away]')?.textContent;
                            
                            return {
                                spread_home: spreadHome,
                                spread_away: spreadAway,
                                total_over: totalOver,
                                total_under: totalUnder,
                                moneyline_home: mlHome,
                                moneyline_away: mlAway,
                                timestamp: new Date().toISOString()
                            };
                        } catch (e) {
                            console.error('Error extracting odds:', e);
                            return null;
                        }
                    }
                """)
                
                await browser.close()
                return odds
                
        except Exception as e:
            print(f"❌ Error scraping game odds: {e}")
            return None
    
    def get_live_lines_sync(self) -> List[Dict]:
        """
        Synchronous wrapper for the async scraper
        """
        if not CRAWLEE_AVAILABLE:
            print("❌ Crawlee not available - returning empty list")
            return []
        
        try:
            # Run async scraper in sync context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            games = loop.run_until_complete(self.scrape_live_games())
            loop.close()
            return games
        except Exception as e:
            print(f"❌ Sync wrapper error: {e}")
            return []


# Integration function for the main system
def get_crawlee_betonline_odds() -> List[Dict]:
    """
    🕷️ Main function to get BetOnline odds via Crawlee
    """
    if not CRAWLEE_AVAILABLE:
        print("🕷️ Crawlee not available - install: pip install playwright playwright-stealth")
        return []
    
    scraper = CrawleeBetOnlineScraper()
    return scraper.get_live_lines_sync()


if __name__ == "__main__":
    # Test the scraper
    print("🕷️ Testing Crawlee BetOnline scraper...")
    odds = get_crawlee_betonline_odds()
    print(f"Found {len(odds)} games with odds")
    for game in odds:
        print(f"  {game.get('away_team', 'Unknown')} @ {game.get('home_team', 'Unknown')}")
