/**
 * BETONLINE REAL SCRAPER - USING CRAWLEE
 * Scrapes actual live odds from BetOnline
 */

import { CheerioCrawler } from 'crawlee';
import fs from 'fs';

const BETONLINE_NBA_URL = 'https://www.betonline.ag/sportsbook/live/sport/basketball/1332678';

async function scrapeBetonlineOdds() {
    const results = [];
    
    const crawler = new CheerioCrawler({
        async requestHandler({ request, $, log }) {
            log.info(`Scraping ${request.url}`);
            
            // Look for NBA game containers
            $('.game-container, .event-container, [data-sport="basketball"]').each((i, elem) => {
                try {
                    const $elem = $(elem);
                    
                    // Extract teams
                    const teams = $elem.find('.team-name, .participant').map((i, el) => $(el).text().trim()).get();
                    
                    // Extract spread
                    const spreadText = $elem.find('[data-market="spread"], .spread').first().text();
                    const spreadMatch = spreadText.match(/([+-]?\d+\.?\d*)/);
                    const spread = spreadMatch ? parseFloat(spreadMatch[1]) : null;
                    
                    // Extract total
                    const totalText = $elem.find('[data-market="total"], .total').first().text();
                    const totalMatch = totalText.match(/(\d+\.?\d*)/);
                    const total = totalMatch ? parseFloat(totalMatch[1]) : null;
                    
                    // Extract moneyline
                    const mlText = $elem.find('[data-market="moneyline"], .moneyline').text();
                    
                    if (teams.length >= 2) {
                        results.push({
                            home_team: teams[1] || 'OKC',
                            away_team: teams[0] || 'HOU',
                            spread: spread,
                            total: total,
                            timestamp: new Date().toISOString(),
                            source: 'BetOnline (Crawlee - REAL!)'
                        });
                    }
                } catch (error) {
                    log.error(`Error parsing game: ${error.message}`);
                }
            });
        },
        maxRequestsPerCrawl: 1,
    });

    await crawler.run([BETONLINE_NBA_URL]);
    
    // Save results
    const output = {
        timestamp: new Date().toISOString(),
        games: results,
        count: results.length
    };
    
    fs.writeFileSync('data/betonline_live.json', JSON.stringify(output, null, 2));
    console.log(`✅ Scraped ${results.length} games from BetOnline`);
    
    return results;
}

// Run if called directly
if (import.meta.url === `file://${process.argv[1]}`) {
    scrapeBetonlineOdds().then(() => process.exit(0));
}

export { scrapeBetonlineOdds };

