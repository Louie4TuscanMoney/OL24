#!/usr/bin/env python3
"""
Autonomous Mamba Prediction System

Runs every 30 seconds on Railway cron
Fetches play-by-play data and triggers Mamba at Q2 6:00
"""

import os
import requests
import psycopg2
from datetime import datetime
import json
import numpy as np

DATABASE_URL = os.getenv('DATABASE_URL')

def main():
    """Main cron job - runs every 30 seconds"""
    print(f"\n{'='*80}")
    print(f"🤖 MAMBA AUTONOMOUS CRON - {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*80}\n")
    
    try:
        # 1. Fetch ALL games (live + completed)
        all_games = fetch_all_games()
        
        live_games = [g for g in all_games if g.get('gameStatus') == 2]
        completed_games = [g for g in all_games if g.get('gameStatus') == 3]
        
        if not live_games and not completed_games:
            print("   ℹ️  No games right now")
            return
        
        print(f"   📡 Found {len(live_games)} live games, {len(completed_games)} completed")
        
        # 2. Process each LIVE game
        for game in live_games:
            process_live_game(game)
        
        # 3. Process COMPLETED games (update results)
        for game in completed_games:
            update_game_result(game)
        
        print(f"\n{'='*80}")
        print("✅ Cron cycle complete")
        print(f"{'='*80}\n")
        
    except Exception as e:
        print(f"❌ Cron error: {e}")
        import traceback
        traceback.print_exc()


def fetch_all_games():
    """
    Fetch ALL games (live + completed) from ESPN API
    
    HARDENED FOR ZERO DOWNTIME:
    - 5 retry attempts with exponential backoff
    - 10 second timeout
    - Data validation
    - Never fails silently
    """
    import time
    
    espn_url = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
    headers = {
        "User-Agent": "OntologicXYZ/1.0 (+https://ontologicxyz.com)"
    }
    max_retries = 5
    backoffs = [0.5, 1.0, 2.0, 4.0, 8.0]
    
    # Retry loop
    for attempt in range(max_retries):
        try:
            response = requests.get(espn_url, timeout=10, headers=headers)
            
            if response.status_code != 200:
                print(f"   ⚠️  ESPN API returned {response.status_code} (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    time.sleep(backoffs[attempt])
                    continue
                return []
            
            data = response.json()
            espn_events = data.get('events', [])
            
            # Map ESPN format to our expected format (similar to nba_api structure)
            games = []
            for event in espn_events:
                try:
                    competition = event.get('competitions', [{}])[0]
                    status = event.get('status', {})
                    competitors = competition.get('competitors', [])
                    
                    # Find home and away teams
                    home_team = next((c for c in competitors if c.get('homeAway') == 'home'), {})
                    away_team = next((c for c in competitors if c.get('homeAway') == 'away'), {})
                    
                    # Determine game status
                    state_type = status.get('type', {}).get('state', 'pre')
                    if state_type == 'in':
                        game_status = 2  # Live
                    elif state_type == 'post':
                        game_status = 3  # Final
                    else:
                        game_status = 1  # Scheduled
                    
                    # Parse scores with validation
                    try:
                        home_score = int(home_team.get('score', 0))
                        away_score = int(away_team.get('score', 0))
                    except (ValueError, TypeError):
                        home_score = 0
                        away_score = 0
                    
                    # Validate required fields
                    game_id = event.get('id')
                    home_abbr = home_team.get('team', {}).get('abbreviation', '')
                    away_abbr = away_team.get('team', {}).get('abbreviation', '')
                    
                    if not game_id or not home_abbr or not away_abbr:
                        print(f"   ⚠️  Skipping game with missing data")
                        continue
                    
                    games.append({
                        'gameId': game_id,
                        'gameStatus': game_status,
                        'period': status.get('period', 0),
                        'gameClock': status.get('displayClock', ''),
                        'homeTeam': {
                            'teamTricode': home_abbr,
                            'score': home_score
                        },
                        'awayTeam': {
                            'teamTricode': away_abbr,
                            'score': away_score
                        }
                    })
                except Exception as e:
                    print(f"   ⚠️  Error parsing game: {e}")
                    continue
            
            print(f"   📡 Fetched {len(games)} games from ESPN API")
            return games
            
        except Exception as e:
            print(f"   ❌ Error fetching ESPN scoreboard (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                print(f"   ⏳ Retrying in {backoffs[attempt]}s...")
                time.sleep(backoffs[attempt])
            else:
                import traceback
                traceback.print_exc()
                return []
    
    return []


def update_game_result(game):
    """Update final result for completed games with Mamba predictions"""
    game_id = game['gameId']
    
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        
        # Check if this game has a Mamba prediction
        cur.execute("""
            SELECT game_id, prediction, home_score, away_score
            FROM mamba_game_cache
            WHERE game_id = %s
            AND final_home_score IS NULL
        """, (game_id,))
        
        result = cur.fetchone()
        
        if not result:
            # No Mamba prediction for this game, skip
            cur.close()
            conn.close()
            return
        
        # Get final scores
        final_home_score = game['homeTeam']['score']
        final_away_score = game['awayTeam']['score']
        final_margin = final_home_score - final_away_score
        
        # Get halftime scores (from game detail API)
        halftime_data = get_halftime_scores(game_id)
        h1_home = halftime_data.get('h1_home', 0) if halftime_data else 0
        h1_away = halftime_data.get('h1_away', 0) if halftime_data else 0
        
        # Calculate 2H scores
        h2_home = final_home_score - h1_home
        h2_away = final_away_score - h1_away
        h2_margin = h2_home - h2_away
        
        # Calculate Mamba performance
        mamba_prediction = result[1]
        mamba_error = abs(mamba_prediction - final_margin)
        mamba_correct = mamba_error <= 5  # Within 5 points = correct
        
        # Update mamba_game_cache
        cur.execute("""
            UPDATE mamba_game_cache
            SET final_home_score = %s,
                final_away_score = %s,
                actual_margin = %s,
                h2_home_score = %s,
                h2_away_score = %s,
                h2_margin = %s,
                mamba_correct = %s,
                mamba_error = %s,
                updated_at = NOW()
            WHERE game_id = %s
        """, (
            final_home_score, final_away_score, final_margin,
            h2_home, h2_away, h2_margin,
            mamba_correct, mamba_error,
            game_id
        ))
        
        conn.commit()
        cur.close()
        conn.close()
        
        print(f"   ✅ Updated result for {game_id}")
        print(f"      Prediction: {mamba_prediction:+.1f}, Actual: {final_margin:+d}")
        print(f"      Error: {mamba_error:.1f}, Correct: {mamba_correct}")
        
    except Exception as e:
        print(f"   ❌ Error updating result for {game_id}: {e}")


def get_halftime_scores(game_id):
    """Get halftime scores from game summary"""
    try:
        url = f"https://cdn.nba.com/static/json/liveData/boxscore/boxscore_{game_id}.json"
        response = requests.get(url, timeout=10)
        
        if response.status_code != 200:
            return None
        
        data = response.json()
        game = data.get('game', {})
        
        # Get period scores
        home_periods = game.get('homeTeam', {}).get('periods', [])
        away_periods = game.get('awayTeam', {}).get('periods', [])
        
        if len(home_periods) >= 2 and len(away_periods) >= 2:
            h1_home = sum(p.get('score', 0) for p in home_periods[:2])  # Q1 + Q2
            h1_away = sum(p.get('score', 0) for p in away_periods[:2])
            
            return {
                'h1_home': h1_home,
                'h1_away': h1_away
            }
        
        return None
        
    except Exception as e:
        print(f"   ⚠️  Could not get halftime scores: {e}")
        return None


def process_live_game(game):
    """Process a single live game - fetch PBP, update win prob, check for Mamba trigger"""
    game_id = game['gameId']
    period = game.get('period', 0)
    clock = game.get('gameClock', '')
    
    home_team = game['homeTeam']['teamTricode']
    away_team = game['awayTeam']['teamTricode']
    home_score = game['homeTeam']['score']
    away_score = game['awayTeam']['score']
    
    print(f"\n   🏀 {away_team} @ {home_team} - Q{period} {clock}")
    print(f"      Score: {away_score}-{home_score}")
    
    # 1. Fetch and store play-by-play
    pbp_stored = fetch_and_store_playbyplay(game_id)
    
    if not pbp_stored:
        print(f"      ⚠️  Failed to store play-by-play")
        return
    
    # 2. Update minute-by-minute win probability (if we have enough data)
    update_win_probability(game_id)
    
    # 3. Check if Mamba should trigger (ONLY at Q2 6:00)
    if period == 2 and clock.startswith('6:0'):
        print(f"      ⚡ MAMBA Q2 6:00 TRIGGER DETECTED!")
        trigger_mamba_prediction(game_id, game)
    else:
        print(f"      ⏳ Waiting for Q2 6:00 (current: Q{period} {clock})")


def fetch_and_store_playbyplay(game_id):
    """Store live scores every minute (simpler than parsing play-by-play)"""
    try:
        # Get current game state from ESPN
        espn_url = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
        response = requests.get(espn_url, timeout=10)
        
        if response.status_code != 200:
            return False
        
        data = response.json()
        events = data.get('events', [])
        
        # Find this specific game
        game = None
        for event in events:
            if str(event.get('id')) == str(int(game_id)):  # Match ESPN game ID format
                game = event
                break
        
        if not game:
            print(f"      ⚠️  Game {game_id} not found in ESPN data")
            return False
        
        competition = game.get('competitions', [{}])[0]
        status = game.get('status', {})
        competitors = competition.get('competitors', [])
        
        home_team = next((c for c in competitors if c.get('homeAway') == 'home'), {})
        away_team = next((c for c in competitors if c.get('homeAway') == 'away'), {})
        
        home_score = int(home_team.get('score', 0))
        away_score = int(away_team.get('score', 0))
        period = status.get('period', 0)
        clock = status.get('displayClock', '')
        
        # Calculate time elapsed
        if ':' in clock:
            parts = clock.split(':')
            mins = int(parts[0])
            secs = int(parts[1]) if len(parts) > 1 else 0
            period_seconds = 720 - (mins * 60 + secs)
        else:
            period_seconds = 0
        
        time_elapsed = (period - 1) * 720 + period_seconds
        
        # Store in database - create a unique timestamp for this minute
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        
        # Create a unique event_num based on current minute
        # This way we store one snapshot per minute
        event_num = int(time_elapsed / 60)  # Minute number (0, 1, 2, ...)
        
        # Insert or update current minute's score snapshot
        cur.execute("""
            INSERT INTO play_by_play (
                game_id, event_num, period, clock,
                time_elapsed_seconds, event_type,
                description, player_id, team_id,
                home_score, away_score, score_margin,
                event_data
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
            ON CONFLICT (game_id, event_num) DO UPDATE SET
                home_score = EXCLUDED.home_score,
                away_score = EXCLUDED.away_score,
                score_margin = EXCLUDED.score_margin,
                clock = EXCLUDED.clock,
                time_elapsed_seconds = EXCLUDED.time_elapsed_seconds,
                period = EXCLUDED.period
        """, (
            game_id,
            event_num,
            period,
            clock,
            time_elapsed,
            'score_snapshot',  # event_type
            f'Score at Q{period} {clock}',  # description
            None,  # player_id
            None,  # team_id
            home_score,
            away_score,
            home_score - away_score,
            json.dumps({
                'home_score': home_score,
                'away_score': away_score,
                'period': period,
                'clock': clock,
                'time_elapsed': time_elapsed
            })
        ))
        
        conn.commit()
        cur.close()
        conn.close()
        
        print(f"      ✅ Stored score snapshot (minute {event_num}, {time_elapsed}s elapsed)")
        return True
        
    except Exception as e:
        print(f"      ❌ PBP error: {str(e)[:50]}")
        return False


def update_win_probability(game_id):
    """Update win probability every minute using Mamba 33 features"""
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        
        # Get recent snapshots (last 12 minutes minimum)
        cur.execute("""
            SELECT home_score, away_score, score_margin, time_elapsed_seconds, period
            FROM play_by_play
            WHERE game_id = %s
            ORDER BY time_elapsed_seconds DESC
            LIMIT 12
        """, (game_id,))
        
        pbp_data = cur.fetchall()
        
        if len(pbp_data) < 6:  # Need at least 6 minutes
            return
        
        # Reverse to get chronological order
        pbp_data = list(reversed(pbp_data))
        
        # Extract 33 features
        features = extract_mamba_features_from_pbp(pbp_data)
        
        # Simple prediction: pattern_mean as margin prediction
        margin_pred = features[0]  # pattern_mean
        
        # Convert margin to win probability using sigmoid
        # margin of +5 ≈ 75% home win, -5 ≈ 25% home win (50% at 0)
        win_prob_home = 1 / (1 + np.exp(-0.2 * margin_pred))  # Sigmoid scaling
        win_prob_away = 1 - win_prob_home
        
        # Get current period and time
        current_period = pbp_data[-1][4]
        current_time = pbp_data[-1][3]
        
        # Store/update win probability timeline
        cur.execute("""
            CREATE TABLE IF NOT EXISTS win_probability_timeline (
                id SERIAL PRIMARY KEY,
                game_id VARCHAR(20) NOT NULL,
                period INT NOT NULL,
                time_elapsed_seconds INT NOT NULL,
                home_win_prob DECIMAL(5,2) NOT NULL,
                away_win_prob DECIMAL(5,2) NOT NULL,
                margin_prediction DECIMAL(5,2),
                confidence DECIMAL(5,2),
                created_at TIMESTAMP DEFAULT NOW(),
                UNIQUE(game_id, time_elapsed_seconds)
            )
        """)
        
        cur.execute("""
            INSERT INTO win_probability_timeline (
                game_id, period, time_elapsed_seconds, 
                home_win_prob, away_win_prob, margin_prediction, confidence
            ) VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (game_id, time_elapsed_seconds) DO UPDATE SET
                home_win_prob = EXCLUDED.home_win_prob,
                away_win_prob = EXCLUDED.away_win_prob,
                margin_prediction = EXCLUDED.margin_prediction,
                confidence = EXCLUDED.confidence,
                created_at = NOW()
        """, (
            game_id, current_period, current_time,
            float(win_prob_home * 100),
            float(win_prob_away * 100),
            float(margin_pred),
            75.0  # Confidence
        ))
        
        conn.commit()
        cur.close()
        conn.close()
        
        print(f"      📊 Win Prob: Home {win_prob_home*100:.0f}% | Away {win_prob_away*100:.0f}%")
        
    except Exception as e:
        print(f"      ⚠️  Win prob update error: {str(e)[:50]}")


def trigger_mamba_prediction(game_id, game):
    """Trigger Mamba prediction ONLY at Q2 6:00"""
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        
        # Check if already triggered
        cur.execute("""
            SELECT game_id FROM mamba_game_cache 
            WHERE game_id = %s
        """, (game_id,))
        
        if cur.fetchone():
            print(f"      ℹ️  Mamba already ran for this game")
            cur.close()
            conn.close()
            return
        
        # Get available play-by-play data (need 18 minutes for Q2 6:00)
        cur.execute("""
            SELECT home_score, away_score, score_margin, time_elapsed_seconds
            FROM play_by_play
            WHERE game_id = %s
            AND time_elapsed_seconds <= 1080
            ORDER BY time_elapsed_seconds ASC
        """, (game_id,))
        
        pbp_data = cur.fetchall()
        
        if len(pbp_data) < 10:
            print(f"      ⚠️  Not enough data ({len(pbp_data)} events, need 10+)")
            cur.close()
            conn.close()
            return
        
        print(f"      📊 Extracting features from {len(pbp_data)} events...")
        
        # Extract 33 Mamba features
        features = extract_mamba_features_from_pbp(pbp_data)
        
        # Make prediction (simplified - just use basic pattern for now)
        # In production, this would load the actual Mamba model
        prediction = features[0]  # Use pattern_mean as proxy
        
        # Store prediction
        cur.execute("""
            INSERT INTO mamba_game_cache (
                game_id, features, prediction, confidence,
                triggered_at, period, clock,
                home_team_id, away_team_id,
                home_score, away_score, current_margin
            ) VALUES (%s, %s, %s, %s, NOW(), 2, '6:00', %s, %s, %s, %s, %s)
        """, (
            game_id,
            json.dumps(features.tolist()),
            float(prediction),
            75.0,  # Placeholder confidence
            str(game['homeTeam']['teamId']),
            str(game['awayTeam']['teamId']),
            game['homeTeam']['score'],
            game['awayTeam']['score'],
            game['homeTeam']['score'] - game['awayTeam']['score']
        ))
        
        conn.commit()
        cur.close()
        conn.close()
        
        print(f"      ✅ MAMBA PREDICTION (Q2 6:00): {prediction:+.1f}")
        print(f"      💾 Stored in mamba_game_cache")
        
    except Exception as e:
        print(f"      ❌ Mamba trigger error: {e}")
        import traceback
        traceback.print_exc()


def extract_mamba_features_from_pbp(pbp_data):
    """
    Extract 33 Mamba features from play-by-play data
    
    Simplified version - extracts basic pattern features
    Full version would include spectral analysis, autocorrelation, etc.
    """
    # Extract score differentials
    margins = np.array([row[2] for row in pbp_data])
    times = np.array([row[3] for row in pbp_data])
    
    # 1. Pattern Statistics (12 features)
    pattern_mean = np.mean(margins)
    pattern_std = np.std(margins)
    pattern_min = np.min(margins)
    pattern_max = np.max(margins)
    pattern_range = pattern_max - pattern_min
    
    # Trend
    from sklearn.linear_model import LinearRegression
    if len(times) > 1:
        lr = LinearRegression()
        lr.fit(times.reshape(-1, 1), margins)
        trend = lr.coef_[0]
    else:
        trend = 0
    
    # Velocity
    velocity = np.mean(np.diff(margins)) if len(margins) > 1 else 0
    
    # Acceleration
    acceleration = np.mean(np.diff(np.diff(margins))) if len(margins) > 2 else 0
    
    # Volatility
    volatility = pattern_std / (abs(pattern_mean) + 1)
    
    # Momentum
    momentum = pattern_mean * velocity
    
    # Lead changes
    lead_changes = np.sum(np.diff(np.sign(margins)) != 0)
    
    # Max swing
    max_swing = pattern_range
    
    # 2-5. Simplified features (would be more complex in production)
    features = np.array([
        pattern_mean, pattern_std, pattern_min, pattern_max, pattern_range,
        trend, velocity, acceleration, volatility, momentum, lead_changes, max_swing,
        # Spectral (6 placeholders)
        0, 0, 0, 0, 0, 0,
        # Autocorrelation (3 placeholders)
        0, 0, 0,
        # Advanced (8 placeholders)
        0, 0, 0, 0, 0, 0, 0, 0,
        # Form (6 placeholders)
        0, 0, 0, 0, 0, 0
    ])
    
    return features


if __name__ == "__main__":
    main()

