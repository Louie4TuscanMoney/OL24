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
    """Fetch ALL games (live + completed) from NBA scoreboard using nba_api"""
    try:
        from nba_api.live.nba.endpoints import scoreboard
        
        # Use nba_api library (same as /api/live-games endpoint)
        games_data = scoreboard.ScoreBoard()
        games_dict = games_data.get_dict()
        
        games = games_dict.get('scoreboard', {}).get('games', [])
        
        print(f"   📡 Fetched {len(games)} games from nba_api")
        
        return games
        
    except Exception as e:
        print(f"   ❌ Error fetching scoreboard: {e}")
        import traceback
        traceback.print_exc()
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
    """Process a single live game - fetch PBP and check for Mamba trigger"""
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
    
    # 2. Check if Mamba should trigger (Q2 6:00)
    if period == 2 and clock.startswith('6:0'):
        print(f"      ⚡ MAMBA TRIGGER DETECTED!")
        trigger_mamba_prediction(game_id, game)
    else:
        print(f"      ⏳ Waiting for Q2 6:00 (current: Q{period} {clock})")


def fetch_and_store_playbyplay(game_id):
    """Fetch play-by-play from NBA API and store in database"""
    try:
        # Fetch play-by-play
        pbp_url = f"https://cdn.nba.com/static/json/liveData/playbyplay/playbyplay_{game_id}.json"
        response = requests.get(pbp_url, timeout=10)
        
        if response.status_code != 200:
            return False
        
        pbp_data = response.json()
        actions = pbp_data.get('game', {}).get('actions', [])
        
        if not actions:
            return False
        
        # Store in database
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        
        stored_count = 0
        
        for action in actions:
            try:
                # Calculate time elapsed
                period = action.get('period', 1)
                clock = action.get('clock', '12:00')
                
                # Convert clock to seconds (e.g., "6:34" -> 394)
                if ':' in clock:
                    parts = clock.split(':')
                    mins = int(parts[0])
                    secs = int(parts[1]) if len(parts) > 1 else 0
                    period_seconds = 720 - (mins * 60 + secs)  # Seconds into period
                else:
                    period_seconds = 0
                
                time_elapsed = (period - 1) * 720 + period_seconds
                
                # Get scores
                home_score = action.get('scoreHome', 0) or 0
                away_score = action.get('scoreAway', 0) or 0
                
                # Insert into database
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
                        time_elapsed_seconds = EXCLUDED.time_elapsed_seconds
                """, (
                    game_id,
                    action.get('actionNumber', 0),
                    period,
                    clock,
                    time_elapsed,
                    action.get('actionType', ''),
                    action.get('description', ''),
                    str(action.get('personId', '')) if action.get('personId') else None,
                    str(action.get('teamId', '')) if action.get('teamId') else None,
                    home_score,
                    away_score,
                    home_score - away_score,
                    json.dumps(action)
                ))
                
                stored_count += 1
                
            except Exception as e:
                continue
        
        conn.commit()
        cur.close()
        conn.close()
        
        print(f"      ✅ Stored {stored_count} play-by-play events")
        return True
        
    except Exception as e:
        print(f"      ❌ PBP error: {str(e)[:50]}")
        return False


def trigger_mamba_prediction(game_id, game):
    """Trigger Mamba prediction at Q2 6:00"""
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
        
        # Get last 18 minutes (1080 seconds) of play-by-play
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
        
        print(f"      ✅ MAMBA PREDICTION: {prediction:+.1f}")
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

