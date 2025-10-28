#!/usr/bin/env python3
"""
Monitor NBA games for Q2 6:00 trigger and test full prediction pipeline
"""

import time
import numpy as np
from datetime import datetime
from nba_api.live.nba.endpoints import scoreboard, playbyplay

def check_games():
    """Check all games for Q2 6:00 trigger"""
    
    board = scoreboard.ScoreBoard()
    games = board.games.get_dict()
    
    print(f"\n{'=' * 80}")
    print(f"⏰ {datetime.now().strftime('%H:%M:%S')} - Checking games...")
    print(f"{'=' * 80}")
    
    triggered_games = []
    
    for game in games:
        game_id = game['gameId']
        home = game['homeTeam']['teamTricode']
        away = game['awayTeam']['teamTricode']
        period = game.get('period', 0)
        clock = game.get('gameClock', '')
        home_score = game['homeTeam']['score']
        away_score = game['awayTeam']['score']
        
        status = "Not started"
        
        if period > 0:
            status = f"Q{period} {clock}"
            
            # Check if we're at Q2 6:00 or past
            if period >= 2:
                if period == 2 and clock and 'PT' in clock:
                    try:
                        minutes = int(clock.split('M')[0].replace('PT', ''))
                        seconds_part = clock.split('M')[1].replace('S', '')
                        seconds = int(seconds_part.split('.')[0]) if '.' in seconds_part else int(seconds_part)
                        
                        print(f"🏀 {away} @ {home} | {away_score}-{home_score} | Q2 {minutes}:{seconds:02d}")
                        
                        if minutes <= 6:
                            print(f"   🚨 TRIGGER READY! Past Q2 6:00!")
                            triggered_games.append((game_id, home, away, home_score, away_score))
                        else:
                            remaining = (minutes - 6) * 60 + seconds
                            print(f"   ⏳ {remaining} seconds until trigger...")
                    except Exception as e:
                        print(f"   ⚠️ Error parsing time: {e}")
                        
                elif period > 2:
                    print(f"🏀 {away} @ {home} | {away_score}-{home_score} | Q{period}")
                    print(f"   🚨 TRIGGER READY! Past Q2!")
                    triggered_games.append((game_id, home, away, home_score, away_score))
            else:
                print(f"🏀 {away} @ {home} | {away_score}-{home_score} | {status}")
    
    return triggered_games


def run_full_prediction(game_id, home, away, home_score, away_score):
    """Run complete prediction pipeline"""
    
    print(f"\n{'=' * 80}")
    print(f"🤖 RUNNING FULL PREDICTION PIPELINE")
    print(f"{'=' * 80}")
    print(f"Game: {away} @ {home}")
    print(f"Current Score: {away_score}-{home_score}")
    print(f"Game ID: {game_id}\n")
    
    # Step 1: Fetch PBP
    print("STEP 1: Fetching 18-minute play-by-play...")
    pbp = playbyplay.PlayByPlay(game_id=game_id)
    actions = pbp.actions.get_dict()
    print(f"✅ Fetched {len(actions)} actions")
    
    # Step 2: Extract pattern
    print("\nSTEP 2: Extracting 18-minute pattern...")
    pattern = []
    
    for action in actions:
        period = action.get('period', 0)
        clock = action.get('clock', '')
        
        if not action.get('scoreHome') or not action.get('scoreAway'):
            continue
        
        try:
            h = int(action['scoreHome'])
            a = int(action['scoreAway'])
            diff = h - a
            
            # First 18 minutes: All of Q1 + first 6 min of Q2
            if period == 1:
                pattern.append(diff)
            elif period == 2 and 'PT' in clock:
                minutes = int(clock.split('M')[0].replace('PT', ''))
                if minutes >= 6:
                    pattern.append(diff)
        except:
            pass
    
    print(f"✅ Extracted {len(pattern)} data points")
    
    if len(pattern) < 50:
        print(f"⚠️ Not enough data yet (need ~150+ points for 18 minutes)")
        return
    
    # Step 3: Extract ALL 33 features
    print("\nSTEP 3: Extracting ALL 33 features...")
    pattern_array = np.array(pattern, dtype=float)
    features = []
    
    # 1. Pattern Analysis (10)
    mean_diff = np.mean(pattern_array)
    std_diff = np.std(pattern_array)
    trend = np.polyfit(range(len(pattern_array)), pattern_array, 1)[0]
    volatility = np.var(pattern_array)
    velocity = pattern_array[-1] - pattern_array[-10] if len(pattern_array) > 10 else 0
    acceleration = velocity - (pattern_array[-10] - pattern_array[-20]) if len(pattern_array) > 20 else 0
    recent_momentum = np.mean(pattern_array[-10:]) if len(pattern_array) > 10 else mean_diff
    
    # Count lead changes
    lead_changes = 0
    for i in range(1, len(pattern_array)):
        if (pattern_array[i] > 0) != (pattern_array[i-1] > 0):
            lead_changes += 1
    
    max_swing = np.max(np.abs(np.diff(pattern_array))) if len(pattern_array) > 1 else 0
    comeback_potential = 1.0 / (1.0 + abs(mean_diff)) if mean_diff != 0 else 0.5
    
    print(f"  1️⃣ Pattern Analysis (10):")
    print(f"     mean_diff={mean_diff:.2f}, std_diff={std_diff:.2f}, trend={trend:.4f}")
    
    # 2. Spectral Features (6)
    fft_result = np.fft.fft(pattern_array)
    power_spectrum = np.abs(fft_result) ** 2
    
    n = len(power_spectrum)
    low_idx = int(n * 0.33)
    mid_idx = int(n * 0.67)
    
    spectral_energy = np.sum(power_spectrum)
    spectral_entropy = -np.sum((power_spectrum / spectral_energy) * np.log(power_spectrum / spectral_energy + 1e-10))
    low_freq_power = np.sum(power_spectrum[:low_idx])
    mid_freq_power = np.sum(power_spectrum[low_idx:mid_idx])
    high_freq_power = np.sum(power_spectrum[mid_idx:])
    dominant_freq = np.argmax(power_spectrum)
    
    print(f"  2️⃣ Spectral Features (6):")
    print(f"     low_freq={low_freq_power:.0f}, mid_freq={mid_freq_power:.0f}, high_freq={high_freq_power:.0f}")
    
    # 3. Autocorrelation (3)
    autocorr_lag1 = np.corrcoef(pattern_array[:-1], pattern_array[1:])[0,1] if len(pattern_array) > 1 else 0
    autocorr_lag3 = np.corrcoef(pattern_array[:-3], pattern_array[3:])[0,1] if len(pattern_array) > 3 else 0
    autocorr_lag5 = np.corrcoef(pattern_array[:-5], pattern_array[5:])[0,1] if len(pattern_array) > 5 else 0
    
    print(f"  3️⃣ Autocorrelation (3):")
    print(f"     lag1={autocorr_lag1:.3f}, lag3={autocorr_lag3:.3f}, lag5={autocorr_lag5:.3f}")
    
    # 4. Advanced Stats Proxies (8)
    pace_proxy = len(pattern) / 18.0
    efg_proxy = 0.5  # Would calculate from scoring events
    ts_proxy = 0.55
    netrtg_proxy = mean_diff / 18.0
    usg_proxy = 0.65
    pm_proxy = pattern_array[-1]
    pie_proxy = 0.5
    four_factors = 0.5
    
    print(f"  4️⃣ Advanced Stats (8):")
    print(f"     pace={pace_proxy:.2f}, netrtg={netrtg_proxy:.3f}")
    
    # 5. Team Form (6) - Using placeholders (would fetch from API)
    team_diff_lag1 = 5.0
    team_mean_lag1 = 110.0
    team_diff_rolling3 = 3.0
    team_volatility = 8.0
    team_form_10 = 2.0
    team_consistency = 0.7
    
    print(f"  5️⃣ Team Form (6): Using placeholders (would fetch from nba_api)")
    
    # Combine all 33 features
    all_features = [
        # Pattern Analysis (10)
        mean_diff, std_diff, trend, volatility, velocity, 
        acceleration, recent_momentum, lead_changes, max_swing, comeback_potential,
        # Spectral (6)
        spectral_energy, spectral_entropy, low_freq_power, mid_freq_power, high_freq_power, dominant_freq,
        # Autocorrelation (3)
        autocorr_lag1, autocorr_lag3, autocorr_lag5,
        # Advanced Stats (8)
        pace_proxy, efg_proxy, ts_proxy, netrtg_proxy, usg_proxy, pm_proxy, pie_proxy, four_factors,
        # Team Form (6)
        team_diff_lag1, team_mean_lag1, team_diff_rolling3, team_volatility, team_form_10, team_consistency
    ]
    
    print(f"\n✅ EXTRACTED ALL 33 FEATURES!")
    print(f"   Feature vector shape: ({len(all_features)},)")
    
    # Step 4: Mock ML Prediction (you'd load your actual model here)
    print(f"\nSTEP 4: ML Model Prediction...")
    print(f"  (Would run: model.predict(features))")
    
    # Mock prediction based on current trend
    predicted_margin = mean_diff + (trend * 30)  # Extrapolate trend
    confidence_width = std_diff * 2  # Use volatility for CI
    
    print(f"\n{'=' * 80}")
    print(f"📊 PREDICTION RESULT")
    print(f"{'=' * 80}")
    print(f"Game: {away} @ {home}")
    print(f"Current: {away_score}-{home_score} (Diff: {home_score - away_score:+d})")
    print(f"\n🤖 ML Prediction:")
    print(f"   Final Margin: {home} {predicted_margin:+.1f}")
    print(f"   Confidence Interval: [{predicted_margin - confidence_width:.1f}, {predicted_margin + confidence_width:.1f}]")
    print(f"\n📈 Pattern Insights:")
    print(f"   Trend: {trend:.4f} (positive = home pulling away)")
    print(f"   Volatility: {volatility:.2f} (higher = more chaotic)")
    print(f"   Low Freq Power: {low_freq_power:.0f} (steady momentum)")
    print(f"   Lead Changes: {lead_changes}")
    print(f"{'=' * 80}")


# Main monitoring loop
if __name__ == "__main__":
    print("🚀 NBA Q2 6:00 TRIGGER MONITOR")
    print("Checking for games every 30 seconds...")
    print("Press Ctrl+C to stop\n")
    
    try:
        while True:
            triggered = check_games()
            
            if triggered:
                print(f"\n🎯 Found {len(triggered)} game(s) ready for prediction!")
                
                # Run prediction on first triggered game
                game_id, home, away, home_score, away_score = triggered[0]
                run_full_prediction(game_id, home, away, home_score, away_score)
                
                print(f"\n✅ Prediction complete! Continuing to monitor...")
            
            print(f"\n⏳ Waiting 30 seconds before next check...")
            time.sleep(30)
            
    except KeyboardInterrupt:
        print(f"\n\n🛑 Monitoring stopped")

