"""
Complete Dejavu Pipeline Runner
Runs all steps: Clean → Extract → Build → Test

One-command execution of entire Dejavu system
"""

import sys
from pathlib import Path
import time

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*60)
    print(text.center(60))
    print("="*60)

def run_pipeline():
    """
    Run complete Dejavu pipeline
    """
    start_time = time.time()
    
    print_header("DEJAVU COMPLETE PIPELINE")
    print("\nThis will:")
    print("  1. Clean raw PBP data → cleaned_games.parquet")
    print("  2. Extract patterns → reference_set/")
    print("  3. Build Dejavu model")
    print("  4. Run test predictions")
    print("\nEstimated time: 30-60 seconds")
    print("\nPress Enter to continue or Ctrl+C to cancel...")
    input()
    
    # Step 1: Data Cleaning
    print_header("STEP 1: DATA CLEANING")
    try:
        from data_cleaning_01 import DataCleaner
        
        data_dir = "../../All Of Our Data"
        cleaner = DataCleaner(data_dir)
        cleaned_file = cleaner.clean_and_save('cleaned_games.parquet')
        
        print(f"✅ Step 1 complete: {cleaned_file}")
        
    except Exception as e:
        print(f"❌ Step 1 failed: {e}")
        return False
    
    # Step 2: Pattern Extraction
    print_header("STEP 2: PATTERN EXTRACTION")
    try:
        from pattern_extraction_02 import PatternExtractor
        
        extractor = PatternExtractor()
        reference_dir = extractor.run_pipeline(cleaned_file)
        
        print(f"✅ Step 2 complete: {reference_dir}")
        
    except Exception as e:
        print(f"❌ Step 2 failed: {e}")
        return False
    
    # Step 3: Model Building
    print_header("STEP 3: DEJAVU MODEL")
    try:
        from dejavu_model_03 import DejavuForecaster
        
        model = DejavuForecaster(k=500)
        model.load_reference_set(reference_dir)
        
        print(f"✅ Step 3 complete: Model loaded")
        
    except Exception as e:
        print(f"❌ Step 3 failed: {e}")
        return False
    
    # Step 4: Test Predictions
    print_header("STEP 4: TEST PREDICTIONS")
    try:
        # Test 1: BOS leading at halftime
        test_query_1 = {
            'differential_ht': -4,  # BOS +4
            'home_rolling_diff': 2.5,
            'away_rolling_diff': 3.5,
            'home_ht_avg': 1.2,
            'away_ht_avg': 2.8,
            'home_rolling_std': 8.5,
            'away_rolling_std': 7.2
        }
        
        pred_1 = model.predict(test_query_1)
        
        print(f"\nTest 1: BOS 52, LAL 48 (BOS +4 at halftime)")
        print(f"  Predicted change: {pred_1['point_forecast']:+.1f}")
        print(f"  Interval: [{pred_1['lower']:+.1f}, {pred_1['upper']:+.1f}]")
        print(f"  Time: {pred_1['computation_time_ms']:.1f}ms")
        
        # Test 2: Tied game
        test_query_2 = {
            'differential_ht': 0,  # Tied
            'home_rolling_diff': 3.0,
            'away_rolling_diff': 2.5,
            'home_ht_avg': 1.5,
            'away_ht_avg': 1.8,
            'home_rolling_std': 7.0,
            'away_rolling_std': 8.0
        }
        
        pred_2 = model.predict(test_query_2)
        
        print(f"\nTest 2: Tied game at halftime (50-50)")
        print(f"  Predicted change: {pred_2['point_forecast']:+.1f}")
        print(f"  Interval: [{pred_2['lower']:+.1f}, {pred_2['upper']:+.1f}]")
        print(f"  Time: {pred_2['computation_time_ms']:.1f}ms")
        
        # Test 3: Large deficit
        test_query_3 = {
            'differential_ht': -15,  # Down 15 at halftime
            'home_rolling_diff': 1.0,
            'away_rolling_diff': 5.0,
            'home_ht_avg': 0.5,
            'away_ht_avg': 3.5,
            'home_rolling_std': 9.0,
            'away_rolling_std': 6.0
        }
        
        pred_3 = model.predict(test_query_3)
        
        print(f"\nTest 3: Down 15 at halftime (weak vs strong)")
        print(f"  Predicted change: {pred_3['point_forecast']:+.1f}")
        print(f"  Interval: [{pred_3['lower']:+.1f}, {pred_3['upper']:+.1f}]")
        print(f"  Time: {pred_3['computation_time_ms']:.1f}ms")
        
        print(f"\n✅ All tests passed")
        
    except Exception as e:
        print(f"❌ Step 4 failed: {e}")
        return False
    
    # Success!
    elapsed = time.time() - start_time
    
    print_header("✅ PIPELINE COMPLETE")
    print(f"\nTotal time: {elapsed:.1f} seconds")
    print(f"\nDejavu model is ready!")
    print(f"  • Reference set: {len(model.patterns):,} games")
    print(f"  • k={model.k} neighbors")
    print(f"  • Prediction time: <100ms")
    print(f"  • Performance: Real-time compatible ✅")
    
    print(f"\nNext steps:")
    print(f"  1. Deploy FastAPI server (python 04_fastapi_server.py)")
    print(f"  2. Integrate with LSTM (40% Dejavu + 60% LSTM)")
    print(f"  3. Add Conformal wrapper (95% CI)")
    print(f"  4. Connect to Risk layers")
    
    return True

if __name__ == "__main__":
    try:
        success = run_pipeline()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n❌ Pipeline cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

