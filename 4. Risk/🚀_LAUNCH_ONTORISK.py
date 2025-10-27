"""
🚀 LAUNCH ONTORISK - ONE-CLICK LAUNCHER

Purpose: Launch complete OntoRisk system
Author: Ontologic XYZ
Date: October 20, 2025

This script:
1. Runs backtest on historical data
2. Generates performance report
3. Optionally launches API server

Usage:
    python 🚀_LAUNCH_ONTORISK.py
"""

import sys
import argparse
from pathlib import Path

# Import OntoRisk components
from ontorisk_complete_system import OntoRiskCompleteSystem


def run_backtest_only():
    """
    Run backtest and generate report
    """
    print("\n" + "="*80)
    print("🔥 ONTORISK - BACKTEST MODE")
    print("="*80 + "\n")
    
    # Initialize system
    system = OntoRiskCompleteSystem(
        model_path="../Action/HYBRID_ULTIMATE_V2_CLEAN.pkl",
        data_path="../Action/ULTRA_ENHANCED_PATTERNS_V3_CHRONOLOGICAL.pkl",
        mae=9.029,
        starting_bankroll=10000,
        kelly_fraction=0.25,
        min_edge=5.0,
        min_p_win=0.55
    )
    
    # Run backtest
    print("\n⏳ Running backtest (this may take a minute)...\n")
    results = system.run_backtest(use_synthetic_spreads=True)
    
    if results is None:
        print("\n❌ Backtest failed")
        return False
    
    # Save results
    system.save_backtest_results(results, "backtest_results.json")
    
    # Print summary
    print("\n" + "="*80)
    print("📊 BACKTEST COMPLETE")
    print("="*80)
    print(f"\n✅ Results saved to backtest_results.json")
    print(f"\n💰 Key Metrics:")
    print(f"   Win Rate: {results.win_rate:.1%}")
    print(f"   ROI: {results.roi:.1%}")
    print(f"   Profit: ${results.total_profit:+,.0f}")
    print(f"   Sharpe: {results.sharpe_ratio:.2f}")
    print("\n" + "="*80)
    
    return True


def launch_api():
    """
    Launch API server
    """
    print("\n" + "="*80)
    print("🔥 ONTORISK - API MODE")
    print("="*80 + "\n")
    
    try:
        from ontorisk_api import start_api
        start_api(host="0.0.0.0", port=8000)
    except ImportError:
        print("❌ FastAPI not installed")
        print("   Install with: pip install fastapi uvicorn")
        return False
    except Exception as e:
        print(f"❌ Failed to start API: {e}")
        return False


def show_menu():
    """
    Interactive menu
    """
    print("\n" + "="*80)
    print("🔥 ONTORISK LAUNCHER")
    print("="*80)
    print("\nWhat would you like to do?\n")
    print("1. Run Backtest (test on historical data)")
    print("2. Launch API Server (for live predictions)")
    print("3. Both (backtest first, then API)")
    print("4. Exit")
    print("\n" + "="*80)
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == "1":
        run_backtest_only()
    elif choice == "2":
        launch_api()
    elif choice == "3":
        if run_backtest_only():
            input("\n✅ Backtest complete. Press Enter to launch API...")
            launch_api()
    elif choice == "4":
        print("\n👋 Goodbye!")
        sys.exit(0)
    else:
        print("\n❌ Invalid choice")
        show_menu()


def main():
    """
    Main launcher
    """
    parser = argparse.ArgumentParser(description="OntoRisk Launcher")
    parser.add_argument(
        "--mode",
        choices=["backtest", "api", "both"],
        default=None,
        help="Launch mode (if not specified, shows menu)"
    )
    
    args = parser.parse_args()
    
    if args.mode == "backtest":
        run_backtest_only()
    elif args.mode == "api":
        launch_api()
    elif args.mode == "both":
        if run_backtest_only():
            input("\n✅ Backtest complete. Press Enter to launch API...")
            launch_api()
    else:
        show_menu()


if __name__ == "__main__":
    main()

