"""
BET PORTFOLIO MANAGER

Purpose: Track all bets manually entered by user, store for analysis
Author: Ontologic XYZ
Date: October 20, 2025

This stores:
- Bets placed
- Outcomes
- P&L tracking
- Performance analytics
- Portfolio statistics
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np


class BetPortfolioManager:
    """
    Manage bet portfolio and performance tracking
    """
    
    def __init__(self, db_path: str = "data/bet_portfolio.db"):
        """
        Initialize portfolio manager
        
        Args:
            db_path: Path to SQLite database
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Create database tables if they don't exist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Bets table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS bets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                game_id TEXT,
                matchup TEXT NOT NULL,
                bet_type TEXT NOT NULL,
                bet_line TEXT NOT NULL,
                stake REAL NOT NULL,
                odds REAL DEFAULT -110,
                prediction REAL,
                market_spread REAL,
                edge REAL,
                p_win REAL,
                book TEXT,
                status TEXT DEFAULT 'PENDING',
                result TEXT,
                actual_score TEXT,
                profit REAL,
                settled_at TEXT,
                notes TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Performance summary table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_summary (
                date TEXT PRIMARY KEY,
                bets_placed INTEGER DEFAULT 0,
                wins INTEGER DEFAULT 0,
                losses INTEGER DEFAULT 0,
                pushes INTEGER DEFAULT 0,
                total_staked REAL DEFAULT 0,
                total_profit REAL DEFAULT 0,
                win_rate REAL DEFAULT 0,
                roi REAL DEFAULT 0,
                bankroll REAL DEFAULT 10000
            )
        ''')
        
        conn.commit()
        conn.close()
        
        print(f"✅ Database initialized: {self.db_path}")
    
    def add_bet(
        self,
        matchup: str,
        bet_type: str,
        bet_line: str,
        stake: float,
        odds: float = -110,
        prediction: Optional[float] = None,
        market_spread: Optional[float] = None,
        edge: Optional[float] = None,
        p_win: Optional[float] = None,
        book: str = "BetOnline",
        game_id: Optional[str] = None,
        notes: Optional[str] = None
    ) -> int:
        """
        Add a bet to the portfolio
        
        Args:
            matchup: e.g., "BOS @ LAL"
            bet_type: "SPREAD", "TOTAL", "MONEYLINE"
            bet_line: e.g., "LAL -3.5"
            stake: Bet amount
            odds: Betting odds (default -110)
            prediction: Our ML prediction
            market_spread: Market spread
            edge: Our calculated edge
            p_win: Win probability
            book: Sportsbook name
            game_id: NBA game ID
            notes: Any notes
            
        Returns:
            Bet ID
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO bets (
                timestamp, game_id, matchup, bet_type, bet_line,
                stake, odds, prediction, market_spread, edge,
                p_win, book, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            game_id,
            matchup,
            bet_type,
            bet_line,
            stake,
            odds,
            prediction,
            market_spread,
            edge,
            p_win,
            book,
            notes
        ))
        
        bet_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        print(f"✅ Bet #{bet_id} logged: {bet_line} for ${stake}")
        
        return bet_id
    
    def settle_bet(
        self,
        bet_id: int,
        result: str,
        actual_score: Optional[str] = None,
        profit: Optional[float] = None
    ):
        """
        Settle a bet with result
        
        Args:
            bet_id: Bet ID to settle
            result: "WIN", "LOSS", or "PUSH"
            actual_score: Actual game score
            profit: Actual profit/loss
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # If profit not provided, calculate from stake and odds
        if profit is None:
            cursor.execute('SELECT stake, odds FROM bets WHERE id = ?', (bet_id,))
            row = cursor.fetchone()
            if row:
                stake, odds = row
                if result == "WIN":
                    profit = stake * (100 / abs(odds)) if odds < 0 else stake * (odds / 100)
                elif result == "LOSS":
                    profit = -stake
                else:  # PUSH
                    profit = 0
        
        cursor.execute('''
            UPDATE bets
            SET status = 'SETTLED',
                result = ?,
                actual_score = ?,
                profit = ?,
                settled_at = ?
            WHERE id = ?
        ''', (result, actual_score, profit, datetime.now().isoformat(), bet_id))
        
        conn.commit()
        conn.close()
        
        print(f"✅ Bet #{bet_id} settled: {result} ({profit:+.2f})")
    
    def get_all_bets(self, status: Optional[str] = None) -> List[Dict]:
        """
        Get all bets
        
        Args:
            status: Filter by status ("PENDING", "SETTLED", or None for all)
            
        Returns:
            List of bet dicts
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        if status:
            cursor.execute('SELECT * FROM bets WHERE status = ? ORDER BY timestamp DESC', (status,))
        else:
            cursor.execute('SELECT * FROM bets ORDER BY timestamp DESC')
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    
    def get_performance_summary(self) -> Dict:
        """
        Get overall performance summary
        
        Returns:
            Performance metrics
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get settled bets
        cursor.execute('''
            SELECT COUNT(*) as total,
                   SUM(CASE WHEN result = 'WIN' THEN 1 ELSE 0 END) as wins,
                   SUM(CASE WHEN result = 'LOSS' THEN 1 ELSE 0 END) as losses,
                   SUM(CASE WHEN result = 'PUSH' THEN 1 ELSE 0 END) as pushes,
                   SUM(stake) as total_staked,
                   SUM(profit) as total_profit
            FROM bets
            WHERE status = 'SETTLED'
        ''')
        
        row = cursor.fetchone()
        conn.close()
        
        if not row or row[0] == 0:
            return {
                'total_bets': 0,
                'wins': 0,
                'losses': 0,
                'pushes': 0,
                'win_rate': 0,
                'total_staked': 0,
                'total_profit': 0,
                'roi': 0,
                'avg_stake': 0,
                'avg_profit_per_bet': 0
            }
        
        total, wins, losses, pushes, staked, profit = row
        
        win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0
        roi = profit / staked if staked > 0 else 0
        
        return {
            'total_bets': total,
            'wins': wins,
            'losses': losses,
            'pushes': pushes,
            'win_rate': win_rate,
            'total_staked': staked,
            'total_profit': profit,
            'roi': roi,
            'avg_stake': staked / total,
            'avg_profit_per_bet': profit / total
        }
    
    def get_pending_bets(self) -> List[Dict]:
        """Get all pending (unsettled) bets"""
        return self.get_all_bets(status='PENDING')
    
    def export_to_csv(self, filepath: str = "bet_portfolio.csv"):
        """Export all bets to CSV"""
        bets = self.get_all_bets()
        df = pd.DataFrame(bets)
        df.to_csv(filepath, index=False)
        print(f"✅ Exported {len(bets)} bets to {filepath}")
    
    def print_summary(self):
        """Print formatted performance summary"""
        summary = self.get_performance_summary()
        
        print("\n" + "="*80)
        print("📊 BET PORTFOLIO SUMMARY")
        print("="*80 + "\n")
        
        print(f"Total Bets: {summary['total_bets']}")
        print(f"Wins: {summary['wins']}")
        print(f"Losses: {summary['losses']}")
        print(f"Pushes: {summary['pushes']}")
        print(f"Win Rate: {summary['win_rate']:.1%}")
        print()
        
        print(f"Total Staked: ${summary['total_staked']:,.2f}")
        print(f"Total Profit: ${summary['total_profit']:+,.2f}")
        print(f"ROI: {summary['roi']:.1%}")
        print()
        
        print(f"Avg Stake: ${summary['avg_stake']:,.2f}")
        print(f"Avg Profit/Bet: ${summary['avg_profit_per_bet']:+,.2f}")
        
        print("\n" + "="*80)


def example_usage():
    """Example: Track bets"""
    print("\n" + "="*80)
    print("🔥 BET PORTFOLIO MANAGER - EXAMPLE")
    print("="*80 + "\n")
    
    # Initialize
    portfolio = BetPortfolioManager()
    
    # Add some bets
    print("📝 Adding bets...\n")
    
    bet1 = portfolio.add_bet(
        matchup="BOS @ LAL",
        bet_type="SPREAD",
        bet_line="LAL -3.5",
        stake=430,
        prediction=+2.5,
        market_spread=-3.5,
        edge=6.0,
        p_win=0.703,
        book="DraftKings"
    )
    
    bet2 = portfolio.add_bet(
        matchup="PHX @ GSW",
        bet_type="SPREAD",
        bet_line="PHX +5.5",
        stake=350,
        prediction=-8.0,
        market_spread=-5.0,
        edge=3.0,
        p_win=0.605,
        book="FanDuel"
    )
    
    # Settle bet 1 (win)
    print("\n💰 Settling bets...\n")
    portfolio.settle_bet(bet1, result="WIN", actual_score="LAL 112, BOS 108 (+4)")
    
    # Settle bet 2 (loss)
    portfolio.settle_bet(bet2, result="LOSS", actual_score="GSW 120, PHX 108 (-12)")
    
    # Show summary
    portfolio.print_summary()
    
    # Show pending bets
    pending = portfolio.get_pending_bets()
    print(f"\n📋 Pending Bets: {len(pending)}")
    
    # Export
    portfolio.export_to_csv()
    
    print("\n" + "="*80)
    print("✅ BET PORTFOLIO MANAGER READY")
    print("="*80)


if __name__ == "__main__":
    example_usage()

