"""
ONTORISK PHASE 1: PROBABILITY CALIBRATION ENGINE

Purpose: Convert MAE-based predictions into calibrated win probabilities
Author: Ontologic XYZ
Date: October 20, 2025

This is the FIRST layer of OntoRisk - takes ML predictions and converts them
to betting probabilities.
"""

import numpy as np
from scipy.stats import norm
from sklearn.isotonic import IsotonicRegression
import pickle
from typing import Dict, Tuple, List
from dataclasses import dataclass


@dataclass
class PredictionProbability:
    """
    Output of probability calibration
    """
    prediction: float  # Our model's prediction (+2.5 = home wins by 2.5)
    spread_line: float  # Market spread (-3.5 = home favored by 3.5)
    p_win: float  # Probability we beat the spread
    p_push: float  # Probability of push (tie)
    confidence_interval: Tuple[float, float]  # 95% CI
    kelly_edge: float  # Edge for Kelly criterion
    bet_side: str  # "OVER" or "UNDER"
    bet_line: str  # "Team A +3.5" (formatted)


class ProbabilityCalibrator:
    """
    Converts MAE-based predictions to calibrated win probabilities
    """
    
    def __init__(self, mae: float = 9.03, method: str = 'gaussian'):
        """
        Args:
            mae: Model's Mean Absolute Error
            method: 'gaussian' or 'isotonic' (isotonic requires training data)
        """
        self.mae = mae
        self.method = method
        self.std_dev = mae * 1.25  # Convert MAE to std dev
        self.isotonic_model = None
        
    def gaussian_probability(
        self, 
        prediction: float, 
        spread_line: float
    ) -> float:
        """
        Calculate P(win) assuming Gaussian error distribution
        
        Args:
            prediction: Our model's prediction (e.g., +2.5)
            spread_line: Market spread (e.g., -3.5)
            
        Returns:
            Probability of beating the spread
            
        Example:
            prediction = +2.5 (we think home wins by 2.5)
            spread_line = -3.5 (market says home wins by 3.5)
            bet_side = OVER (bet home +3.5)
            
            We win if: actual >= -3.5
            Our prediction: +2.5
            Margin: 2.5 - (-3.5) = 6.0 points
            
            Z-score = 6.0 / std_dev
            P(win) = norm.cdf(z_score)
        """
        # Determine bet side
        if prediction > spread_line:
            # We think underdog covers
            bet_side = "OVER"
            # We win if actual >= spread_line
            margin = prediction - spread_line
        else:
            # We think favorite covers
            bet_side = "UNDER"
            # We win if actual <= spread_line
            margin = spread_line - prediction
            
        # Calculate z-score
        z_score = margin / self.std_dev
        
        # Probability of winning
        p_win = norm.cdf(z_score)
        
        return p_win
    
    def train_isotonic_calibration(
        self, 
        predictions: np.ndarray, 
        spreads: np.ndarray,
        actuals: np.ndarray
    ):
        """
        Train isotonic regression for better calibration
        
        This learns the relationship between:
            - Our prediction - spread (edge)
            - Actual win rate
            
        Args:
            predictions: Our model predictions
            spreads: Historical market spreads
            actuals: Actual game results
        """
        # Calculate edges
        edges = np.abs(predictions - spreads)
        
        # Calculate outcomes (did we win the bet?)
        outcomes = []
        for pred, spread, actual in zip(predictions, spreads, actuals):
            if pred > spread:
                # Bet OVER
                win = 1 if actual >= spread else 0
            else:
                # Bet UNDER
                win = 1 if actual <= spread else 0
            outcomes.append(win)
        
        outcomes = np.array(outcomes)
        
        # Train isotonic regression
        self.isotonic_model = IsotonicRegression(out_of_bounds='clip')
        self.isotonic_model.fit(edges, outcomes)
        
        print(f"✅ Isotonic calibration trained on {len(predictions)} games")
        
    def isotonic_probability(
        self, 
        prediction: float, 
        spread_line: float
    ) -> float:
        """
        Use isotonic regression to estimate P(win)
        """
        if self.isotonic_model is None:
            raise ValueError("Must train isotonic calibration first!")
        
        edge = abs(prediction - spread_line)
        p_win = self.isotonic_model.predict([edge])[0]
        
        return p_win
    
    def calculate_probability(
        self,
        prediction: float,
        spread_line: float,
        home_team: str = "Team A",
        away_team: str = "Team B"
    ) -> PredictionProbability:
        """
        Full probability calculation with all outputs
        
        Args:
            prediction: Our prediction (positive = home wins)
            spread_line: Market spread (negative = home favored)
            home_team: Home team name
            away_team: Away team name
            
        Returns:
            PredictionProbability object with all info
        """
        # Calculate P(win)
        if self.method == 'gaussian':
            p_win = self.gaussian_probability(prediction, spread_line)
        elif self.method == 'isotonic' and self.isotonic_model is not None:
            p_win = self.isotonic_probability(prediction, spread_line)
        else:
            # Default to Gaussian
            p_win = self.gaussian_probability(prediction, spread_line)
        
        # Calculate confidence interval (95%)
        ci_lower = prediction - 1.96 * self.std_dev
        ci_upper = prediction + 1.96 * self.std_dev
        
        # Probability of push (within 0.5 points)
        p_push = norm.cdf((spread_line + 0.5 - prediction) / self.std_dev) - \
                 norm.cdf((spread_line - 0.5 - prediction) / self.std_dev)
        
        # Kelly edge
        # edge = (p * odds) - (1 - p)
        # For -110 odds: odds = 1.91
        odds = 1.91
        kelly_edge = (p_win * odds) - (1 - p_win)
        
        # Determine bet side and format
        if prediction > spread_line:
            bet_side = "OVER"
            if spread_line < 0:
                # Underdog
                bet_line = f"{home_team} {spread_line:+.1f}"
            else:
                bet_line = f"{away_team} {-spread_line:+.1f}"
        else:
            bet_side = "UNDER"
            if spread_line < 0:
                # Favorite
                bet_line = f"{away_team} {-spread_line:+.1f}"
            else:
                bet_line = f"{home_team} {spread_line:+.1f}"
        
        return PredictionProbability(
            prediction=prediction,
            spread_line=spread_line,
            p_win=p_win,
            p_push=p_push,
            confidence_interval=(ci_lower, ci_upper),
            kelly_edge=kelly_edge,
            bet_side=bet_side,
            bet_line=bet_line
        )
    
    def batch_calibrate(
        self,
        predictions: List[float],
        spreads: List[float],
        teams: List[Tuple[str, str]] = None
    ) -> List[PredictionProbability]:
        """
        Calibrate multiple predictions at once
        """
        results = []
        
        for i, (pred, spread) in enumerate(zip(predictions, spreads)):
            if teams is not None:
                home, away = teams[i]
            else:
                home, away = "Home", "Away"
            
            prob = self.calculate_probability(pred, spread, home, away)
            results.append(prob)
        
        return results


def example_usage():
    """
    Example of how to use ProbabilityCalibrator
    """
    print("\n" + "="*80)
    print("🎯 ONTORISK PHASE 1: PROBABILITY CALIBRATION")
    print("="*80 + "\n")
    
    # Initialize calibrator
    calibrator = ProbabilityCalibrator(mae=9.03, method='gaussian')
    
    # Example 1: We predict +2.5, market says -3.5
    print("Example 1: LAL @ BOS")
    print("-" * 40)
    prob1 = calibrator.calculate_probability(
        prediction=+2.5,
        spread_line=-3.5,
        home_team="LAL",
        away_team="BOS"
    )
    
    print(f"Our Prediction: LAL wins by {prob1.prediction:+.1f}")
    print(f"Market Spread: LAL {prob1.spread_line:+.1f}")
    print(f"Our Edge: {abs(prob1.prediction - prob1.spread_line):.1f} points")
    print(f"\nRecommended Bet: {prob1.bet_line} ({prob1.bet_side})")
    print(f"P(Win): {prob1.p_win:.1%}")
    print(f"P(Push): {prob1.p_push:.1%}")
    print(f"Confidence Interval: [{prob1.confidence_interval[0]:.1f}, {prob1.confidence_interval[1]:.1f}]")
    print(f"Kelly Edge: {prob1.kelly_edge:.1%}")
    
    # Example 2: We predict -8, market says -5
    print("\n\nExample 2: GSW @ PHX")
    print("-" * 40)
    prob2 = calibrator.calculate_probability(
        prediction=-8.0,
        spread_line=-5.0,
        home_team="GSW",
        away_team="PHX"
    )
    
    print(f"Our Prediction: GSW wins by {abs(prob2.prediction):.1f}")
    print(f"Market Spread: GSW {prob2.spread_line:+.1f}")
    print(f"Our Edge: {abs(prob2.prediction - prob2.spread_line):.1f} points")
    print(f"\nRecommended Bet: {prob2.bet_line} ({prob2.bet_side})")
    print(f"P(Win): {prob2.p_win:.1%}")
    print(f"P(Push): {prob2.p_push:.1%}")
    print(f"Kelly Edge: {prob2.kelly_edge:.1%}")
    
    # Example 3: Batch processing
    print("\n\n" + "="*80)
    print("📊 BATCH CALIBRATION (5 games)")
    print("="*80 + "\n")
    
    predictions = [+2.5, -8.0, +0.5, -12.0, +6.0]
    spreads = [-3.5, -5.0, -2.0, -8.0, +3.0]
    teams = [
        ("LAL", "BOS"),
        ("GSW", "PHX"),
        ("MIA", "DEN"),
        ("MIL", "CLE"),
        ("OKC", "DAL")
    ]
    
    results = calibrator.batch_calibrate(predictions, spreads, teams)
    
    print(f"{'Game':<15} {'Edge':>6} {'P(Win)':>8} {'Bet Side':<12} {'Kelly Edge':>12}")
    print("-" * 65)
    
    for (home, away), result in zip(teams, results):
        edge = abs(result.prediction - result.spread_line)
        game = f"{away}@{home}"
        print(f"{game:<15} {edge:>6.1f} {result.p_win:>8.1%} {result.bet_line:<12} {result.kelly_edge:>11.1%}")
    
    # Filter for high-confidence bets
    print("\n\n" + "="*80)
    print("✅ HIGH-CONFIDENCE BETS (P(Win) ≥ 58%)")
    print("="*80 + "\n")
    
    high_conf = [r for r in results if r.p_win >= 0.58]
    
    if high_conf:
        for result in high_conf:
            print(f"🎯 {result.bet_line}")
            print(f"   P(Win): {result.p_win:.1%}")
            print(f"   Kelly Edge: {result.kelly_edge:.1%}")
            print()
    else:
        print("No high-confidence bets found.")
    
    print("="*80)
    print("✅ PROBABILITY CALIBRATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    example_usage()

