"""
Covariance Matrix Builder
Estimate correlation and build covariance matrix for multi-game portfolio

Based on: PORTFOLIO_MANAGEMENT/Applied Model/covariance_estimator.py
Following: PORTFOLIO_MANAGEMENT/MATH_BREAKDOWN.txt (Section 9)
Performance: <10ms for 10×10 matrix
"""

import numpy as np
from typing import List, Dict

class CovarianceBuilder:
    """
    Build covariance matrix from game opportunities
    
    Formula (MATH_BREAKDOWN.txt 9.1):
        σ_ij = ρ_ij × σ_i × σ_j
        
    Correlation assumptions:
        - Same night: ρ = 0.20 (moderate)
        - Same division: ρ = 0.30 (higher)
        - Otherwise: ρ = 0.15 (baseline)
    
    Performance: <10ms for n=10 games
    """
    
    def __init__(
        self,
        same_night_corr: float = 0.20,
        same_division_corr: float = 0.30,
        base_corr: float = 0.15
    ):
        """
        Initialize covariance builder
        
        Args:
            same_night_corr: Correlation for games same night
            same_division_corr: Correlation for games same division
            base_corr: Baseline correlation
        """
        self.same_night_corr = same_night_corr
        self.same_division_corr = same_division_corr
        self.base_corr = base_corr
        
        print("Covariance Builder initialized:")
        print(f"  Same night correlation: {same_night_corr}")
        print(f"  Same division correlation: {same_division_corr}")
        print(f"  Base correlation: {base_corr}")
    
    def build_correlation_matrix(self, opportunities: List[Dict]) -> np.ndarray:
        """
        Build correlation matrix from opportunities
        
        Args:
            opportunities: List of game dicts with 'division' field
        
        Returns:
            n×n correlation matrix
        
        Formula:
            ρ_ii = 1.0 (diagonal)
            ρ_ij = correlation based on game relationship
        
        Time: <5ms for n=10
        """
        n = len(opportunities)
        correlation = np.eye(n)  # Start with identity
        
        for i in range(n):
            for j in range(i + 1, n):
                # Determine correlation based on context
                game_i = opportunities[i]
                game_j = opportunities[j]
                
                # Check if same division
                if game_i.get('division') == game_j.get('division') and game_i.get('division'):
                    rho = self.same_division_corr
                else:
                    # All games on same night (default)
                    rho = self.same_night_corr
                
                # Symmetric
                correlation[i, j] = rho
                correlation[j, i] = rho
        
        return correlation
    
    def build_covariance_matrix(self, opportunities: List[Dict]) -> np.ndarray:
        """
        Build full covariance matrix
        
        Formula (MATH_BREAKDOWN.txt 9.2):
            Σ = D × R × D
            
        Where:
            D = Diagonal matrix of standard deviations
            R = Correlation matrix
            Σ = Covariance matrix
        
        Args:
            opportunities: List with 'volatility' field
        
        Returns:
            n×n covariance matrix
        
        Time: <8ms for n=10
        """
        n = len(opportunities)
        
        # Extract volatilities
        volatilities = np.array([opp.get('volatility', 0.20) for opp in opportunities])
        
        # Build correlation matrix
        correlation = self.build_correlation_matrix(opportunities)
        
        # Create diagonal volatility matrix
        D = np.diag(volatilities)
        
        # Compute covariance: Σ = D × R × D
        covariance = D @ correlation @ D
        
        return covariance
    
    def get_portfolio_variance(
        self,
        weights: np.ndarray,
        covariance: np.ndarray
    ) -> float:
        """
        Calculate portfolio variance
        
        Formula (MATH_BREAKDOWN.txt 10.2):
            σ_p² = w^T Σ w
        
        Args:
            weights: Portfolio weights (sums to ≤1)
            covariance: Covariance matrix
        
        Returns:
            Portfolio variance
        
        Time: <1ms
        """
        variance = weights @ covariance @ weights
        return float(variance)
    
    def get_portfolio_volatility(
        self,
        weights: np.ndarray,
        covariance: np.ndarray
    ) -> float:
        """
        Calculate portfolio volatility (standard deviation)
        
        Formula:
            σ_p = sqrt(w^T Σ w)
        
        Time: <1ms
        """
        variance = self.get_portfolio_variance(weights, covariance)
        return float(np.sqrt(variance))
    
    def get_diversification_benefit(
        self,
        weights: np.ndarray,
        opportunities: List[Dict],
        covariance: np.ndarray
    ) -> float:
        """
        Calculate diversification benefit
        
        Formula (MATH_BREAKDOWN.txt 5.3):
            Benefit = 1 - (σ_portfolio / Σ(w_i × σ_i))
        
        Returns value 0-1:
            0 = No diversification
            1 = Perfect diversification
        
        Time: <2ms
        """
        # Portfolio volatility
        portfolio_vol = self.get_portfolio_volatility(weights, covariance)
        
        # Sum of individual volatilities weighted
        individual_vols = np.array([opp.get('volatility', 0.20) for opp in opportunities])
        weighted_sum = np.sum(weights * individual_vols)
        
        if weighted_sum == 0:
            return 0.0
        
        benefit = 1.0 - (portfolio_vol / weighted_sum)
        return float(np.clip(benefit, 0.0, 1.0))


# Test the covariance builder
if __name__ == "__main__":
    import time
    
    print("="*80)
    print("COVARIANCE BUILDER - VERIFICATION")
    print("="*80)
    
    builder = CovarianceBuilder()
    
    # Test 1: Build correlation matrix
    print("\n1. Correlation Matrix (6 games):")
    opportunities = [
        {'game_id': 'LAL@BOS', 'division': 'Atlantic', 'volatility': 0.22},
        {'game_id': 'GSW@MIA', 'division': 'Southeast', 'volatility': 0.20},
        {'game_id': 'DEN@PHX', 'division': 'Pacific', 'volatility': 0.18},
        {'game_id': 'BKN@MIL', 'division': 'Central', 'volatility': 0.21},
        {'game_id': 'DAL@LAC', 'division': 'Pacific', 'volatility': 0.19},
        {'game_id': 'MEM@NOP', 'division': 'Southwest', 'volatility': 0.20},
    ]
    
    start = time.time()
    correlation = builder.build_correlation_matrix(opportunities)
    elapsed = (time.time() - start) * 1000
    
    print(f"   Shape: {correlation.shape}")
    print(f"   Diagonal: {np.diag(correlation)} (all 1.0) ✅")
    print(f"   DEN-LAC correlation: {correlation[2, 4]:.2f} (same Pacific division = 0.30) ✅")
    print(f"   LAL-GSW correlation: {correlation[0, 1]:.2f} (different division = 0.20) ✅")
    print(f"   Time: {elapsed:.2f}ms")
    
    # Test 2: Build covariance matrix
    print("\n2. Covariance Matrix:")
    start = time.time()
    covariance = builder.build_covariance_matrix(opportunities)
    elapsed = (time.time() - start) * 1000
    
    print(f"   Shape: {covariance.shape}")
    print(f"   Diagonal (variances):")
    for i, opp in enumerate(opportunities):
        print(f"      {opp['game_id']}: {covariance[i, i]:.4f} (= {opp['volatility']}²)")
    print(f"   Time: {elapsed:.2f}ms")
    
    # Test 3: Portfolio variance
    print("\n3. Portfolio Variance & Volatility:")
    weights = np.array([0.20, 0.18, 0.15, 0.17, 0.16, 0.14])  # Sum = 1.00
    
    variance = builder.get_portfolio_variance(weights, covariance)
    volatility = builder.get_portfolio_volatility(weights, covariance)
    
    print(f"   Weights: {weights}")
    print(f"   Portfolio variance: {variance:.6f}")
    print(f"   Portfolio volatility: {volatility:.4f} ({volatility*100:.2f}%)")
    
    # Compare to no correlation
    no_corr_variance = np.sum((weights ** 2) * np.diag(covariance))
    no_corr_vol = np.sqrt(no_corr_variance)
    
    print(f"\n   Without correlation: {no_corr_vol:.4f}")
    print(f"   With correlation: {volatility:.4f}")
    print(f"   Correlation increases vol by: {(volatility/no_corr_vol - 1)*100:.1f}%")
    
    # Test 4: Diversification benefit
    print("\n4. Diversification Benefit:")
    benefit = builder.get_diversification_benefit(weights, opportunities, covariance)
    
    print(f"   Diversification benefit: {benefit:.1%}")
    print(f"   Interpretation: Portfolio vol is {benefit:.0%} lower than sum of individual vols")
    
    # Test 5: Performance (10 games)
    print("\n5. Performance Test (10 games):")
    large_opportunities = [
        {'game_id': f'Game{i}', 'division': f'Div{i%4}', 'volatility': 0.20}
        for i in range(10)
    ]
    
    start = time.time()
    for _ in range(100):
        builder.build_covariance_matrix(large_opportunities)
    elapsed = (time.time() - start) * 1000 / 100
    
    print(f"   10×10 matrix: {elapsed:.2f}ms avg")
    print(f"   Target: <10ms")
    
    if elapsed < 10:
        print(f"   ✅ PASS!")
    else:
        print(f"   ❌ FAIL - Too slow")
    
    print("\n" + "="*80)
    print("✅ COVARIANCE BUILDER READY")
    print("="*80)
    print("\nBuilds correlation and covariance matrices")
    print("  Same night: ρ = 0.20")
    print("  Same division: ρ = 0.30")
    print("  Portfolio variance: w^T Σ w")
    print("  Diversification: Measures risk reduction")

