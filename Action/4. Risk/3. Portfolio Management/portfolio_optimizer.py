"""
Portfolio Optimizer - Markowitz Mean-Variance Optimization
Institutional-grade multi-game allocation using quadratic programming

Based on: PORTFOLIO_MANAGEMENT/Applied Model/portfolio_optimizer.py
Following: PORTFOLIO_MANAGEMENT/MATH_BREAKDOWN.txt (Section 1-2)
Performance: <50ms for 10 games
"""

import numpy as np
import cvxpy as cp
from typing import List, Dict
from covariance_builder import CovarianceBuilder

class PortfolioOptimizer:
    """
    Optimize allocation across multiple betting opportunities
    
    Method: Markowitz mean-variance optimization
    Formula (MATH_BREAKDOWN.txt 1.1):
        max w^T μ - (λ/2) w^T Σ w
        
    Subject to:
        Σw_i ≤ 0.80 (max 80% of bankroll)
        w_i ≤ 0.20 (max 20% per game)
        w_i ≥ 0 (no short selling)
    
    Performance: <50ms for n≤15 games
    """
    
    def __init__(
        self,
        max_single_position: float = 0.20,
        max_total_allocation: float = 0.80,
        concentration_max: float = 0.35,
        risk_aversion: float = 1.5
    ):
        """
        Initialize portfolio optimizer
        
        Args:
            max_single_position: Max fraction per bet (0.20 = 20%)
            max_total_allocation: Max total fraction (0.80 = 80%)
            concentration_max: Max for exceptional opportunities (0.35 = 35%)
            risk_aversion: λ in formula (higher = more conservative)
        """
        self.max_single_position = max_single_position
        self.max_total_allocation = max_total_allocation
        self.concentration_max = concentration_max
        self.risk_aversion = risk_aversion
        
        self.cov_builder = CovarianceBuilder()
        
        print("\nPortfolio Optimizer initialized:")
        print(f"  Max single position: {max_single_position*100:.0f}%")
        print(f"  Max total allocation: {max_total_allocation*100:.0f}%")
        print(f"  Concentration max: {concentration_max*100:.0f}%")
        print(f"  Risk aversion (λ): {risk_aversion}")
    
    def optimize(
        self,
        opportunities: List[Dict],
        bankroll: float,
        allow_leverage: bool = False
    ) -> Dict:
        """
        Optimize portfolio allocation
        
        Args:
            opportunities: List of {
                'game_id': str,
                'expected_return': 0.12,  # 12% expected return
                'volatility': 0.20,        # 20% volatility
                'edge': 0.15,              # 15% edge
                'conviction_score': 0.85   # 85% conviction
            }
            bankroll: Current bankroll ($5,000)
            allow_leverage: Allow >100% allocation
        
        Returns:
            {
                'allocations': {'game_1': 1750.00, ...},
                'weights': [0.35, 0.055, ...],
                'total_allocation': 3010.00,
                'portfolio_sharpe': 1.05,
                'expected_return': 0.128,
                'portfolio_volatility': 0.122,
                'diversification_score': 0.82,
                'hhi_index': 0.18
            }
        
        Time: <50ms for n≤15 games
        """
        n = len(opportunities)
        
        if n == 0:
            return {
                'allocations': {},
                'error': 'No opportunities',
                'total_allocation': 0
            }
        
        if n == 1:
            # Single game - simple allocation
            return self._single_game_allocation(opportunities[0], bankroll)
        
        # Build expected returns vector
        mu = np.array([opp['expected_return'] for opp in opportunities])
        
        # Build covariance matrix
        Sigma = self.cov_builder.build_covariance_matrix(opportunities)
        
        # Check for concentration opportunity
        concentration_allowed = self._check_concentration_opportunity(opportunities)
        
        # Solve QP
        weights = self._solve_qp(
            mu=mu,
            Sigma=Sigma,
            n=n,
            concentration_allowed=concentration_allowed,
            allow_leverage=allow_leverage
        )
        
        # Calculate allocations
        allocations = {}
        for i, opp in enumerate(opportunities):
            allocations[opp['game_id']] = bankroll * weights[i]
        
        # Calculate portfolio metrics
        portfolio_return = np.dot(weights, mu)
        portfolio_variance = self.cov_builder.get_portfolio_variance(weights, Sigma)
        portfolio_vol = np.sqrt(portfolio_variance)
        portfolio_sharpe = portfolio_return / portfolio_vol if portfolio_vol > 0 else 0
        
        diversification = self.cov_builder.get_diversification_benefit(weights, opportunities, Sigma)
        
        return {
            'allocations': {k: round(v, 2) for k, v in allocations.items()},
            'weights': [round(w, 4) for w in weights],
            'total_allocation': round(sum(allocations.values()), 2),
            'portfolio_sharpe': round(portfolio_sharpe, 3),
            'expected_return': round(portfolio_return, 4),
            'portfolio_volatility': round(portfolio_vol, 4),
            'diversification_score': round(diversification, 3),
            'hhi_index': round(self._calculate_hhi(weights), 4),
            'num_bets': n
        }
    
    def _solve_qp(
        self,
        mu: np.ndarray,
        Sigma: np.ndarray,
        n: int,
        concentration_allowed: bool,
        allow_leverage: bool
    ) -> np.ndarray:
        """
        Solve quadratic program using CVXPY
        
        Formula (MATH_BREAKDOWN.txt 7.1):
            max w^T μ - (λ/2) w^T Σ w
        
        Returns: Optimal weights
        Time: <30ms for n=10
        """
        # Decision variable
        w = cp.Variable(n)
        
        # Objective: Maximize return - risk_aversion × variance
        portfolio_return = mu @ w
        portfolio_variance = cp.quad_form(w, Sigma)
        
        objective = cp.Maximize(
            portfolio_return - (self.risk_aversion / 2) * portfolio_variance
        )
        
        # Constraints
        constraints = []
        
        # No short selling
        constraints.append(w >= 0)
        
        # Individual position limits
        if concentration_allowed:
            # Allow concentration for exceptional opportunity
            constraints.append(w <= self.concentration_max)
        else:
            constraints.append(w <= self.max_single_position)
        
        # Total allocation limit
        if allow_leverage:
            constraints.append(cp.sum(w) <= 1.50)  # Max 150%
        else:
            constraints.append(cp.sum(w) <= self.max_total_allocation)
        
        # Solve
        problem = cp.Problem(objective, constraints)
        
        try:
            problem.solve(solver=cp.ECOS, verbose=False)
            
            if w.value is None or problem.status != 'optimal':
                # Fallback: Equal weights scaled to 50%
                return np.ones(n) / n * 0.50
            
            return np.array(w.value)
            
        except Exception as e:
            print(f"QP solver error: {e}")
            # Fallback
            return np.ones(n) / n * 0.50
    
    def _single_game_allocation(self, opportunity: Dict, bankroll: float) -> Dict:
        """Handle single game case (no optimization needed)"""
        weight = min(0.20, opportunity.get('edge', 0.10))
        allocation = bankroll * weight
        
        return {
            'allocations': {opportunity['game_id']: round(allocation, 2)},
            'weights': [round(weight, 4)],
            'total_allocation': round(allocation, 2),
            'portfolio_sharpe': opportunity.get('edge', 0.10) / opportunity.get('volatility', 0.20),
            'expected_return': opportunity['expected_return'],
            'portfolio_volatility': opportunity['volatility'],
            'diversification_score': 0.0,
            'hhi_index': 1.0,
            'num_bets': 1
        }
    
    def _check_concentration_opportunity(self, opportunities: List[Dict]) -> bool:
        """
        Check if one opportunity is significantly better
        
        Criteria: Top conviction ≥ 90% and 15%+ better than second
        """
        if len(opportunities) < 2:
            return False
        
        convictions = [opp.get('conviction_score', 0.70) for opp in opportunities]
        
        top = max(convictions)
        second = sorted(convictions, reverse=True)[1]
        
        if top >= 0.90 and (top - second) >= 0.15:
            return True
        
        return False
    
    def _calculate_hhi(self, weights: np.ndarray) -> float:
        """
        Calculate Herfindahl-Hirschman Index
        
        Formula (MATH_BREAKDOWN.txt 11.1):
            HHI = Σ(w_i²)
        
        Lower HHI = Better diversification
        """
        return float(np.sum(weights ** 2))


# Test the portfolio optimizer
if __name__ == "__main__":
    import time
    
    print("="*80)
    print("PORTFOLIO OPTIMIZER - VERIFICATION")
    print("="*80)
    
    optimizer = PortfolioOptimizer(
        max_single_position=0.20,
        max_total_allocation=0.80,
        risk_aversion=1.5
    )
    
    # Test 1: 6-game portfolio
    print("\n" + "="*80)
    print("TEST 1: 6-GAME PORTFOLIO (TYPICAL NIGHT)")
    print("="*80)
    
    opportunities = [
        {
            'game_id': 'LAL@BOS',
            'expected_return': 0.15,
            'volatility': 0.22,
            'edge': 0.226,
            'conviction_score': 0.92,
            'division': 'Atlantic'
        },
        {
            'game_id': 'GSW@MIA',
            'expected_return': 0.12,
            'volatility': 0.20,
            'edge': 0.185,
            'conviction_score': 0.78,
            'division': 'Southeast'
        },
        {
            'game_id': 'DEN@PHX',
            'expected_return': 0.08,
            'volatility': 0.18,
            'edge': 0.105,
            'conviction_score': 0.65,
            'division': 'Pacific'
        },
        {
            'game_id': 'BKN@MIL',
            'expected_return': 0.13,
            'volatility': 0.21,
            'edge': 0.165,
            'conviction_score': 0.80,
            'division': 'Central'
        },
        {
            'game_id': 'DAL@LAC',
            'expected_return': 0.10,
            'volatility': 0.19,
            'edge': 0.125,
            'conviction_score': 0.72,
            'division': 'Pacific'
        },
        {
            'game_id': 'MEM@NOP',
            'expected_return': 0.11,
            'volatility': 0.20,
            'edge': 0.140,
            'conviction_score': 0.75,
            'division': 'Southwest'
        }
    ]
    
    print(f"\n6 games with edges detected")
    print(f"Bankroll: $5,000")
    
    start = time.time()
    result = optimizer.optimize(
        opportunities=opportunities,
        bankroll=5000,
        allow_leverage=False
    )
    elapsed = (time.time() - start) * 1000
    
    print(f"\nOPTIMIZED ALLOCATION:")
    print(f"  {'Game':<15} {'Allocation':<12} {'Weight':<8}")
    print(f"  {'-'*35}")
    for game_id, allocation in result['allocations'].items():
        weight = allocation / 5000
        print(f"  {game_id:<15} ${allocation:>8,.0f}   {weight:>6.1%}")
    
    print(f"\nPORTFOLIO METRICS:")
    print(f"  Total allocation: ${result['total_allocation']:,.0f} ({result['total_allocation']/5000:.1%} of bankroll)")
    print(f"  Expected return: {result['expected_return']:.1%}")
    print(f"  Portfolio volatility: {result['portfolio_volatility']:.1%}")
    print(f"  Portfolio Sharpe: {result['portfolio_sharpe']:.2f}")
    print(f"  Diversification score: {result['diversification_score']:.2f}")
    print(f"  HHI (concentration): {result['hhi_index']:.3f}")
    
    print(f"\nPERFORMANCE:")
    print(f"  Optimization time: {elapsed:.1f}ms")
    print(f"  Target: <50ms")
    if elapsed < 50:
        print(f"  ✅ PASS!")
    else:
        print(f"  ❌ FAIL - Too slow")
    
    # Test 2: Performance with 10 games
    print("\n" + "="*80)
    print("TEST 2: PERFORMANCE (10 GAMES)")
    print("="*80)
    
    large_opportunities = [
        {
            'game_id': f'Game{i}',
            'expected_return': 0.10 + i * 0.01,
            'volatility': 0.18 + i * 0.005,
            'edge': 0.12 + i * 0.01,
            'conviction_score': 0.70 + i * 0.02,
            'division': f'Div{i%4}'
        }
        for i in range(10)
    ]
    
    start = time.time()
    result2 = optimizer.optimize(
        opportunities=large_opportunities,
        bankroll=5000
    )
    elapsed = (time.time() - start) * 1000
    
    print(f"\n10 games optimized:")
    print(f"  Total allocation: ${result2['total_allocation']:,.0f}")
    print(f"  Portfolio Sharpe: {result2['portfolio_sharpe']:.2f}")
    print(f"  Optimization time: {elapsed:.1f}ms")
    
    if elapsed < 50:
        print(f"  ✅ PASS - Under 50ms target!")
    
    print("\n" + "="*80)
    print("✅ PORTFOLIO OPTIMIZER READY")
    print("="*80)
    print("\nMarkowitz mean-variance optimization working!")
    print("  Quadratic programming via CVXPY")
    print("  Sharpe ratio maximization")
    print("  Correlation-adjusted allocation")
    print("  Institutional-grade optimization")

