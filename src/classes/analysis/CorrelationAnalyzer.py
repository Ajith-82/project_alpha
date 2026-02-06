"""
Correlation Analyzer Module

Provides stock correlation analysis including:
- Match finding (most correlated stock pairs)
- Clustering (groups of similar stocks)
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

try:
    import multitasking
    MULTITASKING_AVAILABLE = True
except ImportError:
    MULTITASKING_AVAILABLE = False


logger = logging.getLogger(__name__)


@dataclass
class MatchResult:
    """Result of stock matching."""
    match: str  # Matched ticker symbol
    index: int  # Index in ticker list
    distance: float  # Distance metric


@dataclass
class CorrelationResult:
    """Results from correlation analysis."""
    matches: Dict[str, MatchResult]
    clusters: Optional[List[int]] = None  # Cluster assignments


class CorrelationAnalyzer:
    """
    Analyzes correlations between stocks.
    
    Uses derivative of log-price estimates to find:
    - Most similar stock for each ticker (matches)
    - Groups of correlated stocks (clusters)
    """
    
    def __init__(self, max_stocks_vectorized: int = 2000):
        """
        Initialize CorrelationAnalyzer.
        
        Args:
            max_stocks_vectorized: Max stocks for vectorized computation.
                                   Uses multithreading above this.
        """
        self.max_stocks_vectorized = max_stocks_vectorized
    
    def _compute_derivative_estimates(
        self, mu: np.ndarray, tt: np.ndarray
    ) -> np.ndarray:
        """
        Compute derivative of log-price estimates.
        
        Args:
            mu: Polynomial coefficients
            tt: Time array
            
        Returns:
            Derivative estimates (dlogp/dt)
        """
        # Derivative time array: scale by polynomial order
        dtt = np.arange(1, tt.shape[0])[:, None] * tt[1:] / tt[1, None]
        # Apply derivative (exclude constant term)
        dlogp_est = np.dot(mu[:, 1:], dtt)
        return dlogp_est
    
    def estimate_matches(
        self, tickers: List[str], mu: np.ndarray, tt: np.ndarray
    ) -> Dict[str, MatchResult]:
        """
        Find the most correlated stock for each ticker.
        
        Uses sum of squared differences in derivative estimates
        as a distance metric.
        
        Args:
            tickers: List of ticker symbols
            mu: Polynomial coefficients
            tt: Time array
            
        Returns:
            Dictionary mapping ticker → MatchResult
        """
        dlogp_est = self._compute_derivative_estimates(mu, tt)
        num_stocks = len(tickers)
        
        # Choose method based on size
        if num_stocks <= self.max_stocks_vectorized:
            return self._estimate_matches_vectorized(tickers, dlogp_est)
        else:
            return self._estimate_matches_parallel(tickers, dlogp_est)
    
    def _estimate_matches_vectorized(
        self, tickers: List[str], dlogp_est: np.ndarray
    ) -> Dict[str, MatchResult]:
        """
        Vectorized match estimation for smaller datasets.
        
        Uses full distance matrix computation.
        """
        # Compute pairwise distances: (N, N) matrix
        # dist[i,j] = sum((dlogp_est[i] - dlogp_est[j])^2)
        diff = dlogp_est[:, None] - dlogp_est[None]  # (N, N, T)
        match_dist = np.sum(diff**2, axis=2)  # (N, N)
        
        # Find minimum distance (excluding self)
        np.fill_diagonal(match_dist, np.inf)
        match_minidx = np.argmin(match_dist, axis=1)
        match_mindist = np.min(match_dist, axis=1)
        
        matches = {}
        for i, ticker in enumerate(tickers):
            matches[ticker] = MatchResult(
                match=tickers[match_minidx[i]],
                index=int(match_minidx[i]),
                distance=float(match_mindist[i]),
            )
        
        return matches
    
    def _estimate_matches_parallel(
        self, tickers: List[str], dlogp_est: np.ndarray
    ) -> Dict[str, MatchResult]:
        """
        Parallel match estimation for larger datasets.
        
        Uses multithreading to avoid memory issues.
        """
        if not MULTITASKING_AVAILABLE:
            logger.warning("multitasking not available, falling back to sequential")
            return self._estimate_matches_sequential(tickers, dlogp_est)
        
        num_stocks = len(tickers)
        num_threads = min(num_stocks, multitasking.cpu_count() * 2)
        multitasking.set_max_threads(num_threads)
        
        matches = {}
        
        @multitasking.task
        def _estimate_one(i: int):
            dist = np.sum((dlogp_est[i] - dlogp_est) ** 2, axis=1)
            dist[i] = np.inf  # Exclude self
            min_idx = np.argmin(dist)
            matches[tickers[i]] = MatchResult(
                match=tickers[min_idx],
                index=int(min_idx),
                distance=float(dist[min_idx]),
            )
        
        for i in range(num_stocks):
            _estimate_one(i)
        
        # Wait for all tasks
        multitasking.wait_for_tasks()
        
        return matches
    
    def _estimate_matches_sequential(
        self, tickers: List[str], dlogp_est: np.ndarray
    ) -> Dict[str, MatchResult]:
        """Sequential fallback for match estimation."""
        matches = {}
        num_stocks = len(tickers)
        
        for i in range(num_stocks):
            dist = np.sum((dlogp_est[i] - dlogp_est) ** 2, axis=1)
            dist[i] = np.inf
            min_idx = np.argmin(dist)
            matches[tickers[i]] = MatchResult(
                match=tickers[min_idx],
                index=int(min_idx),
                distance=float(dist[min_idx]),
            )
        
        return matches
    
    def estimate_clusters(
        self, tickers: List[str], mu: np.ndarray, tt: np.ndarray
    ) -> List[int]:
        """
        Group stocks into clusters based on correlation.
        
        Uses a union-find like approach where stocks are grouped
        with their closest neighbor.
        
        Args:
            tickers: List of ticker symbols
            mu: Polynomial coefficients
            tt: Time array
            
        Returns:
            List of cluster indices (one per stock)
        """
        dlogp_est = self._compute_derivative_estimates(mu, tt)
        num_stocks = len(tickers)
        
        clusters: List[set] = []
        
        def _unite_clusters(clusters: List[set]) -> List[set]:
            """Merge overlapping clusters."""
            k = 0
            while k < len(clusters):
                merged = False
                for j in range(k + 1, len(clusters)):
                    if clusters[j] & clusters[k]:
                        clusters[j] = clusters[j].union(clusters[k])
                        del clusters[k]
                        merged = True
                        break
                if not merged:
                    k += 1
            return clusters
        
        # Build clusters by pairing each stock with its closest neighbor
        for i in range(num_stocks):
            dist = np.sum((dlogp_est[i] - dlogp_est) ** 2, axis=1)
            closest_two = np.argsort(dist)[:2].tolist()
            clusters.append(set(closest_two))
            clusters = _unite_clusters(clusters)
        
        # Convert to cluster indices
        cluster_indices = []
        for j in range(num_stocks):
            for k, cluster in enumerate(clusters):
                if j in cluster:
                    cluster_indices.append(k)
                    break
        
        return cluster_indices
