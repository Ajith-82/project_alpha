"""
Volatile Analyzer Module

Main orchestrator for the Volatile analysis pipeline.
Combines TrendAnalyzer and CorrelationAnalyzer with Rich progress UI.
"""

import logging
import os
import pickle
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
    from rich.console import Console
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from .VolatileConfig import VolatileConfig, TrainingConfig
from .TrendAnalyzer import TrendAnalyzer, TrendResult, estimate_logprice_statistics, estimate_price_statistics
from .CorrelationAnalyzer import CorrelationAnalyzer, MatchResult


logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    """Complete results from Volatile analysis."""
    
    # Core metrics
    tickers: List[str]
    scores: np.ndarray
    growth: np.ndarray
    volatility: np.ndarray
    rates: List[str]
    
    # Matches (correlation pairs)
    matches: Optional[Dict[str, MatchResult]] = None
    
    # Ranking
    rank: Optional[np.ndarray] = None
    rank_method: str = "rate"
    
    # Price estimates
    p_est: Optional[np.ndarray] = None
    std_p_est: Optional[np.ndarray] = None
    
    # Hierarchical estimates  
    p_mkt_est: Optional[np.ndarray] = None
    std_p_mkt_est: Optional[np.ndarray] = None
    p_sec_est: Optional[np.ndarray] = None
    std_p_sec_est: Optional[np.ndarray] = None
    p_ind_est: Optional[np.ndarray] = None
    std_p_ind_est: Optional[np.ndarray] = None
    
    # Model parameters (for saving)
    params: Optional[tuple] = None
    
    # Additional info
    info: Dict[str, Any] = field(default_factory=dict)
    
    def to_dataframe(
        self,
        data: Dict[str, Any],
        rank_method: str = "rate",
    ) -> pd.DataFrame:
        """
        Convert results to pandas DataFrame.
        
        Args:
            data: Original data dict with sectors/industries
            rank_method: Ranking method used
            
        Returns:
            DataFrame with prediction table
        """
        num_stocks = len(self.tickers)
        rank = self.get_ranking(rank_method)
        
        ranked_tickers = np.array(self.tickers)[rank]
        ranked_scores = self.scores[rank]
        ranked_growth = self.growth[rank]
        ranked_volatility = self.volatility[rank]
        ranked_rates = [self.rates[i] for i in rank]
        
        # Get sectors and industries
        sectors = list(data.get("sectors", {}).values())
        industries = list(data.get("industries", {}).values())
        
        ranked_sectors = [
            sectors[i] if i < len(sectors) else "Unknown"
            for i in rank
        ]
        ranked_industries = [
            industries[i] if i < len(industries) else "Unknown"
            for i in rank
        ]
        
        # Clean NA prefix
        ranked_sectors = [
            s if not s.startswith("NA_") else "Not Available"
            for s in ranked_sectors
        ]
        ranked_industries = [
            s if not s.startswith("NA_") else "Not Available"
            for s in ranked_industries
        ]
        
        # Get prices
        price = data.get("price", np.zeros((num_stocks, 1)))
        currencies = data.get("currencies", ["USD"] * num_stocks)
        
        ranked_prices = [
            f"{np.round(price[i, -1], 2)} {currencies[i]}"
            for i in rank
        ]
        
        # Get matches
        if self.matches:
            ranked_matches = [
                self.matches[t].match if t in self.matches else "None"
                for t in ranked_tickers
            ]
        else:
            ranked_matches = ["None"] * num_stocks
        
        return pd.DataFrame({
            "SYMBOL": ranked_tickers.tolist(),
            "SECTOR": ranked_sectors,
            "INDUSTRY": ranked_industries,
            "PRICE": ranked_prices,
            "RATE": ranked_rates,
            "GROWTH": ranked_growth.tolist(),
            "VOLATILITY": ranked_volatility.tolist(),
            "MATCH": ranked_matches,
        })
    
    def get_ranking(self, method: str = "rate") -> np.ndarray:
        """Get stock ranking by specified method."""
        if method == "rate":
            return np.argsort(self.scores)[::-1]
        elif method == "growth":
            return np.argsort(self.growth)[::-1]
        elif method == "volatility":
            return np.argsort(self.volatility)[::-1]
        else:
            return np.arange(len(self.tickers))


class VolatileAnalyzer:
    """
    Main orchestrator for Volatile analysis.
    
    Coordinates:
    - Model training (correlation + trend models)
    - Trend analysis (scores, growth, volatility, ratings)
    - Correlation analysis (stock matches)
    - Progress tracking with Rich UI
    """
    
    def __init__(self, config: Optional[VolatileConfig] = None):
        """
        Initialize VolatileAnalyzer.
        
        Args:
            config: Analysis configuration
        """
        self.config = config or VolatileConfig()
        self.trend_analyzer = TrendAnalyzer(self.config.thresholds)
        self.correlation_analyzer = CorrelationAnalyzer()
        
        if RICH_AVAILABLE:
            self.console = Console()
        else:
            self.console = None
    
    def _print(self, message: str):
        """Print message using Rich or fallback."""
        if self.console and self.config.verbose:
            self.console.print(message)
        elif self.config.verbose:
            print(message)
    
    def _extract_hierarchical_info(
        self, sectors: Dict[str, str], industries: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Extract hierarchical structure from sector/industry data.
        
        This is needed for the MSIS-MCS model.
        """
        # Import from Tools module
        try:
            from classes.Tools import extract_hierarchical_info
            return extract_hierarchical_info(sectors, industries)
        except ImportError:
            # Fallback implementation
            return self._build_hierarchy_fallback(sectors, industries)
    
    def _build_hierarchy_fallback(
        self, sectors: Dict[str, str], industries: Dict[str, str]
    ) -> Dict[str, Any]:
        """Fallback hierarchy builder."""
        tickers = list(sectors.keys())
        
        # Build sector indices
        unique_sectors = list(set(sectors.values()))
        sector_idx = np.array([unique_sectors.index(sectors[t]) for t in tickers])
        
        # Build industry indices
        unique_industries = list(set(industries.values()))
        industry_idx = np.array([unique_industries.index(industries[t]) for t in tickers])
        
        return {
            "sector_idx": sector_idx,
            "industry_idx": industry_idx,
            "num_sectors": len(unique_sectors),
            "num_industries": len(unique_industries),
        }
    
    def analyze(
        self,
        data: Dict[str, Any],
        rank_method: str = "rate",
    ) -> AnalysisResult:
        """
        Run complete Volatile analysis pipeline.
        
        Args:
            data: Input data dict with keys:
                  - tickers: List of symbols
                  - price: 2D array (stocks × time)
                  - sectors: Dict of ticker → sector
                  - industries: Dict of ticker → industry
                  - currencies: List of currency codes
            rank_method: Ranking method (rate/growth/volatility)
            
        Returns:
            AnalysisResult with all computed metrics
        """
        tickers = data["tickers"]
        logp = np.log(data["price"])
        num_stocks, num_timesteps = logp.shape
        
        # Extract hierarchical info
        info = self._extract_hierarchical_info(
            data.get("sectors", {}),
            data.get("industries", {}),
        )
        
        # Load initial parameters if specified
        initial_params = None
        if self.config.load_model and os.path.exists(self.config.load_model):
            with open(self.config.load_model, "rb") as f:
                initial_params = pickle.load(f)
            self._print(f"[green]Loaded model from {self.config.load_model}[/green]")
        
        matches = None
        
        # Phase 1: Correlation model (only if multiple stocks)
        if num_stocks > 1:
            self._print("\n[bold cyan]Phase 1: Training correlation model...[/bold cyan]")
            
            order = self.config.training.order_correlation
            info["tt"] = self._build_time_array(num_timesteps, order)
            info["order_scale"] = np.ones((1, order + 1), dtype="float32")
            
            # Train correlation model
            correlation_params = self._train_model(
                logp, info,
                num_steps=self.config.training.correlation_steps,
                phase_name="Correlation",
            )
            
            if correlation_params:
                phi = correlation_params[6]  # Stock-level mean params
                matches = self.correlation_analyzer.estimate_matches(
                    tickers, phi.numpy(), info["tt"]
                )
                self._print("[green]✓ Correlation analysis complete[/green]")
        
        # Phase 2: Trend model
        self._print("\n[bold cyan]Phase 2: Training trend model...[/bold cyan]")
        
        order = self.config.training.order_trend
        info["tt"] = self._build_time_array(num_timesteps, order)
        info["order_scale"] = (
            np.linspace(1 / (order + 1), 1, order + 1)[::-1]
            .astype("float32")[None, :]
        )
        
        params = self._train_model(
            logp, info,
            num_steps=self.config.training.trend_steps,
            initial_params=initial_params,
            phase_name="Trend",
        )
        
        if params is None:
            raise RuntimeError("Model training failed")
        
        phi_m, psi_m, phi_s, psi_s, phi_i, psi_i, phi, psi = params[:8]
        
        self._print("[green]✓ Trend analysis complete[/green]")
        
        # Save model if requested
        if self.config.save_model:
            with open(self.config.save_model, "wb") as f:
                pickle.dump([p.numpy() for p in params], f)
            self._print(f"[green]Saved model to {self.config.save_model}[/green]")
        
        # Compute trend metrics
        trend_result = self.trend_analyzer.analyze(
            phi.numpy(),
            psi.numpy(),
            logp,
            data["price"],
            info["tt"],
            order,
            self.config.horizon,
        )
        
        # Compute hierarchical estimates
        logp_mkt_est, std_logp_mkt_est = estimate_logprice_statistics(
            phi_m.numpy(), psi_m.numpy(), info["tt"]
        )
        logp_sec_est, std_logp_sec_est = estimate_logprice_statistics(
            phi_s.numpy(), psi_s.numpy(), info["tt"]
        )
        logp_ind_est, std_logp_ind_est = estimate_logprice_statistics(
            phi_i.numpy(), psi_i.numpy(), info["tt"]
        )
        
        p_mkt_est, std_p_mkt_est = estimate_price_statistics(logp_mkt_est, std_logp_mkt_est)
        p_sec_est, std_p_sec_est = estimate_price_statistics(logp_sec_est, std_logp_sec_est)
        p_ind_est, std_p_ind_est = estimate_price_statistics(logp_ind_est, std_logp_ind_est)
        
        return AnalysisResult(
            tickers=tickers,
            scores=trend_result.scores,
            growth=trend_result.growth,
            volatility=trend_result.volatility,
            rates=trend_result.rates,
            matches=matches,
            p_est=trend_result.p_est,
            std_p_est=trend_result.std_p_est,
            p_mkt_est=p_mkt_est,
            std_p_mkt_est=std_p_mkt_est,
            p_sec_est=p_sec_est,
            std_p_sec_est=std_p_sec_est,
            p_ind_est=p_ind_est,
            std_p_ind_est=std_p_ind_est,
            params=params,
            info=info,
        )
    
    def _build_time_array(self, num_timesteps: int, order: int) -> np.ndarray:
        """Build polynomial time array."""
        return (
            np.linspace(1 / num_timesteps, 1, num_timesteps)
            ** np.arange(order + 1).reshape(-1, 1)
        ).astype("float32")
    
    def _train_model(
        self,
        logp: np.ndarray,
        info: Dict[str, Any],
        num_steps: int,
        initial_params: Optional[tuple] = None,
        phase_name: str = "Model",
    ) -> Optional[tuple]:
        """
        Train the MSIS-MCS model.
        
        Args:
            logp: Log-prices
            info: Hierarchical info dict
            num_steps: Training steps
            initial_params: Pre-trained parameters
            phase_name: Name for progress display
            
        Returns:
            Trained parameters or None on failure
        """
        try:
            from classes.Models import train_msis_mcs
            
            if RICH_AVAILABLE and self.config.verbose:
                with Progress(
                    SpinnerColumn(),
                    TextColumn(f"[cyan]{phase_name}[/cyan]"),
                    BarColumn(),
                    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                    TimeElapsedColumn(),
                    console=self.console,
                ) as progress:
                    task = progress.add_task(phase_name, total=100)
                    
                    # Run training
                    params = train_msis_mcs(
                        logp, info,
                        num_steps=num_steps,
                        plot_losses=self.config.plot_losses,
                        initial_params=initial_params,
                    )
                    
                    progress.update(task, completed=100)
                    return params
            else:
                return train_msis_mcs(
                    logp, info,
                    num_steps=num_steps,
                    plot_losses=self.config.plot_losses,
                    initial_params=initial_params,
                )
        
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            return None
