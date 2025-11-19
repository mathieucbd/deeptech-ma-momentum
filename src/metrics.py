"""
Advanced Metrics Utilities

This module provides financial performance metrics used in Phase 5 (Backtesting).
All functions operate on a Polars Series of PERIODIC SIMPLE RETURNS (e.g., weekly returns).
"""

# --- 1. Standard Library Imports ---
import sys
from typing import Dict

# --- 2. Third-Party Library Imports ---
import polars as pl
import numpy as np

# Annualization factor for weekly data (52 weeks in a year)
ANNUALIZATION_FACTOR = 52

def calculate_sharpe(returns: pl.Series) -> float:
    """Calculates the annualized Sharpe Ratio (Risk-free rate = 0)."""
    if len(returns) < 2: return 0.0
    
    annual_mean = returns.mean() * ANNUALIZATION_FACTOR
    annual_std = returns.std() * np.sqrt(ANNUALIZATION_FACTOR)

    return annual_mean / annual_std if annual_std > 0 else 0.0

def calculate_sortino(returns: pl.Series) -> float:
    """Calculates the annualized Sortino Ratio (Downside Deviation)."""
    if len(returns) < 2: return 0.0

    # Isolate negative returns (downside deviation)
    negative_returns = returns.filter(returns < 0)
    
    if len(negative_returns) == 0:
        # If no downside movement, Sortino is infinite/undefined, return Sharpe or a large number.
        return calculate_sharpe(returns) 
        
    downside_vol = negative_returns.std() * np.sqrt(ANNUALIZATION_FACTOR)
    annual_mean = returns.mean() * ANNUALIZATION_FACTOR
    
    return annual_mean / downside_vol if downside_vol > 0 else 0.0

def calculate_hit_ratio(returns: pl.Series) -> float:
    """Calculates the ratio of positive return periods to total non-zero return periods."""
    if len(returns) == 0: return 0.0
    
    positive_count = len(returns.filter(returns > 0))
    non_zero_count = len(returns.filter(returns != 0))
    
    return positive_count / non_zero_count if non_zero_count > 0 else 0.0

def calculate_max_drawdown(returns: pl.Series) -> float:
    """Calculates the Maximum Drawdown from peak equity."""
    if len(returns) == 0:
        return 0.0

    # 1. Calculate cumulative returns (1 + R1) * (1 + R2) ...
    cumulative_returns = (1 + returns.fill_null(0)).cum_prod()

    if len(cumulative_returns) == 0:
        return 0.0

    # 2. Find the running peak equity value
    peak = cumulative_returns.cum_max()

    # 3. Calculate Drawdown
    drawdown = (cumulative_returns / peak) - 1

    # 4. Max Drawdown is the minimum (most negative) value; guard None
    md = drawdown.min()
    return md if md is not None else 0.0

def calculate_all_metrics(returns: pl.Series) -> Dict[str, float]:
    """Calculates all primary metrics from a single returns series."""
    # Note: Cumulative return is calculated separately as it requires the full history.
    
    return {
        "sharpe_ratio": calculate_sharpe(returns),
        "sortino_ratio": calculate_sortino(returns),
        "max_drawdown": calculate_max_drawdown(returns),
        "hit_ratio": calculate_hit_ratio(returns),
    }