"""
Market Data Retrieval Module

This module fetches historical price data for a list of tickers (ETFs/Stocks)
from Yahoo Finance using the yfinance library and converts the output to Polars.
"""

# --- 1. Standard Library Imports ---
import time
import sys

# --- 2. Third-Party Library Imports ---
import polars as pl
import yfinance as yf
import pandas as pd
from typing import List, Dict

# Define the default benchmark ticker (S&P 500)
BENCHMARK_TICKER = "^GSPC" 


def get_historical_prices(
    tickers: List[str], 
    start_date: str, 
    end_date: str,
    interval: str = '1wk' # Use weekly or monthly data to align with signal frequency
) -> pl.DataFrame:
    """
    Downloads historical adjusted closing prices for a list of tickers.

    Args:
        tickers: List of ticker strings (e.g., ['ARKG', 'QTUM']).
        start_date: Start date string (YYYY-MM-DD).
        end_date: End date string (YYYY-MM-DD).
        interval: Data frequency (e.g., '1wk' for weekly, '1mo' for monthly).

    Returns:
        A Polars DataFrame containing the 'Date', 'Ticker', and 'Adj_Close'.
    """
    
    print(f"Fetching {interval} data for {len(tickers)} tickers...")
    
    try:
        # Download data using yfinance, which returns a Pandas DataFrame
        # Set auto_adjust=False to get 'Adj Close' column explicitly
        data: pd.DataFrame = yf.download(
            tickers=tickers,
            start=start_date,
            end=end_date,
            interval=interval,
            group_by='ticker',
            auto_adjust=False,
            progress=False
        )
    except Exception as e:
        print(f"ERROR: Failed to download data from Yahoo Finance: {e}")
        # Return an empty Polars DataFrame if API call fails
        return pl.DataFrame({"Date": [], "Ticker": [], "Adj_Close": []})


    # Process the multi-index Pandas DataFrame
    # 1. Select only 'Adj Close'
    # 2. Stack the data to go from wide (columns=tickers) to long (column='Ticker')
    if len(tickers) == 1:
        # Handle single ticker case where yfinance output is simpler
        df_long = data[['Adj Close']].copy()
        df_long.columns = ['Adj_Close']
        df_long['Ticker'] = tickers[0]
        df_long = df_long.reset_index()
        
    else:
        # Standard multi-ticker case
        df_adj_close = data.loc[:, (slice(None), 'Adj Close')]
        df_adj_close.columns = df_adj_close.columns.get_level_values(0) # Simplify column names
        
        # Melt/Stack the data to a long format (Date, Ticker, Adj_Close)
        df_long = df_adj_close.stack().reset_index()
        df_long.columns = ['Date', 'Ticker', 'Adj_Close']

    # Convert the resulting Pandas DataFrame to Polars DataFrame for efficient processing
    df_pl = pl.from_pandas(df_long)
    
    # Final Polars cleaning and type adjustment
    df_pl = df_pl.with_columns(
        pl.col("Date").cast(pl.Date), # Ensure Date type
        pl.col("Adj_Close").cast(pl.Float64),
        pl.col("Ticker").cast(pl.Utf8)
    )
    
    print(f"✓ Fetched {len(df_pl):,} market data points.")
    return df_pl

if __name__ == '__main__':
    # Example test run
    # Note: Requires yfinance and pandas to be installed
    start = "2018-01-01"
    end = "2023-12-31"
    tickers = ["ARKG", "QTUM", BENCHMARK_TICKER]
    
    market_data = get_historical_prices(tickers, start, end)
    # print(market_data.head())
    
    if market_data.is_empty():
        sys.exit("Market data retrieval failed.")
    else:
        print("Market data retrieval successful.")