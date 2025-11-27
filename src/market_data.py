# market_data.py (Adapted)

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
    interval: str = '1mo' # Use weekly or monthly data to align with signal frequency
) -> pl.DataFrame:
    """
    Downloads historical adjusted closing prices for a list of tickers.
    If a ticker is new and has no data for the requested start_date, it
    automatically retrieves data from the ticker's first available date.
    
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
        # Download data. We rely on yfinance's behavior to fetch whatever data is available
        # if the start date is too early. Tickers with no data will have NA columns.
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
        return pl.DataFrame({"Date": [], "Ticker": [], "Adj_Close": []})


    # Process the multi-index Pandas DataFrame
    if len(tickers) == 1:
        # Handle single ticker case
        df_long = data[['Adj Close']].copy()
        df_long.columns = ['Adj_Close']
        df_long['Ticker'] = tickers[0]
        df_long = df_long.reset_index()
        
    else:
        # Standard multi-ticker case
        # Select only 'Adj Close'
        df_adj_close = data.loc[:, (slice(None), 'Adj Close')]
        df_adj_close.columns = df_adj_close.columns.get_level_values(0)
        
        # Melt/Stack the data to a long format (Date, Ticker, Adj_Close)
        # This is where NA values for new tickers will naturally appear, 
        # allowing us to discard them later while keeping the date range intact.
        df_long = df_adj_close.stack().reset_index()
        df_long.columns = ['Date', 'Ticker', 'Adj_Close']

    # Convert the resulting Pandas DataFrame to Polars DataFrame
    df_pl = pl.from_pandas(df_long)
    
    # Final Polars cleaning and type adjustment
    df_pl = df_pl.with_columns(
        pl.col("Date").cast(pl.Date),
        pl.col("Adj_Close").cast(pl.Float64),
        pl.col("Ticker").cast(pl.Utf8)
    )
    
    # CRUCIAL STEP: Drop rows where Adj_Close is null. 
    # This removes the early dates for newly launched ETFs where no data exists, 
    # ensuring the time series starts on the ticker's actual inception date.
    df_pl = df_pl.drop_nulls(subset=["Adj_Close"])
    
    print(f"✓ Fetched {len(df_pl):,} market data points.")
    
    # Log the effective start date for verification
    if not df_pl.is_empty():
        min_dates = df_pl.group_by("Ticker").agg(pl.col("Date").min().alias("Effective_Start_Date"))
        print("\nEffective Start Dates per Ticker (May differ from requested start_date):")
        print(min_dates.sort("Effective_Start_Date").to_pandas().to_markdown(index=False))
        
    return df_pl

if __name__ == '__main__':
    # Example test run
    start = "2018-01-01"
    end = "2023-12-31"
    # Example includes an old ticker (SOXX) and a very recent one (TMET, Sep 2023)
    tickers = ["SOXX", "TMET", BENCHMARK_TICKER] 
    
    market_data_test = get_historical_prices(tickers, start, end)
    
    if market_data_test.is_empty():
        sys.exit("Market data retrieval failed.")
    else:
        print("Market data retrieval successful.")