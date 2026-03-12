import os
import yfinance as yf
import pandas as pd
import numpy as np

def fetch_spy_data(start_date: str, end_date: str, filename: str) -> None:
    """
    Downloads historical SPY (S&P 500 ETF) data and calculates daily log returns.
    Saves the cleaned output to the /data/ directory as a CSV.
    """
    print(f"Downloading SPY data from {start_date} to {end_date}...")
    
    # Download the historical daily bars using the stable Ticker object structure
    try:
        spy = yf.Ticker("SPY").history(start=start_date, end=end_date)
    except Exception as e:
        print(f"Failed to fetch data for {filename}: {e}")
        return

    if spy.empty:
        print(f"Warning: No data found for {start_date} to {end_date}")
        return

    # yf.Ticker.history default returns 'Close' which is already adjusted for dividends/splits
    spy['Log_Return'] = np.log(spy['Close'] / spy['Close'].shift(1))
    
    # Drop the very first row since it won't have a yesterday to compute a return
    spy = spy.dropna(subset=['Log_Return'])
    
    # Keep only the essential columns for our engine and visualization
    spy = spy[['Close', 'Log_Return']]
    
    # Build absolute path to save the data securely in the project's /data/ folder
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "..", "data")
    
    # Ensure /data/ dir exists (just in case)
    os.makedirs(data_dir, exist_ok=True)
    
    out_path = os.path.join(data_dir, filename)
    spy.to_csv(out_path)
    
    print(f"✅ Saved {len(spy)} trading days to {out_path}\n")

if __name__ == "__main__":
    print("--- Starting Phase 4 (Part 1): Historical Data Extraction ---\n")
    
    # 1. The 2008 Global Financial Crisis
    # Spans from pre-crash highs to the absolute bottom and slow recovery
    fetch_spy_data(
        start_date="2007-01-01", 
        end_date="2010-01-01", 
        filename="spy_2008_crash.csv"
    )
    
    # 2. The 2020 COVID-19 Flash Crash
    # Spans the massive V-shaped bottom representing intense volatility
    fetch_spy_data(
        start_date="2019-06-01", 
        end_date="2021-06-01", 
        filename="spy_2020_crash.csv"
    )
